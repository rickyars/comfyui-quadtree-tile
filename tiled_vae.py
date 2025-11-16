'''
# ------------------------------------------------------------------------
#
#   Quadtree VAE
#
#   Introducing a revolutionary new optimization designed to make
#   the VAE work with giant images on limited VRAM!
#   Say goodbye to the frustration of OOM and hello to seamless output!
#
# ------------------------------------------------------------------------
#
#   This script is a wild hack that splits the image into tiles,
#   encodes each tile separately, and merges the result back together.
#
#   Advantages:
#   - The VAE can now work with giant images on limited VRAM
#       (~10 GB for 8K images!)
#   - The merged output is completely seamless without any post-processing.
#
#   Drawbacks:
#   - NaNs always appear in for 8k images when you use fp16 (half) VAE
#       You must use --no-half-vae to disable half VAE for that giant image.
#   - The gradient calculation is not compatible with this hack. It
#       will break any backward() or torch.autograd.grad() that passes VAE.
#       (But you can still use the VAE to generate training data.)
#
#   How it works:
#   1. The image is split into tiles, which are then padded with 11/32 pixels' in the decoder/encoder.
#   2. When Fast Mode is disabled:
#       1. The original VAE forward is decomposed into a task queue and a task worker, which starts to process each tile.
#       2. When GroupNorm is needed, it suspends, stores current GroupNorm mean and var, send everything to RAM, and turns to the next tile.
#       3. After all GroupNorm means and vars are summarized, it applies group norm to tiles and continues. 
#       4. A zigzag execution order is used to reduce unnecessary data transfer.
#   3. When Fast Mode is enabled:
#       1. The original input is downsampled and passed to a separate task queue.
#       2. Its group norm parameters are recorded and used by all tiles' task queues.
#       3. Each tile is separately processed without any RAM-VRAM data transfer.
#   4. After all tiles are processed, tiles are written to a result buffer and returned.
#   Encoder color fix = only estimate GroupNorm before downsampling, i.e., run in a semi-fast mode.
#
#   Enjoy!
#
#   @Author: LI YI @ Nanyang Technological University - Singapore
#   @Date: 2023-03-02
#   @License: CC BY-NC-SA 4.0
#
#   Please give https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111
#   a star if you like the project!
#
# -------------------------------------------------------------------------
'''

import gc
import math
from time import time
from tqdm import tqdm

import torch
import torch.version
import torch.nn.functional as F
# import gradio as gr

# import modules.scripts as scripts
# from .modules import devices
# from modules.shared import state
# from modules.ui import gr_show
# from modules.processing import opt_f
# from modules.sd_vae_approx import cheap_approximation
# from ldm.modules.diffusionmodules.model import AttnBlock, MemoryEfficientAttnBlock

# from tile_utils.attn import get_attn_func
# from tile_utils.typing import Processing

import comfy
import comfy.model_management
from comfy.model_management import processing_interrupted
import contextlib

opt_C = 4
opt_f = 8
is_sdxl = False
disable_nan_check = True

# ==================== Quadtree Classes ====================

class QuadtreeNode:
    """Represents a single node in the quadtree structure"""
    def __init__(self, x: int, y: int, w: int, h: int, depth: int = 0):
        self.x = x  # Top-left x coordinate
        self.y = y  # Top-left y coordinate
        self.w = w  # Width
        self.h = h  # Height
        self.depth = depth  # Depth in tree (0 = root)
        self.variance = 0.0  # Content complexity metric
        self.denoise = 0.0  # Denoise value for diffusion
        self.children = []  # Child nodes (empty if leaf)

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)"""
        return len(self.children) == 0

    def subdivide(self):
        """Subdivide this node into 4 children with 8-pixel alignment for VAE compatibility

        Strategy:
        - Always create exactly 4 children (quadtree property)
        - Rectangular parents → 4 rectangular children
        - Children naturally become smaller as we subdivide deeper
        - Stop at leaf level (min_tile_size or max_depth)
        """
        # Ensure subdivisions are aligned to 8-pixel boundaries for VAE encoder/decoder
        # VAE downsamples by 8x, so tiles must be divisible by 8
        half_w = (self.w // 2) // 8 * 8  # Round down to nearest multiple of 8
        half_h = (self.h // 2) // 8 * 8

        # Ensure we have at least 8 pixels
        half_w = max(half_w, 8)
        half_h = max(half_h, 8)

        # QUADTREE PROPERTY: Always create exactly 4 children
        # Children inherit parent's proportions and become smaller as we subdivide deeper

        self.children = [
            QuadtreeNode(self.x, self.y, half_w, half_h, self.depth + 1),  # Top-left
            QuadtreeNode(self.x + half_w, self.y, self.w - half_w, half_h, self.depth + 1),  # Top-right
            QuadtreeNode(self.x, self.y + half_h, half_w, self.h - half_h, self.depth + 1),  # Bottom-left
            QuadtreeNode(self.x + half_w, self.y + half_h, self.w - half_w, self.h - half_h, self.depth + 1),  # Bottom-right
        ]

    def get_bbox(self):
        """Get bounding box in [x1, x2, y1, y2] format"""
        return [self.x, self.x + self.w, self.y, self.y + self.h]


class QuadtreeBuilder:
    """Builds a quadtree from image/latent content based on RGB Euclidean distance variance"""
    def __init__(self,
                 content_threshold: float = 0.03,
                 max_depth: int = 4,
                 min_tile_size: int = 256,
                 min_denoise: float = 0.0,
                 max_denoise: float = 1.0):
        """
        Initialize quadtree builder

        Args:
            content_threshold: Variance threshold to trigger subdivision
            max_depth: Maximum recursion depth
            min_tile_size: Minimum tile size in pixels
            min_denoise: Denoise value for largest tiles
            max_denoise: Denoise value for smallest tiles
        """
        self.content_threshold = content_threshold
        self.max_depth = max_depth
        self.min_tile_size = min_tile_size
        self.min_denoise = min_denoise
        self.max_denoise = max_denoise
        self.max_tile_area = 0

    def calculate_variance(self, tensor: torch.Tensor, x: int, y: int, w: int, h: int) -> float:
        """
        Calculate variance as average Euclidean distance from mean color
        Reference: avg(sqrt((R-R_avg)² + (G-G_avg)² + (B-B_avg)²))

        Args:
            tensor: Input tensor (B, C, H, W) or (C, H, W)
            x, y, w, h: Region coordinates

        Returns:
            Average Euclidean distance in RGB space
        """
        # Extract region
        if tensor.dim() == 4:
            region = tensor[:, :, y:y+h, x:x+w]
            channel_dim = 1
        else:
            region = tensor[:, y:y+h, x:x+w]
            channel_dim = 0

        if region.numel() == 0:
            return 0.0

        # Compute mean color across spatial dimensions
        mean_color = torch.mean(region, dim=(-2, -1), keepdim=True)

        # Euclidean distance in RGB space for each pixel
        diff = region - mean_color
        squared_dist = torch.sum(diff ** 2, dim=channel_dim)
        euclidean_dist = torch.sqrt(squared_dist)

        # Average over all pixels (and batch if present)
        variance = torch.mean(euclidean_dist).item()

        return variance

    def should_subdivide(self, node: QuadtreeNode, variance: float) -> bool:
        """Determine if a node should be subdivided"""
        # Don't subdivide if at max depth
        if node.depth >= self.max_depth:
            return False

        # Don't subdivide if tile would be too small
        half_w_aligned = ((node.w // 2) // 8) * 8
        half_h_aligned = ((node.h // 2) // 8) * 8

        if half_w_aligned < max(self.min_tile_size, 8) or half_h_aligned < max(self.min_tile_size, 8):
            return False

        # NEW: Check 8-pixel alignment for BOTH halves and remainders
        # This prevents creating children that violate VAE's 8-pixel alignment requirement
        remainder_w = node.w - half_w_aligned
        remainder_h = node.h - half_h_aligned

        # All children must be >=8 and multiples of 8
        if (half_w_aligned < 8 or half_h_aligned < 8 or
            remainder_w < 8 or remainder_h < 8 or
            remainder_w % 8 != 0 or remainder_h % 8 != 0):
            return False

        # Subdivide if variance is above threshold
        return variance > self.content_threshold

    def build_tree(self, tensor: torch.Tensor, root_node: QuadtreeNode = None) -> QuadtreeNode:
        """
        Recursively build quadtree from tensor

        Args:
            tensor: Input tensor (B, C, H, W) or (C, H, W)
            root_node: Root node (created automatically if None)

        Returns:
            Root node of the built quadtree
        """
        # Create root node if not provided
        if root_node is None:
            if tensor.dim() == 4:
                _, _, h, w = tensor.shape
            else:
                _, h, w = tensor.shape

            # Create square root covering entire image (power-of-2 multiple of 8)
            root_size = max(w, h)

            if root_size <= 8:
                root_size = 8
            else:
                n = math.ceil(math.log2(root_size / 8))
                root_size = 8 * (2 ** n)

            root_node = QuadtreeNode(0, 0, root_size, root_size, 0)

        # Calculate variance for this node
        variance = self.calculate_variance(tensor, root_node.x, root_node.y, root_node.w, root_node.h)
        root_node.variance = variance

        # Decide whether to subdivide
        if self.should_subdivide(root_node, variance):
            # Subdivide into 4 children
            root_node.subdivide()

            # Recursively build children
            for child in root_node.children:
                self.build_tree(tensor, child)

        return root_node

    def get_leaf_nodes(self, node: QuadtreeNode, leaves: list = None) -> list:
        """Get all leaf nodes from the tree"""
        if leaves is None:
            leaves = []

        if node.is_leaf():
            leaves.append(node)
        else:
            for child in node.children:
                self.get_leaf_nodes(child, leaves)

        return leaves

    def assign_denoise_values(self, root_node: QuadtreeNode):
        """
        Assign denoise values based on tile size
        Larger tiles → lower denoise (preserve)
        Smaller tiles → higher denoise (regenerate)
        """
        leaves = self.get_leaf_nodes(root_node)
        self.max_tile_area = max(leaf.w * leaf.h for leaf in leaves)

        for leaf in leaves:
            tile_area = leaf.w * leaf.h
            size_ratio = tile_area / self.max_tile_area
            leaf.denoise = self.min_denoise + (self.max_denoise - self.min_denoise) * (1.0 - size_ratio)

    def build(self, tensor: torch.Tensor) -> tuple:
        """
        Build complete quadtree and return leaf nodes

        Args:
            tensor: Input tensor to analyze

        Returns:
            (root_node, leaf_nodes) tuple
        """
        root = self.build_tree(tensor)
        self.assign_denoise_values(root)
        leaves = self.get_leaf_nodes(root)

        print(f'[Quadtree]: Built quadtree with {len(leaves)} tiles')

        return root, leaves

# ==================== End Quadtree Classes ====================

class Device: ...
devices = Device()
devices.device = comfy.model_management.get_torch_device()
devices.cpu = torch.device('cpu')
devices.torch_gc = lambda: comfy.model_management.soft_empty_cache()
devices.get_optimal_device = lambda: comfy.model_management.get_torch_device()

class NansException(Exception): ...
def test_for_nans(x, where):
    if disable_nan_check:
        return
    if not torch.all(torch.isnan(x)).item():
        return
    if where == "unet":
        message = "A tensor with all NaNs was produced in Unet."
        if comfy.model_management.unet_dtype(x.device) != torch.float32:
            message += " This could be either because there's not enough precision to represent the picture, or because your video card does not support half type. Try setting the \"Upcast cross attention layer to float32\" option in Settings > Stable Diffusion or using the --no-half commandline argument to fix this."
    elif where == "vae":
        message = "A tensor with all NaNs was produced in VAE."
        if comfy.model_management.unet_dtype(x.device) != torch.float32 and comfy.model_management.vae_dtype()  != torch.float32:
            message += " This could be because there's not enough precision to represent the picture. Try adding --no-half-vae commandline argument to fix this."
    else:
        message = "A tensor with all NaNs was produced."
    message += " Use --disable-nan-check commandline argument to disable this check."
    raise NansException(message)

def _autocast(disable=False):
    if disable:
        return contextlib.nullcontext()

    if comfy.model_management.unet_dtype() == torch.float32 or comfy.model_management.get_torch_device() == torch.device("mps"): # or shared.cmd_opts.precision == "full":
        return contextlib.nullcontext()

    # only cuda
    autocast_device = comfy.model_management.get_autocast_device(comfy.model_management.get_torch_device())
    return torch.autocast(autocast_device)

def without_autocast(disable=False):
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()

devices.test_for_nans = test_for_nans
devices.autocast = _autocast
devices.without_autocast = without_autocast

def cheap_approximation(sample):
    # https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/2

    if is_sdxl:
        coeffs = [
            [ 0.3448,  0.4168,  0.4395],
            [-0.1953, -0.0290,  0.0250],
            [ 0.1074,  0.0886, -0.0163],
            [-0.3730, -0.2499, -0.2088],
        ]
    else:
        coeffs = [
            [ 0.298,  0.207,  0.208],
            [ 0.187,  0.286,  0.173],
            [-0.158,  0.189,  0.264],
            [-0.184, -0.271, -0.473],
        ]

    coefs = torch.tensor(coeffs).to(sample.device)

    x_sample = torch.einsum("...lxy,lr -> ...rxy", sample, coefs)

    return x_sample

def get_rcmd_enc_tsize():
    if torch.cuda.is_available() and devices.device not in ['cpu', devices.cpu]:
        total_memory = torch.cuda.get_device_properties(devices.device).total_memory // 2**20
        if   total_memory > 16*1000: ENCODER_TILE_SIZE = 3072
        elif total_memory > 12*1000: ENCODER_TILE_SIZE = 2048
        elif total_memory >  8*1000: ENCODER_TILE_SIZE = 1536
        else:                        ENCODER_TILE_SIZE = 960
    else:                            ENCODER_TILE_SIZE = 512
    return ENCODER_TILE_SIZE


def get_rcmd_dec_tsize():
    if torch.cuda.is_available() and devices.device not in ['cpu', devices.cpu]:
        total_memory = torch.cuda.get_device_properties(devices.device).total_memory // 2**20
        if   total_memory > 30*1000: DECODER_TILE_SIZE = 256
        elif total_memory > 16*1000: DECODER_TILE_SIZE = 192
        elif total_memory > 12*1000: DECODER_TILE_SIZE = 128
        elif total_memory >  8*1000: DECODER_TILE_SIZE = 96
        else:                        DECODER_TILE_SIZE = 64
    else:                            DECODER_TILE_SIZE = 64
    return DECODER_TILE_SIZE


def inplace_nonlinearity(x):
    # Test: fix for Nans
    return F.silu(x, inplace=True)

def _attn_forward(self, x):
    # From comfy.Idm.modules.diffusionmodules.model.AttnBlock.forward
    # However, the residual & normalization are removed and computed separately.
    h_ = x
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    h_ = self.optimized_attention(q, k, v)
    h_ = self.proj_out(h_)
    return h_

def get_attn_func():
    return _attn_forward

def attn2task(task_queue, net):
    
    attn_forward = get_attn_func()
    task_queue.append(('store_res', lambda x: x))
    task_queue.append(('pre_norm', net.norm))
    task_queue.append(('attn', lambda x, net=net: attn_forward(net, x)))
    task_queue.append(['add_res', None])


def resblock2task(queue, block):
    """
    Turn a ResNetBlock into a sequence of tasks and append to the task queue

    @param queue: the target task queue
    @param block: ResNetBlock

    """
    if block.in_channels != block.out_channels:
        if block.use_conv_shortcut:
            queue.append(('store_res', block.conv_shortcut))
        else:
            queue.append(('store_res', block.nin_shortcut))
    else:
        queue.append(('store_res', lambda x: x))
    queue.append(('pre_norm', block.norm1))
    queue.append(('silu', inplace_nonlinearity))
    queue.append(('conv1', block.conv1))
    queue.append(('pre_norm', block.norm2))
    queue.append(('silu', inplace_nonlinearity))
    queue.append(('conv2', block.conv2))
    queue.append(['add_res', None])


def build_sampling(task_queue, net, is_decoder):
    """
    Build the sampling part of a task queue
    @param task_queue: the target task queue
    @param net: the network
    @param is_decoder: currently building decoder or encoder
    """
    if is_decoder:
        resblock2task(task_queue, net.mid.block_1)
        attn2task(task_queue, net.mid.attn_1)
        resblock2task(task_queue, net.mid.block_2)
        resolution_iter = reversed(range(net.num_resolutions))
        block_ids = net.num_res_blocks + 1
        condition = 0
        module = net.up
        func_name = 'upsample'
    else:
        resolution_iter = range(net.num_resolutions)
        block_ids = net.num_res_blocks
        condition = net.num_resolutions - 1
        module = net.down
        func_name = 'downsample'

    for i_level in resolution_iter:
        for i_block in range(block_ids):
            resblock2task(task_queue, module[i_level].block[i_block])
        if i_level != condition:
            task_queue.append((func_name, getattr(module[i_level], func_name)))

    if not is_decoder:
        resblock2task(task_queue, net.mid.block_1)
        attn2task(task_queue, net.mid.attn_1)
        resblock2task(task_queue, net.mid.block_2)


def build_task_queue(net, is_decoder):
    """
    Build a single task queue for the encoder or decoder
    @param net: the VAE decoder or encoder network
    @param is_decoder: currently building decoder or encoder
    @return: the task queue
    """
    task_queue = []
    task_queue.append(('conv_in', net.conv_in))

    # construct the sampling part of the task queue
    # because encoder and decoder share the same architecture, we extract the sampling part
    build_sampling(task_queue, net, is_decoder)

    if not is_decoder or not net.give_pre_end:
        task_queue.append(('pre_norm', net.norm_out))
        task_queue.append(('silu', inplace_nonlinearity))
        task_queue.append(('conv_out', net.conv_out))
        if is_decoder and net.tanh_out:
            task_queue.append(('tanh', torch.tanh))

    return task_queue


def clone_task_queue(task_queue):
    """
    Clone a task queue
    @param task_queue: the task queue to be cloned
    @return: the cloned task queue
    """
    return [[item for item in task] for task in task_queue]


def get_var_mean(input, num_groups, eps=1e-6):
    """
    Get mean and var for group norm
    """
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c/num_groups)
    input_reshaped = input.contiguous().view(1, int(b * num_groups), channel_in_group, *input.size()[2:])
    var, mean = torch.var_mean(input_reshaped, dim=[0, 2, 3, 4], unbiased=False)
    return var, mean


def custom_group_norm(input, num_groups, mean, var, weight=None, bias=None, eps=1e-6):
    """
    Custom group norm with fixed mean and var

    @param input: input tensor
    @param num_groups: number of groups. by default, num_groups = 32
    @param mean: mean, must be pre-calculated by get_var_mean
    @param var: var, must be pre-calculated by get_var_mean
    @param weight: weight, should be fetched from the original group norm
    @param bias: bias, should be fetched from the original group norm
    @param eps: epsilon, by default, eps = 1e-6 to match the original group norm

    @return: normalized tensor
    """
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c/num_groups)
    input_reshaped = input.contiguous().view(
        1, int(b * num_groups), channel_in_group, *input.size()[2:])

    out = F.batch_norm(input_reshaped, mean.to(input), var.to(input), weight=None, bias=None, training=False, momentum=0, eps=eps)
    out = out.view(b, c, *input.size()[2:])

    # post affine transform
    if weight is not None:
        out *= weight.view(1, -1, 1, 1).to(device=devices.device)
    if bias is not None:
        out += bias.view(1, -1, 1, 1).to(device=devices.device)
    return out


def crop_valid_region(x, input_bbox, target_bbox, is_decoder):
    """
    Crop the valid region from the tile
    @param x: input tile
    @param input_bbox: original input bounding box
    @param target_bbox: output bounding box
    @param scale: scale factor
    @return: cropped tile
    """
    padded_bbox = [i * 8 if is_decoder else i//8 for i in input_bbox]
    margin = [target_bbox[i] - padded_bbox[i] for i in range(4)]
    return x[:, :, margin[2]:x.size(2)+margin[3], margin[0]:x.size(3)+margin[1]]


# ↓↓↓ https://github.com/Kahsolt/stable-diffusion-webui-vae-tile-infer ↓↓↓

def perfcount(fn):
    def wrapper(*args, **kwargs):
        ts = time()

        if torch.cuda.is_available() and devices.device not in ['cpu', devices.cpu]:
            torch.cuda.reset_peak_memory_stats(devices.device)
        devices.torch_gc()
        gc.collect()

        ret = fn(*args, **kwargs)

        devices.torch_gc()
        gc.collect()
        if torch.cuda.is_available() and devices.device not in ['cpu', devices.cpu]:
            vram = torch.cuda.max_memory_allocated(devices.device) / 2**20
            print(f'[Quadtree VAE]: Done in {time() - ts:.3f}s, max VRAM alloc {vram:.3f} MB')
        else:
            print(f'[Quadtree VAE]: Done in {time() - ts:.3f}s')

        return ret
    return wrapper

# ↑↑↑ https://github.com/Kahsolt/stable-diffusion-webui-vae-tile-infer ↑↑↑


class GroupNormParam:

    def __init__(self):
        self.var_list = []
        self.mean_list = []
        self.pixel_list = []
        self.weight = None
        self.bias = None

    def add_tile(self, tile, layer):
        var, mean = get_var_mean(tile, 32)
        # For giant images, the variance can be larger than max float16
        # In this case we create a copy to float32
        if var.dtype == torch.float16 and var.isinf().any():
            fp32_tile = tile.float()
            var, mean = get_var_mean(fp32_tile, 32)
        # ============= DEBUG: test for infinite =============
        # if torch.isinf(var).any():
        #    print('[Tiled VAE]: inf test', var)
        # ====================================================
        self.var_list.append(var)
        self.mean_list.append(mean)
        self.pixel_list.append(
            tile.shape[2]*tile.shape[3])
        if hasattr(layer, 'weight'):
            self.weight = layer.weight
            self.bias = layer.bias
        else:
            self.weight = None
            self.bias = None

    def summary(self):
        """
        summarize the mean and var and return a function
        that apply group norm on each tile
        """
        if len(self.var_list) == 0: return None

        var = torch.vstack(self.var_list)
        mean = torch.vstack(self.mean_list)
        max_value = max(self.pixel_list)
        pixels = torch.tensor(self.pixel_list, dtype=torch.float32, device=devices.device) / max_value
        sum_pixels = torch.sum(pixels)
        pixels = pixels.unsqueeze(1) / sum_pixels
        var = torch.sum(var * pixels, dim=0)
        mean = torch.sum(mean * pixels, dim=0)
        return lambda x:  custom_group_norm(x, 32, mean, var, self.weight, self.bias)

    @staticmethod
    def from_tile(tile, norm):
        """
        create a function from a single tile without summary
        """
        var, mean = get_var_mean(tile, 32)
        if var.dtype == torch.float16 and var.isinf().any():
            fp32_tile = tile.float()
            var, mean = get_var_mean(fp32_tile, 32)
            # if it is a macbook, we need to convert back to float16
            if var.device.type == 'mps':
                # clamp to avoid overflow
                var = torch.clamp(var, 0, 60000)
                var = var.half()
                mean = mean.half()
        if hasattr(norm, 'weight'):
            weight = norm.weight
            bias = norm.bias
        else:
            weight = None
            bias = None

        def group_norm_func(x, mean=mean, var=var, weight=weight, bias=bias):
            return custom_group_norm(x, 32, mean, var, weight, bias, 1e-6)
        return group_norm_func


class VAEHook:

    def __init__(self, net, tile_size, is_decoder:bool, fast_decoder:bool, fast_encoder:bool, color_fix:bool, to_gpu:bool=False,
                 use_quadtree:bool=False, content_threshold:float=0.03, max_depth:int=4, min_tile_size:int=128):
        self.net = net                  # encoder | decoder
        self.tile_size = tile_size
        self.is_decoder = is_decoder
        self.fast_mode = (fast_encoder and not is_decoder) or (fast_decoder and is_decoder)
        self.color_fix = color_fix and not is_decoder
        self.to_gpu = to_gpu
        self.pad = 11 if is_decoder else 32         # FIXME: magic number

        # Quadtree parameters
        self.use_quadtree = use_quadtree
        self.content_threshold = content_threshold
        self.max_depth = max_depth
        self.min_tile_size_quadtree = min_tile_size
        self.quadtree_root = None  # Store quadtree for reuse
        self.quadtree_leaves = None

    def __call__(self, x):
        # original_device = next(self.net.parameters()).device
        try:
            # if self.to_gpu:
            #     self.net = self.net.to(devices.get_optimal_device())
            B, C, H, W = x.shape
            if False:#max(H, W) <= self.pad * 2 + self.tile_size:
                print("[Quadtree VAE]: the input size is tiny and unnecessary to tile.", x.shape, self.pad * 2 + self.tile_size)
                return self.net.original_forward(x)
            else:
                return self.vae_tile_forward(x)
        finally:
            pass
            # self.net = self.net.to(original_device)

    def get_best_tile_size(self, lowerbound, upperbound):
        """
        Get the best tile size for GPU memory
        """
        divider = 32
        while divider >= 2:
            remainer = lowerbound % divider
            if remainer == 0:
                return lowerbound
            candidate = lowerbound - remainer + divider
            if candidate <= upperbound:
                return candidate
            divider //= 2
        return lowerbound

    def split_tiles(self, h, w):
        """
        Tool function to split the image into tiles
        @param h: height of the image
        @param w: width of the image
        @return: tile_input_bboxes, tile_output_bboxes
        """
        tile_input_bboxes, tile_output_bboxes = [], []
        tile_size = self.tile_size
        pad = self.pad
        num_height_tiles = math.ceil((h - 2 * pad) / tile_size)
        num_width_tiles = math.ceil((w - 2 * pad) / tile_size)
        # If any of the numbers are 0, we let it be 1
        # This is to deal with long and thin images
        num_height_tiles = max(num_height_tiles, 1)
        num_width_tiles = max(num_width_tiles, 1)

        # Suggestions from https://github.com/Kahsolt: auto shrink the tile size
        real_tile_height = math.ceil((h - 2 * pad) / num_height_tiles)
        real_tile_width = math.ceil((w - 2 * pad) / num_width_tiles)
        real_tile_height = self.get_best_tile_size(real_tile_height, tile_size)
        real_tile_width = self.get_best_tile_size(real_tile_width, tile_size)

        print(f'[Quadtree VAE]: split to {num_height_tiles}x{num_width_tiles} = {num_height_tiles*num_width_tiles} tiles. ' +
              f'Optimal tile size {real_tile_width}x{real_tile_height}, original tile size {tile_size}x{tile_size}')

        for i in range(num_height_tiles):
            for j in range(num_width_tiles):
                # bbox: [x1, x2, y1, y2]
                # the padding is is unnessary for image borders. So we directly start from (32, 32)
                input_bbox = [
                    pad + j * real_tile_width,
                    min(pad + (j + 1) * real_tile_width, w),
                    pad + i * real_tile_height,
                    min(pad + (i + 1) * real_tile_height, h),
                ]

                # if the output bbox is close to the image boundary, we extend it to the image boundary
                output_bbox = [
                    input_bbox[0] if input_bbox[0] > pad else 0,
                    input_bbox[1] if input_bbox[1] < w - pad else w,
                    input_bbox[2] if input_bbox[2] > pad else 0,
                    input_bbox[3] if input_bbox[3] < h - pad else h,
                ]

                # scale to get the final output bbox
                output_bbox = [x * 8 if self.is_decoder else x // 8 for x in output_bbox]
                tile_output_bboxes.append(output_bbox)

                # indistinguishable expand the input bbox by pad pixels
                tile_input_bboxes.append([
                    max(0, input_bbox[0] - pad),
                    min(w, input_bbox[1] + pad),
                    max(0, input_bbox[2] - pad),
                    min(h, input_bbox[3] + pad),
                ])

        return tile_input_bboxes, tile_output_bboxes

    def split_tiles_quadtree(self, h, w, z_tensor):
        """
        Split image into adaptive quadtree tiles based on content variance

        @param h: height of the image (in current space - latent for decoder, image for encoder)
        @param w: width of the image
        @param z_tensor: tensor to analyze for variance (B, C, H, W)
        @return: tile_input_bboxes, tile_output_bboxes
        """
        pad = self.pad

        # Build quadtree from the tensor
        builder = QuadtreeBuilder(
            content_threshold=self.content_threshold,
            max_depth=self.max_depth,
            min_tile_size=self.min_tile_size_quadtree,
            min_denoise=0.0,  # Will be set by diffusion
            max_denoise=1.0
        )

        root, leaves = builder.build(z_tensor)

        # Store quadtree for potential reuse by diffusion
        self.quadtree_root = root
        self.quadtree_leaves = leaves

        tile_input_bboxes, tile_output_bboxes = [], []

        for leaf in leaves:
            # Quadtree leaves are in the current tensor's space (h x w)
            # For decoder: latent space. For encoder: image space.
            x1, x2, y1, y2 = leaf.x, leaf.x + leaf.w, leaf.y, leaf.y + leaf.h

            # Define the core tile region (without padding) - this is what we want in the output
            core_bbox = [x1, x2, y1, y2]

            # Create output bbox - handle image borders by extending to edge
            output_bbox = [
                core_bbox[0] if core_bbox[0] > pad else 0,
                core_bbox[1] if core_bbox[1] < w - pad else w,
                core_bbox[2] if core_bbox[2] > pad else 0,
                core_bbox[3] if core_bbox[3] < h - pad else h,
            ]

            # Scale output bbox for the target space (decoder: latent→image 8x, encoder: image→latent /8)
            output_bbox = [x * 8 if self.is_decoder else x // 8 for x in output_bbox]
            tile_output_bboxes.append(output_bbox)

            # Expand core bbox by padding to create input bbox
            # This is the region we'll actually process (larger due to padding)
            input_bbox_padded = [
                max(0, core_bbox[0] - pad),
                min(w, core_bbox[1] + pad),
                max(0, core_bbox[2] - pad),
                min(h, core_bbox[3] + pad),
            ]
            tile_input_bboxes.append(input_bbox_padded)

        return tile_input_bboxes, tile_output_bboxes

    @torch.no_grad()
    def estimate_group_norm(self, z, task_queue, color_fix):
        device = z.device
        tile = z
        last_id = len(task_queue) - 1
        while last_id >= 0 and task_queue[last_id][0] != 'pre_norm':
            last_id -= 1
        if last_id <= 0 or task_queue[last_id][0] != 'pre_norm':
            raise ValueError('No group norm found in the task queue')
        # estimate until the last group norm
        for i in range(last_id + 1):
            task = task_queue[i]
            if task[0] == 'pre_norm':
                group_norm_func = GroupNormParam.from_tile(tile, task[1])
                task_queue[i] = ('apply_norm', group_norm_func)
                if i == last_id:
                    return True
                tile = group_norm_func(tile)
            elif task[0] == 'store_res':
                task_id = i + 1
                while task_id < last_id and task_queue[task_id][0] != 'add_res':
                    task_id += 1
                if task_id >= last_id:
                    continue
                task_queue[task_id][1] = task[1](tile)
            elif task[0] == 'add_res':
                tile += task[1].to(device)
                task[1] = None
            elif color_fix and task[0] == 'downsample':
                for j in range(i, last_id + 1):
                    if task_queue[j][0] == 'store_res':
                        task_queue[j] = ('store_res_cpu', task_queue[j][1])
                return True
            else:
                tile = task[1](tile)
            try:
                devices.test_for_nans(tile, "vae")
            except:
                print(f'Nan detected in fast mode estimation. Fast mode disabled.')
                return False

        raise IndexError('Should not reach here')

    @perfcount
    @torch.no_grad()
    def vae_tile_forward(self, z):
        """
        Decode a latent vector z into an image in a tiled manner.
        @param z: latent vector
        @return: image
        """
        device = devices.device # next(self.net.parameters()).device
        net = self.net
        tile_size = self.tile_size
        is_decoder = self.is_decoder

        z = z.detach() # detach the input to avoid backprop

        N, height, width = z.shape[0], z.shape[2], z.shape[3]
        net.last_z_shape = z.shape

        # Split the input into tiles and build a task queue for each tile
        print(f'[Quadtree VAE]: input_size: {z.shape}, tile_size: {tile_size}, padding: {self.pad}')

        # Choose tiling method: quadtree or regular grid
        if self.use_quadtree:
            in_bboxes, out_bboxes = self.split_tiles_quadtree(height, width, z)
        else:
            in_bboxes, out_bboxes = self.split_tiles(height, width)

        # Prepare tiles by split the input latents
        tiles = []
        for input_bbox in in_bboxes:
            # Extract tile with clamping to image boundaries
            x1, x2, y1, y2 = input_bbox
            x1_clamped = max(0, x1)
            x2_clamped = min(width, x2)
            y1_clamped = max(0, y1)
            y2_clamped = min(height, y2)

            # Extract the available region
            tile = z[:, :, y1_clamped:y2_clamped, x1_clamped:x2_clamped]

            # Check if padding is needed (tile extends beyond image)
            pad_left = max(0, -x1)
            pad_right = max(0, x2 - width)
            pad_top = max(0, -y1)
            pad_bottom = max(0, y2 - height)

            # Apply padding if needed
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                # PyTorch pad order: (left, right, top, bottom)

                # Reflection padding has a limit: padding must be < dimension size
                # For tiles that extend far beyond image, use replicate mode instead
                _, _, tile_h, tile_w = tile.shape

                # Check if reflection padding is possible
                can_reflect_h = (pad_top < tile_h) and (pad_bottom < tile_h)
                can_reflect_w = (pad_left < tile_w) and (pad_right < tile_w)

                if can_reflect_h and can_reflect_w:
                    # Use reflection for smoother boundaries
                    tile = F.pad(tile, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
                else:
                    # Fall back to replicate for tiles extending far beyond image
                    tile = F.pad(tile, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')

            tiles.append(tile.cpu())

        num_tiles = len(tiles)
        num_completed = 0

        # Build task queues
        single_task_queue = build_task_queue(net, is_decoder)
        if self.fast_mode:
            # Fast mode: downsample the input image to the tile size,
            # then estimate the group norm parameters on the downsampled image
            scale_factor = tile_size / max(height, width)
            z = z.to(device)
            downsampled_z = F.interpolate(z, scale_factor=scale_factor, mode='nearest-exact')
            # use nearest-exact to keep statictics as close as possible
            print(f'[Quadtree VAE]: Fast mode enabled, estimating group norm parameters on {downsampled_z.shape[3]} x {downsampled_z.shape[2]} image')

            # ======= Special thanks to @Kahsolt for distribution shift issue ======= #
            # The downsampling will heavily distort its mean and std, so we need to recover it.
            std_old, mean_old = torch.std_mean(z, dim=[0, 2, 3], keepdim=True)
            std_new, mean_new = torch.std_mean(downsampled_z, dim=[0, 2, 3], keepdim=True)
            downsampled_z = (downsampled_z - mean_new) / std_new * std_old + mean_old
            del std_old, mean_old, std_new, mean_new
            # occasionally the std_new is too small or too large, which exceeds the range of float16
            # so we need to clamp it to max z's range.
            downsampled_z = torch.clamp_(downsampled_z, min=z.min(), max=z.max())
            estimate_task_queue = clone_task_queue(single_task_queue)
            if self.estimate_group_norm(downsampled_z, estimate_task_queue, color_fix=self.color_fix):
                single_task_queue = estimate_task_queue
            del downsampled_z

        task_queues = [clone_task_queue(single_task_queue) for _ in range(num_tiles)]

        # Dummy result
        result = None
        result_approx = None
        try:
            with devices.autocast():
                result_approx = torch.cat([F.interpolate(cheap_approximation(x).unsqueeze(0), scale_factor=opt_f, mode='nearest-exact') for x in z], dim=0).cpu()
        except: pass
        # Free memory of input latent tensor
        del z

        # Task queue execution
        pbar = tqdm(total=num_tiles * len(task_queues[0]), desc=f"[Quadtree VAE]: Executing {'Decoder' if is_decoder else 'Encoder'} Task Queue: ")
        pbar_comfy = comfy.utils.ProgressBar(num_tiles * len(task_queues[0]))

        # execute the task back and forth when switch tiles so that we always
        # keep one tile on the GPU to reduce unnecessary data transfer
        forward = True
        interrupted = False
        state_interrupted = processing_interrupted()
        #state.interrupted = interrupted
        while True:
            if state_interrupted: interrupted = True ; break

            group_norm_param = GroupNormParam()
            for i in range(num_tiles) if forward else reversed(range(num_tiles)):
                if state_interrupted: interrupted = True ; break

                tile = tiles[i].to(device)
                input_bbox = in_bboxes[i]
                task_queue = task_queues[i]

                interrupted = False
                while len(task_queue) > 0:
                    if state_interrupted: interrupted = True ; break

                    # DEBUG: current task
                    # print('Running task: ', task_queue[0][0], ' on tile ', i, '/', num_tiles, ' with shape ', tile.shape)
                    task = task_queue.pop(0)
                    if task[0] == 'pre_norm':
                        group_norm_param.add_tile(tile, task[1])
                        break
                    elif task[0] == 'store_res' or task[0] == 'store_res_cpu':
                        task_id = 0
                        res = task[1](tile)
                        if not self.fast_mode or task[0] == 'store_res_cpu':
                            res = res.cpu()
                        while task_queue[task_id][0] != 'add_res':
                            task_id += 1
                        task_queue[task_id][1] = res
                    elif task[0] == 'add_res':
                        tile += task[1].to(device)
                        task[1] = None
                    else:
                        tile = task[1](tile)
                    pbar.update(1)
                    pbar_comfy.update(1)


                if interrupted: break

                # check for NaNs in the tile.
                # If there are NaNs, we abort the process to save user's time
                devices.test_for_nans(tile, "vae")

                if len(task_queue) == 0:
                    tiles[i] = None
                    num_completed += 1
                    if result is None:      # NOTE: dim C varies from different cases, can only be inited dynamically
                        result = torch.zeros((N, tile.shape[1], height * 8 if is_decoder else height // 8, width * 8 if is_decoder else width // 8), device=device, requires_grad=False)
                    result[:, :, out_bboxes[i][2]:out_bboxes[i][3], out_bboxes[i][0]:out_bboxes[i][1]] = crop_valid_region(tile, in_bboxes[i], out_bboxes[i], is_decoder)
                    del tile
                elif i == num_tiles - 1 and forward:
                    forward = False
                    tiles[i] = tile
                elif i == 0 and not forward:
                    forward = True
                    tiles[i] = tile
                else:
                    tiles[i] = tile.cpu()
                    del tile

            if interrupted: break
            if num_completed == num_tiles: break

            # insert the group norm task to the head of each task queue
            group_norm_func = group_norm_param.summary()
            if group_norm_func is not None:
                for i in range(num_tiles):
                    task_queue = task_queues[i]
                    task_queue.insert(0, ('apply_norm', group_norm_func))

        # Done!
        pbar.close()
        if interrupted:
            del result, result_approx
            comfy.model_management.throw_exception_if_processing_interrupted()
        vae_dtype = self.net.conv_in.weight.dtype
        return result.to(dtype=vae_dtype, device=device) if result is not None else result_approx.to(device=device, dtype=vae_dtype)

# from .tiled_vae import VAEHook, get_rcmd_enc_tsize, get_rcmd_dec_tsize
from nodes import VAEEncode, VAEDecode
class TiledVAE:
    def process(self, *args, **kwargs):
        samples = kwargs['samples'] if 'samples' in kwargs else (kwargs['pixels'] if 'pixels' in kwargs else args[0])
        _vae = kwargs['vae'] if 'vae' in kwargs else args[1]
        tile_size = kwargs['tile_size'] if 'tile_size' in kwargs else args[2]
        fast = kwargs['fast'] if 'fast' in kwargs else args[3]
        color_fix = kwargs['color_fix'] if 'color_fix' in kwargs else False
        is_decoder = self.is_decoder
        devices.device = _vae.device

        # Extract quadtree parameters (with defaults if not provided)
        use_quadtree = kwargs.get('use_quadtree', False)
        content_threshold = kwargs.get('content_threshold', 0.03)
        max_depth = kwargs.get('max_depth', 4)
        min_tile_size = kwargs.get('min_tile_size', 128)

        # for shorthand
        vae = _vae.first_stage_model
        encoder = vae.encoder
        decoder = vae.decoder
        
        # # undo hijack if disabled (in cases last time crashed)
        # if not enabled:
        #     if self.hooked:
        if isinstance(encoder.forward, VAEHook):
            encoder.forward.net = None
            encoder.forward = encoder.original_forward
        if isinstance(decoder.forward, VAEHook):
            decoder.forward.net = None
            decoder.forward = decoder.original_forward
        #         self.hooked = False
        #     return

        # if devices.get_optimal_device_name().startswith('cuda') and vae.device == devices.cpu and not vae_to_gpu:
        #     print("[Tiled VAE] warn: VAE is not on GPU, check 'Move VAE to GPU' if possible.")

        # do hijack
        # kwargs = {
        #     'fast_decoder': fast_decoder, 
        #     'fast_encoder': fast_encoder, 
        #     'color_fix':    color_fix, 
        #     'to_gpu':       vae_to_gpu,
        # }

        # save original forward (only once)
        if not hasattr(encoder, 'original_forward'): setattr(encoder, 'original_forward', encoder.forward)
        if not hasattr(decoder, 'original_forward'): setattr(decoder, 'original_forward', decoder.forward)

        # self.hooked = True
        
        # encoder.forward = VAEHook(encoder, encoder_tile_size, is_decoder=False, **kwargs)
        # decoder.forward = VAEHook(decoder, decoder_tile_size, is_decoder=True,  **kwargs)
        fn = VAEHook(net=decoder if is_decoder else encoder, tile_size=tile_size // 8 if is_decoder else tile_size,
                        is_decoder=is_decoder, fast_decoder=fast, fast_encoder=fast,
                        color_fix=color_fix, to_gpu=comfy.model_management.vae_device().type != 'cpu',
                        use_quadtree=use_quadtree, content_threshold=content_threshold,
                        max_depth=max_depth, min_tile_size=min_tile_size)
        if is_decoder:
            decoder.forward = fn
        else:
            encoder.forward = fn

        ret = (None,)
        try:
            with devices.without_autocast():
                if not is_decoder:
                    ret = VAEEncode().encode(_vae, samples)
                else:
                    ret = VAEDecode().decode(_vae, samples) if is_decoder else VAEEncode().encode(_vae, samples)
        finally:
            if isinstance(encoder.forward, VAEHook):
                encoder.forward.net = None
                encoder.forward = encoder.original_forward
            if isinstance(decoder.forward, VAEHook):
                decoder.forward.net = None
                decoder.forward = decoder.original_forward
        return ret

class VAEEncodeTiled_TiledDiffusion(TiledVAE):
    @classmethod
    def INPUT_TYPES(s):
        fast = True
        tile_size = get_rcmd_enc_tsize()
        return {"required": {
                    "pixels": ("IMAGE", ),
                    "vae": ("VAE", ),
                    "tile_size": ("INT", {"default": tile_size, "min": 256, "max": 4096, "step": 16}),
                    "fast": ("BOOLEAN", {"default": fast}),
                    "color_fix": ("BOOLEAN", {"default": fast}),
                }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "process"
    CATEGORY = "_for_testing"

    def __init__(self):
        self.is_decoder = False
        super().__init__()

class VAEDecodeTiled_TiledDiffusion(TiledVAE):
    @classmethod
    def INPUT_TYPES(s):
        tile_size = get_rcmd_dec_tsize() * opt_f
        return {"required": {
                    "samples": ("LATENT", ),
                    "vae": ("VAE", ),
                    "tile_size": ("INT", {"default": tile_size, "min": 48*opt_f, "max": 4096, "step": 16}),
                    "fast": ("BOOLEAN", {"default": True}),
                }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "_for_testing"

    def __init__(self):
        self.is_decoder = True
        super().__init__()

class QuadtreeVisualizer:
    """Debug node to visualize quadtree structure on an image"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "content_threshold": ("FLOAT", {
                    "default": 0.03,
                    "min": 0.001,
                    "max": 0.5,
                    "step": 0.001,
                    "tooltip": "Variance threshold for subdivision. Lower = more tiles in detailed areas."
                }),
                "max_depth": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Maximum quadtree depth"
                }),
                "min_tile_size": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 1024,
                    "step": 8,
                    "tooltip": "Minimum tile size in pixels"
                }),
                "min_denoise": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Denoise strength for largest tiles (low complexity areas). 0.0 = preserve completely, 1.0 = regenerate completely"
                }),
                "max_denoise": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Denoise strength for smallest tiles (high complexity areas). Should be >= min_denoise"
                }),
                "line_thickness": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Thickness of quadtree boundary lines"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "QUADTREE",)
    FUNCTION = "visualize"
    CATEGORY = "_for_testing/quadtree"

    def visualize(self, image, content_threshold, max_depth, min_tile_size,
                  min_denoise, max_denoise, line_thickness):
        """
        Visualize the quadtree structure overlaid on the input image

        Args:
            image: Input image tensor (B, H, W, C)
            content_threshold: Variance threshold for subdivision
            max_depth: Maximum tree depth
            min_tile_size: Minimum tile size
            min_denoise: Min denoise value for large tiles
            max_denoise: Max denoise value for small tiles
            line_thickness: Thickness of boundary lines

        Returns:
            Image with quadtree overlay
        """
        import numpy as np

        # ComfyUI images are (B, H, W, C) with values in [0, 1]
        batch_size = image.shape[0]
        results = []

        for b in range(batch_size):
            # Get single image (H, W, C)
            img = image[b].clone()
            h, w, c = img.shape

            # Convert to tensor format for quadtree builder (C, H, W)
            img_tensor = img.permute(2, 0, 1)

            # Build quadtree
            builder = QuadtreeBuilder(
                content_threshold=content_threshold,
                max_depth=max_depth,
                min_tile_size=min_tile_size,
                min_denoise=min_denoise,
                max_denoise=max_denoise
            )
            root, leaves = builder.build(img_tensor)

            # CROP EDGE TILES TO IMAGE BOUNDS
            # The quadtree creates a square root that extends beyond rectangular images
            # Instead of filtering, crop edge tiles to create rectangular tiles at boundaries
            original_leaf_count = len(leaves)
            cropped_leaves = []
            filtered_count = 0
            cropped_count = 0

            for leaf in leaves:
                # Check if tile overlaps with image at all
                if leaf.x >= w or (leaf.x + leaf.w) <= 0 or leaf.y >= h or (leaf.y + leaf.h) <= 0:
                    # Completely outside - skip it
                    filtered_count += 1
                    continue

                # Tile overlaps - crop to image bounds
                new_x = max(0, leaf.x)
                new_y = max(0, leaf.y)
                new_w = min(w, leaf.x + leaf.w) - new_x
                new_h = min(h, leaf.y + leaf.h) - new_y

                # Check if this was actually cropped
                if new_x != leaf.x or new_y != leaf.y or new_w != leaf.w or new_h != leaf.h:
                    cropped_count += 1
                    # Create new cropped leaf
                    cropped_leaf = QuadtreeNode(new_x, new_y, new_w, new_h, leaf.depth)
                    cropped_leaf.variance = leaf.variance
                    cropped_leaf.denoise = leaf.denoise
                    cropped_leaves.append(cropped_leaf)
                else:
                    # Keep original
                    cropped_leaves.append(leaf)

            leaves = cropped_leaves

            if filtered_count > 0 or cropped_count > 0:
                print(f'[Quadtree Visualizer]: Filtered {filtered_count} fully out-of-bounds leaves, cropped {cropped_count} edge tiles to fit {w}x{h} image')

            # Convert image to numpy for drawing
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)

            # Draw quadtree boundaries
            for leaf in leaves:
                # Get denoise value normalized to [0, 1]
                denoise_norm = (leaf.denoise - min_denoise) / (max_denoise - min_denoise + 1e-8)

                # Color coding: Blue (low denoise/preserve) to Red (high denoise/regenerate)
                # Blue = (0, 0, 255), Red = (255, 0, 0)
                red = int(denoise_norm * 255)
                blue = int((1 - denoise_norm) * 255)
                color = [red, 0, blue]

                # Draw rectangle boundaries
                x1, x2, y1, y2 = leaf.x, leaf.x + leaf.w, leaf.y, leaf.y + leaf.h

                # Top edge
                img_np[y1:min(y1+line_thickness, h), x1:x2] = color
                # Bottom edge
                img_np[max(y2-line_thickness, 0):y2, x1:x2] = color
                # Left edge
                img_np[y1:y2, x1:min(x1+line_thickness, w)] = color
                # Right edge
                img_np[y1:y2, max(x2-line_thickness, 0):x2] = color

            # Convert back to tensor (H, W, C) with values in [0, 1]
            result = torch.from_numpy(img_np.astype(np.float32) / 255.0)
            results.append(result)

        # Stack batch
        output = torch.stack(results, dim=0)

        # Print statistics
        print(f'[Quadtree Visualizer]: Built quadtree with {len(leaves)} tiles (after filtering)')
        print(f'[Quadtree Visualizer]: Tile dimensions range from {min(l.w for l in leaves)}x{min(l.h for l in leaves)} to {max(l.w for l in leaves)}x{max(l.h for l in leaves)}')
        print(f'[Quadtree Visualizer]: Variance values range from {min(l.variance for l in leaves):.4f} to {max(l.variance for l in leaves):.4f}')
        print(f'[Quadtree Visualizer]: Denoise values range from {min(l.denoise for l in leaves):.3f} to {max(l.denoise for l in leaves):.3f}')
        print(f'[Quadtree Visualizer]: Max depth reached: {max(l.depth for l in leaves)}')

        # Count tiles by depth to understand tree structure
        from collections import Counter
        depth_counts = Counter(l.depth for l in leaves)
        print(f'[Quadtree Visualizer]: Tiles by depth: {dict(sorted(depth_counts.items()))}')

        # Create quadtree data structure to pass to diffusion
        quadtree_data = {
            'root': root,
            'leaves': leaves,
            'content_threshold': content_threshold,
            'max_depth': max_depth,
            'min_tile_size': min_tile_size,
            'min_denoise': min_denoise,
            'max_denoise': max_denoise,
        }

        return (output, quadtree_data)

NODE_CLASS_MAPPINGS = {
    "VAEEncodeTiled_QuadtreeDiffusion": VAEEncodeTiled_TiledDiffusion,
    "VAEDecodeTiled_QuadtreeDiffusion": VAEDecodeTiled_TiledDiffusion,
    "QuadtreeVisualizer": QuadtreeVisualizer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEEncodeTiled_QuadtreeDiffusion": "Quadtree VAE Encode",
    "VAEDecodeTiled_QuadtreeDiffusion": "Quadtree VAE Decode",
    "QuadtreeVisualizer": "Quadtree Visualizer",
}
