import torch
from torch import Tensor
from typing import List, Union, Tuple, Callable, Dict
from weakref import WeakSet
from collections import defaultdict
import comfy.utils
import comfy.model_patcher
import comfy.model_management
from nodes import ImageScale
from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from comfy.controlnet import ControlNet, T2IAdapter
from comfy.utils import common_upscale
from comfy.model_management import processing_interrupted, loaded_models, load_models_gpu
from math import pi

# Import quadtree classes from tiled_vae
from .tiled_vae import QuadtreeNode, QuadtreeBuilder

opt_C = 4
opt_f = 8

def ceildiv(big, small):
    # Correct ceiling division that avoids floating-point errors and importing math.ceil.
    return -(big // -small)

from enum import Enum
class BlendMode(Enum):  # i.e. LayerType
    FOREGROUND = 'Foreground'
    BACKGROUND = 'Background'

class Processing: ...
class Device: ...
devices = Device()
devices.device = comfy.model_management.get_torch_device()

def null_decorator(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

keep_signature = null_decorator
controlnet     = null_decorator
stablesr       = null_decorator
grid_bbox      = null_decorator
custom_bbox    = null_decorator
noise_inverse  = null_decorator

class BBox:
    ''' grid bbox '''

    def __init__(self, x:int, y:int, w:int, h:int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.box = [x, y, x+w, y+h]
        self.slicer = slice(None), slice(None), slice(y, y+h), slice(x, x+w)

    def __getitem__(self, idx:int) -> int:
        return self.box[idx]

def repeat_to_batch_size(tensor, batch_size, dim=0):
    if dim == 0 and tensor.shape[dim] == 1:
        return tensor.expand([batch_size] + [-1] * (len(tensor.shape) - 1))
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    elif tensor.shape[dim] < batch_size:
        return tensor.repeat(dim * [1] + [ceildiv(batch_size, tensor.shape[dim])] + [1] * (len(tensor.shape) - 1 - dim)).narrow(dim, 0, batch_size)
    return tensor

def extract_tile_with_padding(tensor: Tensor, bbox: BBox, image_w: int, image_h: int) -> Tensor:
    """
    Extract a tile from tensor with reflection padding if it extends beyond boundaries

    Args:
        tensor: Input tensor (B, C, H, W)
        bbox: BBox defining the tile region (may extend beyond tensor)
        image_w: Actual image width
        image_h: Actual image height

    Returns:
        Extracted tile with padding applied if needed
    """
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h

    # Calculate clamped extraction region (what we can actually extract)
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(image_w, x + w)
    y_end = min(image_h, y + h)

    # Extract the available region
    tile = tensor[:, :, y_start:y_end, x_start:x_end]

    # Calculate padding needed
    pad_left = max(0, -x)
    pad_right = max(0, (x + w) - image_w)
    pad_top = max(0, -y)
    pad_bottom = max(0, (y + h) - image_h)

    # Apply padding if needed
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        import torch.nn.functional as F
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

    return tile

def split_bboxes(w:int, h:int, tile_w:int, tile_h:int, overlap:int=16, init_weight:Union[Tensor, float]=1.0) -> Tuple[List[BBox], Tensor]:
    cols = ceildiv((w - overlap) , (tile_w - overlap))
    rows = ceildiv((h - overlap) , (tile_h - overlap))
    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    bbox_list: List[BBox] = []
    weight = torch.zeros((1, 1, h, w), device=devices.device, dtype=torch.float32)
    for row in range(rows):
        y = min(int(row * dy), h - tile_h)
        for col in range(cols):
            x = min(int(col * dx), w - tile_w)

            bbox = BBox(x, y, tile_w, tile_h)
            bbox_list.append(bbox)
            weight[bbox.slicer] += init_weight

    return bbox_list, weight

def split_bboxes_quadtree(w: int, h: int, content_threshold: float, max_depth: int,
                          min_tile_size: int, min_denoise: float, max_denoise: float,
                          init_weight: Union[Tensor, float] = 1.0, device='cpu') -> Tuple[List[BBox], Tensor, List]:
    """
    Create bboxes using adaptive quadtree subdivision

    Note: Unlike image-based quadtree in VAE, this creates a FIXED quadtree structure
    based on dimensions and depth parameters (similar to how grid creates a fixed grid).
    This is because diffusion operates on noisy latents which have no meaningful structure to analyze.

    Returns:
        bbox_list: List of BBox objects for each tile
        weight: Weight tensor for blending
        leaves: List of QuadtreeNode leaf nodes (with denoise values)
    """
    # Build quadtree structure based purely on dimensions and parameters
    # We create a dummy tensor just for dimensions - content doesn't matter for diffusion
    import torch
    dummy_tensor = torch.zeros((1, 1, h, w), device=device)

    builder = QuadtreeBuilder(
        content_threshold=999.0,  # Set very high so it ALWAYS subdivides based on depth/size only
        max_depth=max_depth,
        min_tile_size=min_tile_size,
        min_denoise=min_denoise,
        max_denoise=max_denoise
    )

    root, leaves = builder.build(dummy_tensor)

    # Convert quadtree leaves to BBox list
    bbox_list: List[BBox] = []
    weight = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)

    for leaf in leaves:
        # Clamp tile to actual image bounds (since root may be larger square)
        tile_x = min(leaf.x, w - 1)
        tile_y = min(leaf.y, h - 1)
        tile_w = min(leaf.w, w - tile_x)
        tile_h = min(leaf.h, h - tile_y)

        # Skip tiles that are completely outside the image or too small
        if tile_w <= 0 or tile_h <= 0 or tile_x >= w or tile_y >= h:
            continue

        # For true quadtree, ensure tile remains square after clamping
        tile_size = min(tile_w, tile_h)

        # Create BBox from clamped dimensions
        bbox = BBox(tile_x, tile_y, tile_size, tile_size)

        # Store denoise value in bbox for later use
        bbox.denoise = leaf.denoise
        bbox.variance = leaf.variance

        bbox_list.append(bbox)
        weight[bbox.slicer] += init_weight

    print(f'[Quadtree Diffusion]: Created {len(bbox_list)} adaptive tiles')
    print(f'[Quadtree Diffusion]: Tile sizes: {min(l.w * l.h for l in leaves)} to {max(l.w * l.h for l in leaves)} pixels')
    print(f'[Quadtree Diffusion]: Variance range: {min(l.variance for l in leaves):.4f} to {max(l.variance for l in leaves):.4f}')
    print(f'[Quadtree Diffusion]: Denoise range: {min(l.denoise for l in leaves):.3f} to {max(l.denoise for l in leaves):.3f}')

    # Show distribution of tile sizes for debugging
    size_counts = {}
    for leaf in leaves:
        size_key = (leaf.w, leaf.h)
        size_counts[size_key] = size_counts.get(size_key, 0) + 1
    print(f'[Quadtree Diffusion]: Tile size distribution: {len(size_counts)} unique sizes')
    sorted_sizes = sorted(size_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for (w, h), count in sorted_sizes:
        print(f'  {w}x{h}: {count} tiles')

    return bbox_list, weight, leaves

class CustomBBox(BBox):
    ''' region control bbox '''
    pass

class AbstractDiffusion:
    def __init__(self):
        self.method = self.__class__.__name__
        self.pbar = None

        # Performance: Cache for Gaussian weights to avoid recomputation
        self.gaussian_weight_cache = {}

        self.w: int = 0
        self.h: int = 0
        self.tile_width: int = None
        self.tile_height: int = None
        self.tile_overlap: int = None
        self.tile_batch_size: int = None

        # cache. final result of current sampling step, [B, C=4, H//8, W//8]
        # avoiding overhead of creating new tensors and weight summing
        self.x_buffer: Tensor = None
        # self.w: int = int(self.p.width  // opt_f)       # latent size
        # self.h: int = int(self.p.height // opt_f)
        # weights for background & grid bboxes
        self._weights: Tensor = None
        # self.weights: Tensor = torch.zeros((1, 1, self.h, self.w), device=devices.device, dtype=torch.float32)
        self._init_grid_bbox = None
        self._init_done = None

        # count the step correctly
        self.step_count = 0         
        self.inner_loop_count = 0  
        self.kdiff_step = -1

        # ext. Grid tiling painting (grid bbox)
        self.enable_grid_bbox: bool = False
        self.tile_w: int = None
        self.tile_h: int = None
        self.tile_bs: int = None
        self.num_tiles: int = None
        self.num_batches: int = None
        self.batched_bboxes: List[List[BBox]] = []

        # ext. Region Prompt Control (custom bbox)
        self.enable_custom_bbox: bool = False
        self.custom_bboxes: List[CustomBBox] = []
        # self.cond_basis: Cond = None
        # self.uncond_basis: Uncond = None
        # self.draw_background: bool = True       # by default we draw major prompts in grid tiles
        # self.causal_layers: bool = None

        # ext. ControlNet
        self.enable_controlnet: bool = False
        # self.controlnet_script: ModuleType = None
        self.control_tensor_batch_dict = {}
        self.control_tensor_batch: List[List[Tensor]] = [[]]
        # self.control_params: Dict[str, Tensor] = None # {}
        self.control_params: Dict[Tuple, List[List[Tensor]]] = {}
        self.control_tensor_cpu: bool = None
        self.control_tensor_custom: List[List[Tensor]] = []

        self.draw_background: bool = True       # by default we draw major prompts in grid tiles
        self.control_tensor_cpu = False
        self.weights = None
        self.imagescale = ImageScale()
        self.uniform_distribution = None
        self.sigmas = None

    def reset(self):
        # Preserve essential parameters
        tile_batch_size = self.tile_batch_size
        compression = self.compression
        tile_overlap = getattr(self, 'tile_overlap', 8)
        skip_diffusion_below = getattr(self, 'skip_diffusion_below', 0)

        # Preserve quadtree structure
        use_quadtree = getattr(self, 'use_quadtree', False)
        quadtree_from_visualizer = getattr(self, 'quadtree_from_visualizer', None)

        self.__init__()
        self.compression = compression
        self.tile_batch_size = tile_batch_size
        self.tile_overlap = tile_overlap
        self.skip_diffusion_below = skip_diffusion_below

        # Restore quadtree structure
        self.use_quadtree = use_quadtree
        self.quadtree_from_visualizer = quadtree_from_visualizer

    def repeat_tensor(self, x:Tensor, n:int, concat=False, concat_to=0) -> Tensor:
        ''' repeat the tensor on it's first dim '''
        if n == 1: return x
        B = x.shape[0]
        r_dims = len(x.shape) - 1
        if B == 1:      # batch_size = 1 (not `tile_batch_size`)
            shape = [n] + [-1] * r_dims     # [N, -1, ...]
            return x.expand(shape)          # `expand` is much lighter than `tile`
        else:
            if concat:
                return torch.cat([x for _ in range(n)], dim=0)[:concat_to]
            shape = [n] + [1] * r_dims      # [N, 1, ...]
            return x.repeat(shape)
    def update_pbar(self):
        if self.pbar.n >= self.pbar.total:
            self.pbar.close()
        else:
            # self.pbar.update()
            sampling_step = 20
            if self.step_count == sampling_step:
                self.inner_loop_count += 1
                if self.inner_loop_count < self.total_bboxes:
                    self.pbar.update()
            else:
                self.step_count = sampling_step
                self.inner_loop_count = 0
    def reset_buffer(self, x_in:Tensor):
        # Judge if the shape of x_in is the same as the shape of x_buffer
        if self.x_buffer is None or self.x_buffer.shape != x_in.shape:
            self.x_buffer = torch.zeros_like(x_in, device=x_in.device, dtype=x_in.dtype)
        else:
            self.x_buffer.zero_()

    @grid_bbox
    def init_grid_bbox(self, tile_w:int, tile_h:int, overlap:int, tile_bs:int):
        # if self._init_grid_bbox is not None: return
        # self._init_grid_bbox = True
        self.weights = torch.zeros((1, 1, self.h, self.w), device=devices.device, dtype=torch.float32)
        self.enable_grid_bbox = True

        self.tile_w = min(tile_w, self.w)
        self.tile_h = min(tile_h, self.h)
        overlap = max(0, min(overlap, min(tile_w, tile_h) - 4))
        # split the latent into overlapped tiles, then batching
        # weights basically indicate how many times a pixel is painted
        bboxes, weights = split_bboxes(self.w, self.h, self.tile_w, self.tile_h, overlap, self.get_tile_weights())
        self.weights += weights
        self.num_tiles = len(bboxes)
        self.num_batches = ceildiv(self.num_tiles , tile_bs)
        self.tile_bs = ceildiv(len(bboxes) , self.num_batches)          # optimal_batch_size
        self.batched_bboxes = [bboxes[i*self.tile_bs:(i+1)*self.tile_bs] for i in range(self.num_batches)]

    @grid_bbox
    def init_quadtree_bbox(self, latent_tensor: Tensor, tile_bs: int):
        """Initialize bboxes using quadtree subdivision with overlap"""
        self.weights = torch.zeros((1, 1, self.h, self.w), device=devices.device, dtype=torch.float32)
        self.enable_grid_bbox = True

        # Get overlap parameter (default to 8 pixels in latent space, which is 64 pixels in image space)
        overlap = getattr(self, 'tile_overlap', 8)
        overlap = max(0, min(overlap, min(self.h, self.w) // 4))  # Clamp overlap to reasonable range

        # Check if we have a quadtree from visualizer to reuse
        visualizer_quadtree = getattr(self, 'quadtree_from_visualizer', None)

        if visualizer_quadtree is not None:
            num_leaves = len(visualizer_quadtree["leaves"])
            print(f'[Quadtree Diffusion]: Processing {num_leaves} tiles with {overlap}px overlap (latent space)')

            # Warning for too many tiles
            if num_leaves > 100:
                print(f'[Quadtree Diffusion]: ⚠️  {num_leaves} tiles may be slow. Consider increasing min_tile_size to 512-768px.')

            # Warning for overlap >= smallest tile dimension
            if overlap > 0:
                # Find smallest tile dimension (not area) in latent space
                min_tile_dim = min(min(leaf.w, leaf.h) for leaf in visualizer_quadtree["leaves"]) // 8
                if overlap >= min_tile_dim:
                    print(f'[Quadtree Diffusion]: ⚠️  Overlap ({overlap}px latent) >= smallest tile dimension ({min_tile_dim}px latent = {min_tile_dim*8}px image). Reduce overlap or increase min_tile_size in visualizer.')

            # Reuse the quadtree structure from Visualizer
            # With rectangular quadtree, tiles are naturally within image bounds
            # No cropping, filtering, or capping needed!
            leaves = visualizer_quadtree['leaves']

            print(f'[Quadtree Diffusion]: Processing {len(leaves)} tiles (latent: {self.w}x{self.h})')

            bbox_list = []
            for idx, leaf in enumerate(leaves):
                # Scale from image space to latent space (divide by 8)
                core_x, core_y = leaf.x // 8, leaf.y // 8
                core_w, core_h = leaf.w // 8, leaf.h // 8

                # Cap overlap to 50% of tile dimension to prevent degenerate cases
                max_safe_overlap = min(core_w // 2, core_h // 2, overlap)

                # Add overlap symmetrically on all sides
                # Tiles are rectangular (naturally sized from quadtree)
                x = core_x - max_safe_overlap
                y = core_y - max_safe_overlap
                w = core_w + 2 * max_safe_overlap
                h = core_h + 2 * max_safe_overlap

                bbox = BBox(x, y, w, h)
                bbox.denoise = leaf.denoise
                bbox.variance = leaf.variance
                bbox.core_region = (core_x, core_y, core_w, core_h)  # Store the non-overlapping core
                bbox.pixel_w = leaf.w  # Store original pixel dimensions for skip logic
                bbox.pixel_h = leaf.h
                bbox_list.append(bbox)

                # For overlap mode, accumulate actual Gaussian weights
                # IMPORTANT: Only accumulate weights for pixels INSIDE the image
                if overlap > 0:
                    tile_weights = self.get_weight(w, h)

                    # Calculate the intersection of the tile with the image
                    x_start = max(0, x)
                    y_start = max(0, y)
                    x_end = min(self.w, x + w)
                    y_end = min(self.h, y + h)

                    # Offset into the tile weights tensor
                    tile_x_offset = x_start - x
                    tile_y_offset = y_start - y
                    tile_x_end_offset = tile_x_offset + (x_end - x_start)
                    tile_y_end_offset = tile_y_offset + (y_end - y_start)

                    # Only accumulate weights for the portion of the tile that's inside the image
                    if x_end > x_start and y_end > y_start:
                        self.weights[:, :, y_start:y_end, x_start:x_end] += \
                            tile_weights[tile_y_offset:tile_y_end_offset, tile_x_offset:tile_x_end_offset]
                else:
                    # For non-overlap mode, just count coverage
                    # Only count pixels inside the image
                    x_start = max(0, x)
                    y_start = max(0, y)
                    x_end = min(self.w, x + w)
                    y_end = min(self.h, y + h)
                    if x_end > x_start and y_end > y_start:
                        self.weights[:, :, y_start:y_end, x_start:x_end] += self.get_tile_weights()

            bboxes = bbox_list
            leaves = visualizer_quadtree['leaves']
        else:
            # No quadtree provided - this shouldn't happen in normal usage
            raise ValueError('[Quadtree Diffusion]: No quadtree structure provided! Connect QuadtreeVisualizer output to the quadtree input.')

        # Validate full coverage
        if self.weights.min() < 1e-6:
            uncovered = (self.weights < 1e-6).sum().item()
            raise RuntimeError(f"Quadtree has {uncovered} uncovered pixels! Bug in quadtree implementation.")

        self.quadtree_leaves = leaves  # Store for potential later use
        self.num_tiles = len(bboxes)

        # For quadtree, batch tiles by ACTUAL size (after overlap expansion)
        # Tiles with the same actual dimensions can be batched together efficiently

        # Group tiles by ACTUAL size (after overlap is applied)
        # This allows PyTorch to concatenate them without padding overhead
        actual_size_groups = defaultdict(list)
        for bbox in bboxes:
            # Use actual tile size for grouping
            size_key = (bbox.w, bbox.h)  # (width, height) after overlap
            actual_size_groups[size_key].append(bbox)

        # Create batches: batch same-sized tiles together up to tile_bs
        self.batched_bboxes = []
        for size_key, group_bboxes in actual_size_groups.items():
            # Split this size group into batches of tile_bs tiles each
            for i in range(0, len(group_bboxes), tile_bs):
                batch = group_bboxes[i:i+tile_bs]
                self.batched_bboxes.append(batch)

        self.num_batches = len(self.batched_bboxes)

        # Print statistics
        print(f'[Quadtree Diffusion]: {len(actual_size_groups)} unique tile sizes -> {self.num_batches} batches (batch_size={tile_bs})')

    # detached version of above
    @grid_bbox
    def get_grid_bbox(self, tile_w: int, tile_h: int, overlap: int, tile_bs: int, w: int, h: int, 
                    device: torch.device, get_tile_weights: Callable = lambda: 1.0) -> List[List[BBox]]:
        weights = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)
        # enable_grid_bbox = True

        tile_w = min(tile_w, w)
        tile_h = min(tile_h, h)
        overlap = max(0, min(overlap, min(tile_w, tile_h) - 4))
        # split the latent into overlapped tiles, then batching
        # weights basically indicate how many times a pixel is painted
        bboxes, weights_ = split_bboxes(w, h, tile_w, tile_h, overlap, get_tile_weights())
        weights += weights_
        num_tiles = len(bboxes)
        num_batches = ceildiv(num_tiles, tile_bs)
        tile_bs = ceildiv(len(bboxes), num_batches)          # optimal_batch_size
        batched_bboxes = [bboxes[i*tile_bs:(i+1)*tile_bs] for i in range(num_batches)]
        return batched_bboxes

    @grid_bbox
    def get_tile_weights(self) -> Union[Tensor, float]:
        return 1.0

    @noise_inverse
    def init_noise_inverse(self, steps:int, retouch:float, get_cache_callback, set_cache_callback, renoise_strength:float, renoise_kernel:int):
        self.noise_inverse_enabled = True
        self.noise_inverse_steps = steps
        self.noise_inverse_retouch = float(retouch)
        self.noise_inverse_renoise_strength = float(renoise_strength)
        self.noise_inverse_renoise_kernel = int(renoise_kernel)
        self.noise_inverse_set_cache = set_cache_callback
        self.noise_inverse_get_cache = get_cache_callback

    def init_done(self):
        '''
          Call this after all `init_*`, settings are done, now perform:
            - settings sanity check 
            - pre-computations, cache init
            - anything thing needed before denoising starts
        '''

        # if self._init_done is not None: return
        # self._init_done = True
        self.total_bboxes = 0
        if self.enable_grid_bbox:   self.total_bboxes += self.num_batches
        if self.enable_custom_bbox: self.total_bboxes += len(self.custom_bboxes)
        assert self.total_bboxes > 0, "Nothing to paint! No background to draw and no custom bboxes were provided."

        # sampling_steps = _steps
        # self.pbar = tqdm(total=(self.total_bboxes) * sampling_steps, desc=f"{self.method} Sampling: ")

    @controlnet
    def prepare_controlnet_tensors(self, refresh:bool=False, tensor=None):
        ''' Crop the control tensor into tiles and cache them '''
        if not refresh:
            if self.control_tensor_batch is not None or self.control_params is not None: return
        tensors = [tensor]
        self.org_control_tensor_batch = tensors
        self.control_tensor_batch = []
        for i in range(len(tensors)):
            control_tile_list = []
            control_tensor = tensors[i]
            for bboxes in self.batched_bboxes:
                single_batch_tensors = []
                for bbox in bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tensor.unsqueeze_(0)
                    control_tile = control_tensor[:, :, bbox[1]*opt_f:bbox[3]*opt_f, bbox[0]*opt_f:bbox[2]*opt_f]
                    single_batch_tensors.append(control_tile)
                control_tile = torch.cat(single_batch_tensors, dim=0)
                if self.control_tensor_cpu:
                    control_tile = control_tile.cpu()
                control_tile_list.append(control_tile)
            self.control_tensor_batch.append(control_tile_list)

            if len(self.custom_bboxes) > 0:
                custom_control_tile_list = []
                for bbox in self.custom_bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tensor.unsqueeze_(0)
                    control_tile = control_tensor[:, :, bbox[1]*opt_f:bbox[3]*opt_f, bbox[0]*opt_f:bbox[2]*opt_f]
                    if self.control_tensor_cpu:
                        control_tile = control_tile.cpu()
                    custom_control_tile_list.append(control_tile)
                self.control_tensor_custom.append(custom_control_tile_list)

    @controlnet
    def switch_controlnet_tensors(self, batch_id:int, x_batch_size:int, tile_batch_size:int, is_denoise=False):
        # if not self.enable_controlnet: return
        if self.control_tensor_batch is None: return
        # self.control_params = [0]

        # for param_id in range(len(self.control_params)):
        for param_id in range(len(self.control_tensor_batch)):
            # tensor that was concatenated in `prepare_controlnet_tensors`
            control_tile = self.control_tensor_batch[param_id][batch_id]
            # broadcast to latent batch size
            if x_batch_size > 1: # self.is_kdiff:
                all_control_tile = []
                for i in range(tile_batch_size):
                    this_control_tile = [control_tile[i].unsqueeze(0)] * x_batch_size
                    all_control_tile.append(torch.cat(this_control_tile, dim=0))
                control_tile = torch.cat(all_control_tile, dim=0) # [:x_tile.shape[0]]
                self.control_tensor_batch[param_id][batch_id] = control_tile
            # else:
            #     control_tile = control_tile.repeat([x_batch_size if is_denoise else x_batch_size * 2, 1, 1, 1])
            # self.control_params[param_id].hint_cond = control_tile.to(devices.device)

    def process_controlnet(self, x_noisy, c_in: dict, cond_or_uncond: List, bboxes, batch_size: int, batch_id: int, shifts=None, shift_condition=None):
        control: ControlNet = c_in['control']
        param_id = -1 # current controlnet & previous controlnets
        tuple_key = tuple(cond_or_uncond) + tuple(x_noisy.shape)
        while control is not None:
            param_id += 1

            if tuple_key not in self.control_params:
                self.control_params[tuple_key] = [[None]]

            while len(self.control_params[tuple_key]) <= param_id:
                self.control_params[tuple_key].append([None])

            while len(self.control_params[tuple_key][param_id]) <= batch_id:
                self.control_params[tuple_key][param_id].append(None)

            # Below is taken from comfy.controlnet.py, but we need to additionally tile the cnets.
            # if statement: eager eval. first time when cond_hint is None. 
            if self.refresh or control.cond_hint is None or not isinstance(self.control_params[tuple_key][param_id][batch_id], Tensor):
                if control.cond_hint is not None:
                    del control.cond_hint
                control.cond_hint = None
                compression_ratio = control.compression_ratio
                if getattr(control, 'vae', None) is not None:
                    compression_ratio *= control.vae.downscale_ratio
                else:
                    if getattr(control, 'latent_format', None) is not None:
                        raise ValueError("This Controlnet needs a VAE but none was provided, please use a ControlNetApply node with a VAE input and connect it.")
                PH, PW = self.h * compression_ratio, self.w * compression_ratio

                device = getattr(control, 'device', x_noisy.device)
                dtype = getattr(control, 'manual_cast_dtype', None)
                if dtype is None: dtype = getattr(getattr(control, 'control_model', None), 'dtype', None)
                if dtype is None: dtype = x_noisy.dtype

                if isinstance(control, T2IAdapter):
                    width, height = control.scale_image_to(PW, PH)
                    cns = common_upscale(control.cond_hint_original, width, height, control.upscale_algorithm, "center").float().to(device=device)
                    if control.channels_in == 1 and control.cond_hint.shape[1] > 1:
                        cns = torch.mean(control.cond_hint, 1, keepdim=True)
                elif control.__class__.__name__ == 'ControlLLLiteAdvanced':
                    if getattr(control, 'sub_idxs', None) is not None and control.cond_hint_original.shape[0] >= control.full_latent_length:
                        cns = common_upscale(control.cond_hint_original[control.sub_idxs], PW, PH, control.upscale_algorithm, "center").to(dtype=dtype, device=device)
                    else:
                        cns = common_upscale(control.cond_hint_original, PW, PH, control.upscale_algorithm, "center").to(dtype=dtype, device=device)
                else:
                    cns = common_upscale(control.cond_hint_original, PW, PH, control.upscale_algorithm, 'center').to(dtype=dtype, device=device)
                    cns = control.preprocess_image(cns)
                    if getattr(control, 'vae', None) is not None:
                        loaded_models_ = loaded_models(only_currently_used=True)
                        cns = control.vae.encode(cns.movedim(1, -1))
                        load_models_gpu(loaded_models_)
                    if getattr(control, 'latent_format', None) is not None:
                        cns = control.latent_format.process_in(cns)
                    if len(getattr(control, 'extra_concat_orig', ())) > 0:
                        to_concat = []
                        for c in control.extra_concat_orig:
                            c = c.to(device=device)
                            c = common_upscale(c, cns.shape[3], cns.shape[2], control.upscale_algorithm, "center")
                            to_concat.append(repeat_to_batch_size(c, cns.shape[0]))
                        cns = torch.cat([cns] + to_concat, dim=1)

                    cns = cns.to(device=device, dtype=dtype)

                # Tile the ControlNets
                #
                # Below can be in this if clause because self.refresh will trigger on resolution change,
                # e.g. cause of ConditioningSetArea, so that particular case isn't cached atm.
                cf = control.compression_ratio
                if cns.shape[0] != batch_size:
                    cns = repeat_to_batch_size(cns, batch_size)
                if shifts is not None:
                    control.cns = cns
                    # cns = cns.roll(shifts=tuple(x * cf for x in shifts), dims=(-2,-1))
                    sh_h,sh_w=shifts
                    sh_h *= cf
                    sh_w *= cf
                    if (sh_h,sh_w) != (0,0):
                        if sh_h == 0 or sh_w == 0:
                            cns = control.cns.roll(shifts=(sh_h,sh_w), dims=(-2,-1))
                        else:
                            if shift_condition:
                                cns = control.cns.roll(shifts=sh_h, dims=-2)
                            else:
                                cns = control.cns.roll(shifts=sh_w, dims=-1)
                cns_slices = [cns[:, :, bbox[1]*cf:bbox[3]*cf, bbox[0]*cf:bbox[2]*cf] for bbox in bboxes]
                control.cond_hint = torch.cat(cns_slices, dim=0).to(device=cns.device)
                del cns_slices
                del cns
                self.control_params[tuple_key][param_id][batch_id] = control.cond_hint
            else:
                if hasattr(control,'cns') and shifts is not None:
                    cf = control.compression_ratio
                    # cns = control.cns.roll(shifts=tuple(x * cf for x in shifts), dims=(-2,-1))
                    cns = control.cns
                    sh_h,sh_w=shifts
                    sh_h *= cf
                    sh_w *= cf
                    if (sh_h,sh_w) != (0,0):
                        if sh_h == 0 or sh_w == 0:
                            cns = control.cns.roll(shifts=(sh_h,sh_w), dims=(-2,-1))
                        else:
                            if shift_condition:
                                cns = control.cns.roll(shifts=sh_h, dims=-2)
                            else:
                                cns = control.cns.roll(shifts=sh_w, dims=-1)
                    cns_slices = [cns[:, :, bbox[1]*cf:bbox[3]*cf, bbox[0]*cf:bbox[2]*cf] for bbox in bboxes]
                    control.cond_hint = torch.cat(cns_slices, dim=0).to(device=cns.device)
                    del cns_slices
                    del cns
                else:
                    control.cond_hint = self.control_params[tuple_key][param_id][batch_id]
            control = control.previous_controlnet

import numpy as np
from numpy import pi, exp, sqrt
def gaussian_weights(tile_w:int, tile_h:int) -> Tensor:
    '''
    Copy from the original implementation of Mixture of Diffusers
    https://github.com/albarji/mixture-of-diffusers/blob/master/mixdiff/tiling.py
    This generates gaussian weights to smooth the noise of each tile.
    This is critical for this method to work.

    CRITICAL FIX 1: Increased var from 0.01 to 0.02 to ensure edge weights > 1e-6
    for tiles with overlap. With var=0.01, tiles ≥68×68 had edge weights < 1e-6,
    causing pixels to be marked as "uncovered" even though tiles were processing them.

    CRITICAL FIX 2: Use correct tile dimension (tile_w for x, tile_h for y) in Gaussian formula.
    Previously used tile_w for both axes, which was incorrect for non-square tiles.
    While quadtree always creates square tiles, this fix ensures correctness.
    '''
    # Define separate Gaussian functions for x and y axes with correct scaling
    var = 0.02
    f_x = lambda x, midpoint: exp(-(x-midpoint)*(x-midpoint) / (tile_w*tile_w) / (2*var)) / sqrt(2*pi*var)
    f_y = lambda y, midpoint: exp(-(y-midpoint)*(y-midpoint) / (tile_h*tile_h) / (2*var)) / sqrt(2*pi*var)

    x_probs = [f_x(x, (tile_w - 1) / 2) for x in range(tile_w)]   # -1 because index goes from 0 to latent_width - 1
    y_probs = [f_y(y, (tile_h - 1) / 2) for y in range(tile_h)]   # FIXED: was tile_h / 2, now consistent with x_probs

    w = np.outer(y_probs, x_probs)
    return torch.from_numpy(w).to(devices.device, dtype=torch.float32)

class CondDict: ...

class MultiDiffusion(AbstractDiffusion):

    @torch.inference_mode()
    def __call__(self, model_function: BaseModel.apply_model, args: dict):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]
        cond_or_uncond: List = args["cond_or_uncond"]

        N, C, H, W = x_in.shape

        # Performance: Cache use_quadtree flag once instead of multiple getattr calls
        use_qt = getattr(self, 'use_quadtree', False)

        # comfyui can feed in a latent that's a different size cause of SetArea, so we'll refresh in that case.
        self.refresh = False
        if self.weights is None or self.h != H or self.w != W:
            self.h, self.w = H, W
            self.refresh = True

            if use_qt:
                self.init_quadtree_bbox(x_in, self.tile_batch_size)
            else:
                self.init_grid_bbox(self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)

            # init everything done, perform sanity check & pre-computations
            self.init_done()
        self.h, self.w = H, W
        # clear buffer canvas
        self.reset_buffer(x_in)

        # Background sampling (grid bbox)
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if processing_interrupted():
                    # self.pbar.close()
                    return x_in

                # batching & compute tiles
                x_tile = torch.cat([x_in[bbox.slicer] for bbox in bboxes], dim=0)   # [TB, C, TH, TW]
                t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])
                c_tile = {}
                for k, v in c_in.items():
                    if isinstance(v, torch.Tensor):
                        if len(v.shape) == len(x_tile.shape):
                            bboxes_ = bboxes
                            if v.shape[-2:] != x_in.shape[-2:]:
                                cf = x_in.shape[-1] * self.compression // v.shape[-1] # compression factor
                                bboxes_ = self.get_grid_bbox(
                                    self.width // cf,
                                    self.height // cf,
                                    self.overlap // cf,
                                    self.tile_batch_size,
                                    v.shape[-1],
                                    v.shape[-2],
                                    x_in.device,
                                    self.get_tile_weights,
                                )
                            v = torch.cat([v[bbox_.slicer] for bbox_ in bboxes_[batch_id]])
                        if v.shape[0] != x_tile.shape[0]:
                            v = repeat_to_batch_size(v, x_tile.shape[0])
                    c_tile[k] = v

                # controlnet tiling
                # self.switch_controlnet_tensors(batch_id, N, len(bboxes))
                if 'control' in c_in:
                    self.process_controlnet(x_tile, c_in, cond_or_uncond, bboxes, N, batch_id)
                    c_tile['control'] = c_in['control'].get_control_orig(x_tile, t_tile, c_tile, len(cond_or_uncond), c_in['transformer_options'])

                # stablesr tiling
                # self.switch_stablesr_tensors(batch_id)

                x_tile_out = model_function(x_tile, t_tile, **c_tile)

                for i, bbox in enumerate(bboxes):
                    # Both quadtree and grid tiles use accumulation with overlap
                    self.x_buffer[bbox.slicer] += x_tile_out[i*N:(i+1)*N, :, :, :]
                del x_tile_out, x_tile, t_tile, c_tile

                # update progress bar
                # self.update_pbar()

        # Averaging background buffer
        # Both grid and quadtree modes now use overlap, so average by weights
        x_out = torch.where(self.weights > 1, self.x_buffer / self.weights, self.x_buffer)

        return x_out

from .utils import store

def fibonacci_spacing(x):
    result = torch.zeros_like(x)
    fib = [0, 1]
    while fib[-1] < len(x):
        fib.append(fib[-1] + fib[-2])
    
    used_indices = set()
    for i, val in enumerate(x):
        fib_index = i % len(fib)
        target_index = fib[fib_index] % len(x)
        while target_index in used_indices:
            target_index = (target_index + 1) % len(x)
        result[target_index] = val
        used_indices.add(target_index)
    
    return result

def find_nearest(a,b):
    # Calculate the absolute differences. 
    diff = (a - b).abs()

    # Find the indices of the nearest elements
    nearest_indices = diff.argmin()

    # Get the nearest elements from b
    return b[nearest_indices]

class SpotDiffusion(AbstractDiffusion):
    
    @torch.inference_mode()
    def __call__(self, model_function: BaseModel.apply_model, args: dict):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]
        cond_or_uncond: List = args["cond_or_uncond"]

        N, C, H, W = x_in.shape

        # Performance: Cache use_quadtree flag once instead of multiple getattr calls
        use_qt = getattr(self, 'use_quadtree', False)

        # comfyui can feed in a latent that's a different size cause of SetArea, so we'll refresh in that case.
        self.refresh = False
        if self.weights is None or self.h != H or self.w != W:
            self.h, self.w = H, W
            self.refresh = True

            if use_qt:
                self.init_quadtree_bbox(x_in, self.tile_batch_size)
            else:
                self.init_grid_bbox(self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)

            # init everything done, perform sanity check & pre-computations
            self.init_done()
        self.h, self.w = H, W
        # clear buffer canvas
        self.reset_buffer(x_in)

        if self.uniform_distribution is None:
            sigmas = self.sigmas = store.sigmas
            shift_method = store.model_options.get('tiled_diffusion_shift_method', 'random')
            seed = store.model_options.get('tiled_diffusion_seed', store.extra_args.get('seed', 0))
            th = self.tile_height
            tw = self.tile_width
            cf = self.compression
            if 'effnet' in c_in:
                cf = x_in.shape[-1] * self.compression // c_in['effnet'].shape[-1] # compression factor
                th = self.height // cf
                tw = self.width // cf
            shift_height = torch.randint(0, th, (len(sigmas)-1,), generator=torch.Generator(device='cpu').manual_seed(seed), device='cpu')
            shift_height = (shift_height * cf / self.compression).round().to(torch.int32)
            shift_width = torch.randint(0, tw, (len(sigmas)-1,), generator=torch.Generator(device='cpu').manual_seed(seed), device='cpu')
            shift_width = (shift_width * cf / self.compression).round().to(torch.int32)
            if shift_method == "sorted":
                shift_height = shift_height.sort().values
                shift_width = shift_width.sort().values
            elif shift_method == "fibonacci":
                shift_height = fibonacci_spacing(shift_height.sort().values)
                shift_width = fibonacci_spacing(shift_width.sort().values)
            self.uniform_distribution = (shift_height, shift_width)

        sigmas = self.sigmas
        ts_in = find_nearest(t_in[0], sigmas)
        cur_i = ss.item() if (ss:=(sigmas == ts_in).nonzero()).shape[0] != 0 else 0

        sh_h = self.uniform_distribution[0][cur_i].item()
        sh_w = self.uniform_distribution[1][cur_i].item()
        if min(self.tile_height, x_in.shape[-2]) == x_in.shape[-2]:
            sh_h=0
        if min(self.tile_width, x_in.shape[-1]) == x_in.shape[-1]:
            sh_w=0
        condition = cur_i % 2 == 0 if self.tile_height > self.tile_width else cur_i % 2 != 0
        if (sh_h,sh_w) != (0,0):
            # x_in = x_in.roll(shifts=(sh_h,sh_w), dims=(-2,-1))
            if sh_h == 0 or sh_w == 0:
                x_in = x_in.roll(shifts=(sh_h,sh_w), dims=(-2,-1))
            else:
                if condition:
                    x_in = x_in.roll(shifts=sh_h, dims=-2)
                else:
                    x_in = x_in.roll(shifts=sh_w, dims=-1)

        # Background sampling (grid bbox)
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if processing_interrupted(): 
                    # self.pbar.close()
                    return x_in

                # batching & compute tiles
                x_tile = torch.cat([x_in[bbox.slicer] for bbox in bboxes], dim=0)   # [TB, C, TH, TW]
                t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])
                c_tile = {}
                for k, v in c_in.items():
                    if isinstance(v, torch.Tensor):
                        if len(v.shape) == len(x_tile.shape):
                            bboxes_ = bboxes
                            sh_h_new, sh_w_new = sh_h, sh_w
                            if v.shape[-2:] != x_in.shape[-2:]:
                                cf = x_in.shape[-1] * self.compression // v.shape[-1] # compression factor
                                bboxes_ = self.get_grid_bbox(
                                    self.width // cf,
                                    self.height // cf,
                                    self.overlap // cf,
                                    self.tile_batch_size,
                                    v.shape[-1],
                                    v.shape[-2],
                                    x_in.device,
                                    self.get_tile_weights,
                                )
                                sh_h_new, sh_w_new = round(sh_h * self.compression / cf), round(sh_w * self.compression / cf)
                            v = v.roll(shifts=(sh_h_new, sh_w_new), dims=(-2,-1))
                            v = torch.cat([v[bbox_.slicer] for bbox_ in bboxes_[batch_id]])
                        if v.shape[0] != x_tile.shape[0]:
                            v = repeat_to_batch_size(v, x_tile.shape[0])
                    c_tile[k] = v

                # controlnet tiling
                # self.switch_controlnet_tensors(batch_id, N, len(bboxes))
                if 'control' in c_in:
                    self.process_controlnet(x_tile, c_in, cond_or_uncond, bboxes, N, batch_id, (sh_h,sh_w), condition)
                    c_tile['control'] = c_in['control'].get_control_orig(x_tile, t_tile, c_tile, len(cond_or_uncond), c_in['transformer_options'])

                # stablesr tiling
                # self.switch_stablesr_tensors(batch_id)

                x_tile_out = model_function(x_tile, t_tile, **c_tile)

                for i, bbox in enumerate(bboxes):
                    self.x_buffer[bbox.slicer] = x_tile_out[i*N:(i+1)*N, :, :, :]

                del x_tile_out, x_tile, t_tile, c_tile

                # update progress bar
                # self.update_pbar()

        if (sh_h,sh_w) != (0,0):
            # self.x_buffer = self.x_buffer.roll(shifts=(-sh_h, -sh_w), dims=(-2, -1))
            if sh_h == 0 or sh_w == 0:
                self.x_buffer = self.x_buffer.roll(shifts=(-sh_h, -sh_w), dims=(-2, -1))
            else:
                if condition:
                    self.x_buffer = self.x_buffer.roll(shifts=-sh_h, dims=-2)
                else:
                    self.x_buffer = self.x_buffer.roll(shifts=-sh_w, dims=-1)

        return self.x_buffer

class MixtureOfDiffusers(AbstractDiffusion):
    """
        Mixture-of-Diffusers Implementation
        https://github.com/albarji/mixture-of-diffusers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # weights for custom bboxes
        self.custom_weights: List[Tensor] = []

    def get_weight(self, tile_w: int, tile_h: int) -> Tensor:
        """Generate Gaussian weight matrix for a tile with caching"""
        cache_key = (tile_w, tile_h)
        if cache_key in self.gaussian_weight_cache:
            return self.gaussian_weight_cache[cache_key]

        # Compute Gaussian weights using the original function
        w = gaussian_weights(tile_w, tile_h)

        # Cache before returning
        self.gaussian_weight_cache[cache_key] = w
        return w

    def init_done(self):
        super().init_done()
        # The original gaussian weights can be extremely small, so we rescale them for numerical stability
        self.rescale_factor = 1 / self.weights
        # Meanwhile, we rescale the custom weights in advance to save time of slicing
        for bbox_id, bbox in enumerate(self.custom_bboxes):
            if bbox.blend_mode == BlendMode.BACKGROUND:
                self.custom_weights[bbox_id] *= self.rescale_factor[bbox.slicer]

    @grid_bbox
    def get_tile_weights(self) -> Union[Tensor, float]:
        # For quadtree mode, we still use Gaussian weights but generate them per-tile
        # We return 1.0 here because we'll compute per-tile weights during processing
        if getattr(self, 'use_quadtree', False):
            return 1.0

        # weights for grid bboxes
        # if not hasattr(self, 'tile_weights'):
        # x_in can change sizes cause of ConditioningSetArea, so we have to recalcualte each time
        self.tile_weights = self.get_weight(self.tile_w, self.tile_h)
        return self.tile_weights

    @torch.inference_mode()
    def __call__(self, model_function: BaseModel.apply_model, args: dict):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]
        cond_or_uncond: List= args["cond_or_uncond"]

        N, C, H, W = x_in.shape

        # Check if quadtree mode is enabled (needed throughout the function)
        use_qt = getattr(self, 'use_quadtree', False)

        # Store sigmas for variable denoise (from store if available)
        if not hasattr(self, 'sigmas') or self.sigmas is None:
            try:
                from .utils import store
                if hasattr(store, 'sigmas'):
                    self.sigmas = store.sigmas
                    print(f'[Quadtree Variable Denoise]: Loaded sigmas from store, length={len(self.sigmas)}')
                else:
                    print(f'[Quadtree Variable Denoise]: WARNING - No sigmas in store, variable denoise will NOT work')
            except Exception as e:
                print(f'[Quadtree Variable Denoise]: WARNING - Failed to load sigmas: {e}')
                pass

        self.refresh = False
        # self.refresh = True
        if self.weights is None or self.h != H or self.w != W:
            self.h, self.w = H, W
            self.refresh = True

            if use_qt:
                self.init_quadtree_bbox(x_in, self.tile_batch_size)
                # Print denoise range for quadtree tiles
                if hasattr(self, 'batched_bboxes') and len(self.batched_bboxes) > 0:
                    all_denoise = [bbox.denoise for batch in self.batched_bboxes for bbox in batch if hasattr(bbox, 'denoise')]
                    if all_denoise:
                        min_d, max_d = min(all_denoise), max(all_denoise)
                        print(f'[Quadtree Variable Denoise]: Denoise range: {min_d:.3f} to {max_d:.3f}')
                        if min_d < 1.0:
                            print(f'[Quadtree Variable Denoise]: ENABLED - Tiles will be denoised adaptively')
                        else:
                            print(f'[Quadtree Variable Denoise]: All tiles at max denoise - no variation')
            else:
                self.init_grid_bbox(self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)

            # init everything done, perform sanity check & pre-computations
            self.init_done()
        self.h, self.w = H, W
        # clear buffer canvas
        self.reset_buffer(x_in)

        # self.pbar = tqdm(total=(self.total_bboxes) * sampling_steps, desc=f"{self.method} Sampling: ")
        # self.pbar = tqdm(total=len(self.batched_bboxes), desc=f"{self.method} Sampling: ")

        # PRE-CALCULATE SKIP STATISTICS: Count how many tiles will be skipped
        skip_threshold = getattr(self, 'skip_diffusion_below', 0)
        if use_qt and skip_threshold > 0:
            total_tiles = 0
            total_skip = 0
            skip_tile_sizes = []

            for bboxes in self.batched_bboxes:
                for bbox in bboxes:
                    total_tiles += 1
                    pixel_w = getattr(bbox, 'pixel_w', 0)
                    pixel_h = getattr(bbox, 'pixel_h', 0)
                    min_dimension = min(pixel_w, pixel_h)

                    if min_dimension < skip_threshold:
                        total_skip += 1
                        if len(skip_tile_sizes) < 10:  # Sample first 10 for debugging
                            skip_tile_sizes.append(f"{pixel_w}x{pixel_h}")

            total_process = total_tiles - total_skip
            print(f'[Quadtree Skip]: Will skip {total_skip}/{total_tiles} tiles (min dimension < {skip_threshold}px)')
            print(f'[Quadtree Skip]: Processing {total_process} tiles through model, copying {total_skip} tiles directly')
            if skip_tile_sizes:
                print(f'[Quadtree Skip]: Sample skip sizes: {", ".join(skip_tile_sizes[:5])}{"..." if len(skip_tile_sizes) > 5 else ""}')

        # Global sampling
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):     # batch_id is the `Latent tile batch size`
                if processing_interrupted():
                    # self.pbar.close()
                    return x_in

                # SKIP LOGIC: Separate tiles into skip and process lists
                process_bboxes = []
                skip_bboxes = []

                if use_qt and skip_threshold > 0:
                    for bbox in bboxes:
                        pixel_w = getattr(bbox, 'pixel_w', 0)
                        pixel_h = getattr(bbox, 'pixel_h', 0)
                        min_dimension = min(pixel_w, pixel_h)

                        if min_dimension < skip_threshold:
                            skip_bboxes.append(bbox)
                        else:
                            process_bboxes.append(bbox)
                else:
                    # No skipping, process all tiles
                    process_bboxes = bboxes

                # Process skip tiles: copy input directly to output buffer
                for bbox in skip_bboxes:
                    # Extract the input tile for this region
                    tile_input = extract_tile_with_padding(x_in, bbox, self.w, self.h)

                    # Crop to bbox size
                    if tile_input.shape[-2] > bbox.h or tile_input.shape[-1] > bbox.w:
                        tile_input = tile_input[:, :, :bbox.h, :bbox.w]

                    # Copy directly to output buffer (no diffusion)
                    # Handle overlap blending if enabled
                    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
                    x_start = max(0, x)
                    y_start = max(0, y)
                    x_end = min(self.w, x + w)
                    y_end = min(self.h, y + h)

                    tile_x_offset = x_start - x
                    tile_y_offset = y_start - y

                    valid_tile = tile_input[:, :,
                                          tile_y_offset:tile_y_offset + (y_end - y_start),
                                          tile_x_offset:tile_x_offset + (x_end - x_start)]

                    if self.tile_overlap > 0:
                        tile_weights_full = self.get_weight(bbox.w, bbox.h)
                        tile_weights = tile_weights_full[tile_y_offset:tile_y_offset + (y_end - y_start),
                                                        tile_x_offset:tile_x_offset + (x_end - x_start)]
                        tile_weights = tile_weights.unsqueeze(0).unsqueeze(0)
                        self.x_buffer[:, :, y_start:y_end, x_start:x_end] += valid_tile * tile_weights
                    else:
                        self.x_buffer[:, :, y_start:y_end, x_start:x_end] += valid_tile

                # Skip model inference if no tiles to process
                if len(process_bboxes) == 0:
                    continue

                # batching (only for tiles that need processing)
                bboxes = process_bboxes  # Replace bboxes with filtered list
                x_tile_list     = []

                # For quadtree with overlap, extract tiles with padding if needed
                if use_qt:
                    # Find maximum size in this batch (should all be same if square tiles work correctly)
                    max_h = max(bbox.h for bbox in bboxes)
                    max_w = max(bbox.w for bbox in bboxes)

                    for bbox in bboxes:
                        # Use helper function to extract with reflection padding
                        tile = extract_tile_with_padding(x_in, bbox, self.w, self.h)

                        # Additional padding if tiles in batch have different sizes (shouldn't happen with square tiles)
                        if tile.shape[-2] < max_h or tile.shape[-1] < max_w:
                            pad_h = max_h - tile.shape[-2]
                            pad_w = max_w - tile.shape[-1]
                            tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h), mode='replicate')
                        x_tile_list.append(tile)
                else:
                    for bbox in bboxes:
                        x_tile_list.append(x_in[bbox.slicer])

                x_tile = torch.cat(x_tile_list, dim=0)                     # differs each
                t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])   # just repeat
                c_tile = {}
                for k, v in c_in.items():
                    if isinstance(v, torch.Tensor):
                        if len(v.shape) == len(x_tile.shape):
                            bboxes_ = bboxes
                            if v.shape[-2:] != x_in.shape[-2:]:
                                # Conditioning tensor has different resolution - need to scale bboxes
                                cf = x_in.shape[-1] * self.compression // v.shape[-1] # compression factor
                                # Scale the quadtree bboxes to match conditioning resolution
                                scaled_bboxes = []
                                for batch in self.batched_bboxes:
                                    scaled_batch = []
                                    for bbox in batch:
                                        scaled_bbox = BBox(bbox.x // cf, bbox.y // cf, bbox.w // cf, bbox.h // cf)
                                        scaled_batch.append(scaled_bbox)
                                    scaled_bboxes.append(scaled_batch)
                                bboxes_ = scaled_bboxes[batch_id]
                            v = torch.cat([v[bbox_.slicer] for bbox_ in bboxes_])
                        if v.shape[0] != x_tile.shape[0]:
                            v = repeat_to_batch_size(v, x_tile.shape[0])
                    c_tile[k] = v
                
                # controlnet
                # self.switch_controlnet_tensors(batch_id, N, len(bboxes), is_denoise=True)
                if 'control' in c_in:
                    self.process_controlnet(x_tile, c_in, cond_or_uncond, bboxes, N, batch_id)
                    c_tile['control'] = c_in['control'].get_control_orig(x_tile, t_tile, c_tile, len(cond_or_uncond), c_in['transformer_options'])
                
                # stablesr
                # self.switch_stablesr_tensors(batch_id)

                # denoising: here the x is the noise
                x_tile_out = model_function(x_tile, t_tile, **c_tile)

                # de-batching
                for i, bbox in enumerate(bboxes):
                    # Get the output tile
                    tile_out = x_tile_out[i*N:(i+1)*N, :, :, :]

                    # Crop if we padded this tile earlier
                    if tile_out.shape[-2] > bbox.h or tile_out.shape[-1] > bbox.w:
                        tile_out = tile_out[:, :, :bbox.h, :bbox.w]

                    # VARIABLE DENOISE: Check if this tile should be active at this timestep
                    # Only applies to quadtree mode with denoise values
                    tile_denoise = getattr(bbox, 'denoise', 1.0) if use_qt else 1.0

                    if use_qt and hasattr(self, 'sigmas') and self.sigmas is not None and tile_denoise < 1.0:
                        # Calculate progress through denoising schedule (0 = start, 1 = end)
                        sigmas = self.sigmas
                        ts_in = find_nearest(t_in[0], sigmas)
                        cur_idx = (sigmas == ts_in).nonzero()
                        if cur_idx.shape[0] > 0:
                            current_step = cur_idx.item()
                            total_steps = len(sigmas) - 1

                            # Progress from 0 (high noise) to 1 (low noise)
                            progress = current_step / max(total_steps, 1)

                            # SIMPLIFIED APPROACH: Smooth scaling that works for both txt2img and img2img
                            # Instead of on/off activation, use continuous scaling based on tile denoise
                            #
                            # tile_denoise=0.2 (large tiles, preserve more):
                            #   - Start at scale=0.75, smoothly ramp to 1.0
                            #   - Less aggressive denoising = preserve more
                            #
                            # tile_denoise=0.8 (small tiles, change more):
                            #   - Start at scale=0.90, quickly ramp to 1.0
                            #   - More aggressive denoising = change more

                            # Map tile_denoise to starting scale factor
                            # FIX: Increased minimum scale from 0.4 to 0.70 to preserve color enhancement
                            # Old formula was too aggressive, causing color washing in large tiles
                            # Low denoise (0.2) -> start at 0.75 (75% strength) [was 0.5]
                            # High denoise (0.8) -> start at 0.90 (90% strength) [was 0.9]
                            start_scale = 0.70 + (tile_denoise * 0.25)  # Range: 0.70-0.95

                            # Ramp up to full strength over the schedule
                            # Low denoise tiles ramp slower (gentler curve)
                            # High denoise tiles ramp faster (steeper curve)
                            ramp_curve = 1.0 + tile_denoise  # Range: 1.2-1.8
                            progress_curved = min(1.0, pow(progress, 1.0 / ramp_curve))

                            # Final scale factor: start_scale + remaining distance * curved progress
                            scale_factor = start_scale + (1.0 - start_scale) * progress_curved
                            scale_factor = max(0.70, min(1.0, scale_factor))  # Clamp to [0.70, 1.0]

                            # Log smooth scaling info (first tile only, once per session)
                            if i == 0 and batch_id == 0 and not hasattr(self, '_logged_var_denoise'):
                                print(f'[Quadtree Variable Denoise]: SMOOTH SCALING - tile_denoise={tile_denoise:.3f}, progress={progress:.3f}, start_scale={start_scale:.3f}, scale={scale_factor:.3f}')
                                self._logged_var_denoise = True

                            # Scale the noise prediction
                            tile_out = tile_out * scale_factor

                    if use_qt:
                        # In quadtree mode with square tiles
                        # Calculate intersection with image boundaries
                        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
                        x_start = max(0, x)
                        y_start = max(0, y)
                        x_end = min(self.w, x + w)
                        y_end = min(self.h, y + h)

                        # Calculate offset into tile (accounts for padding we added earlier)
                        tile_x_offset = x_start - x
                        tile_y_offset = y_start - y

                        # Extract the valid portion of the tile (without padding)
                        valid_tile = tile_out[:, :,
                                              tile_y_offset:tile_y_offset + (y_end - y_start),
                                              tile_x_offset:tile_x_offset + (x_end - x_start)]

                        if self.tile_overlap > 0:
                            # With overlap enabled, use weighted blending
                            # Generate Gaussian weights for the FULL tile size (including padding)
                            tile_weights_full = self.get_weight(bbox.w, bbox.h)
                            # Extract only the weights for the valid (non-padded) portion
                            tile_weights = tile_weights_full[tile_y_offset:tile_y_offset + (y_end - y_start),
                                                            tile_x_offset:tile_x_offset + (x_end - x_start)]
                            # Reshape to [1, 1, H, W] for proper broadcasting with 4D tensors
                            tile_weights = tile_weights.unsqueeze(0).unsqueeze(0)
                            # Add weighted tile to buffer - normalization will happen after all tiles are processed
                            self.x_buffer[:, :, y_start:y_end, x_start:x_end] += valid_tile * tile_weights
                        else:
                            # Without overlap, use simple direct assignment
                            self.x_buffer[:, :, y_start:y_end, x_start:x_end] = valid_tile
                    else:
                        # In grid mode, tiles overlap - use weighted blending
                        # These weights can be calcluated in advance, but will cost a lot of vram
                        # when you have many tiles. So we calculate it here.
                        w = self.tile_weights * self.rescale_factor[bbox.slicer]
                        self.x_buffer[bbox.slicer] += tile_out * w
                del x_tile_out, x_tile, t_tile, c_tile

                # self.update_pbar()
                # self.pbar.update()
        # self.pbar.close()
        x_out = self.x_buffer

        # For quadtree with overlap, normalize by accumulated weights
        if use_qt and self.tile_overlap > 0:
            # CRITICAL: Handle division by zero for uncovered pixels
            # Only normalize pixels that were actually covered by tiles (weight > epsilon)
            epsilon = 1e-6
            mask = self.weights > epsilon
            x_out = torch.where(mask, x_out / torch.clamp(self.weights, min=epsilon), x_out)

            # Check for uncovered pixels and warn user
            uncovered_pixels = (~mask).sum().item()
            if uncovered_pixels > 0:
                total_pixels = mask.numel()
                pct = 100.0 * uncovered_pixels / total_pixels
                print(f'[Quadtree Diffusion]: ⚠️  {uncovered_pixels}/{total_pixels} pixels ({pct:.1f}%) not covered by any tile!')
                print(f'[Quadtree Diffusion]: This can cause artifacts. Try: increase overlap, decrease content_threshold, or increase max_depth')

        return x_out

MAX_RESOLUTION=8192
class TiledDiffusion():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL", ),
                                "method": (["MultiDiffusion", "Mixture of Diffusers", "SpotDiffusion"], {"default": "Mixture of Diffusers"}),
                                "quadtree": ("QUADTREE",),
                                "enable_overlap": ("BOOLEAN", {"default": False, "tooltip": "Enable tile overlap for smoother blending. Currently experimental."}),
                                "tile_overlap": ("INT", {"default": 16*opt_f, "min": opt_f, "max": 256*opt_f, "step": opt_f, "tooltip": "Overlap between tiles in pixels (image space). Minimum 8px (1px latent). Gets divided by 8 for latent space."}),
                                "tile_batch_size": ("INT", {"default": 4, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                                "skip_diffusion_below": ("INT", {"default": 0, "min": 0, "max": 512, "step": 8, "tooltip": "Skip diffusion for tiles smaller than this size in pixels (0 = disabled). Preserves original details for small tiles. Must be multiple of 8 for VAE compatibility."}),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "_for_testing"
    instances = WeakSet()

    @classmethod
    def IS_CHANGED(s, *args, **kwargs):
        for o in s.instances:
            if hasattr(o, 'impl') and o.impl is not None:
                o.impl.reset()
        return ""
    
    def __init__(self) -> None:
        self.__class__.instances.add(self)

    def apply(self, model: ModelPatcher, method, quadtree, enable_overlap, tile_overlap, tile_batch_size, skip_diffusion_below):

        skip_enabled = skip_diffusion_below > 0
        print(f'[Quadtree Diffusion]: Overlap: {"enabled" if enable_overlap else "disabled"}{f" ({tile_overlap}px)" if enable_overlap else ""}, Skip: {"enabled" if skip_enabled else "disabled"}{f" (<{skip_diffusion_below}px)" if skip_enabled else ""}')

        if method == "Mixture of Diffusers":
            self.impl = MixtureOfDiffusers()
        elif method == "MultiDiffusion":
            self.impl = MultiDiffusion()
        else:
            self.impl = SpotDiffusion()

        # if noise_inversion:
        #     get_cache_callback = self.noise_inverse_get_cache
        #     set_cache_callback = None # lambda x0, xt, prompts: self.noise_inverse_set_cache(p, x0, xt, prompts, steps, retouch)
        #     self.impl.init_noise_inverse(steps, retouch, get_cache_callback, set_cache_callback, renoise_strength, renoise_kernel_size)

        # Set compression for latent space calculations
        compression = 4 if "CASCADE" in str(model.model.model_type) else 8
        self.impl.compression = compression

        # Store quadtree structure, overlap, batch size, and skip threshold
        self.impl.use_quadtree = True
        self.impl.quadtree_from_visualizer = quadtree
        self.impl.enable_overlap = enable_overlap
        self.impl.tile_overlap = (tile_overlap // compression) if enable_overlap else 0  # Convert to latent space or disable
        self.impl.tile_batch_size = tile_batch_size
        self.impl.skip_diffusion_below = skip_diffusion_below  # Store in pixel space, we'll check during diffusion

        # self.impl.init_grid_bbox(tile_width, tile_height, tile_overlap, tile_batch_size)
        # # init everything done, perform sanity check & pre-computations
        # self.impl.init_done()
        # hijack the behaviours
        # self.impl.hook()
        model = model.clone()
        model.set_model_unet_function_wrapper(self.impl)
        model.model_options['tiled_diffusion'] = True
        return (model,)

class SpotDiffusionParams():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                                "shift_method": (["random", "sorted", "fibonacci"], {"default": "random", "tooltip": "Samples a shift size over a uniform distribution to shift tiles."}),
                                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "_for_testing"

    def apply(self, model: ModelPatcher, shift_method, seed):
        model = model.clone()
        model.model_options['tiled_diffusion_seed'] = seed
        model.model_options['tiled_diffusion_shift_method'] = shift_method
        return (model,)

class NoiseInversion():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                                "positive": ("CONDITIONING", ),
                                "negative": ("CONDITIONING", ),
                                "latent_image": ("LATENT", ),
                                "image": ("IMAGE", ),
                                "steps": ("INT", {"default": 10, "min": 1, "max": 208, "step": 1}),
                                "retouch": ("FLOAT", {"default": 1, "min": 1, "max": 100, "step": 0.1}),
                                "renoise_strength": ("FLOAT", {"default": 1, "min": 1, "max": 2, "step": 0.01}),
                                "renoise_kernel_size": ("INT", {"default": 2, "min": 2, "max": 512, "step": 1}),
                            }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    def sample(self, model: ModelPatcher, positive, negative,
                    latent_image, image, steps, retouch, renoise_strength, renoise_kernel_size):
        return (latent_image,)

NODE_CLASS_MAPPINGS = {
    "QuadtreeDiffusion": TiledDiffusion,
    "SpotDiffusionParams_QuadtreeDiffusion": SpotDiffusionParams,
    # "NoiseInversion": NoiseInversion,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QuadtreeDiffusion": "Quadtree Diffusion",
    "SpotDiffusionParams_QuadtreeDiffusion": "SpotDiffusion Parameters (Quadtree)",
    # "NoiseInversion": "Noise Inversion",
}
