# Encoder3d Compatibility: Implementation Guide

**Status:** Quick Reference for Developers  
**Last Updated:** November 21, 2025

---

## 1. Quick Problem Statement

### Current Error
```
AttributeError: 'WanEncoder3d' object has no attribute 'conv_in'
```

**What's Really Happening:**
- `conv_in` actually EXISTS
- The real failure is later in `build_sampling()` at line 492: `module = net.down`
- Encoder3d uses `down_blocks` (flat list), not `down` (hierarchical structure)

### Why It Fails

```python
# build_sampling expects this structure (Standard VAE):
net.down[i_level].block[i_block]        # Access block at level i, block j
net.down[i_level].downsample            # Access downsample at level i

# But Encoder3d has:
net.down_blocks[j]                       # Flat list - no level structure!
# Can't determine which index is at which level
```

---

## 2. Encoder3d Architecture Reference

### Initialization Pattern

```python
# WanEncoder3d.__init__() simplified:

class WanEncoder3d(nn.Module):
    def __init__(self, in_channels=3, dim=128, z_dim=4, 
                 dim_mult=[1, 2, 4, 4], num_res_blocks=2, ...):
        
        self.conv_in = WanCausalConv3d(in_channels, dim, 3, padding=1)
        
        # FLAT list - no hierarchy!
        self.down_blocks = nn.ModuleList()
        
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Add ResNet blocks
            for _ in range(num_res_blocks):
                self.down_blocks.append(WanResidualBlock(...))
                if in_attention_scales:
                    self.down_blocks.append(WanAttentionBlock(...))
            
            # Add downsample (if not last level)
            if i != len(dim_mult) - 1:
                self.down_blocks.append(WanResample(...))
        
        self.mid_block = WanMidBlock(...)
        self.norm_out = WanRMS_norm(...)
        self.conv_out = WanCausalConv3d(...)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # Sequential processing
        x = self.conv_in(x, feat_cache[idx] if feat_cache else None)
        for layer in self.down_blocks:
            x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        # ... etc
```

### Key Attributes Reference

| Attribute | Type | Available? | Notes |
|-----------|------|-----------|-------|
| `conv_in` | WanCausalConv3d | ✅ YES | Entry layer |
| `down` | ModuleList | ❌ NO | **USE `down_blocks` INSTEAD** |
| `down_blocks` | ModuleList | ✅ YES | Flat list of layers |
| `mid` | Module | ❌ NO | **USE `mid_block` INSTEAD** |
| `mid_block` | WanMidBlock | ✅ YES | Single middle block |
| `norm_out` | WanRMS_norm | ✅ YES | Output normalization |
| `conv_out` | WanCausalConv3d | ✅ YES | Output convolution |
| `give_pre_end` | bool | ❌ NO | Doesn't exist in Encoder3d |
| `tanh_out` | bool | ❌ NO | Doesn't exist in Encoder3d |
| `num_resolutions` | int | ❌ NO | Doesn't exist |
| `num_res_blocks` | int | ✅ YES | BUT used differently |

---

## 3. Detection Code

### How to Identify Encoder3d

```python
def is_encoder3d(net):
    """Check if this is a Wan Encoder3d/Decoder3d"""
    # Encoder3d has down_blocks (flat list), not down (hierarchical)
    has_down_blocks = hasattr(net, 'down_blocks')
    lacks_down = not hasattr(net, 'down')
    
    return has_down_blocks and lacks_down

def is_standard_vae(net):
    """Check if this is a standard SD/SDXL VAE"""
    return hasattr(net, 'down') and hasattr(net, 'num_resolutions')
```

### Detection Points

In `build_task_queue()`:
```python
def build_task_queue(net, is_decoder):
    # ADD THIS AT THE START:
    if hasattr(net, 'down_blocks') and not hasattr(net, 'down'):
        # This is Encoder3d - not compatible with current tiling
        logger.warning(f"Encoder3d detected - tiling not supported")
        return None  # Signal: use full forward pass
    
    # Continue with existing logic
    task_queue = []
    task_queue.append(('conv_in', net.conv_in))
    build_sampling(task_queue, net, is_decoder)
    # ... rest of function
```

---

## 4. Current Code Issues (Detailed)

### Issue 1: Line 492 - Missing `down` Attribute

**Current Code:**
```python
# tiled_vae.py, line 492
else:
    resolution_iter = range(net.num_resolutions)
    block_ids = net.num_res_blocks
    condition = net.num_resolutions - 1
    module = net.down  # ❌ FAILS: Encoder3d has 'down_blocks', not 'down'
    func_name = 'downsample'
```

**Why It Fails:**
- Standard VAE: `net.down` = list of resolution levels with hierarchical structure
- Encoder3d: `net.down_blocks` = flat list with mixed layer types

**Can't Be Fixed Simply** because:
- Even if we changed `net.down` → `net.down_blocks`
- The code expects `module[i_level].block[i_block]` structure
- But Encoder3d has flat list where we can't determine what's at each "level"

---

### Issue 2: Line 497 - Accessing Non-Existent `.block` Attribute

**Current Code:**
```python
# tiled_vae.py, line 497
for i_level in resolution_iter:
    for i_block in range(block_ids):
        resblock2task(task_queue, module[i_level].block[i_block])
        # ❌ FAILS: module[i_level] is a WanResidualBlock (not a container)
        #           It has no .block attribute
```

**What Exists in Encoder3d:**
```python
# module[0] might be WanResidualBlock - has no .block attribute
# Just a single layer, not a container for blocks

# To access the same layer:
# module[0]  # This IS the block, not a container of blocks
```

---

### Issue 3: Line 480-481 - Missing `.block_1`, `.attn_1`, `.block_2`

**Current Code:**
```python
# tiled_vae.py, line 480-482
if is_decoder:
    resblock2task(task_queue, net.mid.block_1)  # ❌ FAILS
    attn2task(task_queue, net.mid.attn_1)       # ❌ FAILS
    resblock2task(task_queue, net.mid.block_2)  # ❌ FAILS
```

**What Encoder3d Has:**
```python
# net.mid_block = WanMidBlock (single object)
# Has: resnets[0], attentions[0], resnets[1], resnets[2]
# Doesn't have: .block_1, .block_2, .attn_1 (as direct attributes)

# To access the same concepts:
net.mid_block.resnets[0]        # First ResNet block
net.mid_block.attentions[0]     # First attention block
net.mid_block.resnets[1]        # Second ResNet block (after attention)
net.mid_block.resnets[2]        # Third ResNet block
```

---

### Issue 4: Lines 521-526 - Non-Existent Attributes

**Current Code:**
```python
# tiled_vae.py, line 521-526
if not is_decoder or not net.give_pre_end:  # ❌ give_pre_end doesn't exist
    task_queue.append(('pre_norm', net.norm_out))
    task_queue.append(('silu', inplace_nonlinearity))
    task_queue.append(('conv_out', net.conv_out))
    if is_decoder and net.tanh_out:  # ❌ tanh_out doesn't exist
        task_queue.append(('tanh', torch.tanh))
```

**In Encoder3d:**
```python
# Encoder3d doesn't have give_pre_end or tanh_out
# But it always applies norm_out and conv_out
# It doesn't have tanh activation

# Should be:
task_queue.append(('pre_norm', net.norm_out))
task_queue.append(('silu', inplace_nonlinearity))
task_queue.append(('conv_out', net.conv_out))
# No tanh for Encoder3d
```

---

## 5. Recommended Implementation

### Option A: Simple Detection & Bypass (15 minutes)

**File:** `tiled_vae.py`

**Change 1: Lines 507-530 (build_task_queue)**

```python
def build_task_queue(net, is_decoder):
    """
    Build a single task queue for the encoder or decoder
    @param net: the VAE decoder or encoder network
    @param is_decoder: currently building decoder or encoder
    @return: the task queue, or None if not compatible
    """
    # NEW: Detect Encoder3d/Decoder3d (not compatible with tiling)
    if hasattr(net, 'down_blocks') and not hasattr(net, 'down'):
        # This is Wan Encoder3d/Decoder3d - uses flat sequential architecture
        # Tiling not supported due to:
        # 1. Flat down_blocks list (not hierarchical)
        # 2. 5D tensor (B,C,T,H,W) with temporal causality
        # 3. Stateful feat_cache management
        print(f"[Quadtree VAE]: {net.__class__.__name__} uses 3D causal architecture")
        print(f"[Quadtree VAE]: Tiling not supported - will use full forward pass")
        return None  # Signal to bypass tiling
    
    # Existing code for standard VAEs
    task_queue = []
    task_queue.append(('conv_in', net.conv_in))

    # construct the sampling part of the task queue
    build_sampling(task_queue, net, is_decoder)

    if not is_decoder or not net.give_pre_end:
        task_queue.append(('pre_norm', net.norm_out))
        task_queue.append(('silu', inplace_nonlinearity))
        task_queue.append(('conv_out', net.conv_out))
        if is_decoder and net.tanh_out:
            task_queue.append(('tanh', torch.tanh))

    return task_queue
```

**Change 2: Lines 999-1002 (vae_tile_forward)**

```python
# Around line 999-1002, where single_task_queue is created:
single_task_queue = build_task_queue(net, is_decoder)

# NEW: Handle None return (Encoder3d case)
if single_task_queue is None:
    print(f'[Quadtree VAE]: Architecture not compatible with tiling, using full forward')
    # Use full forward pass without tiling
    if self.fast_mode:
        print(f'[Quadtree VAE]: Fast mode ignored for this architecture')
    return self.net.original_forward(z, **kwargs)  # Pass through kwargs!

# Continue with existing tiling logic
if self.fast_mode:
    # ... rest of code
```

### Option B: Advanced Detection with Fallback (30 minutes)

Add this to `VAEHook.__call__()`:

```python
def __call__(self, x, **kwargs):
    try:
        # NEW: Detect Encoder3d with 5D input
        is_5d = len(x.shape) == 5
        is_encoder3d = hasattr(self.net, 'down_blocks') and not hasattr(self.net, 'down')
        
        if is_encoder3d and is_5d:
            print(f"[Quadtree VAE]: Encoder3d detected with 5D tensor - using full pass")
            return self.net.original_forward(x, **kwargs)
        
        # Handle 4D Qwen input (squeeze from 5D)
        if is_5d and not is_encoder3d:
            B, C, T, H, W = x.shape
            if T != 1:
                print(f"[Quadtree VAE]: Multi-frame 5D tensor, bypassing tiling")
                return self.net.original_forward(x, **kwargs)
            x = x.squeeze(2)  # Single frame - continue with tiling
        
        # Existing code
        # ... rest of function
        
        if is_5d and result.dim() == 4:
            result = result.unsqueeze(2)
        
        return result
```

---

## 6. Testing Checklist

After implementing detection/bypass:

- [ ] **Standard VAEs Still Work**
  - [ ] SD 1.5 encoder/decoder
  - [ ] SDXL encoder/decoder
  - [ ] Other SD variants

- [ ] **Qwen VAE Works**
  - [ ] Qwen VAE encoder runs without errors
  - [ ] Qwen VAE decoder runs without errors
  - [ ] Output quality is acceptable (compare with no-tiling baseline)
  - [ ] No feature cache errors

- [ ] **5D Tensor Handling**
  - [ ] 5D input (B,C,T,H,W) is properly handled
  - [ ] Single-frame 5D input (T=1) works
  - [ ] Multi-frame 5D input (T>1) bypasses tiling
  - [ ] Result shape is correct

- [ ] **Messages Are Clear**
  - [ ] Warning message appears when Encoder3d detected
  - [ ] Message explains why tiling is disabled
  - [ ] No confusing error messages

---

## 7. Performance Impact

### Memory Usage (Expected)

**Standard VAE with Tiling:** 
- 8K image: ~4-6 GB VRAM with tiling vs 12+ GB without

**Qwen VAE without Tiling:**
- Large video: Must fit entirely in VRAM
- Typical 512x512 video 32 frames: ~6-8 GB

### Workarounds for Qwen Users

If users hit OOM with Qwen:

1. **Reduce resolution** before encoding
2. **Process in chunks** (split video into smaller clips)
3. **Use CPU offloading** (move non-active layers to CPU)
4. **Use bfloat16** for reduced precision (if model supports)

---

## 8. Error Messages Reference

### What Users Might See

**Before Fix:**
```
AttributeError: 'WanEncoder3d' object has no attribute 'down'
```

**After Fix:**
```
[Quadtree VAE]: WanEncoder3d uses 3D causal architecture
[Quadtree VAE]: Tiling not supported - will use full forward pass
[Quadtree VAE]: Execution completed without tiling
```

---

## 9. Code Changes Summary

**Total Changes Needed:** ~20 lines of code

| File | Function | Lines | Changes |
|------|----------|-------|---------|
| tiled_vae.py | `build_task_queue()` | 507-530 | Add Encoder3d detection, return None |
| tiled_vae.py | `vae_tile_forward()` | 999-1025 | Handle None from build_task_queue |
| tiled_vae.py | `VAEHook.__call__()` | 717-751 | Optional: Add 5D/Encoder3d handling |

---

## 10. Implementation Order

1. **Step 1:** Add `if hasattr(net, 'down_blocks')...` check to `build_task_queue()`
2. **Step 2:** Handle `None` return in `vae_tile_forward()`
3. **Step 3:** Test with both standard VAE and Qwen VAE
4. **Step 4:** Verify error messages are helpful
5. **Step 5:** Document in README

**Estimated Time:** 20-30 minutes for implementation + testing

---

## References

- Wan VAE Architecture: https://github.com/Wan-Video/Wan2.1
- HuggingFace Diffusers Implementation: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_wan.py
- ComfyUI Issues: https://github.com/comfyorg/ComfyUI/issues

