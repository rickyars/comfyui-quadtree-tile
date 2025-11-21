# VAEHook feat_cache Issue - Code Reference Guide

## Quick Visual Overview

```
ComfyUI VAEEncode().encode(_vae, samples)
    |
    v
encoder.forward(input_tensor)  <-- This is now VAEHook.__call__()
    |
    v
VAEHook.__call__(x)  <-- PROBLEM: Qwen passes feat_cache=..., feat_idx=...
    |
    v
TypeError: got an unexpected keyword argument 'feat_cache'
```

## Current Broken Code Locations

### 1. Main Problem: VAEHook.__call__ (Line 717-730)

**File:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`

```python
# CURRENT (BROKEN):
def __call__(self, x):  # <-- Only accepts x
    # original_device = next(self.net.parameters()).device
    try:
        # if self.to_gpu:
        #     self.net = self.net.to(devices.get_optimal_device())
        B, C, H, W = x.shape
        if False:#max(H, W) <= self.pad * 2 + self.tile_size:
            print("[Quadtree VAE]: the input size is tiny and unnecessary to tile.", x.shape, self.pad * 2 + self.tile_size)
            return self.net.original_forward(x)  # <-- Also needs kwargs
        else:
            return self.vae_tile_forward(x)  # <-- Also needs kwargs
    finally:
        pass

# SHOULD BE:
def __call__(self, x, **kwargs):  # <-- Accept kwargs
    try:
        B, C, H, W = x.shape
        if False:
            return self.net.original_forward(x, **kwargs)  # <-- Pass kwargs
        else:
            return self.vae_tile_forward(x, **kwargs)  # <-- Pass kwargs
    finally:
        pass
```

### 2. Secondary Problem: vae_tile_forward (Line 911-1103)

**File:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`

```python
# CURRENT (INCOMPLETE):
@perfcount
@torch.no_grad()
def vae_tile_forward(self, z):  # <-- Doesn't accept kwargs
    """
    Decode a latent vector z into an image in a tiled manner.
    @param z: latent vector
    @return: image
    """
    # ... processing code ...

# SHOULD BE:
@perfcount
@torch.no_grad()
def vae_tile_forward(self, z, **kwargs):  # <-- Accept kwargs
    """
    Decode a latent vector z into an image in a tiled manner.
    @param z: latent vector
    @param kwargs: Additional parameters (feat_cache, feat_idx, first_chunk, etc.)
    @return: image
    """
    # Extract parameters from kwargs
    feat_cache = kwargs.get('feat_cache', None)
    feat_idx = kwargs.get('feat_idx', None)
    first_chunk = kwargs.get('first_chunk', False)
    
    # ... rest of processing code ...
```

### 3. Hook Installation (Line 1159-1167)

**File:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`

```python
fn = VAEHook(net=decoder if is_decoder else encoder, 
             tile_size=tile_size // 8 if is_decoder else tile_size,
             is_decoder=is_decoder, fast_decoder=fast, fast_encoder=fast,
             color_fix=color_fix, to_gpu=comfy.model_management.vae_device().type != 'cpu',
             use_quadtree=use_quadtree, content_threshold=content_threshold,
             max_depth=max_depth, min_tile_size=min_tile_size)

if is_decoder:
    decoder.forward = fn
else:
    encoder.forward = fn

# This replaces encoder.forward with VAEHook instance
# When VAEEncode().encode() calls encoder(), it calls VAEHook.__call__()
```

### 4. The Call Site (Line 1173)

**File:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`

```python
ret = (None,)
try:
    with devices.without_autocast():
        if not is_decoder:
            ret = VAEEncode().encode(_vae, samples)  # <-- Line 1173
            # This calls: encoder(samples) which is now VAEHook.__call__(samples)
            # Qwen encoder tries: encoder(samples, feat_cache=..., feat_idx=...)
            # Our hook signature: VAEHook.__call__(self, x) <- FAILS
        else:
            ret = VAEDecode().decode(_vae, samples) if is_decoder else VAEEncode().encode(_vae, samples)
finally:
    # Cleanup hooks
    if isinstance(encoder.forward, VAEHook):
        encoder.forward.net = None
        encoder.forward = encoder.original_forward
```

## Expected Qwen VAE Signatures

From HuggingFace Diffusers (AutoencoderKLWan):

```python
# Qwen/Wan Encoder
class WanEncoder3d(nn.Module):
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        """
        Args:
            x: Input tensor
            feat_cache: Optional feature cache list
            feat_idx: Cache index tracker [0]
        """
        # ... causal 3D conv processing ...
        return encoded_output

# Qwen/Wan Decoder
class WanDecoder3d(nn.Module):
    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        """
        Args:
            x: Input latent tensor
            feat_cache: Optional feature cache list
            feat_idx: Cache index tracker [0]
            first_chunk: Processing first chunk flag
        """
        # ... causal 3D conv processing ...
        return decoded_output
```

## Parameter Details

### feat_cache
- **Type:** `list[Optional[torch.Tensor]]` or `None`
- **Purpose:** Stores intermediate feature maps from causal convolutions
- **Default:** `None`
- **Usage:** For maintaining temporal/spatial continuity across tiles
- **Example:** `feat_cache = [None] * num_conv_layers`

### feat_idx
- **Type:** `list[int]` - mutable list
- **Purpose:** Tracks current position in cache during forward pass
- **Default:** `[0]`
- **Usage:** Gets incremented as processing progresses through layers
- **Example:** `feat_idx = [0]` -> incremented to `[1]`, `[2]`, etc.

### first_chunk (Decoder only)
- **Type:** `bool`
- **Purpose:** Indicates if processing the first chunk/frame
- **Default:** `False`
- **Usage:** Affects how cache is initialized and concatenated

## Impact Analysis

### What Breaks
- Any Qwen or Wan VAE model used with our tiled VAE hooks
- ComfyUI nodes using Qwen models with `VAEEncodeTiled_QuadtreeDiffusion` or `VAEDecodeTiled_QuadtreeDiffusion`
- Custom workflows using Qwen VAEs

### What Still Works
- All standard SD/SDXL/Flux VAE models (they don't use feat_cache)
- ComfyUI's built-in tiled VAE nodes
- Non-tiled VAE processing

### Backwards Compatibility
- The fix should be fully backwards compatible
- Standard VAEs will just ignore the extra kwargs
- No breaking changes to existing API

## Fix Priority & Complexity

| Aspect | Difficulty | Priority | Time |
|--------|-----------|----------|------|
| Accept kwargs | Easy | HIGH | 5 min |
| Pass through params | Easy | HIGH | 5 min |
| Fallback strategy | Medium | HIGH | 15 min |
| Full tiling support | Hard | LOW | 1-2 hr |
| Testing | Medium | HIGH | 30 min |

## Recommended Implementation Order

1. **Step 1 (5 min):** Update `__call__` signature to accept `**kwargs`
2. **Step 2 (5 min):** Update `vae_tile_forward` signature to accept `**kwargs`
3. **Step 3 (5 min):** Pass kwargs through to `original_forward` calls
4. **Step 4 (10 min):** Add fallback for feat_cache (bypass tiling)
5. **Step 5 (30 min):** Test with Qwen models
6. **Step 6 (60 min):** Implement proper feat_cache handling in tiling (if needed)

## Testing Checklist

- [ ] Qwen VAE encoder works with tiling
- [ ] Qwen VAE decoder works with tiling
- [ ] Standard SD VAE still works (no regression)
- [ ] SDXL models still work
- [ ] Flux models still work
- [ ] Quadtree processing with Qwen VAE
- [ ] Feature cache is maintained correctly
- [ ] Memory usage is reasonable

