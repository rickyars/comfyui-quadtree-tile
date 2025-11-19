# Skip Feature Implementation Analysis

**Date**: 2025-11-19
**Purpose**: Analyze skip feature for potential use in variable denoise blending approach

---

## Executive Summary

The **skip feature** is a mechanism that preserves original content for small tiles by returning a velocity/noise prediction that **moves the sampler toward the original latent**, effectively bypassing model inference for those tiles.

**Key Findings**:
1. ✅ Skip accesses **clean original latent** from `store.latent_image` (img2img workflows)
2. ✅ Computes velocity prediction: `original - x_in` (points FROM noisy TO clean)
3. ✅ Uses **Gaussian weighting** when overlap is enabled
4. ⚠️ **Binary approach** - tiles are either 100% skipped or 100% processed
5. 💡 **CAN be adapted for variable denoise blending** with modifications

**Recommendation**: The skip approach provides the missing piece for variable denoise - **accessing the original clean latent**. However, the current implementation is binary. We need to **blend** between model output and original-preserving velocity based on denoise strength.

---

## 1. Skip Feature Overview

### What It Does

The skip feature allows small tiles (below a threshold size) to **preserve their original content** in img2img workflows by:
1. Bypassing expensive model inference for those tiles
2. Computing a velocity/noise prediction that restores the original content
3. Adding this prediction to the output buffer with proper weighting

### Parameters

**File**: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:1545`

```python
"skip_diffusion_below": ("INT", {
    "default": 0,
    "min": 0,
    "max": 512,
    "step": 8,
    "tooltip": "Skip diffusion for tiles smaller than this size in pixels (0 = disabled). "
               "Preserves original details for small tiles. Must be multiple of 8 for VAE compatibility."
})
```

**Usage**:
- `0` = disabled (all tiles processed through model)
- `256` = tiles with min(width, height) < 256px are skipped

---

## 2. Complete Skip Implementation

### 2.1 Original Latent Storage

**File**: `/home/user/comfyui-quadtree-tile/utils.py:32-37`

The original latent is captured during sampling initialization:

```python
# Capture latent_image for img2img workflows (preserves original content for skipped tiles)
latent_image = kwargs.get('latent_image') if 'latent_image' in kwargs else (args[6] if len(args) > 6 else None)
if latent_image is not None:
    store.latent_image = latent_image  # CLEAN original latent (before noise added)
else:
    _delattr(store, 'latent_image')  # txt2img has no latent_image
```

**Critical Detail**: `store.latent_image` contains the **CLEAN original latent** (before noise is added by the sampler). This is the encoded VAE output from the original image.

**File**: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:1170-1180`

The latent is loaded into the diffusion implementation:

```python
# Store original latent for img2img skip feature (from store if available)
if not hasattr(self, 'original_latent'):
    try:
        from .utils import store
        if hasattr(store, 'latent_image'):
            self.original_latent = store.latent_image  # CLEAN original latent
            print(f'[Quadtree Skip]: Loaded original latent for img2img, shape={self.original_latent.shape}')
        else:
            self.original_latent = None  # txt2img has no original latent
    except Exception as e:
        print(f'[Quadtree Skip]: WARNING - Failed to load original latent: {e}')
        self.original_latent = None
```

### 2.2 Tile Classification

**File**: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:1280-1296`

Tiles are separated into "skip" and "process" lists based on size:

```python
# SKIP LOGIC: Separate tiles into skip and process lists
process_bboxes = []
skip_bboxes = []

if use_qt and skip_threshold > 0:
    for bbox in bboxes:
        pixel_w = getattr(bbox, 'pixel_w', 0)  # Tile width in PIXEL space
        pixel_h = getattr(bbox, 'pixel_h', 0)  # Tile height in PIXEL space
        min_dimension = min(pixel_w, pixel_h)

        if min_dimension < skip_threshold:
            skip_bboxes.append(bbox)  # Too small - skip model inference
        else:
            process_bboxes.append(bbox)  # Normal size - process through model
else:
    # No skipping, process all tiles
    process_bboxes = bboxes
```

**Binary Decision**: Each tile is either:
- **Skipped** (min dimension < threshold) → preserve original
- **Processed** (min dimension >= threshold) → run through model

### 2.3 Skip Tile Processing

**File**: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:1298-1349`

This is the core of the skip implementation:

```python
# SKIP TILES: Handle tiles that will not go through model inference
if len(skip_bboxes) > 0:
    if hasattr(self, 'original_latent') and self.original_latent is not None:
        # img2img: Compute model prediction that moves toward original content
        # FLUX uses velocity prediction (Rectified Flow): velocity points toward target
        # Formula: predicted_velocity = original - x_in (direction toward target)
        # Sampler does: x_next = x_in + predicted_velocity * dt → converges to original ✓
        for bbox in skip_bboxes:
            # Extract tiles from both current noisy state and original clean latent
            x_in_tile = extract_tile_with_padding(x_in, bbox, self.w, self.h)
            original_tile = extract_tile_with_padding(self.original_latent, bbox, self.w, self.h)

            # Crop both to bbox size if padded
            if x_in_tile.shape[-2] > bbox.h or x_in_tile.shape[-1] > bbox.w:
                x_in_tile = x_in_tile[:, :, :bbox.h, :bbox.w]
            if original_tile.shape[-2] > bbox.h or original_tile.shape[-1] > bbox.w:
                original_tile = original_tile[:, :, :bbox.h, :bbox.w]

            # Compute velocity/noise prediction that will restore original
            # FLUX (velocity): velocity = original - x_in (points toward target)
            # SD1.5/SDXL (noise): noise = x_in - original (what to subtract)
            # FLUX is velocity-based (additive), so use: original - x_in
            model_prediction = original_tile - x_in_tile

            # Calculate intersection with image boundaries
            x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
            x_start = max(0, x)
            y_start = max(0, y)
            x_end = min(self.w, x + w)
            y_end = min(self.h, y + h)

            # Calculate offset into tiles
            tile_x_offset = x_start - x
            tile_y_offset = y_start - y

            # Extract valid portion (remove padding)
            valid_prediction = model_prediction[:, :,
                                          tile_y_offset:tile_y_offset + (y_end - y_start),
                                          tile_x_offset:tile_x_offset + (x_end - x_start)]

            # Add to buffer with Gaussian weighting if overlap enabled
            if self.tile_overlap > 0:
                tile_weights_full = self.get_weight(bbox.w, bbox.h)
                tile_weights = tile_weights_full[tile_y_offset:tile_y_offset + (y_end - y_start),
                                                tile_x_offset:tile_x_offset + (x_end - x_start)]
                tile_weights = tile_weights.unsqueeze(0).unsqueeze(0)
                self.x_buffer[:, :, y_start:y_end, x_start:x_end] += valid_prediction * tile_weights
            else:
                self.x_buffer[:, :, y_start:y_end, x_start:x_end] = valid_prediction
    # else: txt2img - zero velocity/noise prediction (do nothing, stays at zero in x_buffer)
    # This preserves current noisy state, which for txt2img will appear gray
    # Recommendation: Use variable_denoise instead for txt2img workflows
```

**Key Points**:
1. **Extracts two tiles**: current noisy state (`x_in`) and clean original (`original_latent`)
2. **Computes velocity**: `original - x_in` (for FLUX) points FROM noisy TO clean
3. **Applies Gaussian weighting**: If overlap enabled, uses same weighting as regular tiles
4. **Adds to buffer**: Accumulated like normal model outputs

---

## 3. Mathematical Analysis

### 3.1 How Skip Preserves Original Content

**Sampler Formula** (FLUX Euler):
```
x_next = x_in + velocity * dt
```

**Skip Formula**:
```
velocity = original - x_in
```

**Result After One Step**:
```
x_next = x_in + (original - x_in) * dt
       = x_in * (1 - dt) + original * dt
       = interpolate(x_in, original, weight=dt)
```

**Progressive Convergence**:
- At early steps (large dt): Moves significantly toward original
- At later steps (small dt): Fine-tunes toward original
- At final step: Reaches original exactly (if dt=1.0)

**Effect**: The sampler **linearly interpolates** from noisy state to clean original over the denoising process.

### 3.2 Comparison with Model Output

**Normal Tile** (processed through model):
```
velocity = model(x_in, t, conditioning)  # Model decides what to generate
x_next = x_in + velocity * dt
```

**Skip Tile** (bypasses model):
```
velocity = original - x_in  # Fixed target: restore original
x_next = x_in + velocity * dt
```

**Key Difference**:
- Normal tiles: Model **generates new content** based on prompt
- Skip tiles: **Converge to original** regardless of prompt

---

## 4. Original Latent Access Details

### 4.1 What is `store.latent_image`?

**Source**: Passed as `latent_image` parameter to KSampler

**Content**: The **clean VAE-encoded latent** of the original image (img2img input)

**Not Noisy**: This is the latent BEFORE noise is added by the sampler:
```python
# In img2img workflow:
# 1. Original image → VAE encode → store.latent_image (CLEAN)
# 2. store.latent_image + noise → noisy_latent (NOISY)
# 3. Sampling starts from noisy_latent
```

### 4.2 Availability

✅ **img2img**: Always available (comes from VAE encode of input image)
❌ **txt2img**: Not available (no input image exists)
❌ **Some custom nodes**: May not pass latent_image parameter

### 4.3 Shape and Format

**Shape**: `[B, C, H, W]` - Same as model's latent space
- B = batch size (usually 1)
- C = channels (4 for SD1.5/SDXL, 16 for FLUX)
- H, W = latent dimensions (image_size / compression)

**Data Type**: `torch.float32` or `torch.float16`
**Device**: Same device as model (GPU)

---

## 5. Bug History and Fix

### 5.1 The Film Negative Bug

**Symptom**: Skipped tiles showed inverted colors (film negative effect) with FLUX

**Root Cause**: Wrong sign in velocity calculation
```python
# WRONG (caused bug):
model_prediction = x_in_tile - original_tile  # Points AWAY from original

# CORRECT (fixed):
model_prediction = original_tile - x_in_tile  # Points TOWARD original
```

**Why SD1.5/SDXL Worked**: They use noise prediction with opposite sign convention
```python
# For noise prediction models (SD1.5, SDXL):
# Sampler: x_next = x_in - noise * dt
# To preserve: noise = (x_in - original) / dt
# Sign happens to be opposite, so wrong formula was correct!
```

### 5.2 The Fix

**Commit**: ee6ac60
**Change**: Corrected velocity formula to `original - x_in`
**Impact**: Skip now works correctly for FLUX (velocity models)

**Full Analysis**: See `/home/user/comfyui-quadtree-tile/docs/bug-analysis/SKIP_TILE_VELOCITY_BUG.md`

---

## 6. Can We Use This for Variable Denoise?

### 6.1 Current Variable Denoise Problem

**Current Approach** (scales model output):
```python
tile_out = tile_out * scale_factor  # scale ∈ [0.7, 1.0]
```

**Problem**:
- Low scale → weak model output → **noisy results**
- No mechanism to preserve original content when scale is low
- Only works by "weakening" the denoising, not by blending with original

### 6.2 Skip-Based Blending Solution

**Concept**: Instead of scaling model output, **blend** between:
1. **Model velocity** (generates new content based on prompt)
2. **Skip velocity** (preserves original content)

**Formula**:
```python
# For each tile:
model_velocity = model(x_in_tile, t, conditioning)  # What model wants to generate
skip_velocity = original_tile - x_in_tile           # What preserves original

# Blend based on denoise strength:
final_velocity = denoise * model_velocity + (1 - denoise) * skip_velocity
```

**Effect**:
- `denoise = 1.0` → 100% model output (full regeneration)
- `denoise = 0.5` → 50/50 blend (partial preservation)
- `denoise = 0.0` → 100% skip (complete preservation)

### 6.3 Mathematical Proof

**Sampler Update**:
```
x_next = x_in + final_velocity * dt
```

**Case 1: denoise = 1.0 (Full Regeneration)**
```
final_velocity = 1.0 * model_velocity + 0.0 * skip_velocity
               = model_velocity
x_next = x_in + model_velocity * dt  ✅ Standard model output
```

**Case 2: denoise = 0.0 (Complete Preservation)**
```
final_velocity = 0.0 * model_velocity + 1.0 * skip_velocity
               = skip_velocity
               = original - x_in
x_next = x_in + (original - x_in) * dt
       = x_in * (1-dt) + original * dt  ✅ Converges to original
```

**Case 3: denoise = 0.3 (Mostly Preserve)**
```
final_velocity = 0.3 * model_velocity + 0.7 * skip_velocity
x_next = x_in + (0.3 * model_velocity + 0.7 * (original - x_in)) * dt
       = x_in + 0.3 * model_velocity * dt + 0.7 * (original - x_in) * dt
       = [x_in + 0.3 * model_velocity * dt] * (1 - 0.7*dt) + original * (0.7*dt)
```
✅ **Blends toward both model output AND original**

### 6.4 Advantages Over Current Approach

| Aspect | Current (Scale Model) | Proposed (Blend Velocities) |
|--------|----------------------|----------------------------|
| **Low denoise quality** | Weak, noisy output | Smooth blend to original |
| **Original preservation** | No mechanism | Direct access to original |
| **Mathematical soundness** | Arbitrary scaling | Linear interpolation |
| **Works for txt2img?** | Yes (but weak) | ❌ No (no original) |
| **Works for img2img?** | Yes (but noisy) | ✅ Yes (perfect blend) |

---

## 7. Implementation Approach for Variable Denoise

### 7.1 Required Changes

**Step 1: Access Original Latent**
```python
# Already implemented for skip feature!
if hasattr(self, 'original_latent') and self.original_latent is not None:
    # img2img workflow - can use blending
    use_blending = True
else:
    # txt2img workflow - fall back to current scaling approach
    use_blending = False
```

**Step 2: Extract Original Tile**
```python
# For each tile being processed:
original_tile = extract_tile_with_padding(self.original_latent, bbox, self.w, self.h)
if original_tile.shape[-2] > bbox.h or original_tile.shape[-1] > bbox.w:
    original_tile = original_tile[:, :, :bbox.h, :bbox.w]
```

**Step 3: Compute Skip Velocity**
```python
skip_velocity = original_tile - x_in_tile  # Direction toward original
```

**Step 4: Blend with Model Output**
```python
# tile_out contains model's velocity prediction
# tile_denoise is the denoise strength for this tile (from bbox.denoise)

if use_blending:
    # Blend between model output and skip velocity
    blended_output = tile_denoise * tile_out + (1 - tile_denoise) * skip_velocity
else:
    # Fall back to current scaling approach (txt2img)
    blended_output = tile_out * scale_factor
```

### 7.2 Progressive Scaling with Blending

**Current Approach**: Scales model output progressively over denoising steps

**Enhanced Approach**: Progressively reduce blending strength
```python
# Early steps: More blending (preserve original structure)
# Later steps: Less blending (allow model creativity)

if use_blending:
    # Adjust denoise strength based on progress
    effective_denoise = start_denoise + (tile_denoise - start_denoise) * progress_curved
    effective_denoise = max(0.0, min(1.0, effective_denoise))

    # Blend
    blended_output = effective_denoise * tile_out + (1 - effective_denoise) * skip_velocity
else:
    # Original scaling approach
    blended_output = tile_out * scale_factor
```

### 7.3 Edge Cases

**Case 1: txt2img (No Original)**
```python
if not hasattr(self, 'original_latent') or self.original_latent is None:
    # Fall back to current scaling approach
    blended_output = tile_out * scale_factor
```

**Case 2: tile_denoise = 1.0**
```python
# No blending needed
blended_output = tile_out  # 100% model output
```

**Case 3: tile_denoise = 0.0**
```python
# Full preservation
blended_output = skip_velocity  # 100% skip velocity
```

---

## 8. Potential Issues and Concerns

### 8.1 ⚠️ Model Prediction Type Compatibility

**Issue**: Different models use different prediction types
- FLUX: velocity prediction (`v`)
- SD1.5/SDXL: noise prediction (`ε`)

**Current Implementation**: Uses `original - x_in` (correct for velocity)

**For Noise Models**: May need opposite sign
```python
# For noise prediction models (SD1.5, SDXL):
# Sampler: x_next = x_in - noise * dt
# To preserve: noise = (x_in - original) / dt

if is_noise_model:
    skip_prediction = x_in_tile - original_tile  # Opposite sign
else:
    skip_prediction = original_tile - x_in_tile  # FLUX (velocity)
```

**Recommendation**: Detect model type and use correct formula (see bug analysis doc)

### 8.2 ⚠️ Blending in Different Prediction Spaces

**Issue**: Blending velocities assumes linear interpolation is meaningful

**Analysis**:
- **Velocity models (FLUX)**: ✅ Linear blending makes sense (both are velocities)
- **Noise models (SD1.5/SDXL)**: ✅ Linear blending makes sense (both are noise predictions)
- **x₀ models**: ⚠️ Would need different approach (blend in latent space, not prediction space)

**Conclusion**: Should work for FLUX and SD1.5/SDXL (most common models)

### 8.3 ⚠️ Gaussian Weighting Interaction

**Issue**: Both blending and Gaussian weighting happen

**Current Flow**:
```python
# 1. Model inference
tile_out = model(x_in_tile, ...)

# 2. Blending (proposed)
blended_out = denoise * tile_out + (1-denoise) * skip_velocity

# 3. Gaussian weighting
self.x_buffer[...] += blended_out * gaussian_weights
```

**Analysis**:
✅ **This is correct** - blending happens per-tile, weighting happens during accumulation

**Order matters**:
1. First blend the predictions (per-tile operation)
2. Then apply Gaussian weights (accumulation operation)

### 8.4 ⚠️ Performance Impact

**Current Skip**: Skips model inference entirely for small tiles (saves compute)

**Proposed Blending**: Still runs model inference for all tiles (no performance gain)

**Trade-off**:
- Skip feature: Fast but binary (all or nothing)
- Blending approach: Slower but smooth (gradual preservation)

**Recommendation**: Keep both features separate
- Skip for performance optimization (binary preservation)
- Variable denoise blending for quality (smooth preservation)

### 8.5 ⚠️ txt2img Degradation

**Issue**: Blending requires original latent, which doesn't exist in txt2img

**Current Behavior**: Scaling approach works for txt2img (though weak at low scale)

**Proposed Behavior**: Fall back to scaling for txt2img
```python
if original_latent is not None:
    # img2img: Use blending (better quality)
    output = blend_with_original(tile_out, original_tile, denoise)
else:
    # txt2img: Use scaling (current approach)
    output = tile_out * scale_factor
```

**Result**:
- ✅ img2img gets better quality (blending with original)
- ✅ txt2img still works (scaling, same as before)

---

## 9. Comparison: Skip vs Variable Denoise Blending

| Feature | Skip Feature | Variable Denoise (Current) | Variable Denoise (Blending) |
|---------|-------------|---------------------------|----------------------------|
| **Decision** | Binary (skip/process) | Continuous (scale 0.7-1.0) | Continuous (blend 0.0-1.0) |
| **Mechanism** | Bypass model inference | Scale model output | Blend model + skip velocities |
| **Original access** | ✅ Yes (clean latent) | ❌ No | ✅ Yes (clean latent) |
| **Performance** | ⚡ Fast (skips inference) | 🐌 Same as normal | 🐌 Same as normal |
| **Quality (low denoise)** | ✅ Perfect preservation | ❌ Noisy/weak | ✅ Smooth blend |
| **Works img2img** | ✅ Yes | ⚠️ Yes (but noisy) | ✅ Yes (perfect) |
| **Works txt2img** | ❌ No (stays gray) | ✅ Yes (weak) | ⚠️ Fallback to scale |
| **Use case** | Small tiles, performance | All tiles, smooth | All tiles, img2img quality |

---

## 10. Recommendations

### 10.1 Immediate Actions

1. **✅ Adopt blending approach for variable denoise in img2img**
   - Use skip velocity formula: `original - x_in`
   - Blend based on tile denoise strength
   - Fall back to scaling for txt2img

2. **✅ Keep skip feature separate**
   - Skip is for performance (binary preservation)
   - Variable denoise is for quality (smooth blending)
   - Document the difference clearly

3. **✅ Detect model type**
   - Check if model uses velocity or noise prediction
   - Use correct sign for skip velocity
   - Warn if model type is unknown

### 10.2 Implementation Priority

**High Priority** (implement first):
1. Access original latent in variable denoise code
2. Compute skip velocity: `original - x_in`
3. Blend: `denoise * model_out + (1-denoise) * skip_velocity`
4. Test with FLUX (velocity model)

**Medium Priority** (implement next):
1. Model type detection (velocity vs noise)
2. Correct sign for noise prediction models
3. Fallback to scaling for txt2img

**Low Priority** (nice to have):
1. Progressive blending strength over steps
2. Performance optimizations
3. Better error messages

### 10.3 Testing Strategy

**Unit Tests**:
- Test skip velocity formula: `original - x_in`
- Test blending at denoise=0.0, 0.5, 1.0
- Test fallback to scaling when no original

**Integration Tests**:
- img2img with denoise range 0.1-0.9 (FLUX)
- img2img with denoise range 0.1-0.9 (SD1.5)
- txt2img with same range (should use scaling)

**Visual Tests**:
- Verify smooth blend from original to generated
- Check for tile boundaries (should be seamless)
- Compare to current scaling approach

---

## 11. Code Locations Reference

**Skip Implementation**:
- Original latent storage: `/home/user/comfyui-quadtree-tile/utils.py:32-37`
- Original latent loading: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:1170-1180`
- Tile classification: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:1280-1296`
- Skip processing: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:1298-1349`
- Skip parameter: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:1545`

**Variable Denoise Implementation**:
- Current scaling: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:1326-1376`
- Denoise assignment: `/home/user/comfyui-quadtree-tile/tiled_vae.py:296-308`

**Helper Functions**:
- `extract_tile_with_padding`: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:72-101`
- `get_weight` (Gaussian): Used in skip processing for overlap weighting

---

## 12. Conclusion

### Key Findings

1. **✅ Skip feature provides the missing piece**: Access to clean original latent
2. **✅ Skip velocity formula is correct**: `original - x_in` (for FLUX)
3. **✅ Blending is mathematically sound**: Linear interpolation in velocity space
4. **⚠️ Need model type detection**: Different signs for velocity vs noise models
5. **⚠️ txt2img needs fallback**: No original latent, use scaling approach

### Proposed Solution

**For img2img workflows**:
```python
# Extract original tile
original_tile = extract_tile_with_padding(self.original_latent, bbox, self.w, self.h)

# Compute skip velocity (preserve original)
skip_velocity = original_tile - x_in_tile

# Get model velocity (generate new content)
model_velocity = model(x_in_tile, t, conditioning)

# Blend based on denoise strength
tile_denoise = bbox.denoise  # 0.0 to 1.0
final_velocity = tile_denoise * model_velocity + (1 - tile_denoise) * skip_velocity

# Add to buffer with Gaussian weighting
self.x_buffer[...] += final_velocity * weights
```

**For txt2img workflows**:
```python
# Fall back to current scaling approach
final_velocity = model_velocity * scale_factor
```

### Impact

**Quality Improvement**:
- ✅ Low denoise tiles: Smooth blend to original (no noise/weakness)
- ✅ High denoise tiles: Full model output (no change)
- ✅ Seamless transition: Denoise becomes true preservation strength

**Compatibility**:
- ✅ img2img: Major improvement (blending with original)
- ✅ txt2img: No regression (fallback to scaling)
- ✅ FLUX: Correct sign (velocity prediction)
- ⚠️ SD1.5/SDXL: Need sign detection (noise prediction)

**Performance**:
- 🐌 No improvement (all tiles still processed)
- 💡 Could combine with skip for best of both worlds

---

**Status**: ✅ READY FOR IMPLEMENTATION
**Confidence**: HIGH - Mathematical foundation is sound, skip feature already proven
**Risk**: LOW - Fallback ensures no regression for txt2img or edge cases

---

**Author**: Analysis via Claude Code
**Date**: 2025-11-19
