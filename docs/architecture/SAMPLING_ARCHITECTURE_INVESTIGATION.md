# ComfyUI Sampling Architecture and Tiled Diffusion Integration

**Investigation Date**: 2025-11-18
**Purpose**: Understand how ComfyUI's sampling/scheduling works and why per-tile variable denoise is fundamentally limited
**Status**: ✅ COMPLETE

---

## Executive Summary

**The Core Problem**: ComfyUI's sampling architecture uses a **single global timestep** for the entire image at each denoising step. While tiled diffusion can process tiles separately, **all tiles must use the same timestep/sigma value**. This architectural constraint means true per-tile variable denoise (where different tiles are at different stages of denoising) is **fundamentally impossible** without major changes to ComfyUI's core sampling code.

**Current Implementation**: The existing variable denoise feature (lines 1419-1465 in `tiled_diffusion.py`) works by **scaling the model output** after prediction. This is a post-hoc approximation that modulates the strength of predictions, not true per-tile denoising control.

**Why Scaling Works (Partially)**: Scaling the noise/velocity prediction effectively reduces the step size for that tile, creating a weaker denoising effect. However, it's not equivalent to running tiles through different denoising schedules.

---

## 1. ComfyUI Sampling Architecture

### 1.1 The Denoising Loop

ComfyUI's samplers follow this high-level flow:

```python
# Pseudocode of ComfyUI's sampling loop
latent = initial_noise  # or noised image for img2img

for i, (t_curr, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
    # Single timestep for entire image
    model_output = model.apply_model(
        input=latent,
        timestep=t_curr,  # ← ONE VALUE FOR ENTIRE IMAGE
        c=conditioning
    )

    # Update latent based on sampler type (Euler, DPM++, etc.)
    latent = sampler_update(latent, model_output, t_curr, t_next)

return latent
```

**Key Observation**: The timestep `t_curr` is a **scalar value** (or batch of identical values) that applies to the **entire latent tensor**. There is no mechanism for different spatial regions to have different timesteps.

### 1.2 Sigma Schedule

The `sigmas` tensor defines the noise schedule:

```python
# Example sigma schedule (from utils.py:25-28)
sigmas = [14.6, 10.3, 7.3, 5.1, ..., 0.0]  # High to low noise
```

- **Length**: `num_steps + 1` (e.g., 20 steps → 21 sigmas)
- **Direction**: High noise → low noise (denoising progression)
- **Global**: One schedule for the entire image

At each step, the sampler picks one sigma from this schedule and uses it for the entire image.

### 1.3 Model Function Signature

```python
# From BaseModel.apply_model (inferred from code)
def apply_model(input, timestep, c, cond_or_uncond):
    """
    Args:
        input: Noisy latent [B, C, H, W]
        timestep: Current timestep [B] or scalar - SAME for all spatial locations
        c: Conditioning dict with various tensors
        cond_or_uncond: List indicating which samples are conditional

    Returns:
        prediction: Model output [B, C, H, W]
            - Noise prediction (SD1.5/SDXL)
            - Velocity prediction (FLUX)
            - Denoised prediction (some models)
    """
```

**Critical Point**: The `timestep` parameter is **not spatially varying**. It's a single value (or batch of identical values) that applies to all pixels.

---

## 2. Tiled Diffusion Integration

### 2.1 Model Wrapping

**Location**: `tiled_diffusion.py:1594`

```python
model = model.clone()
model.set_model_unet_function_wrapper(self.impl)
```

This **wraps** the model's `apply_model` function. When the sampler calls `model.apply_model()`, it actually calls our wrapper.

### 2.2 Wrapper Call Signature

**Location**: `tiled_diffusion.py:747, 915, 1139` (three implementations)

```python
def __call__(self, model_function: BaseModel.apply_model, args: dict):
    """
    Args:
        model_function: Original model.apply_model function
        args: Dict containing:
            - "input": Noisy latent [B, C, H, W]
            - "timestep": Current timestep [B]
            - "c": Conditioning dict
            - "cond_or_uncond": List of cond flags

    Returns:
        output: Model prediction [B, C, H, W]
    """
    x_in = args["input"]      # Full noisy latent
    t_in = args["timestep"]   # ONE timestep for entire image
    c_in = args["c"]          # Conditioning
```

**What the Wrapper Does**:
1. Receives the FULL latent and ONE timestep
2. Splits the latent into tiles
3. Calls `model_function(tile, t_in, ...)` for each tile **with the SAME timestep**
4. Blends tile predictions back into full prediction
5. Returns combined prediction to sampler

### 2.3 Tile Processing Flow

```python
# Simplified from MixtureOfDiffusers.__call__ (lines 1269-1404)
for batch_id, bboxes in enumerate(self.batched_bboxes):
    # Extract tiles from full latent
    x_tile = torch.cat([x_in[bbox.slicer] for bbox in bboxes], dim=0)

    # Repeat the SAME timestep for all tiles
    t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])  # ← SAME VALUE REPEATED

    # Call model on tiles
    x_tile_out = model_function(x_tile, t_tile, **c_tile)

    # Blend predictions
    for i, bbox in enumerate(bboxes):
        tile_out = x_tile_out[i*N:(i+1)*N, :, :, :]
        self.x_buffer[bbox.slicer] += tile_out * weights
```

**Key Insight**: Even though tiles are processed separately, they **all receive the SAME timestep** (`t_in`). This is fundamental to how ComfyUI's wrapper API works.

---

## 3. Current Variable Denoise Implementation

### 3.1 What It Does

**Location**: `tiled_diffusion.py:1419-1465` (MixtureOfDiffusers)

```python
# Get tile's assigned denoise value (0.0-1.0)
tile_denoise = getattr(bbox, 'denoise', 1.0)

if use_qt and hasattr(self, 'sigmas') and tile_denoise < 1.0:
    # Calculate progress through schedule (0 = start, 1 = end)
    sigmas = self.sigmas
    ts_in = find_nearest(t_in[0], sigmas)
    cur_idx = (sigmas == ts_in).nonzero()
    current_step = cur_idx.item()
    progress = current_step / total_steps

    # Calculate scale factor based on tile denoise and progress
    start_scale = 0.70 + (tile_denoise * 0.25)  # 0.70-0.95
    ramp_curve = 1.0 + tile_denoise              # 1.2-1.8
    progress_curved = min(1.0, pow(progress, 1.0 / ramp_curve))
    scale_factor = start_scale + (1.0 - start_scale) * progress_curved

    # Scale the model output
    tile_out = tile_out * scale_factor  # ← POST-HOC SCALING
```

**Behavior**:
- **Large tiles** (low denoise): `scale_factor ≈ 0.70-0.75` early, ramps to 1.0
- **Small tiles** (high denoise): `scale_factor ≈ 0.90-0.95` early, ramps to 1.0
- **Progressive**: Scale increases over denoising steps

### 3.2 Why This Approach Was Chosen

Given the architectural constraint (one global timestep), scaling the output is the **only option available**:

1. ✅ **Can be done in wrapper**: Post-processing after model call
2. ✅ **No ComfyUI core changes**: Works within existing API
3. ✅ **Per-tile control**: Can scale each tile differently
4. ✅ **Preserves sampling algorithm**: Doesn't break sampler logic

### 3.3 What Scaling Actually Does

Scaling the model output has this effect:

**For Noise Prediction (SD1.5/SDXL)**:
```python
# Normal step
x_next = x_t - sigma * noise_prediction

# With scaling (scale < 1.0)
x_next = x_t - sigma * (scale * noise_prediction)
       = x_t - (sigma * scale) * noise_prediction
       # Effectively reduces sigma (noise level) for this tile
```

**For Velocity Prediction (FLUX)**:
```python
# Normal step
x_next = x_t + dt * velocity_prediction

# With scaling (scale < 1.0)
x_next = x_t + dt * (scale * velocity_prediction)
       = x_t + (dt * scale) * velocity_prediction
       # Effectively reduces dt (step size) for this tile
```

**Result**: Scaling makes the denoising step **weaker** for that tile, preserving more of its current state.

---

## 4. Why True Per-Tile Denoise Is Impossible

### 4.1 What "True" Variable Denoise Would Require

For different tiles to be at different denoising stages, you would need:

```python
# Hypothetical (impossible with current architecture)
for batch_id, bboxes in enumerate(self.batched_bboxes):
    for i, bbox in enumerate(bboxes):
        # Each tile gets its OWN timestep based on denoise value
        tile_timestep = calculate_tile_timestep(
            base_timestep=t_in,
            tile_denoise=bbox.denoise,
            progress=current_step
        )

        # Process tile at its specific denoising stage
        tile_out = model_function(
            x_tile[i],
            tile_timestep,  # ← DIFFERENT per tile
            **c_tile
        )
```

**Why This Won't Work**:
1. Model expects batched input with **same timestep** for efficiency
2. Sampler tracks ONE denoising trajectory for the entire image
3. Can't have tile A at step 5 and tile B at step 15 simultaneously
4. Final combined latent would be **temporally incoherent**

### 4.2 The Temporal Coherence Problem

If tiles were at different denoising stages:

```
Time:     t=5        t=10       t=15
Tile A:   ████████   ██████     ████      (mostly denoised)
Tile B:   ████████   ████████   ██████    (still noisy)
          ↓
Combined: [INCOHERENT - different noise levels]
```

When sampler does the next update, it would apply ONE timestep to a latent that's already at **different stages** in different regions. This breaks the math of iterative denoising.

### 4.3 Architectural Constraints

**ComfyUI's Sampling API** (`set_model_unet_function_wrapper`):
- ✅ Can intercept model calls
- ✅ Can split input spatially
- ✅ Can process tiles separately
- ❌ **Cannot** change timestep per tile
- ❌ **Cannot** run tiles through different schedules
- ❌ **Cannot** maintain per-tile temporal state

**To implement true variable denoise would require**:
1. Modifying ComfyUI's core sampler code
2. Tracking separate denoising states per region
3. Custom blending logic that handles temporal incoherence
4. Likely breaking compatibility with standard samplers

---

## 5. Why Current Scaling Approach Works (Partially)

### 5.1 Mathematical Justification

Scaling the prediction approximates a **weaker denoising step**:

**Noise Prediction Models**:
```
Normal:  x_{t-1} = x_t - σ_t * ε
Scaled:  x_{t-1} = x_t - σ_t * (α * ε)  where α < 1
                 = x_t - (α * σ_t) * ε
Effect: Acts like using smaller sigma (less denoising)
```

**Velocity Prediction Models**:
```
Normal:  x_{t+1} = x_t + Δt * v
Scaled:  x_{t+1} = x_t + Δt * (α * v)  where α < 1
                 = x_t + (α * Δt) * v
Effect: Acts like using smaller step size (less denoising)
```

### 5.2 Progressive Ramping

The implementation ramps `scale_factor` from 0.70-0.95 → 1.0 over the schedule:

```python
# At early steps (high noise)
tile_denoise=0.2: scale=0.75 (25% reduction in step size)
tile_denoise=0.8: scale=0.90 (10% reduction in step size)

# At late steps (low noise)
All tiles: scale=1.0 (full strength, detail refinement)
```

**Why This Makes Sense**:
- **Early steps**: Large tiles take smaller steps (preserve structure)
- **Late steps**: All tiles take full steps (refine details)
- **Result**: Large tiles preserve content, small tiles regenerate

### 5.3 Limitations

This approach is **NOT** equivalent to true variable denoise because:

1. **Still uses same timestep**: All tiles see the same noise level
2. **Post-hoc approximation**: Not how schedulers are designed to work
3. **Limited control range**: Can only reduce strength, not increase
4. **No temporal independence**: Can't have tiles at different stages

**But it works well enough for practical use** because:
- img2img: Preserves large regions while enhancing details
- txt2img: Creates smooth variation without hard boundaries
- Blending: Gaussian weights smooth any artifacts

---

## 6. Alternative Approaches Considered

### 6.1 Skip Feature (Different Mechanism)

**Location**: `tiled_diffusion.py:1274-1343`

Instead of weak denoising, **completely skip** model inference for small tiles:

```python
if min_dimension < skip_threshold:
    # Don't process through model
    # For img2img: Copy prediction that preserves original
    # For txt2img: Contribute zero to buffer
    skip_bboxes.append(bbox)
```

**Advantages**:
- ✅ Faster (no model inference for skipped tiles)
- ✅ Perfect preservation (img2img)
- ✅ No approximation artifacts

**Disadvantages**:
- ❌ All-or-nothing (no gradual control)
- ❌ Requires original latent for img2img
- ❌ txt2img produces gray regions for skipped tiles

**Status**: Implemented and working (after velocity bug fix)

### 6.2 Multi-Pass Denoising (Theoretical)

Process different tiles through **separate denoising passes**:

```python
# Pass 1: Large tiles with weak schedule
large_tiles = denoise(large_regions, steps=5, strength=0.2)

# Pass 2: Small tiles with strong schedule
small_tiles = denoise(small_regions, steps=20, strength=0.8)

# Blend results
final = blend(large_tiles, small_tiles)
```

**Advantages**:
- ✅ True independent denoising per region
- ✅ Full control over schedules

**Disadvantages**:
- ❌ Requires multiple full sampling passes (very slow)
- ❌ Blending between different denoising stages is hard
- ❌ Can't use within ComfyUI's wrapper API
- ❌ Would need custom sampler implementation

**Status**: Not implemented (too complex, too slow)

### 6.3 Adaptive Step Size (Per-Tile Sigma Modulation)

Modify the **effective sigma** for each tile:

```python
# Instead of: t_tile = repeat(t_in, num_tiles)
# Use:       t_tile = [modulate_sigma(t_in, bbox.denoise) for bbox in bboxes]
```

**Problem**: The model's timestep embedding is **fixed** for the batch. You can't feed different timesteps to different tiles in the same batch without breaking the model's internal conditioning.

**Status**: Not feasible with current architecture

---

## 7. Recommendations

### 7.1 Current Implementation Is Good Enough

The existing scaling approach (lines 1419-1465) is:
- ✅ Mathematically sound (approximates weaker steps)
- ✅ Practical (works within API constraints)
- ✅ Effective (achieves desired preservation/regeneration)
- ✅ Fast (no extra inference)

**Recommendation**: **Keep current implementation** and focus on parameter tuning.

### 7.2 Parameter Tuning Guidelines

**For img2img upscaling** (preserve structure, enhance details):
```
min_denoise: 0.1-0.3  (large tiles barely touched)
max_denoise: 0.7-0.9  (small tiles heavily processed)
```

**For txt2img** (uniform generation):
```
min_denoise: 0.7-0.8  (same value)
max_denoise: 0.7-0.8  (same value)
```

**For subtle enhancement**:
```
min_denoise: 0.3-0.5
max_denoise: 0.5-0.7  (narrow range)
```

### 7.3 Improve User Communication

**Add to node description**:
> "Variable denoise modulates prediction strength per tile (not true per-tile scheduling). Large tiles get weaker denoising steps to preserve content, small tiles get stronger steps to regenerate details."

**Add warnings**:
- When `min_denoise == max_denoise`: Inform user this creates uniform denoising
- When sigmas not available: Clear error message (not just console warning)

### 7.4 Consider Exposing Scaling Parameters

Currently hardcoded:
```python
start_scale = 0.70 + (tile_denoise * 0.25)  # Range: 0.70-0.95
ramp_curve = 1.0 + tile_denoise             # Range: 1.2-1.8
```

**Could make tunable**:
```python
INPUT_TYPES:
    "min_scale": (0.5, 1.0, default=0.70)
    "ramp_strength": (0.5, 2.0, default=1.0)
```

This gives advanced users more control over the approximation.

---

## 8. Potential Future Improvements

### 8.1 Hybrid Skip + Scaling

Combine skip feature with variable denoise:
- Very large tiles (>70% of max): **skip** entirely
- Medium tiles: **scale** based on size
- Small tiles: **full strength**

This could be faster and more accurate than pure scaling.

### 8.2 Sampler-Aware Scaling

Different samplers have different sensitivities to output scaling:
- **Euler**: Scaling works well (linear updates)
- **DPM++**: Might need adjustment (multi-step)
- **Ancestral samplers**: Scaling might interfere with noise injection

**Recommendation**: Test with various samplers and document compatibility.

### 8.3 Model Type Detection

Currently assumes scaling works the same for all models:
```python
tile_out = tile_out * scale_factor  # Works for both noise and velocity
```

This happens to be correct, but could explicitly detect model type:
```python
model_sampling = model.model.model_sampling
is_flux = 'Flux' in str(type(model_sampling))
# Could adjust scaling formula if needed
```

---

## 9. Code Locations Reference

### Integration Points
- **Model wrapper setup**: `tiled_diffusion.py:1594`
- **Wrapper call**: `tiled_diffusion.py:747, 915, 1139`
- **Sigma capture**: `utils.py:25-28`

### Variable Denoise Implementation
- **Sigma loading**: `tiled_diffusion.py:1150-1161`
- **Scaling logic**: `tiled_diffusion.py:1419-1465`
- **Denoise assignment**: `tiled_vae.py:296-308`

### Skip Feature (Alternative)
- **Skip logic**: `tiled_diffusion.py:1274-1343`
- **Velocity fix**: Commit `ee6ac60`

---

## 10. Conclusion

**The Problem**: Users reported variable denoise "not working" because they expected different tiles to be at different stages of denoising (true per-tile schedules).

**The Reality**: ComfyUI's architecture uses **one global timestep** per step. True per-tile scheduling is impossible without major core changes.

**The Solution**: The current implementation **scales model outputs** to approximate weaker denoising for certain tiles. This is:
- The **only feasible approach** within ComfyUI's wrapper API
- **Mathematically sound** (approximates reduced step size)
- **Practically effective** (achieves desired preservation/enhancement)
- **Already implemented correctly**

**Key Insight**: This is **not a bug** - it's an **architectural limitation** of the wrapper API. The scaling approach is the correct solution given the constraints.

**If the user reports it's "not working"**, the likely issues are:
1. **Sigmas not loaded** (check console for warnings)
2. **Parameters too subtle** (try wider range like 0.1-0.9)
3. **Wrong expectations** (expecting true per-tile scheduling)
4. **Visualizer not showing** (denoise values not reaching diffusion)

**Next Steps**: Review parameter values and ensure sigmas are being loaded correctly. The architecture and implementation are sound.

---

**Investigation Complete**
**Status**: Architecture fully understood
**Recommendation**: Current implementation is optimal given constraints
