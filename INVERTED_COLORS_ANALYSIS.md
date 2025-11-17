# Inverted Colors (Film Negative) in Skipped Tiles - Root Cause Analysis

## Executive Summary

**CRITICAL BUG**: Skipped tiles show film negative (inverted colors) when using FLUX and other velocity-based models.

**ROOT CAUSE**: The skip implementation assumes noise prediction (epsilon) but FLUX uses velocity prediction. The formula has the **OPPOSITE SIGN** for velocity models.

**IMPACT**:
- FLUX users: Skipped tiles appear as film negative
- SD1.5/SDXL users: Works correctly (these use noise prediction)

---

## 1. Understanding What `x_buffer` Contains

### Initialization (Line 337)
```python
self.x_buffer = torch.zeros_like(x_in, device=x_in.device, dtype=x_in.dtype)
```
**`x_buffer` is initialized to ZEROS**

### Accumulation (Lines 1402, 1411)
```python
# For quadtree with overlap:
self.x_buffer[:, :, y_start:y_end, x_start:x_end] += valid_tile * tile_weights

# For grid:
self.x_buffer[bbox.slicer] += tile_out * w
```
**`x_buffer` accumulates weighted MODEL OUTPUTS**

### Normalization (Lines 1421-1426)
```python
x_out = torch.where(mask, x_out / torch.clamp(self.weights, min=epsilon), x_out)
return x_out
```
**`x_buffer` is normalized by weights and returned to sampler**

### Conclusion
**`x_buffer` contains whatever `model_function` returns!**
- For noise prediction models: contains predicted noise ε
- For velocity prediction models: contains predicted velocity v
- For x₀ prediction models: contains predicted denoised image

---

## 2. What Does `model_function` Return?

The `model_function` is `BaseModel.apply_model` which returns **different things for different model types**:

### Noise Prediction (SD1.5, SDXL)
- Training objective: Predict noise ε
- Model output: `ε = model(x_t, t, c)`
- Sampler formula: `x_{t-1} = x_t - σ_t * ε`

### Velocity Prediction (FLUX, Rectified Flow)
- Training objective: Predict velocity v
- Model output: `v = model(x_t, t, c)`
- Sampler formula: `x_{t+1} = x_t + Δt * v`

### Denoised Prediction (Some models)
- Training objective: Predict clean image x₀
- Model output: `x₀ = model(x_t, t, c)`
- Sampler formula: `x_{t-1} = f(x_t, x₀)`

---

## 3. Tracing a Normal Tile Through the Pipeline

### Step 1: Extract Tile (Line 1268)
```python
tile = extract_tile_with_padding(x_in, bbox, self.w, self.h)
```
- `x_in` = Current noisy latent at timestep t
- `tile` = Region of noisy latent

### Step 2: Model Inference (Line 1311)
```python
x_tile_out = model_function(x_tile, t_tile, **c_tile)
```
- For FLUX: Returns **velocity** `v(x_t, t)`
- For SD1.5/SDXL: Returns **noise** `ε(x_t, t)`

### Step 3: Accumulate to Buffer (Line 1402)
```python
self.x_buffer[:, :, y_start:y_end, x_start:x_end] += valid_tile * tile_weights
```
- Adds weighted model output to buffer

### Step 4: Return to Sampler (Line 1426)
```python
x_out = x_buffer / weights
return x_out
```
- Returns normalized model outputs
- **For FLUX**: Returns velocity field
- **For SD1.5/SDXL**: Returns noise field

---

## 4. Mathematical Analysis: Why We're Seeing Film Negative

### Current Skip Implementation (Lines 1221)
```python
# For skipped tiles:
noise_prediction = x_in_tile - original_tile
```

This formula assumes the sampler will do:
```
x_next = x_in - noise_prediction
```

### For Noise Prediction Models (SD1.5, SDXL) ✅ CORRECT

**Sampler formula:**
```
x_{t-1} = x_t - σ_t * ε
```

**Goal:** Preserve original content
```
original = x_t - σ_t * ε
```

**Solving for ε:**
```
ε = (x_t - original) / σ_t
```

**Current code (ignoring σ_t scaling):**
```python
noise_prediction = x_in - original  # ✅ CORRECT (same sign)
```

**Result:**
```
x_next = x_in - (x_in - original) = original  ✅ Works correctly
```

### For Velocity Prediction Models (FLUX) ❌ WRONG SIGN

**Sampler formula:**
```
x_{t+1} = x_t + Δt * v
```

**Goal:** Preserve original content
```
original = x_t + Δt * v
```

**Solving for v:**
```
v = (original - x_t) / Δt
```

**Current code (ignoring Δt scaling):**
```python
noise_prediction = x_in - original  # ❌ WRONG (opposite sign!)
```

**What we're actually computing:**
```
noise_prediction = x_in - original = -(original - x_in)
```

**Result:**
```
x_next = x_in + Δt * (x_in - original)
       = x_in + Δt * x_in - Δt * original
       = (1 + Δt) * x_in - Δt * original
```

This is **NOT** equal to `original`! The sign inversion causes the sampler to:
1. Move **away from** the original content instead of **toward** it
2. Create a "film negative" effect where colors are inverted
3. Progressively make the error worse over denoising steps

---

## 5. What FLUX Expects (Flow Matching Details)

### FLUX uses Rectified Flow

From research:
> "FLUX uses Rectified Flow which leverages a velocity-based formulation. The model predicts a velocity vector v(x_t, t) representing the target vector from noise to data."

### Flow Matching Formula
```
dx/dt = v(x_t, t)
```

Discrete update:
```
x_{t+1} = x_t + Δt * v(x_t, t)
```

### For Skipped Tiles
To make `x_{t+1} = original`:
```
original = x_t + Δt * v
v = (original - x_t) / Δt
```

**The sign is CRITICAL:**
- Correct: `v ∝ (original - x_in)` → Points FROM current TO target
- Wrong: `v ∝ (x_in - original)` → Points AWAY from target

---

## 6. Why Film Negative Specifically?

The "film negative" effect occurs because:

1. **Early denoising steps** (high noise):
   - `x_in` is mostly noise
   - Wrong velocity pushes AWAY from original
   - Creates inverted color values

2. **Progressive inversion**:
   - Each step reinforces the error
   - Colors become increasingly inverted
   - Final result is a film negative

3. **Pixel positions correct**:
   - The spatial structure is preserved
   - Only the COLOR VALUES are inverted
   - This is characteristic of sign errors in diffusion

---

## 7. The Correct Formula for Skipped Tiles

### Detection Strategy

First, detect the model's prediction type:

```python
# In apply() method:
model_sampling = model.model.model_sampling
prediction_type = getattr(model_sampling, 'prediction_type', 'epsilon')
# Or check class name: 'ModelSamplingFlux' vs 'ModelSamplingDiscrete'

# Store for later use:
self.impl.model_prediction_type = prediction_type
```

### Fixed Skip Implementation

```python
# For skipped tiles (lines 1221):
if hasattr(self, 'original_latent') and self.original_latent is not None:
    x_in_tile = extract_tile_with_padding(x_in, bbox, self.w, self.h)
    original_tile = extract_tile_with_padding(self.original_latent, bbox, self.w, self.h)

    # Detect prediction type
    if self.model_prediction_type == 'velocity' or 'Flux' in str(type(self.model_sampling)):
        # FLUX / Velocity prediction: v = (original - x_in)
        prediction = original_tile - x_in_tile  # ✅ CORRECT for velocity
    else:
        # SD1.5/SDXL / Noise prediction: ε = (x_in - original)
        prediction = x_in_tile - original_tile  # ✅ CORRECT for noise

    # ... rest of the code
```

---

## 8. Summary of Findings

### What `x_buffer` Contains
- **Not** image data
- **Not** always noise
- **Exactly** whatever `model_function` returns (velocity for FLUX, noise for SD)

### What the Sampler Expects
- **FLUX**: Velocity field `v` where `x_next = x_in + Δt * v`
- **SD1.5/SDXL**: Noise field `ε` where `x_next = x_in - σ * ε`

### Why We're Seeing Film Negative
- Current formula: `prediction = x_in - original`
- FLUX needs: `velocity = original - x_in`
- **OPPOSITE SIGN** → Colors inverted

### The Fix
```python
# Detection:
is_velocity_model = 'Flux' in str(type(model.model.model_sampling))

# Correct formula:
if is_velocity_model:
    prediction = original - x_in  # For FLUX
else:
    prediction = x_in - original  # For SD1.5/SDXL
```

---

## 9. Code Changes Required

### File: `tiled_diffusion.py`

**Change 1:** Detect model type in `apply()` method (around line 1466)
```python
def apply(self, model: ModelPatcher, method, quadtree, ...):
    # ... existing code ...

    # NEW: Detect model prediction type
    model_sampling = model.model.model_sampling
    is_flux = 'Flux' in str(type(model_sampling))

    self.impl.is_velocity_model = is_flux
    print(f'[Quadtree Diffusion]: Model type: {"FLUX (velocity)" if is_flux else "SD (noise)"}')
```

**Change 2:** Fix skip formula (around line 1221)
```python
# Compute prediction that will restore original
if getattr(self, 'is_velocity_model', False):
    # FLUX: velocity = original - x_in
    prediction = original_tile - x_in_tile
else:
    # SD1.5/SDXL: noise = x_in - original
    prediction = x_in_tile - original_tile
```

---

## 10. Testing Strategy

### Test Case 1: FLUX with Skip
- Model: FLUX.1-dev
- Workflow: img2img with denoise=0.55
- Skip threshold: 256px
- Expected: Skipped tiles preserve original colors ✅

### Test Case 2: SDXL with Skip
- Model: SDXL
- Workflow: img2img with denoise=0.55
- Skip threshold: 256px
- Expected: No regression, still works ✅

### Test Case 3: txt2img
- Model: FLUX or SDXL
- Workflow: txt2img
- Skip threshold: 256px
- Expected: Minimal impact (use variable_denoise instead)

---

## 11. Additional Notes

### Why Didn't This Show Up in Testing?

1. **Most testing was on SD1.5/SDXL** where the formula is correct
2. **FLUX is relatively new** (August 2024)
3. **Skip feature is experimental** and not widely used yet

### Alternative Solution: Variable Denoise

Instead of skipping entirely, the existing **variable denoise** feature provides similar benefits:
- Small tiles get low denoise values (e.g., 0.1)
- They're still processed but with minimal change
- **Works correctly for ALL model types**
- Recommended for txt2img workflows

### Long-term Recommendation

Consider deprecating the `skip_diffusion_below` feature in favor of the more robust `variable_denoise` approach:
- Simpler implementation
- No model-specific logic needed
- Works for both txt2img and img2img
- Already implemented and tested

---

## 12. References

- **FLUX Architecture**: "Rectified Flow: Straight is Fast" (https://rectifiedflow.github.io/)
- **Flow Matching**: "Demystifying Flux Architecture" (arXiv:2507.09595)
- **ComfyUI Model Types**: ModelSamplingFlux vs ModelSamplingDiscrete
- **Velocity vs Noise**: "Three Stable Diffusion Training Losses: x0, epsilon, and v-prediction"

---

**Author**: Senior Python/ML Engineer Analysis
**Date**: 2025-11-17
**Status**: CRITICAL - Requires immediate fix for FLUX compatibility
