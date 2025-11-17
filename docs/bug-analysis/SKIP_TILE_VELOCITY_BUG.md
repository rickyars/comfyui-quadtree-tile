# Skip Tile Velocity Prediction Bug - Complete Analysis

## Executive Summary

**The Bug**: Skipped tiles in img2img workflows showed inverted colors (film negative effect) when using FLUX and other velocity-based models.

**Root Cause**: The skip implementation contained a critical sign error. FLUX uses velocity prediction (Rectified Flow) where velocity should point FROM current state TO target. The code incorrectly computed `velocity = x_in - original` when it should be `velocity = original - x_in`.

**Impact**:
- **FLUX users**: Skipped tiles appeared as film negative (inverted colors)
- **SD1.5/SDXL users**: Worked correctly (these use noise prediction with opposite sign)
- Made the skip feature completely unusable for img2img workflows with FLUX

**Resolution**: Fixed in commit ee6ac60 by correcting the velocity formula to `original - x_in`.

---

## Timeline of Discovery and Resolution

### Phase 1: Initial Discovery - "Pure Noise" Bug

**Initial Symptom**: Skipped tiles appeared as pure noise in final output.

**Initial Hypothesis**: The skip implementation was copying noisy latent input (`x_in`) into the noise prediction buffer (`x_buffer`), causing the sampler to receive garbage data.

**Analysis**: Believed that `x_buffer` should contain noise predictions, and copying image data into it was mixing incompatible data types.

**Status**: This analysis was partially incorrect - the fundamental issue wasn't about noise vs image data, but about velocity prediction types.

### Phase 2: Understanding Model Prediction Types

**Discovery**: Different models return different prediction types:
- **Noise Prediction (SD1.5, SDXL)**: Model predicts noise ε, sampler subtracts it
- **Velocity Prediction (FLUX)**: Model predicts velocity v, sampler adds it
- **Denoised Prediction**: Model predicts clean image x₀ directly

**Key Insight**: `x_buffer` contains whatever `model_function` returns - not always noise!

### Phase 3: Film Negative Effect Discovery

**Updated Symptom**: With img2img workflows, skipped tiles showed **film negative effect** - pixels in right place but colors inverted.

**Root Cause Identified**: Sign error in velocity calculation:
- Current (wrong): `velocity = x_in - original`
- Correct: `velocity = original - x_in`

**Mathematical Proof**:
```
FLUX sampler: x_next = x_in + velocity * dt
Goal: x_next = original

Wrong formula (current):
  velocity = x_in - original
  x_next = x_in + (x_in - original) * dt
        = x_in + x_in*dt - original*dt
        = (1 + dt)*x_in - dt*original  ❌ Moves AWAY from original

Correct formula:
  velocity = original - x_in
  x_next = x_in + (original - x_in) * dt
        = x_in + original*dt - x_in*dt
        = x_in*(1-dt) + original*dt  ✓ Interpolates toward original
```

### Phase 4: Resolution

**Fix Applied**: Changed sign in velocity calculation (commit ee6ac60)
**Result**: Skipped tiles now correctly preserve original content in img2img workflows

---

## Technical Deep Dive

### 1. Understanding What `x_buffer` Contains

#### Initialization
```python
# Line 337: tiled_diffusion.py
self.x_buffer = torch.zeros_like(x_in, device=x_in.device, dtype=x_in.dtype)
```
**`x_buffer` is initialized to ZEROS**

#### Accumulation
```python
# Lines 1402, 1411: Accumulate weighted model outputs
self.x_buffer[:, :, y_start:y_end, x_start:x_end] += valid_tile * tile_weights
```
**`x_buffer` accumulates weighted MODEL OUTPUTS**

#### Return to Sampler
```python
# Lines 1421-1426: Normalize and return
x_out = torch.where(mask, x_out / torch.clamp(self.weights, min=epsilon), x_out)
return x_out
```

**Critical Understanding**: `x_buffer` contains whatever `model_function` returns:
- For noise prediction models (SD1.5, SDXL): contains predicted noise ε
- For velocity prediction models (FLUX): contains predicted velocity v
- For denoised prediction models: contains predicted x₀

### 2. Model Prediction Types Explained

#### Noise Prediction (SD1.5, SDXL)
- **Training objective**: Predict noise ε
- **Model output**: `ε = model(x_t, t, c)`
- **Sampler formula**: `x_{t-1} = x_t - σ_t * ε`
- **Direction**: Subtractive (remove noise)

#### Velocity Prediction (FLUX, Rectified Flow)
- **Training objective**: Predict velocity v
- **Model output**: `v = model(x_t, t, c)`
- **Sampler formula**: `x_{t+1} = x_t + Δt * v`
- **Direction**: Additive (flow toward target)

From FLUX Euler sampler code:
```python
pred = model_forward(...)  # Returns velocity prediction
img = img + (t_prev - t_curr) * pred
```

### 3. The Sign Error Explained

For **Noise Prediction Models (SD1.5, SDXL)** - ✅ CORRECT:

```
Sampler formula: x_{t-1} = x_t - σ_t * ε
Goal: Preserve original content
  original = x_t - σ_t * ε
Solving for ε:
  ε = (x_t - original) / σ_t

Current code (ignoring σ_t scaling):
  noise_prediction = x_in - original  ✅ CORRECT sign

Result:
  x_next = x_in - (x_in - original) = original  ✅ Works!
```

For **Velocity Prediction Models (FLUX)** - ❌ WRONG SIGN:

```
Sampler formula: x_{t+1} = x_t + Δt * v
Goal: Preserve original content
  original = x_t + Δt * v
Solving for v:
  v = (original - x_t) / Δt

Current code (ignoring Δt scaling):
  velocity = x_in - original  ❌ WRONG (opposite sign!)

Result:
  x_next = x_in + (x_in - original) * Δt
        = (1 + Δt) * x_in - Δt * original
        ≠ original  ❌ Creates film negative!
```

### 4. Why Film Negative Specifically?

The film negative effect occurs because wrong-sign velocity creates proportional color inversion:

1. **For dark pixels** (low values like 0.2):
   - Original = 0.2, Current noisy = 0.5
   - Wrong velocity = 0.5 - 0.2 = 0.3 (points away from dark)
   - Result: Gets brighter → becomes light in negative

2. **For bright pixels** (high values like 0.8):
   - Original = 0.8, Current noisy = 0.5
   - Wrong velocity = 0.5 - 0.8 = -0.3 (points away from bright)
   - Result: Gets darker → becomes dark in negative

3. **Spatial structure preserved**:
   - Tiles are placed correctly
   - Only COLOR VALUES inverted
   - Classic sign error signature in diffusion

### 5. Understanding Rectified Flow

FLUX uses Rectified Flow which leverages velocity-based formulation:

**Flow Matching Formula**:
```
dx/dt = v(x_t, t)
```

**Discrete update**:
```
x_{t+1} = x_t + Δt * v(x_t, t)
```

**For Skipped Tiles**:
To make `x_{t+1} = original`:
```
original = x_t + Δt * v
v = (original - x_t) / Δt
```

**The sign is CRITICAL**:
- Correct: `v ∝ (original - x_in)` → Points FROM current TO target
- Wrong: `v ∝ (x_in - original)` → Points AWAY from target

---

## The Fix

### Code Changes Required

**File**: `tiled_diffusion.py`

**Change 1**: Detect model type in `apply()` method (around line 1466)
```python
def apply(self, model: ModelPatcher, method, quadtree, ...):
    # ... existing code ...

    # Detect model prediction type
    model_sampling = model.model.model_sampling
    is_flux = 'Flux' in str(type(model_sampling))

    self.impl.is_velocity_model = is_flux
    print(f'[Quadtree Diffusion]: Model type: {"FLUX (velocity)" if is_flux else "SD (noise)"}')
```

**Change 2**: Fix skip formula with correct sign (around line 1221)
```python
# Compute prediction that will restore original
if getattr(self, 'is_velocity_model', False):
    # FLUX: velocity = original - x_in (points toward target)
    prediction = original_tile - x_in_tile  ✅ CORRECT for velocity
else:
    # SD1.5/SDXL: noise = x_in - original
    prediction = x_in_tile - original_tile  ✅ CORRECT for noise
```

**Change 3**: Update comments and variable names
```python
# OLD (misleading):
# Compute noise prediction that restores original content
# Formula: predicted_noise = x_in - original
noise_prediction = x_in_tile - original_tile

# NEW (accurate):
# Compute velocity/noise prediction that moves toward original content
# For velocity models (FLUX): predicted_velocity = original - x_in
# For noise models (SD): predicted_noise = x_in - original
if is_velocity_model:
    velocity_prediction = original_tile - x_in_tile
else:
    noise_prediction = x_in_tile - original_tile
```

---

## Testing Strategy

### Test Case 1: FLUX with Skip (Primary Fix Target)
- **Model**: FLUX.1-dev
- **Workflow**: img2img with denoise=0.55
- **Skip threshold**: 256px
- **Expected**: Skipped tiles preserve original colors ✅
- **Before fix**: Film negative effect
- **After fix**: Correct colors

### Test Case 2: SDXL with Skip (Regression Test)
- **Model**: SDXL
- **Workflow**: img2img with denoise=0.55
- **Skip threshold**: 256px
- **Expected**: No regression, still works ✅

### Test Case 3: txt2img (Sanity Check)
- **Model**: FLUX or SDXL
- **Workflow**: txt2img
- **Skip threshold**: 256px
- **Expected**: Minimal impact

---

## Evolution of Understanding

### Initial Misconception (SKIP_TILE_BUG_ANALYSIS.md)
- Believed `x_buffer` should only contain noise predictions
- Thought copying `x_in` was mixing "image data" with "noise data"
- Proposed using zero noise predictions
- **Problem**: This understanding was based on noise-prediction models only

### Intermediate Understanding (INVERTED_COLORS_ANALYSIS.md)
- Discovered different models return different prediction types
- Identified that FLUX uses velocity, not noise
- Recognized the sign error: `x_in - original` vs `original - x_in`
- **Breakthrough**: Understood the mathematical root cause

### Final Understanding (FILM_NEGATIVE_BUG_ANALYSIS.md)
- Confirmed sign error with detailed mathematical proof
- Explained why film negative effect occurs (proportional color inversion)
- Provided clear fix with model type detection
- **Resolution**: Single-line fix with correct sign

---

## Why This Bug Wasn't Caught Earlier

1. **Most testing was on SD1.5/SDXL** where the formula happens to be correct
2. **FLUX is relatively new** (released August 2024)
3. **Skip feature is experimental** and not widely used
4. **Different workflows affected differently**:
   - txt2img: Less obvious (no original content exists)
   - img2img: Very obvious (color inversion visible)

---

## Alternative Solutions Considered

### Option A: Variable Denoise (Recommended Alternative)
Instead of skipping entirely, use the existing `variable_denoise` feature:
- Small tiles get low denoise values (e.g., 0.1)
- They're still processed but with minimal change
- **Works correctly for ALL model types** (no sign error possible)
- **Simpler implementation** (no model-specific logic needed)
- **Already implemented and tested**

### Option B: Zero Velocity/Noise
Return zero predictions for skipped tiles:
- **Problem for txt2img**: Skipped regions stay noisy
- **Problem for img2img**: Needs original clean latent (not available)
- **Complexity**: Requires weight adjustment

### Option C: Store Original Latent
For img2img, store original encoded latent before noise addition:
- **Problem**: Not available in all workflows
- **Problem**: Doesn't work for txt2img
- **Complexity**: Requires sampler cooperation

---

## Long-term Recommendations

### 1. Document Model Prediction Types
Create clear documentation about:
- Which models use which prediction types
- How to detect prediction type programmatically
- Implications for custom diffusion implementations

### 2. Consider Deprecating Skip Feature
The `variable_denoise` feature provides similar benefits with:
- Simpler implementation
- No model-specific logic
- Works for both txt2img and img2img
- Already proven reliable

### 3. Add Model Type Warnings
When using skip feature, detect and warn about model compatibility:
```python
if skip_diffusion_below > 0:
    if is_velocity_model:
        print("[WARNING] Skip feature with velocity models requires correct sign")
```

---

## References

- **FLUX Architecture**: "Rectified Flow: Straight is Fast" (https://rectifiedflow.github.io/)
- **Flow Matching**: "Demystifying Flux Architecture" (arXiv:2507.09595)
- **ComfyUI Model Types**: ModelSamplingFlux vs ModelSamplingDiscrete
- **Velocity vs Noise**: "Three Stable Diffusion Training Losses: x0, epsilon, and v-prediction"
- **Commit History**:
  - `be0ecca`: Original correct implementation (zero noise)
  - `5feb41a`: Introduced bug (wrong sign optimization)
  - `ee6ac60`: Fixed bug (corrected velocity sign)

---

## Conclusion

This bug demonstrates the importance of:
1. **Understanding model fundamentals**: Different diffusion models use different prediction types
2. **Mathematical rigor**: Sign errors have dramatic visual effects
3. **Comprehensive testing**: Test across multiple model architectures
4. **Clear documentation**: Variable naming and comments must reflect reality

The fix was ultimately simple (one sign change), but the journey to understanding it required deep analysis of:
- Diffusion model theory (noise vs velocity prediction)
- Rectified Flow mathematics
- FLUX sampling implementation
- Multi-step diffusion dynamics

**Key Takeaway**: When working with diffusion models, always verify:
- What does the model predict? (ε, v, or x₀)
- What does the sampler expect?
- Are the signs correct for the update formula?

---

**Status**: ✅ RESOLVED (commit ee6ac60)
**Date**: 2025-11-17
**Impact**: Critical bug fix enabling FLUX compatibility for skip feature
