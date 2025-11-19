# FLUX img2img Velocity Prediction Investigation

**Date**: 2025-11-19
**Status**: 🔍 **CRITICAL BUG IDENTIFIED**

---

## Executive Summary

**The Problem**: Users report that with `min_denoise=0` (intended for complete preservation), large tiles are NOT preserved but instead look "underdeveloped", "super faint", or "too much noise".

**Root Cause**: The variable denoise feature **scales the model's velocity prediction**, but in img2img workflows, **scaling velocity to 0 does NOT preserve the original** — it **freezes the tile in its noisy state**.

**Why This Happens**: The recent commit (98e3cbf) changed the formula from `start_scale = 0.70 + tile_denoise * 0.25` to `start_scale = tile_denoise`, allowing scale to reach 0.0. However, **zero velocity in img2img means "stay noisy", not "preserve original"**.

---

## 1. What Does the Latent Contain at Each Step in img2img?

### Initial State (Before Sampling)

In ComfyUI img2img workflows:

```
1. Input image → VAE encode → clean_latent (X_1)
2. Add noise based on denoise parameter:
   - If denoise = 1.0: Start at pure noise (t=0, X_0 = noise)
   - If denoise = 0.5: Start at 50% noisy (t=0.5, X_0.5 = 0.5*clean + 0.5*noise)
   - If denoise = 0.0: Start at clean (t=1.0, X_1 = clean_latent)
```

**Key Point**: The `latent_image` captured in `utils.py` is the **CLEAN VAE-encoded latent** (X_1), NOT the noisy initial state.

**Location**: `/home/user/comfyui-quadtree-tile/utils.py:33-35`
```python
latent_image = kwargs.get('latent_image') if 'latent_image' in kwargs else (args[6] if len(args) > 6 else None)
if latent_image is not None:
    store.latent_image = latent_image  # ← This is the CLEAN latent
```

### During Denoising

At each sampling step:

```
x_in = current noisy state at timestep t (X_t)
     = Mixture of clean_latent and noise
     = Getting progressively cleaner as denoising proceeds
```

**What `x_in` contains**:
- **Early steps** (high noise): Mostly noise with some structure
- **Middle steps**: Mix of noise and clean content
- **Late steps** (low noise): Mostly clean with fine noise
- **Final step**: Clean image (ideally converges to original clean_latent)

---

## 2. What Does FLUX Velocity Prediction Mean in img2img?

### Rectified Flow Basics

FLUX uses Rectified Flow with linear interpolation:

```
X_t = t * X_1 + (1-t) * X_0

Where:
- X_0 = pure noise (Gaussian)
- X_1 = clean image/latent (target)
- t ∈ [0, 1] is the time parameter
- X_t = noisy state at time t
```

### Velocity Definition

The velocity is the time derivative:

```
v = dX_t/dt = X_1 - X_0

Intuitively:
- v points FROM noise (X_0) TO clean image (X_1)
- v is constant throughout the trajectory (rectified = straight line)
- v represents the "direction of denoising"
```

### Model Prediction

The model is trained to predict this velocity:

```
v_θ(X_t, t) ≈ X_1 - X_0

In practice, for img2img:
- Model sees: noisy latent X_t at time t
- Model predicts: velocity pointing toward clean target
- Prediction: v ≈ E[X_1 - X_0 | X_t]
```

### Sampler Update Formula

```
x_next = x_current + v * dt

Where:
- x_current = current noisy state (X_t)
- v = velocity prediction from model
- dt = step size (sigma_current - sigma_next)
- x_next = next state (X_{t+dt})
```

**Critical Understanding**: The velocity v is **additive**. The sampler **adds** it to move toward the target.

---

## 3. How to Achieve "No Changes" to Original in img2img?

### What Users Expect

When `min_denoise=0`, users expect:
- Large tiles should be **completely preserved**
- Result should be **identical to the original clean latent**
- **NO changes, NO noise, NO modifications**

### What Actually Happens with velocity=0

**Current Implementation** (line 1468 in tiled_diffusion.py):
```python
scale_factor = tile_denoise  # 0.0 for min_denoise=0
tile_out = tile_out * scale_factor  # velocity * 0 = 0
```

**Sampler receives**:
```
v_actual = 0
x_next = x_in + 0 * dt = x_in  ← STAYS NOISY!
```

**Result**: Tile **freezes in its current noisy state**, does NOT return to clean original.

### Why This Looks "Underdeveloped"

```
Step 1:  x_in = 80% noise + 20% clean
         v = 0
         x_next = 80% noise + 20% clean  ← stays noisy

Step 2:  x_in = 80% noise + 20% clean  (unchanged)
         v = 0
         x_next = 80% noise + 20% clean  ← still noisy

...

Final:   x_final = 80% noise + 20% clean  ← looks "super faint" / "underdeveloped"
```

The tile never denoises because velocity=0 means "don't move".

### What Velocity Should We Return to Preserve Original?

**Correct Formula for Preservation**:
```python
v_preserve = original_clean_latent - x_in
```

**Why This Works**:
```
x_next = x_in + (original - x_in) * dt
       = x_in + original*dt - x_in*dt
       = x_in*(1-dt) + original*dt
       → Interpolates toward original
       → After enough steps, converges to original ✓
```

**This is EXACTLY what the skip feature does** (line 1320 in tiled_diffusion.py):
```python
model_prediction = original_tile - x_in_tile  # ✓ CORRECT
```

---

## 4. The Original Latent - Where Is It?

### Storage Location

**File**: `/home/user/comfyui-quadtree-tile/utils.py:33-35`

```python
# Captured during sampler initialization
latent_image = kwargs.get('latent_image') if 'latent_image' in kwargs else (args[6] if len(args) > 6 else None)
if latent_image is not None:
    store.latent_image = latent_image  # ← Stored here
```

### Retrieval Location

**File**: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:1169-1180`

```python
# Loaded once per generation in __call__ method
if not hasattr(self, 'original_latent'):
    try:
        from .utils import store
        if hasattr(store, 'latent_image'):
            self.original_latent = store.latent_image  # ← Retrieved here
            print(f'[Quadtree Skip]: Loaded original latent for img2img, shape={self.original_latent.shape}')
        else:
            self.original_latent = None  # txt2img has no original latent
    except Exception as e:
        self.original_latent = None
```

### What Is It?

**Answer**: It is the **CLEAN VAE-encoded latent**, NOT the noisy latent.

**Evidence**:
1. Captured as `latent_image` parameter from KSampler (the input image after VAE encoding)
2. Used in skip feature as the target: `velocity = original - x_in`
3. Bug analysis document (SKIP_TILE_VELOCITY_BUG.md) confirms it's the clean target

**What it represents**:
- **img2img**: Clean VAE-encoded version of input image (X_1 in rectified flow)
- **txt2img**: Does not exist (None)

### Can We Use It for Variable Denoise?

**Current Status**:
- ✅ **Skip feature** uses it correctly: `velocity = original - x_in`
- ❌ **Variable denoise feature** does NOT use it at all

**Location of variable denoise code**: Lines 1421-1468 (MixtureOfDiffusers), similar in MultiDiffusion and SpotDiffusion

**Problem**: Variable denoise only scales the model's prediction, it does NOT have access to or use `self.original_latent`.

---

## 5. The Problem We're Seeing - Detailed Analysis

### User Report

With `min_denoise=0` (should preserve original completely):
- ✅ **Small tiles**: Look good (regenerated with high denoise)
- ❌ **Large tiles**: NOT preserved, look "underdeveloped" / "super faint" / "too much noise"
- ❌ **Large tiles ARE changing** when they shouldn't

### Root Cause Timeline

#### Commit 98e3cbf (2025-11-18): "Expand variable denoise range to full 0.0-1.0 scale"

**Change**:
```python
# OLD:
start_scale = 0.70 + (tile_denoise * 0.25)  # Range: 0.70-0.95

# NEW:
start_scale = tile_denoise  # Range: 0.0-1.0
```

**Intent** (from commit message):
> "min_denoise=0: Large tiles should get ZERO changes (complete preservation)"
> "Large tiles: Completely unchanged (scale starts at 0.0)"

**Reality**: This is **fundamentally wrong** for img2img workflows!

#### Why the Change Was Wrong

The commit author **misunderstood** what `velocity * 0` means in img2img:

**Misconception**:
```
velocity * 0 = 0 → no changes → preservation ❌ WRONG
```

**Reality**:
```
velocity * 0 = 0 → stay at current noisy state → looks underdeveloped ✓ CORRECT
```

**Why the old 0.70 minimum worked better**:
```
OLD: scale_factor = 0.70
     velocity = model_prediction * 0.70
     x_next = x_in + (0.70 * velocity) * dt
     → Still moves 70% toward target
     → Eventually denoises (slowly)
     → Preserves structure, removes most noise ✓

NEW: scale_factor = 0.0
     velocity = model_prediction * 0.0 = 0
     x_next = x_in + 0 * dt = x_in
     → NEVER moves
     → Stays permanently noisy
     → Looks underdeveloped ❌
```

### Mathematical Proof

**For FLUX sampler**: `x_next = x_in + v * dt`

**Scenario 1: velocity = model_prediction (normal)**
```
v = model(x_in) ≈ clean_target - x_in
x_next = x_in + (clean_target - x_in) * dt
       = x_in * (1-dt) + clean_target * dt
       → Interpolates toward clean_target
       → Over multiple steps: converges to clean_target ✓
```

**Scenario 2: velocity = 0 (current min_denoise=0)**
```
v = 0
x_next = x_in + 0 * dt = x_in
       → Stays at current noisy state
       → NEVER denoises
       → Remains noisy at end ❌
```

**Scenario 3: velocity = original - x_in (what we need)**
```
v = original_clean - x_in
x_next = x_in + (original_clean - x_in) * dt
       = x_in * (1-dt) + original_clean * dt
       → Interpolates toward original_clean
       → Over multiple steps: converges to original_clean ✓
```

**Scenario 4: velocity = model_prediction * 0.70 (old formula)**
```
v = 0.70 * model(x_in) ≈ 0.70 * (clean_target - x_in)
x_next = x_in + 0.70 * (clean_target - x_in) * dt
       = x_in + 0.70*dt*clean_target - 0.70*dt*x_in
       = x_in * (1 - 0.70*dt) + clean_target * (0.70*dt)
       → Interpolates toward clean_target (slower)
       → Over multiple steps: still converges (70% speed)
       → Preserves structure, removes noise (gradually) ✓
```

---

## 6. Comparison: Skip Feature vs Variable Denoise

### Skip Feature (Lines 1298-1349)

**Access to original**: ✅ YES
```python
if hasattr(self, 'original_latent') and self.original_latent is not None:
    original_tile = extract_tile_with_padding(self.original_latent, bbox, self.w, self.h)
    x_in_tile = extract_tile_with_padding(x_in, bbox, self.w, self.h)

    # Compute correct preservation velocity
    model_prediction = original_tile - x_in_tile  # ✓ Points toward original
```

**Result**: ✅ **Perfect preservation** - tiles converge to original clean latent

**How it works**:
1. Extracts tile from CLEAN original latent
2. Extracts tile from current NOISY state
3. Computes velocity: `original - noisy`
4. Returns this as model prediction
5. Sampler applies: `x_next = x_in + (original - x_in) * dt`
6. Result: Interpolates toward original → perfect preservation

### Variable Denoise Feature (Lines 1421-1468)

**Access to original**: ❌ NO
```python
# Get model prediction (knows nothing about original)
tile_out = model_function(x_in, t_in, **c_in)

# Scale the prediction
scale_factor = tile_denoise  # 0.0 for min_denoise=0
tile_out = tile_out * scale_factor  # ← Just scales, doesn't use original
```

**Result**: ❌ **Stays noisy** - when scale=0, velocity=0, stays at current state

**How it works**:
1. Model predicts velocity toward its learned target
2. We scale this velocity by tile_denoise
3. If tile_denoise=0: velocity becomes 0
4. Sampler applies: `x_next = x_in + 0 * dt = x_in`
5. Result: Tile never moves → stays noisy → looks underdeveloped

### Why Variable Denoise Can't Achieve Perfect Preservation

**Fundamental limitation**: The model's prediction is trained to denoise toward a **generated target** (based on the prompt), not toward the **original input image**.

```
Model prediction: v_model ≈ generated_target - x_in
Preservation needs: v_preserve = original_clean - x_in

These are DIFFERENT targets!
```

**Example**:
- Original image: A blue car
- Prompt: "A red sports car"
- Model prediction: v → toward RED car (based on prompt)
- Preservation needs: v → toward BLUE car (original)

**Scaling doesn't fix this**:
```
v_model * 0.7 = 0.7 * (red_car - x_in) → Still points toward red car (just slower)
v_model * 0.0 = 0 → Doesn't move at all (stays noisy)

Neither preserves the original blue car!
```

**Only solution**: Need access to `original_clean` to compute `v = original_clean - x_in`.

---

## 7. Solution Options

### Option A: Revert to 0.70 Minimum (Quick Fix)

**Change**:
```python
# Revert line 1451 (and similar in MultiDiffusion, SpotDiffusion)
start_scale = max(0.70, tile_denoise)  # Never go below 0.70
```

**Pros**:
- ✅ Quick fix (one line change in 3 locations)
- ✅ Prevents "stays noisy" problem
- ✅ Large tiles still get gentle denoising (70% speed)
- ✅ Backward compatible

**Cons**:
- ❌ Not true preservation (tiles still change slightly)
- ❌ Can't achieve "zero changes" that users expect
- ❌ Doesn't match user expectations for min_denoise=0

### Option B: Use Original Latent for Low Denoise (Correct Fix)

**Change**: Modify variable denoise to use original latent when available and scale is low.

**Pseudocode**:
```python
if tile_denoise < threshold and hasattr(self, 'original_latent') and self.original_latent is not None:
    # Use preservation velocity (like skip feature)
    original_tile = extract_tile_with_padding(self.original_latent, bbox, self.w, self.h)
    preservation_velocity = original_tile - x_in_tile

    # Blend between preservation and model prediction
    tile_out = preservation_velocity * (1 - scale_factor) + tile_out * scale_factor
else:
    # Use scaled model prediction (current behavior)
    tile_out = tile_out * scale_factor
```

**Pros**:
- ✅ Achieves true preservation when min_denoise=0
- ✅ Smooth blend between preservation and generation
- ✅ Works correctly for img2img
- ✅ Matches user expectations

**Cons**:
- ❌ More complex implementation
- ❌ Requires access to original_latent (already available in img2img)
- ❌ Needs careful blending to avoid artifacts

### Option C: Document Limitation and Recommend Skip Feature

**Change**: Update documentation to explain:
- Variable denoise with min_denoise=0 does NOT preserve original
- For preservation, use `skip_diffusion_below` parameter instead
- Variable denoise is for "gentle vs aggressive denoising", not "preserve vs regenerate"

**Pros**:
- ✅ No code changes needed
- ✅ Skip feature already works correctly
- ✅ Clear separation of use cases

**Cons**:
- ❌ Confusing to users (why doesn't min_denoise=0 preserve?)
- ❌ Skip feature is all-or-nothing (no smooth variation)
- ❌ Doesn't fix the "underdeveloped tiles" problem

### Option D: Hybrid Approach (Recommended)

**Combine Options A and C**:

1. **Revert to 0.70 minimum** for variable denoise
   - Prevents "stays noisy" problem
   - Provides reasonable "gentle denoising" behavior

2. **Update documentation** to clarify:
   - `min_denoise=0` means "gentle denoising" (70% speed), NOT "no changes"
   - For true preservation, use `skip_diffusion_below` parameter
   - Variable denoise is for controlling denoising strength, not preservation

3. **Consider adding warning** when min_denoise < 0.5:
   ```python
   if min_denoise < 0.5:
       print("[WARNING] Very low min_denoise may result in incomplete denoising")
       print("[INFO] For preservation, use skip_diffusion_below parameter instead")
   ```

**Pros**:
- ✅ Quick fix prevents immediate problem
- ✅ Clear user guidance
- ✅ Maintains separation between features
- ✅ No complex blending code needed

**Cons**:
- ❌ Still doesn't achieve true "zero changes" preservation
- ❌ Requires documentation updates

---

## 8. Recommendations

### Immediate Action (Fix the Bug)

1. **Revert commit 98e3cbf** to restore 0.70 minimum:
   ```python
   start_scale = 0.70 + (tile_denoise * 0.25)  # Range: 0.70-0.95
   ```
   - **OR** use: `start_scale = max(0.70, tile_denoise)`

2. **Update commit message and comments** to clarify:
   - "min_denoise=0 means gentle denoising (70% strength), not preservation"
   - "For preservation in img2img, use skip_diffusion_below parameter"

### Documentation Updates

1. **Add to README** or **User Guide**:
   ```markdown
   ### Variable Denoise vs Skip Feature

   **Variable Denoise** (min_denoise / max_denoise):
   - Controls denoising STRENGTH (70%-100%)
   - Large tiles: gentle denoising
   - Small tiles: aggressive denoising
   - All tiles eventually converge and denoise

   **Skip Feature** (skip_diffusion_below):
   - Controls whether to PROCESS tiles at all
   - Large tiles: completely preserved (img2img)
   - Small tiles: fully processed
   - Perfect preservation for large regions
   ```

2. **Add warning in code** when very low min_denoise is used

### Long-term Considerations

1. **Consider implementing Option B** (use original latent for low denoise):
   - Would provide "true" variable denoise from 0-100%
   - More aligned with user expectations
   - Requires careful implementation and testing

2. **Consider deprecating overlap between features**:
   - Variable denoise: Focus on strength modulation (0.5-1.0 range)
   - Skip feature: Focus on preservation (binary on/off)
   - Clear separation of concerns

3. **Add automated tests** for this scenario:
   - Test that min_denoise=0 doesn't leave tiles noisy
   - Test preservation quality in img2img workflows
   - Verify that tiles actually denoise over time

---

## 9. Technical Details for Reference

### FLUX Rectified Flow Formula Summary

```
Forward process (adding noise):
  X_t = t * X_1 + (1-t) * X_0
  where t: 0 (pure noise) → 1 (clean data)

Velocity field:
  v(X_t, t) = dX_t/dt = X_1 - X_0

Sampler update (Euler):
  X_{t+dt} = X_t + v(X_t, t) * dt

For preservation:
  v = X_1 - X_t  (where X_1 is the original clean target)
  Result: X_{t+dt} = X_t + (X_1 - X_t) * dt → converges to X_1
```

### Variable Denoise vs Skip Feature Matrix

| Feature | Access to Original? | Preservation Quality | Use Case |
|---------|-------------------|---------------------|----------|
| **Skip Feature** | ✅ YES | ✅ Perfect (img2img) | Binary preserve/regenerate |
| **Variable Denoise (current)** | ❌ NO | ❌ Poor (stays noisy at scale=0) | Strength modulation |
| **Variable Denoise (with fix)** | ⚠️ Partial | ✅ Good (at low scales) | Smooth preserve-to-regenerate |

### File Locations Reference

- **Original latent storage**: `/home/user/comfyui-quadtree-tile/utils.py:33-35`
- **Original latent retrieval**: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:1169-1180`
- **Skip feature implementation**: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:1298-1349`
- **Variable denoise (MixtureOfDiffusers)**: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:1421-1468`
- **Variable denoise (MultiDiffusion)**: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:835-871`
- **Variable denoise (SpotDiffusion)**: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:1037-1072`
- **Problematic commit**: `98e3cbf` (2025-11-18)

---

## 10. Conclusion

**The bug is clear**: Allowing `scale_factor = 0` in img2img workflows causes tiles to freeze in their noisy state, producing "underdeveloped" / "super faint" results.

**The fix is straightforward**: Restore the 0.70 minimum to ensure tiles always denoise (just more gently for large tiles).

**The lesson learned**: In rectified flow (velocity-based) img2img:
- `velocity = 0` means "stay at current state" (which is noisy)
- `velocity = original - current` means "move toward original" (preservation)
- Scaling model prediction can't achieve true preservation without access to original

**The path forward**:
1. Fix the immediate bug (restore 0.70 minimum)
2. Update documentation (clarify feature differences)
3. Consider hybrid approach for true 0-100% range (future enhancement)

---

**Status**: 🔍 **Investigation Complete - Ready for Fix**
