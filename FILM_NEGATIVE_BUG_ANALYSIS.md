# Film Negative Bug Analysis: Inverted Colors in Skipped Tiles

## Executive Summary

**The Bug**: Skipped tiles in img2img show INVERTED COLORS (film negative effect)

**Root Cause**: SIGN ERROR - The velocity prediction has the wrong sign because:
1. FLUX uses Rectified Flow and returns VELOCITY predictions (not noise)
2. Velocity should point FROM current state TO target (original)
3. Current code computes: `velocity = x_in - original` (BACKWARDS!)
4. Correct formula: `velocity = original - x_in` (direction toward target)

**Impact**: Any skipped tiles show color inversion, making the feature unusable for img2img workflows.

---

## 1. Understanding FLUX and Rectified Flow

### What does FLUX return?

Unlike traditional diffusion models that predict NOISE (epsilon), FLUX uses **Rectified Flow** which predicts **VELOCITY** (v).

From research and code analysis:
- **Traditional diffusion**: Model predicts noise ε, sampler does `x_next = x - noise`
- **FLUX (Rectified Flow)**: Model predicts velocity v, sampler does `x_next = x + velocity * dt`

### The FLUX Euler Sampler Formula

From XLabs FLUX sampling code:
```python
pred = model_forward(...)  # Returns velocity prediction
img = img + (t_prev - t_curr) * pred
```

Simplified:
```
x_next = x_current + dt * velocity
```

Where:
- `velocity` = model output (direction to move in latent space)
- `dt` = timestep difference
- Motion is ADDITIVE (not subtractive like noise-based diffusion)

---

## 2. The Current Buggy Implementation

### Lines 1220-1221 in tiled_diffusion.py:

```python
# Compute noise prediction that will restore original
noise_prediction = x_in_tile - original_tile
```

**Problems**:
1. **Wrong name**: This is velocity prediction, not noise prediction
2. **Wrong sign**: Velocity should be `original - x_in`, not `x_in - original`

### Why This Causes Film Negative Effect

Let's trace through one sampling step:

**What we have**:
- `x_in_tile` = noisy/corrupted latent (e.g., value = 0.5)
- `original_tile` = clean target latent (e.g., value = 0.8)
- Goal: Move FROM 0.5 TO 0.8

**Current buggy code**:
```python
velocity = x_in_tile - original_tile  # = 0.5 - 0.8 = -0.3
x_next = x_in + velocity * dt         # = 0.5 + (-0.3) * dt = 0.5 - 0.3*dt
# Result: Moves AWAY from target (toward 0.2, not 0.8)
```

**Correct formula**:
```python
velocity = original_tile - x_in_tile  # = 0.8 - 0.5 = 0.3
x_next = x_in + velocity * dt         # = 0.5 + 0.3 * dt = 0.5 + 0.3*dt
# Result: Moves TOWARD target (toward 0.8) ✓
```

**Why "film negative"?**

When you subtract in the wrong direction:
- Dark pixels (low values) get pushed darker (negative direction)
- Bright pixels (high values) get pushed brighter (negative direction)
- The error is proportional to the distance from target
- Result: Color inversion, like a photographic negative!

---

## 3. Model Output Type Verification

### From SKIP_TILE_BUG_ANALYSIS.md:

The analysis states:
> "The noise predictor estimates the noise of the image, the predicted noise is subtracted from the image"
> Sampler: `x_next = x_in - predicted_noise` (simplified)

**This is INCORRECT for FLUX!** This describes traditional epsilon-prediction diffusion, not Rectified Flow.

### Correct Understanding:

From web research and FLUX sampling code:
1. **Rectified Flow formula**: `dx/dt = v(x, t)` where v is velocity
2. **Euler integration**: `x_next = x_current + v * dt`
3. **FLUX model output**: Velocity vector in latent space
4. **Direction**: Points FROM noise TOWARD clean data

---

## 4. What's in x_buffer?

Looking at the code flow:

```python
# Line 337-339: Initialize to zeros
self.x_buffer = torch.zeros_like(x_in)

# Line 1311: Model predicts for processed tiles
x_tile_out = model_function(x_tile, t_tile, **c_tile)

# Line 1245: Accumulate for skipped tiles (img2img)
self.x_buffer[:, :, y_start:y_end, x_start:x_end] += valid_noise * tile_weights

# Line 1418: Return accumulated predictions
return self.x_buffer
```

**What x_buffer contains**:
- For FLUX: **Velocity predictions** (not noise!)
- For processed tiles: Model-predicted velocities
- For skipped tiles (current bug): WRONG-SIGN velocities
- Sampler receives this and does: `x_next = x_in + x_buffer * dt`

---

## 5. The Fix

### Change line 1221 from:
```python
noise_prediction = x_in_tile - original_tile
```

### To:
```python
velocity_prediction = original_tile - x_in_tile
```

### Why this works:

**Rectified Flow Interpretation**:
- Velocity points FROM current state TO target state
- `v = target - current` gives the direction vector
- Sampler integrates: `x_next = x_current + v * dt`
- After integration, `x_next` moves toward `target`

**For img2img skipped tiles**:
- Current state: `x_in_tile` (noisy latent)
- Target state: `original_tile` (clean original)
- Velocity: `original_tile - x_in_tile` (direction toward original)
- Result: Skipped tiles gradually restore to original ✓

---

## 6. Additional Issues to Check

### Variable Naming

The variable is called `noise_prediction` but should be `velocity_prediction`:

```python
# Current (misleading):
noise_prediction = original_tile - x_in_tile

# Better (accurate):
velocity_prediction = original_tile - x_in_tile
```

### Comments Need Updating

Lines 1206-1208 have outdated comments:
```python
# img2img: Compute noise prediction that restores original content
# Formula: predicted_noise = x_in - original
# Sampler does: x_next = x_in - predicted_noise = x_in - (x_in - original) = original ✓
```

Should be:
```python
# img2img: Compute velocity prediction that moves toward original content
# Formula: predicted_velocity = original - x_in
# Sampler does: x_next = x_in + predicted_velocity * dt = x_in + (original - x_in) * dt
# Over sampling steps, x converges to original ✓
```

---

## 7. Why Pixels Are in Right Place But Colors Wrong

The user observed:
> "Pixels are in the right place but colors are reversed"

This confirms the sign error:
- **Spatial position**: Correct (tiles are placed correctly)
- **Color values**: Inverted (velocity has wrong sign)

The bug is purely mathematical (sign error), not architectural (wrong tiles or wrong positions).

---

## 8. Testing the Fix

### Expected Behavior After Fix:

**Before fix** (current):
- Skipped tiles show film negative effect
- Dark areas become bright, bright areas become dark
- Colors are inverted

**After fix**:
- Skipped tiles should preserve original content
- Colors should be correct
- Tiles should blend seamlessly with processed tiles

### Test Cases:

1. **img2img with skip tiles**:
   - Input: Photo with clear colors
   - Set low skip threshold to create skip tiles
   - Expected: Skipped areas preserve original colors

2. **Variable denoise + skip**:
   - Some tiles skipped, some with variable denoise
   - Expected: Smooth transition, no color inversion

3. **txt2img** (sanity check):
   - Should still work (uses zero velocity for skip tiles)
   - No regression

---

## 9. Related Code Locations

### Files to modify:
1. `tiled_diffusion.py` line 1221: Fix the sign
2. `tiled_diffusion.py` lines 1206-1208: Update comments
3. `tiled_diffusion.py` line 1220: Rename variable

### Files to review:
1. `SKIP_TILE_BUG_ANALYSIS.md`: Contains outdated assumptions about noise prediction

---

## 10. Conclusion

The film negative bug is a **classic sign error** caused by:
1. Misunderstanding FLUX's prediction type (velocity, not noise)
2. Using subtraction in the wrong direction
3. Misleading variable names and comments

**The one-line fix**:
```python
# Change this:
noise_prediction = x_in_tile - original_tile

# To this:
velocity_prediction = original_tile - x_in_tile
```

This aligns with Rectified Flow's velocity-based formulation where motion is toward the target, not away from it.
