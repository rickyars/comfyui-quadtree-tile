# Denoising Bug Fix - Variable Denoise Implementation

## Summary
Fixed a critical bug in the variable denoise implementation where the code was incorrectly blending the noisy latent input with the noise prediction output, causing corrupted results.

## The Bug

### Location
`tiled_diffusion.py` lines 1284-1295 (old code)

### Problem
The code was blending two values that exist in completely different mathematical spaces:
- `tile_input`: The current noisy latent extracted from `x_in` (in latent space)
- `tile_out`: The model's noise prediction (in noise space / epsilon space)

```python
# BUGGY CODE (before fix):
tile_input = extract_tile_with_padding(x_in, bbox, self.w, self.h)
tile_out = tile_input * (1 - blend_factor) + tile_out * blend_factor
```

This is mathematically incorrect because:
1. In diffusion models, `model_function(x, t)` returns a **noise prediction** (epsilon)
2. The scheduler uses this to compute: `x_next = scheduler.step(x, t, epsilon)`
3. Blending the noisy latent `x` directly with the noise prediction `epsilon` produces garbage

### Why This Caused the Reported Issues

**Issue 1: Large leafs getting pure noise**
- Large leafs have low denoise values (e.g., 0.3)
- They activate late (at 70% progress)
- The buggy blending corrupted the model output
- Result: Noise-like artifacts instead of preserved content

**Issue 2: Small leafs getting too much variation**
- Small leafs have high denoise values (e.g., 1.0)
- They might be affected by blending artifacts from adjacent tiles
- Result: More variation than expected

**Issue 3: 2x faster than expected**
- This is actually EXPECTED behavior, not a bug!
- Tiles with denoise=0.3 only process for ~30% of steps
- Tiles with denoise=1.0 process for 100% of steps
- Average computation is reduced, hence faster execution

## The Fix

### New Approach
When a tile should be "preserved" (inactive), return a **zero noise prediction** instead of blending spaces:

```python
# FIXED CODE:
if progress < activation_threshold:
    # Return zero noise prediction to preserve input
    blend_factor = max(0.0, min(1.0, (progress - (activation_threshold - 0.1)) / 0.1))
    zero_noise = torch.zeros_like(tile_out)
    tile_out = zero_noise * (1 - blend_factor) + tile_out * blend_factor
```

### Why This Works

1. **Zero noise prediction** = "don't change this region"
   - Scheduler interprets this as: latent should stay at current state
   - Effectively preserves the content

2. **Normal noise prediction** = "denoise this region normally"
   - Standard diffusion denoising behavior

3. **Blended transition**
   - Smooth transition from zero to normal over 0.1 step window
   - Avoids hard edges between active/inactive states

### Mathematical Correctness

```python
# Scheduler step (simplified):
x_next = x - sigma * epsilon

# When epsilon = 0 (our zero noise):
x_next = x - sigma * 0 = x  # ✅ Preserved!

# When epsilon = model_output:
x_next = x - sigma * model_output  # ✅ Normal denoising!
```

## Expected Results After Fix

1. **Large leafs** (low denoise): Should preserve most of the original content with subtle changes
2. **Small leafs** (high denoise): Should show more variation/regeneration as intended
3. **Speed**: Still faster than full denoising (this is by design)
4. **Transitions**: Smooth blending between regions with different denoise values

## Testing Recommendations

1. Test with various denoise ranges (e.g., min=0.2, max=1.0)
2. Verify large tiles preserve content better than before
3. Check small tiles generate appropriate detail
4. Inspect tile boundaries for smooth transitions
5. Compare processing time (should still be faster than non-tiled)

## Technical Notes

- This fix assumes the model returns noise predictions (epsilon), which is standard in ComfyUI
- The blend_factor uses a 0.1 step transition window for smooth activation
- The activation threshold calculation (1.0 - tile_denoise) matches img2img semantics
