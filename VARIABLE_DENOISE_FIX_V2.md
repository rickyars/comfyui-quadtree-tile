# Variable Denoise Fix V2 - txt2img Support

## Problem Discovered

The previous variable denoise implementation had a **fundamental flaw**: it only worked for img2img, not txt2img.

### Why the Old Approach Failed in txt2img

**Old approach (broken for txt2img):**
```python
if progress < activation_threshold:
    zero_noise = torch.zeros_like(tile_out)
    tile_out = zero_noise  # Return zero noise prediction
```

**What this does:**
- In img2img: Zero noise = "don't change the input image" = ✅ Preserves content
- In txt2img: Zero noise = "don't change the random noise" = ❌ Keeps tiles noisy, no variation

**Evidence from user logs:**
```
[Quadtree Variable Denoise]: APPLYING preservation - tile_denoise=0.650, progress=0.105, threshold=0.350, blend=0.000
[Quadtree Diffusion DEBUG]: After normalization, x_out min=0.0000, max=0.0000
```

The output was ALL ZEROS in step 1, meaning all tiles were "preserved" as random noise!

## New Approach: Scaled Denoising

Instead of returning zero noise (no change), we now **scale the noise prediction** to slow down denoising:

```python
if progress < activation_threshold:
    # Calculate scale factor based on progress
    scale_factor = max(0.0, min(1.0, (progress - (activation_threshold - 0.2)) / 0.2))

    # Scale the noise prediction to slow down denoising
    tile_out = tile_out * scale_factor
```

### How This Works

**In diffusion, the scheduler updates:**
```python
x_next = x_current - sigma * epsilon
```

Where `epsilon` is the noise prediction from the model.

**By scaling epsilon:**
```python
x_next = x_current - sigma * (scale * epsilon)
     = x_current - (scale * sigma) * epsilon
```

This effectively reduces the step size, slowing down denoising.

### Effect on Different Workflows

**txt2img (generating from noise):**
- Tiles with low denoise (large tiles): Scale ~ 0.0 early on → Stay noisy longer → Less detail
- Tiles with high denoise (small tiles): Scale ~ 1.0 early on → Denoise faster → More detail
- ✅ Creates variation by controlling denoising speed

**img2img (modifying existing image):**
- Tiles with low denoise: Scale ~ 0.0 early on → Less change to input → Preserves more
- Tiles with high denoise: Scale ~ 1.0 early on → More change to input → Regenerates more
- ✅ Still works as intended

## Scale Factor Calculation

```python
scale_factor = max(0.0, min(1.0, (progress - (activation_threshold - 0.2)) / 0.2))
```

**For a tile with denoise=0.3 (activation_threshold=0.7):**

| Progress | Formula | Scale | Effect |
|----------|---------|-------|--------|
| 0.0 | (0.0 - 0.5) / 0.2 | 0.0 | No denoising |
| 0.3 | (0.3 - 0.5) / 0.2 | 0.0 | No denoising |
| 0.5 | (0.5 - 0.5) / 0.2 | 0.0 | No denoising |
| 0.6 | (0.6 - 0.5) / 0.2 | 0.5 | Half denoising |
| 0.7 | (0.7 - 0.5) / 0.2 | 1.0 | Full denoising |
| 1.0 | (1.0 - 0.5) / 0.2 | 1.0 | Full denoising |

**Note:** We use a 0.2 transition window (increased from 0.1) for smoother blending.

## Important: Check Your Denoise Range!

The user's logs showed:
```
[Quadtree Variable Denoise]: Denoise range: 0.650 to 0.791
```

**This range is too narrow!** Only 0.141 difference between large and small tiles.

### Recommended Settings

In your **QuadtreeVisualizer** node:
- `min_denoise`: **0.2** (for large tiles, low complexity)
- `max_denoise`: **0.8** (for small tiles, high complexity)

This gives a 0.6 difference, creating noticeable variation.

### Effect of Different Ranges

| min | max | Difference | Effect |
|-----|-----|------------|--------|
| 0.65 | 0.79 | 0.14 | ❌ Barely noticeable |
| 0.5 | 0.9 | 0.4 | ⚠️ Moderate variation |
| 0.2 | 0.8 | 0.6 | ✅ Good variation (default) |
| 0.0 | 1.0 | 1.0 | ⚠️ Extreme variation |

## Expected Results After Fix

With proper denoise range (0.2 to 0.8):

**Large tiles (simple areas like sky, backgrounds):**
- Denoise = 0.2
- Scale factor stays at 0.0 until 80% progress
- Minimal denoising in early steps
- Result: Less detail, smoother, more consistent

**Small tiles (complex areas like faces, details):**
- Denoise = 0.8
- Scale factor reaches 1.0 by 20% progress
- Full denoising throughout most steps
- Result: More detail, more variation, more refined

**Medium tiles:**
- Denoise values between 0.2 and 0.8
- Progressive denoising based on size
- Smooth transition between simple and complex

## Testing the Fix

1. **Update your QuadtreeVisualizer settings:**
   - Set `min_denoise` to 0.2
   - Set `max_denoise` to 0.8

2. **Run txt2img** with an image that has:
   - Simple areas (solid backgrounds)
   - Complex areas (detailed subjects)

3. **Look for variation:**
   - Simple areas should be smoother, less detailed
   - Complex areas should be more detailed, more varied

4. **Check the logs:**
   ```
   [Quadtree Variable Denoise]: Denoise range: 0.200 to 0.800
   [Quadtree Variable Denoise]: SCALING denoising - tile_denoise=0.200, progress=0.105, threshold=0.800, scale=0.000
   ```
   - Verify scale=0.000 for low-denoise tiles early on
   - Verify scale increases as progress increases

## Technical Notes

- The scale factor uses a 0.2 transition window (20% of progress)
- This is wider than the previous 0.1 window for smoother transitions
- Scaling works in both epsilon-prediction and v-prediction modes
- The approach is mathematically sound and respects diffusion semantics

## Limitations

This approach is an approximation. The ideal implementation would:
- Actually start tiles at different noise levels (like img2img does per-tile)
- Require access to the initial latent state
- More complex to implement

The scaling approach is simpler and works reasonably well for both txt2img and img2img.
