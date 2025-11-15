# Variable Denoise Troubleshooting Guide

## Issue
Variable denoising based on quadtree leaf node size doesn't appear to be working.

## What Variable Denoise Should Do

With variable denoise enabled:
- **Large tiles** (low complexity areas) should get **low denoise** values (e.g., 0.2)
  - These tiles should preserve most of the original content
  - They become active late in the denoising process (at ~80% progress for denoise=0.2)
  - Result: Subtle changes, mostly preserved

- **Small tiles** (high complexity areas) should get **high denoise** values (e.g., 0.8)
  - These tiles should regenerate more content
  - They become active early in the denoising process (at ~20% progress for denoise=0.8)
  - Result: More variation and detail

## Debug Logging Added

I've added comprehensive logging to help diagnose the issue. When you run the workflow, you should see messages like:

### 1. Sigma Loading
```
[Quadtree Variable Denoise]: Loaded sigmas from store, length=21
```
Or if there's a problem:
```
[Quadtree Variable Denoise]: WARNING - No sigmas in store, variable denoise will NOT work
```

### 2. Denoise Range (during quadtree initialization)
```
[Quadtree Diffusion]: Denoise range: 0.200 to 0.800
[Quadtree Variable Denoise]: Denoise range: 0.200 to 0.800
[Quadtree Variable Denoise]: ENABLED - Tiles will be denoised adaptively
```

### 3. First Tile Check (during first denoising step)
```
[Quadtree Variable Denoise]: First tile denoise=0.352
[Quadtree Variable Denoise]: use_qt=True, has_sigmas=True, sigmas_not_none=True
[Quadtree Variable Denoise]: Variable denoise IS ACTIVE for tiles with denoise < 1.0
```

### 4. Preservation Application (when a tile is being preserved)
```
[Quadtree Variable Denoise]: APPLYING preservation - tile_denoise=0.200, progress=0.150, threshold=0.800, blend=0.000
```

## Possible Issues and Solutions

### Issue 1: Sigmas Not Available ❌
**Symptom:** You see:
```
[Quadtree Variable Denoise]: WARNING - No sigmas in store
```

**Cause:** The sampling scheduler isn't providing sigma values to the store.

**Solution:**
- Check that you're using a standard KSampler or KSampler Advanced node
- Some custom samplers may not provide sigmas
- The utils.py hooks might not be properly registered

### Issue 2: All Tiles Have denoise >= 1.0 ❌
**Symptom:** You see:
```
[Quadtree Variable Denoise]: All tiles at max denoise - no variation
```

**Cause:** Your `max_denoise` parameter is set to 1.0 and all tiles are small.

**Solution:**
- In the QuadtreeVisualizer node, set `max_denoise` to 0.8 or lower (default is 0.8)
- Variable denoise logic only activates for tiles with denoise < 1.0

### Issue 3: No Variation in Tile Sizes ⚠️
**Symptom:** All tiles are the same size.

**Cause:**
- Your image has uniform complexity
- `content_threshold` is too high (quadtree doesn't subdivide)
- `max_depth` is too low

**Solution:**
- Lower `content_threshold` to subdivide more aggressively
- Increase `max_depth` to allow smaller tiles
- Check the tile size distribution in the logs

### Issue 4: Variable Denoise Not Being Applied ❌
**Symptom:** You never see the "APPLYING preservation" message.

**Possible Causes:**
1. **Progress always >= activation_threshold**
   - This means all tiles are active from the start
   - Check that you have tiles with low denoise values (< 0.5)

2. **Sigmas not loaded**
   - Check for sigma warning messages

3. **Not using quadtree mode**
   - Verify `use_qt=True` in the first tile check log

## How to Test Variable Denoise is Working

### Test 1: Check the Logs
Run your workflow and check for all 4 types of log messages above. They should appear in the console.

### Test 2: Visual Inspection
- Create an image with both simple and complex areas (e.g., solid background + detailed foreground)
- Use the QuadtreeVisualizer to see the tile distribution
- Run denoising and compare:
  - Large tiles (background) should change very little
  - Small tiles (details) should change more

### Test 3: Compare Denoise Values
Try these settings and compare outputs:
- **Test A:** min_denoise=0.2, max_denoise=0.8 (default)
- **Test B:** min_denoise=0.8, max_denoise=0.8 (uniform, no variation)
- **Test C:** min_denoise=0.0, max_denoise=1.0 (extreme variation)

If variable denoise is working, Test A should look different from Test B.

## Expected Behavior

When variable denoise is working correctly:

1. **Early in denoising** (progress = 0.0 to ~0.2):
   - Tiles with high denoise (0.8) are active and denoising
   - Tiles with low denoise (0.2) are preserved (zero noise prediction)
   - You should see "APPLYING preservation" messages

2. **Middle of denoising** (progress = 0.2 to ~0.8):
   - More tiles become active as their thresholds are reached
   - Smooth transitions as blend_factor goes from 0.0 to 1.0

3. **Late in denoising** (progress = 0.8 to 1.0):
   - All tiles are active and denoising normally
   - No more preservation, all tiles use model output

## Parameters to Check

### QuadtreeVisualizer Node
- `min_denoise`: Default 0.2 (lower = preserve more)
- `max_denoise`: Default 0.8 (must be < 1.0 for variable denoise to work)
- `content_threshold`: Lower values = more subdivision
- `max_depth`: Higher values = allow smaller tiles

### TiledDiffusion Node
- `method`: Should be "Mixture of Diffusers" (recommended for variable denoise)
- `quadtree`: Connect from QuadtreeVisualizer output

### KSampler
- Use standard KSampler or KSamplerAdvanced
- Steps: 20+ recommended to see the effect
- Denoise: 1.0 (or img2img with appropriate denoise for the overall image)

## Still Not Working?

If variable denoise still isn't working after checking all the above:

1. **Capture the full console output** including all log messages
2. **Note which messages appear and which don't**
3. **Check your workflow parameters** (QuadtreeVisualizer, TiledDiffusion, KSampler)
4. **Report the issue** with:
   - Console logs
   - Workflow settings
   - Description of expected vs actual behavior

## Technical Note: How It Works

The variable denoise system works by:
1. Assigning each tile a denoise value based on its area (0.0 to 1.0)
2. Converting denoise to an activation threshold (threshold = 1.0 - denoise)
3. Calculating progress through the denoising schedule (0.0 to 1.0)
4. When progress < threshold: Return zero noise prediction (preserves tile)
5. When progress >= threshold: Return model prediction (denoises tile normally)
6. Smooth blending over 0.1 step window around threshold

This is mathematically correct and follows diffusion model semantics.
