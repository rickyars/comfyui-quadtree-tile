# Skip Tile Bug Analysis: Why Skipped Tiles Show Pure Noise

## Executive Summary

**The Bug**: Skipped tiles appear as pure noise in the final output.

**Root Cause**: The skip implementation copies the noisy latent input (`x_in`) into the noise prediction buffer (`x_buffer`), causing the sampler to receive garbage data instead of valid noise predictions.

**Impact**: Any tiles below the `skip_diffusion_below` threshold will render as pure noise, making the feature completely broken.

---

## 1. Understanding the Diffusion Flow

### What is `x_in` at different denoising steps?

The diffusion process works like this:

```
Step 0 (start):  x_in = pure noise (random gaussian)
Step 5:          x_in = partially denoised (still very noisy)
Step 10:         x_in = more denoised (less noisy)
Step 15:         x_in = mostly denoised
Step 20 (end):   x_in = clean image
```

**Key point**: `x_in` changes at EVERY denoising step. It's the current state of the noisy latent, progressively getting cleaner.

### How does the model work?

From web research and code analysis:
- "The noise predictor estimates the noise of the image, the predicted noise is subtracted from the image"
- "From the noisy image xₜ and timestep t, the model predicts the noise ϵ"

So:
1. **Input**: `x_in` (noisy latent), `t` (timestep)
2. **Model Output**: Predicted noise that should be subtracted
3. **Sampler**: `x_next = x_in - predicted_noise` (simplified)

---

## 2. When Does `__call__` Get Invoked?

The `__call__` method (line 1047 in tiled_diffusion.py) is invoked **ONCE PER DENOISING STEP**.

For a typical 20-step sampling:
- Called 20 times
- Each time with different `x_in` (progressively less noisy)
- Each time it must return a noise prediction

**This is NOT a one-time operation!**

---

## 3. What Should `x_buffer` Contain?

Looking at the code flow:

```python
# Line 337-339: Initialize buffer to ZEROS
self.x_buffer = torch.zeros_like(x_in)

# Line 1265: Model predicts noise for each tile
x_tile_out = model_function(x_tile, t_tile, **c_tile)

# Line 1356: Accumulate weighted noise predictions
self.x_buffer[:, :, y_start:y_end, x_start:x_end] += valid_tile * tile_weights

# Line 1372-1380: Normalize and return
x_out = self.x_buffer / self.weights
return x_out
```

**`x_buffer` is the accumulated NOISE PREDICTION, not image data!**

---

## 4. The Bug: Mixing Apples and Oranges

### Current Buggy Implementation (lines 1173-1204)

```python
# Process skip tiles: copy input directly to output buffer
for bbox in skip_bboxes:
    tile_input = extract_tile_with_padding(x_in, bbox, self.w, self.h)  # ← NOISY LATENT
    # ...
    self.x_buffer[:, :, y_start:y_end, x_start:x_end] += valid_tile * tile_weights  # ← WRONG!
```

**What's wrong**:
- `x_in` = noisy latent at current timestep (image data)
- `x_buffer` = noise prediction buffer (should be noise, not image!)
- Copying image data into noise prediction buffer is mixing incompatible data types

### What the Sampler Receives

The sampler expects:
```
predicted_noise = tiled_diffusion.__call__(x_in, t, c)
x_next = sampler_step(x_in, predicted_noise)  # Subtract noise
```

But for skipped tiles, it receives:
```
predicted_noise[skipped_region] = x_in[skipped_region]  # ← Garbage!
```

This causes the sampler to do:
```
x_next = x_in - x_in  # = zero (or weird artifacts due to weighting)
```

---

## 5. Why Does This Cause Noise?

Let's trace through one denoising step:

**Step 1 (high noise)**:
- `x_in` = very noisy (like 90% noise, 10% signal)
- Skipped region: `x_buffer = x_in` (noisy data)
- Sampler: `x_next = x_in - x_in = 0` (black) or garbage

**Step 10 (medium noise)**:
- `x_in` = medium noise (like 50% noise, 50% signal)
- Skipped region: `x_buffer = x_in` (still copying wrong data)
- Sampler: continues to receive garbage predictions

**Result**: Skipped regions never denoise properly, appear as noise or black holes.

---

## 6. Commit History Analysis

### Commit `be0ecca` (CORRECT Implementation)

```python
# Returns zero noise prediction for tiles below threshold
if min_dimension < skip_threshold:
    tile_out = torch.zeros_like(tile_out)  # ← CORRECT: Zero noise = no change
```

**Why this works**:
- Zero noise prediction means "don't change this region"
- Sampler: `x_next = x_in - 0 = x_in` (preserves current state)
- Over multiple steps, region gradually denoises naturally with surrounding context

### Commit `5feb41a` (INTRODUCED BUG)

Commit message claimed:
> "Skip tiles: Copy input directly to output buffer (no model call)"

```python
# Copy directly to output buffer (no diffusion)
tile_input = extract_tile_with_padding(x_in, bbox, self.w, self.h)
self.x_buffer[:, :, y_start:y_end, x_start:x_end] += valid_tile * tile_weights
```

**What went wrong**:
- Tried to optimize by skipping model call entirely
- But copied WRONG data type (`x_in` instead of noise prediction)
- Fundamental misunderstanding of what `x_buffer` represents

---

## 7. Proper Fix Options

### Option A: Return Zero Noise (Recommended)

**Revert to original `be0ecca` approach but keep the optimization**:

```python
# Process skip tiles: add ZERO noise prediction
for bbox in skip_bboxes:
    # Don't add anything to x_buffer (it's already zeros)
    # Just ensure weights are properly accounted for (they already are from init)
    pass  # No-op: zero noise = preserve current state
```

**Wait, there's a problem**: If we don't add anything to `x_buffer`, but weights were already accumulated during `init_quadtree_bbox`, the normalization will divide by incorrect weights!

**Solution**: Don't accumulate weights for skipped tiles during init, OR add zeros explicitly.

### Option B: Store Original Clean Latent (Complex)

For img2img workflows:
- Store the original encoded latent before adding noise
- Copy from original latent instead of noisy `x_in`

**Problems**:
- Requires access to original latent (may not be available in all workflows)
- Doesn't work for txt2img (no original content exists)
- More complex implementation

### Option C: Use Simple Diffusion (Compromise)

- Don't skip model entirely
- Run a simpler/faster model version for small tiles
- Still get valid noise predictions but with less compute

### Option D: Skip Only After Partial Denoising

- Skip only in later denoising steps when image is mostly clean
- Early steps: process all tiles normally
- Late steps (e.g., last 5): skip small tiles

---

## 8. Recommended Fix: Zero Noise with Weight Adjustment

The cleanest fix:

```python
# Process skip tiles: return zero noise prediction
for bbox in skip_bboxes:
    # Zero noise prediction = preserve current state
    # x_buffer already initialized to zeros, so we don't need to add anything
    # BUT we need to handle weights correctly!
    
    if self.tile_overlap > 0:
        # IMPORTANT: Don't add to weights for skipped tiles
        # They were added during init, but we need to subtract them OR
        # add zero with same weights (so normalization works correctly)
        
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(self.w, x + w)
        y_end = min(self.h, y + h)
        
        # Since x_buffer[region] is already zero, and we divide by weights,
        # the result will be zero (which is what we want)
        # No action needed! The zeros are already there.
```

**Actually, even simpler**: Just don't process skipped tiles at all! The `x_buffer` is already zeros, which represents zero noise prediction. The weights were accumulated during init, so the normalization will work correctly:

```
x_out = x_buffer / weights = 0 / weights = 0
```

Zero noise prediction means the sampler will preserve the current state:
```
x_next = x_in - 0 = x_in
```

Perfect! The region will naturally denoise through the diffusion process without explicit denoising.

---

## 9. Wait, Does This Actually Work?

Let me reconsider... If we return zero noise:
- Early steps: `x_next = very_noisy_x - 0 = very_noisy_x` (stays noisy)
- Late steps: `x_next = slightly_noisy_x - 0 = slightly_noisy_x` (still slightly noisy)
- End: Skipped regions will still be noisy!

**This means zero noise doesn't preserve original content, it preserves the NOISE!**

For **txt2img**: This is bad, skipped regions stay noisy.
For **img2img**: This is bad, skipped regions don't preserve original.

---

## 10. The REAL Solution

After deeper analysis, I realize:

**The feature goal is unclear!** What does "skip diffusion to preserve original details" mean?

### For txt2img:
- There IS no "original content" - it starts from pure noise
- Skipping diffusion = leaving noise as-is = BAD
- **Solution**: Don't skip at all for txt2img, or use lighter denoising

### For img2img:
- There IS original content (encoded image)
- Goal: Preserve original in small tiles, only denoise large tiles
- **Solution**: Need access to the original CLEAN latent (before noise was added)
- Then copy from original latent, not from `x_in`

### The Problem:
The `__call__` method doesn't have access to the original clean latent! It only receives:
- `x_in`: Current noisy state
- `t`: Timestep
- `c`: Conditioning

**We need to store the original latent somewhere accessible!**

---

## 11. Final Recommendations

### Recommendation 1: Fix for img2img (Store Original Latent)

```python
# In the apply() method when model is wrapped:
self.impl.original_latent = None  # Will be set by sampler if available

# During first __call__:
if self.original_latent is None and denoise_strength < 1.0:
    # This is img2img - store the clean latent
    # We can detect this by checking if we're not at maximum noise
    # and infer the clean latent by denoising x_in backward
    pass  # Complex - need sampler cooperation

# For skip tiles:
if self.original_latent is not None:
    # Copy from original clean latent
    tile_input = extract_tile_with_padding(self.original_latent, bbox, self.w, self.h)
else:
    # txt2img: Use zero noise (or just process normally)
    pass
```

### Recommendation 2: Simpler Fix (Reduce Denoise Strength)

Instead of skipping entirely, use the **variable denoise** feature that already exists:
- Set very low denoise values for small tiles (e.g., 0.1)
- They'll still be processed but with minimal change
- This already works in the codebase!

### Recommendation 3: Just Remove the Feature

The skip feature is broken and complex to fix. The variable denoise feature already provides similar benefits (minimal processing for small tiles).

---

## 12. What To Do Next

1. **Immediate**: Remove or disable the skip feature (set default to 0, warn users)
2. **Short-term**: Use variable denoise instead (already implemented)
3. **Long-term**: Implement proper skip with original latent storage (complex)

The current implementation is fundamentally broken and cannot work correctly without major changes.
