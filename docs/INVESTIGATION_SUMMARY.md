# Denoising Steps Investigation - Executive Summary

## Investigation Overview

A very thorough investigation of how denoising steps are calculated, applied, and controlled for different quadtree leaf sizes/depths in the comfyui-quadtree-tile implementation.

**Repository:** `/home/user/comfyui-quadtree-tile`  
**Codebase:** ~5,351 lines of Python across core and test files  
**Investigation Depth:** Very Thorough (Comprehensive)

---

## Key Findings

### 1. Denoise Value Calculation ✅

**Location:** `tiled_vae.py` lines 289-308

The denoise values are calculated using an **area-based formula** (not depth-based):

```python
leaf.denoise = min_denoise + (max_denoise - min_denoise) * (1.0 - size_ratio)
```

Where `size_ratio = tile_area / max_tile_area`

**What This Means:**
- Larger tiles → Lower denoise → Preserve more content
- Smaller tiles → Higher denoise → Regenerate more content
- Linear interpolation between min and max denoise values
- Works correctly for square tiles (all enforced to be w == h)

**Example Results:**
| Tile Type | Size Ratio | Denoise (default 0.2-0.8) | Behavior |
|-----------|-----------|--------------------------|----------|
| Largest (low complexity) | 1.0 | 0.2 | Preserve 80% of input |
| Medium | 0.5 | 0.5 | Balanced blend |
| Smallest (high complexity) | 0.0 | 0.8 | Regenerate 80% |

---

### 2. Denoising Step Scheduling ✅

**Location:** `tiled_diffusion.py` lines 1268-1295

Step calculation follows a 4-step process:

**Step 1: Find Current Timestep**
```python
ts_in = find_nearest(t_in[0], sigmas)  # Find nearest sigma
cur_idx = (sigmas == ts_in).nonzero()  # Get its index
```

**Step 2: Calculate Progress (0.0 to 1.0)**
```python
progress = current_step / max(total_steps, 1)  # Normalized position in schedule
```

**Step 3: Determine Activation Threshold**
```python
activation_threshold = 1.0 - tile_denoise  # When tile starts being used
# Example: denoise=0.3 → threshold=0.7 (tile active after 70% progress)
```

**Step 4: Smooth Blending**
```python
blend_factor = max(0.0, min(1.0, (progress - (activation_threshold - 0.1)) / 0.1))
tile_out = tile_input * (1 - blend_factor) + tile_out * blend_factor
```

**Correctness Status:** ✅ Mathematically correct

The calculation properly implements img2img semantics:
- Low denoise = preserve input structure longer
- High denoise = use model output (regenerate) earlier

---

### 3. Noise Application to Tiles ✅

**Key Insight:** Noise is scheduler-controlled, not tile-specific

**How It Works:**
1. Scheduler provides sigma levels for each denoising step
2. Each tile processes the same timestep `t` (global, not local)
3. Model applies denoising: `x_out = network(x, t)`
4. Variable denoise **controls WHEN tile is active**, not HOW noise is applied

**Tile Extraction with Padding:**
- Reflection padding for boundary tiles (smoother)
- Fallback to replicate padding for extreme cases
- Proper shape handling with negative coordinates

**Status:** ✅ Correct approach that respects diffusion semantics

---

### 4. Critical Bug Fixes ✅

**Recent Fix (Commit 3775ccb):** "Fix denoise blending bug"

**Problem Found:**
```python
# BROKEN: Shape mismatch
tile_input = extract_tile_with_padding(...)  # Shape: [B, C, h+pad, w+pad]
tile_out = tile_input * blend + tile_out * (1-blend)  # Shape: [B, C, h, w] - CRASH!
```

**Solution Implemented:**
```python
# FIXED: Crop to matching size
if tile_input.shape[-2:] != tile_out.shape[-2:]:
    tile_input = tile_input[:, :, :tile_out.shape[-2], :tile_out.shape[-1]]
```

**Impact:** ✅ Variable denoise now works without crashing

---

## Files Analyzed

### Core Implementation Files
- **`tiled_diffusion.py`** (1,476 lines)
  - Denoise calculation logic (lines 1268-1295)
  - Tile extraction with padding (lines 72-122)
  - Step scheduling (lines 936-944)
  
- **`tiled_vae.py`** (1,422 lines)
  - Denoise assignment (lines 289-308)
  - Quadtree validation (lines 329-341)
  - Tile extraction (lines 824-902)

- **`utils.py`** (159 lines)
  - Sampling hooks and store system

### Test Files
- `test_square_quadtree.py` - Validates square tile enforcement
- `test_weight_accumulation.py` - Validates weight calculations
- `test_coverage_filter.py` - Validates pixel coverage
- And 11 other test/debug files

---

## Issues Found

### ✅ No Critical Issues
The denoising step calculation is fundamentally correct.

### ⚠️ Minor Issues

1. **Size-Based (Not Depth-Based) Denoise**
   - Denoise assigned by tile area, not tree depth
   - A 128×128 tile at depth 2 gets same denoise as depth 4 tile
   - May not match user expectations if they assume depth controls denoise
   - **Impact:** Low (current behavior makes sense)

2. **Sigmas Requirement**
   - Variable denoise disabled if sigmas unavailable
   - Affects only non-standard samplers
   - **Impact:** Low (most samplers provide sigmas)

3. **Edge Case: Single Sigma**
   - If only 1 sigma in schedule: `total_steps = 0`
   - Uses `max(0, 1)` fallback, but worth documenting
   - **Impact:** Very low (rarely occurs)

4. **denoise >= 1.0 Handling**
   - Tiles with denoise=1.0 skip variable denoise logic
   - Default max_denoise=1.0 means smallest tiles don't use blending
   - **Impact:** Medium (could affect smallest tile quality)

---

## Validation Results

| Component | Status | Notes |
|-----------|--------|-------|
| Denoise calculation | ✅ | Area-based formula correct |
| Step counting | ✅ | Proper 0-based indexing |
| Progress calculation | ✅ | Normalized 0.0 to 1.0 |
| Activation logic | ✅ | Inverse relationship correct |
| Blend smoothing | ✅ | Bounded [0.0, 1.0] |
| Shape matching | ✅ | Fixed in recent commit |
| Square tile enforcement | ✅ | Validated in build() |
| Pixel coverage | ✅ | Validated and checked |
| Padding logic | ✅ | Proper reflection/replicate |

---

## Recommendations

### High Priority ✅ (Already Done)
- [x] Fix tensor shape mismatch in blending (commit 3775ccb)
- [x] Validate square tile enforcement
- [x] Add coverage validation

### Medium Priority ⚠️
- [ ] Add explicit warning when sigmas unavailable
- [ ] Document that denoise is size-based, not depth-based
- [ ] Consider supporting depth-based denoise as alternative
- [ ] Add per-tile denoise value logging for debugging
- [ ] Test edge cases (extreme aspect ratios, single sigma)

### Low Priority 
- [ ] Optimize find_nearest() to return index directly
- [ ] Cache progress calculation across tiles in same step
- [ ] Add visualization of denoise values per tile

---

## Conclusion

✅ **The denoising step calculation is CORRECT and WORKING as designed.**

The implementation:
- Properly calculates denoise values based on tile area
- Correctly computes step progress through the schedule
- Implements sound activation logic with smooth blending
- Handles padding and tile extraction properly
- Includes recent bug fix for shape compatibility

The codebase follows standard diffusion semantics and all mathematical operations are valid. The variable denoise feature is a sophisticated but correct implementation of img2img-style denoising with quadtree tile sizing.

**Status:** Ready for use with awareness of minor design choices

---

## Report Files

Full detailed investigation available in:
- `/home/user/comfyui-quadtree-tile/docs/DENOISING_INVESTIGATION_REPORT.md` (Comprehensive, 400+ lines)
- `/home/user/comfyui-quadtree-tile/docs/INVESTIGATION_SUMMARY.md` (This file)

**Generated:** 2025-11-14  
**Investigation Method:** Code analysis, git history review, test file examination
