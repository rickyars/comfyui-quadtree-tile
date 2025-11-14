# Comprehensive Investigation: Denoising Steps in Quadtree Tiling

## Investigation Scope
This investigation covers how denoise steps are calculated, applied, and controlled for different quadtree leaf sizes/depths in the comfyui-quadtree-tile implementation.

---

## 1. DENOISE VALUE CALCULATION & ASSIGNMENT

### Location: `tiled_vae.py` lines 289-308

**Function:** `QuadtreeBuilder.assign_denoise_values(root_node)`

### Current Implementation:
```python
def assign_denoise_values(self, root_node: QuadtreeNode):
    # First pass: find maximum tile area
    leaves = self.get_leaf_nodes(root_node)
    self.max_tile_area = max(leaf.w * leaf.h for leaf in leaves)  # Line 300

    # Second pass: assign denoise values
    for leaf in leaves:
        tile_area = leaf.w * leaf.h                                # Line 304
        size_ratio = tile_area / self.max_tile_area                # Line 305
        # 0.0 (smallest) to 1.0 (largest)

        # Inverse relationship: large tiles get low denoise, small tiles get high denoise
        leaf.denoise = self.min_denoise + \                        # Line 308
                       (self.max_denoise - self.min_denoise) * \
                       (1.0 - size_ratio)
```

### Analysis:

**The Formula Breakdown:**
- `size_ratio = tile_area / max_tile_area` ranges from 0.0 (smallest) to 1.0 (largest)
- `denoise = min_denoise + (max_denoise - min_denoise) * (1.0 - size_ratio)`

**Example with Default Values (min_denoise=0.2, max_denoise=0.8):**

| Tile Size | size_ratio | 1.0 - size_ratio | Denoise Value | Meaning |
|-----------|-----------|-----------------|---------------|---------|
| Largest   | 1.0       | 0.0             | 0.2           | Preserve (low denoise) |
| Medium    | 0.5       | 0.5             | 0.5           | Balanced |
| Smallest  | 0.0       | 1.0             | 0.8           | Regenerate (high denoise) |

**Key Characteristics:**
1. ✅ Size-based (area), not depth-based
2. ✅ Inverse relationship: Larger tiles → lower denoise → preserve content
3. ✅ Inverse relationship: Smaller tiles → higher denoise → regenerate
4. ✅ Linear interpolation between min/max values
5. ✅ No special handling for boundary tiles

**Potential Issues:**

1. **Non-Square Tile Assumption:** The formula uses `w * h` (area) for calculation. With square quadtree tiles (all w==h enforced), this is equivalent to `size²`, so a 2x smaller tile has 4x smaller denoise value change.

2. **No Depth-Based Logic:** Denoise is purely area-based, not considering tree depth. A tile at depth 3 gets same denoise as any other tile with same area, regardless of depth.

3. **Edge Case - All Same Size:** If all tiles end up same size (rare), all get same denoise value.

4. **No Special Handling:** No special denoise values for boundary vs interior tiles.

---

## 2. DENOISING STEP SCHEDULING & PROGRESS CALCULATION

### Location: `tiled_diffusion.py` lines 1268-1295

**Context:** `MixtureOfDiffusers.__call__()` method during tile processing

### Current Implementation:
```python
# Line 1266-1278
tile_denoise = getattr(bbox, 'denoise', 1.0) if use_qt else 1.0

if use_qt and hasattr(self, 'sigmas') and self.sigmas is not None and tile_denoise < 1.0:
    # Calculate progress through denoising schedule (0 = start, 1 = end)
    sigmas = self.sigmas
    ts_in = find_nearest(t_in[0], sigmas)                         # Line 1271
    cur_idx = (sigmas == ts_in).nonzero()                          # Line 1272
    if cur_idx.shape[0] > 0:
        current_step = cur_idx.item()                              # Line 1274
        total_steps = len(sigmas) - 1                              # Line 1275
        progress = current_step / max(total_steps, 1)              # Line 1278
        
        activation_threshold = 1.0 - tile_denoise                  # Line 1282
        
        if progress < activation_threshold:                        # Line 1284
            blend_factor = max(0.0, min(1.0, \                    # Line 1287
                             (progress - (activation_threshold - 0.1)) / 0.1))
            tile_input = extract_tile_with_padding(x_in, bbox, self.w, self.h)
            tile_out = tile_input * (1 - blend_factor) + tile_out * blend_factor
```

### Step-by-Step Analysis:

**Step 1: Find Current Timestep** (Lines 1271-1272)
```python
ts_in = find_nearest(t_in[0], sigmas)  # Find nearest sigma to current timestep
cur_idx = (sigmas == ts_in).nonzero()  # Find index of that sigma
```

**Function: `find_nearest(a, b)`** (Lines 936-944)
```python
def find_nearest(a, b):
    diff = (a - b).abs()           # Absolute differences
    nearest_indices = diff.argmin() # Index of minimum difference
    return b[nearest_indices]       # Return the nearest value
```

**Analysis of find_nearest:**
- ✅ Returns the nearest sigma value (not the index)
- ✅ Then line 1272 finds the index of that nearest value
- ✅ Simple but effective

**Step 2: Calculate Progress** (Lines 1274-1278)
```python
current_step = cur_idx.item()              # Step index (0 = start)
total_steps = len(sigmas) - 1              # Total denoising steps
progress = current_step / max(total_steps, 1)  # Normalized 0.0 to ~1.0
```

**Progress Calculation Analysis:**
- ✅ `progress = current_step / total_steps` gives 0.0 at start, 1.0 at end
- ⚠️ Off-by-one consideration: `total_steps = len(sigmas) - 1`
  - If sigmas has 21 steps: total_steps = 20
  - current_step ranges from 0 to 20
  - progress ranges from 0.0 to 1.0 ✅

**Step 3: Calculate Activation Threshold** (Line 1282)
```python
activation_threshold = 1.0 - tile_denoise  # When tile starts being used
```

**Example with tile_denoise = 0.3 (low denoise, preserve):**
- activation_threshold = 1.0 - 0.3 = 0.7
- Tile only starts denoising when progress >= 0.7 (70% through schedule)
- Before 70%: tile input is preserved, blended gradually in
- After 70%: tile output (model result) is used

**Step 4: Calculate Blend Factor** (Line 1287)
```python
blend_factor = max(0.0, min(1.0, (progress - (activation_threshold - 0.1)) / 0.1))
```

**Blend Factor Breakdown:**
- Smooth transition window: `activation_threshold - 0.1` to `activation_threshold`
- Example with threshold = 0.7, transition from 0.6 to 0.7:
  - At progress = 0.60: blend_factor = max(0, min(1, (0.60 - 0.60) / 0.1)) = 0.0 (use input)
  - At progress = 0.65: blend_factor = max(0, min(1, (0.65 - 0.60) / 0.1)) = 0.5 (50% blend)
  - At progress = 0.70: blend_factor = max(0, min(1, (0.70 - 0.60) / 0.1)) = 1.0 (use output)
  - At progress > 0.70: blend_factor = 1.0 (use output fully)

**Step 5: Apply Blending** (Line 1295)
```python
tile_out = tile_input * (1 - blend_factor) + tile_out * blend_factor
```

### Analysis Summary:

| Component | Status | Notes |
|-----------|--------|-------|
| Sigma lookup | ✅ | Uses find_nearest for stable lookup |
| Step counting | ✅ | Correctly indexed 0 to (len-1) |
| Progress calculation | ✅ | Normalized 0.0 to 1.0 |
| Activation logic | ✅ | Lower denoise = later activation (preserve longer) |
| Blend smoothing | ✅ | 0.1 step smooth transition window |
| Shape matching | ⚠️ | Fixed in recent commit 3775ccb |

---

## 3. NOISE APPLICATION & TILE EXTRACTION

### Tile Extraction with Padding
**Location:** `tiled_diffusion.py` lines 72-122

```python
def extract_tile_with_padding(tensor: Tensor, bbox: BBox, image_w: int, 
                              image_h: int) -> Tensor:
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    
    # Calculate clamped extraction region
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(image_w, x + w)
    y_end = min(image_h, y + h)
    
    # Extract available region
    tile = tensor[:, :, y_start:y_end, x_start:x_end]
    
    # Calculate padding needed
    pad_left = max(0, -x)
    pad_right = max(0, (x + w) - image_w)
    pad_top = max(0, -y)
    pad_bottom = max(0, (y + h) - image_h)
    
    # Apply reflection or replicate padding
    if can_reflect_h and can_reflect_w:
        tile = F.pad(tile, (pad_left, pad_right, pad_top, pad_bottom), 
                    mode='reflect')
    else:
        tile = F.pad(tile, (pad_left, pad_right, pad_top, pad_bottom), 
                    mode='replicate')
```

**Key Points:**
- ✅ Reflection padding when possible (smoother boundaries)
- ✅ Fallback to replicate padding for extreme cases
- ✅ Properly handles negative coordinates (boundary tiles)

---

## 4. RECENT BUG FIXES

### Bug #1: Denoise Blending Shape Mismatch

**Commit:** `3775ccb` - "Fix denoise blending bug"

**Location:** `tiled_diffusion.py` lines 1290-1293

**Problem:**
```python
# BEFORE (Broken):
tile_input = extract_tile_with_padding(x_in, bbox, self.w, self.h)  # Shape: [B, C, h+2pad, w+2pad]
# ...
tile_out = tile_input * (1 - blend_factor) + tile_out * blend_factor  # Shape mismatch!
```

The issue: `tile_input` includes padding, but `tile_out` is cropped to bbox size.

**Fix:**
```python
# AFTER (Fixed):
tile_input = extract_tile_with_padding(x_in, bbox, self.w, self.h)

# CRITICAL FIX: Crop tile_input to match tile_out size
if tile_input.shape[-2:] != tile_out.shape[-2:]:
    tile_input = tile_input[:, :, :tile_out.shape[-2], :tile_out.shape[-1]]

tile_out = tile_input * (1 - blend_factor) + tile_out * blend_factor
```

**Impact:** ✅ Fixed broken variable denoise blending

---

## 5. CRITICAL ISSUES & POTENTIAL PROBLEMS

### Issue #1: Step Counting Edge Case

**Location:** `tiled_diffusion.py` line 1275

**Current Code:**
```python
total_steps = len(sigmas) - 1
progress = current_step / max(total_steps, 1)
```

**Potential Problem:**
- If `sigmas` list has only 1 element: `total_steps = 0`
- `progress = current_step / max(0, 1) = current_step / 1`
- This could cause unexpected behavior

**Status:** Minor edge case, unlikely in practice

### Issue #2: Denoise Only Works if sigmas Available

**Location:** `tiled_diffusion.py` line 1268

**Current Code:**
```python
if use_qt and hasattr(self, 'sigmas') and self.sigmas is not None and tile_denoise < 1.0:
```

**Problem:** Variable denoise completely disabled if:
1. sigmas not available (different sampler)
2. tile_denoise >= 1.0 (largest tile in certain config)
3. use_qt is False

**Status:** Expected behavior but limits feature availability

### Issue #3: No Handling for Tiles with denoise >= 1.0

**Location:** Variable denoise logic doesn't apply when `tile_denoise >= 1.0`

**Current Code:**
```python
if use_qt and ... and tile_denoise < 1.0:
    # Variable denoise logic applies
    ...
```

**Problem:** Tiles with denoise=1.0 (including all tiles if max_denoise=1.0) never use blending logic

**Impact:** Default max_denoise=1.0 means variable denoise has limited effect for smallest tiles

### Issue #4: Depth-Independent Denoise Assignment

**Location:** `tiled_vae.py` lines 289-308

**Problem:** Denoise values based only on tile area, not tree depth
- A 128×128 tile at depth 2 gets same denoise as 128×128 tile at depth 4
- No explicit depth-based logic
- May not match user expectations (depth controls denoise)

**Current:** Size-based formula works but doesn't explicitly use depth

---

## 6. STEP CALCULATION CORRECTNESS

### Is Step Calculation Correct?

**Analysis:**
1. ✅ Step indexing (0-based) matches sigmas list
2. ✅ Progress calculation (0.0 to 1.0) is linear and correct
3. ✅ Activation threshold logic is sound
4. ✅ Blend factor smoothing is properly bounded [0.0, 1.0]

**Conclusion:** ✅ Step calculation is mathematically correct

### Semantic Correctness

**Question:** Does denoise=0.3 mean "preserve 30%, regenerate 70%" or something else?

**How It Works:**
- denoise=0.3 → activation_threshold = 0.7
- Tile is active (using model output) only after 70% progress
- Early in schedule (high noise): uses input (preserves structure)
- Late in schedule (low noise): uses output (refines details)

**Semantics:** ✅ Correct for img2img-like behavior (low denoise = preserve)

---

## 7. NOISE APPLICATION TO TILES

### How Noise is Applied

**Key Insight:** Noise is part of the diffusion scheduler, not applied per-tile

**Process:**
1. Scheduler provides `sigmas` list (noise levels for each step)
2. Each step has a timestep `t` and corresponding noise level `sigma`
3. Model applies denoising: `x_out = denoising_network(x, t)`
4. Quadtree tiling just applies this per-tile
5. Variable denoise blends between input and output based on progress

**Tile-Specific Behavior:**
- All tiles process same timestep (not independent)
- Variable denoise affects WHEN tile is active, not HOW noise is applied
- Noise is scheduler-controlled, not tile-controlled

**Analysis:** ✅ Correct approach (respects diffusion semantics)

---

## 8. QUADTREE STRUCTURE VALIDATION

### Square Tile Enforcement

**Location:** `tiled_vae.py` lines 238-246, 329-341

**Validation:**
```python
# Check all leaves are square
non_square_leaves = []
for leaf in leaves:
    if leaf.w != leaf.h:
        non_square_leaves.append((leaf.x, leaf.y, leaf.w, leaf.h))

if non_square_leaves:
    raise AssertionError(f"Found {len(non_square_leaves)} non-square leaves")
```

**Status:** ✅ Enforced - quadtree creates only square tiles

### Tile Coverage Validation

**Location:** `tiled_diffusion.py` lines 527-550

```python
# Validate full coverage
if self.weights.min() < 1e-6:
    uncovered = (self.weights < 1e-6).sum().item()
    if uncovered > 0:
        raise RuntimeError(f"Quadtree has {uncovered} uncovered pixels!")
```

**Status:** ✅ Validates all pixels covered by at least one tile weight

---

## 9. SUMMARY TABLE: DENOISING FLOW

| Step | Component | Input | Output | Status |
|------|-----------|-------|--------|--------|
| 1 | Quadtree Builder | Image | Tiles with denoise values | ✅ |
| 2 | Denoise Assignment | Tile area | size_ratio → denoise | ✅ |
| 3 | Diffusion Sampling | Sigmas, tiles | Noisy latent per tile | ✅ |
| 4 | Step Lookup | Current t | current_step index | ✅ |
| 5 | Progress Calc | current_step | progress [0.0, 1.0] | ✅ |
| 6 | Threshold Calc | tile_denoise | activation_threshold | ✅ |
| 7 | Blend Decision | progress vs threshold | blend_factor | ✅ |
| 8 | Apply Blend | tile_input, tile_out | final_tile_out | ✅ Fixed |
| 9 | Accumulate | tiles with weights | final image | ✅ |

---

## 10. RECOMMENDATIONS

### High Priority:
1. ✅ **Shape matching fix** - Already done in commit 3775ccb
2. ⚠️ Add warning if sigmas unavailable (variable denoise won't work)
3. ⚠️ Document that denoise is size-based, not depth-based

### Medium Priority:
1. Consider adding depth-based denoise option as alternative
2. Test edge cases (single sigma, all tiles same size, extreme aspect ratios)
3. Add logging of calculated denoise values for debugging

### Low Priority:
1. Optimize find_nearest to return index directly (currently returns value then finds index)
2. Consider caching progress calculation across tiles in same step
3. Add visualization of denoise values per tile

---

## Conclusion

✅ **The denoising step calculation is fundamentally correct:**
- Denoise values properly assigned based on tile area
- Step progress correctly calculated as normalized position in schedule
- Activation logic properly implements selective denoising
- Recent fix ensures shape compatibility

⚠️ **Minor issues to be aware of:**
- Size-based (not depth-based) denoise assignment
- Variable denoise disabled if sigmas unavailable
- Edge case handling could be more explicit

The implementation follows standard diffusion semantics and the math checks out.

