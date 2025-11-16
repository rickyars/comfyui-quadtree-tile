# Edge Node Size Issue - Findings Summary

## Problem Statement

For a **2241x3600 pixel image**:
- Log shows: "Tile dimensions range from **64x16** to **1024x2048**"
- **5 edge tiles were cropped** to fit the image bounds
- Some tiles are **too small** (64x16) for diffusion models
- Some tiles are **too large** (with overlap added), causing ComfyUI freezes

## Root Cause

**The minimum tile size check happens BEFORE edge cropping, not after.**

1. Quadtree creates square tiles >= `min_tile_size` (default 256px)
2. Square root (4096x4096) extends beyond rectangular image (2241x3600)
3. Edge tiles are cropped to fit image bounds **WITHOUT re-checking minimum size**
4. Result: Tiles can become as small as 64x16 (8x2 in latent space)

## Code Analysis

### 1. Where Edge Nodes Are Created

**File:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`

**Square Root Creation (lines 236-246):**
```python
# Creates 4096x4096 square root for 2241x3600 image
root_size = max(w, h)
if root_size > 8:
    n = math.ceil(math.log2(root_size / 8))
    root_size = 8 * (2 ** n)  # Result: 4096
root_node = QuadtreeNode(0, 0, root_size, root_size, 0)
```

**Subdivision Logic (lines 192-217):**
```python
def should_subdivide(self, node: QuadtreeNode, variance: float) -> bool:
    # Minimum size check - BEFORE cropping
    half_w_aligned = ((node.w // 2) // 8) * 8
    half_h_aligned = ((node.h // 2) // 8) * 8

    if half_w_aligned < max(self.min_tile_size, 8):
        return False  # Stops subdivision

    # ... 8-pixel alignment checks ...

    return variance > self.content_threshold
```

✓ **Working as designed** - creates valid square tiles >= min_tile_size

### 2. Where Sizes Are Determined (PROBLEM HERE!)

**File:** `/home/user/comfyui-quadtree-tile/tiled_diffusion.py`

**Edge Cropping in Diffusion (lines 399-435):**
```python
for leaf in leaves:
    # Convert to latent space
    core_start_x = leaf.x // 8
    core_start_y = leaf.y // 8
    core_end_x = (leaf.x + leaf.w) // 8
    core_end_y = (leaf.y + leaf.h) // 8

    # Filter completely outside tiles
    if core_start_x >= self.w or core_end_x <= 0:
        filtered_count += 1
        continue

    # ⚠️ CROP TO IMAGE BOUNDS - NO SIZE CHECK!
    new_core_x = max(0, core_start_x)
    new_core_y = max(0, core_start_y)
    new_core_w = min(self.w, core_end_x) - new_core_x
    new_core_h = min(self.h, core_end_y) - new_core_y

    # Convert back to image space
    new_x = new_core_x * 8
    new_y = new_core_y * 8
    new_w = new_core_w * 8  # ← Can be 64!
    new_h = new_core_h * 8  # ← Can be 16!

    # Create BBox with potentially tiny dimensions
    bbox = BBox(tile_x, tile_y, tile_size, tile_size)  # line 192
```

✗ **MISSING:** No check for `new_core_w < MIN_SIZE` or `new_core_h < MIN_SIZE`

**File:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`

**Edge Cropping in Visualizer (lines 1309-1332):**
```python
for leaf in leaves:
    # Similar cropping logic - same problem
    new_x = max(0, leaf.x)
    new_y = max(0, leaf.y)
    new_w = min(w, leaf.x + leaf.w) - new_x
    new_h = min(h, leaf.y + leaf.h) - new_y

    # ⚠️ NO SIZE CHECK HERE EITHER!
    cropped_leaf = QuadtreeNode(new_x, new_y, new_w, new_h, leaf.depth)
```

### 3. Minimum Size Parameter Usage

**File:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`

**QuadtreeBuilder (lines 140-157):**
```python
def __init__(self,
             content_threshold: float = 0.03,
             max_depth: int = 4,
             min_tile_size: int = 256,  # ← Default minimum
             min_denoise: float = 0.0,
             max_denoise: float = 1.0):
    self.min_tile_size = min_tile_size
```

**Used in should_subdivide (line 202):**
```python
if half_w_aligned < max(self.min_tile_size, 8):
    return False
```

✓ Correctly prevents subdivision below minimum
✗ But cropping happens AFTER this check!

### 4. Why Edge Nodes Can Be Too Large

**File:** `/home/user/comfyui-quadtree-tile/tiled_diffusion.py`

**Overlap Addition (lines 467-476):**
```python
# Add overlap symmetrically on all sides
x = core_x - overlap
y = core_y - overlap
w = core_w + 2 * overlap  # ← Can make tiles huge!
h = core_h + 2 * overlap

bbox = BBox(x, y, w, h)
```

**Example:**
- Core tile: 2048x2048 (256x256 latent)
- Overlap: 128 pixels (16 latent)
- Result: 2048 + 2*128 = **2304x2304 pixels** (288x288 latent)

This is processed as a single tensor - can cause OOM or freezes!

## Concrete Example: 64x16 Tile

**Input:** 2241x3600 image, quadtree creates 512x512 tile at position (2176, 3584)

**Step 1 - Original tile (image space):**
```
Position: (2176, 3584)
Size: 512 x 512
Valid: ✓ (>= min_tile_size=256)
```

**Step 2 - Convert to latent space (÷8):**
```
Position: (272, 448)
Size: 64 x 64
Image bounds: 280 x 450 latent
```

**Step 3 - Crop to latent bounds:**
```
new_x = max(0, 272) = 272
new_y = max(0, 448) = 448
new_w = min(280, 272+64) - 272 = 280 - 272 = 8  ← Too small!
new_h = min(450, 448+64) - 448 = 450 - 448 = 2  ← Too small!
```

**Step 4 - Convert back to image space (×8):**
```
Position: (2176, 3584)
Size: 64 x 16  ← PROBLEM!
```

**Result:** 8x2 latent tile - far too small for stable diffusion!

## Specific Fix Recommendations

### Fix #1: Enforce Minimum After Cropping (CRITICAL)

**File:** `/home/user/comfyui-quadtree-tile/tiled_diffusion.py`
**Location:** After line 423, inside the loop starting at line 399
**Priority:** HIGH - Prevents degenerate tiles

```python
# After line 423 (after calculating new_w and new_h):

# CRITICAL FIX: Skip tiles that are too small after cropping
MIN_TILE_DIM_LATENT = 16  # 128 pixels in image space (16*8)

if new_core_w < MIN_TILE_DIM_LATENT or new_core_h < MIN_TILE_DIM_LATENT:
    filtered_count += 1
    if new_w < MIN_TILE_DIM_LATENT*8 or new_h < MIN_TILE_DIM_LATENT*8:
        print(f'[Quadtree Diffusion]: Filtered edge tile {new_w}x{new_h}px ' +
              f'(below minimum {MIN_TILE_DIM_LATENT*8}px per dimension)')
    continue
```

**Impact:**
- Prevents 64x16, 128x32, and other degenerate tiles
- Slightly reduces edge coverage (acceptable trade-off)
- Tiles remain >= 128x128 pixels (16x16 latent)

### Fix #2: Same Fix in Visualizer (IMPORTANT)

**File:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`
**Location:** After line 1320, inside the loop starting at line 1309
**Priority:** MEDIUM - Keeps visualizer consistent with diffusion

```python
# After line 1320 (after calculating new_w and new_h):

MIN_TILE_DIM = 128  # pixels

# Skip tiles that are too small after cropping
if new_w < MIN_TILE_DIM or new_h < MIN_TILE_DIM:
    filtered_count += 1
    continue
```

**Impact:**
- Visualizer shows only tiles that will actually be used
- User gets accurate preview of quadtree structure

### Fix #3: Add Warning for Large Tiles (RECOMMENDED)

**File:** `/home/user/comfyui-quadtree-tile/tiled_diffusion.py`
**Location:** After line 476, after overlap is added
**Priority:** MEDIUM - Helps users avoid freezes

```python
# After line 476 (after creating bbox with overlap):

MAX_TILE_DIM_LATENT = 192  # 1536 pixels (reasonable for most GPUs)

if bbox.w > MAX_TILE_DIM_LATENT or bbox.h > MAX_TILE_DIM_LATENT:
    print(f'[Quadtree Diffusion]: ⚠️  Large tile: {bbox.w*8}x{bbox.h*8}px ' +
          f'(latent: {bbox.w}x{bbox.h}). May cause slowness or OOM.')
    print(f'[Quadtree Diffusion]: Consider: reduce max_depth, ' +
          f'increase min_tile_size, or reduce overlap')
```

**Impact:**
- Warns before processing tiles that may freeze
- Gives actionable suggestions
- No behavioral change, just informational

### Fix #4: Update UI Constraints (NICE TO HAVE)

**File:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`
**Location:** Line 1225-1230 (min_tile_size parameter)
**Priority:** LOW - Prevents users from setting bad values

```python
"min_tile_size": ("INT", {
    "default": 256,
    "min": 128,  # Changed from 64 - enforce reasonable minimum
    "max": 1024,
    "step": 8,
    "tooltip": "Minimum tile size in pixels. Must be >= 128 to ensure edge tiles " +
               "remain usable after cropping. Lower = more tiles (slower, more detail). " +
               "Higher = fewer tiles (faster, less adaptive)."
}),
```

**Impact:**
- Prevents users from creating tiny tiles
- Clear guidance on trade-offs
- Better UX overall

## Summary Table

| Issue | Location | Line Numbers | Severity | Fix Priority |
|-------|----------|--------------|----------|--------------|
| No minimum size after crop (diffusion) | tiled_diffusion.py | 399-435 | CRITICAL | HIGH |
| No minimum size after crop (visualizer) | tiled_vae.py | 1309-1332 | MEDIUM | MEDIUM |
| No warning for large tiles | tiled_diffusion.py | 467-476 | MEDIUM | MEDIUM |
| UI allows min_tile_size < 128 | tiled_vae.py | 1225-1230 | LOW | LOW |

## Expected Results After Fixes

**Before:**
```
[Quadtree Visualizer]: Tile dimensions range from 64x16 to 1024x2048
[Quadtree Diffusion]: 5 edge tiles were cropped to fit
```

**After:**
```
[Quadtree Visualizer]: Filtered 5 edge tiles below 128px minimum
[Quadtree Visualizer]: Tile dimensions range from 256x256 to 1024x1024
[Quadtree Diffusion]: Filtered 5 edge tiles below 128px minimum
[Quadtree Diffusion]: Processing 47 tiles (was 52)
[Quadtree Diffusion]: ⚠️  Large tile: 2304x2304px (may cause slowness)
```

## Testing Recommendations

1. **Test with 2241x3600 image** - Should no longer create 64x16 tiles
2. **Check edge coverage** - Verify filtering doesn't leave large gaps
3. **Test with extreme aspect ratios** - 1000x4000, 4000x1000
4. **Monitor memory usage** - Ensure large tile warnings are helpful
5. **Compare visualizer vs diffusion** - Should show same tile count

## Files to Modify

1. `/home/user/comfyui-quadtree-tile/tiled_diffusion.py` (critical)
2. `/home/user/comfyui-quadtree-tile/tiled_vae.py` (important)
