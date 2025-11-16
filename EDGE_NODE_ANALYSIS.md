# Edge Node Size Analysis

## Problem Summary

Edge nodes in the quadtree tiling implementation can become either:
1. **Too small** - dimensions like 64x16 pixels (8x2 in latent space)
2. **Too large** - dimensions up to 2048x2048 + overlap, causing ComfyUI freezes

For a 2241x3600 image:
- Log shows: "Tile dimensions range from 64x16 to 1024x2048"
- 5 edge tiles were cropped to fit
- Some tiles are problematic for diffusion models

## Root Cause Analysis

### 1. Square Root Creation (tiled_vae.py:236-246)

**Location:** `/home/user/comfyui-quadtree-tile/tiled_vae.py` lines 236-246

```python
# Create square root covering entire image (power-of-2 multiple of 8)
root_size = max(w, h)

if root_size <= 8:
    root_size = 8
else:
    n = math.ceil(math.log2(root_size / 8))
    root_size = 8 * (2 ** n)

root_node = QuadtreeNode(0, 0, root_size, root_size, 0)
```

**Issue:** For a 2241x3600 image:
- `max(2241, 3600) = 3600`
- `n = ceil(log2(3600/8)) = ceil(log2(450)) = 9`
- `root_size = 8 * 2^9 = 4096 pixels`

This creates a **4096x4096 square root** for a **2241x3600 rectangular image**, resulting in:
- 1855 pixels of horizontal overhang (4096 - 2241)
- 496 pixels of vertical overhang (4096 - 3600)

### 2. Subdivision Logic (tiled_vae.py:102-128, 192-217)

**Location:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`

**`QuadtreeNode.subdivide()` (lines 102-128):**
```python
def subdivide(self):
    # Ensure subdivisions are aligned to 8-pixel boundaries for VAE
    half_w = (self.w // 2) // 8 * 8  # Round down to nearest multiple of 8
    half_h = (self.h // 2) // 8 * 8

    half_w = max(half_w, 8)
    half_h = max(half_h, 8)

    # Always create exactly 4 children
    self.children = [
        QuadtreeNode(self.x, self.y, half_w, half_h, self.depth + 1),  # Top-left
        QuadtreeNode(self.x + half_w, self.y, self.w - half_w, half_h, self.depth + 1),  # Top-right
        QuadtreeNode(self.x, self.y + half_h, half_w, self.h - half_h, self.depth + 1),  # Bottom-left
        QuadtreeNode(self.x + half_w, self.y + half_h, self.w - half_w, self.h - half_h, self.depth + 1),  # Bottom-right
    ]
```

**`QuadtreeBuilder.should_subdivide()` (lines 192-217):**
```python
def should_subdivide(self, node: QuadtreeNode, variance: float) -> bool:
    # Don't subdivide if at max depth
    if node.depth >= self.max_depth:
        return False

    # Don't subdivide if tile would be too small
    half_w_aligned = ((node.w // 2) // 8) * 8
    half_h_aligned = ((node.h // 2) // 8) * 8

    if half_w_aligned < max(self.min_tile_size, 8) or half_h_aligned < max(self.min_tile_size, 8):
        return False

    # Check 8-pixel alignment for BOTH halves and remainders
    remainder_w = node.w - half_w_aligned
    remainder_h = node.h - half_h_aligned

    # All children must be >=8 and multiples of 8
    if (half_w_aligned < 8 or half_h_aligned < 8 or
        remainder_w < 8 or remainder_h < 8 or
        remainder_w % 8 != 0 or remainder_h % 8 != 0):
        return False

    # Subdivide if variance is above threshold
    return variance > self.content_threshold
```

**Issue:** The `min_tile_size` check happens **BEFORE** edge cropping. Once the quadtree is built with valid square tiles, they are later cropped to fit the rectangular image, bypassing the minimum size constraint.

### 3. Edge Tile Cropping in Diffusion (tiled_diffusion.py:390-440)

**Location:** `/home/user/comfyui-quadtree-tile/tiled_diffusion.py` lines 390-440

```python
for leaf in leaves:
    # Calculate CORE bounds (without overlap) in latent space
    core_start_x = leaf.x // 8
    core_start_y = leaf.y // 8
    core_end_x = (leaf.x + leaf.w) // 8
    core_end_y = (leaf.y + leaf.h) // 8

    # Check if core overlaps with latent image at all
    if core_start_x >= self.w or core_end_x <= 0 or core_start_y >= self.h or core_end_y <= 0:
        # Completely outside - skip it
        filtered_count += 1
        continue

    # Core overlaps - crop to latent image bounds
    new_core_x = max(0, core_start_x)
    new_core_y = max(0, core_start_y)
    new_core_w = min(self.w, core_end_x) - new_core_x
    new_core_h = min(self.h, core_end_y) - new_core_y

    # Convert back to image space for storage
    new_x = new_core_x * 8
    new_y = new_core_y * 8
    new_w = new_core_w * 8
    new_h = new_core_h * 8
```

**Issue:** This cropping can create **arbitrarily small tiles** with no minimum size enforcement.

**Example Calculation:**

For image 2241x3600 (latent 280x450):
- Suppose quadtree creates a 512x512 pixel tile at position (2176, 3584)
- In latent space: position (272, 448), size (64, 64)

After cropping to latent bounds (280x450):
- `new_core_x = max(0, 272) = 272`
- `new_core_y = max(0, 448) = 448`
- `new_core_w = min(280, 272+64) - 272 = 280 - 272 = 8`
- `new_core_h = min(450, 448+64) - 448 = 450 - 448 = 2`

**Result:** Tile becomes (272*8, 448*8, 8*8, 2*8) = **(2176, 3584, 64, 16)** in image space!

This is **8x2 in latent space** - far too small for stable diffusion processing.

### 4. Edge Tile Cropping in Visualizer (tiled_vae.py:1301-1337)

**Location:** `/home/user/comfyui-quadtree-tile/tiled_vae.py` lines 1301-1337

Similar cropping logic in the QuadtreeVisualizer node:

```python
for leaf in leaves:
    # Check if tile overlaps with image at all
    if leaf.x >= w or (leaf.x + leaf.w) <= 0 or leaf.y >= h or (leaf.y + leaf.h) <= 0:
        # Completely outside - skip it
        filtered_count += 1
        continue

    # Tile overlaps - crop to image bounds
    new_x = max(0, leaf.x)
    new_y = max(0, leaf.y)
    new_w = min(w, leaf.x + leaf.w) - new_x
    new_h = min(h, leaf.y + leaf.h) - new_y
```

**Issue:** Same problem - no minimum size enforcement after cropping.

### 5. Why Tiles Can Be Too Large

**Location:** `/home/user/comfyui-quadtree-tile/tiled_diffusion.py` lines 467-476

When overlap is added to tiles:

```python
# Add overlap symmetrically on all sides
x = core_x - overlap
y = core_y - overlap
w = core_w + 2 * overlap
h = core_h + 2 * overlap

bbox = BBox(x, y, w, h)
```

**Issue:** For large tiles:
- A 2048x2048 tile with 128 pixel overlap becomes 2304x2304 in latent space (288x288)
- A 1024x1024 tile with 128 pixel overlap becomes 1280x1280 in latent space (160x160)
- These are processed as single tensors and can cause OOM or freezes

The diffusion model must process the entire tile at once, and tiles this large can exceed available VRAM or cause extremely slow processing.

## Impact Analysis

### Small Tiles (e.g., 64x16)

**Problems:**
1. **Insufficient context** - Diffusion models need adequate spatial context
2. **Poor quality** - Very thin tiles don't provide enough information for coherent generation
3. **Numerical instability** - Gaussian weights may underflow for very small dimensions
4. **Wasted compute** - Small tiles have high overhead relative to useful area

### Large Tiles (e.g., 2048x2048 + overlap)

**Problems:**
1. **Memory issues** - Can exceed VRAM, causing OOM errors or system freezes
2. **Slow processing** - Large tiles take exponentially longer to process
3. **Timeout risks** - ComfyUI may freeze or timeout on very large tiles

## Parameter Configuration at Issue Time

Based on the log output for 2241x3600 image:
- `min_tile_size`: 256 pixels (default from QuadtreeVisualizer)
- Tiles range from 64x16 to 1024x2048
- 5 edge tiles were cropped
- Latent dimensions: approximately 280x450

## Specific Recommendations

### Fix 1: Enforce Minimum Dimensions After Cropping (CRITICAL)

**File:** `/home/user/comfyui-quadtree-tile/tiled_diffusion.py`
**Lines:** 399-435 (in `init_quadtree_bbox`)

Add minimum dimension check after cropping:

```python
# After calculating new_core_w and new_core_h (line 416)
MIN_TILE_DIM_LATENT = 16  # 128 pixels in image space (16*8)

# Skip tiles that are too small after cropping
if new_core_w < MIN_TILE_DIM_LATENT or new_core_h < MIN_TILE_DIM_LATENT:
    filtered_count += 1
    print(f'[Quadtree Diffusion]: Skipping too-small edge tile: {new_core_w*8}x{new_core_h*8}px ' +
          f'(below minimum {MIN_TILE_DIM_LATENT*8}px per dimension)')
    continue
```

**Rationale:**
- 128 pixels (16 latent) is a reasonable minimum for diffusion
- Prevents degenerate tiles like 64x16
- Slightly reduces coverage at edges, but these tiny tiles aren't useful anyway

### Fix 2: Enforce Minimum Dimensions in Visualizer

**File:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`
**Lines:** 1309-1332 (in QuadtreeVisualizer.visualize)

Add the same check in the visualizer:

```python
# After calculating new_w and new_h (line 1320)
MIN_TILE_DIM = 128  # pixels

# Skip tiles that are too small after cropping
if new_w < MIN_TILE_DIM or new_h < MIN_TILE_DIM:
    filtered_count += 1
    continue
```

**Rationale:**
- Keeps visualizer and diffusion consistent
- Prevents user from seeing tiles that will be skipped during diffusion

### Fix 3: Add Maximum Tile Dimension Check (IMPORTANT)

**File:** `/home/user/comfyui-quadtree-tile/tiled_diffusion.py`
**Lines:** After line 476 (after overlap is added)

Add maximum dimension check to prevent freezes:

```python
# After creating bbox with overlap (line 476)
MAX_TILE_DIM_LATENT = 192  # 1536 pixels in image space (192*8)

# Warn about very large tiles
if bbox.w > MAX_TILE_DIM_LATENT or bbox.h > MAX_TILE_DIM_LATENT:
    print(f'[Quadtree Diffusion]: ⚠️  Very large tile detected: {bbox.w*8}x{bbox.h*8}px ' +
          f'(latent: {bbox.w}x{bbox.h}). This may cause slowness or OOM.')
    print(f'[Quadtree Diffusion]: Consider: reduce max_depth, increase min_tile_size, or reduce overlap')

    # Optional: Clamp tile size to maximum (commented out by default)
    # This would require adjusting overlap or core region
    # bbox.w = min(bbox.w, MAX_TILE_DIM_LATENT)
    # bbox.h = min(bbox.h, MAX_TILE_DIM_LATENT)
```

**Rationale:**
- 1536 pixels (192 latent) is reasonable for most GPUs
- Warning gives user actionable feedback
- Optional clamping can prevent freezes (but may affect coverage)

### Fix 4: Add Validation in QuadtreeVisualizer UI

**File:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`
**Lines:** 1207-1254 (INPUT_TYPES)

Add helpful tooltips and constraints:

```python
"min_tile_size": ("INT", {
    "default": 256,
    "min": 128,  # Changed from 64 - enforce reasonable minimum
    "max": 1024,
    "step": 8,
    "tooltip": "Minimum tile size in pixels. Must be >= 128 to ensure edge tiles remain usable after cropping. Lower = more tiles (slower, more detail). Higher = fewer tiles (faster, less adaptive)."
}),
```

**Rationale:**
- Prevents users from setting `min_tile_size` too low
- Clear guidance on trade-offs
- 128 minimum ensures even heavily cropped tiles are >= 64 pixels

### Fix 5: Consider Rectangular Root (OPTIONAL - MAJOR CHANGE)

**File:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`
**Lines:** 236-246 (in QuadtreeBuilder.build_tree)

Alternative: Create rectangular root that better matches image aspect ratio:

```python
# Instead of square root, create rectangular root that's closer to image size
def calculate_root_size(dimension):
    """Calculate power-of-2 multiple of 8 for a dimension"""
    if dimension <= 8:
        return 8
    n = math.ceil(math.log2(dimension / 8))
    return 8 * (2 ** n)

root_w = calculate_root_size(w)
root_h = calculate_root_size(h)
root_node = QuadtreeNode(0, 0, root_w, root_h, 0)
```

**For 2241x3600:**
- `root_w = calculate_root_size(2241) = 2048` (overhang: 0, actually clips 193 pixels)
- Wait, that's wrong. Let me recalculate:
  - `n_w = ceil(log2(2241/8)) = ceil(log2(280.125)) = 9`
  - `root_w = 8 * 2^9 = 4096`
- `root_h = calculate_root_size(3600) = 4096` (same as before)

Actually, this doesn't help much. The issue is that power-of-2 sizing creates large steps.

**Better approach:** Use nearest power-of-2 that's >= image dimension:

```python
def next_power_of_2_multiple_of_8(dimension):
    """Find smallest power-of-2 multiple of 8 that's >= dimension"""
    if dimension <= 8:
        return 8
    # Find next power of 2
    power = 1
    while power < dimension:
        power *= 2
    # Ensure it's a multiple of 8
    return max(8, (power // 8) * 8)

root_w = next_power_of_2_multiple_of_8(w)
root_h = next_power_of_2_multiple_of_8(h)
```

For 2241x3600:
- `root_w = next_power_of_2(2241) = 4096`
- `root_h = next_power_of_2(3600) = 4096`

Still the same! The issue is inherent to power-of-2 sizing with odd dimensions.

**Conclusion:** Rectangular root doesn't help much. Better to enforce minimum dimensions after cropping (Fixes 1-2).

## Summary

**Critical Issue:** Edge tile cropping creates tiles smaller than `min_tile_size` because the minimum size check happens before cropping, not after.

**Primary Solution:** Enforce minimum dimensions (128 pixels / 16 latent) after cropping in both:
1. `init_quadtree_bbox` (diffusion)
2. `QuadtreeVisualizer.visualize` (visualizer)

**Secondary Solutions:**
3. Add warnings for very large tiles
4. Update UI constraints and tooltips
5. Consider rectangular root (optional, limited benefit)

**Expected Impact:**
- No more 64x16 tiles - minimum 128x128 after cropping
- Slightly reduced edge coverage (acceptable trade-off)
- Better warnings for large tiles
- More consistent behavior between visualizer and diffusion
