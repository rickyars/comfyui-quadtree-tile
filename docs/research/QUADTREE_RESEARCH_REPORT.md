# Quadtree Square Tiles Research Report

**Date:** 2025-11-13
**Objective:** Create a proper quadtree with SQUARE tiles providing FULL COVERAGE for rectangular images

---

## Executive Summary

After analyzing the codebase and researching standard quadtree algorithms, I've identified the root cause of rectangular tiles and propose a **Square Root + Edge Padding** solution that guarantees:
- ✓ All tiles are square (except potentially smallest boundary tiles if needed)
- ✓ 100% image coverage (no gaps/black output)
- ✓ Maintains true quadtree structure (4 children per node)
- ✓ Works with VAE 8-pixel alignment requirements

---

## Part 1: Root Cause Analysis

### Current Implementation Behavior

**File:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`
**Method:** `QuadtreeNode.subdivide()` (lines 102-129)

```python
def subdivide(self):
    # Calculate half dimensions with 8-pixel alignment
    half_w = (self.w // 2) // 8 * 8
    half_h = (self.h // 2) // 8 * 8

    # Create 4 children by splitting width and height independently
    self.children = [
        QuadtreeNode(self.x, self.y, half_w, half_h, ...),                      # Top-left
        QuadtreeNode(self.x + half_w, self.y, self.w - half_w, half_h, ...),    # Top-right
        QuadtreeNode(self.x, self.y + half_h, half_w, self.h - half_h, ...),    # Bottom-left
        QuadtreeNode(self.x + half_w, self.y + half_h,
                     self.w - half_w, self.h - half_h, ...),                    # Bottom-right
    ]
```

### Why Tiles Are Rectangular

**Root Cause:** The subdivision splits width and height *independently*, so rectangular parents produce rectangular children.

**Example: 1920×1080 image**
```
Root: 1920×1080 (rectangular)
  ├─ half_w = 960, half_h = 536
  ├─ Child 1 (TL): 960×536 (rectangular)
  ├─ Child 2 (TR): 960×536 (rectangular)
  ├─ Child 3 (BL): 960×544 (rectangular)
  └─ Child 4 (BR): 960×544 (rectangular)

This propagates down the entire tree → all leaves are rectangular
```

### Previous Fix Attempt (commit b6ea610)

**Approach:** Used `half_size = min(half_w, half_h)` to force square children

**Why It Failed:**
```python
# For 1920×1080:
half_w = 960, half_h = 536
half_size = min(960, 536) = 536  # Force square

# 4 children each 536×536
# Coverage: 1072×1072 (2×536)
# Image size: 1920×1080
# GAPS: 848 pixels horizontally! → Black output
```

**Critical Issue:** Shrinking tiles to squares created massive coverage gaps

---

## Part 2: How Real Quadtrees Handle Rectangular Images

### Research Findings

From academic literature and standard implementations, there are 4 main approaches:

#### **Approach A: Square Root (Padding/Expansion)** ⭐ RECOMMENDED
- **Method:** Root is max(W, H) × max(W, H), aligned to requirements
- **Subdivision:** Standard quadtree (each child is half parent size, all square)
- **Edge handling:** Tiles extending beyond image are clipped OR image is padded
- **Coverage:** 100% guaranteed
- **Tiles:** All square (or clipped at boundaries)
- **Used by:** MATLAB qtdecomp, most image compression algorithms

**Example:**
```
1920×1080 image → Root: 1920×1920
Subdivision creates perfect squares
Tiles beyond y=1080 are handled via padding/clipping
```

#### **Approach B: Allow Rectangular Tiles**
- **Method:** Accept that children inherit parent aspect ratio
- **Coverage:** 100% guaranteed
- **Tiles:** Rectangular for rectangular images
- **Issue:** ✗ Violates "all tiles must be square" requirement

#### **Approach C: Multiple Square Regions**
- **Method:** Divide rectangular image into multiple square zones
- **Complexity:** High - need to merge results, handle boundaries
- **Coverage:** 100% but complex

#### **Approach D: Largest Square + Gap Tiles**
- **Method:** Build quadtree for largest inscribed square, add strip tiles for remainder
- **Issue:** Remainder tiles break quadtree structure

---

## Part 3: Proposed Solution - Square Root with Edge Padding

### Strategy Overview

**Core Principle:** Build a square quadtree in a conceptual square space that encloses the image

### Implementation Steps

#### **Step 1: Create Square Root Node**

Modify `QuadtreeBuilder.build_tree()` (lines 220-259):

```python
# OLD CODE (line 243):
root_node = QuadtreeNode(0, 0, w_aligned, h_aligned, 0)

# NEW CODE:
# Create square root: size = max dimension
root_size = max(w_aligned, h_aligned)
root_node = QuadtreeNode(0, 0, root_size, root_size, 0)
root_node.actual_image_w = w_aligned  # Store actual dimensions
root_node.actual_image_h = h_aligned
```

#### **Step 2: Modify Subdivision to Always Create Squares**

Modify `QuadtreeNode.subdivide()` (lines 102-129):

```python
def subdivide(self):
    """Subdivide into 4 SQUARE children"""

    # CRITICAL CHANGE: Use same dimension for both width and height
    # This ensures all children are square
    half_size = (self.w // 2) // 8 * 8  # Assuming w == h for square nodes
    half_size = max(half_size, 8)  # Minimum 8 pixels

    # Create 4 SQUARE children
    self.children = [
        QuadtreeNode(self.x, self.y, half_size, half_size, self.depth + 1),
        QuadtreeNode(self.x + half_size, self.y, half_size, half_size, self.depth + 1),
        QuadtreeNode(self.x, self.y + half_size, half_size, half_size, self.depth + 1),
        QuadtreeNode(self.x + half_size, self.y + half_size, half_size, half_size, self.depth + 1),
    ]
```

**Note:** Since root is now square (w == h), all children will be square, and this propagates down the tree.

#### **Step 3: Handle Edge Tiles (Beyond Image Bounds)**

Three options for tiles extending beyond actual image dimensions:

##### **Option 3A: Clipping (Simple)** ⚠️
```python
# When extracting tile from image, clip to actual bounds
tile_x2 = min(node.x + node.w, actual_image_w)
tile_y2 = min(node.y + node.h, actual_image_h)
tile_w = tile_x2 - node.x
tile_h = tile_y2 - node.y
# Result: boundary tiles may be rectangular
```
**Pros:** Simple, no data modification
**Cons:** Boundary tiles are rectangular (violates requirement)

##### **Option 3B: Reflection Padding** ⭐ RECOMMENDED
```python
# When tile extends beyond image, pad using reflection
if node.y + node.h > actual_image_h:
    # Tile extends below image
    overhang = (node.y + node.h) - actual_image_h
    # Pad by reflecting the bottom edge upward
    tile = F.pad(tile, (0, 0, 0, overhang), mode='reflect')
# Result: all tiles remain square
```
**Pros:** All tiles square, smooth blending, common in image processing
**Cons:** Slight additional computation

##### **Option 3C: Skip Out-of-Bounds Tiles**
```python
# Don't process tiles that are mostly outside image bounds
if node.x >= actual_image_w or node.y >= actual_image_h:
    return  # Skip this tile entirely
```
**Pros:** Efficient
**Cons:** May leave small gaps at edges

#### **Step 4: Update Variance Calculation**

Modify `QuadtreeBuilder.calculate_variance()` (lines 161-192):

```python
def calculate_variance(self, tensor, node):
    """Calculate variance, clipping to actual image bounds"""
    # Clip region to actual image dimensions
    x1 = node.x
    y1 = node.y
    x2 = min(node.x + node.w, tensor.shape[-1])  # Actual image width
    y2 = min(node.y + node.h, tensor.shape[-2])  # Actual image height

    # Extract region (may be smaller than node.w × node.h at edges)
    if tensor.dim() == 4:
        region = tensor[:, :, y1:y2, x1:x2]
    else:
        region = tensor[:, y1:y2, x1:x2]

    if region.numel() == 0:
        return 0.0  # Outside image bounds

    # Calculate variance on actual image content
    avg_color = torch.mean(region, dim=(-2, -1), keepdim=True)
    deviations = torch.abs(region - avg_color)
    return torch.mean(deviations).item()
```

---

## Part 4: Complete Implementation Plan

### File: `/home/user/comfyui-quadtree-tile/tiled_vae.py`

#### **Change 1: Update `build_tree()` method** (lines 220-259)

**Location:** Line ~243
**Current:**
```python
root_node = QuadtreeNode(0, 0, w_aligned, h_aligned, 0)
```

**Replace with:**
```python
# Create SQUARE root node using maximum dimension
root_size = max(w_aligned, h_aligned)
root_node = QuadtreeNode(0, 0, root_size, root_size, 0)

# Store actual image dimensions for boundary handling
root_node.actual_w = w_aligned
root_node.actual_h = h_aligned
```

#### **Change 2: Update `subdivide()` method** (lines 102-129)

**Replace entire method with:**
```python
def subdivide(self):
    """Subdivide this node into 4 SQUARE children with 8-pixel alignment

    Strategy for SQUARE TILES:
    - Parent is always square (w == h)
    - Split into 4 equal square quadrants
    - All children are square (w/2 × h/2 = square)
    """
    # Ensure subdivisions are aligned to 8-pixel boundaries
    half_size = (self.w // 2) // 8 * 8  # Since w == h, we can use either
    half_size = max(half_size, 8)  # Ensure minimum 8 pixels

    # Sanity check: parent should be square
    if self.w != self.h:
        raise ValueError(f"subdivide() called on non-square node: {self.w}×{self.h}")

    # QUADTREE PROPERTY: Create exactly 4 SQUARE children
    self.children = [
        QuadtreeNode(self.x, self.y, half_size, half_size, self.depth + 1),  # Top-left
        QuadtreeNode(self.x + half_size, self.y, half_size, half_size, self.depth + 1),  # Top-right
        QuadtreeNode(self.x, self.y + half_size, half_size, half_size, self.depth + 1),  # Bottom-left
        QuadtreeNode(self.x + half_size, self.y + half_size, half_size, half_size, self.depth + 1),  # Bottom-right
    ]
```

#### **Change 3: Update `calculate_variance()` method** (lines 161-192)

**Modify to handle nodes extending beyond image bounds:**

**Location:** Line ~174-177
**Current:**
```python
if tensor.dim() == 4:
    region = tensor[:, :, y:y+h, x:x+w]
else:
    region = tensor[:, y:y+h, x:x+w]
```

**Replace with:**
```python
# Get actual image dimensions
if tensor.dim() == 4:
    img_h, img_w = tensor.shape[2], tensor.shape[3]
else:
    img_h, img_w = tensor.shape[1], tensor.shape[2]

# Clip region to actual image bounds
x1 = x
y1 = y
x2 = min(x + w, img_w)
y2 = min(y + h, img_h)

# Extract region
if tensor.dim() == 4:
    region = tensor[:, :, y1:y2, x1:x2]
else:
    region = tensor[:, y1:y2, x1:x2]
```

#### **Change 4: Update `split_tiles_quadtree()` method** (lines 803-880)

**Add tile padding/clipping logic:**

**Location:** After line ~836, before creating input/output bboxes
**Add:**
```python
for leaf in leaves:
    # Check if tile extends beyond actual image bounds
    extends_beyond_w = (leaf.x + leaf.w) > w
    extends_beyond_h = (leaf.y + leaf.h) > h

    if not extends_beyond_w and not extends_beyond_h:
        # Tile is fully within image - standard processing
        # ... existing bbox code ...
    else:
        # Tile extends beyond image - handle with padding/clipping
        # ... add padding logic or mark for special handling ...
```

---

## Part 5: Test Cases and Expected Results

### Test Case 1: 1920×1080 (16:9 rectangular)

**Setup:**
- Image: 1920×1080
- Max depth: 3
- Min tile size: 256

**Expected Behavior:**

```
Quadtree Structure:
  Root: 1920×1920 (square) [extends beyond image by 840 pixels vertically]
    ├─ Depth 1: 4×(960×960) squares
    ├─ Depth 2: 16×(480×480) squares (if variance triggers)
    └─ Depth 3: 64×(240×240) squares (if variance triggers)

Coverage:
  - X-axis: 0-1920 (full coverage ✓)
  - Y-axis: 0-1080 (full coverage ✓)
  - Tiles below y=1080: Use reflection padding or skip

Tile Counts (example with moderate variance):
  - Fully within image: ~40 tiles
  - Partially beyond (y > 1080): ~8 tiles (need padding/clipping)
  - Total: ~48 tiles
  - All tiles: SQUARE ✓
```

### Test Case 2: 512×768 (2:3 vertical)

**Setup:**
- Image: 512×768
- Max depth: 4
- Min tile size: 128

**Expected Behavior:**

```
Quadtree Structure:
  Root: 768×768 (square) [extends beyond image by 256 pixels horizontally]
    └─ All children are square subdivisions

Coverage:
  - X-axis: 0-512 (full coverage ✓)
  - Y-axis: 0-768 (full coverage ✓)
  - Tiles beyond x=512: Use reflection padding or skip

Tile Counts:
  - Fully within: ~25 tiles
  - Partially beyond (x > 512): ~8 tiles
  - All tiles: SQUARE ✓
```

### Test Case 3: 1024×1024 (square)

**Setup:**
- Image: 1024×1024
- Max depth: 4
- Min tile size: 128

**Expected Behavior:**

```
Quadtree Structure:
  Root: 1024×1024 (square) [perfect fit]
    └─ All subdivisions within bounds

Coverage:
  - 100% coverage, no edge cases ✓
  - No padding/clipping needed ✓

Tile Counts (depends on variance):
  - Depth 1: up to 4 tiles (512×512)
  - Depth 2: up to 16 tiles (256×256)
  - Depth 3: up to 64 tiles (128×128)
  - All tiles: SQUARE ✓
```

---

## Part 6: Coverage Verification Algorithm

To ensure 100% coverage with no gaps:

```python
def verify_coverage(leaves, image_w, image_h):
    """Verify that tiles provide complete coverage"""
    import numpy as np

    # Create coverage map
    coverage = np.zeros((image_h, image_w), dtype=bool)

    # Mark covered pixels
    for leaf in leaves:
        x1 = max(0, leaf.x)
        y1 = max(0, leaf.y)
        x2 = min(image_w, leaf.x + leaf.w)
        y2 = min(image_h, leaf.y + leaf.h)

        coverage[y1:y2, x1:x2] = True

    # Check for gaps
    uncovered_pixels = np.sum(~coverage)
    coverage_pct = 100 * np.sum(coverage) / (image_w * image_h)

    print(f"Coverage: {coverage_pct:.2f}%")
    print(f"Uncovered pixels: {uncovered_pixels}")

    if uncovered_pixels > 0:
        print("WARNING: Gaps detected!")
        return False

    return True
```

---

## Part 7: Alternative Approaches Considered (and why rejected)

### Alternative 1: Adaptive Square Packing
**Idea:** Use different-sized non-overlapping squares to tile the rectangle
**Issue:** Breaks quadtree structure (not exactly 4 children per node)
**Verdict:** ✗ Too complex, loses quadtree benefits

### Alternative 2: Overlapping Square Tiles
**Idea:** Shift boundary tiles inward to keep them square but create overlaps
**Issue:** Overlaps cause double-processing, complex blending required
**Verdict:** ✗ Adds complexity, unclear how to merge overlapping results

### Alternative 3: Allow Rectangular Leaves Only at Boundaries
**Idea:** Keep quadtree square internally, only clip final leaves
**Issue:** Some tiles would be rectangular (violates requirement)
**Verdict:** ~ Acceptable compromise if padding is undesirable

### Alternative 4: Resample Image to Square
**Idea:** Resize image to square before processing
**Issue:** Distorts aspect ratio, quality loss
**Verdict:** ✗ Unacceptable quality impact

---

## Part 8: Recommended Implementation Priority

### Phase 1: Core Changes (REQUIRED)
1. ✅ Modify `build_tree()` to create square root node
2. ✅ Update `subdivide()` to create square children
3. ✅ Update `calculate_variance()` to handle boundary clipping

### Phase 2: Edge Handling (REQUIRED)
4. ✅ Implement reflection padding for boundary tiles in `split_tiles_quadtree()`
5. ✅ Update tile extraction logic in `vae_tile_forward()`

### Phase 3: Validation (RECOMMENDED)
6. ✅ Add coverage verification function
7. ✅ Add debug logging for tile dimensions
8. ✅ Test with 1920×1080, 512×768, 1024×1024 images

### Phase 4: Optimization (OPTIONAL)
9. ⚪ Cache padding computations
10. ⚪ Skip tiles that are entirely outside image bounds
11. ⚪ Optimize reflection padding for performance

---

## Part 9: Expected Outcomes

### Benefits
- ✅ **All tiles are square** (except possibly smallest boundary tiles if clipping)
- ✅ **100% coverage** guaranteed (no black output)
- ✅ **True quadtree structure** maintained (4 children per node)
- ✅ **Content-adaptive** (variance-based subdivision still works)
- ✅ **VAE-compatible** (8-pixel alignment preserved)

### Trade-offs
- **Slight computation overhead:** Boundary tiles need padding/reflection (~10-15% of tiles for 16:9 images)
- **Memory:** Square root node may be larger than image (e.g., 1920×1920 vs 1920×1080 = +78% conceptual space)
- **Complexity:** Need careful handling of tiles extending beyond actual image

### Performance Impact
- **Minimal:** Padding operations are fast (reflection/replication)
- **Memory efficient:** Only process tiles that overlap actual image
- **Parallelizable:** Independent tile processing unchanged

---

## Part 10: Validation Checklist

Before deploying:

- [ ] All subdivisions create square children (w == h for all nodes)
- [ ] Root node size is max(image_w, image_h)
- [ ] Variance calculation handles boundary nodes correctly
- [ ] Coverage verification shows 100% for test images
- [ ] No NaN/black output on 1920×1080 test image
- [ ] Tile dimensions logged confirm all are square
- [ ] Visual inspection shows no gaps in processed image
- [ ] Performance acceptable (< 10% overhead vs rectangular tiles)

---

## Conclusion

**The RECOMMENDED SOLUTION is:**

**Square Root with Reflection Padding (Approach A + Option 3B)**

This approach:
1. Creates a square quadtree in conceptual square space
2. All nodes (including leaves) are perfect squares
3. Boundary tiles extending beyond image use reflection padding to maintain squareness
4. Guarantees 100% coverage with no gaps
5. Maintains true quadtree properties
6. Minimal complexity and performance overhead

**Next Step:** Implement the changes outlined in Part 4 (Implementation Plan)
