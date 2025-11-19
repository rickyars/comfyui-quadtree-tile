# Implementation Guide: Square Quadtree Tiles

**File:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`

This guide provides exact code changes to implement square tiles with full coverage.

---

## Change 1: Add actual dimensions to QuadtreeNode

**Location:** Lines 86-96 (QuadtreeNode.__init__)

**Current Code:**
```python
class QuadtreeNode:
    """Represents a single node in the quadtree structure"""
    def __init__(self, x: int, y: int, w: int, h: int, depth: int = 0):
        self.x = x  # Top-left x coordinate
        self.y = y  # Top-left y coordinate
        self.w = w  # Width
        self.h = h  # Height
        self.depth = depth  # Depth in tree (0 = root)
        self.variance = 0.0  # Content complexity metric
        self.denoise = 0.0  # Denoise value for diffusion
        self.children = []  # Child nodes (empty if leaf)
```

**New Code:**
```python
class QuadtreeNode:
    """Represents a single node in the quadtree structure"""
    def __init__(self, x: int, y: int, w: int, h: int, depth: int = 0):
        self.x = x  # Top-left x coordinate
        self.y = y  # Top-left y coordinate
        self.w = w  # Width
        self.h = h  # Height
        self.depth = depth  # Depth in tree (0 = root)
        self.variance = 0.0  # Content complexity metric
        self.denoise = 0.0  # Denoise value for diffusion
        self.children = []  # Child nodes (empty if leaf)
        # Store actual image dimensions (for root node only)
        self.actual_image_w = None  # Actual image width (may be < w for square root)
        self.actual_image_h = None  # Actual image height (may be < h for square root)
```

---

## Change 2: Modify subdivide() to create square children

**Location:** Lines 102-129

**Current Code:**
```python
def subdivide(self):
    """Subdivide this node into 4 children with 8-pixel alignment for VAE compatibility

    Strategy:
    - Always create exactly 4 children (quadtree property)
    - Rectangular parents → 4 rectangular children
    - Children naturally become smaller as we subdivide deeper
    - Stop at leaf level (min_tile_size or max_depth)
    """
    # Ensure subdivisions are aligned to 8-pixel boundaries for VAE encoder/decoder
    # VAE downsamples by 8x, so tiles must be divisible by 8
    half_w = (self.w // 2) // 8 * 8  # Round down to nearest multiple of 8
    half_h = (self.h // 2) // 8 * 8

    # Ensure we have at least 8 pixels
    half_w = max(half_w, 8)
    half_h = max(half_h, 8)

    # QUADTREE PROPERTY: Always create exactly 4 children
    # Children inherit parent's proportions and become smaller as we subdivide deeper

    self.children = [
        QuadtreeNode(self.x, self.y, half_w, half_h, self.depth + 1),  # Top-left
        QuadtreeNode(self.x + half_w, self.y, self.w - half_w, half_h, self.depth + 1),  # Top-right
        QuadtreeNode(self.x, self.y + half_h, half_w, self.h - half_h, self.depth + 1),  # Bottom-left
        QuadtreeNode(self.x + half_w, self.y + half_h, self.w - half_w, self.h - half_h, self.depth + 1),  # Bottom-right
    ]
```

**New Code:**
```python
def subdivide(self):
    """Subdivide this node into 4 SQUARE children with 8-pixel alignment

    Strategy for SQUARE TILES:
    - Parent must be square (w == h) - enforced by square root
    - Split into 4 equal square quadrants
    - All children are square (size = parent_size / 2)
    - Maintains 8-pixel alignment for VAE compatibility
    """
    # Sanity check: verify parent is square
    if self.w != self.h:
        raise ValueError(
            f"subdivide() called on non-square node: {self.w}×{self.h}. "
            f"Root node must be square for square tile generation."
        )

    # Calculate half size with 8-pixel alignment
    # Since w == h, we can use either dimension
    half_size = (self.w // 2) // 8 * 8  # Round down to nearest multiple of 8
    half_size = max(half_size, 8)  # Ensure minimum 8 pixels

    # QUADTREE PROPERTY: Create exactly 4 SQUARE children
    # All children have same dimensions: half_size × half_size
    self.children = [
        QuadtreeNode(self.x, self.y, half_size, half_size, self.depth + 1),  # Top-left
        QuadtreeNode(self.x + half_size, self.y, half_size, half_size, self.depth + 1),  # Top-right
        QuadtreeNode(self.x, self.y + half_size, half_size, half_size, self.depth + 1),  # Bottom-left
        QuadtreeNode(self.x + half_size, self.y + half_size, half_size, half_size, self.depth + 1),  # Bottom-right
    ]

    # Propagate actual image dimensions to children (if set on parent)
    if self.actual_image_w is not None:
        for child in self.children:
            child.actual_image_w = self.actual_image_w
            child.actual_image_h = self.actual_image_h
```

---

## Change 3: Create square root node in build_tree()

**Location:** Lines 227-256 (QuadtreeBuilder.build_tree method)

**Current Code (lines 231-244):**
```python
# Create root node if not provided
if root_node is None:
    if tensor.dim() == 4:
        _, _, h, w = tensor.shape
    else:
        _, h, w = tensor.shape

    # Ensure root dimensions are aligned to 8-pixel boundaries
    # This is critical for VAE encoder/decoder compatibility
    w_aligned = (w // 8) * 8
    h_aligned = (h // 8) * 8

    # Root node matches the actual image dimensions (can be rectangular)
    root_node = QuadtreeNode(0, 0, w_aligned, h_aligned, 0)
```

**New Code:**
```python
# Create root node if not provided
if root_node is None:
    if tensor.dim() == 4:
        _, _, h, w = tensor.shape
    else:
        _, h, w = tensor.shape

    # Ensure dimensions are aligned to 8-pixel boundaries
    w_aligned = (w // 8) * 8
    h_aligned = (h // 8) * 8

    # CRITICAL CHANGE: Create SQUARE root node using maximum dimension
    # This ensures all subdivisions create square children
    root_size = max(w_aligned, h_aligned)
    root_node = QuadtreeNode(0, 0, root_size, root_size, 0)

    # Store actual image dimensions for boundary handling
    root_node.actual_image_w = w_aligned
    root_node.actual_image_h = h_aligned

    print(f'[Quadtree VAE]: Square root created: {root_size}×{root_size} for image {w_aligned}×{h_aligned}')
    if root_size > max(w_aligned, h_aligned):
        overhang_w = root_size - w_aligned
        overhang_h = root_size - h_aligned
        print(f'[Quadtree VAE]: Overhang: +{overhang_w}px width, +{overhang_h}px height (will use padding)')
```

---

## Change 4: Update calculate_variance() to handle boundaries

**Location:** Lines 161-192 (QuadtreeBuilder.calculate_variance method)

**Current Code (lines 173-191):**
```python
# Handle both batched and unbatched tensors
if tensor.dim() == 4:
    region = tensor[:, :, y:y+h, x:x+w]
else:
    region = tensor[:, y:y+h, x:x+w]

# Ensure region has valid size
if region.numel() == 0:
    return 0.0

# Calculate average color for this region
avg_color = torch.mean(region, dim=(-2, -1), keepdim=True)

# Calculate mean absolute deviation
# This is "the average distance between each pixel's color and the region's average color"
# matching the reference implementation
deviations = torch.abs(region - avg_color)
mean_absolute_deviation = torch.mean(deviations).item()

return mean_absolute_deviation
```

**New Code:**
```python
# Get actual image dimensions
if tensor.dim() == 4:
    img_h, img_w = tensor.shape[2], tensor.shape[3]
else:
    img_h, img_w = tensor.shape[1], tensor.shape[2]

# Clip region to actual image bounds
# (node may extend beyond image for square root)
x1 = max(0, x)
y1 = max(0, y)
x2 = min(x + w, img_w)
y2 = min(y + h, img_h)

# Check if node is entirely outside image bounds
if x1 >= img_w or y1 >= img_h or x2 <= x1 or y2 <= y1:
    return 0.0  # Node outside image - no variance

# Extract region (only the part within image bounds)
if tensor.dim() == 4:
    region = tensor[:, :, y1:y2, x1:x2]
else:
    region = tensor[:, y1:y2, x1:x2]

# Ensure region has valid size
if region.numel() == 0:
    return 0.0

# Calculate average color for this region
avg_color = torch.mean(region, dim=(-2, -1), keepdim=True)

# Calculate mean absolute deviation
deviations = torch.abs(region - avg_color)
mean_absolute_deviation = torch.mean(deviations).item()

return mean_absolute_deviation
```

---

## Change 5: Update should_subdivide() to handle boundaries

**Location:** Lines 194-218 (QuadtreeBuilder.should_subdivide method)

**Current Code (lines 205-218):**
```python
# Don't subdivide if at max depth
if node.depth >= self.max_depth:
    return False

# Don't subdivide if tile would be too small
# Also ensure child tiles would be at least 8 pixels (VAE requirement)
half_w_aligned = ((node.w // 2) // 8) * 8
half_h_aligned = ((node.h // 2) // 8) * 8

if half_w_aligned < max(self.min_tile_size, 8) or half_h_aligned < max(self.min_tile_size, 8):
    return False

# Subdivide if variance is above threshold
return variance > self.content_threshold
```

**New Code:**
```python
# Don't subdivide if at max depth
if node.depth >= self.max_depth:
    return False

# For square nodes, calculate child size
# Since w == h, we only need to check one dimension
half_size_aligned = ((node.w // 2) // 8) * 8

# Don't subdivide if child tiles would be too small
if half_size_aligned < max(self.min_tile_size, 8):
    return False

# Don't subdivide if node is entirely outside actual image bounds
# (only relevant for boundary nodes in square root quadtrees)
if node.actual_image_w is not None:
    if node.x >= node.actual_image_w or node.y >= node.actual_image_h:
        return False  # Entirely outside image

# Subdivide if variance is above threshold
return variance > self.content_threshold
```

---

## Change 6: Add padding helper function

**Location:** Add new helper function after QuadtreeBuilder class (around line 325)

**Add this new function:**
```python
def pad_tile_to_square(tile: torch.Tensor, target_size: int, pad_mode: str = 'reflect') -> torch.Tensor:
    """
    Pad a tile to square dimensions using reflection padding

    Args:
        tile: Input tile tensor (B, C, H, W) or (C, H, W)
        target_size: Target square size (both width and height)
        pad_mode: Padding mode ('reflect', 'replicate', or 'constant')

    Returns:
        Padded square tile of size target_size × target_size
    """
    if tile.dim() == 4:
        _, _, h, w = tile.shape
    else:
        _, h, w = tile.shape

    # Calculate padding needed
    pad_right = target_size - w
    pad_bottom = target_size - h

    if pad_right < 0 or pad_bottom < 0:
        raise ValueError(f"Tile {w}×{h} is larger than target size {target_size}")

    if pad_right == 0 and pad_bottom == 0:
        return tile  # Already square

    # Apply padding (left, right, top, bottom)
    # PyTorch pad order: (left, right, top, bottom) for 2D
    padded_tile = F.pad(tile, (0, pad_right, 0, pad_bottom), mode=pad_mode)

    return padded_tile


def crop_padded_result(result: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
    """
    Crop padded result back to actual image dimensions

    Args:
        result: Padded result tensor (B, C, H, W)
        target_w: Target width (actual image width)
        target_h: Target height (actual image height)

    Returns:
        Cropped tensor of size target_h × target_w
    """
    if result.dim() == 4:
        return result[:, :, :target_h, :target_w]
    else:
        return result[:, :target_h, :target_w]
```

---

## Change 7: Update split_tiles_quadtree() to handle padding

**Location:** Lines 803-880 (VAEHook.split_tiles_quadtree method)

**Find this section (around line 833-861):**
```python
for leaf in leaves:
    # Quadtree leaves are in the current tensor's space (h x w)
    # For decoder: latent space. For encoder: image space.
    x1, x2, y1, y2 = leaf.x, leaf.x + leaf.w, leaf.y, leaf.y + leaf.h

    # Define the core tile region (without padding) - this is what we want in the output
    core_bbox = [x1, x2, y1, y2]

    # Create output bbox - handle image borders by extending to edge
    output_bbox = [
        core_bbox[0] if core_bbox[0] > pad else 0,
        core_bbox[1] if core_bbox[1] < w - pad else w,
        core_bbox[2] if core_bbox[2] > pad else 0,
        core_bbox[3] if core_bbox[3] < h - pad else h,
    ]

    # Scale output bbox for the target space (decoder: latent→image 8x, encoder: image→latent /8)
    output_bbox = [x * 8 if self.is_decoder else x // 8 for x in output_bbox]
    tile_output_bboxes.append(output_bbox)

    # Expand core bbox by padding to create input bbox
    # This is the region we'll actually process (larger due to padding)
    input_bbox_padded = [
        max(0, core_bbox[0] - pad),
        min(w, core_bbox[1] + pad),
        max(0, core_bbox[2] - pad),
        min(h, core_bbox[3] + pad),
    ]
    tile_input_bboxes.append(input_bbox_padded)
```

**Replace with:**
```python
for leaf in leaves:
    # Check if tile extends beyond actual image bounds
    actual_w = leaf.actual_image_w if leaf.actual_image_w is not None else w
    actual_h = leaf.actual_image_h if leaf.actual_image_h is not None else h

    x1, x2, y1, y2 = leaf.x, leaf.x + leaf.w, leaf.y, leaf.y + leaf.h

    # Skip tiles entirely outside image bounds
    if x1 >= actual_w or y1 >= actual_h:
        continue

    # Clip tile to actual image bounds for output
    x2_clipped = min(x2, actual_w)
    y2_clipped = min(y2, actual_h)

    # Define the core tile region (clipped to actual image)
    core_bbox = [x1, x2_clipped, y1, y2_clipped]

    # Create output bbox - handle image borders
    output_bbox = [
        core_bbox[0] if core_bbox[0] > pad else 0,
        core_bbox[1] if core_bbox[1] < actual_w - pad else actual_w,
        core_bbox[2] if core_bbox[2] > pad else 0,
        core_bbox[3] if core_bbox[3] < actual_h - pad else actual_h,
    ]

    # Scale output bbox for the target space
    output_bbox = [x * 8 if self.is_decoder else x // 8 for x in output_bbox]
    tile_output_bboxes.append(output_bbox)

    # Expand core bbox by padding to create input bbox
    # Clip to actual image bounds
    input_bbox_padded = [
        max(0, x1 - pad),
        min(actual_w, x2_clipped + pad),
        max(0, y1 - pad),
        min(actual_h, y2_clipped + pad),
    ]
    tile_input_bboxes.append(input_bbox_padded)

    # Store whether this tile needs padding to maintain square shape
    needs_padding_w = (x2_clipped - x1) < leaf.w
    needs_padding_h = (y2_clipped - y1) < leaf.h

    if needs_padding_w or needs_padding_h:
        print(f'[Quadtree VAE]: Tile at ({x1},{y1}) needs padding: w={needs_padding_w}, h={needs_padding_h}')
```

---

## Change 8: Add tile padding in vae_tile_forward()

**Location:** Lines 927-1086 (VAEHook.vae_tile_forward method)

**Find the section where tiles are prepared (around line 954-956):**
```python
# Prepare tiles by split the input latents
tiles = []
for input_bbox in in_bboxes:
    tile = z[:, :, input_bbox[2]:input_bbox[3], input_bbox[0]:input_bbox[1]].cpu()
    tiles.append(tile)
```

**Replace with:**
```python
# Prepare tiles by split the input latents
tiles = []
for i, input_bbox in enumerate(in_bboxes):
    # Extract tile from image
    tile = z[:, :, input_bbox[2]:input_bbox[3], input_bbox[0]:input_bbox[1]]

    # If using quadtree, check if tile needs padding to maintain square shape
    if self.use_quadtree and self.quadtree_leaves is not None:
        leaf = self.quadtree_leaves[i]
        tile_h = input_bbox[3] - input_bbox[2]
        tile_w = input_bbox[1] - input_bbox[0]

        # Pad to square if needed (leaf.w == leaf.h, but extracted tile may be clipped)
        if tile_w < leaf.w or tile_h < leaf.h:
            target_size = leaf.w  # Square size
            tile = pad_tile_to_square(tile, target_size, pad_mode='reflect')
            print(f'[Quadtree VAE]: Padded tile {i} from {tile_w}×{tile_h} to {target_size}×{target_size}')

    tiles.append(tile.cpu())
```

---

## Validation Code

Add this test function to verify square tiles:

```python
def validate_square_quadtree(root: QuadtreeNode, leaves: list) -> bool:
    """
    Validate that quadtree has square tiles

    Args:
        root: Root node
        leaves: Leaf nodes

    Returns:
        True if all tiles are square, False otherwise
    """
    print("[Quadtree Validation]")

    # Check root is square
    if root.w != root.h:
        print(f"  ✗ Root is NOT square: {root.w}×{root.h}")
        return False
    print(f"  ✓ Root is square: {root.w}×{root.h}")

    # Check all leaves are square
    non_square_leaves = []
    for i, leaf in enumerate(leaves):
        if leaf.w != leaf.h:
            non_square_leaves.append((i, leaf.w, leaf.h))

    if non_square_leaves:
        print(f"  ✗ Found {len(non_square_leaves)} non-square leaves:")
        for i, w, h in non_square_leaves[:5]:  # Show first 5
            print(f"    Leaf {i}: {w}×{h}")
        return False

    print(f"  ✓ All {len(leaves)} leaves are square")

    # Check coverage
    if root.actual_image_w and root.actual_image_h:
        print(f"  Image dimensions: {root.actual_image_w}×{root.actual_image_h}")
        print(f"  Quadtree space: {root.w}×{root.h}")
        overhang_w = root.w - root.actual_image_w
        overhang_h = root.h - root.actual_image_h
        print(f"  Overhang: {overhang_w}px width, {overhang_h}px height")

    return True
```

---

## Testing Checklist

After implementing changes:

```python
# Test 1: Verify square root creation
# Expected: Root is max(1920, 1080) = 1920×1920
image_1920x1080 = torch.randn(1, 3, 1080, 1920)
builder = QuadtreeBuilder()
root, leaves = builder.build(image_1920x1080)
assert root.w == 1920 and root.h == 1920, "Root should be 1920×1920"
assert root.actual_image_w == 1920 and root.actual_image_h == 1080

# Test 2: Verify all leaves are square
for leaf in leaves:
    assert leaf.w == leaf.h, f"Leaf should be square, got {leaf.w}×{leaf.h}"

# Test 3: Verify coverage
# (use validate_square_quadtree function)
assert validate_square_quadtree(root, leaves)

# Test 4: Test portrait image
image_512x768 = torch.randn(1, 3, 768, 512)
root, leaves = builder.build(image_512x768)
assert root.w == 768 and root.h == 768, "Root should be 768×768"
for leaf in leaves:
    assert leaf.w == leaf.h, f"Leaf should be square"

# Test 5: Test square image (no padding needed)
image_1024x1024 = torch.randn(1, 3, 1024, 1024)
root, leaves = builder.build(image_1024x1024)
assert root.w == 1024 and root.h == 1024
for leaf in leaves:
    assert leaf.w == leaf.h
```

---

## Summary of Changes

| Change | File | Lines | Impact |
|--------|------|-------|--------|
| 1. Add actual dimensions | tiled_vae.py | ~96 | Store real image size |
| 2. Square subdivide() | tiled_vae.py | 102-129 | Force square children |
| 3. Square root node | tiled_vae.py | 231-244 | Create square root |
| 4. Boundary variance | tiled_vae.py | 173-191 | Handle edge tiles |
| 5. Boundary subdivision | tiled_vae.py | 205-218 | Skip out-of-bounds |
| 6. Padding helpers | tiled_vae.py | ~325 | New functions |
| 7. Split with padding | tiled_vae.py | 833-861 | Handle boundaries |
| 8. Tile padding | tiled_vae.py | 954-956 | Pad to square |
| 9. Validation | tiled_vae.py | New | Test function |

**Total Lines Changed:** ~150
**New Lines Added:** ~100
**Complexity:** Moderate
**Risk:** Low (all changes localized to quadtree logic)
