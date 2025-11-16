# Edge Node Size Issue - Visual Example

## Concrete Example: 2241x3600 Image

### Step 1: Square Root Creation

```
Image: 2241 x 3600 pixels
         ↓
max(2241, 3600) = 3600
         ↓
ceil(log2(3600/8)) = ceil(log2(450)) = 9
         ↓
Root size = 8 * 2^9 = 4096 pixels

┌─────────────────────────────────┐
│                                 │  ← 4096 pixels
│  ┌──────────────────┐          │
│  │                  │          │
│  │  Actual Image    │  ← Overhang
│  │  2241 x 3600     │     1855 px (width)
│  │                  │      496 px (height)
│  └──────────────────┘          │
│                                 │
└─────────────────────────────────┘
     Square root: 4096 x 4096
```

### Step 2: Quadtree Subdivision (with min_tile_size=256)

The quadtree subdivides the 4096x4096 root into square tiles:

```
Depth 0: 4096 x 4096 (1 tile)
         ↓ subdivide
Depth 1: 2048 x 2048 (4 tiles)
         ↓ subdivide (high variance areas)
Depth 2: 1024 x 1024 (some tiles)
         ↓ subdivide (very high variance)
Depth 3: 512 x 512   (some tiles)
         ↓ subdivide (extreme variance)
Depth 4: 256 x 256   (some tiles - minimum reached)
```

All tiles are **perfectly square** at this point.

### Step 3: Edge Tiles Extend Beyond Image

Example edge tiles (all square, valid according to min_tile_size=256):

```
Tile A: x=2048, y=3584, w=512, h=512 (bottom-right corner)
Tile B: x=3584, y=2048, w=512, h=512 (right-middle edge)
Tile C: x=0,    y=3584, w=512, h=512 (bottom-left corner)
Tile D: x=2176, y=3584, w=512, h=512 (bottom edge, near corner)
```

Visual representation:

```
          0       2176  2241       3584       4096
          ┌─────────┬────┬──────────┬──────────┐
          │         │    │          │          │
          │         │    │          │          │
          │  Image  │    │   OUT    │   OUT    │
      3600├─────────┼────┤─────┬────┼──────────┤
          │         │ D  │     │ A  │          │
          │   C     │    │ OUT ├────┤   OUT    │
      3584├─────────┴────┴─────┴────┼──────────┤
          │         OUT              │   OUT    │
      4096└──────────────────────────┴──────────┘

Legend:
- C, D, A: Square tiles (512x512) before cropping
- OUT: Regions outside actual image bounds
```

### Step 4: Cropping to Image Bounds (IMAGE SPACE)

**Tile A: (2048, 3584, 512, 512)**
```
Original:
  x=2048, y=3584, w=512, h=512
  x_end=2560, y_end=4096

Crop to image (2241 x 3600):
  new_x = max(0, 2048) = 2048
  new_y = max(0, 3584) = 3584
  new_w = min(2241, 2048+512) - 2048 = min(2241, 2560) - 2048 = 193
  new_h = min(3600, 3584+512) - 3584 = min(3600, 4096) - 3584 = 16

Result: (2048, 3584, 193, 16) ← Very thin!
```

**Tile D: (2176, 3584, 512, 512)**
```
Original:
  x=2176, y=3584, w=512, h=512
  x_end=2688, y_end=4096

Crop to image (2241 x 3600):
  new_x = max(0, 2176) = 2176
  new_y = max(0, 3584) = 3584
  new_w = min(2241, 2176+512) - 2176 = min(2241, 2688) - 2176 = 65
  new_h = min(3600, 3584+512) - 3584 = min(3600, 4096) - 3584 = 16

Result: (2176, 3584, 65, 16) ← PROBLEM: Very small!
```

### Step 5: Conversion to Latent Space (÷8)

**Tile D in latent space:**
```
Image:  (2176, 3584, 65, 16)
          ↓ divide by 8
Latent: (272, 448, 8.125, 2)
          ↓ round/truncate
Latent: (272, 448, 8, 2) ← 8x2 latent pixels!
```

When displayed, log shows: **64x16** (because 8*8=64, 2*8=16)

### Step 6: Adding Overlap Makes It Worse

If overlap = 16 pixels (latent space):

```
Tile D latent: (272, 448, 8, 2)
                ↓ add 2*overlap to each dimension
With overlap: (272-16, 448-16, 8+32, 2+32)
            = (256, 432, 40, 34)
```

The tile extends **beyond the image** in both directions!
- Left: 256 < 0? No, but close
- Right: 256+40 = 296 > 280 (image width in latent)? YES!
- Top: 432 < 0? No
- Bottom: 432+34 = 466 > 450 (image height in latent)? YES!

This requires **padding with reflection/replication**, adding complexity and potential artifacts.

## The Core Problem

```
min_tile_size check ────┐
happens HERE            │
      ↓                 │
                        │
Quadtree                │  ✓ All tiles >= 256x256
Subdivision    ─────────┴─────────────────────────

Edge Cropping           │
happens HERE            │
      ↓                 │  ✗ Some tiles become 64x16!
                        │  (NO size check)
Cropped Tiles  ─────────┴─────────────────────────
```

## Why This Happens

1. **Square root assumption** - Quadtree creates square root larger than rectangular image
2. **Subdivision before cropping** - min_tile_size enforced during subdivision, not after
3. **No post-crop validation** - Cropping can make tiles arbitrarily small
4. **Edge tiles extend beyond** - Square tiles at edges naturally extend past rectangular boundaries

## What Should Happen

```
                        │
Quadtree                │  ✓ All tiles >= 256x256
Subdivision    ─────────┴─────────────────────────

Edge Cropping           │  ✓ Crop to image bounds
                        │
Post-Crop Filter ←──────┼──── ADD THIS CHECK!
                        │  ✓ Skip tiles < 128x128
                        │
Final Tiles    ─────────┴─────────────────────────
```

With minimum dimension check of 128 pixels:
- Tile A (2048, 3584, 193, 16) → **REJECTED** (height=16 < 128)
- Tile D (2176, 3584, 65, 16) → **REJECTED** (both < 128)
- Tile C (0, 3584, 512, 16) → **REJECTED** (height=16 < 128)

Result: No more problematic edge tiles!

## Implementation Location

**File:** `/home/user/comfyui-quadtree-tile/tiled_diffusion.py`
**Function:** `init_quadtree_bbox`
**Lines:** 399-435

Current code (line 415-423):
```python
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

**Add after line 423:**
```python
# CRITICAL FIX: Enforce minimum dimensions after cropping
MIN_TILE_DIM_LATENT = 16  # 128 pixels in image space
if new_core_w < MIN_TILE_DIM_LATENT or new_core_h < MIN_TILE_DIM_LATENT:
    filtered_count += 1
    print(f'[Quadtree Diffusion]: Filtered edge tile {new_w}x{new_h}px ' +
          f'(below {MIN_TILE_DIM_LATENT*8}px minimum)')
    continue
```

This ensures tiles are >= 128x128 pixels (16x16 latent) after cropping.
