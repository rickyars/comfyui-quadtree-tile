# Quadtree Parameter Tuning Guide

## Quick Start: Getting Good Results

**Problem**: Quadtree cuts look random or don't match your image's detail areas?

**Solution**: Start with these settings and adjust from there.

---

## Parameter Overview

### 1. **content_threshold** (MOST IMPORTANT)
**Default**: 0.03
**Range**: 0.001 - 0.5
**What it does**: Controls how "different" a region needs to be before it gets subdivided into smaller tiles.

**Lower value** (0.01 - 0.03):
- ✓ More sensitive to detail
- ✓ More subdivisions in textured areas
- ✗ May subdivide too much (slower)

**Higher value** (0.05 - 0.15):
- ✓ Only subdivides very high-contrast areas
- ✓ Faster (fewer tiles)
- ✗ May miss subtle details

**Recommended starting points**:
- **Portraits/faces**: 0.02 - 0.03 (captures facial features)
- **Landscapes**: 0.03 - 0.05 (balances sky vs detail)
- **Architectural**: 0.02 - 0.04 (captures edges and patterns)
- **Abstract/smooth**: 0.05 - 0.10 (only major features)

---

### 2. **variance_mode** (HOW detail is measured)
**Default**: "combined"
**Options**: color, gradient, combined

**color**:
- Measures color variation (like "how different are the colors in this region?")
- ✓ Fast
- ✗ Can't tell smooth gradient from sharp edge
- ✗ Misses low-contrast texture

**gradient**:
- Measures edges and texture (like "how many edges are in this region?")
- ✓ Great for finding edges, text, patterns
- ✗ Slower
- ✗ May over-subdivide smooth gradients

**combined** (RECOMMENDED):
- Uses both color AND gradient
- ✓ Best of both worlds
- ✓ Detects both color changes AND texture
- ✗ Slightly slower than color-only

**When to use each**:
- **color**: Your image has solid color blocks (cartoons, graphic design)
- **gradient**: Your image has lots of edges (text, wireframes, line art)
- **combined**: Most photos and realistic images

---

### 3. **color_weight** + **gradient_weight**
**Default**: 0.5 + 0.5 (equal balance)
**Range**: 0.0 - 1.0 each

These control the balance when using "combined" mode. They're automatically normalized, so:
- `0.5 + 0.5` = 50% color, 50% gradient
- `1.0 + 1.0` = 50% color, 50% gradient (same thing)
- `1.0 + 0.0` = 100% color, 0% gradient (equivalent to color mode)
- `0.2 + 0.8` = 20% color, 80% gradient

**If your quadtree is...**

**Subdividing smooth gradients too much**:
- Increase color_weight: `0.7 + 0.3`
- This makes it care more about color changes, less about gradients

**Missing textured areas**:
- Increase gradient_weight: `0.3 + 0.7`
- This makes it care more about edges and texture

**Missing both**:
- Lower content_threshold instead

---

### 4. **max_depth**
**Default**: 4
**Range**: 1 - 8

Controls how many times a tile can be subdivided.

**Depth levels** (assuming 256px min tile size):
- Depth 1: Only 2 levels (e.g., 512px → 256px)
- Depth 2: 3 levels (e.g., 1024px → 512px → 256px)
- Depth 3: 4 levels (e.g., 2048px → 1024px → 512px → 256px)
- Depth 4: 5 levels (e.g., 4096px → 2048px → 1024px → 512px → 256px)

**Higher depth**:
- ✓ More granular control
- ✓ Can have very small tiles in detailed areas
- ✗ More tiles = slower generation

**Lower depth**:
- ✓ Fewer tiles = faster
- ✗ Less granular adaptation

**Recommended**:
- **Small images** (<2048px): depth 2-3
- **Medium images** (2048-4096px): depth 3-4
- **Large images** (>4096px): depth 4-5

---

### 5. **min_tile_size**
**Default**: 256
**Range**: 64 - 1024 (must be multiple of 8)

The smallest tile size allowed (in pixels).

**Smaller** (128-256):
- ✓ Can create tiny tiles for very detailed areas
- ✗ Many tiny tiles = much slower generation

**Larger** (512-1024):
- ✓ Fewer, larger tiles = faster
- ✗ Can't adapt to small detailed regions

**Recommended**:
- **High detail needed**: 256px
- **Balanced**: 384px or 512px
- **Speed priority**: 512px or 768px

---

### 6. **min_denoise** / **max_denoise**
**Defaults**: 0.2 / 0.8
**Range**: 0.0 - 1.0

Controls how much each tile gets "regenerated" based on its size:
- **Largest tiles** (low detail areas) use min_denoise
- **Smallest tiles** (high detail areas) use max_denoise

**Denoise scale**:
- **0.0**: Keep original completely
- **0.5**: Mix 50/50 original and generated
- **1.0**: Regenerate completely

**If you want to...**

**Preserve smooth areas, regenerate details**:
- min_denoise: 0.1 (barely touch large tiles)
- max_denoise: 0.9 (heavily regenerate small tiles)

**Regenerate everything evenly**:
- min_denoise: 0.7
- max_denoise: 0.7 (same value = no tile-based variation)

**Only add detail to edges**:
- min_denoise: 0.0 (preserve large tiles completely)
- max_denoise: 0.8 (regenerate detail tiles)

---

## Troubleshooting Common Issues

### "The quadtree cuts look random!"

**Problem**: Tiles don't match your visual perception of detail.

**Likely causes**:
1. **Wrong variance mode**: Using "color" on a photo with textures
   - **Fix**: Switch to "combined" mode

2. **Threshold too high**: Missing subtle details
   - **Fix**: Lower content_threshold from 0.05 to 0.02

3. **Wrong weight balance**: Smooth gradients getting subdivided
   - **Fix**: Increase color_weight to 0.7, decrease gradient_weight to 0.3

---

### "Too many tiles / too slow!"

**Problem**: Quadtree creates hundreds of tiles, generation takes forever.

**Fixes** (try in order):
1. **Increase content_threshold**: 0.03 → 0.05 (fewer subdivisions)
2. **Increase min_tile_size**: 256 → 384 or 512 (larger tiles)
3. **Decrease max_depth**: 4 → 3 (less recursion)
4. **Use color mode**: Faster than combined

---

### "Not enough tiles in detailed areas!"

**Problem**: Faces, text, or complex areas aren't getting subdivided.

**Fixes** (try in order):
1. **Lower content_threshold**: 0.05 → 0.02 (more sensitive)
2. **Switch to combined mode**: Better at detecting detail
3. **Increase gradient_weight**: 0.5 → 0.7 (for edges/texture)
4. **Increase max_depth**: 3 → 4 (allow more subdivision)

---

### "Smooth sky/gradients are getting subdivided!"

**Problem**: Areas that look uniform to you are being split up.

**Fixes**:
1. **Increase color_weight**: 0.5 → 0.7 (care more about color changes)
2. **Decrease gradient_weight**: 0.5 → 0.3 (care less about gradients)
3. **Increase content_threshold**: Make it less sensitive overall

---

## Recommended Presets

### Portrait (Face Details)
```
content_threshold: 0.025
variance_mode: combined
color_weight: 0.4
gradient_weight: 0.6  (emphasize facial features/edges)
max_depth: 4
min_tile_size: 256
```

### Landscape (Sky + Ground)
```
content_threshold: 0.04
variance_mode: combined
color_weight: 0.6
gradient_weight: 0.4  (smooth sky doesn't over-subdivide)
max_depth: 4
min_tile_size: 384
```

### Architectural / Line Art
```
content_threshold: 0.03
variance_mode: gradient  (focus on edges)
color_weight: 0.3
gradient_weight: 0.7
max_depth: 4
min_tile_size: 256
```

### Abstract / Stylized
```
content_threshold: 0.06
variance_mode: color  (fast, focuses on color blocks)
color_weight: 0.8
gradient_weight: 0.2
max_depth: 3
min_tile_size: 512
```

### Maximum Detail (Slow)
```
content_threshold: 0.015  (very sensitive)
variance_mode: combined
color_weight: 0.5
gradient_weight: 0.5
max_depth: 5
min_tile_size: 256
```

### Speed Priority (Fast)
```
content_threshold: 0.08  (less sensitive)
variance_mode: color  (fastest mode)
max_depth: 3
min_tile_size: 512
```

---

## How to Tune Systematically

**Step 1**: Start with the **Portrait** preset above

**Step 2**: Look at the quadtree visualization:
- Do the tiles match where YOU see detail?
- Are smooth areas getting subdivided?
- Are detailed areas being missed?

**Step 3**: Adjust ONE parameter at a time:

**If tiles don't match detail areas**:
→ Try content_threshold ± 0.01

**If smooth gradients are over-subdivided**:
→ Increase color_weight by 0.1

**If textures are missed**:
→ Increase gradient_weight by 0.1

**If too many/few tiles overall**:
→ Adjust content_threshold (higher = fewer tiles)

**Step 4**: Test on a few different images of the same type

**Step 5**: Save your settings as a preset!

---

## Technical Details

### How variance is calculated:

**Color variance** (MAD - Mean Absolute Deviation):
1. Calculate average color of the region
2. Measure how far each pixel is from that average
3. Higher value = more color variation

**Gradient variance** (Sobel edge detection):
1. Apply horizontal and vertical edge filters
2. Calculate edge strength: √(horizontal² + vertical²)
3. Average across the region
4. Higher value = more edges/texture

**Combined variance**:
```
combined = (color_weight × color_variance) + (gradient_weight × gradient_variance)
```

Then compared to `content_threshold`:
- If `combined >= threshold`: SUBDIVIDE (this area is complex)
- If `combined < threshold`: KEEP (this area is simple)

---

## Need More Help?

**Check the visualizer output console**:
```
[Quadtree Visualizer]: Built quadtree with 47 tiles (after filtering)
[Quadtree Visualizer]: Tile dimensions range from 256x256 to 1024x1024
```

This tells you:
- **Number of tiles**: Too many? Raise threshold. Too few? Lower threshold.
- **Size range**: All big? Lower threshold or increase depth. All small? Raise threshold.

---

**Still not working?** Share:
1. Your image type (portrait, landscape, etc.)
2. Current parameter values
3. What's wrong (too many tiles, wrong areas subdivided, etc.)
