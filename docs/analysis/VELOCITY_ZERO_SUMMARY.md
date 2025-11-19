# Summary: velocity=0 Analysis and Root Cause Identification

**Date**: 2025-11-19
**Status**: 🚨 **CRITICAL BUG IDENTIFIED**
**Related**: VELOCITY_ZERO_ANALYSIS.md, VELOCITY_ZERO_VISUAL.md

---

## Executive Summary

**User Report**: "min_denoise=0 tiles look underdeveloped / noisy"

**Root Cause**: Commit 98e3cbf changed the scaling formula from `start_scale = 0.70 + tile_denoise * 0.25` (range 0.70-0.95) to `start_scale = tile_denoise` (range 0.0-1.0), allowing velocity to become zero.

**The Problem**: The commit message contained a **critical misconception**:
> "min_denoise=0 → scale=0.0 (ZERO changes, complete preservation)"

**The Reality**: velocity=0 does **NOT** preserve the original - it **freezes the tile in its noisy initialization state**.

**Mathematical Proof**: See VELOCITY_ZERO_ANALYSIS.md for complete derivation.

---

## What the Analysis Proves

### 1. velocity=0 Produces Noisy Output, NOT Clean Original

**FLUX Sampler Formula**:
```python
x_next = x_current + dt * velocity
```

**If velocity=0**:
```python
x_next = x_current + dt * 0 = x_current
```

**Result**: Tile stays at initial noisy state forever.

**In img2img with denoise=0.55**:
- Initial state: `x_noisy = 0.45 * x_clean + 0.55 * noise`
- With velocity=0: Tile stays at `x_noisy` (never reaches `x_clean`)
- Final appearance: **Noisy, underdeveloped, hazy** (exactly as users reported!)

### 2. Scaled Velocity v × 0.0 is Identical to velocity=0

```python
velocity_scaled = velocity * 0.0 = 0
x_next = x_current + dt * 0 = x_current
```

**Result**: Same as above - frozen in noisy state.

### 3. Correct Preservation Requires Non-Zero Velocity

To preserve the original clean latent:
```python
velocity_correct = x_clean - x_current  # Points toward target
```

**NOT**:
```python
velocity_wrong = 0  # Stays at current noisy state
```

---

## The Problematic Commit

**Commit**: 98e3cbf0c363889e85d4d0b475dd62df4c9e46cd
**Date**: 2025-11-18
**Title**: "Expand variable denoise range to full 0.0-1.0 scale"

### What Changed

**BEFORE** (working correctly):
```python
start_scale = 0.70 + (tile_denoise * 0.25)  # Range: 0.70-0.95
scale_factor = max(0.70, min(1.0, scale_factor))  # Clamp to [0.70, 1.0]
```

Results:
- min_denoise=0 → start_scale=0.70 → velocity at 70% (still moving)
- max_denoise=1 → start_scale=0.95 → velocity at 95% (strong)
- **No tile ever frozen** (minimum 70% velocity)

**AFTER** (introduced bug):
```python
start_scale = tile_denoise  # Range: 0.0-1.0
scale_factor = max(0.0, min(1.0, scale_factor))  # Clamp to [0.0, 1.0]
```

Results:
- min_denoise=0 → start_scale=0.0 → velocity at 0% (**FROZEN!**)
- max_denoise=1 → start_scale=1.0 → velocity at 100%
- **Large tiles frozen** at noisy state (produces reported issues)

### The Misconception in Commit Message

From commit 98e3cbf:
> "## User Expectation
> - min_denoise=0: Large tiles should get ZERO changes (complete preservation)"

**This is WRONG**. The author confused:
- "Zero changes from **current state**" (what velocity=0 does)
- "Zero changes from **original clean image**" (what user expects)

In img2img:
- Current state = noisy (mixture of clean + noise)
- Original = clean (what user wants preserved)
- velocity=0 preserves **current noisy state**, NOT **original clean state**

---

## Validation: User Reports Match Math

### User Report
> "min_denoise=0 tiles look underdeveloped / noisy"

### Mathematical Prediction
- velocity=0 → tile frozen → stays at `x_noisy = 0.45*x_clean + 0.55*noise`
- Appearance: Noisy, underdeveloped, hazy

### Conclusion
✅ **EXACT MATCH** - User observations perfectly match mathematical predictions.

---

## Why The Old Implementation Was Correct

**Old formula** (0.70-0.95 range):
```python
start_scale = 0.70 + (tile_denoise * 0.25)
```

**Benefits**:
1. ✅ Never allows velocity=0 (minimum 70%)
2. ✅ Tiles always make progress toward clean state
3. ✅ No frozen/noisy appearance
4. ✅ Still provides differentiation (70% vs 95% is visible)

**Trade-off**:
- ⚠️ Less extreme effect (can't get full 0-100% range)
- But this is actually **GOOD** - prevents degenerate behavior

---

## Why The New Implementation Is Broken

**New formula** (0.0-1.0 range):
```python
start_scale = tile_denoise
```

**Problems**:
1. ❌ Allows velocity=0 (min_denoise=0)
2. ❌ Tiles freeze at noisy state
3. ❌ Produces "underdeveloped/noisy" appearance
4. ❌ Violates user expectations (they want clean, not noisy)

**False benefit**:
- "Full 0-100% range for maximum effect visibility"
- But velocity=0 produces **WRONG effect** (noisy instead of preserved)

---

## The Fix

### Option 1: Revert to Old Formula (Recommended)

```python
# Proven to work correctly
start_scale = 0.70 + (tile_denoise * 0.25)  # Range: 0.70-0.95
scale_factor = max(0.70, min(1.0, scale_factor))  # Clamp to [0.70, 1.0]
```

**Rationale**: This was working correctly and didn't cause user complaints.

### Option 2: Compromise (Wider Range, But Not Zero)

```python
# Wider range while avoiding velocity=0
start_scale = 0.30 + (tile_denoise * 0.70)  # Range: 0.30-1.0
scale_factor = max(0.30, min(1.0, scale_factor))  # Clamp to [0.30, 1.0]
```

**Benefits**:
- Provides wider effect range (70% variation instead of 25%)
- Still prevents velocity=0 (minimum 30%)
- Avoids frozen/noisy appearance

### Option 3: Use Skip Feature Instead

For extreme preservation, use the **skip feature** (lines 1274-1349):
```python
# Skip model inference entirely and return correct velocity
if has_original_latent:
    velocity = original_tile - x_in_tile  # Correct preservation
```

**Benefits**:
- ✅ Actually preserves original (correct velocity formula)
- ✅ Faster (no model inference)
- ✅ No approximation artifacts

**Requirements**:
- Only works for img2img (needs original latent)
- Already implemented and working (after velocity bug fix)

---

## Recommended Action

### Immediate Fix (Revert Commit)

```bash
# Revert the problematic commit
git revert 98e3cbf

# Or manually change the formula back to:
start_scale = 0.70 + (tile_denoise * 0.25)  # Range: 0.70-0.95
scale_factor = max(0.70, min(1.0, scale_factor))  # Clamp to [0.70, 1.0]
```

### Update Documentation

**Add to node description**:
> ⚠️ **Variable Denoise Behavior**: This feature scales the strength of denoising predictions, not the number of denoising steps. Setting min_denoise too low may cause tiles to remain partially noisy.
>
> **Recommended ranges**:
> - img2img preservation: min=0.2-0.3, max=0.8-0.9
> - txt2img uniform: min=max=0.7
> - Extreme preservation: Use skip feature instead

**Parameter tooltips**:
- min_denoise: "Minimum denoising for largest tiles (0.2-0.5 recommended, avoid 0.0)"
- max_denoise: "Maximum denoising for smallest tiles (0.7-0.9 recommended)"

### Add Warning

```python
if tile_denoise < 0.2:
    print(f"[Warning] tile_denoise={tile_denoise:.2f} is very low. "
          f"Tile may appear underdeveloped/noisy. Recommended minimum: 0.2")
```

---

## Technical Details

### Full Trace: 20 Steps with velocity=0

See VELOCITY_ZERO_ANALYSIS.md Section 6.2 for complete step-by-step trace showing:
- Initial state: `x[0] = [0.28, 0.63, 0.195, 0.475]` (noisy)
- Steps 1-19: Identical (no movement)
- Final state: `x[20] = [0.28, 0.63, 0.195, 0.475]` (still noisy!)
- Target: `x_clean = [0.5, 0.3, 0.8, 0.2]` (never reached)

### Visual Comparison

See VELOCITY_ZERO_VISUAL.md for diagrams showing:
- Latent space trajectories (normal vs frozen)
- Distance to target over time
- Visual appearance timeline
- Color histogram comparisons

---

## Questions Answered

### Q1: If we return v=0, what happens?

**A**: Tile stays frozen at current noisy state. In img2img, this is `0.45*x_clean + 0.55*noise`, which looks underdeveloped/noisy.

### Q2: Does it converge to anything?

**A**: No. It stays at the initial noisy state forever. No convergence occurs.

### Q3: At the end, is it still noisy?

**A**: Yes, 100%. The final output equals the initial noisy state `x_noisy`.

### Q4: What velocity SHOULD we return to preserve original?

**A**: `velocity = x_clean - x_current` (points toward clean target). This is what the skip feature does correctly.

### Q5: Do we need to know x_clean?

**A**: Yes. To preserve the original, we need access to the clean latent. This is why:
- img2img: Can use skip feature (has `self.original_latent`)
- txt2img: Cannot preserve original (no clean latent exists)

---

## Conclusion

### The Core Issue

Commit 98e3cbf was based on a **fundamental misunderstanding** of what velocity=0 does:

**Incorrect belief** (from commit message):
> "min_denoise=0 → scale=0.0 (ZERO changes, complete preservation)"

**Mathematical reality**:
> "velocity=0 → frozen at noisy state (partial preservation of noisy mixture)"

### The Evidence

1. ✅ **Mathematical proof**: velocity=0 produces `x_final = x_noisy`, not `x_clean`
2. ✅ **User reports**: Match predicted behavior exactly ("underdeveloped/noisy")
3. ✅ **Code history**: Old formula (0.70-0.95) didn't have these issues
4. ✅ **Commit analysis**: Shows the misconception that introduced the bug

### The Solution

**Revert to the old formula** (0.70-0.95 range) or use a **compromise range** (0.30-1.0) that avoids velocity=0 while providing stronger effects.

For extreme preservation needs, **use the skip feature** which implements the correct formula: `velocity = x_clean - x_current`.

---

## Related Documents

- **VELOCITY_ZERO_ANALYSIS.md**: Complete mathematical analysis with proofs
- **VELOCITY_ZERO_VISUAL.md**: Visual diagrams and comparisons
- **SAMPLING_ARCHITECTURE_INVESTIGATION.md**: Background on FLUX velocity prediction
- **SKIP_TILE_VELOCITY_BUG.md**: History of velocity prediction issues

---

**Status**: ✅ Analysis Complete
**Severity**: 🚨 Critical Bug (causes user-visible quality issues)
**Recommendation**: Revert commit 98e3cbf or implement compromise formula
**Timeline**: Should be fixed before next release
