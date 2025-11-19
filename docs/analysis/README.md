# Analysis Documents Index

This directory contains detailed technical analyses of the ComfyUI Quadtree Tiled Diffusion codebase.

---

## 🚨 Critical Issue: velocity=0 in FLUX img2img

**User Report**: "min_denoise=0 tiles look underdeveloped / noisy"

**Status**: Bug identified in commit 98e3cbf

### Read These Documents (In Order)

1. **[VELOCITY_ZERO_SUMMARY.md](VELOCITY_ZERO_SUMMARY.md)** - Start here!
   - Executive summary of the issue
   - Root cause identification
   - Quick fix recommendations
   - **5-minute read**

2. **[VELOCITY_ZERO_EXAMPLE.md](VELOCITY_ZERO_EXAMPLE.md)** - Simple explanation
   - Concrete numerical example with real pixel values
   - Shows exactly what happens step-by-step
   - For non-technical users
   - **10-minute read**

3. **[VELOCITY_ZERO_ANALYSIS.md](VELOCITY_ZERO_ANALYSIS.md)** - Deep dive
   - Complete mathematical analysis
   - Formal proofs
   - Answers all technical questions
   - **30-minute read**

4. **[VELOCITY_ZERO_VISUAL.md](VELOCITY_ZERO_VISUAL.md)** - Visual aids
   - Diagrams and charts
   - Side-by-side comparisons
   - Visual timeline progressions
   - **15-minute read**

---

## Quick Summary

### The Problem

When `min_denoise=0`, the code sets `velocity = 0`, which causes tiles to **freeze in their noisy initialization state** instead of denoising properly.

### Why It Happens

Commit 98e3cbf changed the scaling formula from:
```python
scale = 0.70 + denoise × 0.25  # Range: 70%-95% ✓ Working
```

To:
```python
scale = denoise  # Range: 0%-100% ✗ Broken
```

This allows velocity to become zero, which freezes tiles.

### What Users See

- Large tiles (min_denoise=0) look **underdeveloped**
- Appearance is **noisy, hazy, wrong colors**
- Looks like image "not fully rendered"

### The Fix

Revert to old formula or use compromise:
```python
scale = 0.30 + denoise × 0.70  # Range: 30%-100% (compromise)
```

Or use **skip feature** for true preservation (img2img only).

---

## Key Findings

### ✅ Validated

1. **velocity=0 does NOT preserve original** - it freezes at noisy state
2. **User reports are 100% accurate** - math predicts exact behavior
3. **Old formula (0.70-0.95) was correct** - prevented this issue
4. **New formula (0.0-1.0) introduced bug** - allows velocity=0

### ❌ Common Misconception

**Wrong belief** (from commit 98e3cbf):
> "min_denoise=0 → velocity=0 → no changes → preserves original"

**Reality**:
> "velocity=0 → frozen at noisy state → does NOT preserve original"

The confusion:
- Original clean image: What user wants preserved
- Current noisy state: What velocity=0 actually preserves
- These are **different** in img2img!

---

## Document Structure

```
docs/analysis/
├── README.md (this file)
│   └── Index and quick reference
│
├── VELOCITY_ZERO_SUMMARY.md
│   ├── Executive summary
│   ├── Root cause analysis
│   └── Fix recommendations
│
├── VELOCITY_ZERO_EXAMPLE.md
│   ├── Simple numerical example
│   ├── Real pixel values
│   └── Step-by-step trace
│
├── VELOCITY_ZERO_ANALYSIS.md
│   ├── Mathematical proofs
│   ├── FLUX sampling theory
│   ├── Complete derivations
│   └── Technical deep dive
│
└── VELOCITY_ZERO_VISUAL.md
    ├── Trajectory diagrams
    ├── Timeline charts
    ├── Comparison tables
    └── Visual aids
```

---

## Related Background Documents

Located in `docs/architecture/` and `docs/reports/`:

### Sampling Architecture
- **[SAMPLING_ARCHITECTURE_INVESTIGATION.md](../architecture/SAMPLING_ARCHITECTURE_INVESTIGATION.md)**
  - How ComfyUI sampling works
  - Why per-tile timesteps are impossible
  - Variable denoise implementation details

### Variable Denoise Feature
- **[VARIABLE_DENOISE_ANALYSIS.md](../reports/VARIABLE_DENOISE_ANALYSIS.md)**
  - Feature implementation analysis
  - Edge cases and conditions
  - How scaling works

### Velocity Prediction Issues
- **[SKIP_TILE_VELOCITY_BUG.md](../bug-analysis/SKIP_TILE_VELOCITY_BUG.md)**
  - Previous velocity sign error
  - FLUX vs SD prediction types
  - Film negative bug fix

---

## For Developers

### Quick Debugging Guide

**If user reports "tiles look noisy"**:

1. Check `min_denoise` value
   - If `< 0.3`: Likely velocity too close to zero
   - Recommend: Use 0.3-0.5 instead

2. Check current scaling formula
   - Location: `tiled_diffusion.py:1451`
   - Should be: `start_scale = 0.70 + (tile_denoise * 0.25)`
   - NOT: `start_scale = tile_denoise`

3. Check for warnings in console
   - "No sigmas in store" → Feature disabled
   - "Variable denoise will NOT work" → Fix sigma loading

### Testing velocity=0 Hypothesis

```python
# In tiled_diffusion.py, add debug logging:
if scale_factor < 0.1:
    print(f"WARNING: scale_factor={scale_factor:.3f} is very low!")
    print(f"  tile_denoise={tile_denoise:.3f}")
    print(f"  This tile may appear noisy/underdeveloped")
```

Then test with:
- min_denoise=0.0, max_denoise=1.0
- Check console for warnings on large tiles
- Verify if large tiles look noisy (they should with current code)

---

## Mathematical Quick Reference

### FLUX Sampler Update

```
x_next = x_current + dt × velocity
```

### With velocity=0

```
x_next = x_current + dt × 0 = x_current  (frozen!)
```

### With scaled velocity

```
v_scaled = velocity × scale_factor
x_next = x_current + dt × (velocity × scale_factor)
       = x_current + (dt × scale_factor) × velocity

If scale_factor = 0:
  x_next = x_current + 0 = x_current  (frozen!)
```

### Correct preservation (skip feature)

```
velocity = x_clean - x_current  (points toward target)
x_next = x_current + dt × (x_clean - x_current)
       = (1 - dt) × x_current + dt × x_clean
       → Interpolates toward x_clean ✓
```

---

## Timeline of Issue

1. **Before commit 98e3cbf**: Formula `0.70 + denoise × 0.25` works correctly
2. **Commit 98e3cbf** (2025-11-18): Changed to `denoise` range (0.0-1.0)
3. **User reports**: "min_denoise=0 tiles look underdeveloped/noisy"
4. **Analysis** (2025-11-19): Identified velocity=0 as root cause
5. **Status**: **Bug confirmed, fix needed**

---

## Recommendations

### Immediate Action
- [ ] Revert commit 98e3cbf or implement compromise formula
- [ ] Update documentation to warn about low min_denoise values
- [ ] Add console warning when scale_factor < 0.2

### Long-term Improvements
- [ ] Consider deprecating variable denoise in favor of skip feature
- [ ] Add UI tooltips explaining behavior
- [ ] Expose scaling parameters for advanced users
- [ ] Test with various samplers and document compatibility

---

## Questions?

- **"Why does this only affect FLUX?"** - It doesn't! All velocity/noise-based models are affected. FLUX just makes it more visible because it uses velocity prediction.

- **"Can't we just use velocity=0 to preserve original?"** - No! In img2img, the "current state" is noisy, not clean. velocity=0 preserves the noisy state.

- **"What about txt2img?"** - In txt2img, there is no "original" to preserve. velocity=0 leaves tiles at their random noise initialization (appears gray).

- **"How is skip feature different?"** - Skip feature returns `velocity = x_clean - x_current`, which actually points toward the clean original. velocity=0 does not.

---

## Document History

- **2025-11-19**: Initial analysis documents created
  - Identified root cause in commit 98e3cbf
  - Mathematical proof of velocity=0 behavior
  - Validated against user reports

---

**For more details, start with [VELOCITY_ZERO_SUMMARY.md](VELOCITY_ZERO_SUMMARY.md)**
