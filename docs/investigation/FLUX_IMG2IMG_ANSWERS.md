# FLUX img2img Workflow - Direct Answers

**Investigation Date**: 2025-11-19

---

## Critical Questions - Answered

### 1. What does the latent contain at each step in img2img?

#### At Start (Before Sampling)
```
Input Image → VAE Encode → clean_latent (stored as store.latent_image)
                          ↓
                      Add noise based on denoise parameter
                          ↓
              x_initial = noisy starting point

Examples:
- denoise = 1.0: x_initial = pure noise (t=0, start from scratch)
- denoise = 0.5: x_initial = 50% clean + 50% noise (t=0.5, moderate changes)
- denoise = 0.3: x_initial = 70% clean + 30% noise (t=0.7, subtle changes)
```

**Formula**: `X_t = t * X_clean + (1-t) * X_noise`

#### During Denoising
```
Step 1:  x_in = 60% noise + 40% clean  (high noise)
         ↓ model predicts velocity toward clean
Step 2:  x_in = 40% noise + 60% clean  (medium noise)
         ↓ model predicts velocity toward clean
Step 3:  x_in = 20% noise + 80% clean  (low noise)
         ↓ model predicts velocity toward clean
Final:   x_in = 0% noise + 100% clean  (fully denoised)
```

**State**: `x_in` is always a NOISY latent that progressively becomes cleaner.

#### What the Model Predicts
```
Model receives: x_in (current noisy state at time t)
Model predicts: velocity v ≈ (target_clean - x_in)
                → Direction pointing FROM noisy TO clean
```

---

### 2. What does FLUX velocity prediction mean in img2img context?

#### The Mathematical Definition

FLUX uses **Rectified Flow** with linear interpolation:
```
X_t = t * X_1 + (1-t) * X_0

Where:
  X_0 = pure Gaussian noise
  X_1 = clean target image (in latent space)
  t ∈ [0, 1] = time parameter
  X_t = noisy state at time t

Velocity:
  v = dX_t/dt = X_1 - X_0
    = clean_target - noise
    = direction from noise to clean
```

#### What the Model Learns

**Training**: Model learns to predict the velocity field `v(X_t, t)` that points from any noisy state toward the clean target.

**Prediction**: `v_θ(X_t, t) ≈ E[X_1 - X_0 | X_t]`
- "Given current noisy state X_t, what's the direction toward clean?"

#### In img2img Specifically

```
img2img workflow:
  Input: Original image → VAE encode → X_original (clean)
  Start: X_initial = noisy version of X_original (based on denoise param)

During sampling:
  x_in = current noisy state
  v_model = model's predicted velocity ≈ (X_target - x_in)

Key point:
  X_target = what the model THINKS should be there (based on prompt)
  X_original = what WAS originally there (the input image)

These can be DIFFERENT!
```

#### Example

```
Original image: Blue car
Prompt: "Make it red"

Model predicts:
  v = red_car_clean - x_in_noisy
  → Points toward RED car (the prompted target)

NOT:
  v = blue_car_clean - x_in_noisy
  → Would point toward BLUE car (the original)
```

#### How It Differs from txt2img

**txt2img**:
```
- Start: Pure random noise (no original image)
- Target: Generated from prompt only
- Velocity: noise → generated_target
```

**img2img**:
```
- Start: Noisy version of input image
- Target: Modified based on prompt + input image structure
- Velocity: noisy_input → modified_target
- Original clean input is AVAILABLE but NOT the prediction target
```

---

### 3. How to achieve "no changes" to original in img2img?

#### What Users Expect with min_denoise=0

"Large tiles should be COMPLETELY unchanged - identical to original input image"

#### What Actually Happens with velocity=0 (WRONG!)

**Current implementation** (commit 98e3cbf):
```python
scale_factor = 0.0  # when min_denoise=0
tile_out = model_prediction * 0.0 = 0
```

**Sampler receives**:
```
v = 0
x_next = x_in + v * dt = x_in + 0 = x_in

Result: Tile STAYS AT CURRENT NOISY STATE
        → Never denoises
        → Looks "underdeveloped" / "super faint" / "too much noise"
```

**Why this is wrong**:
- x_in is NOISY at every step (until final convergence)
- velocity=0 means "don't move" = stay noisy
- This is NOT preservation, it's "freezing in noisy state"

#### What Velocity We Should Return (CORRECT)

**For true preservation**:
```python
velocity = original_clean_latent - x_in_current_noisy

Sampler applies:
x_next = x_in + (original - x_in) * dt
       = x_in * (1-dt) + original * dt
       → Interpolates toward original clean latent
       → Converges to original after multiple steps ✓
```

**This is what the skip feature does**:
```python
# tiled_diffusion.py:1320
model_prediction = original_tile - x_in_tile  # ✓ CORRECT
```

#### Do We Need Access to Original Clean Latent?

**YES, for true preservation!**

```
Without original_clean:
  Can only scale model's prediction
  Model predicts: v → prompt_target (not original)
  Scaling to 0: v = 0 → stays noisy ❌

With original_clean:
  Can compute preservation velocity
  v = original - x_in → toward original
  Result: converges to original ✓
```

#### Why velocity=0 Doesn't Work

**Fundamental misunderstanding**:
```
❌ WRONG: velocity=0 → no changes → preservation
✓ RIGHT: velocity=0 → no movement → stays noisy
```

**What actually happens**:
```
Step 0:  x = 80% noise + 20% clean
         v = 0 (scaled model prediction)
         x_next = x + 0 = 80% noise + 20% clean

Step 1:  x = 80% noise + 20% clean  (unchanged!)
         v = 0
         x_next = 80% noise + 20% clean

...

Final:   x = 80% noise + 20% clean  ← Still noisy! Looks "underdeveloped"
```

#### Three Velocity Scenarios Compared

**Scenario A: Normal denoising (scale=1.0)**
```
v = model_prediction ≈ (target - x_in)
x_next = x_in + v * dt
       → Moves toward target
       → Eventually denoises completely ✓
```

**Scenario B: Scaled denoising (scale=0.7)**
```
v = 0.7 * model_prediction ≈ 0.7 * (target - x_in)
x_next = x_in + 0.7 * v * dt
       → Moves toward target (70% speed)
       → Still denoises, just slower ✓
       → Preserves structure better (gentler changes) ✓
```

**Scenario C: Zero velocity (scale=0.0) - BROKEN**
```
v = 0.0 * model_prediction = 0
x_next = x_in + 0 * dt = x_in
       → Doesn't move at all
       → NEVER denoises
       → Stays noisy forever ❌
```

**Scenario D: Preservation velocity (correct for preservation)**
```
v = original_clean - x_in
x_next = x_in + (original - x_in) * dt
       → Moves toward ORIGINAL (not prompt target)
       → Converges to original clean latent ✓
       → True preservation ✓
```

---

### 4. The Original Latent - Where Is It?

#### Storage Location

**File**: `/home/user/comfyui-quadtree-tile/utils.py:33-35`

```python
# Captured when KSampler is called
latent_image = kwargs.get('latent_image') if 'latent_image' in kwargs else (args[6] if len(args) > 6 else None)
if latent_image is not None:
    store.latent_image = latent_image  # ← Stored here
```

**When**: During sampler initialization (before denoising begins)

#### Retrieval Location

**File**: `/home/user/comfyui-quadtree-tile/tiled_diffusion.py:1169-1180`

```python
if not hasattr(self, 'original_latent'):
    try:
        from .utils import store
        if hasattr(store, 'latent_image'):
            self.original_latent = store.latent_image  # ← Retrieved here
            print(f'[Quadtree Skip]: Loaded original latent for img2img, shape={self.original_latent.shape}')
        else:
            self.original_latent = None  # txt2img has no original
```

**When**: Once per generation, in the `__call__` method before processing tiles

#### Is It CLEAN or NOISY?

**Answer**: It is the **CLEAN** VAE-encoded latent.

**Evidence**:
1. **Source**: It's the `latent_image` parameter from KSampler, which is the VAE-encoded input image BEFORE noise is added
2. **Usage in skip feature**: Treated as clean target:
   ```python
   velocity = original_tile - x_in_tile  # Velocity toward clean original
   ```
3. **Bug analysis document** confirms it's the clean target (X_1 in rectified flow)

**What it contains**:
```
Original RGB Image
       ↓ VAE Encoder
  Clean Latent  ← This is stored as store.latent_image
       ↓ Add noise (based on denoise parameter)
  Noisy Latent  ← This is x_initial (starting point for sampling)
```

#### Can We Use It to Calculate Correct Preservation Velocity?

**YES!** This is exactly what the skip feature does.

**Skip feature** (tiled_diffusion.py:1300-1320):
```python
if hasattr(self, 'original_latent') and self.original_latent is not None:
    # Extract from CLEAN original
    original_tile = extract_tile_with_padding(self.original_latent, bbox, self.w, self.h)

    # Extract from current NOISY state
    x_in_tile = extract_tile_with_padding(x_in, bbox, self.w, self.h)

    # Compute preservation velocity
    model_prediction = original_tile - x_in_tile  # ✓ Points toward clean original

    # Return this to sampler
    # Sampler applies: x_next = x_in + (original - x_in) * dt
    # Result: Converges to original ✓
```

**Variable denoise feature does NOT use it**:
```python
# tiled_diffusion.py:1421-1468
tile_out = model_function(x_in, t_in, ...)  # Model's prediction

# Just scales the prediction, doesn't use original_latent at all
tile_out = tile_out * scale_factor  # ❌ No access to original
```

---

## The Problem We're Seeing - Root Cause

### User Report
```
Settings: min_denoise=0, max_denoise=0.8

Expected:
- Large tiles: Completely preserved (unchanged from original)
- Small tiles: Regenerated (high denoise)

Actual:
- Large tiles: Look "underdeveloped" / "super faint" / "too much noise"
- Large tiles ARE changing (not preserved)
```

### Why This Happens

**Commit 98e3cbf** changed the formula:
```python
OLD: start_scale = 0.70 + (tile_denoise * 0.25)  # Range: 0.70-0.95
NEW: start_scale = tile_denoise                  # Range: 0.0-1.0
```

**Intent**: "min_denoise=0 should give ZERO changes"

**Reality**: min_denoise=0 gives `scale=0` → `velocity=0` → **stays noisy** → looks underdeveloped

### Why the Old Formula Worked Better

```
OLD (min_denoise=0):
  scale_factor = 0.70
  velocity = model_prediction * 0.70
  x_next = x_in + (0.70 * velocity) * dt
  → Still moves toward target (70% speed)
  → Eventually denoises completely
  → Preserves structure, removes noise ✓

NEW (min_denoise=0):
  scale_factor = 0.0
  velocity = model_prediction * 0.0 = 0
  x_next = x_in + 0 * dt = x_in
  → Never moves
  → Stays permanently noisy
  → Looks "super faint" / "underdeveloped" ❌
```

---

## Solutions

### Option A: Revert to 0.70 Minimum (Quick Fix)

**Change**:
```python
# Line 1451 in tiled_diffusion.py (and similar in other methods)
start_scale = max(0.70, tile_denoise)  # Never go below 0.70
```

**Result**:
- min_denoise=0 → scale=0.70 → gentle denoising (still moves)
- Tiles denoise completely (just slower)
- No "underdeveloped" appearance

**Trade-off**: Not true "zero changes", but reasonable "gentle denoising"

### Option B: Use Original Latent for Preservation (Correct Fix)

**Change**: Blend preservation velocity with model prediction
```python
if tile_denoise < 0.7 and hasattr(self, 'original_latent') and self.original_latent is not None:
    # Extract original
    original_tile = extract_tile_with_padding(self.original_latent, bbox, self.w, self.h)

    # Compute preservation velocity
    preservation_velocity = original_tile - x_in_tile

    # Blend: low denoise → use preservation, high denoise → use model
    tile_out = preservation_velocity * (1 - scale_factor) + tile_out * scale_factor
else:
    # Use scaled model prediction
    tile_out = tile_out * scale_factor
```

**Result**:
- min_denoise=0 → uses preservation_velocity → converges to original ✓
- max_denoise=1 → uses model_prediction → full regeneration ✓
- Smooth blend in between

**Trade-off**: More complex, requires careful implementation

### Option C: Document and Use Skip Feature Instead

**Recommendation**: Use skip feature for preservation, variable denoise for strength control.

```
Variable Denoise (min/max_denoise):
  - Controls denoising STRENGTH (70%-100%)
  - All tiles eventually denoise
  - Use for: gentle vs aggressive denoising

Skip Feature (skip_diffusion_below):
  - Binary: preserve or regenerate
  - Skipped tiles perfectly preserved (img2img)
  - Use for: preserve large regions exactly
```

---

## Summary Table

| Question | Answer |
|----------|--------|
| **What's in latent at start?** | Noisy version of clean original (amount of noise = denoise param) |
| **What's in x_in during denoising?** | Progressively denoising latent (noisy → clean over time) |
| **What does model predict?** | Velocity pointing from current state toward target (v ≈ target - x_in) |
| **What is velocity in FLUX?** | Direction from noise to clean: v = X_1 - X_0 |
| **What does velocity=0 do?** | Freezes at current state (stays noisy!) ❌ |
| **What velocity preserves original?** | v = original_clean - x_in (points toward original) ✓ |
| **Where is original_latent stored?** | utils.py:35 → store.latent_image |
| **Is it clean or noisy?** | CLEAN (VAE-encoded input before noise added) |
| **Can we use it for preservation?** | YES! Skip feature does this correctly |
| **Why doesn't variable denoise work?** | Doesn't use original_latent, only scales model prediction |
| **What's the fix?** | Either: (A) Revert to 0.70 min, or (B) Use original_latent for blend |

---

**Status**: Investigation complete, bug identified, solutions proposed.
