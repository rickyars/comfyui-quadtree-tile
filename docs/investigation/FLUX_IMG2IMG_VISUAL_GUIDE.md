# FLUX img2img Visual Guide - Understanding Velocity and Preservation

**Date**: 2025-11-19

---

## Visual: The img2img Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMFYUI IMG2IMG WORKFLOW                      │
└─────────────────────────────────────────────────────────────────┘

Input Image (RGB)
    │
    ├─→ [VAE Encoder]
    │
    └─→ CLEAN LATENT (X_1) ←─────────── Stored as store.latent_image
             │                           self.original_latent
             │
             ├─→ [Add Noise based on denoise parameter]
             │
             └─→ NOISY LATENT (X_t)
                      │
                      ├─→ Starting point for sampling
                      │   (denoise=0.5 → 50% clean + 50% noise)
                      │
                      v
              ┌───────────────┐
              │   Denoising   │
              │     Loop      │
              └───────────────┘
                      │
                      v
              Each step processes x_in (current noisy state)
                      │
                      v
              ┌─────────────────────┐
              │  Model predicts v   │
              │  (velocity toward   │
              │   target based on   │
              │   prompt)           │
              └─────────────────────┘
                      │
                      v
              x_next = x_in + v * dt
                      │
                      v
              Gradually converges to clean target
```

---

## Visual: What's in the Latent at Each Stage?

```
TIME:     t=0.0         t=0.3         t=0.6         t=0.9         t=1.0
STATE:    [noise]     [mostly      [balanced]   [mostly       [clean]
                       noise]                     clean]

VISUAL:   ▓▓▓▓▓▓      ▓▓▓▓▒▒       ▓▒▒▒░░       ▒░░░░░        ░░░░░░
          ▓▓▓▓▓▓      ▓▓▓▒▒▒       ▓▒▒░░░       ▒░░░░░        ░░░░░░
          ▓▓▓▓▓▓      ▓▓▒▒▒░       ▓▒░░░░       ▒░░░░░        ░░░░░░

          Pure        80% noise    60% noise    20% noise     Clean
          random      20% clean    40% clean    80% clean     target
          noise

          ↑                                                    ↑
          X_0                                                  X_1
          (noise)                                            (clean)

FORMULA:  X_t = t * X_1 + (1-t) * X_0

At t=0.0: X_0 = 0.0*clean + 1.0*noise = pure noise
At t=0.3: X_0.3 = 0.3*clean + 0.7*noise
At t=0.5: X_0.5 = 0.5*clean + 0.5*noise
At t=0.9: X_0.9 = 0.9*clean + 0.1*noise
At t=1.0: X_1 = 1.0*clean + 0.0*noise = clean
```

---

## Visual: FLUX Velocity Prediction

```
┌─────────────────────────────────────────────────────────────────┐
│                    VELOCITY = DIRECTION                          │
└─────────────────────────────────────────────────────────────────┘

    X_0 (noise)                                    X_1 (clean)
         •━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━•
         ↑                       ↑                       ↑
         │                       │                       │
    t=0 (start)            t=0.5 (middle)           t=1 (end)
    Pure noise            50% denoised              Clean

    Velocity v = X_1 - X_0  (direction from noise to clean)
                 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→
                 "This way to clean!"

Model learns: Given current position X_t, predict velocity v
              that points toward clean target X_1

Sampler applies: X_{t+dt} = X_t + v * dt
                 "Move in direction v by amount dt"
```

---

## Visual: The Problem with velocity=0

```
┌─────────────────────────────────────────────────────────────────┐
│          WHAT HAPPENS WITH DIFFERENT VELOCITY VALUES             │
└─────────────────────────────────────────────────────────────────┘

CASE A: Normal velocity (scale=1.0)
────────────────────────────────────
X_0 (noise)    →    →    →    →    →    X_1 (clean)
    •━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━•
    │          │    │    │    │    │          │
    │    v     │    v    │    v    │    v     │
    │          │         │         │          │
    x_0   →   x_1   →   x_2   →   x_3   →   x_4

Result: Moves steadily toward clean target ✓


CASE B: Scaled velocity (scale=0.7)
────────────────────────────────────
X_0 (noise)   →   →   →   →   →   →   →   X_1 (clean)
    •━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━•
    │      v   │    v   │    v   │    v   │  │
    │  (70%)   │  (70%) │  (70%) │  (70%) │  │
    x_0    →  x_1   →  x_2   →  x_3   →  x_4 x_5

Result: Moves slower but still reaches clean target ✓
        (Takes more steps, gentler changes)


CASE C: Zero velocity (scale=0.0) - BROKEN!
────────────────────────────────────
X_0 (noise)                            X_1 (clean)
    •━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━•
    │                                         │
    │   v=0  v=0  v=0  v=0  v=0              │
    │   (no velocity = no movement)          │
    │                                         │
    x ══════════════════════════════════════ (stuck!)
    │
    └─→ Frozen at 40% noise + 60% clean
        Never fully denoises!
        Looks "underdeveloped" / "super faint" ❌


CASE D: Preservation velocity (v = original - x_in) - CORRECT!
────────────────────────────────────
X_0 (noise)    X_current         X_original (what we want)
    •━━━━━━━━━━━━━•━━━━━━━━━━━━━━━━━━━━━━•
                   │                       │
                   │   v = X_orig - X_cur │
                   │   ──────────────────→ │
                   │                       │
                   x_0  →  x_1  →  x_2  →  x_3 (original)

Result: Converges to original clean latent ✓
        (True preservation)
```

---

## Visual: Variable Denoise vs Skip Feature

```
┌─────────────────────────────────────────────────────────────────┐
│               VARIABLE DENOISE (Current - Broken)                │
└─────────────────────────────────────────────────────────────────┘

Large Tile (min_denoise=0):
────────────────────────────
    x_in (noisy)                Model Target (prompt-based)
         •━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→•
         │                                    │
         │ Model predicts: v = target - x_in │
         │                                    │
         │ We scale: v_actual = v * 0 = 0    │
         │                                    │
         │ Sampler: x_next = x_in + 0 = x_in │
         │                                    │
         │ Result: STUCK AT NOISY STATE ❌    │
         •

Small Tile (max_denoise=1.0):
────────────────────────────
    x_in (noisy)  →  →  →  →  →  →  →  Model Target
         •━━━━━━━━━━━━━━━━━━━━━━━━━━━━→•
         │    v_full                     │
         │    (scale=1.0)                │
         └→  Fully denoises ✓


┌─────────────────────────────────────────────────────────────────┐
│                  SKIP FEATURE (Correct for img2img)              │
└─────────────────────────────────────────────────────────────────┘

Large Tile (skipped):
────────────────────────────
    x_in (noisy)                X_original (stored clean)
         •━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→•
         │                                    │
         │ We compute: v = original - x_in   │
         │             ────────────────────→  │
         │                                    │
         │ Sampler: x_next = x_in + v * dt   │
         │          → moves toward original  │
         │                                    │
         │ Result: CONVERGES TO ORIGINAL ✓   │
         │━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→•
                                              │
                                         (preserved!)

Small Tile (processed):
────────────────────────────
    x_in (noisy)  →  →  →  →  →  →  →  Model Target
         •━━━━━━━━━━━━━━━━━━━━━━━━━━━━→•
         │    Normal model inference      │
         └→  Fully regenerated ✓
```

---

## Visual: Why Original Latent Access Matters

```
┌─────────────────────────────────────────────────────────────────┐
│              WITHOUT ORIGINAL LATENT ACCESS                      │
│                  (Current Variable Denoise)                      │
└─────────────────────────────────────────────────────────────────┘

                    ┌──────────────┐
                    │ Model knows: │
                    │  - Prompt    │
                    │  - x_in      │
                    │  - timestep  │
                    └──────────────┘
                           │
                           v
                  ┌────────────────┐
                  │ Predicts v →   │
                  │ toward prompt  │
                  │ target (NEW)   │
                  └────────────────┘
                           │
                           v
                  We scale: v * α
                           │
                           v
              If α=0: v=0 → stays noisy ❌

    Original Image: [UNKNOWN to model]
                     ↑
                     └─ Model doesn't know what was originally there!
                        Only knows what SHOULD be there (prompt)


┌─────────────────────────────────────────────────────────────────┐
│               WITH ORIGINAL LATENT ACCESS                        │
│                     (Skip Feature)                               │
└─────────────────────────────────────────────────────────────────┘

    Original Image (stored)
           │
           v
    ┌──────────────┐
    │ Clean latent │  ←─── We have this!
    │ (X_original) │
    └──────────────┘
           │
           └────────────────┐
                            │
                            v
                   ┌────────────────┐
                   │ Current noisy  │
                   │ state (x_in)   │
                   └────────────────┘
                            │
                            v
               ┌────────────────────────┐
               │ v = original - x_in   │
               │ (direct calculation)   │
               └────────────────────────┘
                            │
                            v
               Converges to original ✓
```

---

## Visual: The Three States in img2img

```
┌─────────────────────────────────────────────────────────────────┐
│         THREE IMPORTANT LATENT STATES IN IMG2IMG                 │
└─────────────────────────────────────────────────────────────────┘

STATE 1: X_original (Clean Original Latent)
──────────────────────────────────────────
    • Stored in: store.latent_image
    • What: VAE-encoded input image (CLEAN, no noise)
    • When: Captured before sampling begins
    • Used by: Skip feature for preservation

    [Beautiful image of a blue car]
    ↓ VAE Encode
    [ Clean latent - sharp, clear, no noise ]


STATE 2: x_initial (Noisy Starting Point)
──────────────────────────────────────────
    • Starts as: X_original + added_noise
    • Amount of noise: Based on denoise parameter
    • When: At t=0 of sampling loop
    • Example: denoise=0.5 → 50% clean + 50% noise

    [ Clean latent ]  +  [ Gaussian noise ]
         ↓                      ↓
    [ Noisy starting latent - somewhat fuzzy ]


STATE 3: x_in (Current Denoising State)
──────────────────────────────────────────
    • Changes at: Every sampling step
    • Progression: Noisy → Clean over time
    • What model sees: This noisy intermediate state
    • Model predicts: How to move toward clean

    Step 1: [ 80% noise, 20% clean - very fuzzy ]
    Step 2: [ 60% noise, 40% clean - getting clearer ]
    Step 3: [ 40% noise, 60% clean - mostly clear ]
    Step 4: [ 20% noise, 80% clean - nearly clean ]
    Final:  [ 0% noise, 100% clean - sharp result ]


KEY INSIGHT:
───────────
    X_original ≠ x_in  (at most steps)
         ↑         ↑
         │         │
      Clean    Currently noisy

    velocity = 0 means:
        "Stay at x_in" (noisy) ❌

    velocity = X_original - x_in means:
        "Move toward X_original" (clean) ✓
```

---

## Visual: The Commit That Broke It

```
┌─────────────────────────────────────────────────────────────────┐
│            COMMIT 98e3cbf: BEFORE VS AFTER                       │
└─────────────────────────────────────────────────────────────────┘

BEFORE (Working):
─────────────────
    start_scale = 0.70 + (tile_denoise * 0.25)
                  └─────┘
                  Minimum of 0.70

    min_denoise=0 → scale=0.70

    Behavior:
    ────────
    X_0 (noisy)  ───→───→───→───→───→  X_1 (clean)
         •━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━•
         │   v*0.7  v*0.7  v*0.7          │
         │   (70% speed)                  │
         │                                │
         x_0 → x_1 → x_2 → x_3 → x_4 → x_5

    Result: Denoises completely (slowly) ✓
            Looks clean but preserved


AFTER (Broken):
───────────────
    start_scale = tile_denoise
                  └──────────┘
                  Can be 0.0!

    min_denoise=0 → scale=0.0

    Behavior:
    ────────
    X_0 (noisy)                      X_1 (clean)
         •━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━•
         │                               │
         │  v*0=0  v*0=0  v*0=0          │
         │  (no movement!)               │
         │                               │
         x ═══════════════════════════════

    Result: Stays noisy forever ❌
            Looks "underdeveloped"


COMMIT MESSAGE (Incorrect Understanding):
──────────────────────────────────────────
    "min_denoise=0: Large tiles should get ZERO changes"

    ❌ WRONG: velocity=0 ≠ zero changes
    ✓ RIGHT: velocity=0 = frozen in noisy state
```

---

## Visual: Solution Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                  SOLUTION A: REVERT TO 0.70                      │
└─────────────────────────────────────────────────────────────────┘

    start_scale = max(0.70, tile_denoise)

    min_denoise=0 → scale=0.70 → gentle denoising

    Pros:                      Cons:
    ─────────────────         ──────────────────────
    • Quick fix               • Not true "zero changes"
    • No noisy tiles          • Still modifies slightly
    • Backward compatible     • Doesn't match user expectation

    Result: Acceptable compromise ✓


┌─────────────────────────────────────────────────────────────────┐
│            SOLUTION B: BLEND WITH ORIGINAL LATENT                │
└─────────────────────────────────────────────────────────────────┘

    if scale < threshold and original_latent available:
        v_preserve = original - x_in
        v_model = model_prediction
        v_final = v_preserve * (1-scale) + v_model * scale

    min_denoise=0 → scale=0 → v_final = v_preserve (100%)
    max_denoise=1 → scale=1 → v_final = v_model (100%)

    Pros:                      Cons:
    ─────────────────         ──────────────────────
    • True preservation       • More complex
    • Smooth blend            • Requires testing
    • Matches expectations    • Blend artifacts?

    Result: Ideal solution ✓


┌─────────────────────────────────────────────────────────────────┐
│              SOLUTION C: USE SKIP FEATURE INSTEAD                │
└─────────────────────────────────────────────────────────────────┘

    Variable Denoise: For strength control (0.7-1.0)
    Skip Feature: For preservation (binary on/off)

    Pros:                      Cons:
    ─────────────────         ──────────────────────
    • Already works           • Two separate features
    • No code changes         • User confusion
    • Clear separation        • Skip is all-or-nothing

    Result: Workaround, not fix ⚠️
```

---

## Quick Reference: Velocity Formulas

```
┌─────────────────────────────────────────────────────────────────┐
│                    VELOCITY FORMULA CHEATSHEET                   │
└─────────────────────────────────────────────────────────────────┘

Normal Denoising:
─────────────────
    v = model_prediction(x_in, t, prompt)
    x_next = x_in + v * dt
    Result: Denoises toward prompt target ✓

Scaled Denoising:
─────────────────
    v = α * model_prediction(x_in, t, prompt)
    x_next = x_in + α*v * dt
    Result: Slower denoising toward prompt target ✓

Zero Velocity (BROKEN):
───────────────────────
    v = 0
    x_next = x_in + 0 * dt = x_in
    Result: STUCK AT NOISY STATE ❌

Preservation Velocity (CORRECT):
────────────────────────────────
    v = X_original_clean - x_in_current_noisy
    x_next = x_in + v * dt
    Result: Converges to original ✓

Hybrid (Proposed):
──────────────────
    v_preserve = X_original - x_in
    v_model = model_prediction(x_in, t, prompt)
    v_final = v_preserve * (1-α) + v_model * α
    x_next = x_in + v_final * dt
    Result: Blend between preservation and generation ✓
```

---

**Visual guide complete. For detailed analysis, see FLUX_IMG2IMG_VELOCITY_INVESTIGATION.md**
