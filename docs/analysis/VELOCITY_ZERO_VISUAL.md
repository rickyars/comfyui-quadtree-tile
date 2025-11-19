# Visual Demonstration: velocity=0 vs Normal Denoising

**Companion to**: VELOCITY_ZERO_ANALYSIS.md
**Purpose**: Visual illustrations of what happens when velocity=0

---

## 1. Latent Space Trajectory Comparison

### Normal Denoising (20 steps, denoise=0.55)

```
Progress through latent space (→ = movement toward clean target):

Step 0:  [████████░░] x_noisy (55% noise + 45% clean)
Step 1:  [███████░░░] → Moving toward x_clean
Step 2:  [██████░░░░] → Removing noise
Step 3:  [█████░░░░░] → Continuing
Step 4:  [████░░░░░░] → Getting cleaner
Step 5:  [███░░░░░░░] → Major structures visible
         ...
Step 10: [██░░░░░░░░] → Most noise gone
Step 15: [█░░░░░░░░░] → Fine details emerging
Step 20: [░░░░░░░░░░] x_clean (0% noise, 100% clean) ✓

Legend: █ = noise content, ░ = clean content
```

### With velocity=0 (frozen tile)

```
Progress through latent space (✗ = no movement):

Step 0:  [████████░░] x_noisy (55% noise + 45% clean)
Step 1:  [████████░░] ✗ No change (velocity = 0)
Step 2:  [████████░░] ✗ Still frozen
Step 3:  [████████░░] ✗ Not moving
Step 4:  [████████░░] ✗ Stuck
Step 5:  [████████░░] ✗ Same state
         ...
Step 10: [████████░░] ✗ Still noisy
Step 15: [████████░░] ✗ Nothing happening
Step 20: [████████░░] x_noisy (55% noise + 45% clean) ✗

Result: Tile stays in initial noisy state forever!
```

---

## 2. Distance to Target Over Time

### Graph: Distance from x_clean

```
Distance (lower is better)
1.0 ┤
    │                                    Legend:
0.9 ┤                                    ─── Normal denoising
    │   velocity=0                       ─── velocity=0 (frozen)
0.8 ┤ ━━━━━━━━━━━━━━━━━━━━━━━━━━
    │ (frozen - no progress!)
0.7 ┤
    │
0.6 ┤
    │     Normal
0.5 ┤    ╱───╮
    │   ╱     ╲
0.4 ┤  ╱       ╲
    │ ╱         ╲
0.3 ┤╱           ╲
    │             ╲
0.2 ┤              ╲
    │               ╲___
0.1 ┤                    ╲___
    │                        ╲______
0.0 ┼─────────────────────────────────╲──
    0  2  4  6  8  10 12 14 16 18 20
                Steps →

Observations:
• Normal: Distance decreases steadily → reaches 0 → perfect reconstruction
• velocity=0: Distance stays constant → never improves → stays noisy
```

---

## 3. Visual Appearance Timeline

### Normal Denoising (what user sees)

```
Step 0:   [Noisy blob]
          ▓▓▒▒░░▒▒▓▓
          ▓▒░░░░░░▒▓
          ▒░░▓▓▓░░▒
          ▓▒░░░░░░▒▓
          ▓▓▒▒░░▒▒▓▓

Step 5:   [Rough shapes forming]
          ▓▒▒░░░░▒▒▓
          ▒░░░░░░░░▒
          ░░░▓▓▓░░░
          ▒░░░░░░░░▒
          ▓▒▒░░░░▒▒▓

Step 10:  [Details emerging]
          ▒░░░░░░░░▒
          ░░░░░░░░░░
          ░░░▓▓▓░░░
          ░░░░░░░░░░
          ▒░░░░░░░░▒

Step 20:  [Clean, sharp]
          ░░░░░░░░░░
          ░░░░░░░░░░
          ░░░███░░░
          ░░░░░░░░░░
          ░░░░░░░░░░

Quality: ✓ Sharp, clean, fully denoised
```

### With velocity=0 (what user sees)

```
Step 0:   [Noisy blob]
          ▓▓▒▒░░▒▒▓▓
          ▓▒░░░░░░▒▓
          ▒░░▓▓▓░░▒
          ▓▒░░░░░░▒▓
          ▓▓▒▒░░▒▒▓▓

Step 5:   [Still noisy - NO CHANGE]
          ▓▓▒▒░░▒▒▓▓
          ▓▒░░░░░░▒▓
          ▒░░▓▓▓░░▒
          ▓▒░░░░░░▒▓
          ▓▓▒▒░░▒▒▓▓

Step 10:  [Still noisy - NO CHANGE]
          ▓▓▒▒░░▒▒▓▓
          ▓▒░░░░░░▒▓
          ▒░░▓▓▓░░▒
          ▓▒░░░░░░▒▓
          ▓▓▒▒░░▒▒▓▓

Step 20:  [Still noisy - NO CHANGE]
          ▓▓▒▒░░▒▒▓▓
          ▓▒░░░░░░▒▓
          ▒░░▓▓▓░░▒
          ▓▒░░░░░░▒▓
          ▓▓▒▒░░▒▒▓▓

Quality: ✗ Noisy, underdeveloped, hazy
         (Exactly as users reported!)
```

Legend:
- `█` = Fully formed features
- `▓` = Heavy noise
- `▒` = Moderate noise
- `░` = Clean/light

---

## 4. Pixel Value Comparison (Concrete Example)

### Original Clean Image (target)

```
Pixel [100, 150]:
  R: 128
  G: 64
  B: 192
  (Purple pixel)
```

### After Noise Addition (denoise=0.55)

```
Noisy pixel = 0.45 * clean + 0.55 * noise
            = 0.45 * [128, 64, 192] + 0.55 * [200, 180, 40]
            = [57.6, 28.8, 86.4] + [110, 99, 22]
            = [167.6, 127.8, 108.4]
            (Grayish-brown - wrong color!)
```

### Normal Denoising Progression

```
Step 0:  [167.6, 127.8, 108.4] ← Noisy start
Step 5:  [155.2, 105.4, 135.8] ← Getting closer to purple
Step 10: [141.6, 87.2, 161.3]  ← More purple
Step 15: [132.4, 71.6, 182.7]  ← Almost there
Step 20: [128.0, 64.0, 192.0]  ← Perfect! ✓

Final: Correct purple color restored
```

### With velocity=0 Progression

```
Step 0:  [167.6, 127.8, 108.4] ← Noisy start
Step 5:  [167.6, 127.8, 108.4] ← Frozen (velocity=0)
Step 10: [167.6, 127.8, 108.4] ← Still frozen
Step 15: [167.6, 127.8, 108.4] ← No movement
Step 20: [167.6, 127.8, 108.4] ← Still wrong! ✗

Final: Wrong color (grayish-brown instead of purple)
       Looks "underdeveloped" / "noisy"
```

---

## 5. Velocity Vector Visualization

### Normal Denoising (velocity points toward target)

```
Current state (x_current)     Target (x_clean)
       [*]  ──────velocity───────→  [○]

x_current = [167, 127, 108]
x_clean   = [128, 64, 192]
velocity  = x_clean - x_current
          = [-39, -63, +84]

Next step: x_next = x_current + dt * velocity
                  = [167, 127, 108] + 0.05 * [-39, -63, 84]
                  = [167, 127, 108] + [-1.95, -3.15, 4.2]
                  = [165.05, 123.85, 112.2]

Result: Moved closer to target ✓
```

### With velocity=0 (no movement)

```
Current state (x_current)     Target (x_clean)
       [*]    (velocity=0)          [○]
        │                            │
        │    NO CONNECTION           │
        │                            │
        └──────────────────────────────┘
           (stays at current state)

x_current = [167, 127, 108]
x_clean   = [128, 64, 192]
velocity  = 0

Next step: x_next = x_current + dt * 0
                  = [167, 127, 108]

Result: No movement, stays wrong ✗
```

---

## 6. Multi-Tile Comparison (Quadtree with Variable Denoise)

### Normal Configuration (min_denoise=0.3, max_denoise=0.9)

```
┌────────────────────────────────┐
│ Large Tile (512×512)           │
│ denoise=0.3                    │
│ scale=0.3→1.0                  │
│                                │
│ ┌────────┬────────┐            │
│ │ Medium │ Medium │            │
│ │ d=0.6  │ d=0.6  │            │
│ │ s=0.6  │ s=0.6  │            │
│ ├────────┼────────┤            │
│ │  Small │  Small │            │
│ │  d=0.9 │  d=0.9 │            │
│ │  s=0.9 │  s=0.9 │            │
│ └────────┴────────┘            │
│                                │
└────────────────────────────────┘

Result:
  Large tile:  Gentle preservation (scale starts 0.3)
  Medium:      Balanced (scale starts 0.6)
  Small:       Heavy regeneration (scale starts 0.9)
  Appearance:  ✓ Natural variation, smooth blending
```

### Extreme Configuration (min_denoise=0.0, max_denoise=1.0)

```
┌────────────────────────────────┐
│ Large Tile (512×512)           │
│ denoise=0.0  ← PROBLEM!        │
│ scale=0.0→1.0                  │
│ FROZEN for most steps!         │
│                                │
│ ┌────────┬────────┐            │
│ │ Medium │ Medium │            │
│ │ d=0.5  │ d=0.5  │            │
│ │ s=0.5  │ s=0.5  │            │
│ ├────────┼────────┤            │
│ │  Small │  Small │            │
│ │  d=1.0 │  d=1.0 │            │
│ │  s=1.0 │  s=1.0 │            │
│ └────────┴────────┘            │
│                                │
└────────────────────────────────┘

Result:
  Large tile:  ✗ Stays noisy! (frozen early)
  Medium:      ⚠ Partially denoised (insufficient steps)
  Small:       ✓ Full denoising
  Appearance:  ✗ Visible tile boundaries, noisy large areas
               (As users reported!)
```

---

## 7. Timeline: Scaling Factor Progression

### min_denoise=0.0 (problematic)

```
Scale Factor Over Time

1.0 ┤                              ╱─────
    │                            ╱
0.9 ┤                          ╱
    │                        ╱
0.8 ┤                      ╱
    │                    ╱
0.7 ┤                  ╱
    │                ╱
0.6 ┤              ╱
    │            ╱
0.5 ┤          ╱
    │        ╱
0.4 ┤      ╱
    │    ╱
0.3 ┤  ╱
    │╱
0.2 ┤
   ╱│
0.1╱ │
  ╱  │
0.0 ─┼──────────────────────────────────
    0  2  4  6  8 10 12 14 16 18 20
              Steps →

Problem Areas:
  Steps 0-10: scale < 0.5  ← Tile barely moves (stays noisy)
  Steps 11-15: scale 0.5-0.7  ← Partial denoising
  Steps 16-20: scale > 0.8  ← Too late, trying to catch up

Result: Insufficient denoising → looks underdeveloped
```

### min_denoise=0.3 (better)

```
Scale Factor Over Time

1.0 ┤                    ╱──────────────
    │                  ╱
0.9 ┤                ╱
    │              ╱
0.8 ┤            ╱
    │          ╱
0.7 ┤        ╱
    │      ╱
0.6 ┤    ╱
    │  ╱
0.5 ┤╱
   ╱│
0.4╱ │
  ╱  │
0.3 ──┼──────────────────────────────────
    0  2  4  6  8 10 12 14 16 18 20
              Steps →

Benefits:
  Steps 0-10: scale 0.3-0.6  ← Always moving (not frozen)
  Steps 11-15: scale 0.7-0.9  ← Strong denoising
  Steps 16-20: scale ≈ 1.0  ← Full strength

Result: Sufficient denoising → looks clean and developed
```

---

## 8. Color Space Comparison

### Original (target)

```
Color histogram (should be sharp peaks):

    Frequency
    │     ▄
 50 │    ███
    │   █████
 40 │  ███████
    │ █████████
 30 │ █████████
    │███████████
 20 │███████████
    │███████████
 10 │███████████
    │███████████
  0 └───────────────
    0   64  128  192  255
         Pixel Value →

Sharp peaks = Clean image with distinct colors
```

### After Normal Denoising

```
Color histogram (clean reconstruction):

    Frequency
    │     ▄
 50 │    ███
    │   █████
 40 │  ███████
    │ █████████
 30 │ █████████
    │███████████
 20 │███████████
    │███████████
 10 │███████████
    │███████████
  0 └───────────────
    0   64  128  192  255
         Pixel Value →

Result: ✓ Matches original (sharp peaks)
```

### With velocity=0 (frozen, noisy)

```
Color histogram (noisy mixture):

    Frequency
    │
 50 │
    │
 40 │
    │  ▄▄▄▄▄▄▄▄▄▄▄▄
 30 │ ██████████████
    │ ██████████████
 20 │ ██████████████
    │ ██████████████
 10 │ ██████████████
    │ ██████████████
  0 └───────────────
    0   64  128  192  255
         Pixel Value →

Broad, flat distribution = Noise still present
Result: ✗ Looks washed out, hazy, "underdeveloped"
```

---

## 9. User Report Validation Visual

### What User Sees (Large Tile with min_denoise=0)

```
Expected:                  Actual Result:
┌─────────────┐           ┌─────────────┐
│             │           │▒▒▒▒▒▒▒▒▒▒▒▒▒│
│   Clean     │           │▒▒▒Noisy▒▒▒▒▒│
│   Sky       │    VS     │▒▒▒▒▒▒▒▒▒▒▒▒▒│
│   Area      │           │▒Underdeveloped│
│             │           │▒▒▒▒▒▒▒▒▒▒▒▒▒│
└─────────────┘           └─────────────┘
   (What user              (What user
    wanted)                 actually got)

User comments:
  "Looks underdeveloped" ✓ Correct!
  "Noisy"                ✓ Correct!
  "Not fully rendered"   ✓ Correct!
  "Hazy appearance"      ✓ Correct!
```

---

## 10. Side-by-Side Comparison Table

| Aspect | Normal Denoising | velocity=0 (frozen) |
|--------|------------------|---------------------|
| **Movement** | ✓ Moves toward clean target | ✗ Stays at noisy start |
| **Distance** | ✓ Decreases each step | ✗ Constant (no progress) |
| **Final noise level** | ✓ 0% (clean) | ✗ 55% (still noisy) |
| **Visual quality** | ✓ Sharp, clean | ✗ Blurry, hazy |
| **Color accuracy** | ✓ Matches original | ✗ Wrong colors (mixture) |
| **User satisfaction** | ✓ "Looks great" | ✗ "Looks underdeveloped" |
| **Preserves original?** | ✓ YES | ✗ NO (preserves noisy state) |

---

## 11. Conclusion Diagram

```
Question: What happens with velocity=0?

                    ┌──────────────┐
                    │  velocity=0  │
                    └──────┬───────┘
                           │
                           ↓
              ┌────────────────────────┐
              │ x_next = x_current + 0 │
              │ x_next = x_current     │
              └────────┬───────────────┘
                       │
                       ↓
              ┌────────────────────┐
              │  Tile FROZEN       │
              │  at noisy state    │
              └────────┬───────────┘
                       │
                       ↓
         ┌─────────────────────────────┐
         │  Never reaches x_clean      │
         │  Stays at:                  │
         │  x = 0.45·clean + 0.55·noise│
         └─────────────┬───────────────┘
                       │
                       ↓
              ┌────────────────────┐
              │  Visual Result:    │
              │  • Noisy           │
              │  • Underdeveloped  │
              │  • Hazy            │
              │  • Wrong colors    │
              └────────────────────┘

Answer: It stays NOISY, NOT clean! ✗
```

---

**End of Visual Analysis**
