# Concrete Numerical Example: velocity=0 in FLUX img2img

**Purpose**: Simple, easy-to-follow example with real numbers
**For**: Non-technical users who want to understand the issue

---

## The Setup

Let's trace what happens to a **single pixel** through the entire sampling process.

### Starting Values

**Original clean image pixel** (what we want):
```
R = 200  (bright red)
G = 50   (low green)
B = 50   (low blue)
→ Result: Bright red color
```

**Random noise** (what gets added):
```
R = 0
G = 200
B = 100
→ Result: Teal/cyan color
```

**img2img with denoise=0.55** (mixed together):
```
Noisy pixel = 0.45 × original + 0.55 × noise
            = 0.45 × [200, 50, 50] + 0.55 × [0, 200, 100]
            = [90, 22.5, 22.5] + [0, 110, 55]
            = [90, 132.5, 77.5]
→ Result: Muddy greenish color (wrong!)
```

---

## Normal Denoising (Working Correctly)

### What Should Happen (20 steps)

| Step | R | G | B | Color | Progress |
|------|---|---|---|-------|----------|
| 0 | 90 | 132 | 77 | Muddy green | Starting (noisy) |
| 5 | 125 | 102 | 62 | Orange-brown | Getting warmer |
| 10 | 160 | 75 | 55 | Orange-red | Almost there |
| 15 | 185 | 60 | 52 | Red-orange | Very close |
| 20 | 200 | 50 | 50 | **Bright red** | **✓ Perfect!** |

**Visual progression**:
```
Step 0:  🟢 (Muddy green - wrong color)
Step 5:  🟠 (Orange-brown - getting closer)
Step 10: 🔴 (Orange-red - almost there)
Step 15: 🔴 (Red-orange - nearly perfect)
Step 20: 🔴 (Bright red - correct!)
```

**Result**: ✓ Image looks **sharp and correct**

---

## With velocity=0 (The Bug)

### What Actually Happens (20 steps)

| Step | R | G | B | Color | Progress |
|------|---|---|---|-------|----------|
| 0 | 90 | 132 | 77 | Muddy green | Starting (noisy) |
| 5 | 90 | 132 | 77 | Muddy green | **No change!** |
| 10 | 90 | 132 | 77 | Muddy green | **Still frozen!** |
| 15 | 90 | 132 | 77 | Muddy green | **Not moving!** |
| 20 | 90 | 132 | 77 | **Muddy green** | **✗ Wrong!** |

**Visual progression**:
```
Step 0:  🟢 (Muddy green - noisy start)
Step 5:  🟢 (Muddy green - FROZEN)
Step 10: 🟢 (Muddy green - FROZEN)
Step 15: 🟢 (Muddy green - FROZEN)
Step 20: 🟢 (Muddy green - still wrong!)
```

**Result**: ✗ Image looks **noisy and underdeveloped**

---

## The Math Behind It

### Normal Denoising (Each Step)

**FLUX formula**:
```
next_value = current_value + step_size × velocity

Step 1:
  velocity = model_prediction = [11, -3, -1.5]  (random example)
  R_next = 90 + 0.3 × 11 = 90 + 3.3 = 93.3
  G_next = 132 + 0.3 × (-3) = 132 - 0.9 = 131.1
  B_next = 77 + 0.3 × (-1.5) = 77 - 0.45 = 76.55
  → Moved slightly toward red (progress!)

Step 2:
  velocity = [12, -4, -2]  (model predicts again)
  R_next = 93.3 + 0.3 × 12 = 96.9
  G_next = 131.1 + 0.3 × (-4) = 129.9
  B_next = 76.55 + 0.3 × (-2) = 76.0
  → Moving more toward red (progress!)

... continues for 20 steps ...

Step 20:
  Final: [200, 50, 50] ✓ Correct bright red
```

### With velocity=0 (Each Step)

**With scaling = 0.0**:
```
velocity_scaled = velocity × 0.0 = [0, 0, 0]
next_value = current_value + step_size × 0 = current_value

Step 1:
  velocity = [0, 0, 0]  (scaled to zero!)
  R_next = 90 + 0.3 × 0 = 90
  G_next = 132 + 0.3 × 0 = 132
  B_next = 77 + 0.3 × 0 = 77
  → No change (frozen!)

Step 2:
  velocity = [0, 0, 0]  (still zero!)
  R_next = 90 + 0.3 × 0 = 90
  G_next = 132 + 0.3 × 0 = 132
  B_next = 77 + 0.3 × 0 = 77
  → No change (still frozen!)

... stays frozen for all 20 steps ...

Step 20:
  Final: [90, 132, 77] ✗ Wrong muddy green color
```

---

## Why It Looks "Underdeveloped"

### Expected Result

```
┌─────────────────┐
│                 │
│  🔴🔴🔴🔴🔴      │  ← Bright, saturated red
│  🔴🔴🔴🔴🔴      │     Sharp edges
│  🔴🔴🔴🔴🔴      │     Clear details
│                 │
└─────────────────┘
Quality: ✓ "Fully rendered" "Sharp" "Vibrant"
```

### Actual Result (velocity=0)

```
┌─────────────────┐
│                 │
│  🟢🟢🟢🟢🟢      │  ← Muddy, desaturated green
│  🟢🟢🟢🟢🟢      │     Hazy appearance
│  🟢🟢🟢🟢🟢      │     Looks "unfinished"
│                 │
└─────────────────┘
Quality: ✗ "Underdeveloped" "Noisy" "Hazy"
```

**Why users say "underdeveloped"**:
- Colors are wrong (mixture of original + noise)
- Image looks hazy/foggy (noise not removed)
- Details are blurred (noise obscures fine features)
- Looks like a photo taken out of developer too early

---

## Comparison Table

| Aspect | Normal Denoising | velocity=0 (Bug) |
|--------|------------------|------------------|
| **Starting color** | [90, 132, 77] Muddy green | [90, 132, 77] Muddy green |
| **Step 5** | [125, 102, 62] Orange-brown | [90, 132, 77] Muddy green ✗ |
| **Step 10** | [160, 75, 55] Orange-red | [90, 132, 77] Muddy green ✗ |
| **Step 20** | [200, 50, 50] Bright red ✓ | [90, 132, 77] Muddy green ✗ |
| **Final appearance** | Correct color | Wrong color |
| **User experience** | "Looks great!" | "Looks underdeveloped" |

---

## The Key Insight

### What People Think velocity=0 Does

> "If I set velocity=0, the tile won't change from the **original**"

**Translation**: They think current state = original clean image

### What velocity=0 Actually Does

> "If I set velocity=0, the tile won't change from the **current state**"

**Reality**: Current state = noisy mixture (45% original + 55% noise)

### The Confusion

```
Original clean:     [200, 50, 50]  ← What user wants preserved
                           ↓
                     Add noise (img2img)
                           ↓
Current noisy:      [90, 132, 77]  ← What velocity=0 actually preserves
                           ↓
                     velocity=0
                           ↓
Final output:       [90, 132, 77]  ← Wrong! Not what user wanted!
```

**The problem**: velocity=0 preserves the **NOISY** state, not the **CLEAN** state!

---

## Real-World Example: Sky Region

### Original Photo (Clean)
```
Sky pixel: [135, 206, 235]  (Light blue sky color)
```

### After Noise Addition (denoise=0.55)
```
Noisy sky = 0.45 × [135, 206, 235] + 0.55 × [random noise]
          = [60, 92, 105] + [70, 30, 80]
          = [130, 122, 185]  (Purple-ish, wrong color!)
```

### Normal Denoising (20 steps)
```
Step 0:  [130, 122, 185]  Purple-ish (wrong)
Step 5:  [132, 155, 215]  Blue-purple (getting better)
Step 10: [134, 180, 225]  Light blue-purple (close)
Step 15: [135, 198, 232]  Almost perfect sky blue
Step 20: [135, 206, 235]  Perfect sky blue! ✓
```

**Result**: Sky looks clean and natural

### With velocity=0
```
Step 0:  [130, 122, 185]  Purple-ish (wrong)
Step 5:  [130, 122, 185]  Still purple-ish (frozen)
Step 10: [130, 122, 185]  Still purple-ish (frozen)
Step 15: [130, 122, 185]  Still purple-ish (frozen)
Step 20: [130, 122, 185]  Still purple-ish! ✗
```

**Result**: Sky looks **muddy, hazy, wrong color** - exactly what users report!

---

## The Correct Formula (For Preservation)

If you want to **actually preserve** the original, you need:

```python
velocity = original_clean - current_noisy

Example:
  original = [200, 50, 50]    (bright red)
  current  = [90, 132, 77]    (muddy green)
  velocity = [200-90, 50-132, 50-77]
           = [110, -82, -27]

Step 1:
  next = [90, 132, 77] + 0.3 × [110, -82, -27]
       = [90, 132, 77] + [33, -24.6, -8.1]
       = [123, 107.4, 68.9]  → Moving toward red! ✓
```

This is what the **skip feature** does correctly.

But setting `velocity = 0` does **NOT** do this!

---

## Summary in Simple Terms

### The Bug

When `min_denoise=0`:
1. Model predicts velocity: `[11, -3, -1.5]`
2. Code scales it by 0.0: `[11, -3, -1.5] × 0.0 = [0, 0, 0]`
3. Sampler applies it: `current + step × 0 = current` (no change!)
4. Tile stays noisy forever
5. User sees: "Underdeveloped, hazy, wrong colors"

### Why This Happens

The commit that introduced this thought:
- "velocity=0 = no changes = preserves original" ✗ WRONG

But the reality is:
- "velocity=0 = no changes **from current noisy state**" ✓ CORRECT
- Current state is noisy, not clean!
- So you preserve noisy state, not original

### The Fix

Change the formula so velocity can never be zero:

**Old (working)**:
```python
scale = 0.70 + denoise × 0.25  # Range: 70% to 95%
```
- min_denoise=0 → scale=70% → velocity reduced but not zero
- Tile still moves, just slower
- Looks clean at the end ✓

**New (broken)**:
```python
scale = denoise  # Range: 0% to 100%
```
- min_denoise=0 → scale=0% → velocity becomes zero!
- Tile frozen in noisy state
- Looks underdeveloped ✗

**Fix**: Use old formula or compromise (e.g., 30% to 100%)

---

## Conclusion

**User report**: "Tiles look underdeveloped / noisy"

**Cause**: velocity=0 freezes tiles in noisy state

**Proof**: Mathematical analysis + numerical example shows:
- Normal: [90,132,77] → [200,50,50] (muddy green → bright red) ✓
- velocity=0: [90,132,77] → [90,132,77] (muddy green → muddy green) ✗

**Solution**: Revert to old scaling formula (0.70-0.95 range) or use skip feature

---

**This is why velocity=0 is bad!**
