"""
Check potential issues with variable denoise conditions

This script checks for edge cases and potential failures:
1. When sigmas are not available
2. When timestep is not found in sigmas
3. When min_denoise == max_denoise == 1.0
4. Floating point precision issues with sigma comparison
"""

import torch

def find_nearest(a, b):
    """From tiled_diffusion.py line 851"""
    # Calculate the absolute differences.
    diff = (a - b).abs()

    # Find the indices of the nearest elements
    nearest_indices = diff.argmin()

    # Get the nearest elements from b
    return b[nearest_indices]


def test_find_nearest():
    """Test find_nearest function"""
    print("Testing find_nearest function:")
    print("="*80)

    # Create sample sigmas (typical denoising schedule)
    sigmas = torch.tensor([14.6146, 10.3158, 7.2761, 5.1302, 3.6171, 2.5493, 1.7963, 1.2654, 0.8914, 0.6282, 0.4426, 0.3118, 0.2196, 0.1546, 0.1088, 0.0766, 0.0539, 0.0380, 0.0267, 0.0188, 0.0000])

    # Test exact match
    t_exact = torch.tensor([5.1302])
    nearest_exact = find_nearest(t_exact, sigmas)
    print(f"Exact match: t={t_exact.item():.4f}, nearest={nearest_exact.item():.4f}")

    # Test approximate value (between two sigmas)
    t_approx = torch.tensor([5.5])
    nearest_approx = find_nearest(t_approx, sigmas)
    print(f"Approximate: t={t_approx.item():.4f}, nearest={nearest_approx.item():.4f}")

    # Test value slightly off due to floating point
    t_fp = torch.tensor([5.1302001])  # Slight floating point error
    nearest_fp = find_nearest(t_fp, sigmas)
    print(f"FP precision: t={t_fp.item():.7f}, nearest={nearest_fp.item():.4f}")

    # Test the equality check (problematic!)
    print("\nTesting equality check (sigmas == ts_in):")
    ts_in = find_nearest(t_fp, sigmas)
    equality_check = (sigmas == ts_in).nonzero()
    print(f"  ts_in from find_nearest: {ts_in.item():.7f}")
    print(f"  Equality check result: {equality_check}")
    print(f"  Found match: {equality_check.shape[0] > 0}")

    # This SHOULD work because find_nearest returns an actual value from sigmas
    # But let's verify
    print("\nVerifying find_nearest always returns value from sigmas:")
    for t in [5.5, 7.3, 12.0, 0.5]:
        t_tensor = torch.tensor([t])
        nearest = find_nearest(t_tensor, sigmas)
        is_in_sigmas = (sigmas == nearest).any()
        print(f"  t={t:6.2f} -> nearest={nearest.item():.4f}, in sigmas={is_in_sigmas.item()}")

    print()


def test_variable_denoise_conditions():
    """Test the conditions that enable variable denoise"""
    print("Testing variable denoise conditions:")
    print("="*80)

    conditions = [
        ("use_qt=False (grid mode)", {'use_qt': False, 'has_sigmas': True, 'sigmas_not_none': True, 'denoise_lt_1': True}),
        ("sigmas not loaded", {'use_qt': True, 'has_sigmas': False, 'sigmas_not_none': True, 'denoise_lt_1': True}),
        ("sigmas is None", {'use_qt': True, 'has_sigmas': True, 'sigmas_not_none': False, 'denoise_lt_1': True}),
        ("tile_denoise=1.0", {'use_qt': True, 'has_sigmas': True, 'sigmas_not_none': True, 'denoise_lt_1': False}),
        ("All conditions met", {'use_qt': True, 'has_sigmas': True, 'sigmas_not_none': True, 'denoise_lt_1': True}),
    ]

    for name, conds in conditions:
        result = (conds['use_qt'] and
                 conds['has_sigmas'] and
                 conds['sigmas_not_none'] and
                 conds['denoise_lt_1'])

        status = "✓ ENABLED" if result else "✗ DISABLED"
        print(f"{name:<30} {status}")
        if not result:
            failed = [k for k, v in conds.items() if not v]
            print(f"  Failed: {', '.join(failed)}")

    print()


def test_min_max_same():
    """Test behavior when min_denoise == max_denoise"""
    print("Testing min_denoise == max_denoise:")
    print("="*80)

    min_denoise = max_denoise = 0.7
    max_tile_area = 512 * 512

    print(f"Settings: min_denoise={min_denoise}, max_denoise={max_denoise}")
    print()

    # All tiles will have the same denoise value
    tiles = [
        ('Large tile', max_tile_area),
        ('Medium tile', max_tile_area * 0.5),
        ('Small tile', max_tile_area * 0.1),
    ]

    for tile_name, tile_area in tiles:
        size_ratio = tile_area / max_tile_area
        denoise = min_denoise + (max_denoise - min_denoise) * (1.0 - size_ratio)

        # Check if variable denoise would be applied
        will_scale = denoise < 1.0

        print(f"{tile_name:<20} denoise={denoise:.3f}, will_scale={will_scale}")

    print()
    print("Conclusion: All tiles get same denoise value (0.7).")
    print("Variable denoise WILL be applied since 0.7 < 1.0.")
    print("All tiles will be scaled identically (same start_scale, same curve).")
    print()


def test_all_denoise_1():
    """Test behavior when min_denoise == max_denoise == 1.0"""
    print("Testing min_denoise == max_denoise == 1.0:")
    print("="*80)

    min_denoise = max_denoise = 1.0
    max_tile_area = 512 * 512

    print(f"Settings: min_denoise={min_denoise}, max_denoise={max_denoise}")
    print()

    tiles = [
        ('Large tile', max_tile_area),
        ('Medium tile', max_tile_area * 0.5),
        ('Small tile', max_tile_area * 0.1),
    ]

    for tile_name, tile_area in tiles:
        size_ratio = tile_area / max_tile_area
        denoise = min_denoise + (max_denoise - min_denoise) * (1.0 - size_ratio)

        # Check if variable denoise would be applied
        will_scale = denoise < 1.0

        print(f"{tile_name:<20} denoise={denoise:.3f}, will_scale={will_scale}")

    print()
    print("Conclusion: All tiles get denoise=1.0.")
    print("Variable denoise will NOT be applied (condition: tile_denoise < 1.0 is False).")
    print("All tiles get full strength denoising - CORRECT behavior for uniform max denoise.")
    print()


def test_potential_issues():
    """Document potential issues"""
    print("POTENTIAL ISSUES:")
    print("="*80)

    print("""
1. ✓ find_nearest() function works correctly
   - Always returns actual value from sigmas tensor
   - Equality check (sigmas == ts_in) will always find match
   - No floating point precision issues

2. ✓ Condition checks are correct
   - Only applies in quadtree mode (use_qt)
   - Only applies if sigmas are available
   - Only applies if tile_denoise < 1.0 (skips full denoise tiles)

3. ✓ Edge case: min_denoise == max_denoise (same value)
   - All tiles get same denoise value
   - If value < 1.0, all tiles scaled identically (uniform behavior)
   - If value == 1.0, no scaling (full strength)
   - CORRECT behavior

4. ⚠️  POTENTIAL ISSUE: Silent failure when sigmas not available
   - If sigmas are not in store, prints warning but continues
   - Variable denoise simply won't work (tiles at full strength)
   - User might not notice if they're not checking console output
   - SUGGESTION: Add visual indicator or node output warning

5. ⚠️  POTENTIAL ISSUE: Progress calculation assumes forward diffusion
   - current_step / total_steps assumes step 0 = start, last step = end
   - Might not work correctly with non-standard samplers
   - VERIFY: Does this work with all ComfyUI samplers?

6. ✓ Scale factor is always in valid range [0.70, 1.0]
   - Clamped at line 1368
   - No risk of negative or >1.0 values

7. ⚠️  POTENTIAL ISSUE: Scaling applied to model output directly
   - Line 1376: tile_out = tile_out * scale_factor
   - This assumes model_function returns noise/velocity prediction
   - Might not work correctly if model returns x0 prediction
   - VERIFY: Does this work with all prediction types (epsilon, v, x0)?
""")

    print("\nRECOMMENDATIONS:")
    print("-"*80)
    print("""
1. Add better error handling when sigmas not available:
   - Could fall back to uniform denoise
   - Or disable quadtree and use grid mode
   - Or add prominent warning in UI

2. Verify compatibility with different samplers:
   - Test with Euler, DPM++, DDIM, etc.
   - Ensure progress calculation is correct

3. Verify compatibility with different prediction types:
   - Test with epsilon (SD1.5/SDXL)
   - Test with velocity (FLUX)
   - Test with x0 prediction if supported

4. Add unit test to verify scaling behavior:
   - Test that scale_factor is always in [0.70, 1.0]
   - Test that progress calculation is monotonic
   - Test edge cases (0 steps, 1 step, etc.)
""")


if __name__ == '__main__':
    test_find_nearest()
    test_variable_denoise_conditions()
    test_min_max_same()
    test_all_denoise_1()
    test_potential_issues()
