# Qwen VAE feat_cache Fix - Implementation Checklist

## Quick Summary
Add `**kwargs` support to VAEHook to accept `feat_cache`, `feat_idx`, and `first_chunk` parameters from Qwen VAE models.

## Phase 1: Basic Kwargs Support (15 minutes)

### Change 1: Update VAEHook.__call__ signature (Line 717)
```diff
- def __call__(self, x):
+ def __call__(self, x, **kwargs):
```

### Change 2: Pass kwargs to vae_tile_forward (Line 727)
```diff
- return self.vae_tile_forward(x)
+ return self.vae_tile_forward(x, **kwargs)
```

### Change 3: Pass kwargs to original_forward (Line 725)
```diff
- return self.net.original_forward(x)
+ return self.net.original_forward(x, **kwargs)
```

### Change 4: Update vae_tile_forward signature (Line 911)
```diff
- def vae_tile_forward(self, z):
+ def vae_tile_forward(self, z, **kwargs):
```

**Verification:** After these changes, Qwen VAE should not crash with feat_cache error.

---

## Phase 2: Smart Fallback (Optional but Recommended - 10 minutes)

Add this at the start of `__call__` method to preserve Qwen's cache behavior:

```python
def __call__(self, x, **kwargs):
    try:
        # If using feat_cache, bypass tiling to preserve cache coherence
        if 'feat_cache' in kwargs or 'feat_idx' in kwargs:
            return self.net.original_forward(x, **kwargs)
        
        # Normal tiled processing
        B, C, H, W = x.shape
        if False:
            return self.net.original_forward(x, **kwargs)
        else:
            return self.vae_tile_forward(x, **kwargs)
    finally:
        pass
```

**Benefit:** Prevents breaking Qwen's cache mechanism while still supporting tiling for other VAEs.

---

## Phase 3: Full feat_cache Integration (OPTIONAL - 1-2 hours)

Only if Phase 2's fallback causes memory issues. Requires implementing cache splitting logic in `vae_tile_forward`.

Would need to:
1. Extract `feat_cache` and `feat_idx` from kwargs
2. Initialize cache per tile if needed
3. Pass cache through tile processing
4. Merge caches after tile processing
5. Ensure cache index tracking remains correct

---

## Testing Checklist

- [ ] Code compiles without syntax errors
- [ ] No runtime errors with standard SD VAE
- [ ] No runtime errors with SDXL VAE
- [ ] No runtime errors with Flux VAE
- [ ] Qwen encoder doesn't crash
- [ ] Qwen decoder doesn't crash
- [ ] Tiled processing still produces correct output
- [ ] Quadtree mode works with Qwen
- [ ] Fast mode works with Qwen
- [ ] Color fix works with Qwen
- [ ] Memory usage is reasonable

---

## Files to Modify

**Primary:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`

**Specific Methods:**
- `VAEHook.__init__` - No changes needed
- `VAEHook.__call__` - **CHANGE THIS** (Line 717)
- `VAEHook.vae_tile_forward` - **CHANGE THIS** (Line 911)
- `VAEHook.estimate_group_norm` - No changes needed
- Various internal calls to `original_forward` - **CHANGE THESE**

---

## Code Review Points

When implementing, verify:

1. **Kwargs Propagation:**
   - [ ] __call__ accepts **kwargs
   - [ ] vae_tile_forward accepts **kwargs
   - [ ] original_forward calls include **kwargs
   - [ ] All decorator functions preserve kwargs

2. **Backwards Compatibility:**
   - [ ] Standard VAEs work without feat_cache
   - [ ] No required kwargs (all should have defaults)
   - [ ] No breaking API changes

3. **Error Handling:**
   - [ ] Graceful handling of unknown kwargs
   - [ ] No crashes if kwargs is empty dict
   - [ ] Proper error messages if needed

4. **Performance:**
   - [ ] No overhead from kwargs unpacking
   - [ ] Cache bypass doesn't hurt performance
   - [ ] No memory leaks from cache handling

---

## Quick Testing Commands

After implementation, test with:

```python
# Test 1: Standard VAE (should work as before)
from comfy.sd import VAE
vae = VAE.load_model("default")
samples = torch.randn(1, 4, 64, 64)
# Should work fine

# Test 2: Qwen VAE (new support)
vae = VAE.load_model("qwen")
samples = torch.randn(1, 4, 64, 64)
# Should now work instead of crashing

# Test 3: With feat_cache (advanced)
feat_cache = [None] * 33
feat_idx = [0]
samples = torch.randn(1, 4, 64, 64)
# Should handle gracefully
```

---

## Rollback Plan

If something breaks:

1. Revert changes to `__call__` and `vae_tile_forward` signatures
2. Ensure standard VAEs work again
3. Debug specific issue
4. Re-implement with fix

---

## Documentation Updates Needed

After implementation, update:
- [ ] README.md - Add Qwen VAE support note
- [ ] CHANGELOG.md - Document feat_cache support
- [ ] Add example workflow for Qwen models
- [ ] Update supported models list

---

## Related Issues to Close

Once implemented and tested:
- Close any GitHub issues about Qwen VAE incompatibility
- Link to this implementation in issue comments
- Update related documentation

---

## Time Estimates

- **Phase 1 (Critical):** 15 minutes
  - 5 min: Update __call__ signature
  - 5 min: Update vae_tile_forward signature
  - 5 min: Update internal calls
  
- **Phase 2 (Recommended):** 10 minutes
  - 10 min: Add feat_cache detection and fallback
  
- **Testing:** 30 minutes
  - 15 min: Test with standard VAEs
  - 15 min: Test with Qwen VAE
  
- **Phase 3 (Optional):** 1-2 hours
  - Complex cache integration logic

**Total for Phases 1+2+Testing:** ~55 minutes

---

## Success Criteria

Implementation is complete when:

1. Qwen VAE models work with our tiled VAE hooks
2. No errors with feat_cache, feat_idx parameters
3. Standard SD/SDXL/Flux VAEs still work
4. All existing features (quadtree, fast mode, color fix) work
5. Tests pass without regressions
6. Code is documented and clean

