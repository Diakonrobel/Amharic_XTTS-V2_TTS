# Dynamic G2P Backend Selection - Fix Documentation

## Problem Summary

**Issue**: The Amharic training script was hardcoded to use `"rule_based"` G2P backend, ignoring the installed Transphone backend and user preferences.

**Location**: `utils/gpt_train.py` (line 360) and `utils/amharic_g2p_dataset_wrapper.py` (line 200)

**Impact**: 
- Transphone G2P was installed and working for inference
- Training ignored Transphone and used inferior rule-based backend
- No way to dynamically select or fallback between backends

---

## Solution Implemented

### 1. Created Dynamic Backend Selector (`utils/g2p_backend_selector.py`)

A new utility module that:
- **Detects available backends dynamically** (Transphone, Epitran, rule_based)
- **Respects user preferences** when specified
- **Falls back intelligently** if preferred backend unavailable
- **NO hardcoded selection** - fully dynamic detection
- **Provides clear feedback** on detection and selection

**Key Features**:
```python
from utils.g2p_backend_selector import select_g2p_backend

# Auto-select best available backend
backend, reason = select_g2p_backend(preferred=None, fallback=True)

# Use specific backend with fallback
backend, reason = select_g2p_backend(preferred='transphone', fallback=True)
```

**Priority Order** (auto-selection):
1. **Transphone** (highest quality, state-of-the-art)
2. **Epitran** (good quality, rule-based)
3. **Rule-based** (fallback, always available)

---

### 2. Updated Training Script (`utils/gpt_train.py`)

**Changes** (lines 356-375):

```python
# OLD: Hardcoded backend
g2p_backend = "rule_based"  # ❌ HARDCODED

# NEW: Dynamic backend selection
from utils.g2p_backend_selector import select_g2p_backend

selected_backend, reason = select_g2p_backend(
    preferred=None,  # Auto-select best available
    fallback=True,
    verbose=False
)
print(f" > Selected G2P backend: {selected_backend} ({reason})")

# Pass dynamically selected backend
train_samples, eval_samples, new_language = apply_g2p_to_training_data(
    train_samples=train_samples,
    eval_samples=eval_samples,
    train_csv_path=train_csv,
    eval_csv_path=eval_csv,
    language=language,
    g2p_backend=selected_backend  # ✅ DYNAMIC
)
```

---

### 3. Updated Dataset Wrapper (`utils/amharic_g2p_dataset_wrapper.py`)

**Changes** (lines 200-208):

```python
# OLD: Direct instantiation, ignored backend parameter
from amharic_tts.tokenizer.xtts_tokenizer_wrapper import XTTSAmharicTokenizer

tokenizer = XTTSAmharicTokenizer(
    vocab_file=None,
    use_phonemes=True
    # ❌ Backend parameter not passed!
)

# NEW: Factory function respects backend
from amharic_tts.tokenizer.xtts_tokenizer_wrapper import create_xtts_tokenizer

tokenizer = create_xtts_tokenizer(
    vocab_file=None,
    use_phonemes=True,
    g2p_backend=g2p_backend  # ✅ Backend passed correctly
)
```

**How it works**:
1. `create_xtts_tokenizer()` creates config with specified backend
2. Config sets `backend_order = [selected_backend, RULE_BASED]` for fallback
3. Config is passed through: `XTTSAmharicTokenizer` → `HybridAmharicTokenizer` → `EnhancedAmharicG2P`
4. G2P converter uses the configured backend

---

## Verification

### Test Script

Run the comprehensive test:

```bash
python test_g2p_backend_dynamic.py
```

**Expected output**:
```
======================================================================
DYNAMIC G2P BACKEND SELECTION - VERIFICATION TEST
======================================================================

Test 1: Backend Detection
----------------------------------------------------------------------
✅ transphone    - Priority: 1
✅ epitran       - Priority: 2
✅ rule_based    - Priority: 3

✅ Backend detection working correctly!

Test 2: Backend Selection (Auto-select)
----------------------------------------------------------------------
Selected backend: transphone
Reason: Auto-selected (highest priority available)

✅ Auto-selection working correctly!

...
```

### Training Log Messages

When training starts with Amharic G2P enabled, look for:

```
> Applying Amharic G2P preprocessing to training data...
> Current effective_language: 'am'
> Will convert Amharic text → IPA phonemes

> Selected G2P backend: transphone (Auto-selected (highest priority available))

> Loading Amharic G2P tokenizer (backend: transphone)...
> ✓ Transphone G2P loaded successfully (language code: 'amh')
> ✓ Amharic G2P tokenizer loaded successfully (backend: transphone)

> G2P preprocessing complete:
  ✓ Converted: 1842 samples
  ○ Skipped (non-Amharic): 0 samples
  ✗ Failed: 0 samples
```

---

## Benefits

### Before (Hardcoded)
- ❌ Always used `rule_based` backend
- ❌ Ignored installed Transphone
- ❌ No fallback mechanism
- ❌ Inconsistent with inference (which used Transphone)
- ❌ Lower quality G2P conversion

### After (Dynamic)
- ✅ **Auto-detects best available backend**
- ✅ **Uses Transphone if installed** (highest quality)
- ✅ **Intelligent fallback** if backend unavailable
- ✅ **Consistent** between training and inference
- ✅ **User can specify preferred backend** if needed
- ✅ **Clear logging** of which backend is used
- ✅ **Better G2P quality** = better training results

---

## File Summary

### Files Modified
1. ✅ `utils/gpt_train.py` - Dynamic backend selection in training
2. ✅ `utils/amharic_g2p_dataset_wrapper.py` - Respects backend parameter

### Files Created
1. ✅ `utils/g2p_backend_selector.py` - Dynamic backend selector utility
2. ✅ `test_g2p_backend_dynamic.py` - Comprehensive verification test
3. ✅ `DYNAMIC_G2P_BACKEND_FIX.md` - This documentation

### Files Unchanged (but now used correctly)
- `amharic_tts/tokenizer/xtts_tokenizer_wrapper.py` - Already had `create_xtts_tokenizer(g2p_backend=...)`
- `amharic_tts/tokenizer/hybrid_tokenizer.py` - Already passed config to G2P
- `amharic_tts/g2p/amharic_g2p_enhanced.py` - Already supported backend selection via config

---

## Next Steps

### For Training

1. **Start fresh training** with Amharic G2P enabled
2. **Verify logs** show `transphone` is selected (if installed)
3. **Monitor training** - should see better convergence with higher quality G2P
4. **Compare results** with previous runs using rule_based backend

### If Transphone Not Installed

The system will automatically fallback to `rule_based`:
```
> Selected G2P backend: rule_based (Fallback (preferred 'transphone' not available))
```

To install Transphone:
```bash
pip install transphone
```

### Advanced Usage

You can manually specify a backend preference in the code:
```python
selected_backend, reason = select_g2p_backend(
    preferred='epitran',  # Force specific backend
    fallback=True         # Still fallback if unavailable
)
```

---

## Testing Checklist

- [x] Backend detection works
- [x] Auto-selection picks highest priority backend
- [x] Manual backend preference is respected
- [x] Fallback works when preferred unavailable
- [x] Tokenizer receives correct backend
- [x] G2P conversion uses selected backend
- [x] Training integration passes backend correctly
- [x] Log messages are clear and informative

---

## Technical Details

### Backend Priority Logic

The selector uses a simple priority system:
- **Priority 1**: Transphone (best quality)
- **Priority 2**: Epitran (good quality)
- **Priority 3**: Rule-based (always available fallback)

Auto-selection always picks the **lowest priority number** that is **available**.

### Fallback Chain

If Transphone fails:
1. Check if Epitran is available
2. If not, use rule-based (always available)

This ensures training **never fails** due to missing backends.

### Config Flow

```
select_g2p_backend()
    ↓
g2p_backend = "transphone"
    ↓
create_xtts_tokenizer(g2p_backend="transphone")
    ↓
AmharicTTSConfig(g2p.backend_order=[TRANSPHONE, RULE_BASED])
    ↓
XTTSAmharicTokenizer(config=config)
    ↓
HybridAmharicTokenizer(config=config)
    ↓
EnhancedAmharicG2P(config=config)
    ↓
Uses config.g2p.backend_order to try backends
```

---

## Conclusion

The dynamic G2P backend selection is now **fully implemented** and **tested**. Training will automatically use the best available backend (Transphone if installed), with intelligent fallback to ensure reliability. This matches inference behavior and improves training quality.

**Key Achievement**: NO MORE HARDCODED BACKENDS! 🎉
