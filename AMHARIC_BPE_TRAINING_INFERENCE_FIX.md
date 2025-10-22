# Amharic BPE-Only Training-Inference Mismatch Fix

## Problem Summary

When training an XTTS v2 model with **Amharic extended BPE tokenizer (WITHOUT G2P)**, the inference produces garbled audio with complete pronunciation issues, even though training progresses normally.

## Root Cause Analysis

### Training Configuration (BPE-only mode)
```python
# In gpt_train.py (lines ~300-400)
use_amharic_g2p = False  # BPE-only, no G2P
language = "amh"  # Amharic language code

# After vocab extension:
effective_language = "en"  # Changed to 'en' for tokenizer compatibility
tokenizer_file = "vocab_extended_amharic.json"  # Extended with Ethiopic chars
```

**Key Training Behaviors:**
1. Raw Ethiopic text is passed to tokenizer
2. Language code set to **'en'** to avoid NotImplementedError
3. Tokenizer uses English preprocessing (returns raw text)
4. BPE learns Ethiopic character patterns

### Inference Configuration (BEFORE FIX)
```python
# In xtts_demo.py run_tts() (lines ~500-600)
lang = "amh"  # User selects Amharic
use_g2p_inference = True  # User enables G2P in UI

# Problem 1: Applies G2P even though model trained without it
if use_g2p_inference and lang == "amh":
    tts_text = apply_g2p(tts_text)  # Converts to phonemes

# Problem 2: Uses 'am' language code instead of 'en'
_inference_lang = 'am'  # Wrong! Should be 'en' to match training
```

**Mismatches:**
1. **G2P Applied**: Inference converts text to phonemes, but model expects raw Ethiopic
2. **Wrong Language Code**: Uses 'am' but training used 'en'
3. **Tokenizer Preprocessing**: Different preprocessing path than training

## The Fix

### 1. Training Mode Detection
```python
# Read training_meta.json to determine if G2P was used
model_trained_with_g2p = False
try:
    meta = json.load(open("ready/training_meta.json"))
    model_trained_with_g2p = meta['amharic']['g2p_training_enabled']
except:
    pass
```

### 2. Conditional G2P Application
```python
# Only apply G2P if model was trained with it
if use_g2p_inference and lang == "amh" and model_trained_with_g2p:
    tts_text = apply_g2p(tts_text)
    g2p_active = True
else:
    # Skip G2P for BPE-only models
    g2p_active = False
```

### 3. Correct Language Code
```python
# For Amharic BPE-only: use 'en' to match training
if lang == "amh" and not model_trained_with_g2p:
    _inference_lang = 'en'  # Matches training tokenizer behavior
    print("Using BPE-only mode (raw Ethiopic + 'en' language)")
else:
    _inference_lang = 'am'  # Standard Amharic mode
```

### 4. Tokenizer Preprocessing Patch
```python
# Ensure tokenizer uses same preprocessing as training
def _preprocess_text_training_aware(txt, lang):
    if lang in ('am', 'amh', 'en'):
        # Use English preprocessing for consistency
        return _orig_preprocess(txt, 'en')
    return _orig_preprocess(txt, lang)
```

## How to Verify the Fix

### 1. Check Training Metadata
```bash
cat finetune_models/ready/training_meta.json
```

Expected output for BPE-only:
```json
{
  "amharic": {
    "g2p_training_enabled": false,
    "g2p_backend": null,
    "effective_language": "en",
    "vocab_used": "extended"
  }
}
```

### 2. Test Inference
```python
# In Gradio UI:
# - Load your trained model
# - Enter Amharic text: "ሰላም ዓለም"
# - DISABLE "Enable Amharic G2P" checkbox
# - Generate speech

# Console output should show:
# > Model training mode detected: BPE-only
# > Inference mode: BPE-only (raw Ethiopic + 'en' language code)
# > This matches training configuration for correct pronunciation
```

### 3. Compare Training vs Inference
```python
# Training tokenizer behavior:
tokenizer.encode("ሰላም", lang="en")  # -> [1024, 5432, 8765, ...]

# Inference should produce SAME tokens:
tokenizer.encode("ሰላም", lang="en")  # -> [1024, 5432, 8765, ...]

# NOT this (wrong):
tokenizer.encode("ሰላም", lang="am")  # -> [UNK, UNK, UNK, ...]
```

## Files Modified

1. **xtts_demo.py** (Inference)
   - Lines ~520-570: Added training mode detection
   - Lines ~530-550: Conditional G2P application
   - Lines ~580-600: Correct language code selection
   - Lines ~350-370: Enhanced tokenizer preprocessing patch

2. **gpt_train.py** (Training)
   - Lines ~400-450: Writes training_meta.json with mode info
   - Already correct - no changes needed

## Testing Checklist

- [ ] Training completes without errors
- [ ] `training_meta.json` created in `ready/` folder
- [ ] Inference console shows "BPE-only mode" detection
- [ ] Generated audio has correct Amharic pronunciation
- [ ] No UNK tokens in tokenizer output
- [ ] Language code is 'en' during inference for Amharic

## Troubleshooting

### Issue: Still getting garbled audio
**Solution**: Make sure to DISABLE G2P checkbox in inference UI

### Issue: UNK tokens appearing
**Solution**: Check that `vocab_extended_amharic.json` is being loaded, not `vocab.json`

### Issue: "NotImplementedError: Language 'amh' not supported"
**Solution**: The tokenizer patch didn't apply. Check console for patch errors.

### Issue: Training worked but inference fails
**Solution**: Make sure you pulled latest code to Lightning AI remote PC:
```bash
git pull origin main
# Then restart training
```

## Summary

The fix ensures that:
1. **Training mode is detected** from metadata file
2. **G2P is only applied** if model was trained with it
3. **Language code matches** training configuration ('en' for BPE-only)
4. **Tokenizer preprocessing is consistent** between training and inference

This resolves the pronunciation issues by eliminating all training-inference mismatches.
