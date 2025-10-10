# Amharic TTS Pronunciation Fix

This document explains the fix for Amharic TTS pronunciation issues in the XTTS fine-tuning system.

## The Problem

When inferring with Amharic text after fine-tuning a model, the pronunciation was poor because:

1. During **training**:
   - Amharic text was converted to phonemes using G2P
   - The language code was switched from 'am' to 'en' for XTTS tokenizer compatibility

2. During **inference**:
   - Amharic text was converted to phonemes using G2P
   - The language code was incorrectly overridden to 'en' when G2P was active
   - This caused the model to apply English pronunciation rules to Amharic phonemes

## The Fix

The fix ensures consistent language code handling between training and inference:

1. The `normalize_xtts_lang()` function in `xtts_demo.py` now correctly preserves 'am' for Amharic
2. The `run_tts()` function no longer overrides the language code to 'en' when G2P is active

This ensures that Amharic phonemes are processed with the 'am' language code during inference, preserving the correct pronunciation context.

## Verifying the Fix

### On Your Local Machine

Run the standalone verification script:

```bash
python amharic_fix_verification_standalone.py
```

This script tests the language normalization and G2P conversion logic without requiring the full TTS model.

### On Lightning AI Machine

1. Push the changes to GitHub:
   ```bash
   git add xtts_demo.py amharic_fix_verification_standalone.py AMHARIC_FIX_README.md
   git commit -m "Fix Amharic TTS pronunciation issues"
   git push
   ```

2. Pull the changes on your Lightning AI machine:
   ```bash
   git pull
   ```

3. Run the verification script:
   ```bash
   python amharic_fix_verification_standalone.py
   ```

4. Test with actual inference using your fine-tuned model.

## G2P Backend Installation

For optimal phoneme conversion, ensure that the G2P backends are properly installed:

```bash
pip install transphone epitran
```

Transphone is the recommended backend for Amharic G2P conversion.

## Technical Details

### Key Files Modified

- `xtts_demo.py`: Fixed language code handling in the `run_tts()` function

### Verification Scripts

- `amharic_fix_verification_standalone.py`: Standalone script to verify the fix
- `test_amharic_inference_fix_verification.py`: Full verification script (requires TTS module)
- `tests/test_language_normalization_fix.py`: Unit tests for language normalization

## Expected Results

After applying the fix:

1. Amharic text will be converted to phonemes during inference
2. The language code will remain 'am' (not overridden to 'en')
3. The model will receive phonemes with the correct language context
4. Pronunciation quality should significantly improve