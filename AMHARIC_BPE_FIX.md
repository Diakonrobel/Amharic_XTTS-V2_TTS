# Amharic BPE-Only Training Fix

## Problem

When training XTTS with Amharic language (`amh` or `am` code) in **BPE-only mode** (without G2P preprocessing), the training crashes with:

```
NotImplementedError: Language 'amh' is not supported.
```

This error occurs in `TTS/tts/layers/xtts/tokenizer.py` at line 676 during dataset loading, even though the Gradio WebUI provides a "Use Amharic G2P" checkbox that should allow BPE-only training.

## Root Cause

The XTTS tokenizer's `preprocess_text()` method checks if a language is supported for G2P preprocessing. Amharic (`amh`) is not in the supported language list, so it raises `NotImplementedError` regardless of whether G2P is actually needed.

The issue occurs during **dataset loading** in `dataset.py:110` → `tokenizer.encode()` → `preprocess_text()`, which happens **before** any instance-level patches can be applied.

## Solution

A **global monkey-patch** applied at the **class level** before any tokenizer instances are created:

### Files Created/Modified

1. **`utils/amharic_bpe_tokenizer_patch.py`** (NEW)
   - Global class-level patch for `VoiceBpeTokenizer`
   - Adds `'am'` and `'amh'` to `char_limits`
   - Redirects Amharic preprocessing to English (returns raw text for BPE)

2. **`utils/gpt_train.py`** (MODIFIED)
   - Imports and applies patch before model initialization (line 22-27)

3. **`xtts_demo.py`** (MODIFIED)
   - Imports and applies patch at module level (line 36-41)

4. **`test_amharic_bpe_fix.py`** (NEW)
   - Verification script to test the fix

## How It Works

### The Patch

```python
from utils.amharic_bpe_tokenizer_patch import apply_global_amharic_bpe_patch
apply_global_amharic_bpe_patch()
```

This patches the `VoiceBpeTokenizer` class to:

1. **Add language codes** to `char_limits`:
   ```python
   VoiceBpeTokenizer.char_limits['am'] = 200
   VoiceBpeTokenizer.char_limits['amh'] = 200
   ```

2. **Wrap `preprocess_text` method**:
   - If language is `'am'` or `'amh'`:
     - Check if text contains IPA markers (for G2P mode detection)
     - If IPA: keep as-is
     - If Ethiopic: redirect to English preprocessing (returns raw text)
   - Other languages: use original behavior

### Why This Works

- **English preprocessing** in XTTS simply returns the text unchanged
- This is perfect for BPE-only mode with Ethiopic script
- All tokenizer instances (model, dataset, trainer) inherit the patched behavior
- No system files are modified (stays in user's project directory)

## Usage

### In Gradio WebUI

1. Set language to `amh` or `am`
2. **Uncheck** "Use Amharic G2P" checkbox
3. Start training

The training will:
- Use raw Ethiopic characters from your dataset
- Apply BPE tokenization directly (no phoneme conversion)
- Not require transphone, epitran, or any G2P dependencies

### Testing the Fix

Run the verification script:

```bash
python test_amharic_bpe_fix.py
```

Expected output:
```
======================================================================
🧪 TESTING AMHARIC BPE-ONLY TOKENIZER FIX
======================================================================

Step 1: Applying global tokenizer patch...
======================================================================
🩹 APPLYING GLOBAL AMHARIC BPE TOKENIZER PATCH
======================================================================
 > ✅ Added 'am' and 'amh' to char_limits
 > ✅ Patched VoiceBpeTokenizer.preprocess_text()
 > ℹ️  All tokenizer instances will now support 'am'/'amh' codes
 > ℹ️  Ethiopic text → raw BPE (no g2p required)
======================================================================

✅ Patch applied

Step 2: Importing TTS libraries...
✅ TTS libraries imported

Step 3: Creating tokenizer instance...
✅ Tokenizer created

Step 4: Testing Amharic text encoding...

  Testing: Hello world in Amharic (ISO 639-3)
    Text: 'ሰላም ዓለም'
    Lang: 'amh'
    ✅ Encoded: 14 tokens
    ✅ Decoded successfully

======================================================================
✅ ALL TESTS PASSED!
======================================================================

🎉 The fix is working correctly!
```

## Technical Details

### Patch Application Order

1. **Before TTS imports**: Patch is imported in module headers
2. **After TTS imports**: `apply_global_amharic_bpe_patch()` is called
3. **Class-level modification**: All future instances inherit the fix
4. **Idempotent**: Safe to call multiple times

### Compatibility

- ✅ Works with existing Amharic G2P mode (detects IPA markers)
- ✅ Works with BPE-only mode (raw Ethiopic)
- ✅ Doesn't affect other languages
- ✅ No system file modifications
- ✅ Version-control friendly (stays in project)

### What Gets Patched

```python
VoiceBpeTokenizer.char_limits  # Class attribute
VoiceBpeTokenizer.preprocess_text  # Class method
```

All instances created after patch automatically inherit:
- Model tokenizer
- Dataset tokenizers (train/eval)
- Trainer tokenizers

## Troubleshooting

### Error: "TTS library not available"

The patch tried to run before TTS was installed. This is OK - it will be applied when TTS is imported.

### Still getting NotImplementedError

1. Check that patch is applied:
   ```python
   from utils.amharic_bpe_tokenizer_patch import patch_status
   print(patch_status())  # Should return True
   ```

2. Restart Python session (old tokenizer instances may exist)

3. Verify imports:
   ```python
   from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
   print('amh' in VoiceBpeTokenizer.char_limits)  # Should be True
   ```

### G2P Mode Still Works?

Yes! The patch detects IPA markers in text:
- If text has `ə ɨ ʔ ʕ` etc. → keeps text as-is (G2P mode)
- If text has only Ethiopic → uses raw BPE (BPE-only mode)

Both modes work seamlessly.

## Benefits Over Previous Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Instance patching** (old) | Simple | Too late - dataset already loaded |
| **System file edit** (old) | Direct | Breaks on TTS updates, not portable |
| **Global class patch** (new) | ✅ Early enough<br>✅ No system mods<br>✅ Version controlled | Requires import order |

## Credits

This fix was developed with assistance from the MCP servers:
- `clear-thought` - Problem analysis and solution design
- `context7-mcp` - XTTS documentation research

Following the user's rule:
> User requires the use of the hybrid g2p system for their target language without fallback to other g2p systems, and mandates calling MCP servers 'clear-thought' and 'context7-mcp' together to fix critical issues.

## Related Files

- `utils/amharic_bpe_tokenizer_patch.py` - The patch implementation
- `utils/gpt_train.py` - Training pipeline (applies patch)
- `xtts_demo.py` - Gradio WebUI (applies patch)
- `test_amharic_bpe_fix.py` - Verification script
- `amharic_tts/config/bpe_only_config.py` - BPE-only configuration presets
