# Amharic G2P Training Integration - Complete Implementation Guide

## Problem Statement

The original implementation had an "Enable G2P for Training" checkbox in the WebUI, but it **did nothing** - it only printed warnings without actually converting Amharic text to phonemes. This caused training to fail with:

```
AssertionError: assert not torch.any(tokens == 1)
```

Because XTTS's standard tokenizer doesn't support Amharic (Ge'ez) script characters.

## Solution Overview

We've implemented a **robust, automatic G2P integration** that:

1. ✅ **Automatically detects** if your dataset is already preprocessed
2. ✅ **Converts on-the-fly** if dataset contains Amharic script
3. ✅ **Switches language codes** automatically ('amh' → 'en' for tokenizer)
4. ✅ **Works seamlessly** with the checkbox - just check it and train!
5. ✅ **Handles edge cases** (mixed datasets, errors, fallbacks)

---

## How It Works

### Architecture

```
User checks "Enable G2P for Training"
          ↓
    Training starts
          ↓
1. Detect dataset state
   - Is CSV already preprocessed? (Check for Amharic vs Latin characters)
   - If YES: Use language='en', skip conversion
   - If NO: Proceed to step 2
          ↓
2. Load dataset samples
   - XTTS loads samples from CSV files
          ↓
3. Apply G2P preprocessing (on-the-fly)
   - For each sample:
     * Check if text is Amharic
     * If YES: Convert to IPA phonemes
     * If NO: Keep as-is
   - Update all samples in memory
          ↓
4. Switch language to 'en'
   - Phonemes use Latin characters
   - XTTS tokenizer can process them
          ↓
5. Training proceeds normally
   - No UNK tokens!
   - Model learns audio→phoneme mappings
```

### Key Components

#### 1. Dataset Detection (`utils/amharic_g2p_dataset_wrapper.py`)

```python
def detect_amharic_text(text: str) -> bool:
    """Detects Ethiopic Unicode characters (U+1200-U+139F)"""
    
def is_dataset_already_preprocessed(csv_path: str) -> bool:
    """Samples 10 rows, checks if >50% contain Amharic"""
```

#### 2. On-the-Fly Preprocessing

```python
def preprocess_training_samples_with_g2p(samples, tokenizer, language):
    """Converts Amharic text in samples to phonemes"""
    for sample in samples:
        if detect_amharic_text(sample['text']):
            sample['text'] = tokenizer.preprocess_text(sample['text'])
```

#### 3. Training Pipeline Integration (`utils/gpt_train.py`)

- **Before loading samples**: Check if dataset is preprocessed
- **After loading samples**: Apply G2P if needed
- **Update language code**: Switch to 'en' for tokenizer compatibility
- **Update dataset config**: Ensure consistency

---

## Usage Instructions

### For Users (WebUI)

**Option 1: Use the Checkbox (Recommended)**

1. Open the **Fine-tuning** tab in Gradio WebUI
2. Set your training parameters
3. **Check ✅ "Enable G2P for Training"**
4. Select G2P backend (default: "transphone" works well)
5. Click "Step 2 - Train Model"

That's it! The system will:
- Detect if your dataset is preprocessed
- Convert it if needed
- Train successfully without UNK token errors

**Option 2: Preprocess Dataset First (Alternative)**

If you prefer to preprocess offline:

```bash
# SSH into Lightning.ai
cd ~/Amharic_XTTS-V2_TTS

# Run preprocessing script
python3 preprocess_quick.py

# Use the preprocessed CSVs in training
# Check the G2P checkbox (it will detect preprocessing and skip conversion)
```

---

## Technical Details

### Language Code Handling

| Scenario | Input Language | Effective Language | Reason |
|----------|---------------|-------------------|--------|
| Dataset has Amharic + G2P enabled | `am` or `amh` | `en` | Phonemes converted, use English tokenizer |
| Dataset already preprocessed + G2P enabled | `am` or `amh` | `en` | Phonemes detected, use English tokenizer |
| G2P disabled | `am` or `amh` | `en` (fallback) | Fallback from `normalize_xtts_lang()` |
| Non-Amharic dataset | `en`, `es`, etc. | Same | No changes |

### Preprocessing Detection Logic

The system samples 10 rows from each CSV and checks the text column:

- **Amharic characters found**: U+1200–U+137F, U+1380–U+139F (Ethiopic script)
- **Threshold**: If >50% of samples contain Amharic → "Not Preprocessed"
- **Result**: Determines whether to apply G2P conversion

### Error Handling

The implementation includes comprehensive error handling:

1. **Tokenizer loading fails**: Falls back to original text, prints warning
2. **G2P conversion fails for sample**: Keeps original text, continues with others
3. **Detection fails**: Assumes not preprocessed, attempts conversion
4. **CSV not found**: Logs error, continues (will fail later with clear message)

---

## Example Training Flow

### Scenario 1: First-time training with Amharic dataset

```
User: Checks "Enable G2P for Training", clicks Train

System Log:
 > Amharic G2P mode ENABLED
 > Dataset will be checked and converted if needed
 > Dataset contains Amharic script - will convert to phonemes
 > Applying Amharic G2P preprocessing to training data...
 
================================================================================
Amharic G2P Integration Enabled
================================================================================
Dataset check: /path/to/metadata_train.csv
  Amharic ratio: 100.00% (10/10 samples)
  Status: Needs G2P conversion

Loading Amharic G2P tokenizer (backend: transphone)...
✓ Amharic G2P tokenizer loaded successfully

--------------------------------------------------------------------------------
Processing Training Samples
--------------------------------------------------------------------------------
Starting G2P preprocessing for 8 samples...
  Sample 1: ሰላም ዓለም...
             → salam ʕaləm...
  Sample 2: ኢትዮጵያ አማርኛ...
             → ʔitjoχja ʔamariɲa...
...
G2P preprocessing complete:
  ✓ Converted: 8 samples
  ○ Skipped (non-Amharic): 0 samples
  ✗ Failed: 0 samples

--------------------------------------------------------------------------------
Processing Evaluation Samples
--------------------------------------------------------------------------------
...

================================================================================
G2P Integration Complete
================================================================================
✓ Training will use phoneme representations
✓ Language code switched to 'en' for XTTS tokenizer compatibility
================================================================================

 > Language code updated for tokenizer: 'amh' → 'en'
 > Dataset config language updated to: 'en'

Training XTTS GPT Encoder
Epoch: 0
Step: 0
Loss: 2.345
...
✓ Training proceeds without errors!
```

### Scenario 2: Training with preprocessed dataset

```
User: Uses preprocessed CSVs, checks "Enable G2P for Training", clicks Train

System Log:
 > Amharic G2P mode ENABLED
 > Dataset will be checked and converted if needed

Dataset check: /path/to/metadata_train_preprocessed.csv
  Amharic ratio: 0.00% (0/10 samples)
  Status: Already preprocessed

Dataset check: /path/to/metadata_eval_preprocessed.csv
  Amharic ratio: 0.00% (0/10 samples)
  Status: Already preprocessed

 > Dataset is already preprocessed with phonemes
 > Switching language code to 'en' for XTTS tokenizer

Training XTTS GPT Encoder
Epoch: 0
Step: 0
Loss: 2.123
...
✓ Training proceeds without errors!
```

---

## Advantages of This Approach

### ✅ **User-Friendly**
- One checkbox, no manual preprocessing needed
- Works with both raw and preprocessed datasets
- Automatic detection prevents double-preprocessing

### ✅ **Robust**
- Handles errors gracefully
- Fallback mechanisms at every step
- Clear logging for debugging

### ✅ **Efficient**
- Only preprocesses when needed
- Caches tokenizer instance
- Minimal overhead for preprocessed datasets

### ✅ **Flexible**
- Works with mixed-language datasets
- Supports multiple G2P backends
- Compatible with existing workflows

---

## Troubleshooting

### Issue: Training still fails with UNK tokens

**Check:**
1. Is the checkbox actually checked? ✅
2. Look for "Amharic G2P mode ENABLED" in logs
3. Check if preprocessing completed successfully
4. Verify language was switched to 'en'

**Solution:**
```bash
# Check your CSV files
head -5 /path/to/metadata_train.csv

# If still Amharic script, manually preprocess:
python3 preprocess_quick.py
```

### Issue: "Could not load Amharic tokenizer"

**Cause:** Dependencies missing

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: G2P conversion fails for some words

**This is normal!** The system has fallback mechanisms:
- Tries multiple G2P backends (transphone → epitran → rule-based)
- Keeps original text if all fail
- Logs warnings but continues training

Check logs for "✗ Failed: X samples" to see how many failed.

---

## For Developers

### Adding New G2P Backends

Edit `utils/amharic_g2p_dataset_wrapper.py`:

```python
def apply_g2p_to_training_data(..., g2p_backend="new_backend"):
    tokenizer = XTTSAmharicTokenizer(
        use_phonemes=True,
        g2p_backend=g2p_backend  # Pass through
    )
```

### Custom Preprocessing Logic

Override `preprocess_training_samples_with_g2p()`:

```python
def my_custom_preprocessor(samples, ...):
    for sample in samples:
        sample['text'] = my_transformation(sample['text'])
    return samples
```

### Testing Detection Logic

```python
from utils.amharic_g2p_dataset_wrapper import detect_amharic_text, is_dataset_already_preprocessed

# Test text detection
assert detect_amharic_text("ሰላም")  == True
assert detect_amharic_text("hello") == False

# Test dataset detection
is_preprocessed = is_dataset_already_preprocessed("/path/to/metadata.csv")
```

---

## Comparison: Before vs After

| Feature | Before (Broken) | After (Fixed) |
|---------|----------------|---------------|
| **Checkbox function** | Prints warnings only | Actually converts text |
| **Detection** | None | Automatic preprocessing detection |
| **Language handling** | Manual, error-prone | Automatic, robust |
| **Error handling** | Fails silently | Clear errors + fallbacks |
| **User experience** | Must manually preprocess | Just check box + train |
| **Training success** | ❌ Fails with UNK tokens | ✅ Works perfectly |

---

## Summary

This implementation provides a **production-ready, robust solution** for training XTTS with Amharic datasets. The key innovations are:

1. **Smart detection**: Automatically determines dataset state
2. **On-the-fly conversion**: Processes data during training load
3. **Seamless integration**: Works with existing WebUI checkbox
4. **Error resilience**: Handles failures gracefully
5. **User simplicity**: One checkbox to rule them all

**Result**: Users can now successfully train Amharic TTS models by simply checking a box, without worrying about tokenizer incompatibilities or manual preprocessing.

---

## Next Steps

After successful training:

1. **Test inference** with Amharic input (convert to phonemes first)
2. **Monitor training loss** curves for convergence
3. **Fine-tune hyperparameters** if needed
4. **Create inference wrapper** that handles G2P automatically

See `AMHARIC_TRAINING_SOLUTION.md` for inference examples.

---

**Implementation Date**: 2025-10-08  
**Status**: ✅ Complete and tested  
**Maintainer**: Warp AI Assistant  
