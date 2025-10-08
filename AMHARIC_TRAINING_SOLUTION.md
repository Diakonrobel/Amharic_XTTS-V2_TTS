# Amharic XTTS Training Solution

## Problem Overview

Your Amharic XTTS training was failing with the following error:

```
AssertionError: assert not torch.any(tokens == 1)
```

**Root Cause:** XTTS's standard tokenizer doesn't support Amharic (Ge'ez) script characters. When the tokenizer encounters Amharic text, it returns UNK (unknown) tokens with ID 1, which triggers the assertion failure during training.

## Solution: Preprocess Dataset with G2P

The solution is to **convert your Amharic text to IPA phonemes** before training. This way:
- The standard XTTS tokenizer can process the phonemes (which use Latin characters)
- Your model will learn to map audio to phoneme sequences
- At inference time, you can use the same G2P conversion for input text

### Why This Works

XTTS learns the relationship between:
1. **Audio features** (spectrograms)
2. **Text representations** (token sequences)

It doesn't matter if the text is in Amharic script or phonemes - as long as the tokenizer can process it and the model can learn the pattern. By converting Amharic → IPA phonemes, you're giving the model a representation it can work with.

---

## Step-by-Step Solution

### Step 1: Pull Latest Changes on Lightning.ai

SSH into your Lightning instance and pull the preprocessing scripts:

```bash
cd ~/Amharic_XTTS-V2_TTS
git pull origin main
```

### Step 2: Run the Preprocessing Script

Option A - **Quick Auto-Detection (Recommended)**:

```bash
python3 preprocess_quick.py
```

This script will:
- Auto-detect your project directory
- Find your filtered CSV files
- Convert all Amharic text to IPA phonemes
- Save new CSV files with `_preprocessed` suffix

Option B - **Manual Specification**:

```bash
python3 preprocess_amharic_dataset.py \
  --input-train /path/to/metadata_train_filtered.csv \
  --input-eval /path/to/metadata_eval_filtered.csv \
  --output-train /path/to/metadata_train_preprocessed.csv \
  --output-eval /path/to/metadata_eval_preprocessed.csv
```

**Example Output:**

```
================================================================================
🔧 Amharic Dataset Preprocessor for XTTS
================================================================================

📦 Loading Amharic tokenizer with G2P...
   ✅ Tokenizer loaded successfully

================================================================================
Processing Training Data
================================================================================

📝 Processing: .../metadata_train_filtered.csv
   Output to: .../metadata_train_filtered_preprocessed.csv
   ✓ ሰላም ዓለም...
     → salam ʕaləm...
   ✓ ኢትዮጵያ አማርኛ...
     → ʔitjoχja ʔamariɲa...
   ✅ Processed 8 Amharic entries

================================================================================
📊 Summary
================================================================================
Training:   8 processed, 0 failed
Evaluation: 2 processed, 0 failed

✅ Preprocessed files saved:
   📄 .../metadata_train_filtered_preprocessed.csv
   📄 .../metadata_eval_filtered_preprocessed.csv
```

### Step 3: Update Training Configuration in WebUI

1. **Access the WebUI** using your Gradio share URL
2. Navigate to the **Training** tab
3. **Update the CSV paths** to use the preprocessed files:
   - Training CSV: `/path/to/metadata_train_filtered_preprocessed.csv`
   - Eval CSV: `/path/to/metadata_eval_filtered_preprocessed.csv`
4. **Change the language setting to `en`** (English)
   - This is important! The phonemes use Latin characters, so the tokenizer needs to treat them as English
5. Keep all other settings the same

### Step 4: Start Training

Click the "Train XTTS Model" button and monitor the output. You should now see:

```
Epoch: 0
Step: 0
Loss: X.XXX
```

Without any `AssertionError` about unknown tokens!

---

## Understanding the Preprocessed Data

### Before Preprocessing:
```
/path/to/audio1.wav|ሰላም ዓለም|speaker1
/path/to/audio2.wav|ኢትዮጵያ አማርኛ|speaker1
```

### After Preprocessing:
```
/path/to/audio1.wav|salam ʕaləm|speaker1
/path/to/audio2.wav|ʔitjoχja ʔamariɲa|speaker1
```

The audio files remain the same - only the text column is converted to IPA phonemes.

---

## For Inference (After Training)

When you want to use your trained model for inference:

1. **Load the trained model** as usual
2. **Convert input Amharic text to phonemes** before inference:

```python
from amharic_tts.tokenizer.xtts_tokenizer_wrapper import XTTSAmharicTokenizer

# Create tokenizer with G2P
tokenizer = XTTSAmharicTokenizer(use_phonemes=True)

# Convert Amharic text to phonemes
amharic_text = "ሰላም ዓለም"
phoneme_text = tokenizer.preprocess_text(amharic_text, lang="am")

# Use phoneme text for TTS
run_tts(lang="en", tts_text=phoneme_text, ...)
```

Or use the WebUI and enable the "Use Amharic G2P" checkbox (if implemented).

---

## Alternative Solution (Advanced)

If you want deeper integration, you can modify the XTTS training pipeline to use the custom Amharic tokenizer directly. This requires:

1. Modifying `utils/gpt_train.py` to inject the custom tokenizer
2. Patching the XTTS dataset class to use the custom tokenizer
3. Ensuring vocab compatibility

However, the preprocessing approach is **simpler, faster, and equally effective** for most use cases.

---

## Troubleshooting

### Issue: "Could not load tokenizer"

**Solution:** Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: "No Amharic text found"

**Solution:** Check your CSV file encoding. It should be UTF-8:
```bash
file -i metadata_train.csv
```

If not UTF-8, convert it:
```bash
iconv -f ISO-8859-1 -t UTF-8 metadata_train.csv > metadata_train_utf8.csv
```

### Issue: Training still fails with UNK tokens

**Solution:** Make sure you:
1. Updated both train and eval CSV paths in the WebUI
2. Changed language to `en` in the WebUI
3. Are using the preprocessed files (with `_preprocessed` suffix)

### Issue: G2P conversion fails for some words

**Solution:** The G2P system has fallback mechanisms. If a word fails:
- It tries multiple backends (transphone, epitran, rule-based)
- Falls back to keeping original text if all fail
- Check the preprocessing output for failed entries

---

## Summary

✅ **Problem Solved:** XTTS tokenizer can't handle Amharic script  
✅ **Solution:** Convert Amharic text to IPA phonemes before training  
✅ **Result:** Training proceeds without UNK token errors  
✅ **Bonus:** You have reusable preprocessing scripts for future datasets  

The key insight is that XTTS doesn't care about the *script* - it learns audio-to-token mappings. By using phonemes, you're giving it a representation it understands while maintaining the phonetic information of Amharic.

---

## Next Steps After Successful Training

1. ✅ Monitor training loss curves
2. ✅ Test inference with sample Amharic text (converted to phonemes)
3. ✅ Fine-tune hyperparameters if needed
4. ✅ Create a wrapper function for easy Amharic TTS (handles G2P automatically)

Good luck with your training! 🚀
