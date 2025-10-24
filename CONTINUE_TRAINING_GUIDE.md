# Continue Training from Checkpoint - Fidel Pronunciation Fix

## Overview

You've trained for 10 epochs (~15k steps) and your model produces good speech but struggles with specific Amharic fidels: **ጨ ጠ ፀ ጰ ቀ**

**Good news**: You DON'T need to retrain from scratch! You can continue training from your existing checkpoint.

---

## Quick Diagnosis

First, run the diagnostic to understand the issue:

```bash
python diagnose_fidel_coverage.py
```

Follow prompts to check:
1. Are these fidels in your vocabulary?
2. How often do they appear in your dataset?

---

## Solution Paths

### Path 1: Continue Training (Recommended - No Data Loss)

If fidels are in vocab but underrepresented:

#### Step 1: Augment Dataset (Optional but Recommended)

```bash
python augment_fidel_dataset.py
```

This creates augmented data emphasizing problematic fidels.

#### Step 2: Merge Augmented Data

```bash
# Combine original + augmented
cat output/metadata_train.csv augmented_data/metadata_fidel_augmented.csv > output/metadata_train_enhanced.csv

# Same for eval
cat output/metadata_eval.csv augmented_data/metadata_fidel_augmented.csv > output/metadata_eval_enhanced.csv
```

#### Step 3: Continue Training

**Via Web UI:**

1. Launch: `python xtts_demo.py`
2. Go to Tab 2 (Fine-tuning)
3. **CRITICAL**: Select "Custom model" option
4. Point to your trained model directory (e.g., `output/ready/`)
5. Set language: `amh`
6. Set epochs: `5-10` (additional epochs)
7. Use enhanced CSV files
8. Click "Train Model"

**Via Command Line:**

```bash
python headlessXttsTrain.py \
  --continue_from output/ready/model.pth \
  --train_csv output/metadata_train_enhanced.csv \
  --eval_csv output/metadata_eval_enhanced.csv \
  --lang amh \
  --epochs 10 \
  --output_path output/continued_training
```

#### Step 4: Monitor Training

Watch for:
- Loss continuing to decrease
- Eval loss improving on fidel-rich examples
- Test pronunciation every 2-3 epochs

---

### Path 2: Targeted Fine-tuning on Fidels Only

Train a few epochs ONLY on fidel-heavy data, then return to full dataset:

```bash
# Train 3 epochs on fidel data only
python headlessXttsTrain.py \
  --continue_from output/ready/model.pth \
  --train_csv augmented_data/metadata_fidel_augmented.csv \
  --eval_csv augmented_data/metadata_fidel_augmented.csv \
  --lang amh \
  --epochs 3 \
  --output_path output/fidel_focused

# Then continue with full dataset
python headlessXttsTrain.py \
  --continue_from output/fidel_focused/model.pth \
  --train_csv output/metadata_train.csv \
  --eval_csv output/metadata_eval.csv \
  --lang amh \
  --epochs 5 \
  --output_path output/final
```

---

### Path 3: Fix Vocabulary (If Fidels Are Missing)

If diagnostics show fidels are missing from vocab:

#### Step 1: Rebuild Vocabulary

```bash
python utils/vocab_extension.py \
  --input-vocab base_models/v2.0.2/vocab.json \
  --output-vocab output/vocab_fixed.json \
  --dataset-csv output/metadata_train.csv
```

#### Step 2: Unfortunately, This Requires Retraining

If vocab was incomplete, you'll need to retrain with the fixed vocab. But this is rare - usually the extended Amharic vocab includes all Ethiopic characters.

To check:
```python
python diagnose_fidel_coverage.py
# If it says "All fidels present" → use Path 1 or 2
# If it says "Missing fidels" → use Path 3
```

---

## Training Parameters for Continuation

Recommended settings for continuing training:

```python
{
    "epochs": 5-10,           # Start with 5, increase if needed
    "batch_size": 4,          # Keep same as original training
    "grad_accumulation": 2,   # Keep same as original
    "learning_rate": 5e-6,    # Slightly lower than initial (original was likely 1e-5)
    "lr_schedule": "constant_with_warmup",
    "warmup_steps": 100
}
```

### Why Lower Learning Rate?

Your model has already learned general patterns. Use a lower LR for fine-grained adjustments.

---

## Expected Results

After 5-10 additional epochs:

- **Epochs 11-13**: Model starts recognizing fidel patterns
- **Epochs 14-16**: Pronunciation improves noticeably  
- **Epochs 17-20**: Full proficiency on problematic fidels

**Total training**: 20-30 epochs (~30-45k steps)

---

## Testing During Training

Test pronunciation every 2-3 epochs:

```python
from TTS.api import TTS

# Load checkpoint
tts = TTS(model_path="output/continued_training/epoch_12/model.pth")

# Test problematic fidels
test_words = [
    "ጨዋ",   # cha
    "ጠዋት",  # tta
    "ፀሐይ",  # tsa
    "ቀን",   # qha
]

for word in test_words:
    tts.tts_to_file(
        text=word,
        speaker_wav="reference_audio.wav",
        language="amh",
        file_path=f"test_{word}.wav"
    )
```

---

## Troubleshooting

### Issue: "Cannot find checkpoint"

**Solution**: Ensure checkpoint path is correct
```bash
ls output/ready/
# Should show: model.pth, config.json, vocab.json, etc.
```

### Issue: "Vocabulary mismatch"

**Solution**: Use same vocab file as original training
```bash
# Check vocab path in training command
--vocab_file output/ready/vocab.json
```

### Issue: "Loss not decreasing"

**Possible causes**:
1. Learning rate too low → increase to 1e-5
2. Not enough fidel examples → use Path 2 (targeted training)
3. Batch size too small → increase if GPU allows

### Issue: "Still mispronouncing after 10 epochs"

**Solutions**:
1. Record more fidel-specific audio (use `augmented_data/fidel_wordlist.txt`)
2. Try Path 2: Targeted fine-tuning on fidels only
3. Check if using correct language code (`amh` not `am`)

---

## Alternative: Switch to G2P Mode (Advanced)

If character-based training isn't working, consider switching to phoneme-based:

### Pros:
- Explicit phoneme representation
- May handle rare fidels better
- Consistent with linguistic theory

### Cons:
- **Requires full retrain** (can't continue from character-based checkpoint)
- More preprocessing complexity
- Need G2P backend (transphone, epitran, or rule-based)

**Only do this if Paths 1-2 fail after 20 total epochs.**

To enable G2P:
```bash
python headlessXttsTrain.py \
  --input_audio your_audio.wav \
  --lang amh \
  --epochs 20 \
  --use_g2p \
  --g2p_backend transphone  # or epitran, rule_based
```

---

## Summary

**Best Approach**:
1. ✅ Run `diagnose_fidel_coverage.py` to understand the issue
2. ✅ Use `augment_fidel_dataset.py` to create focused training data
3. ✅ Continue training from your epoch 10 checkpoint with enhanced data
4. ✅ Train 5-10 more epochs with slightly lower learning rate
5. ✅ Test pronunciation every few epochs

**Expected time**: 2-4 hours of additional training (depending on GPU)

**Result**: Improved pronunciation while preserving your 15k steps of existing training!

---

## Quick Commands Reference

```bash
# 1. Diagnose
python diagnose_fidel_coverage.py

# 2. Augment data
python augment_fidel_dataset.py

# 3. Merge datasets
cat output/metadata_train.csv augmented_data/metadata_fidel_augmented.csv > output/metadata_train_enhanced.csv

# 4. Continue training (adjust paths as needed)
python headlessXttsTrain.py \
  --continue_from output/ready/model.pth \
  --train_csv output/metadata_train_enhanced.csv \
  --eval_csv output/metadata_eval_enhanced.csv \
  --lang amh \
  --epochs 10 \
  --batch_size 4 \
  --grad_acumm 2
```

---

**Questions?** Check existing docs:
- `VOCAB_EXTENSION_GUIDE.md` - Vocabulary details
- `AMHARIC_TOKENIZER_ARCHITECTURE.md` - How tokenization works
- `diagnose_amharic_issue.py` - General pronunciation diagnostics

Good luck! 🚀
