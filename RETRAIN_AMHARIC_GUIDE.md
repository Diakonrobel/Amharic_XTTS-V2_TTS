# Proper Amharic Model Retraining Guide

## Why Retrain?

Your current model has a vocabulary mismatch (7536 vs 7537 tokens) that makes correct pronunciation impossible. This guide ensures you retrain with perfect vocab consistency.

## ⚠️ Critical Rules

1. **ONE vocabulary file** used for EVERYTHING
2. **Same G2P setting** for dataset and training
3. **Language code 'amh'** throughout
4. **Document everything** for future reference

---

## Step-by-Step Retraining

### Phase 1: Prepare Clean Vocabulary

```bash
# On your local machine or server

# Option A: Use existing vocab (recommended if it has all phonemes)
cp vocab_extended_amharic.json vocab_reference_7537.json

# Option B: Create fresh extended vocab
python utils/vocab_extension.py \
    --base_vocab base_vocab.json \
    --output vocab_reference_7537.json \
    --add_amharic_phonemes

# Verify size
python -c "import json; print(len(json.load(open('vocab_reference_7537.json'))['model']['vocab']), 'tokens')"
```

**Save this vocab file!** You'll use it for everything.

### Phase 2: Prepare Dataset with G2P

```bash
# Preprocess your Amharic audio + transcripts with G2P

python preprocess_amharic_dataset.py \
    --audio_dir /path/to/amharic/audio \
    --transcript_file /path/to/transcripts.txt \
    --output_dir dataset_amharic_g2p \
    --lang amh \
    --use_g2p \
    --g2p_backend rule_based \
    --vocab vocab_reference_7537.json
```

**What this does:**
- Converts Amharic text → IPA phonemes
- Creates `metadata.csv` with phonemized text
- Copies audio files
- Uses YOUR reference vocab for tokenization

**Verify:**
```bash
# Check output
head dataset_amharic_g2p/metadata.csv
# Should show phonemes, not Amharic characters
```

### Phase 3: Create Training Dataset

```bash
# Tab 1 in WebUI OR use command line:

python xtts_demo.py --create_dataset \
    --audio_dir dataset_amharic_g2p/wavs \
    --metadata dataset_amharic_g2p/metadata.csv \
    --lang amh \
    --out_path ready_amharic \
    --vocab vocab_reference_7537.json \
    --eval_split 0.1
```

**Critical:** This must use the SAME `vocab_reference_7537.json`!

**Verify:**
```bash
# Check that ready/ uses your vocab
python -c "import json; print(len(json.load(open('ready_amharic/vocab.json'))['model']['vocab']), 'tokens')"
# MUST show 7537
```

### Phase 4: Train with Proper Config

```bash
# Headless training (recommended):

python headlessXttsTrain.py \
    --input_audio your_speaker_reference.wav \
    --lang amh \
    --train_csv ready_amharic/metadata_train.csv \
    --eval_csv ready_amharic/metadata_eval.csv \
    --vocab ready_amharic/vocab.json \
    --epochs 30 \
    --batch_size 4 \
    --grad_acumm 4 \
    --save_step 5000 \
    --use_g2p \
    --g2p_backend rule_based \
    --out_path training_amharic_g2p
```

**Key parameters:**
- `--lang amh` - Use ISO 639-3 code
- `--use_g2p` - Match dataset preprocessing
- `--vocab ready_amharic/vocab.json` - Use dataset's vocab
- `--epochs 30` - Minimum for good quality (50-100 better)

### Phase 5: Monitor Training

Watch for:
- Loss should decrease steadily
- After epoch 10-15, test samples should sound better
- By epoch 30, pronunciation should be mostly correct

```bash
# Check training logs
tail -f training_amharic_g2p/trainer_0_log.txt

# Test intermediate checkpoints
# Use best_model.pth when training completes
```

### Phase 6: Verify Before Deployment

```bash
# Run diagnostics on trained model:

python diagnose_amharic_issue.py
# When prompted:
#   Checkpoint: training_amharic_g2p/ready/best_model.pth
#   Vocab: training_amharic_g2p/ready/vocab.json

# Should show:
#   ✅ Vocabulary sizes match (7537 = 7537)
#   ✅ IPA phoneme markers in vocab: YES
```

### Phase 7: Test Inference

```python
# Test with G2P enabled (matching training)

# In Gradio:
# - Text: "ሰላም ኢትዮጵያ"
# - Language: amh
# - Enable Amharic G2P: ✅ CHECKED
# - Generate

# Should produce clear Amharic speech!
```

---

## Configuration Checklist

Before starting training, verify:

- [ ] Have ONE reference vocabulary file (7537 tokens recommended)
- [ ] Dataset preprocessing used G2P with that vocab
- [ ] Training config points to that same vocab
- [ ] Language code is 'amh' everywhere
- [ ] G2P enabled consistently (dataset + training + inference)
- [ ] Have good quality Amharic audio (minimum 30 minutes, 1-2 hours better)

---

## Training Configuration Template

Save this as `TRAINING_CONFIG_v2.md` alongside your checkpoint:

```markdown
# Amharic XTTS Training Configuration v2

## Model Info
- Checkpoint: `best_model.pth`
- Training Date: 2025-01-12
- Total Epochs: 30
- Final Loss: 0.X

## Vocabulary
- File: `vocab_reference_7537.json`
- Size: **7537 tokens**
- Includes: Amharic phonemes (IPA)

## Preprocessing
- G2P Enabled: ✅ YES
- G2P Backend: rule_based
- Language: amh (ISO 639-3)
- Dataset Size: 2 hours

## Training
- Lang Code: amh
- G2P During Training: ✅ YES  
- Batch Size: 4
- Gradient Accumulation: 4
- Effective Batch Size: 16

## Inference Requirements
**CRITICAL:** Use these exact settings:

```python
lang = "amh"
use_g2p = True
g2p_backend = "rule_based"
vocab_file = "vocab_reference_7537.json"  # MUST be 7537 tokens
```

## Audio Quality
- Sample Rate: 22050 Hz
- Channels: Mono
- Format: WAV
```

---

## Common Pitfalls to Avoid

### ❌ DON'T:
1. Use different vocab files in different stages
2. Mix G2P and non-G2P datasets
3. Use 'am' and 'amh' language codes interchangeably
4. Train for <10 epochs and expect good results
5. Change vocab size mid-training

### ✅ DO:
1. Use ONE vocabulary file throughout
2. Enable G2P consistently (or disable consistently)
3. Always use 'amh' language code
4. Train for minimum 30 epochs (50-100 for production)
5. Keep detailed training notes

---

## Quick Start Commands

```bash
# Full workflow in one script:

# 1. Prepare
python create_reference_vocab.sh  # Creates vocab_reference_7537.json

# 2. Preprocess
python preprocess_amharic_dataset.py \
    --audio_dir raw_audio/ \
    --use_g2p \
    --vocab vocab_reference_7537.json

# 3. Create dataset
python xtts_demo.py --create_dataset \
    --vocab vocab_reference_7537.json \
    ...

# 4. Train
python headlessXttsTrain.py \
    --lang amh \
    --use_g2p \
    --vocab ready/vocab.json \
    --epochs 30

# 5. Verify
python diagnose_amharic_issue.py

# 6. Deploy
# Copy best_model.pth and vocab to production
# Use G2P at inference
```

---

## Expected Timeline

- **Data prep**: 1-2 hours
- **Dataset creation**: 30 minutes  
- **Training (30 epochs)**: 
  - GPU (V100): ~4-6 hours
  - GPU (T4): ~8-12 hours
  - GPU (A100): ~2-3 hours
- **Testing**: 30 minutes

**Total: ~1 day** for complete retraining with proper setup

---

## When Training Completes

1. **Test thoroughly** with multiple Amharic sentences
2. **Compare** with previous model
3. **Document** what works and what doesn't
4. **Save** training config with checkpoint
5. **Backup** your reference vocab file

Remember: **Vocabulary consistency is EVERYTHING!** 

Good luck with retraining! 🚀🇪🇹
