# BPE-Only Training Guide for Amharic TTS

## Overview

This guide explains how to train your XTTS model using **only BPE (Byte Pair Encoding) tokenization** on raw Ethiopic text, **without requiring G2P (Grapheme-to-Phoneme) conversion**.

## When to Use BPE-Only Training

✅ **Use BPE-only when:**
- G2P backends (Transphone/Epitran) are not available or difficult to install
- You want faster preprocessing without phoneme conversion
- Your dataset has good pronunciation coverage
- You prefer the model to learn character-level representations directly
- You're working with Ethiopic script's inherent syllabic structure

❌ **Consider G2P when:**
- You need maximum pronunciation accuracy for rare words
- You have access to G2P backends
- You're fine with slower preprocessing
- Your dataset has inconsistent spelling/pronunciation

## Benefits vs Trade-offs

### Benefits
- ⚡ **Faster preprocessing** - No G2P conversion step
- 🔧 **No external dependencies** - No need for Transphone, Epitran, etc.
- 📝 **Direct learning** - Model learns from Ethiopic characters directly
- 🎯 **Works well for syllabic scripts** - Ethiopic is naturally syllabic

### Trade-offs
- 📊 **Larger vocabulary** - Ethiopic characters require more tokens
- 🎤 **Less explicit phonology** - Model learns patterns implicitly
- ⚠️ **Rare word handling** - May be less accurate for uncommon words

## How to Train (Web UI)

### Step 1: Data Processing
1. Launch the web UI: `python xtts_demo.py`
2. Go to **Tab 1 (Data Processing)**
3. Select `amh` (Amharic) as the language
4. **IMPORTANT:** Do NOT enable "Amharic G2P preprocessing"
5. Upload your Amharic audio files
6. Click "Step 1 - Create dataset"

### Step 2: Fine-tuning
1. Go to **Tab 2 (Fine-tuning)**
2. **IMPORTANT:** Do NOT enable "Amharic G2P for training"
3. Configure training parameters:
   - Epochs: 10-20 (start with 10)
   - Batch size: 2
   - Grad accumulation: 84
4. Click "Step 2 - Run the training"

### Step 3: Inference
1. Go to **Tab 3 (Inference)**
2. **IMPORTANT:** Do NOT enable "Use Amharic G2P for inference"
3. Test your trained model with Amharic text
4. The model will process raw Ethiopic characters directly

## How to Train (Headless)

```bash
# Basic BPE-only training (NO G2P)
python headlessXttsTrain.py \
  --input_audio amharic_speaker.wav \
  --lang amh \
  --epochs 10 \
  --batch_size 2 \
  --grad_acumm 84

# DO NOT add --use_g2p flag!
# That would enable G2P preprocessing
```

## How to Train (Python API)

```python
from utils.gpt_train import train_gpt

# Train with BPE-only (G2P disabled)
train_gpt(
    custom_model=None,
    version="v2.0.2",
    language="amh",
    num_epochs=10,
    batch_size=2,
    grad_acumm=84,
    train_csv="output/metadata_train.csv",
    eval_csv="output/metadata_eval.csv",
    output_path="./output",
    use_amharic_g2p=False,  # ← IMPORTANT: Set to False!
    g2p_backend_train=None,  # Not used when G2P disabled
)
```

## Configuration File Approach

For advanced users, you can use the dedicated BPE-only configuration:

```python
from amharic_tts.config.bpe_only_config import (
    create_bpe_only_config,
    validate_bpe_only_environment
)

# Validate environment
success, message = validate_bpe_only_environment()
print(message)

# Create BPE-only configuration
config = create_bpe_only_config(preset="quality")

# Convert to full config
full_config = config.to_amharic_config()

# Use in training (pass to train_gpt or your training script)
```

### Available Presets

```python
# Fast training (smaller vocab)
config = create_bpe_only_config("fast")
# - Vocab size: 512
# - Faster training, less nuanced

# Quality training (larger vocab) 
config = create_bpe_only_config("quality")
# - Vocab size: 2048
# - Better representation, slower training

# Minimal (for testing)
config = create_bpe_only_config("minimal")
# - Vocab size: 256
# - Quick tests only

# Custom
config = create_bpe_only_config(
    preset="default",
    vocab_size=1536,  # Custom size
    normalize_variants=True
)
```

## Text Preprocessing

Even without G2P, the system applies these normalizations:

### Ethiopic Character Variants
```
Input:  ሥላም (variant ሥ)
Output: ስላም (normalized ስ)

Input:  ዕለት (variant ዕ)  
Output: እለት (normalized እ)
```

### Number Expansion (Optional)
```python
from amharic_tts.preprocessing.number_expander import AmharicNumberExpander

expander = AmharicNumberExpander()
text = expander.expand_number("2024")  # → "ሁለት ሺህ ሃያ አራት"
```

### Punctuation Preservation
```
Ethiopic punctuation is preserved:
። (sentence end)
፣ (comma)
፤ (semicolon)
፥ (colon)
```

## Verifying BPE-Only Mode

Run this validation script to confirm G2P is disabled:

```python
from amharic_tts.config.bpe_only_config import validate_bpe_only_environment

success, message = validate_bpe_only_environment()
print(message)
```

Expected output:
```
==================================================================
🔍 BPE-ONLY TRAINING ENVIRONMENT VALIDATION
==================================================================
✅ Tokenizer          : BPE mode confirmed
✅ TTS Library        : XTTS available
✅ PyTorch            : CUDA: True
==================================================================

✅ All checks passed - BPE-only training ready!
```

## Troubleshooting

### Problem: G2P is accidentally enabled

**Symptoms:**
- Training logs show "Applying Amharic G2P preprocessing"
- You see IPA phonemes in preprocessed text
- Transphone/Epitran errors appear

**Solution:**
```python
# In Web UI: Uncheck ALL G2P checkboxes in Tabs 1, 2, 3

# In Python:
train_gpt(
    ...
    use_amharic_g2p=False,  # ← Make sure this is False!
    ...
)
```

### Problem: UNK (unknown) tokens in output

**Symptoms:**
- Model outputs contain <UNK> tokens
- Poor quality synthesis

**Causes:**
1. Vocabulary too small - increase vocab_size
2. Characters not in training data
3. Special characters not normalized

**Solution:**
```python
# Increase vocabulary size
config = create_bpe_only_config(vocab_size=2048)

# Enable character normalization
config.normalize_ethiopic_variants = True
```

### Problem: Slow training or convergence

**Solution:**
```python
# Adjust learning rate for character-level learning
learning_rate_override = 5e-6  # Slightly higher than default

# Reduce batch size if memory issues
batch_size = 1
grad_acumm = 168  # Compensate with more accumulation
```

## Comparing BPE vs G2P Results

| Aspect | BPE-Only | G2P + BPE |
|--------|----------|-----------|
| **Preprocessing Speed** | ⚡ Fast (seconds) | 🐌 Slow (minutes) |
| **Dependencies** | ✅ None | ⚠️ Transphone/Epitran |
| **Rare Word Accuracy** | 😐 Good | ✅ Excellent |
| **Common Word Accuracy** | ✅ Excellent | ✅ Excellent |
| **Vocabulary Size** | 📊 Larger (1024-2048) | 📊 Smaller (512-1024) |
| **Training Time** | ⚡ Faster | 🐌 Slower |
| **Model Size** | 📦 Slightly larger | 📦 Standard |

## Best Practices

### 1. Start with BPE-Only
For most users, BPE-only is the recommended starting point:
```bash
python headlessXttsTrain.py \
  --input_audio speaker.wav \
  --lang amh \
  --epochs 10
```

### 2. Use Character Normalization
Always enable Ethiopic variant normalization:
```python
config.normalize_ethiopic_variants = True
```

### 3. Choose Appropriate Vocab Size
```python
# Small dataset (<1000 samples)
vocab_size = 512

# Medium dataset (1000-5000 samples)
vocab_size = 1024

# Large dataset (>5000 samples)
vocab_size = 2048
```

### 4. Monitor Training
Watch for:
- ✅ Decreasing loss
- ✅ No <UNK> tokens in validation
- ⚠️ Overfitting (val loss increases)

## Example: Complete BPE-Only Workflow

```python
#!/usr/bin/env python3
"""
Complete example of BPE-only Amharic TTS training
"""

# Step 1: Validate environment
from amharic_tts.config.bpe_only_config import validate_bpe_only_environment

success, msg = validate_bpe_only_environment()
assert success, f"Environment check failed: {msg}"

# Step 2: Prepare dataset (assuming already done)
train_csv = "output/metadata_train.csv"
eval_csv = "output/metadata_eval.csv"

# Step 3: Train with BPE-only
from utils.gpt_train import train_gpt

train_gpt(
    custom_model=None,
    version="v2.0.2",
    language="amh",  # Amharic
    num_epochs=10,
    batch_size=2,
    grad_acumm=84,
    train_csv=train_csv,
    eval_csv=eval_csv,
    output_path="./output",
    max_audio_length=255995,
    save_step=1000,
    
    # BPE-ONLY SETTINGS (no G2P!)
    use_amharic_g2p=False,  # ← Critical: Disable G2P
    g2p_backend_train=None,  # Not used
    
    # Optional optimizations
    enable_grad_checkpoint=False,
    enable_sdpa=False,
    freeze_encoder=True,
    learning_rate_override=5e-6
)

print("✅ BPE-only training complete!")
```

## FAQ

**Q: Will BPE-only training work as well as G2P?**
A: For most use cases, yes! BPE-only works very well for Amharic, especially with medium to large datasets. The syllabic nature of Ethiopic script helps the model learn phonological patterns.

**Q: Can I switch between BPE-only and G2P?**
A: Yes, but you'll need to retrain from scratch. The tokenization is fundamentally different.

**Q: What vocab size should I use?**
A: Start with 1024 for most cases. Increase to 2048 for large datasets or better quality.

**Q: Do I need to install Transphone or Epitran?**
A: No! That's the whole point of BPE-only training. Zero G2P dependencies.

**Q: How do I verify G2P is really disabled?**
A: Check training logs - you should NOT see "Applying Amharic G2P preprocessing". Also run `validate_bpe_only_environment()`.

## Conclusion

BPE-only training is a **robust, dependency-free approach** for Amharic TTS that works excellently for most use cases. It's:
- ✅ Faster to train
- ✅ Easier to set up (no G2P dependencies)
- ✅ Works well with Ethiopic script's syllabic structure
- ✅ Recommended for most users

Only consider G2P if you specifically need:
- Maximum accuracy for rare/unusual words
- Explicit phonological control
- You already have G2P backends installed

**For most Amharic TTS projects, BPE-only is the recommended approach!**
