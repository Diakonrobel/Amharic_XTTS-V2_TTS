# Amharic Pronunciation Fix Guide

## Problem: Nonsense/Incorrect Amharic Speech Output

If your Amharic TTS model produces nonsense or incorrect pronunciation, this is typically caused by one of these issues:

### Root Causes

1. **Vocabulary Size Mismatch** (Most Critical)
   - Training used a different vocab size than inference
   - Causes: Every token maps to wrong embedding → complete gibberish

2. **G2P Mode Mismatch**
   - Training with G2P but inference without (or vice versa)
   - Model learned phonemes but receives characters (or vice versa)

3. **Language Code Inconsistency**
   - Training with 'am' but inference with 'amh' (or vice versa)
   - Triggers different tokenization/preprocessing paths

4. **Insufficient Training**
   - Model hasn't learned Amharic patterns yet
   - Needs more epochs or better data

---

## Diagnostic Steps

### Step 1: Run Diagnostics Script

```bash
python diagnose_amharic_issue.py
```

When prompted, provide:
- Path to your checkpoint file (.pth)
- Path to your vocab file (.json)

**Look for**:
- ❌ Vocabulary size mismatch → **MUST RETRAIN**
- ⚠️ G2P markers presence/absence → Try different inference modes
- ✅ Sizes match → Test inference modes

### Step 2: Test Different Inference Modes

If vocab sizes match, the issue is likely G2P/language code mismatch.

Run the test script:
```bash
python test_amharic_modes.py
```

Or manually test via Gradio UI with text "ሰላም":

| Test | Language Code | G2P Enabled | What It Tests |
|------|---------------|-------------|---------------|
| 1 | amh | ❌ No | Character-based, ISO 639-3 |
| 2 | am | ❌ No | Character-based, ISO 639-1 |
| 3 | en | ❌ No | Treats Amharic as English |
| 4 | amh | ✅ Yes | Phoneme-based with 'amh' |
| 5 | en | ✅ Yes | Phoneme mode (TTS standard) |

**Listen to each output and identify which sounds correct!**

---

## Solutions Based on Diagnosis

### Solution A: Vocabulary Size Mismatch (CRITICAL)

**Symptom**: `diagnose_amharic_issue.py` shows different vocab sizes

**Example**:
```
Checkpoint vocabulary size: 7536
Vocab file vocabulary size: 7537
❌ MISMATCH DETECTED!
```

**Why This Happens**:
- Checkpoint trained with 7536-token vocab
- Loading with 7537-token vocab
- Every token ID is off by 1 → complete misalignment

**Fix**: **YOU MUST RETRAIN** - No workaround exists

#### Retraining Steps:

1. **Find or create matching vocab**:
   ```bash
   # Option 1: Find original vocab used in training
   # Look in your ready/ directory for vocab with 7536 tokens
   
   # Option 2: Use current vocab and retrain from scratch
   # This is recommended for consistency
   ```

2. **Ensure consistent vocab throughout**:
   - Use SAME vocab file for dataset preprocessing
   - Use SAME vocab file for training
   - Use SAME vocab file for inference

3. **Retrain with correct configuration**:
   ```bash
   # Basic retraining command
   python headlessXttsTrain.py \
       --input_audio your_amharic_speaker.wav \
       --lang amh \
       --epochs 20 \
       --batch_size 4 \
       --grad_acumm 2
   
   # With G2P (recommended for better pronunciation)
   python headlessXttsTrain.py \
       --input_audio your_amharic_speaker.wav \
       --lang amh \
       --use_g2p \
       --g2p_backend rule_based \
       --epochs 20
   ```

4. **Verify vocab consistency**:
   ```python
   # Check all vocab files have same size
   import json
   
   files = [
       "path/to/dataset/vocab.json",
       "path/to/checkpoint/ready/vocab.json",
       "path/to/inference/vocab.json"
   ]
   
   for f in files:
       with open(f) as vf:
           size = len(json.load(vf)['model']['vocab'])
           print(f"{f}: {size} tokens")
   ```

---

### Solution B: G2P Mode Mismatch (COMMON)

**Symptom**: Vocab sizes match, but pronunciation is wrong

**Diagnosis**: One of the test modes (1-5) produces correct output

#### If Test 1 Works (amh, no G2P):
```python
# Your model was trained WITHOUT G2P
# Always use these settings:
lang = "amh"
use_g2p = False
```

Update default in `xtts_demo.py`:
```python
# Around line 432
use_g2p_inference = False  # Change default to False
```

#### If Test 4 Works (amh, with G2P):
```python
# Your model was trained WITH G2P
# Always use these settings:
lang = "amh"  
use_g2p = True
```

This is the recommended configuration!

#### If Test 5 Works (en, with G2P):
```python
# Model expects phonemes in 'en' phoneme mode
lang = "en"
use_g2p = True
```

Update inference to convert language:
```python
# In xtts_demo.py run_tts function
if use_g2p_inference:
    # Convert to phonemes
    processed_text = g2p.convert(tts_text)
    # Use 'en' as phoneme language
    inference_lang = "en"
else:
    inference_lang = lang
```

---

### Solution C: None of the Tests Work

**This means**: Vocabulary mismatch OR severely undertrained model

1. **Confirm vocab mismatch**:
   ```bash
   python diagnose_amharic_issue.py
   ```

2. **If mismatch confirmed**: See Solution A (retrain required)

3. **If sizes match but still broken**:
   - Model may be undertrained
   - Try training for 50-100 more epochs
   - Check training logs for loss convergence
   - Verify training data quality

---

## Preventing Future Issues

### 1. Use Consistent Vocabulary

Create a reference vocab and use it everywhere:

```bash
# Step 1: Create/identify your canonical vocab
cp vocab_extended_amharic.json vocab_reference.json

# Step 2: Use it for dataset creation
python preprocess_amharic_dataset.py \
    --vocab vocab_reference.json

# Step 3: Use it for training  
# (Ensure train_gpt.py uses this vocab)

# Step 4: Use it for inference
# (Load this vocab in xtts_demo.py)
```

### 2. Document Your Training Configuration

Create a `TRAINING_CONFIG.md` file:

```markdown
# Training Configuration for Checkpoint XYZ

- **Vocab File**: vocab_extended_amharic.json (7537 tokens)
- **G2P Enabled**: Yes
- **G2P Backend**: rule_based
- **Language Code**: amh
- **Epochs**: 50
- **Dataset**: 2 hours Amharic speech

## Inference Requirements
- Use lang='amh'
- Enable G2P
- Vocab must have exactly 7537 tokens
```

### 3. Add Vocab Validation to Training

Add to your training script:

```python
def validate_vocab_consistency():
    """Ensure dataset, checkpoint, and config use same vocab."""
    import json
    
    dataset_vocab = "ready/vocab.json"
    config_vocab = "config/vocab.json"
    
    with open(dataset_vocab) as f:
        dataset_size = len(json.load(f)['model']['vocab'])
    
    with open(config_vocab) as f:
        config_size = len(json.load(f)['model']['vocab'])
    
    assert dataset_size == config_size, \
        f"Vocab mismatch: dataset={dataset_size}, config={config_size}"
    
    print(f"✅ Vocab validation passed: {dataset_size} tokens")

# Call before training
validate_vocab_consistency()
```

---

## Quick Reference

### Symptoms → Solutions

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Complete gibberish | Vocab size mismatch | Retrain (Solution A) |
| Wrong but structured speech | G2P mode mismatch | Test modes (Solution B) |
| Silent or error | Language code issue | Use 'amh' consistently |
| Gradual improvement over epochs | Undertrained | Continue training |

### File Checklist

Before deploying a model, verify:

- [ ] Checkpoint vocab size matches current vocab file
- [ ] G2P settings documented
- [ ] Language code documented ('amh' recommended)
- [ ] Test inference produces correct output
- [ ] All vocab files in project are identical

---

## Emergency Quick Fix

If you need speech NOW and can't retrain:

1. Find the original vocab file from training:
   ```bash
   # Check all ready/ directories
   find . -name "vocab*.json" -exec sh -c 'echo "$1: $(python -c "import json; print(len(json.load(open(\"$1\"))[\"model\"][\"vocab\"]))")" tokens' _ {} \;
   ```

2. Use the vocab that matches checkpoint size (e.g., 7536 tokens)

3. Test inference modes to find working combination

4. **Then plan proper retraining with consistent vocab**

---

## Need Help?

If none of these solutions work:

1. Run both diagnostic scripts and save output
2. Note which test mode (if any) works partially
3. Check training logs for errors
4. Verify your training data is good quality Amharic speech
5. Consider starting fresh with a clean training run

Remember: **Vocabulary consistency is the #1 priority!**
