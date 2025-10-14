# XTTS Amharic Training Diagnosis Report

**Date:** 2025-10-14  
**Training Run:** GPT_XTTS_FT-October-13-2025_09+33PM  
**Status:** CRITICAL - Severe Overfitting & Inference Quality Degradation

---

## Executive Summary

The Amharic XTTS-V2 training exhibits **catastrophic overfitting** by epoch 90, with training loss reaching ~0.0005 while evaluation loss increases to ~8.6 (a 17,000x divergence). Inference quality deteriorates significantly with poor Amharic pronunciation, unnatural prosody, incorrect word boundaries, and artificial breathing artifacts.

---

## 1. Loss Trajectory Analysis

### Training Loss Evolution
```
Epoch 0:  train=0.639, eval=3.415
Epoch 1:  train=0.100, eval=4.717  ⚠️ Eval loss increases
Epoch 5:  train=0.005, eval=5.678  ⚠️ Continues increasing
Epoch 10: train=0.002, eval=6.419
Epoch 20: train=0.001, eval=6.974
Epoch 30: train=0.001, eval=7.790  🔴 Critical divergence
Epoch 50: train=0.001, eval=8.108
Epoch 70: train=0.001, eval=8.350
Epoch 90: train=0.0005, eval=8.591 🔴 CATASTROPHIC
```

### Key Observations

1. **Immediate Overfitting**: Eval loss starts increasing from epoch 1, indicating the model memorizes training data instantly
2. **Text Loss Collapse**: `loss_text_ce` drops to ~0.000001 by epoch 90 (near-zero cross-entropy means perfect memorization)
3. **Mel Loss Explosion**: `loss_mel_ce` increases continuously, showing the model cannot generalize mel-spectrogram patterns
4. **No Recovery**: The trend is monotonic - no improvement in validation loss after epoch 0

---

## 2. Root Cause Analysis

### A. Insufficient Dataset Size/Diversity
**Evidence:**
- Training loss reaches near-zero very quickly (by epoch 5)
- Model has 520M parameters but dataset appears small
- High variance in eval loss indicates insufficient coverage

**Impact:**
- Model memorizes exact training samples
- Cannot generalize to slight variations in speech patterns
- Poor handling of Amharic phonemes not well-represented in training

### B. Lack of Regularization
**Evidence:**
```python
# Current config issues:
- No dropout mentioned in logs
- No weight decay visible
- No data augmentation
- Fixed learning rate (5e-06) throughout training
```

**Impact:**
- Model overfits to training noise
- No penalty for complex solutions
- Learns spurious correlations

### C. Learning Rate Schedule Issues
**Evidence:**
- Constant learning rate of 5e-06 for all 90+ epochs
- No warmup, no decay, no cosine annealing

**Impact:**
- Model continues to overfit training data without constraint
- No gradual refinement phase
- Cannot escape local minima on validation set

### D. Missing Early Stopping
**Evidence:**
- Training continues for 90+ epochs despite eval loss increasing from epoch 1
- No automatic checkpoint selection based on validation performance
- Best model saved at epoch 0, yet training continues

**Impact:**
- Massive compute waste
- Increasingly worse model quality
- User confusion about which checkpoint to use

### E. Dataset Quality Issues
**Evidence from inference problems:**
- Poor Amharic pronunciation → phoneme representation issues
- Unnatural word boundaries → inadequate text-audio alignment
- Artificial breathing → training data may contain artifacts
- Poor exclamation handling → insufficient prosody examples

---

## 3. Inference Quality Degradation Analysis

### A. Amharic Pronunciation Issues

**Problem:** Model produces incorrect Amharic sounds

**Root Causes:**
1. **Phoneme Tokenization**: XTTS uses English-centric phoneme set; Amharic ejectives (ጠ, ቀ, ጨ) may not map correctly
2. **Overfitting**: Model memorizes exact training pronunciations, cannot generalize to new Amharic words
3. **Limited Training Data**: Insufficient coverage of Amharic phoneme combinations

**Diagnostic Tests Needed:**
```python
# Check phoneme coverage
1. Extract all unique phonemes from training data
2. Compare with Amharic IPA phoneme inventory
3. Identify missing/poorly-mapped sounds
```

### B. Word Boundary & Timing Issues

**Problem:** Poor speech timing, awkward pauses between words

**Root Causes:**
1. **Audio Preprocessing**: Silence trimming may be too aggressive
2. **Text Alignment**: Duration predictor overfits to training examples
3. **Token Boundaries**: Amharic word boundaries may not align with model's tokenization

**Evidence:**
- Text loss near-zero means perfect token prediction
- Mel loss high means timing/duration completely wrong
- Model knows WHAT to say but not WHEN/HOW LONG

### C. Unnatural Breathing & Artifacts

**Problem:** Artificial breath sounds, unnatural pauses

**Root Causes:**
1. **Training Data Quality**: Original audio may contain artifacts
2. **Mel-Spectrogram Generation**: Model learns to reproduce training artifacts
3. **Overfitting**: Memorizes specific breath patterns from training samples

**Fix Priority:** HIGH (indicates fundamental data quality issue)

### D. Prosody & Exclamation Handling

**Problem:** Cannot handle exclamations, emotional tone

**Root Causes:**
1. **Limited Prosody Examples**: Training data may lack varied emotional expressions
2. **Overfitting**: Cannot generalize prosody patterns beyond training
3. **No Prosody Conditioning**: Model not explicitly trained on prosody features

---

## 4. Critical Metrics Summary

### Epoch 90 Statistics:
```
Training:
- loss_text_ce: 3.67e-06  (near perfect memorization)
- loss_mel_ce: 0.00424    (still learning, but incorrectly)
- total_loss:  0.00053    (misleading - only training performance)

Evaluation:
- loss_text_ce: 0.1395    (38,000x worse than training!)
- loss_mel_ce: 8.451      (1,994x worse than training!)
- total_loss:  8.591      (16,209x worse than training!)
```

**Interpretation:** The model has completely memorized the training set and has zero generalization ability.

---

## 5. Systemic Issues in Training Configuration

### Current Setup Problems:

1. ❌ **No validation-based early stopping**
2. ❌ **No learning rate scheduling** (constant 5e-06)
3. ❌ **No regularization** (dropout, weight decay)
4. ❌ **No data augmentation**
5. ❌ **Checkpoint selection ignores eval loss**
6. ❌ **No gradient clipping visible**
7. ❌ **100 epoch target despite clear overfitting at epoch 1**

### Environmental Factors:

- **Hardware:** Lightning AI GPU (adequate)
- **Mixed Precision:** Disabled (could enable for stability)
- **Batch Processing:** Working correctly
- **Data Loading:** No bottlenecks observed

---

## 6. Recommended Diagnostic Steps

### Immediate Actions:

1. **Stop Current Training** - It's making the model worse
2. **Use Best Checkpoint** - Likely epoch 0 or very early checkpoint
3. **Analyze Dataset:**
   ```bash
   # Check dataset size
   ls -lh dataset/
   
   # Inspect audio quality
   python scripts/analyze_dataset.py
   
   # Check phoneme coverage
   python scripts/check_phonemes.py
   ```

4. **Validate Preprocessing:**
   - Check audio sample rates
   - Verify silence trimming isn't too aggressive
   - Ensure text normalization preserves Amharic characters

### Data Quality Checks:

```python
# Required checks:
1. Total training samples: ??? (need to verify)
2. Audio quality: Check for artifacts, clipping, noise
3. Text quality: Verify Amharic Unicode correctness
4. Duration distribution: Look for outliers
5. Phoneme coverage: Map to Amharic IPA
```

---

## 7. Impact Assessment

### Severity: **CRITICAL**

- **Training Efficiency:** 0/10 (wasting compute after epoch 1)
- **Model Quality:** 2/10 (worse than initial checkpoint)
- **Inference Usability:** 3/10 (significant quality issues)
- **Data Pipeline:** 5/10 (functional but needs quality improvements)

### Business Impact:

- ❌ Cannot deploy current model
- ❌ Training time wasted (90+ epochs × ~30min = 45+ hours)
- ❌ GPU credits wasted on Lightning AI
- ✅ Infrastructure works (can retrain with fixes)
- ✅ Some good samples in early checkpoints

---

## 8. Next Steps

See `training_fixes.md` for detailed solution implementation plan.

**Priority Order:**
1. IMMEDIATE: Implement early stopping + LR scheduling
2. HIGH: Expand/clean dataset for Amharic
3. MEDIUM: Add regularization techniques
4. MEDIUM: Improve text preprocessing for Amharic
5. LOW: Add prosody-specific training enhancements

---

## Appendix: Training Configuration Snapshot

```
Model: XTTS-V2 (520M parameters)
Optimizer: Adam, lr=5e-06 (constant)
Batch Size: Unknown from logs
Epochs: 0-91 (ongoing)
GPU: 1x (Lightning AI)
Precision: float32
Mixed Precision: False
Gradient Accumulation: Unknown
```

**Last Updated:** 2025-10-14 06:03 UTC
