# XTTS Training Knowledge Base

**Last Updated:** 2025-10-14 06:15 UTC

---

## 🚨 CRITICAL ALERT: Current Training Status

**DO NOT USE EPOCH 90+ CHECKPOINTS**

Your current training has **catastrophically overfit**. Use `best_model_569.pth` (Epoch 0) instead.

---

## 📚 Quick Navigation

### Current Incident (ACTIVE):
- 📄 **[Training Diagnosis](./training_diagnosis.md)** - Detailed analysis of overfitting issue
- 🔧 **[Training Fixes](./training_fixes.md)** - Complete solution implementation guide
- 📋 **[Incident Report](./incidents/2025-10-14-training-overfitting.md)** - Tracking document

### Key Findings:

**Problem:** 
- Training loss: 0.0005 ✅ (misleading - perfect memorization)
- Validation loss: 8.59 ❌ (cannot generalize)
- Gap: **17,000x** 🔴 CATASTROPHIC

**Root Causes:**
1. ❌ No early stopping
2. ❌ Insufficient dataset (too small for 520M model)
3. ❌ No regularization (dropout, weight decay)
4. ❌ Fixed learning rate (no scheduling)
5. ❌ Poor checkpoint selection

**Symptoms:**
- Poor Amharic pronunciation
- Unnatural word boundaries  
- Artificial breathing sounds
- Cannot handle exclamations
- Model memorized training data, can't generalize

---

## 🎯 Immediate Actions Required

### STOP and Use Best Checkpoint:
```bash
# On Lightning AI:
cd /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/finetune_models/run/training/GPT_XTTS_FT-October-13-2025_09+33PM-da15f5a/

# Use epoch 0 model (the actual best one)
cp best_model_569.pth best_model_FINAL.pth

# Test this checkpoint for inference
```

### Before Retraining - Implementation Checklist:

```python
# Priority 1: CRITICAL (Do Today)
[ ] Implement early stopping (patience=10 epochs)
[ ] Add LR scheduler (CosineAnnealing or ReduceLROnPlateau)
[ ] Enable gradient clipping (max_norm=1.0)
[ ] Stop current training immediately

# Priority 2: HIGH (This Week)
[ ] Expand dataset to 10+ hours
[ ] Filter dataset quality (>0.7 score)
[ ] Normalize Amharic text properly
[ ] Verify phoneme coverage

# Priority 3: MEDIUM (Next Week)  
[ ] Add regularization (dropout, weight decay)
[ ] Enable data augmentation
[ ] Set up monitoring dashboard
[ ] Create inference quality tests
```

---

## 📖 Document Guide

### `BACKGROUND_MUSIC_REMOVAL.md` ⭐ NEW!
**Purpose:** Remove background music from audio for cleaner TTS datasets  
**Contains:**
- Complete guide to using Demucs for audio source separation
- Integration with YouTube downloader and batch processor
- Performance benchmarks and best practices
- Troubleshooting guide

**Read this if:** You want to improve dataset quality by removing background music from downloaded YouTube videos or other audio sources

### `training_diagnosis.md`
**Purpose:** Understand what went wrong  
**Contains:**
- Loss trajectory analysis (epoch 0-90)
- Root cause breakdown
- Inference quality issues explained
- Critical metrics summary

**Read this if:** You want to understand the overfitting problem in depth

---

### `training_fixes.md`
**Purpose:** Fix the problems and retrain successfully  
**Contains:**
- Immediate fixes with code examples
- Dataset expansion strategies
- Amharic-specific preprocessing
- Complete training configuration
- Implementation timeline

**Read this if:** You're ready to implement fixes and retrain

**Quick Links in Document:**
- Part 1: Immediate Fixes (Early stopping, LR scheduling, gradient clipping)
- Part 2: Dataset Expansion (YouTube, Common Voice, augmentation)
- Part 3: Regularization (Dropout, weight decay, label smoothing)
- Part 4: Advanced Features (Phoneme mapping, prosody, multi-speaker)
- Part 5: Training Restart Checklist

---

### `incidents/2025-10-14-training-overfitting.md`
**Purpose:** Track incident for future reference  
**Contains:**
- Detailed timeline
- Impact assessment
- Lessons learned
- Prevention measures

**Read this if:** You want to prevent this from happening again

---

## 🔍 Key Metrics to Monitor

### Healthy Training:
```
Epoch 0:  train=3.5, eval=3.5, gap=~0
Epoch 10: train=1.5, eval=1.8, gap=0.3  ✅ Good!
Epoch 20: train=0.8, eval=1.2, gap=0.4  ✅ Acceptable
```

### Unhealthy Training (STOP IMMEDIATELY):
```
Epoch 1:  train=0.1, eval=4.7, gap=47x   🚩 Warning!
Epoch 5:  train=0.005, eval=5.7, gap=1140x  🔴 Critical!
Epoch 10: train=0.002, eval=6.4, gap=3200x  ⛔ STOP NOW!
```

**Red Flags:**
- 🚩 Val loss increases 3+ consecutive epochs → STOP
- 🚩 Train/val gap > 3x → Add regularization
- 🚩 Text loss < 0.001 → Overfitting text tokens
- 🚩 Mel loss diverging → Audio quality or preprocessing issue

---

## 💡 Best Practices Learned

### Always Do:
✅ Implement early stopping FIRST (before starting training)  
✅ Analyze dataset BEFORE training (size, quality, coverage)  
✅ Use learning rate scheduling (warmup + decay)  
✅ Monitor validation loss, not training loss  
✅ Test inference every 5-10 epochs  
✅ Stop at first sign of overfitting (don't wait)

### Never Do:
❌ Train without early stopping  
❌ Use constant learning rate for 90+ epochs  
❌ Ignore validation loss increasing  
❌ Trust training loss as quality indicator  
❌ Wait until training completes to test inference  
❌ Use latest checkpoint without validation

---

## 📞 Quick Reference Commands

### Analyze Training Logs:
```bash
# Check validation loss trend
grep "avg_loss:" trainer_0_log.txt | grep "EVAL PERFORMANCE"

# Find best epoch (lowest val loss)
grep "avg_loss:" trainer_0_log.txt | grep "EVAL PERFORMANCE" | sort -k5 -n | head -5
```

### Test Inference Quality:
```bash
# Create test file with Amharic sentences
echo "ሰላም\nእንደምን ነህ?\nዛሬ የአየር ሁኔታው ጥሩ ነው።" > test_sentences.txt

# Test with specific checkpoint
python scripts/test_inference.py \
    --checkpoint path/to/checkpoint.pth \
    --text_file test_sentences.txt \
    --output_dir inference_tests/
```

### Dataset Analysis:
```bash
# Count total samples
find dataset/ -name "*.wav" | wc -l

# Check total duration
python -c "import librosa; import glob; print(sum(librosa.get_duration(filename=f) for f in glob.glob('dataset/**/*.wav', recursive=True)) / 3600, 'hours')"

# Verify text quality
cat dataset/metadata.csv | cut -d'|' -f2 | head -20
```

---

## 🎓 Additional Resources

**XTTS Official:**
- Documentation: https://docs.coqui.ai/en/latest/models/xtts.html
- GitHub: https://github.com/coqui-ai/TTS
- Training Guide: https://github.com/coqui-ai/TTS/blob/dev/docs/source/training.md

**Amharic Resources:**
- Common Voice Dataset: https://commonvoice.mozilla.org/am/datasets
- Amharic IPA: https://en.wikipedia.org/wiki/Amharic#Phonology
- Unicode Range: U+1200 to U+137F

**Machine Learning:**
- Early Stopping: https://en.wikipedia.org/wiki/Early_stopping
- Learning Rate Schedules: https://www.deeplearning.ai/ai-notes/optimization/
- Regularization Techniques: https://developers.google.com/machine-learning/crash-course/regularization

---

## 📝 Change Log

**2025-10-14:**
- Initial knowledge base created
- Diagnosed catastrophic overfitting (Epoch 0-91)
- Created comprehensive fix guide
- Established incident tracking

---

## 🤝 Contributing to This Knowledge Base

When adding new information:

1. **Diagnoses:** Add to `training_diagnosis.md` or create new diagnosis file
2. **Solutions:** Add to `training_fixes.md` or create specific fix guides
3. **Incidents:** Create new file in `incidents/` folder
4. **Updates:** Always update this README.md with quick links

---

**Status:** Knowledge base established ✅  
**Next Update:** After implementing fixes and retraining

---

*This knowledge base tracks solutions for XTTS Amharic TTS training issues. All documentation is in `.warp/` directory.*
