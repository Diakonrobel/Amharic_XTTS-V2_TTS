# Restart Training Guide - Fixed Learning Rate

## Problem Fixed
- ❌ **Old LR**: 1e-06 (too low for 37.7hr dataset)
- ✅ **New LR**: 1e-05 (10x higher, appropriate for your data size)
- ✅ **Better Scheduler**: ReduceLROnPlateau (adapts to validation loss)

## Option 1: Start Fresh Training (Recommended)

**Best if:**
- Current training hasn't improved much (which is the case)
- You want clean metrics from the start

```bash
# 1. Delete old checkpoints (optional, keeps previous attempts)
# rm -rf "Amharic_XTTS-V2_TTS/finetune_models/run/training/GPT_XTTS_FT-October-17-2025_06+40AM-14fa6ae"

# 2. Start fresh training
python headlessXttsTrain.py \
  --train_csv "path/to/train.csv" \
  --eval_csv "path/to/eval.csv" \
  --out_path "Amharic_XTTS-V2_TTS/finetune_models" \
  --num_epochs 20 \
  --batch_size 2 \
  --language "amharic"
```

## Option 2: Resume from Checkpoint (If you prefer)

**Resume from a previous checkpoint:**

```bash
# Resume from last checkpoint (epoch 2)
python headlessXttsTrain.py \
  --train_csv "path/to/train.csv" \
  --eval_csv "path/to/eval.csv" \
  --out_path "Amharic_XTTS-V2_TTS/finetune_models" \
  --num_epochs 20 \
  --batch_size 2 \
  --language "amharic" \
  --restore_path "Amharic_XTTS-V2_TTS/finetune_models/run/training/GPT_XTTS_FT-October-17-2025_06+40AM-14fa6ae/checkpoint_8000.pth"
```

**Note:** Resuming will use the NEW learning rate config we just fixed!

## What to Expect with New Settings

### Proper Training Progress:
```
Epoch 0: loss ~0.50 → ~0.35   (much faster improvement)
Epoch 1: loss ~0.35 → ~0.25
Epoch 2: loss ~0.25 → ~0.20
...
```

### Validation Loss Should:
- ✅ Decrease alongside training loss
- ✅ Stay close to training loss (gap <20%)
- ✅ Not increase (was happening before)

## Monitoring Training

Watch for these **good signs**:
1. Loss decreases steadily (not stuck)
2. Validation loss improves
3. LR reduces when validation plateaus (scheduler working)

Watch for **bad signs** (shouldn't happen now):
1. ❌ Loss barely moving → LR too low (shouldn't happen)
2. ❌ Validation loss increasing → Overfitting (scheduler will reduce LR)
3. ❌ Loss exploding → LR too high (unlikely with 1e-05)

## If Training Still Has Issues

### If loss decreases too slowly:
```python
# Increase LR further in utils/gpt_train.py line 368:
lr=2e-05,  # Try doubling
```

### If validation loss increases:
```python
# Training will auto-reduce LR with ReduceLROnPlateau
# No action needed - scheduler handles it
```

### If loss explodes (unlikely):
```python
# Reduce LR in utils/gpt_train.py line 368:
lr=5e-06,  # Reduce by half
```

## Expected Training Time

With 37.7 hours of data:
- **Epoch duration**: ~2-3 hours (depends on GPU)
- **Total training**: 40-60 hours for 20 epochs
- **Early stopping**: Likely stops at 10-15 epochs when optimal

## After Training Completes

1. Check the best model:
   - Look for `best_model_*.pth` in checkpoint directory
   - This has the lowest validation loss

2. Test inference:
   ```python
   from TTS.api import TTS
   tts = TTS(model_path="path/to/best_model.pth")
   tts.tts_to_file(text="ሰላም ዓለም", file_path="test.wav")
   ```

3. Compare with previous bad checkpoint

## Questions?

Check training logs in real-time:
```bash
tail -f trainer_0_log.txt
```

Monitor GPU usage:
```bash
nvidia-smi -l 1
```

---

**Good luck with the retrain! The new settings should work much better for your 37.7hr dataset!** 🚀
