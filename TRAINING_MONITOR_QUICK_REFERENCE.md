# 🎯 Training Monitoring Quick Reference

**Keep this open while training to make quick decisions!**

---

## ✅ Decision Matrix

| Epoch | Train Loss | Eval Loss | Gap | Action |
|-------|-----------|-----------|-----|--------|
| 0 | 0.45 | 0.48 | 6.7% | ✅ **Continue** - Healthy start |
| 1 | 0.35 | 0.40 | 14% | ✅ **Continue** - Good progress |
| 2 | 0.28 | 0.36 | 29% | ✅ **Continue** - Acceptable gap |
| 3 | 0.24 | 0.34 | 42% | ⚠️ **Monitor** - Gap widening |
| 4 | 0.21 | 0.40 | 90% | 🛑 **STOP** - Eval increased! |

---

## 🚦 Traffic Light System

### ✅ GREEN - Continue Training
```
Train Loss: Decreasing ↓
Eval Loss:  Decreasing ↓
Gap:        <30%

Example:
  Epoch 2: train=0.32, eval=0.38 (19% gap) ✅
```

### ⚠️ YELLOW - Watch Carefully
```
Train Loss: Decreasing ↓
Eval Loss:  Flat or small increase →
Gap:        30-50%

Example:
  Epoch 3: train=0.25, eval=0.38 (52% gap) ⚠️
  → Watch next epoch closely
```

### 🛑 RED - STOP IMMEDIATELY
```
Train Loss: Decreasing ↓
Eval Loss:  Increasing ↑ (2 epochs in a row)
Gap:        >100%

Example:
  Epoch 3: train=0.24, eval=0.34 (OK)
  Epoch 4: train=0.20, eval=0.40 (BAD - increased!)
  Epoch 5: train=0.18, eval=0.50 (WORSE - STOP!) 🛑
  
  → Use checkpoint from Epoch 3
```

---

## 📊 Epoch-by-Epoch Checklist

### After Each Epoch:

1. **Find the EVALUATION section** in logs
   ```
   > EVALUATION
   > avg_loss: X.XXX
   ```

2. **Record the losses:**
   - Training loss (last step before eval)
   - Eval loss (from EVALUATION section)

3. **Calculate gap:**
   ```
   Gap % = ((eval_loss - train_loss) / train_loss) × 100
   ```

4. **Make decision:**
   - Gap <30% → ✅ Continue
   - Gap 30-50% → ⚠️ Watch next epoch
   - Gap >50% OR eval increasing 2x → 🛑 STOP

---

## 🎯 Quick Commands

### Monitor Training (Real-time)
```bash
# On Lightning AI
tail -f finetune_models/run/training/GPT_XTTS_FT-*/trainer_0_log.txt | grep -E "EPOCH|EVALUATION|avg_loss"
```

### Find Eval Losses
```bash
grep "EVALUATION" finetune_models/run/training/GPT_XTTS_FT-*/trainer_0_log.txt -A 5
```

### Stop Training
```
Ctrl+C in terminal
OR
Kill process in Lightning AI
```

---

## 📋 Training Log Template

**Fill this out after each epoch:**

```
=== TRAINING PROGRESS ===

Epoch 0:
  Train Loss: _____
  Eval Loss:  _____
  Gap:        _____% 
  Decision:   ✅ Continue / ⚠️ Watch / 🛑 Stop

Epoch 1:
  Train Loss: _____
  Eval Loss:  _____
  Gap:        _____% 
  Decision:   ✅ Continue / ⚠️ Watch / 🛑 Stop

Epoch 2: [LR reduces to 1e-06]
  Train Loss: _____
  Eval Loss:  _____
  Gap:        _____% 
  Decision:   ✅ Continue / ⚠️ Watch / 🛑 Stop

Epoch 3:
  Train Loss: _____
  Eval Loss:  _____
  Gap:        _____% 
  Decision:   ✅ Continue / ⚠️ Watch / 🛑 Stop

Epoch 4: [LR reduces to 5e-07]
  Train Loss: _____
  Eval Loss:  _____
  Gap:        _____% 
  Decision:   ✅ Continue / ⚠️ Watch / 🛑 Stop

Epoch 5:
  Train Loss: _____
  Eval Loss:  _____
  Gap:        _____% 
  Decision:   ✅ Continue / ⚠️ Watch / 🛑 Stop

BEST EPOCH: _____ (lowest eval_loss)
```

---

## 🎯 What to Look For

### Good Signs ✅
- Both losses decreasing steadily
- Gap stays below 30%
- LR reductions cause slight slowdown (expected)
- Eval loss keeps improving or stable

### Warning Signs ⚠️
- Gap increasing beyond 30%
- Eval loss stops improving
- Training loss approaches zero too fast
- Large fluctuations in eval loss

### Stop Signs 🛑
- Eval loss increases 2 epochs in a row
- Gap exceeds 100%
- Training loss near zero but eval loss high
- Audio quality gets worse

---

## 🚀 After Training Stops

### 1. Identify Best Checkpoint
```
Best epoch = epoch with LOWEST eval_loss
```

### 2. Use Checkpoint Manager
```
1. Open WebUI → Inference tab
2. Click "Scan Training Runs"
3. Find run "GPT_XTTS_FT-October-14-2025..."
4. Select checkpoint from best epoch
5. Click "Copy to Ready Folder"
6. Test audio generation
```

### 3. Test Quality
Generate sample audio with Amharic text and verify:
- ✅ Clear pronunciation
- ✅ Natural prosody
- ✅ No artifacts
- ✅ Correct Amharic phonemes

---

## 💡 Pro Tips

1. **Check eval loss IMMEDIATELY** after each epoch finishes
2. **Don't wait** - if you see 2 consecutive increases, stop NOW
3. **Keep this doc open** on second monitor/tab
4. **Set alarm** for end of each epoch (~10-15 min)
5. **Screenshot eval losses** for record-keeping

---

## 🔢 Example Calculations

### Scenario 1: Good Training
```
Epoch 2:
  train_loss = 0.30
  eval_loss = 0.36
  
Gap = ((0.36 - 0.30) / 0.30) × 100 = 20%
Decision: ✅ Continue (gap <30%)
```

### Scenario 2: Warning
```
Epoch 3:
  train_loss = 0.25
  eval_loss = 0.38
  
Gap = ((0.38 - 0.25) / 0.25) × 100 = 52%
Decision: ⚠️ Watch closely (gap >50%)
```

### Scenario 3: Stop
```
Epoch 3: eval_loss = 0.34 ✅
Epoch 4: eval_loss = 0.40 ⚠️ (increased by 0.06)
Epoch 5: eval_loss = 0.48 🛑 (increased again by 0.08)

Decision: 🛑 STOP NOW - use checkpoint from Epoch 3
```

---

## 📞 Need Help?

**If eval_loss is confusing:**
- Lower = Better
- Should decrease over epochs
- Closer to train_loss = Better generalization

**If gap is confusing:**
```
Gap = How much worse model performs on unseen data
<20% = Excellent
20-30% = Good
30-50% = Concerning
>50% = Overfitting
```

---

**Keep this reference handy during training! Good luck!** 🚀
