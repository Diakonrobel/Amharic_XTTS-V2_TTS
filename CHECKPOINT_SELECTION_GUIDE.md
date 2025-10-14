# 📦 Checkpoint Selection Feature - User Guide

## Overview

The **Checkpoint Selection** feature allows you to browse and select specific training checkpoints from your training runs, giving you precise control over which model version to use for inference. This is especially useful for:

- **Avoiding overfitting**: Select early checkpoints that generalize better
- **Testing different stages**: Compare model performance at different training epochs
- **Recovery**: Use a checkpoint if the final model is corrupted or overfitted

---

## 🎯 Quick Start

### Step 1: Complete a Training Run

Train your model using the Fine-tuning tab. The system automatically saves checkpoints during training (default: every 1000 steps).

### Step 2: Navigate to Inference Tab

Go to the **🎤 Inference** tab in the WebUI.

### Step 3: Scan for Checkpoints

1. Find the **🔄 Checkpoint Selection (Advanced)** section
2. Click **🔍 Scan Checkpoints**
3. Wait for the system to scan your latest training run

### Step 4: Review Available Checkpoints

The display will show:
- All available checkpoints
- Epoch and step numbers
- Evaluation loss (if available)
- Recommended checkpoint (marked with 💡)
- File sizes and save times

### Step 5: Select and Use

1. Choose a checkpoint from the dropdown
2. Click **✅ Use Selected Checkpoint**
3. The system will copy it to the ready folder
4. Click **▶️ Step 3 - Load Model** to load it
5. Generate speech as normal!

---

## 🔍 Feature Details

### Checkpoint Information Display

When you scan checkpoints, you'll see information like this:

```
📦 **Available Checkpoints from Latest Training Run**
======================================================================

💡 **RECOMMENDED**: Epoch 0 | Step 569 | (Loss: 3.415) [485.2 MB]
   Reason: Lowest eval loss

**All Checkpoints:**

  ➜ 1. Epoch 0 | Step 569 | (Loss: 3.415) [485.2 MB]
      Saved: 2025-10-14 01:30:45

   2. Epoch 5 | Step 3000 | (Loss: 5.678) [485.2 MB]
      Saved: 2025-10-14 02:15:30

   3. Epoch 10 | Step 6000 | (Loss: 6.419) [485.2 MB]
      Saved: 2025-10-14 03:00:15
```

### Recommendations

The system automatically recommends the **best checkpoint** based on:

1. **Lowest evaluation loss** (if eval data is available)
2. **Early checkpoints** (epochs 0-5) to avoid overfitting
3. **"Best model"** saved by trainer (fallback)

---

## 📊 Overfitting Analysis

Click **📊 Analyze Overfitting** to get a detailed analysis:

### What It Does

- Compares evaluation losses across checkpoints
- Detects if eval loss increased over time (overfitting indicator)
- Shows loss trend graph
- Recommends the safest checkpoint

### Example Analysis

```
📊 **Overfitting Analysis Report**
======================================================================

⚠️  OVERFITTING DETECTED!

Eval loss increased from 3.415 (epoch 0) to 8.591 (final).
Recommended safe checkpoint: Epoch 0 | Step 569 | (Loss: 3.415)

This checkpoint has the lowest evaluation loss.

**Evaluation Loss Trend:**

  Epoch  0 | Step   569 | Loss: 3.415
  Epoch  1 | Step  1138 | Loss: 4.717
  Epoch  2 | Step  1707 | Loss: 5.302
  Epoch  5 | Step  3000 | Loss: 5.678
  Epoch 10 | Step  6000 | Loss: 6.419
  Epoch 91 | Step 51879 | Loss: 8.591

**Recommended Safe Checkpoint:**
  Epoch 0 | Step 569 | (Loss: 3.415)
```

### Interpreting Results

| Status | Meaning | Action |
|--------|---------|--------|
| ✅ No significant overfitting | Eval loss stable or decreasing | Use latest checkpoint or recommended |
| ⚠️ Overfitting detected | Eval loss increased >50% | **Use early checkpoint!** (highlighted) |
| ℹ️ Not enough data | < 3 checkpoints with eval loss | Manual inspection recommended |

---

## 🛠️ How to Use This Feature

### Scenario 1: Your Training Overfitted (Most Common!)

**Problem**: You trained for 100 epochs and the final model sounds robotic or distorted.

**Solution**:
1. Click **🔍 Scan Checkpoints**
2. Click **📊 Analyze Overfitting**
3. Look for the **RECOMMENDED** checkpoint (usually epoch 0-10)
4. Select it from the dropdown
5. Click **✅ Use Selected Checkpoint**
6. Load and test!

### Scenario 2: Compare Different Training Stages

**Goal**: Test how your model improved over time.

**Steps**:
1. Scan checkpoints
2. Select an early checkpoint (e.g., Epoch 2)
3. Load and generate speech → save the output
4. Select a middle checkpoint (e.g., Epoch 10)
5. Load and generate speech → save the output
6. Compare audio quality!

### Scenario 3: Recovery After Accidental Overtraining

**Problem**: You accidentally trained for 200 epochs and the model is now unusable.

**Solution**:
1. The checkpoint system **automatically saved** checkpoints every 1000 steps
2. Scan checkpoints to see all available versions
3. Select the recommended early checkpoint
4. Your model is **recovered**! The early checkpoint is much better.

---

## ⚙️ Advanced Usage

### Checkpoint Naming Convention

The system recognizes these checkpoint patterns:

| Pattern | Example | Description |
|---------|---------|-------------|
| `checkpoint_STEP.pth` | `checkpoint_3000.pth` | Regular checkpoint at step 3000 |
| `best_model_STEP.pth` | `best_model_569.pth` | Best model saved at step 569 |
| `model_STEP.pth` | `model_5000.pth` | Alternative naming |

### Checkpoint Save Frequency

Default: **1000 steps**

To change:
1. Go to Fine-tuning tab
2. Adjust **"Checkpoint Save Frequency"** slider
3. Lower values = more checkpoints (more disk space)
4. Higher values = fewer checkpoints (less disk space)

**Recommended**: 1000 steps (good balance)

### Backup System

When you select a checkpoint:
- The system **automatically backs up** your current `model.pth`
- Backup is saved as `model_backup_YYYYMMDD_HHMMSS.pth`
- You can always restore the previous model manually

---

## 🎓 Best Practices

### 1. **Always Scan After Training**

After every training run:
1. Go to Inference tab
2. Click **🔍 Scan Checkpoints**
3. Click **📊 Analyze Overfitting**
4. Use the recommended checkpoint if overfitting is detected

### 2. **Don't Always Use the Final Checkpoint**

The **last checkpoint is NOT always the best**. Overfitting is common, especially with:
- Small datasets (< 10 minutes of audio)
- Too many epochs (> 20 for small datasets)
- Insufficient regularization

### 3. **Save Checkpoints Frequently**

For important training runs:
- Set checkpoint frequency to **500-1000 steps**
- This gives you more options if overfitting occurs

### 4. **Test Before Committing**

Before using a checkpoint:
1. Select it
2. Load it
3. Generate test speech
4. Listen and verify quality
5. If good → keep it!
6. If bad → try another checkpoint

### 5. **Keep Training Logs**

The checkpoint analysis depends on training logs. **Don't delete**:
- `finetune_models/run/training/GPT_XTTS_FT-*/trainer_0_log.txt`

This file contains evaluation losses used for recommendations.

---

## 🐛 Troubleshooting

### "No checkpoints found"

**Cause**: No training run completed or training folder deleted.

**Fix**:
- Complete at least one training run
- Checkpoints are in `finetune_models/run/training/GPT_XTTS_FT-*/`

### "Not enough data to determine overfitting"

**Cause**: Training log doesn't contain evaluation loss data, or < 3 checkpoints.

**Fix**:
- This is OK! You can still select checkpoints manually
- Choose early checkpoints (Epoch 0-5) to be safe

### Selected checkpoint doesn't sound better

**Cause**: Dataset issues, not training issues.

**Fix**:
- If all checkpoints sound bad, the problem is your dataset quality
- Check: audio quality, transcription accuracy, speaker consistency
- Re-train with cleaned dataset

### Checkpoint selector shows old training

**Cause**: The system shows the **latest** training run by modification time.

**Fix**:
- If you have multiple training runs, it shows the most recent
- To use an older run, manually navigate to its folder:
  - `finetune_models/run/training/GPT_XTTS_FT-DATE-TIME/`
  - Copy desired checkpoint to `finetune_models/ready/model.pth`

---

## 📋 Real-World Example

### Your Case: Amharic Training (91 Epochs, Overfitted)

**Situation**:
- Trained for 91 epochs (way too many!)
- Training loss: 0.000044 (memorized)
- Validation loss: 8.591 (terrible generalization)

**Using Checkpoint Selection**:

1. **Scan Checkpoints**:
   ```
   Found checkpoints:
   - best_model_569.pth (Epoch 0, Loss: 3.415) ← RECOMMENDED
   - checkpoint_3000.pth (Epoch 5, Loss: 5.678)
   - checkpoint_6000.pth (Epoch 10, Loss: 6.419)
   - ... many more ...
   - checkpoint_51879.pth (Epoch 91, Loss: 8.591) ← OVERFITTED!
   ```

2. **Analysis Shows**:
   ```
   ⚠️ OVERFITTING DETECTED!
   
   Eval loss increased from 3.415 → 8.591 (151% increase)
   Recommended: best_model_569.pth (Epoch 0)
   ```

3. **Action**:
   - Select `best_model_569.pth`
   - Click **✅ Use Selected Checkpoint**
   - Load and test
   - **Result**: Much better speech quality! ✅

4. **Next Steps**:
   - Retrain with **10-15 epochs max**
   - Enable early stopping
   - Use the recommended checkpoint this time

---

## 🎉 Summary

### Key Takeaways

✅ **Checkpoint selection gives you control** over which model version to use  
✅ **Early checkpoints often sound better** than the final model  
✅ **Overfitting analysis helps you avoid bad models** automatically  
✅ **The system recommends the best checkpoint** based on eval loss  
✅ **You can always recover from overtraining** using saved checkpoints  

### Workflow Recommendation

```
Train Model
    ↓
Scan Checkpoints
    ↓
Analyze Overfitting
    ↓
Select Recommended Checkpoint
    ↓
Load & Test
    ↓
Generate Speech!
```

---

## 📚 Related Documentation

- `TRAINING_DIAGNOSIS_AND_FIX.md` - Overfitting diagnosis and solutions
- `README.md` - Main WebUI documentation
- `TROUBLESHOOTING.md` - General troubleshooting guide

---

**Feature Added**: October 14, 2025  
**Version**: 1.0  
**Author**: Warp AI Agent Mode

For questions or issues, check the console logs when using the feature. The system prints detailed diagnostic information to help debug any problems.
