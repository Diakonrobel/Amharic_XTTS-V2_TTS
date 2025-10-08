# Checkpoint Saving Guide

## Overview
This guide explains how model checkpoints are saved during XTTS fine-tuning and how to configure the checkpoint saving behavior through the WebUI.

---

## 📍 Default Checkpoint Settings

### Current Configuration (Before Update)
- **Save Frequency**: Every **1000 steps**
- **Number Retained**: **1 checkpoint** (only the latest)
- **Save Location**: `<output_path>/run/training/`

### After Update (Commit `9131199`)
You can now **adjust these settings directly in the WebUI**!

---

## 🎛️ WebUI Controls (NEW)

In the **Fine-tuning tab** under **Training Parameters**, you'll now see:

### 1. Checkpoint Save Frequency (steps)
- **Range**: 100 - 5000 steps
- **Default**: 1000 steps
- **Description**: How often to save a checkpoint during training
- **Recommendation**:
  - **Fast experimentation**: 500-1000 steps
  - **Long training runs**: 1000-2000 steps
  - **Production/critical training**: 500 steps (more frequent)

### 2. Keep N Checkpoints
- **Range**: 1 - 10 checkpoints
- **Default**: 1 checkpoint
- **Description**: How many recent checkpoints to retain
- **Recommendation**:
  - **Limited disk space**: 1-2 checkpoints
  - **Experimentation**: 3-5 checkpoints
  - **Production**: 5-10 checkpoints (compare multiple versions)

---

## 💾 Checkpoint Files and Locations

### Training Checkpoints
During training, checkpoints are saved to:
```
<output_path>/run/training/
├── checkpoint_1000.pth
├── checkpoint_2000.pth
├── checkpoint_3000.pth
└── ...
```

**Important**: Only the last N checkpoints are kept (based on "Keep N Checkpoints" setting)

### Best Model
The best performing model (based on validation loss) is automatically saved as:
```
<output_path>/run/training/best_model.pth
```

This file is **always retained** regardless of checkpoint settings.

### Final Model Location
After training completes, the best model is copied to:
```
<output_path>/ready/
├── unoptimize_model.pth  ← Copy of best_model.pth
├── config.json
├── vocab.json
├── speakers_xtts.pth
└── reference.wav
```

### Optimized Model
After running **"Step 2.5 - Optimize Model"**, the final optimized model is saved:
```
<output_path>/ready/
└── model.pth  ← Optimized version (smaller, no optimizer state)
```

---

## 🔄 Checkpoint Contents

Each checkpoint file contains:
- **Model weights**: Full GPT model parameters
- **Optimizer state**: Adam optimizer state (for resuming training)
- **Training metadata**: Step number, epoch, loss history
- **DVAE weights**: Discrete VAE audio encoder weights

**Note**: The optimized model (`model.pth`) has optimizer state and DVAE removed to reduce file size.

---

## ⚙️ Advanced Configuration

### Manual Configuration
If you need more control, you can directly edit `utils/gpt_train.py`:

```python
config = GPTTrainerConfig(
    # ... other parameters ...
    print_step=50,          # Console log frequency
    plot_step=100,          # TensorBoard plot frequency  
    log_model_step=100,     # Model logging frequency
    save_step=1000,         # ← Checkpoint save frequency
    save_n_checkpoints=1,   # ← Number of checkpoints to keep
    save_checkpoints=True,  # ← Enable/disable checkpoint saving
)
```

### Related Settings
- **`print_step=50`**: How often to print training progress to console
- **`plot_step=100`**: How often to log metrics to TensorBoard
- **`log_model_step=100`**: How often to log model architecture

---

## 💡 Best Practices

### 1. **Disk Space Management**
- Each checkpoint is ~500MB-2GB depending on model size
- Calculate required space: `checkpoint_size × keep_N_checkpoints`
- Use fewer checkpoints if disk space is limited

### 2. **Training Duration**
- For short training (< 1 epoch): Save every 200-500 steps
- For medium training (1-3 epochs): Save every 500-1000 steps  
- For long training (> 3 epochs): Save every 1000-2000 steps

### 3. **Experimentation**
- Keep 3-5 checkpoints to compare different training stages
- Resume from earlier checkpoints if overfitting occurs

### 4. **Production Training**
- Save more frequently (500 steps)
- Keep more checkpoints (5-10)
- Monitor TensorBoard to identify best checkpoint

---

## 🚨 Important Notes

1. **Best Model is Always Saved**: Regardless of checkpoint settings, the best model (lowest validation loss) is always saved as `best_model.pth`

2. **Disk Space**: Monitor disk usage during long training runs, especially with frequent saves and multiple retained checkpoints

3. **Resume Training**: If training is interrupted, it can resume from the last checkpoint (if available)

4. **Extended Vocabulary**: When training with Amharic G2P enabled, checkpoints include the extended vocabulary embeddings

5. **Checkpoint Loading**: The system handles vocabulary size mismatches automatically when loading checkpoints with extended vocab

---

## 🔍 Monitoring Checkpoints

### Console Output
During training, you'll see:
```
 > Checkpoint saved at step 1000
 > Checkpoint saved at step 2000
 > Best model updated at step 1500 (loss: 0.234)
```

### TensorBoard
View training metrics in real-time:
```bash
tensorboard --logdir <output_path>/run/training
```

---

## 📊 Example Configurations

### Fast Experimentation (Limited Resources)
- **Save Frequency**: 1000 steps
- **Keep Checkpoints**: 1
- **Disk Usage**: ~1-2GB

### Balanced (Recommended)
- **Save Frequency**: 500 steps  
- **Keep Checkpoints**: 3
- **Disk Usage**: ~3-6GB

### Production (Maximum Safety)
- **Save Frequency**: 500 steps
- **Keep Checkpoints**: 10  
- **Disk Usage**: ~5-20GB

### Very Long Training
- **Save Frequency**: 2000 steps
- **Keep Checkpoints**: 5
- **Disk Usage**: ~5-10GB

---

## 🛠️ Troubleshooting

### Issue: "Out of disk space during training"
**Solution**: Reduce `save_n_checkpoints` to 1 or increase `save_step` to 2000

### Issue: "Training crashed and lost progress"
**Solution**: Use more frequent saves (save_step=500) and keep multiple checkpoints (save_n_checkpoints=3)

### Issue: "Want to resume from earlier checkpoint"
**Solution**: 
1. Keep more checkpoints (save_n_checkpoints=5-10)
2. Manually copy desired checkpoint to use as custom model
3. Use it in the "Custom Model Path" field

### Issue: "Checkpoint loading fails with size mismatch"
**Solution**: This is handled automatically for extended vocabulary. If issues persist, ensure you're using the correct vocab.json file.

---

## 📝 Summary

| Setting | Default | Recommended Range | Purpose |
|---------|---------|-------------------|---------|
| Save Frequency | 1000 steps | 500-2000 steps | How often to save |
| Keep N Checkpoints | 1 | 1-5 (3 ideal) | How many to retain |
| Print Step | 50 steps | - | Console logging |
| Plot Step | 100 steps | - | TensorBoard |

---

**Updated**: 2025-10-08  
**Commit**: `9131199` - Added configurable checkpoint controls to WebUI
