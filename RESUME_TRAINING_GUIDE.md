# Resume Training vs Fresh Training Guide

## 🔄 When to Resume Training

**Use Resume Training when:**
- ✅ You want to add more epochs to the same dataset
- ✅ Training was interrupted (power loss, crash, manual stop)
- ✅ Experimenting with different epoch counts
- ✅ **The dataset has NOT changed**

**How:**
1. Keep dataset unchanged
2. Enable "Resume from Checkpoint"
3. Select checkpoint from dropdown
4. Train continues seamlessly

---

## 🆕 When to Start Fresh Training (with Checkpoint as Base)

**Use Fresh Training when:**
- ✅ You added new audio files to dataset
- ✅ You modified existing transcripts
- ✅ You changed preprocessing settings (VAD, G2P, etc.)
- ✅ You want incremental learning with new data

**How:**
1. Process new/modified dataset normally
2. Load parameters from output folder
3. **Enter previous checkpoint in "Custom Model Path":**
   ```
   output/run/training/best_model.pth
   OR
   output/ready/model.pth (if you optimized it)
   ```
4. **Disable "Resume from Checkpoint"** (leave unchecked)
5. Start training - it will use the old model as base but with fresh optimizer

---

## Why This Matters

### Resume Training (Same Dataset)
```
Checkpoint → [Model + Optimizer State + Dataset State]
          ↓
   Continue exactly where you left off
```

### Fresh Training with Base Model (New Dataset)
```
Old Model → [Model Weights Only]
          ↓
   New Optimizer + New Dataset
          ↓
   Stable incremental learning
```

---

## Example Workflows

### Workflow 1: Add More Epochs (Same Dataset)
```bash
# Initial training
Epochs: 10
Result: output/run/training/best_model.pth

# Continue for 5 more epochs
☑️ Resume from Checkpoint
Checkpoint: run/training/best_model.pth
Epochs: 5 (additional)
Result: Continues from epoch 10 → 15
```

### Workflow 2: Add New Dataset (Incremental Learning)
```bash
# Initial training
Dataset: 500 audio files
Epochs: 10
Result: output/ready/model.pth (after optimization)

# Add 300 more audio files
1. Process new dataset (total now 800 files)
2. Custom Model Path: output/ready/model.pth
3. ☐ Resume from Checkpoint (unchecked)
4. Epochs: 10 (fresh training with all 800 files)
Result: New model trained on 800 files, starting from previous knowledge
```

### Workflow 3: Fix Dataset Issues
```bash
# Initial training
Dataset: Had some bad audio
Epochs: 10
Result: output/run/training/checkpoint_3000.pth (early checkpoint)

# Fixed dataset (removed bad audio, fixed transcripts)
1. Process corrected dataset
2. Custom Model Path: output/run/training/checkpoint_3000.pth
3. ☐ Resume from Checkpoint (unchecked)
4. Epochs: 10 (fresh training with clean dataset)
Result: Better quality model
```

---

## Technical Details

### What's in a Checkpoint?

```python
checkpoint = {
    'model': model_state_dict,           # ✅ Model weights
    'optimizer': optimizer_state_dict,    # ⚠️ Tied to dataset
    'epoch': current_epoch,               # ⚠️ Tied to dataset size
    'iteration': current_iteration,       # ⚠️ Tied to dataset size
    'scheduler': lr_scheduler_state,      # ⚠️ Tied to training schedule
}
```

### Resume Training (Uses Everything)
- Loads all checkpoint state
- Expects exact same dataset
- Continues learning rate schedule
- Optimizer momentum matches old gradients

### Fresh Training with Base Model (Model Only)
- Uses only model weights
- Fresh optimizer state
- Fresh learning rate schedule
- Adapts to new dataset naturally

---

## Best Practices

1. **Always backup checkpoints before modifying dataset**
   ```bash
   cp -r output/run/training output/run/training_backup
   ```

2. **Keep training logs organized**
   - Use different output folders for different dataset versions
   - Name folders descriptively: `output_v1_500samples`, `output_v2_800samples`

3. **Test new data incrementally**
   - Don't add 1000 files at once
   - Add 100-200 at a time and retrain
   - Monitor quality after each addition

4. **Document your datasets**
   ```
   dataset_v1/
   ├── metadata_train.csv  (500 samples)
   └── lang.txt
   
   dataset_v2/
   ├── metadata_train.csv  (800 samples, +300 new)
   └── lang.txt
   └── CHANGES.txt  (what was added/fixed)
   ```

---

## Quick Decision Tree

```
Do you want to continue training?
│
├─ Same dataset? → ✅ Use Resume Training
│  └─ Enable "Resume from Checkpoint"
│
└─ Different dataset? → ✅ Use Fresh Training with Base Model
   └─ Custom Model Path: previous checkpoint
   └─ Disable "Resume from Checkpoint"
```

---

## Summary

| Scenario | Resume Training | Custom Model Path | Result |
|----------|----------------|-------------------|---------|
| Add more epochs (same data) | ✅ Yes | Leave empty | Continue smoothly |
| Add new audio files | ❌ No | Previous model | Stable incremental learning |
| Fix dataset errors | ❌ No | Early checkpoint | Fresh start with fixed data |
| Interrupted training | ✅ Yes | Leave empty | Recover training |
| Experiment with epochs | ✅ Yes | Leave empty | Try different lengths |

**Golden Rule:** 
- **Resume = Same Dataset Only**
- **Dataset Changed = Fresh Training + Custom Model**
