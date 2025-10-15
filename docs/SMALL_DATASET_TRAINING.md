# Training XTTS v2 with Small Datasets (<3000 samples)

## Problem

When fine-tuning XTTS v2 (520M parameters) on small datasets (1-3 hours, 1000-3000 samples), the model tends to overfit:
- Training loss decreases
- Validation loss increases
- Model memorizes training data instead of learning general patterns

## Solution: Comprehensive Anti-Overfitting Strategy

### 1. **Layer Freezing** (Most Important)

Freeze most of the pre-trained model and only train what's necessary:

```python
from utils.xtts_small_dataset_config import XTTSSmallDatasetConfig

# Configuration
FREEZE_ENCODER = True  # Don't train audio encoder
FREEZE_FIRST_N_GPT_LAYERS = 28  # Only train last 2 of 30 GPT layers
TRAIN_TEXT_EMBEDDING = True  # Train embeddings (for new tokens)
TRAIN_TEXT_HEAD = True  # Train output layer

# Apply freezing
total_params, trainable_params = XTTSSmallDatasetConfig.apply_layer_freezing(model)
# Result: Train only ~5-10% of model parameters
```

**Why it works:** The pre-trained model already knows speech synthesis. We only need to adapt the last few layers to your specific voice/language.

### 2. **Very Low Learning Rate**

```python
LEARNING_RATE = 5e-7  # 10x lower than default
```

**Why:** Small datasets need gentle updates. High LR causes overfitting quickly.

### 3. **Minimal Epochs**

```python
MAX_EPOCHS = 2  # Only 2 epochs!
```

**Why:** With <2000 samples, the model sees everything quickly. More epochs = more memorization.

### 4. **Early Stopping**

```python
from utils.xtts_small_dataset_config import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(
    patience=1,  # Stop after 1 epoch without improvement
    min_delta=0.01
)

# In training loop
if early_stopping(val_loss, epoch):
    break  # Stop training
```

**Why:** Automatically stops when validation loss starts increasing.

### 5. **Small Batch Size + High Gradient Accumulation**

```python
BATCH_SIZE = 1  # Very small batches
GRAD_ACCUM_STEPS = 16  # Accumulate gradients
# Effective batch size = 16
```

**Why:** Small batches provide better gradients for limited data, but we accumulate to maintain stability.

### 6. **High Regularization**

```python
WEIGHT_DECAY = 0.1  # Strong L2 regularization
GRAD_CLIP_NORM = 0.5  # Prevent exploding gradients
```

**Why:** Prevents the model from fitting noise in the training data.

### 7. **Audio Augmentation** (Optional but Recommended)

```python
from utils.audio_augmentation import SimpleAudioAugmenter

augmenter = SimpleAugmenter(
    noise_prob=0.3,
    noise_level_range=(0.001, 0.01)
)

# During data loading
waveform = augmenter.augment(waveform)
```

**Why:** Adds variation to limited data, making it harder to memorize exact patterns.

## Complete Training Configuration

```python
from utils.xtts_small_dataset_config import XTTSSmallDatasetConfig

# Print configuration summary
XTTSSmallDatasetConfig.print_config_summary()

# Apply to training
config = GPTTrainerConfig(
    epochs=XTTSSmallDatasetConfig.MAX_EPOCHS,
    batch_size=XTTSSmallDatasetConfig.BATCH_SIZE,
    lr=XTTSSmallDatasetConfig.LEARNING_RATE,
    optimizer_params={
        "weight_decay": XTTSSmallDatasetConfig.WEIGHT_DECAY,
        ...
    },
    ...
)

# Apply layer freezing
model = GPTTrainer.init_from_config(config)
XTTSSmallDatasetConfig.apply_layer_freezing(model)

# Setup early stopping
early_stopping = EarlyStoppingCallback(
    patience=XTTSSmallDatasetConfig.EARLY_STOP_PATIENCE,
    min_delta=XTTSSmallDatasetConfig.EARLY_STOP_MIN_DELTA
)

# Training loop with early stopping
for epoch in range(config.epochs):
    train_loss = trainer.train_epoch()
    val_loss = trainer.eval_epoch()
    
    if early_stopping(val_loss, epoch):
        print("Early stopping triggered!")
        break
```

## Expected Results

With proper configuration for 1,845 samples:

### Bad Configuration (Current Issue):
```
Epoch 0: val_loss = 3.412 ✅
Epoch 1: val_loss = 3.427 ⚠️  (+0.015)
Epoch 2: val_loss = 3.607 🚨 (+0.180)
Result: Overfitting
```

### Good Configuration (With Anti-Overfitting):
```
Epoch 0: val_loss = 3.412 ✅
Epoch 1: val_loss = 3.350 ✅ (-0.062)
Epoch 2: val_loss = 3.380 ⚠️  (+0.030) → Early Stop
Result: Best model at epoch 1
```

## Key Metrics to Monitor

1. **Training vs Validation Loss:**
   - Both should decrease together
   - If train ↓ but val ↑ = overfitting

2. **Validation Loss Change:**
   - Should decrease by 5-15% per epoch
   - Any increase = warning sign

3. **Parameter Count:**
   - Should train <10% of total parameters
   - 520M total → ~50M trainable

## Troubleshooting

### Issue: Still Overfitting
**Solutions:**
- Reduce learning rate further (1e-7 or 5e-8)
- Freeze more layers (train only last 1 GPT layer)
- Use only 1 epoch
- Increase weight decay to 0.15

### Issue: Not Learning Anything
**Solutions:**
- Increase learning rate slightly (1e-6)
- Train more layers (last 3-4 GPT layers)
- Check if embeddings are trainable
- Verify dataset quality

### Issue: Training Too Slow
**Solutions:**
- Increase batch size to 2 (if you have GPU memory)
- Reduce gradient accumulation to 8
- Use mixed precision training
- Enable gradient checkpointing

## Best Practices for Small Datasets

1. **Dataset Quality > Quantity**
   - 1000 high-quality samples > 3000 poor quality
   - Clean audio, accurate transcriptions
   - Consistent recording conditions

2. **Start Conservative**
   - Begin with minimal config (2 epochs, very low LR)
   - Only increase if underfitting

3. **Save All Checkpoints**
   - Best model might be epoch 0 or 1
   - Don't rely on final checkpoint

4. **Monitor Closely**
   - Watch validation loss every epoch
   - Stop immediately if it increases

5. **Consider Data Augmentation**
   - Can effectively double/triple dataset size
   - Especially useful for <1500 samples

## References

- [Coqui TTS Fine-tuning Guide](https://github.com/coqui-ai/TTS)
- [XTTS-Finetune-WebUI Best Practices](https://github.com/daswer123/xtts-finetune-webui)
- Community discussions on small dataset training
- Transfer learning best practices

## Files

- `utils/xtts_small_dataset_config.py` - Configuration class
- `utils/audio_augmentation.py` - Audio augmentation
- `utils/gpt_train.py` - Training script (updated)
- `docs/SMALL_DATASET_TRAINING.md` - This guide
