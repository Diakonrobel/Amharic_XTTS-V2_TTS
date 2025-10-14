# XTTS Training Fixes - Implementation Plan

**Date:** 2025-10-14  
**Related:** `training_diagnosis.md`  
**Status:** Ready for Implementation

---

## Quick Action Summary

**STOP CURRENT TRAINING IMMEDIATELY** - Use best_model_569.pth (Epoch 0) for inference.

**Priority Fixes:**
1. 🔴 CRITICAL: Early stopping + LR scheduler (1-2 hours)
2. 🟠 HIGH: Dataset expansion & quality (4-8 hours)  
3. 🟡 MEDIUM: Regularization techniques (2-3 hours)
4. 🟢 LOW: Advanced prosody features (8+ hours)

---

## PART 1: IMMEDIATE FIXES (Implement Today)

### 1.1 Early Stopping Implementation

**File:** `TTS/trainer/trainer.py` or training config

**Add these parameters:**
```python
early_stopping_config = {
    'monitor': 'val_loss',           # Watch validation loss
    'patience': 10,                   # Stop after 10 epochs without improvement
    'min_delta': 0.001,              # Minimum change to qualify as improvement
    'mode': 'min',                   # We want to minimize loss
    'restore_best_weights': True,    # Revert to best checkpoint
    'verbose': True
}
```

**Expected Impact:** 
- Save 80+ hours of wasted compute
- Automatically select best model (likely epoch 0-5)
- Prevent quality degradation

---

### 1.2 Learning Rate Scheduling

**Replace constant LR with cosine annealing:**

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# In training config:
lr_config = {
    'initial_lr': 5e-06,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'T_0': 10,                      # Restart every 10 epochs
    'T_mult': 2,                    # Double restart period each time
    'eta_min': 1e-07,               # Minimum learning rate
    'warmup_epochs': 3              # Warmup for first 3 epochs
}

# Alternative: ReduceLROnPlateau (simpler)
lr_plateau = {
    'scheduler': 'ReduceLROnPlateau',
    'monitor': 'val_loss',
    'factor': 0.5,                  # Reduce by 50%
    'patience': 5,                  # After 5 epochs without improvement
    'min_lr': 1e-07
}
```

**Expected Impact:**
- Prevent aggressive overfitting in early epochs
- Allow fine-tuning in later epochs (if training continues)
- Better convergence on validation set

---

### 1.3 Gradient Clipping

**Add to training loop:**
```python
gradient_clip_config = {
    'max_norm': 1.0,                # Clip gradients to max norm of 1.0
    'norm_type': 2.0                # L2 norm
}

# In optimizer step:
torch.nn.utils.clip_grad_norm_(
    model.parameters(), 
    max_norm=gradient_clip_config['max_norm']
)
```

**Expected Impact:**
- Stabilize training
- Prevent gradient explosions
- Smoother loss curves

---

### 1.4 Checkpoint Selection Strategy

**Current Problem:** Saves based on training loss, not validation

**Fix:**
```python
checkpoint_config = {
    'save_best_after_epoch': 0,     # Start tracking from epoch 0
    'monitor': 'val_loss',          # Use validation loss
    'save_top_k': 3,                # Keep top 3 checkpoints
    'mode': 'min',
    'filename': 'best-epoch{epoch:02d}-valloss{val_loss:.4f}'
}
```

**Manual Fix (Immediate):**
```bash
# On Lightning AI, identify best checkpoint:
cd /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/finetune_models/run/training/GPT_XTTS_FT-October-13-2025_09+33PM-da15f5a/

# Based on diagnosis, best_model_569.pth (epoch 0) should be used
cp best_model_569.pth best_model_FINAL.pth

# Test inference with this checkpoint
```

---

## PART 2: HIGH PRIORITY FIXES (This Week)

### 2.1 Dataset Expansion Strategy

**Current Problem:** Insufficient data causing severe overfitting

**Action Plan:**

#### A. Quantify Current Dataset
```bash
# Create analysis script
python scripts/analyze_dataset.py --output .warp/dataset_stats.json

# Required metrics:
- Total samples
- Total duration (hours)
- Average sample length
- Speaker diversity
- Phoneme coverage
- Text variety (unique words)
```

**Target Goals:**
- Minimum 10 hours of clean Amharic speech
- 5,000+ unique sentences
- Coverage of all Amharic phonemes
- Multiple speakers (3-5 minimum)

#### B. Data Collection Methods

**Option 1: YouTube Batch Processing (Local)**
```powershell
# Use the local PowerShell script already provided
.\download_youtube_local.ps1

# Process locally to avoid IP blocking
# Upload processed dataset to Lightning incrementally
```

**Option 2: Common Voice Amharic**
```bash
# Download Common Voice Amharic dataset
# URL: https://commonvoice.mozilla.org/am/datasets

wget https://commonvoice.mozilla.org/datasets/cv-corpus-XX-am.tar.gz
tar -xzf cv-corpus-XX-am.tar.gz

# Filter for quality
python scripts/filter_common_voice.py \
    --input cv-corpus/am/ \
    --output dataset/common_voice_filtered/ \
    --min_duration 2 \
    --max_duration 15 \
    --min_quality 0.8
```

**Option 3: Synthetic Data Augmentation**
```python
# Use existing good samples with augmentation:
augmentation_config = {
    'pitch_shift': [-2, -1, 1, 2],    # Semitones
    'speed_change': [0.9, 1.1],        # Tempo
    'add_noise': 0.005,                # Light background noise
    'time_stretch': [0.95, 1.05]       # Time warping
}

# Generates 5x more samples from existing data
python scripts/augment_dataset.py \
    --input dataset/original/ \
    --output dataset/augmented/ \
    --config augmentation_config.json
```

---

### 2.2 Dataset Quality Improvement

**Create Quality Filter Script:**

```python
# scripts/filter_dataset_quality.py

import librosa
import numpy as np
from pathlib import Path

def check_audio_quality(audio_path):
    """
    Returns quality score 0-1
    """
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Check 1: Signal-to-noise ratio
    snr = calculate_snr(y)
    if snr < 20:  # dB
        return 0.0
    
    # Check 2: Clipping detection
    if np.max(np.abs(y)) > 0.99:
        return 0.3
    
    # Check 3: Silence ratio
    silence_ratio = detect_silence_ratio(y, sr)
    if silence_ratio > 0.3:  # More than 30% silence
        return 0.5
    
    # Check 4: Frequency content
    if not has_good_frequency_range(y, sr):
        return 0.6
    
    return 1.0

def filter_dataset(input_dir, output_dir, min_quality=0.7):
    """
    Filter dataset, keeping only high-quality samples
    """
    for audio_file in Path(input_dir).glob('*.wav'):
        quality = check_audio_quality(audio_file)
        
        if quality >= min_quality:
            # Copy to output with quality score in metadata
            copy_with_metadata(audio_file, output_dir, quality)
        else:
            print(f"Rejected {audio_file.name}: quality {quality:.2f}")

# Run filter
if __name__ == '__main__':
    filter_dataset('dataset/raw/', 'dataset/filtered/', min_quality=0.7)
```

**Usage:**
```bash
python scripts/filter_dataset_quality.py \
    --input dataset/raw/ \
    --output dataset/filtered/ \
    --min_quality 0.7 \
    --report .warp/quality_report.txt
```

---

### 2.3 Text Preprocessing for Amharic

**Create Amharic-specific normalizer:**

```python
# scripts/normalize_amharic_text.py

import re
import unicodedata

class AmharicTextNormalizer:
    def __init__(self):
        # Amharic Unicode ranges
        self.amharic_range = range(0x1200, 0x137F)
        
    def normalize(self, text):
        """
        Normalize Amharic text for TTS
        """
        # 1. Normalize Unicode (NFC form)
        text = unicodedata.normalize('NFC', text)
        
        # 2. Remove non-Amharic characters (except punctuation)
        text = self.keep_amharic_and_punctuation(text)
        
        # 3. Normalize punctuation
        text = self.normalize_punctuation(text)
        
        # 4. Handle numbers (convert to Amharic words)
        text = self.numbers_to_amharic_words(text)
        
        # 5. Expand abbreviations
        text = self.expand_abbreviations(text)
        
        return text.strip()
    
    def keep_amharic_and_punctuation(self, text):
        # Keep Amharic characters and basic punctuation
        allowed = set(chr(i) for i in self.amharic_range)
        allowed.update([' ', '.', ',', '!', '?', ':', ';'])
        
        return ''.join(c for c in text if c in allowed)
    
    def normalize_punctuation(self, text):
        # Replace various quote styles
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Normalize ellipsis
        text = re.sub(r'\.{2,}', '...', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def numbers_to_amharic_words(self, text):
        # Map digits to Amharic number words
        # Example: "123" -> "አንድ መቶ ሃያ ሶስት"
        # (Implementation depends on Amharic number system)
        return text  # Placeholder
    
    def expand_abbreviations(self, text):
        # Common Amharic abbreviations
        abbrev_map = {
            'ዓ.ም': 'ዓመተ ምሕረት',
            'ወዘተ': 'ወዘተረፈ',
            # Add more
        }
        for abbrev, expansion in abbrev_map.items():
            text = text.replace(abbrev, expansion)
        return text

# Apply to entire dataset
def preprocess_dataset_text(dataset_dir):
    normalizer = AmharicTextNormalizer()
    
    for metadata_file in Path(dataset_dir).glob('**/metadata.csv'):
        # Read, normalize, write back
        pass  # Implementation

if __name__ == '__main__':
    preprocess_dataset_text('dataset/')
```

---

## PART 3: MEDIUM PRIORITY FIXES (Next Week)

### 3.1 Regularization Techniques

**Add to model config:**

```python
regularization_config = {
    # Dropout
    'dropout': 0.1,                   # 10% dropout in feed-forward layers
    'attention_dropout': 0.1,         # Dropout in attention
    
    # Weight decay
    'weight_decay': 0.01,             # L2 regularization
    
    # Label smoothing
    'label_smoothing': 0.1,           # Smooth hard targets
    
    # Stochastic depth (for transformer layers)
    'stochastic_depth': 0.1,          # Random layer dropping
}
```

**Implementation points:**
- Modify model architecture to include dropout
- Add weight decay to optimizer
- Use label smoothing in loss calculation

---

### 3.2 Data Augmentation Pipeline

**Create augmentation config:**

```python
# config/augmentation.yaml

audio_augmentation:
  - type: pitch_shift
    probability: 0.3
    semitones: [-2, -1, 1, 2]
  
  - type: time_stretch
    probability: 0.3
    factors: [0.9, 0.95, 1.05, 1.1]
  
  - type: add_noise
    probability: 0.2
    snr_db: [20, 30, 40]
  
  - type: spec_augment           # SpecAugment for mel-spectrograms
    probability: 0.4
    freq_mask_param: 15
    time_mask_param: 35
    num_masks: 2

text_augmentation:
  - type: synonym_replacement     # Replace words with Amharic synonyms
    probability: 0.1
    num_replacements: 1
  
  - type: punctuation_variation   # Vary punctuation
    probability: 0.2
```

---

### 3.3 Mixed Precision Training

**Enable for better stability:**

```python
training_config = {
    'mixed_precision': True,
    'precision': 'fp16',              # or 'bf16' if available
    'gradient_scaling': True,
    'loss_scale': 'dynamic'
}
```

**Benefits:**
- Faster training
- Lower memory usage
- Can train with larger batch sizes
- Better numerical stability (in some cases)

---

## PART 4: ADVANCED FIXES (Long-term)

### 4.1 Phoneme Mapping for Amharic

**Create custom phoneme mapper:**

```python
# scripts/amharic_phoneme_mapper.py

class AmharicPhonemeMapper:
    """
    Map Amharic characters to IPA phonemes compatible with XTTS
    """
    def __init__(self):
        self.phoneme_map = {
            # Amharic consonants
            'ሀ': 'h', 'ለ': 'l', 'ሐ': 'h', 'መ': 'm',
            'ሠ': 's', 'ረ': 'r', 'ሰ': 's', 'ሸ': 'ʃ',
            'ቀ': 'kʼ',  # Ejective k
            'በ': 'b', 'ተ': 't', 'ቸ': 'tʃ',
            'ኀ': 'h', 'ነ': 'n', 'ኘ': 'ɲ',
            'አ': 'ʔ',  # Glottal stop
            'ከ': 'k', 'ኸ': 'k', 'ወ': 'w',
            'ዐ': 'ʕ',  # Pharyngeal
            'ዘ': 'z', 'ዠ': 'ʒ', 'የ': 'j',
            'ደ': 'd', 'ጀ': 'dʒ', 'ገ': 'g',
            'ጠ': 'tʼ',  # Ejective t
            'ጨ': 'tʃʼ', # Ejective ch
            'ጰ': 'pʼ',  # Ejective p
            'ጸ': 'sʼ',  # Ejective s
            'ፀ': 'sʼ',
            'ፈ': 'f', 'ፐ': 'p',
            
            # Add vowel modifications
            # e.g., 'ለ' + vowel markers
        }
    
    def text_to_phonemes(self, amharic_text):
        """
        Convert Amharic text to IPA phonemes
        """
        phonemes = []
        for char in amharic_text:
            if char in self.phoneme_map:
                phonemes.append(self.phoneme_map[char])
            else:
                # Handle unknown characters
                phonemes.append(char)
        return ' '.join(phonemes)
```

**Integration:**
- Modify XTTS tokenizer to use custom phoneme mapper
- Ensure all Amharic sounds are represented
- Test coverage on dataset

---

### 4.2 Prosody Enhancement

**Add prosody features to model:**

```python
prosody_config = {
    'enable_prosody_predictor': True,
    'prosody_features': [
        'pitch_contour',
        'energy',
        'duration',
        'speaking_rate'
    ],
    'prosody_loss_weight': 0.1,
    'use_reference_encoder': True    # For style transfer
}
```

**Implementation:**
1. Extract prosody features during preprocessing
2. Add prosody predictor module to model
3. Include prosody loss in training
4. Allow prosody conditioning at inference

---

### 4.3 Multi-Speaker Training

**If collecting multiple speakers:**

```python
multi_speaker_config = {
    'num_speakers': 5,
    'speaker_embedding_dim': 256,
    'speaker_conditioning': 'concat',  # or 'add', 'film'
    'speaker_encoder_type': 'x-vector'
}
```

**Benefits:**
- Better generalization
- Style flexibility
- More robust phoneme learning

---

## PART 5: TRAINING RESTART CHECKLIST

### Before Retraining:

- [ ] Implement early stopping
- [ ] Add learning rate scheduler
- [ ] Enable gradient clipping
- [ ] Expand dataset to 10+ hours
- [ ] Filter dataset for quality (>0.7 score)
- [ ] Normalize all Amharic text
- [ ] Verify phoneme coverage
- [ ] Add regularization (dropout, weight decay)
- [ ] Enable mixed precision training
- [ ] Configure proper checkpoint selection (val_loss based)

### New Training Configuration:

```yaml
# config/training_config_v2.yaml

model:
  name: XTTS_v2
  parameters: 520M

training:
  max_epochs: 100
  batch_size: 8
  gradient_accumulation: 4
  mixed_precision: true
  
  optimizer:
    type: Adam
    lr: 5e-06
    weight_decay: 0.01
    
  scheduler:
    type: CosineAnnealingWarmRestarts
    T_0: 10
    T_mult: 2
    eta_min: 1e-07
    warmup_epochs: 3
  
  early_stopping:
    monitor: val_loss
    patience: 10
    min_delta: 0.001
    mode: min
  
  gradient_clipping:
    max_norm: 1.0
  
  checkpoint:
    monitor: val_loss
    save_top_k: 3
    mode: min

regularization:
  dropout: 0.1
  attention_dropout: 0.1
  label_smoothing: 0.1

data_augmentation:
  audio:
    pitch_shift: true
    time_stretch: true
    spec_augment: true
  text:
    enabled: false  # Enable when ready

dataset:
  min_duration: 2.0
  max_duration: 15.0
  sample_rate: 22050
  text_normalizer: amharic_custom
```

### Expected Results After Fixes:

**Epoch 0:**
- train_loss: ~3.5
- val_loss: ~3.5
- Gap: ~0

**Epoch 10 (with fixes):**
- train_loss: ~1.5
- val_loss: ~1.8
- Gap: ~0.3 (acceptable)

**Epoch 20 (should stop here):**
- train_loss: ~0.8
- val_loss: ~1.2
- Gap: ~0.4 (reasonable)

**Early stopping triggers at epoch ~15-25** (validation loss plateaus)

---

## PART 6: Monitoring & Validation

### Key Metrics to Track:

```python
metrics_to_watch = {
    'train_loss': 'Should decrease steadily',
    'val_loss': 'Should decrease then plateau',
    'loss_gap': 'Should stay < 2x',
    'text_ce_loss': 'Should decrease but not to near-zero',
    'mel_ce_loss': 'Should decrease on both train and val',
    'grad_norm': 'Should be stable (< 10)',
    'learning_rate': 'Should follow schedule'
}
```

### Inference Quality Tests:

**Create test set:**
```python
# test_sentences.txt (Amharic)
ሰላም
ሰላም ነው
እንደምን ነህ?
ምንም ችግር የለም።
ዛሬ የአየር ሁኔታው ጥሩ ነው።
(Add 50+ diverse sentences)
```

**Test every 5 epochs:**
```bash
python scripts/test_inference_quality.py \
    --checkpoint checkpoints/epoch_X/ \
    --test_file test_sentences.txt \
    --output_dir test_outputs/epoch_X/ \
    --metrics .warp/inference_quality.csv
```

### Red Flags to Watch:

🚩 Val loss increases for 3+ consecutive epochs → Stop training  
🚩 Train/val gap > 3x → Reduce model capacity or add regularization  
🚩 Text loss near-zero (< 0.001) → Overfitting text, check dataset  
🚩 Mel loss not decreasing → Audio quality or preprocessing issue  
🚩 Gradient norm > 20 → Reduce learning rate or check for bad samples

---

## PART 7: Implementation Timeline

### Day 1 (Today):
- ✅ Create diagnostic document
- ✅ Create fixes document
- ⏳ Implement early stopping
- ⏳ Add LR scheduler
- ⏳ Enable gradient clipping
- ⏳ Stop current training, select best checkpoint

### Week 1:
- Expand dataset to 10+ hours
- Implement quality filtering
- Apply Amharic text normalization
- Set up augmentation pipeline
- Configure regularization

### Week 2:
- Restart training with all fixes
- Monitor closely for first 20 epochs
- Validate inference quality
- Adjust hyperparameters if needed

### Week 3:
- Fine-tune best checkpoint on high-quality data
- Test thoroughly on diverse Amharic sentences
- Document final results
- Deploy if quality acceptable

---

## Success Criteria

Training is considered successful when:

✅ Validation loss stays within 2x of training loss  
✅ Validation loss decreases for at least 10 epochs  
✅ Early stopping triggers (not just hitting max epochs)  
✅ Inference quality is consistently good:
  - Clear Amharic pronunciation
  - Natural word boundaries
  - Appropriate prosody
  - No artificial artifacts
✅ Model generalizes to unseen Amharic text

---

## Resources & References

- **XTTS Documentation:** https://docs.coqui.ai/en/latest/models/xtts.html
- **TTS Training Guide:** https://github.com/coqui-ai/TTS
- **Amharic Phonology:** IPA chart for Amharic
- **Common Voice Amharic:** https://commonvoice.mozilla.org/am
- **Lightning AI Docs:** https://lightning.ai/docs

---

**Last Updated:** 2025-10-14 06:10 UTC  
**Next Review:** After implementing immediate fixes
