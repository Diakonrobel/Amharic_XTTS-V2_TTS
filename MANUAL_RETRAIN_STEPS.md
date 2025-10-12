# Manual Step-by-Step Retraining Commands

## 🎯 What You Need Before Starting

1. **Amharic audio files** (`.wav` format, 22050 Hz recommended)
2. **Transcriptions** for each audio file
3. **At least 30 minutes** of audio (1-2 hours is better)
4. **GPU access** on your server

---

## 📋 Step 1: Check Your Current Setup

```bash
cd ~/Amharic_XTTS-V2_TTS

# Pull latest code
git pull origin main

# Check what you have
ls -lh finetune_models/dataset/wavs/ | head -10
ls -lh finetune_models/dataset/metadata.csv
```

**Expected:**
- Audio files in `finetune_models/dataset/wavs/`
- `metadata.csv` with format: `filename.wav|Amharic text transcription`

**If you don't have these**, see the "Prepare Dataset" section at the end.

---

## 📚 Step 2: Create Clean Reference Vocabulary

```bash
cd ~/Amharic_XTTS-V2_TTS

# Use the 7537-token backup (or current vocab)
cp finetune_models/ready/vocab_extended_amharic.json vocab_reference_clean.json

# Verify size
python3 << EOF
import json
vocab = json.load(open('vocab_reference_clean.json'))
size = len(vocab['model']['vocab'])
print(f"Vocab size: {size} tokens")
if size == 7537:
    print("✅ Perfect size!")
elif size == 7536:
    print("✅ Compatible size")
else:
    print(f"⚠️  Unusual size: {size}")
EOF
```

---

## 🎤 Step 3: Choose Speaker Reference Audio

Pick ONE audio file that represents your speaker's voice well:

```bash
# List your audio files
ls finetune_models/dataset/wavs/*.wav | head -20

# Choose one (example):
SPEAKER_REF="finetune_models/dataset/wavs/audio_001.wav"

# Verify it exists and check duration
ffprobe -i "$SPEAKER_REF" -show_entries format=duration -v quiet -of csv="p=0"
# Should be between 3-30 seconds
```

---

## 🏗️ Step 4: Create Training Dataset (Method 1 - Using WebUI)

**This is the easiest method:**

```bash
# Start the web interface
./launch.sh
```

Then in the web browser (Gradio interface):

1. **Go to Tab 1** (Dataset Processing)
2. **Upload or select** your audio files folder
3. **Upload metadata.csv** with your transcriptions
4. **Language**: Select "amh" (or type "amh")
5. **Enable "Use G2P Amharic phonemes"** ✅
6. **Click "Step 1 - Create Dataset"**
7. Wait for processing (10-30 minutes)

**Output** will be in `run/dataset/XTTS_FT_[timestamp]/`

---

## 🏗️ Step 4: Create Training Dataset (Method 2 - Command Line)

If WebUI doesn't work, use command line:

```bash
cd ~/Amharic_XTTS-V2_TTS

# Create output directory
mkdir -p finetune_models/ready_clean

# Run dataset creation
python3 << 'PYTHON_SCRIPT'
import os
import sys
sys.path.insert(0, os.getcwd())

from utils.amharic_g2p_dataset_wrapper import create_amharic_dataset
from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P

# Initialize G2P
g2p = AmharicG2P(backend='rule_based')

# Process dataset
create_amharic_dataset(
    audio_dir='finetune_models/dataset/wavs',
    metadata_file='finetune_models/dataset/metadata.csv',
    output_dir='finetune_models/ready_clean',
    lang='amh',
    g2p_converter=g2p,
    vocab_file='vocab_reference_clean.json',
    eval_split=0.1
)

print("\n✅ Dataset created successfully!")
print("📁 Location: finetune_models/ready_clean")
PYTHON_SCRIPT
```

---

## 🚀 Step 5: Start Training

```bash
cd ~/Amharic_XTTS-V2_TTS

# Set variables
SPEAKER_REF="finetune_models/dataset/wavs/audio_001.wav"  # Change this!
READY_DIR="finetune_models/ready_clean"
EPOCHS=30

# Optional: Use screen/tmux for long training
screen -S amharic_train

# Start training
python headlessXttsTrain.py \
    --input_audio "$SPEAKER_REF" \
    --lang amh \
    --train_csv "$READY_DIR/metadata_train.csv" \
    --eval_csv "$READY_DIR/metadata_eval.csv" \
    --out_path "$READY_DIR" \
    --epochs $EPOCHS \
    --batch_size 4 \
    --grad_acumm 4 \
    --save_step 5000 \
    --use_g2p \
    --g2p_backend rule_based \
    --custom_vocab "$READY_DIR/vocab.json"
```

**Training will take 4-8 hours depending on:**
- GPU type (A100 fastest, T4 slowest)
- Dataset size
- Batch size settings

**To detach from screen**: Press `Ctrl+A` then `D`  
**To reattach**: `screen -r amharic_train`

---

## 📊 Step 6: Monitor Training

```bash
# Watch training logs
tail -f finetune_models/ready_clean/trainer_0_log.txt

# Check for checkpoints
ls -lh finetune_models/ready_clean/*.pth
```

**What to look for:**
- Loss should decrease over epochs
- Check samples generated every few thousand steps
- By epoch 15-20, you should hear improvement

---

## ✅ Step 7: Test the New Model

After training completes:

```bash
cd ~/Amharic_XTTS-V2_TTS

# Find the best checkpoint
BEST_MODEL="finetune_models/ready_clean/best_model.pth"
NEW_VOCAB="finetune_models/ready_clean/vocab.json"

# Verify files exist
ls -lh "$BEST_MODEL"
ls -lh "$NEW_VOCAB"

# Test vocab size matches
python3 << EOF
import json
import torch

# Check vocab
vocab = json.load(open('$NEW_VOCAB'))
vocab_size = len(vocab['model']['vocab'])
print(f"New vocab size: {vocab_size}")

# Check checkpoint
ckpt = torch.load('$BEST_MODEL', map_location='cpu')
embed_size = ckpt['model']['gpt.text_embedding.weight'].shape[0]
print(f"Checkpoint embedding size: {embed_size}")

if vocab_size == embed_size:
    print("✅ PERFECT MATCH!")
else:
    print(f"❌ MISMATCH: {vocab_size} vs {embed_size}")
EOF
```

---

## 🎤 Step 8: Use New Model for Inference

### Option A: Update xtts_demo.py paths

Edit `xtts_demo.py` around line 50-60:

```python
# Change these paths:
xtts_checkpoint = "finetune_models/ready_clean/best_model.pth"
xtts_vocab = "finetune_models/ready_clean/vocab.json"
```

Then restart:
```bash
./launch.sh
```

### Option B: Load manually in Gradio

In the Gradio UI:
1. **Tab 2 - Load Model**
2. **Browse and select**:
   - Checkpoint: `finetune_models/ready_clean/best_model.pth`
   - Config: `finetune_models/ready_clean/config.json`
   - Vocab: `finetune_models/ready_clean/vocab.json`
3. **Click "Load Model"**

### Test Inference:

- **Text**: "ሰላም ኢትዮጵያ"
- **Language**: amh
- **Enable Amharic G2P**: ✅ YES
- **Generate**

**Expected**: Clear, intelligible Amharic speech!

---

## 🎉 Success Checklist

- [ ] Training completed without errors
- [ ] Vocab size matches checkpoint embedding size
- [ ] Test inference produces clear speech
- [ ] No more "nonsense" or gibberish output
- [ ] Pronunciation is accurate

---

## 🆘 If You Don't Have a Dataset Yet

### Quick Dataset Preparation:

1. **Collect Amharic audio**:
   ```bash
   mkdir -p finetune_models/dataset/wavs
   # Copy your .wav files here
   ```

2. **Create metadata.csv**:
   ```bash
   # Format: filename.wav|Amharic text
   cat > finetune_models/dataset/metadata.csv << 'EOF'
   audio_001.wav|ሰላም ኢትዮጵያ።
   audio_002.wav|እንዴት ነህ?
   audio_003.wav|ደህና ሁን።
   EOF
   ```

3. **Verify format**:
   ```bash
   head finetune_models/dataset/metadata.csv
   ```

### Minimum Requirements:
- **50+ audio files** (for testing)
- **200+ audio files** (for decent quality)
- **500+ audio files** (for production quality)
- **Total duration**: 30 minutes minimum (1-2 hours recommended)
- **Audio format**: 22050 Hz, mono, WAV
- **Clear speech**: No background noise, good quality

---

## 💡 Tips for Best Results

1. **More data = better results**: Try to get at least 1 hour of audio
2. **Clean audio**: Remove noise, normalize volume
3. **Accurate transcriptions**: Double-check your Amharic text
4. **Consistent speaker**: Use recordings from the same speaker
5. **Monitor training**: Check samples every 5,000 steps
6. **Train longer**: 30 epochs minimum, 50-100 for production

---

## 📞 Need Help?

If you encounter errors:

1. **Check error message** and search for it in the logs
2. **Verify dataset** format is correct
3. **Test with smaller dataset** (10-20 files) first
4. **Share error logs** for specific troubleshooting

Good luck with retraining! 🚀🇪🇹
