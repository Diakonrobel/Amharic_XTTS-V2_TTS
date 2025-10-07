# Product Overview

## Why This Exists

### Problems Solved
1. **High Barrier to Entry:** Fine-tuning TTS models typically requires:
   - Deep understanding of ML training pipelines
   - Manual audio preprocessing and segmentation
   - Complex configuration of hyperparameters
   - Command-line comfort and debugging skills
   
   **Solution:** Gradio web UI guides users through 3 simple tabs with sensible defaults.

2. **Lack of Ethiopic Language Support:** Amharic and other Ethiopian languages have:
   - Complex orthography (Ethiopic script, 340+ characters)
   - Sparse TTS research and tooling
   - No standard phoneme representations
   
   **Solution:** Multi-backend G2P system with automatic fallback (Transphone → Epitran → Rule-based).

3. **Dataset Preparation Burden:** Creating training datasets requires:
   - Manual audio transcription
   - Precise timestamp alignment
   - Voice activity detection
   - Sentence-level segmentation
   
   **Solution:** Automated pipeline using Faster Whisper with VAD, creates ready-to-use metadata.

4. **Model Deployment Complexity:** Trained models are scattered across checkpoint files:
   - Multiple .pth files in training folders
   - Separate config, vocab, speaker files
   - Unclear which checkpoint is "best"
   
   **Solution:** One-click optimization creates single `ready/` folder with all necessary files.

## How It Works

### User Journey

#### Web UI Flow (Interactive)
```
1. Tab 1: Data Processing
   ↓ Upload audio files (WAV/MP3/FLAC)
   ↓ Select language (Amharic, English, Spanish, etc.)
   ↓ Optional: Enable Amharic G2P preprocessing
   ↓ Click "Step 1 - Create Dataset"
   → Automatic transcription & segmentation
   → Creates metadata_train.csv and metadata_eval.csv

2. Tab 2: Fine-tuning
   ↓ Select base XTTS version (v2.0.1, v2.0.2, main)
   ↓ Configure training (epochs, batch size, etc.)
   ↓ Optional: Use custom model as base
   ↓ Click "Step 2 - Run Training"
   → Downloads base model (if not cached)
   → Trains GPT encoder on your voice
   → Saves checkpoints every 1000 steps
   
   ↓ Click "Step 2.5 - Optimize Model" (after training)
   → Finds best checkpoint
   → Creates deployment-ready folder

3. Tab 3: Inference
   ↓ Load trained model from ready/ folder
   ↓ Input text to synthesize
   ↓ Upload reference audio (3-10 seconds)
   ↓ Adjust voice parameters (temperature, etc.)
   ↓ Click "Step 3 - Generate Speech"
   → Produces speech in your voice
```

#### Headless Flow (Automated)
```bash
python headlessXttsTrain.py \
  --input_audio my_voice.wav \
  --lang amh \
  --epochs 10 \
  --use_g2p
```
Runs entire pipeline: prep → dataset → train → optimize

### Core Features

#### 1. Automated Dataset Creation
- **Input:** Raw audio files (2-60 minutes of clean speech)
- **Process:** 
  - Faster Whisper transcribes with word-level timestamps
  - VAD filter removes silence and non-speech
  - Segments split on sentence boundaries (!, ?, ., 。)
  - Text cleaned via XTTS multilingual_cleaners
- **Output:** CSV metadata with audio paths, transcriptions, speaker labels

#### 2. Incremental Updates
- Tracks processed files by prefix in metadata
- New audio added without re-processing old files
- Maintains language consistency via `lang.txt`

#### 3. Language-Specific Processing
- **Japanese:** Special tokenization (cutlet/fugashi), `num_workers=0`
- **Amharic:** G2P conversion, Ethiopic normalization, number expansion
- **General:** ISO 639-3 language codes for all XTTS-supported languages

#### 4. Training Orchestration
- Downloads and caches base XTTS models
- Configures GPTTrainer with optimized hyperparameters
- Supports custom model as base (fine-tune on fine-tuned)
- Saves best checkpoint based on validation loss

#### 5. Model Optimization
- Locates `best_model.pth` from training run
- Copies to `ready/` folder with all dependencies:
  - model.pth (or unoptimize_model.pth)
  - config.json, vocab.json, speakers_xtts.pth
  - reference.wav (example audio)
- Optional cleanup of large training folders

## User Experience Goals

### For Beginners
- **No code required:** Click buttons, see progress bars
- **Clear feedback:** Status messages explain each step
- **Sensible defaults:** Works out-of-box for most use cases
- **Error recovery:** Can restart UI without losing progress

### For Researchers
- **Amharic focus:** State-of-art Ethiopic script handling
- **G2P flexibility:** Choose backend based on accuracy vs. speed needs
- **Reproducibility:** All training parameters visible and adjustable
- **Extensibility:** Add new languages following `amharic_tts/` pattern

### For Developers
- **Headless mode:** Integrate into CI/CD or batch processing
- **File-based state:** Easy debugging and inspection
- **Docker support:** Reproducible environments
- **API-friendly:** All functions callable programmatically

## Key Metrics
- **Time to first model:** < 2 hours (10 min audio, 10 epochs)
- **Audio quality:** Subjective MOS > 3.5 (intelligibility + naturalness)
- **Amharic accuracy:** G2P phoneme error rate < 10% vs. manual annotation
- **GPU efficiency:** Training fits in 6GB VRAM with batch_size=2
