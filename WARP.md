# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**XTTS Fine-Tuning WebUI** - A modified Gradio-based web interface for fine-tuning XTTS (Cross-lingual Text-to-Speech) models with comprehensive Amharic language support and advanced dataset processing capabilities.

### Key Capabilities
- Fine-tune XTTS v2 models on custom voice datasets
- Process audio using Faster Whisper for automatic transcription
- **Advanced Dataset Processing:**
  - SRT subtitle files + media synchronization
  - YouTube video download with automatic transcript extraction (yt-dlp)
  - RMS-based intelligent audio slicing
  - Multi-format support (SRT, VTT, JSON transcripts)
- Support for 20+ languages including advanced Amharic (Ethiopian) TTS with G2P
- Web UI and headless training modes
- Model optimization and deployment-ready output

## Essential Commands

### Installation

**Windows:**
```powershell
# Install dependencies (uses smart_install.py)
.\install.bat

# Or manual installation:
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Linux:**
```bash
bash install.sh
```

**macOS (Apple Silicon):**
```bash
pip install --no-deps -r apple_silicon_requirements.txt
```

### Running the Application

**Start Web UI:**
```powershell
# Windows
.\start.bat

# Linux/Mac
bash start.sh

# Or directly:
python xtts_demo.py

# With custom options:
python xtts_demo.py --port 5003 --share --num_epochs 10
```

**Access the interface:** Navigate to `http://127.0.0.1:5003`

### Headless Training

```bash
# Basic training
python headlessXttsTrain.py --input_audio speaker.wav --lang en --epochs 10

# Amharic with G2P preprocessing
python headlessXttsTrain.py --input_audio amharic_speaker.wav --lang amh --epochs 10 --use_g2p

# Process SRT + media file
python headlessXttsTrain.py --srt_file subtitles.srt --media_file video.mp4 --lang en --epochs 10

# Download and process YouTube video
python headlessXttsTrain.py --youtube_url "https://youtube.com/watch?v=VIDEO_ID" --lang en --epochs 10

# View all options
python headlessXttsTrain.py --help
```

### Docker

```bash
# Run pre-built container
docker run -it --gpus all --pull always -p 7860:7860 --platform=linux/amd64 athomasson2/fine_tune_xtts:huggingface python app.py

# Build from Dockerfile
docker build -t xtts-finetune .
docker run -it --gpus all -p 5003:5003 xtts-finetune
```

### Testing

```bash
# Run Amharic integration tests
python tests/test_amharic_integration.py

# Run comprehensive G2P tests
python tests/test_amharic_g2p_comprehensive.py

# Test advanced features (SRT, YouTube, Audio Slicer)
python test_advanced_features.py

# Test specific Amharic modules
python -m pytest tests/ -v
```

## Architecture Overview

### Core Entry Points

1. **`xtts_demo.py`** - Main Gradio web UI application
   - 3 tabs: Data Processing → Fine-tuning → Inference
   - Handles audio preprocessing, training orchestration, and model testing
   - Uses `utils/formatter.py` for dataset creation
   - Uses `utils/gpt_train.py` for model training

2. **`headlessXttsTrain.py`** - CLI-based training script
   - Automated pipeline: audio prep → dataset → training → optimization
   - FFmpeg integration for audio conversion and trimming
   - Supports all web UI features without graphical interface

3. **`smart_install.py`** - Intelligent dependency installer
   - Detects CUDA availability and installs appropriate PyTorch version
   - Platform-specific installation logic

### Key Modules

#### `utils/` - Core Utilities

- **`formatter.py`** - Audio dataset preparation
  - Uses Faster Whisper for transcription with VAD filtering
  - Splits audio into sentence-level segments
  - Creates train/eval metadata CSV files
  - Applies multilingual text cleaning
  - Incremental dataset updates (skips processed files)

- **`srt_processor.py`** - SRT subtitle processing
  - Parse SRT files and synchronize with media (audio/video)
  - Extract audio segments based on subtitle timestamps
  - Support for VTT and JSON transcript formats
  - FFmpeg-based media extraction and conversion
  - Creates dataset-ready audio clips with aligned transcriptions

- **`youtube_downloader.py`** - YouTube content acquisition
  - Download videos using yt-dlp with format selection
  - Extract auto-generated or manual subtitles/transcripts
  - Support for multiple languages and fallback options
  - Automatic transcript format detection (VTT, SRT, JSON)
  - Cookie support for age-restricted content

- **`audio_slicer.py`** - Intelligent audio segmentation
  - RMS (Root Mean Square) based silence detection
  - Configurable thresholds and minimum lengths
  - Hop length and silence padding customization
  - Export segments with metadata for dataset creation
  - Integration with dataset preprocessing pipeline

- **`gpt_train.py`** - XTTS model training orchestration
  - Downloads base XTTS models (v2.0.1, v2.0.2, main)
  - Configures GPTTrainer with custom parameters
  - Supports custom model as base (fine-tune on fine-tuned models)
  - Amharic G2P integration hooks
  - Saves checkpoints to `output_path/run/training/`

- **`tokenizer.py`** - Enhanced multilingual text processing
  - Japanese support (cutlet/fugashi)
  - Language-specific cleaners

#### `amharic_tts/` - Amharic Language Support

**Architecture:** Multi-backend G2P system with automatic fallback

```
amharic_tts/
├── g2p/                              # Grapheme-to-Phoneme conversion
│   ├── amharic_g2p_enhanced.py       # Multi-backend G2P (Transphone/Epitran/Rule-based)
│   ├── ethiopic_g2p_table.py         # 259 character mappings (IPA-compliant)
│   └── amharic_g2p.py                # Basic G2P converter
├── tokenizer/
│   ├── hybrid_tokenizer.py           # G2P + BPE hybrid tokenization
│   └── xtts_tokenizer_wrapper.py     # XTTS-compatible wrapper
├── preprocessing/
│   ├── text_normalizer.py            # Ethiopic character normalization (ሥ→ስ, etc.)
│   └── number_expander.py            # Amharic number-to-word (123 → አንድ መቶ ሃያ ሶስት)
├── config/
│   └── amharic_config.py             # G2P backend configuration, phoneme inventory
└── utils/
    ├── dependency_installer.py       # Auto-install optional G2P backends
    └── preprocess_dataset.py         # Batch G2P preprocessing for datasets
```

**G2P Backend Fallback Order:**
1. **Transphone** (primary): ML-based, 7500+ languages, best accuracy
2. **Epitran** (secondary): Rule-based, explicit Ethiopic support
3. **Rule-based** (fallback): Always available, zero dependencies, 259 character mappings

**Key Features:**
- Full Ethiopic script support (U+1200-U+137F, 340+ characters)
- Ejective consonants (tʼ, pʼ, sʼ, etc.)
- Labiovelar consonants (qʷ, kʷ, gʷ, xʷ)
- Epenthetic vowel insertion (kɨt → kɨtɨ)
- Quality validation with automatic backend switching

### Training Pipeline Flow

```
1. Audio Input (Multiple Methods)
   
   A) Direct Audio Upload → utils/formatter.py
      ├─ FFmpeg conversion (if needed)
      ├─ Faster Whisper transcription (with VAD)
      ├─ Sentence segmentation
      ├─ Text cleaning (multilingual_cleaners)
      └─ CSV metadata (train/eval split)
   
   B) SRT + Media File → utils/srt_processor.py
      ├─ Parse SRT timestamps and text
      ├─ Extract audio segments using FFmpeg
      ├─ Align transcriptions with audio clips
      ├─ Text cleaning and normalization
      └─ CSV metadata (train/eval split)
   
   C) YouTube URL → utils/youtube_downloader.py
      ├─ Download video/audio with yt-dlp
      ├─ Extract available transcripts/subtitles
      ├─ Convert to SRT format if needed
      ├─ Process via SRT pipeline (method B)
      └─ CSV metadata (train/eval split)
   
   D) Audio Slicing → utils/audio_slicer.py
      ├─ RMS-based silence detection
      ├─ Intelligent segmentation
      ├─ Export clips with timestamps
      ├─ Optional: Whisper transcription
      └─ CSV metadata (train/eval split)

2. Dataset Creation → output/dataset/
   ├─ wavs/ (audio segments)
   ├─ metadata_train.csv
   ├─ metadata_eval.csv
   └─ lang.txt

3. Model Training → utils/gpt_train.py
   ├─ Download base model (if needed)
   ├─ Configure GPTTrainer (epochs, batch size, etc.)
   ├─ Train on dataset
   └─ Save checkpoints → output/run/training/

4. Optimization → xtts_demo.py (Tab 2.5)
   ├─ Find best checkpoint (best_model.pth)
   ├─ Copy to output/ready/
   │   ├─ model.pth (or unoptimize_model.pth)
   │   ├─ config.json
   │   ├─ vocab.json
   │   ├─ speakers_xtts.pth
   │   └─ reference.wav
   └─ Optional: Delete training folders

5. Inference → Tab 3
   ├─ Load model from output/ready/
   ├─ Generate speech with custom parameters
   └─ Test with reference audio
```

### Directory Structure

```
output/ or finetune_models/          # Default output directory
├── dataset/                          # Processed training data
│   ├── wavs/                         # Audio segments
│   ├── metadata_train.csv            # Training metadata
│   ├── metadata_eval.csv             # Evaluation metadata
│   └── lang.txt                      # Dataset language
├── run/training/                     # Training checkpoints
│   └── GPT_XTTS_FT-*/               # Training run folders
│       ├── best_model.pth            # Best checkpoint
│       └── *.pth                     # Other checkpoints
└── ready/                            # Deployment-ready model
    ├── model.pth                     # Optimized model
    ├── config.json                   # Model configuration
    ├── vocab.json                    # Tokenizer vocabulary
    ├── speakers_xtts.pth             # Speaker embeddings
    └── reference.wav                 # Example reference audio

base_models/                          # Downloaded base XTTS models
├── v2.0.1/                           # XTTS version directories
├── v2.0.2/
└── main/
```

## Development Patterns

### Adding Language Support

1. **Update `TTS.tts.layers.xtts.tokenizer`** with language-specific cleaners
2. **Add to `utils/tokenizer.py`** if special tokenization needed
3. **Update `utils/gpt_train.py`** for language-specific workers (`num_workers = 0` for some languages)
4. **Test with both Web UI and headless mode**

### Amharic G2P Customization

```python
# In utils/gpt_train.py, Amharic G2P is enabled via:
if use_amharic_g2p and language == "am":
    from amharic_tts.tokenizer.xtts_tokenizer_wrapper import create_xtts_tokenizer
    # G2P preprocessing applied during training
```

To modify G2P behavior:
- **Backend selection:** Edit `amharic_tts/config/amharic_config.py`
- **Phoneme mappings:** Edit `amharic_tts/g2p/ethiopic_g2p_table.py`
- **Phonological rules:** Modify `amharic_tts/g2p/amharic_g2p_enhanced.py`

### Model Checkpointing

- Base models cached in `base_models/{version}/`
- Training creates checkpoints every 1000 steps (`save_step=1000`)
- Best model selected by lowest loss
- Use `find_latest_best_model(folder_path)` from `utils/formatter.py`

### Custom Model Fine-tuning

Fine-tune an already fine-tuned model:
```python
# In Web UI Tab 2: Select "Custom model" and provide path to model.pth
# In headless mode: Not directly supported, modify gpt_train.py
```

## Common Tasks

### Process New Audio for Existing Dataset
1. Place new audio files in source folder
2. Run preprocessing (Tab 1 or headless)
3. System automatically skips previously processed files
4. New segments added to existing metadata CSVs

### Process SRT Subtitles with Media Files
1. Prepare your SRT file and corresponding audio/video file
2. In Web UI: Upload both files in Tab 1
3. Or headless: `python headlessXttsTrain.py --srt_file subs.srt --media_file video.mp4 --lang en`
4. System extracts audio segments aligned with subtitle timestamps
5. Creates dataset with text-audio pairs ready for training

### Download and Process YouTube Videos
1. Find YouTube video with good audio quality and available subtitles
2. In Web UI: Enter YouTube URL in Tab 1
3. Or headless: `python headlessXttsTrain.py --youtube_url "URL" --lang en`
4. System downloads video, extracts subtitles/transcripts
5. Processes via SRT pipeline for dataset creation
6. Note: Respects YouTube Terms of Service - for personal use only

### Use RMS Audio Slicer for Long Recordings
1. Have long audio file without transcription
2. Use audio slicer to intelligently segment based on silence
3. Parameters: `threshold_db=-40`, `min_length=5.0`, `silence_window=0.3`
4. Optionally run Whisper on segments for automatic transcription
5. Creates dataset with evenly-sized, content-aware segments

### Change Training Parameters
Edit in `utils/gpt_train.py`:
- `BATCH_SIZE` (default: from UI, typically 2-4)
- `GRAD_ACUMM_STEPS` (gradient accumulation)
- `max_wav_length` (default: 255995 ≈ 11.6 seconds)
- `max_text_length` (default: 200 characters)
- Learning rate: `lr=5e-06`

### Training with Small Datasets (<3000 samples)

**Problem:** XTTS v2 (520M parameters) easily overfits on small datasets (1-3 hours / 1000-3000 samples).

**Solution:** Use optimized configuration from `utils/xtts_small_dataset_config.py`:

```python
from utils.xtts_small_dataset_config import XTTSSmallDatasetConfig, EarlyStoppingCallback

# Print configuration summary
XTTSSmallDatasetConfig.print_config_summary()

# Apply to training
config = GPTTrainerConfig(
    epochs=XTTSSmallDatasetConfig.MAX_EPOCHS,  # 2 epochs only
    batch_size=XTTSSmallDatasetConfig.BATCH_SIZE,  # 1
    lr=XTTSSmallDatasetConfig.LEARNING_RATE,  # 5e-7
    optimizer_params={"weight_decay": XTTSSmallDatasetConfig.WEIGHT_DECAY},  # 0.1
    ...
)

# Apply layer freezing (train only ~5-10% of parameters)
model = GPTTrainer.init_from_config(config)
total, trainable = XTTSSmallDatasetConfig.apply_layer_freezing(model)

# Setup early stopping
early_stopping = EarlyStoppingCallback(patience=1, min_delta=0.01)

# Training loop with early stopping
for epoch in range(config.epochs):
    train_loss = trainer.train_epoch()
    val_loss = trainer.eval_epoch()
    
    if early_stopping(val_loss, epoch):
        print("🛑 Early stopping - preventing overfitting")
        break
```

**Key Techniques:**

1. **Layer Freezing** (Most Important)
   - Freeze encoder and first 28 of 30 GPT layers
   - Only train text embeddings, last 2 GPT layers, and output head
   - Reduces trainable params from 520M → ~50M (90% frozen)

2. **Very Low Learning Rate**
   - Use 5e-7 (10x lower than default)
   - Small datasets need gentle updates

3. **Minimal Epochs**
   - Only 2 epochs for <2000 samples
   - More epochs = memorization not learning

4. **Early Stopping**
   - Automatically stops when validation loss increases
   - Prevents wasting compute on overfitting

5. **Small Batch + High Gradient Accumulation**
   - Batch size = 1, Gradient accumulation = 16
   - Effective batch size = 16
   - Better gradients for limited data

6. **High Regularization**
   - Weight decay = 0.1 (stronger L2 regularization)
   - Gradient clipping = 0.5

7. **Audio Augmentation** (Optional)
   ```python
   from utils.audio_augmentation import SimpleAudioAugmenter
   
   augmenter = SimpleAudioAugmenter(noise_prob=0.3)
   waveform = augmenter.augment(waveform)
   ```

**Expected Results:**
```
Bad (Overfitting):
  Epoch 0: val_loss = 3.412 ✅
  Epoch 1: val_loss = 3.427 ⚠️  (+0.015)
  Epoch 2: val_loss = 3.607 🚨 (+0.180)

Good (With Anti-Overfitting):
  Epoch 0: val_loss = 3.412 ✅
  Epoch 1: val_loss = 3.350 ✅ (-0.062)
  Epoch 2: val_loss = 3.380 ⚠️  → Early Stop
  Best model: Epoch 1
```

**See:** `docs/SMALL_DATASET_TRAINING.md` for complete guide

### Optimize Model for Deployment
After training:
1. Go to Tab 2.5 in Web UI
2. Click "Optimize Model" - copies best checkpoint to `ready/`
3. Optionally delete training folders to save space
4. Use files in `ready/` for inference or distribution

### Run Inference with Custom Settings
```python
# In Tab 3 of Web UI, toggle "Use Inference Settings" to customize:
# - temperature (0.1-1.0, lower = more deterministic)
# - length_penalty (1.0-2.0, controls length)
# - repetition_penalty (1.0-10.0, reduces repetition)
# - top_k (1-100, sampling diversity)
# - top_p (0.1-1.0, nucleus sampling)
# - sentence_split (enable for long texts)
```

## Technical Constraints

### Hardware Requirements
- **GPU:** CUDA-capable GPU required for training (RTX 3060+ recommended)
- **VRAM:** Minimum 6GB, 12GB+ recommended for larger batches
- **RAM:** 16GB+ recommended
- **Storage:** ~5GB for base models, ~10-50GB per training run

### Audio Requirements
- **Formats:** WAV, MP3, FLAC, MP4, MKV, AVI (converted to WAV internally via FFmpeg)
- **Duration:** 2-60 minutes recommended (headless auto-trims to 40 min)
- **Quality:** 22.05kHz or 44.1kHz, mono preferred
- **Content:** Clean speech, minimal background noise
- **SRT Processing:** Media file must match SRT duration, supports video extraction
- **YouTube:** 1080p max recommended, automatic audio extraction

### Language Support
Supported languages (use ISO 639-3 codes):
- `en` (English), `es` (Spanish), `fr` (French), `de` (German), `it` (Italian)
- `pt` (Portuguese), `pl` (Polish), `tr` (Turkish), `ru` (Russian)
- `nl` (Dutch), `cs` (Czech), `ar` (Arabic), `zh-cn` (Chinese)
- `ja` (Japanese, requires special handling)
- `amh` or `am` (Amharic, with G2P support)
- More: See XTTS documentation

### Known Limitations
- **Japanese:** Requires `num_workers=0` in training (cutlet/fugashi compatibility)
- **Long audio:** Auto-trimmed to prevent OOM (max_audio_length=255995 frames)
- **Incremental training:** Cannot easily resume interrupted training runs
- **Model mixing:** Mixing different XTTS versions may cause issues
- **YouTube downloads:** Requires active internet connection, respects rate limits
- **SRT sync:** Requires accurate timestamps, may need manual adjustment for poorly-synced subtitles
- **FFmpeg:** Required for media processing, audio extraction, and format conversion

## Important Notes

### Advanced Features Dependencies
Required for advanced dataset processing:
```bash
pip install pysrt        # SRT file parsing
pip install yt-dlp       # YouTube video downloading
pip install soundfile    # Audio file I/O for slicing
# FFmpeg must be installed and in PATH
```

### Amharic G2P Installation
Optional backends for better quality:
```bash
pip install transphone  # Best quality (primary)
pip install epitran     # Good fallback
# Rule-based backend always available (no install needed)
```

### Web UI Behavior
- Interface runs on `127.0.0.1:5003` by default
- Use `--share` flag for public Gradio link
- Logs are **not** displayed in UI (check console)
- Can restart safely - state preserved in files

### Model Paths
- Never hardcode paths - use Path() from pathlib
- Default output: `Path.cwd() / "finetune_models"`
- Base models: `Path.cwd() / "base_models"`

### Dataset Language Consistency
- `lang.txt` file in dataset folder enforces language consistency
- Changing language requires confirming mismatch warning
- Prevents accidental mixing of language datasets

## Documentation References

- **Implementation Plan:** `IMPLEMENTATION_PLAN.md` - Complete integration roadmap
- **Advanced Features Tests:** `test_advanced_features.py` - Test suite and usage examples
- **Amharic G2P backends:** `docs/G2P_BACKENDS_EXPLAINED.md`
- **Amharic phonology:** `amharic_tts/g2p/README.md`
- **Usage examples:** `tests/test_amharic_integration.py`
- **Original XTTS:** https://github.com/coqui-ai/TTS
- **Dataset-Maker Project:** https://github.com/JarodMica/dataset-maker

## Platform-Specific Notes

### Windows (PowerShell)
- Use `.\script.bat` not `script.bat`
- PyTorch CUDA installation requires explicit index URL
- Virtual env: `python -m venv venv; .\venv\Scripts\activate`

### Linux
- Ensure CUDA toolkit installed for GPU training
- May need `libsndfile1` for audio: `sudo apt install libsndfile1`
- Use `bash` not `sh` for install/start scripts

### macOS (Apple Silicon)
- Use special requirements file: `apple_silicon_requirements.txt`
- Training slower than CUDA (MPS backend)
- Some features may require Rosetta 2

## Workflow Integration

This repository uses `.warp/workflows/` for structured development:
- **`plan.md`** - Feature planning workflow
- **`implement.md`** - Implementation workflow with TDD
- **`specify.md`** - Specification creation
- **`tasks.md`** - Task tracking

Follow the Spec-Driven Development lifecycle when adding features.
