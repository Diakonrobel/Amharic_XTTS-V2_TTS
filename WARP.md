# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**XTTS Fine-Tuning WebUI** - A modified Gradio-based web interface for fine-tuning XTTS (Cross-lingual Text-to-Speech) models with comprehensive Amharic language support.

### Key Capabilities
- Fine-tune XTTS v2 models on custom voice datasets
- Process audio using Faster Whisper for automatic transcription
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
1. Audio Upload → utils/formatter.py
   ├─ FFmpeg conversion (if needed)
   ├─ Faster Whisper transcription (with VAD)
   ├─ Sentence segmentation
   ├─ Text cleaning (multilingual_cleaners)
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

### Change Training Parameters
Edit in `utils/gpt_train.py`:
- `BATCH_SIZE` (default: from UI, typically 2-4)
- `GRAD_ACUMM_STEPS` (gradient accumulation)
- `max_wav_length` (default: 255995 ≈ 11.6 seconds)
- `max_text_length` (default: 200 characters)
- Learning rate: `lr=5e-06`

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
- **Formats:** WAV, MP3, FLAC (converted to WAV internally)
- **Duration:** 2-60 minutes recommended (headless auto-trims to 40 min)
- **Quality:** 22.05kHz or 44.1kHz, mono preferred
- **Content:** Clean speech, minimal background noise

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

## Important Notes

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

- **Amharic G2P backends:** `docs/G2P_BACKENDS_EXPLAINED.md`
- **Amharic phonology:** `amharic_tts/g2p/README.md`
- **Usage examples:** `tests/test_amharic_integration.py`
- **Original XTTS:** https://github.com/coqui-ai/TTS

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
