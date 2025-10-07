# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is an XTTS (eXtended Text-To-Speech) fine-tuning web interface that allows users to train custom voice models using their own audio data. It's a modified version of the official Coqui TTS webui with enhanced features for dataset processing, model training, and inference.

## Key Commands

### Setup and Installation
```bash
# Windows installation
install.bat

# Linux/Mac installation  
bash install.sh

# Manual setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\scripts\activate     # Windows
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Running the Application
```bash
# Windows
start.bat

# Linux/Mac
bash start.sh

# Manual start
python xtts_demo.py
```

### Headless Training
```bash
# Run complete training pipeline headlessly
python headlessXttsTrain.py --input_audio speaker.wav --lang en --epochs 10

# See all available parameters
python headlessXttsTrain.py --help
```

### Docker Usage
```bash
# Run with Docker
docker run -it --gpus all --pull always -p 7860:7860 --platform=linux/amd64 athomasson2/fine_tune_xtts:huggingface python app.py
```

## Architecture Overview

### Core Components

**Main Application (`xtts_demo.py`)**
- Gradio-based web interface with 3 main tabs:
  1. Data Processing - converts audio to training dataset
  2. Fine-tuning - trains XTTS model on dataset  
  3. Inference - loads trained model for TTS generation
- Handles file uploads, model downloads, and user interactions

**Headless Training (`headlessXttsTrain.py`)**
- Command-line interface for automated training pipeline
- Includes audio preprocessing, dataset creation, training, optimization, and inference
- Supports FFmpeg integration for audio format conversion and trimming

**Utilities (`utils/`)**
- `formatter.py` - Audio processing and dataset creation using Whisper ASR
- `gpt_train.py` - Core XTTS training logic with Coqui TTS trainer
- `tokenizer.py` - Multilingual text processing and normalization

### Data Flow
1. **Input**: Raw audio files (MP3, WAV, FLAC)
2. **Processing**: Whisper transcription → segmentation → dataset creation
3. **Training**: XTTS fine-tuning using base model + custom dataset
4. **Optimization**: Model size reduction by removing optimizer states and DVAE weights
5. **Output**: Ready-to-use TTS model files

### Key Features
- VAD (Voice Activity Detection) filtering during transcription
- Incremental dataset building (won't reprocess existing data)
- Support for 16 languages including Japanese with specialized handling
- Model optimization for reduced file size
- Multiple reference audio sample rates (16kHz, 24kHz)
- Language validation between dataset and training configuration

## File Structure

```
├── xtts_demo.py           # Main Gradio web interface
├── headlessXttsTrain.py   # CLI training pipeline
├── utils/
│   ├── formatter.py       # Audio processing & dataset creation
│   ├── gpt_train.py      # Training logic
│   └── tokenizer.py      # Text normalization
├── requirements.txt      # Python dependencies
├── install.bat/sh        # Setup scripts
├── start.bat/sh         # Launch scripts
└── Dockerfile           # Container configuration
```

### Output Directory Structure
When training, the following structure is created:
```
output_folder/
├── dataset/              # Processed audio segments and metadata
│   ├── wavs/            # Individual audio clips
│   ├── metadata_train.csv
│   ├── metadata_eval.csv
│   └── lang.txt         # Dataset language marker
├── run/                 # Training checkpoints and logs
│   └── training/
└── ready/               # Final optimized model files
    ├── model.pth        # Optimized model
    ├── config.json      # Model configuration
    ├── vocab.json       # Tokenizer vocabulary
    ├── speakers_xtts.pth # Speaker embeddings
    └── reference*.wav   # Reference audio files
```

## Development Notes

### Audio Processing Pipeline
- Uses `faster-whisper` for transcription with VAD filtering
- Segments audio at sentence boundaries with configurable buffer
- Supports multilingual cleaning via `multilingual_cleaners()`
- Minimum segment length: 1/3 second
- Default max audio length: 11 seconds for training

### Training Configuration
- Base models: XTTS v2.0.0 through v2.0.3
- Custom base model support (local files or URLs)
- GPU memory optimization with `torch.cuda.empty_cache()`
- Automatic language detection from dataset
- Supports custom model continuing (fine-tune an already fine-tuned model)

### Language Support
Fully supported languages: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, hu, ko, ja

Special handling for:
- Japanese: Uses `cutlet` for romanization, reduced worker threads
- Chinese: Uses `pypinyin` for transliteration
- Korean: Uses `hangul_romanize` for transliteration

### GPU Requirements
- CUDA-compatible GPU recommended
- Minimum 6GB VRAM for training
- CPU fallback available but significantly slower

## Common Issues and Solutions

### Memory Issues
- Reduce batch size if encountering CUDA OOM errors
- Use gradient accumulation instead of larger batches
- Clear GPU cache between major operations

### Audio Quality
- Ensure input audio is clean with minimal background noise
- Recommended: 44.1kHz or higher sample rate input
- Minimum 2 minutes of total speech for quality results

### Language Mismatches
- The system validates dataset language against training language
- Language is automatically detected and saved during dataset creation
- Mismatches will be auto-corrected with warnings

## Default Configuration
- Default port: 5003 (accessible at `127.0.0.1:5003`)
- Default output directory: `finetune_models/`
- Default epochs: 6 (headless: 10)
- Default batch size: 2
- Default Whisper model: large-v3

## Amharic TTS Enhancement (In Progress)

This repository is being enhanced with comprehensive Amharic language support, including:

### Planned Features
1. **G2P (Grapheme-to-Phoneme) Conversion**
   - Transphone backend (primary)
   - Epitran backend (fallback)
   - Custom rule-based system (offline)
   - Phonological rules: epenthesis, gemination, labiovelars

2. **Ethiopic Script Support**
   - Full Unicode range (U+1200-U+137F, 340+ characters)
   - Proper handling of Amharic punctuation (።፣፤፥)
   - Character normalization for variants (ሥ→ስ, ዕ→እ)

3. **Amharic Text Processing**
   - Number-to-word expansion (123 → "አንድ መቶ ሃያ ሶስት")
   - Abbreviation handling (ዓ.ም, ክ.ክ, ት.ቤት)
   - Mixed script handling (Amharic-English code-switching)

4. **Enhanced Tokenizer**
   - Ethiopic character vocabulary
   - IPA phoneme vocabulary  
   - BPE for common Amharic syllables
   - Dual-mode operation (text or phonemes)

### Architecture
The Amharic support is implemented as a modular extension in `amharic_tts/`:
```
amharic_tts/
├── g2p/                      # G2P conversion module
│   ├── amharic_g2p.py       # Main G2P converter
│   └── phoneme_rules.py     # Phonological rules
├── tokenizer/
│   └── amharic_tokenizer.py # Ethiopic script tokenizer
├── preprocessing/
│   ├── text_normalizer.py   # Text cleaning
│   └── number_expander.py   # Number to words
└── config/
    ├── amharic_config.yaml  # Configuration
    └── phoneme_mapping.json # Grapheme→Phoneme mappings
```

### Memory Bank
Comprehensive project documentation is maintained in `.warp/rules/memory-bank/`:
- `brief.md` - Project mission and goals
- `product.md` - Product vision and user experience
- `context.md` - Current state and recent changes
- `architecture.md` - System architecture and integration points
- `tech.md` - Technology stack and development setup

### Usage (When Complete)
```bash
# Web Interface - Select "Amharic" from language dropdown
python xtts_demo.py

# Headless Training with Amharic
python headlessXttsTrain.py --input_audio amharic_speaker.wav --lang amh --epochs 10
```

### Quality Targets
- MOS Score: ≥ 4.0 (near-human quality)
- WER: < 10% (high intelligibility)
- RTF: < 0.3 (faster than real-time)
- Speaker similarity: > 0.85 (accurate voice cloning)

### Design Principles
- Backward compatible (no breaking changes)
- Modular architecture (isolated in `amharic_tts/`)
- Extensible to other Ethiopian languages (Tigrinya, Oromo)
- Scientifically accurate phonological modeling
