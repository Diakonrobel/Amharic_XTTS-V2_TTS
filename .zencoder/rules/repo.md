---
description: Repository Information Overview
alwaysApply: true
---

# XTTS Fine-tuning WebUI Information

## Summary
A web interface for fine-tuning XTTS (Coqui TTS) models with advanced features for Amharic language support. This project extends the official XTTS fine-tuning capabilities with improved dataset processing, training optimizations, and comprehensive Amharic language support through multiple Grapheme-to-Phoneme (G2P) conversion backends.

## Structure
- **Root**: Main Python scripts, installation files, and documentation
- **utils/**: Core utility modules for audio processing, training, and dataset management
- **amharic_tts/**: Specialized modules for Amharic language support
- **docs/**: Additional documentation for specific features
- **tests/**: Test scripts for functionality verification

## Language & Runtime
**Language**: Python 3.8+
**Version**: Python 3.11 recommended (Docker uses 3.11-slim-bookworm)
**Build System**: pip
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- torch==2.1.2
- torchaudio==2.1.2
- torchvision==0.16.2
- coqui-tts[languages]==0.24.2
- faster_whisper==1.0.3
- gradio==4.44.1
- spacy==3.7.5
- fastapi==0.103.1
- pydantic==2.3.0

**Advanced Features Dependencies**:
- pysrt>=1.1.2
- yt-dlp>=2024.1.0
- youtube-transcript-api>=0.6.0
- soundfile>=0.12.1

**Optional G2P Backends**:
- transphone
- epitran

## Build & Installation
```bash
# Windows installation
python smart_install.py

# Linux/macOS installation
python smart_install.py

# With optional G2P backends
python smart_install.py --with-backends

# Force CPU-only installation
python smart_install.py --cpu-only

# Specify CUDA version
python smart_install.py --cuda-version 11.8
```

## Docker
**Dockerfile**: Dockerfile in root directory
**Image**: Based on python:3.11-slim-bookworm
**Configuration**: CUDA-enabled with NVIDIA GPU support
**Run Command**:
```bash
docker run -it --gpus all --pull always -p 7860:7860 --platform=linux/amd64 athomasson2/fine_tune_xtts:huggingface python app.py
```

## Main Entry Points
**Web Interface**: xtts_demo.py
**Headless Training**: headlessXttsTrain.py
**Dataset Checking**: check_dataset.py
**Dataset Filtering**: filter_dataset.py
**Demo Script**: xtts_demo.py

## Headless Usage
```bash
# Basic audio processing
python headlessXttsTrain.py --input_audio speaker.wav --lang en --epochs 10

# Process SRT + media file
python headlessXttsTrain.py --srt_file subtitles.srt --media_file video.mp4 --lang en --epochs 10

# Download and process YouTube video
python headlessXttsTrain.py --youtube_url "https://youtube.com/watch?v=VIDEO_ID" --lang en --epochs 10
```

## Key Features
**Data Processing**:
- Faster-whisper 0.10.0 integration with large-v3 model support
- VAD (Voice Activity Detection) filtering
- SRT/VTT subtitle processing with media files
- YouTube video downloading with transcript extraction
- RMS-based audio slicing for intelligent segmentation

**Fine-tuning**:
- Custom base model selection
- Model optimization
- Amharic language support with G2P conversion
- Vocabulary extension

**Inference**:
- Customizable inference settings
- Phoneme-based generation for Amharic

## Amharic Support
**G2P Backends**:
- Transphone (primary)
- Epitran (fallback)
- Custom Rule-Based (offline)

**Components**:
- Text normalization
- Number expansion
- Phonological processing
- Ethiopic script support

## Testing
**Framework**: Python unittest
**Test Location**: tests/ directory
**Test Files**: test_amharic_g2p_comprehensive.py, test_amharic_integration.py, test_advanced_features.py
**Run Command**:
```bash
python -m unittest discover tests
```