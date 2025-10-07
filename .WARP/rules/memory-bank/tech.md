# Technology Stack

## Core Technologies

### Python & Frameworks
- **Python**: 3.9+ (3.10 recommended for M1/M2 Macs)
- **PyTorch**: 2.1.1-2.1.2 (CUDA 11.8 build)
- **TorchAudio**: 2.1.1-2.1.2
- **Gradio**: 4.44.1 (web interface)
- **FastAPI**: 0.103.1 (for API extensions)

### TTS & Speech Processing
- **Coqui TTS**: 0.24.2 (XTTS v2 implementation)
- **Faster-Whisper**: 1.0.3 (ASR for transcription)
- **SpaCy**: 3.7.5 (text processing)
- **Librosa**: Audio analysis

### Amharic-Specific (NEW)
- **Transphone**: 0.1.0+ (primary G2P backend)
- **Epitran**: 1.24+ (fallback G2P backend)
- **PyPinyin**: For Chinese comparison/patterns
- **Cutlet**: Japanese romanization (pattern reference)

## Development Environment

### Platform Support
- **Windows**: Native support via .bat scripts
- **Linux**: Native support via .sh scripts  
- **macOS**: Apple Silicon support (M1/M2) with special requirements

### GPU Requirements
- **Minimum**: NVIDIA GPU with 6GB VRAM
- **Recommended**: 8GB+ VRAM
- **CUDA**: 11.8+ (cu118 builds)
- **Compute Capability**: 3.5+ (Check with `nvidia-smi`)

### CPU Fallback
- Works but significantly slower (10-50x)
- Not recommended for training
- OK for inference on short texts

## Project Setup

### Installation Methods

#### Windows
```powershell
# Run installation script
install.bat

# Manually:
python -m venv venv
venv\Scripts\activate
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### Linux
```bash
# Run installation script
bash install.sh

# Manually:
python -m venv venv
source venv/bin/activate
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### Apple Silicon (M1/M2)
```bash
pip install --no-deps -r apple_silicon_requirements.txt
```

#### Manjaro (x86)
```bash
pip install --no-deps -r ManjaroX86Python3.11.11_requirements.txt
```

### Additional Dependencies for Amharic

Create `requirements-amharic.txt`:
```
transphone>=0.1.0
epitran>=1.24
```

Install:
```bash
pip install -r requirements-amharic.txt
```

## Running the Application

### Web Interface
```bash
# Windows
start.bat

# Linux/Mac
bash start.sh

# Manual
python xtts_demo.py --port 5003
```

Access at: `http://127.0.0.1:5003`

### Headless Training
```bash
python headlessXttsTrain.py \
  --input_audio speaker.wav \
  --lang en \
  --epochs 10 \
  --batch_size 2
```

### Docker
```bash
docker run -it --gpus all --pull always \
  -p 7860:7860 \
  --platform=linux/amd64 \
  athomasson2/fine_tune_xtts:huggingface \
  python app.py
```

## Development Tools

### Recommended IDE Setup
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - isort
- **PyCharm Professional** (alternative)

### Code Quality Tools
```bash
# Formatting
black xtts_demo.py utils/ amharic_tts/
isort xtts_demo.py utils/ amharic_tts/

# Linting
flake8 xtts_demo.py utils/ amharic_tts/
pylint xtts_demo.py utils/ amharic_tts/

# Type Checking
mypy xtts_demo.py utils/ amharic_tts/
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=utils --cov=amharic_tts tests/

# Run specific test
pytest tests/test_amharic_g2p.py -v
```

## File Formats & Specifications

### Audio Formats
**Input**: WAV, MP3, FLAC
**Output**: WAV (22050 Hz, 16-bit PCM, mono)
**Training Sample Rate**: 22050 Hz
**Inference Sample Rate**: 24000 Hz

### Text Encoding
**Required**: UTF-8
**Ethiopic Unicode Range**: U+1200 to U+137F
**Amharic Punctuation**: U+1361 to U+1368

### Dataset Structure
```
output_folder/
├── dataset/
│   ├── wavs/               # Segmented audio clips
│   ├── metadata_train.csv  # Training metadata
│   ├── metadata_eval.csv   # Evaluation metadata
│   └── lang.txt           # Language marker
├── run/
│   └── training/          # Checkpoints during training
└── ready/
    ├── model.pth          # Optimized model
    ├── config.json        # Model configuration
    ├── vocab.json         # Tokenizer vocabulary
    ├── speakers_xtts.pth  # Speaker embeddings
    └── reference.wav      # Reference audio
```

### Metadata CSV Format
```csv
audio_file|text|speaker_name
wavs/sample_00000001.wav|ሰላም ዓለም|speaker_001
wavs/sample_00000002.wav|እንዴት ነህ|speaker_001
```

## Technical Constraints

### Memory Limits
- **Training**: 8GB GPU RAM (with batch_size=2)
- **Inference**: 2GB GPU RAM
- **Dataset Creation**: 4GB system RAM per worker

### Processing Limits
- **Max Audio Length**: 11 seconds per training segment
- **Max Text Length**: 256 characters (Amharic)
- **Min Dataset Size**: 2 minutes of speech (120 seconds)
- **Recommended Dataset**: 20+ minutes per speaker

### Model Limits
- **XTTS v2 Token Limit**: 512 tokens
- **Max Speakers**: Unlimited (multi-speaker fine-tuning)
- **Model Size**: ~1.5GB (unoptimized), ~700MB (optimized)

## Dependency Management

### Core Dependencies (requirements.txt)
```
faster_whisper==1.0.3
gradio==4.44.1
spacy==3.7.5
coqui-tts[languages]==0.24.2
cutlet==0.5.0
fugashi[unidic-lite]==1.4.0
fastapi==0.103.1
pydantic==2.3.0
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2
```

### Python Version Requirements
- **Minimum**: Python 3.9
- **Recommended**: Python 3.10
- **Maximum**: Python 3.11 (M1/M2 Macs)

### Virtual Environment
Always use virtual environment to avoid conflicts:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

## Configuration Files

### Gradio Configuration
- **Port**: 5003 (default)
- **Share**: False (local only)
- **Analytics**: Disabled (`GRADIO_ANALYTICS_ENABLED=False`)

### Training Configuration
- **Default Epochs**: 6 (web), 10 (headless)
- **Default Batch Size**: 2
- **Gradient Accumulation**: 1
- **Max Audio Length**: 11 seconds
- **Learning Rate**: 5e-6
- **Optimizer**: AdamW

### Whisper Configuration
- **Default Model**: large-v3
- **VAD Filter**: Enabled
- **Language**: Auto-detect or specified
- **Compute Type**: float16 (GPU), float32 (CPU)

## Environment Variables

### Optional Settings
```bash
# Disable Gradio analytics
export GRADIO_ANALYTICS_ENABLED=False

# CUDA device selection
export CUDA_VISIBLE_DEVICES=0

# Whisper model cache
export WHISPER_MODEL=large-v3

# Output directory
export XTTS_OUTPUT_DIR=./finetune_models
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Solution**: Reduce batch_size or use gradient accumulation
```bash
python xtts_demo.py --batch_size 1 --grad_acumm 2
```

#### 2. Whisper Model Download Fails
**Solution**: Pre-download model manually
```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3')"
```

#### 3. UTF-8 Encoding Errors
**Solution**: Ensure system locale supports UTF-8
```bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

#### 4. FFmpeg Not Found (Headless Mode)
**Solution**: Install FFmpeg
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from ffmpeg.org and add to PATH
```

### Performance Optimization

#### Training Speed
- Use batch_size=2 (max for 8GB GPU)
- Enable mixed precision (automatic)
- Use gradient accumulation for larger effective batch size
- Ensure CUDA is properly installed

#### Inference Speed
- Load model once, reuse for multiple generations
- Use optimized model (model.pth vs unoptimize_model.pth)
- Enable GPU if available
- Use caching for repeated text

#### Dataset Processing
- Use faster Whisper model for development (medium vs large-v3)
- Process audio in parallel (num_workers parameter)
- Pre-filter silent segments
- Use VAD filter to skip non-speech

## Tool Usage Patterns

### Common Commands
```bash
# Start web interface
python xtts_demo.py

# Train headless with Amharic
python headlessXttsTrain.py --input_audio amharic.wav --lang amh --epochs 10

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check installed packages
pip list | grep -E "torch|tts|whisper|gradio"

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Directory Navigation
```bash
# Output structure
finetune_models/
└── {model_name}/
    ├── dataset/
    ├── run/
    └── ready/
```

### Logs and Debugging
- **Gradio logs**: Console output during web session
- **Training logs**: `run/training/trainer_0_log.txt` (deleted after training)
- **Error messages**: Console + exceptions in UI

## Integration Patterns

### Adding Amharic Support
1. Install Amharic dependencies:
   ```bash
   pip install transphone epitran
   ```

2. Create amharic_tts/ module structure
3. Extend utils/tokenizer.py with Amharic
4. Add "amh" to language dropdowns
5. Test with Amharic audio dataset

### External Tool Integration
- **FFmpeg**: Audio conversion (headless mode)
- **Whisper**: ASR transcription
- **SpaCy**: Text processing
- **Transphone/Epitran**: G2P conversion (NEW)

## Version Control

### Git Configuration
```bash
# Ignore virtual environment
echo "venv/" >> .gitignore

# Ignore output folders
echo "finetune_models/" >> .gitignore
echo "output/" >> .gitignore

# Ignore model weights
echo "*.pth" >> .gitignore
echo "base_models/" >> .gitignore
```

### Recommended .gitignore
```
venv/
__pycache__/
*.pyc
*.pth
*.pth.tar
finetune_models/
output/
base_models/
.DS_Store
*.log
.vscode/
.idea/
```
