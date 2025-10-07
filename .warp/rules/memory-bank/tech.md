# Technology Stack

## Core Technologies

### Python (3.10+)
- **Primary language** for all code
- Requires 3.10+ for type hints and modern features
- **Platform support:** Windows, Linux, macOS

### Deep Learning Framework
- **PyTorch 2.1.2** (with CUDA 11.8 or CPU)
- **torchaudio 2.1.2** - Audio processing and I/O
- **torchvision 0.16.2** - Required by some TTS dependencies

### Text-to-Speech
- **Coqui TTS 0.24.2** (`coqui-tts[languages]`)
  - XTTS v2 architecture
  - GPTTrainer for fine-tuning
  - Multilingual tokenizer
  - Source: https://github.com/coqui-ai/TTS

### Audio Processing
- **Faster Whisper 1.0.3** - Speech-to-text transcription
  - OpenAI Whisper optimized with CTranslate2
  - VAD (Voice Activity Detection) built-in
  - Word-level timestamps
  - Models: large-v3, large-v2, large, medium, small
  
- **librosa** - Audio analysis utilities
- **FFmpeg** - Audio format conversion (external dependency)

### Web UI
- **Gradio 4.44.1** - Interactive web interface
  - Tab-based layout
  - File uploads
  - Progress bars
  - Audio playback

### Data Processing
- **pandas** - CSV metadata manipulation
- **numpy** - Numerical operations

## Language-Specific Dependencies

### Japanese Support
- **cutlet 0.5.0** - Romaji conversion
- **fugashi[unidic-lite] 1.4.0** - Morphological analysis
- **Note:** Requires `num_workers=0` in training

### Amharic Support (Optional)
- **transphone** (optional) - Zero-shot G2P for 7500+ languages
- **epitran** (optional) - Rule-based G2P with Ethiopic support
- **Rule-based backend** (built-in) - Zero dependencies, always available

### Natural Language Processing
- **spacy 3.7.5** - Text processing utilities

## Development Tools

### Installation
- **smart_install.py** - Custom installer script
  - Platform detection (Windows/Linux/macOS)
  - CUDA availability check
  - Automatic PyTorch version selection
  
### Batch Files / Shell Scripts
- **Windows:** `install.bat`, `start.bat`
- **Linux/Mac:** `install.sh`, `start.sh`
- **Docker:** `Dockerfile`, `start-container.sh`

## Technical Constraints

### Hardware Requirements

**Training (Minimum):**
- GPU: NVIDIA GPU with CUDA support
- VRAM: 6GB minimum (RTX 3060 or equivalent)
- RAM: 16GB system RAM
- Storage: 20GB free (base models + training data)

**Training (Recommended):**
- GPU: RTX 3090, RTX 4080, or A100
- VRAM: 12GB+ (enables larger batch sizes)
- RAM: 32GB system RAM
- Storage: 50GB+ free

**Inference (CPU):**
- Possible but slow (~10-30x slower than GPU)
- 8GB+ RAM recommended

### CUDA Setup
**Windows:**
```powershell
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**Linux:**
```bash
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

**macOS (Apple Silicon):**
```bash
pip install --no-deps -r apple_silicon_requirements.txt
# Uses MPS backend (slower than CUDA)
```

### Audio Constraints
- **Max segment length:** 255995 frames (~11.6 seconds at 22.05kHz)
- **Min segment length:** ~0.33 seconds
- **Sample rates:** 22.05kHz (training), 24kHz (output)
- **Channels:** Mono preferred (stereo auto-converted)
- **Formats:** WAV, MP3, FLAC (FFmpeg converts to WAV)

### Model Constraints
- **Base models cached:** ~5GB per XTTS version
- **Training checkpoints:** 1-10GB per training run
- **Final model:** ~500MB-1GB (optimized)

## Development Setup

### Quick Start (Windows)
```powershell
# 1. Clone repository
git clone <repository-url>
cd xtts-finetune-webui-fresh

# 2. Install dependencies
.\install.bat

# 3. Start web UI
.\start.bat
```

### Quick Start (Linux)
```bash
# 1. Install system dependencies
sudo apt install ffmpeg libsndfile1

# 2. Install Python dependencies
bash install.sh

# 3. Start web UI
bash start.sh
```

### Virtual Environment (Manual)
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Docker Setup
```bash
# Build image
docker build -t xtts-finetune .

# Run container
docker run -it --gpus all -p 5003:5003 xtts-finetune

# Or use pre-built image
docker run -it --gpus all -p 7860:7860 athomasson2/fine_tune_xtts:huggingface python app.py
```

## Tool Usage Patterns

### Faster Whisper
```python
from faster_whisper import WhisperModel

# Initialize model
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# Transcribe with VAD and word timestamps
segments, info = model.transcribe(
    audio_path,
    vad_filter=True,           # Remove silence
    word_timestamps=True,      # Word-level timing
    language="en"              # Or auto-detect
)
```

### XTTS Training
```python
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer, GPTTrainerConfig, GPTArgs

# Configure model
model_args = GPTArgs(
    max_wav_length=255995,
    max_text_length=200,
    xtts_checkpoint="path/to/base/model.pth",
    tokenizer_file="path/to/vocab.json"
)

# Configure training
config = GPTTrainerConfig(
    epochs=10,
    batch_size=2,
    lr=5e-06,
    save_step=1000
)

# Initialize and train
model = GPTTrainer.init_from_config(config)
trainer = Trainer(...)
trainer.fit()
```

### XTTS Inference
```python
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig

# Load model
config = XttsConfig()
config.load_json("config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path="model.pth", vocab_path="vocab.json")

# Get conditioning latents from reference audio
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path="reference.wav"
)

# Generate speech
output = model.inference(
    text="Hello world",
    language="en",
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    temperature=0.7
)
```

## Dependency Management

### requirements.txt
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

### Platform-Specific Requirements
- **apple_silicon_requirements.txt** - macOS ARM (M1/M2/M3)
- **ManjaroX86Python3.11.11_requirements.txt** - Manjaro Linux specific

### Optional Dependencies
```bash
# For best Amharic G2P quality
pip install transphone

# Alternative Amharic G2P
pip install epitran
```

## Performance Considerations

### Training Speed
- **Batch size = 2:** ~5-10 minutes per epoch (10 min audio)
- **Batch size = 4:** ~3-5 minutes per epoch (requires 12GB+ VRAM)
- **Gradient accumulation:** Simulate larger batches on smaller GPUs

### Inference Speed
- **GPU (CUDA):** 0.5-2 seconds for 10-word sentence
- **GPU (MPS/Apple Silicon):** 2-5 seconds
- **CPU:** 10-30 seconds

### Memory Optimization
- **Clear GPU cache:** `torch.cuda.empty_cache()` between operations
- **Limit audio length:** max_wav_length prevents OOM
- **Checkpoint only best model:** save_n_checkpoints=1

## Known Issues & Workarounds

### Japanese Training
**Issue:** Training hangs with default num_workers
**Solution:** Set `num_workers=0` in GPTTrainerConfig

### FFmpeg Not Found
**Issue:** Audio conversion fails
**Solution:** Install FFmpeg and add to PATH
- Windows: `winget install FFmpeg` or download from ffmpeg.org
- Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

### CUDA Out of Memory
**Issue:** Training crashes with OOM
**Solutions:**
- Reduce batch_size (try 2 or 1)
- Reduce max_wav_length
- Use gradient accumulation
- Close other GPU applications

### Gradio Interface Slow
**Issue:** UI freezes during processing
**Solution:** Processing happens in background, check console for progress

### Model Download Fails
**Issue:** XTTS base model download interrupted
**Solution:** Delete incomplete files in `base_models/{version}/` and retry

## External Documentation

- **Coqui TTS:** https://github.com/coqui-ai/TTS
- **Faster Whisper:** https://github.com/SYSTRAN/faster-whisper
- **Gradio:** https://www.gradio.app/docs
- **PyTorch:** https://pytorch.org/docs
- **Transphone:** https://github.com/xinjli/transphone
- **Epitran:** https://github.com/dmort27/epitran
