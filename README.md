# xtts-finetune-webui

This webui is a slightly modified copy of the [official webui](https://github.com/coqui-ai/TTS/pull/3296) for finetune xtts.

If you are looking for an option for normal XTTS use look here [https://github.com/daswer123/xtts-webui](https://github.com/daswer123/xtts-webui)

## TODO
- [ ] Add the ability to use via console 

## Key features:

### Data processing

1. Updated faster-whisper to 0.10.0 with the ability to select a larger-v3 model.
2. Changed output folder to output folder inside the main folder.
3. If there is already a dataset in the output folder and you want to add new data, you can do so by simply adding new audio, what was there will not be processed again and the new data will be automatically added
4. Turn on VAD filter
5. After the dataset is created, a file is created that specifies the language of the dataset. This file is read before training so that the language always matches. It is convenient when you restart the interface
6. **NEW: SRT + Media File Processing** - Upload subtitle files (SRT/VTT) with corresponding audio/video files for precise timestamp-based dataset creation
7. **NEW: YouTube Video Download** - Download YouTube videos with automatic transcript extraction using yt-dlp
8. **NEW: RMS-Based Audio Slicing** - Intelligently segment long audio files based on silence detection with configurable parameters

### Fine-tuning XTTS Encoder

1. Added the ability to select the base model for XTTS, as well as when you re-training does not need to download the model again.
2. Added ability to select custom model as base model during training, which will allow finetune already finetune model.
3. Added possibility to get optimized version of the model for 1 click ( step 2.5, put optimized version in output folder).
4. You can choose whether to delete training folders after you have optimized the model
5. When you optimize the model, the example reference audio is moved to the output folder
6. Checking for correctness of the specified language and dataset language

### Inference

1. Added possibility to customize infer settings during model checking.

### Other

1. If you accidentally restart the interface during one of the steps, you can load data to additional buttons
2. Removed the display of logs as it was causing problems when restarted
3. The finished result is copied to the ready folder, these are fully finished files, you can move them anywhere and use them as a standard model
4. Added support for finetune Japanese

## 🎬 Advanced Dataset Processing Features

Version 2.0 introduces powerful dataset creation tools for working with various media sources!

### 📝 SRT + Media File Processing

Process subtitle files synchronized with audio or video for perfectly aligned training data.

**Supported formats:**
- Subtitle files: SRT, VTT, JSON transcripts
- Media files: MP4, MKV, AVI, WAV, MP3, FLAC

**Usage (Web UI):**
1. Navigate to Tab 1 (Data Processing)
2. Upload your SRT/VTT file
3. Upload corresponding media file (video or audio)
4. Select language and click Process
5. System extracts audio segments aligned with subtitle timestamps

**Usage (Headless):**
```bash
python headlessXttsTrain.py \
  --srt_file "path/to/subtitles.srt" \
  --media_file "path/to/video.mp4" \
  --lang en \
  --epochs 10
```

**Benefits:**
- Perfect text-audio alignment from pre-existing subtitles
- No need for automatic transcription (faster processing)
- Ideal for dubbing projects, movie/TV show voice cloning
- Supports multiple speakers if SRT includes speaker labels

### 🎵 Background Music Removal (NEW!)

Remove background music from audio files using state-of-the-art AI (Demucs by Meta) to extract clean vocals for TTS training.

**Features:**
- AI-powered vocal separation (Demucs model)
- Multiple quality presets (fast, balanced, best)
- GPU/CPU support with automatic detection
- Integrated with YouTube downloader and batch processing
- Preserves audio quality while removing music

**Benefits for TTS:**
- ✅ Improves model training quality
- ✅ Produces clearer synthetic voices
- ✅ Reduces unwanted artifacts
- ✅ Results in more natural-sounding output

**Installation:**
```bash
pip install demucs
```

**Usage (Python API):**
```python
from utils.audio_background_remover import remove_background_music

# Remove background music
clean_audio = remove_background_music(
    input_audio="podcast_with_music.wav",
    output_audio="podcast_clean.wav",
    quality="balanced"  # Options: fast, balanced, best
)
```

**Usage (YouTube Integration):**
```python
from utils import youtube_downloader

audio_path, srt_path, info = youtube_downloader.download_and_process_youtube(
    url="https://youtube.com/watch?v=VIDEO_ID",
    output_dir="./downloads",
    language="en",
    remove_background_music=True,  # Enable background removal
    background_removal_quality="balanced"
)
```

**See `.warp/BACKGROUND_MUSIC_REMOVAL.md` for complete documentation.**

### 📹 YouTube Video Download with Transcripts

Automatically download YouTube videos and extract available transcripts/subtitles.

**Features:**
- Auto-download video/audio using yt-dlp
- Extract auto-generated or manual subtitles
- Multi-language support with fallback options
- Automatic format detection and conversion
- Cookie support for age-restricted content

**Usage (Web UI):**
1. Navigate to Tab 1 (Data Processing)
2. Enter YouTube URL
3. Select language (for transcript extraction)
4. Click Download & Process
5. System downloads, extracts transcripts, and creates dataset

**Usage (Headless):**
```bash
python headlessXttsTrain.py \
  --youtube_url "https://youtube.com/watch?v=VIDEO_ID" \
  --lang en \
  --epochs 10
```

**Notes:**
- Requires active internet connection
- Respects YouTube rate limits and Terms of Service
- For personal/educational use only
- Works best with videos that have high-quality subtitles/transcripts

### ✂️ RMS-Based Audio Slicing

Intelligently segment long audio files based on silence detection without manual editing.

**Features:**
- RMS (Root Mean Square) based silence detection
- Configurable threshold, minimum length, and padding
- Creates content-aware segments (not just fixed-length chunks)
- Optional automatic transcription of segments
- Preserves audio quality during slicing

**Usage (Python API):**
```python
from utils.audio_slicer import AudioSlicer

slicer = AudioSlicer(
    threshold_db=-40,      # Silence threshold
    min_length=5.0,        # Minimum segment length (seconds)
    min_interval=0.3,      # Minimum silence interval
    hop_size=10,           # Analysis hop size (ms)
    max_sil_kept=0.5       # Silence padding (seconds)
)

segments = slicer.slice_audio(
    audio_path="long_recording.wav",
    output_dir="output/segments/"
)
```

**Parameters:**
- `threshold_db`: Volume threshold for silence detection (default: -40 dB)
- `min_length`: Minimum segment duration (default: 5 seconds)
- `min_interval`: Minimum silence duration to split (default: 0.3 seconds)
- `hop_size`: Analysis window step size (default: 10 ms)
- `max_sil_kept`: Silence padding at segment boundaries (default: 0.5 seconds)

**Use Cases:**
- Podcast/audiobook processing
- Long interview or lecture recordings
- Radio show archival processing
- Any audio where manual segmentation is impractical

### 📦 Dependencies for Advanced Features

```bash
# SRT processing
pip install pysrt

# YouTube downloading
pip install yt-dlp

# Audio slicing
pip install soundfile

# FFmpeg (required for all advanced features)
# Windows: Download from ffmpeg.org and add to PATH
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg
```

All dependencies are included in `requirements.txt` except FFmpeg, which must be installed separately.

---

## 🇪🇹 Amharic TTS Support

This project includes comprehensive support for **Amharic language** (Ethiopian) with advanced Grapheme-to-Phoneme (G2P) conversion!

### Features

✅ **Multiple G2P Backends** with automatic fallback:
- **Transphone** (primary) - Zero-shot G2P for 7500+ languages including Amharic
- **Epitran** (fallback) - Rule-based G2P with Ethiopic script support  
- **Custom Rule-Based** (offline) - Comprehensive Amharic phoneme mapping, always available

✅ **Ethiopic Script Support**:
- Full support for 340+ Ethiopic characters (U+1200-U+137F)
- Character variant normalization (ሥ→ስ, ዕ→እ, etc.)
- Proper handling of Amharic punctuation (።፣፤፥)

✅ **Phonological Processing**:
- Epenthetic vowel insertion (kɨt → kɨtɨ)
- Gemination handling (doubled consonants)
- Labiovelar consonants (kʷ, gʷ, qʷ)

✅ **Amharic-Specific Preprocessing**:
- Number-to-word expansion in Amharic (123 → አንድ መቶ ሃያ ሶስት)
- Text normalization and cleaning
- Automatic language detection

### Quick Start with Amharic

#### Web Interface
1. Launch the webui: `python xtts_demo.py`
2. In **Tab 1 (Data processing)**:
   - Select `amh` from the **Dataset Language** dropdown
   - Optionally enable **Amharic G2P preprocessing** in the accordion
   - Choose your preferred G2P backend (transphone/epitran/rule_based)
   - Upload Amharic audio files

3. In **Tab 2 (Fine-tuning)**:
   - Enable **Amharic G2P for training** if you want phoneme-based training
   - Select G2P backend for training
   - Configure other training parameters

4. Train and test your Amharic TTS model!

#### Headless Training
```bash
# Basic Amharic training
python headlessXttsTrain.py --input_audio amharic_speaker.wav --lang amh --epochs 10

# With G2P preprocessing (requires transphone or epitran)
python headlessXttsTrain.py --input_audio amharic_speaker.wav --lang amh --epochs 10 --use_g2p
```

### Installing G2P Backends (Optional)

For best quality, install the Transphone backend:
```bash
pip install transphone
```

Or Epitran as an alternative:
```bash
pip install epitran
```

**Note**: The rule-based backend is always available and requires no additional installation!

### Amharic Module Structure

```
amharic_tts/
├── g2p/
│   ├── amharic_g2p.py              # Basic G2P converter
│   └── amharic_g2p_enhanced.py     # Enhanced with multiple backends
├── tokenizer/
│   ├── hybrid_tokenizer.py         # G2P + BPE hybrid tokenizer
│   └── xtts_tokenizer_wrapper.py   # XTTS-compatible wrapper
├── preprocessing/
│   ├── text_normalizer.py          # Character normalization
│   └── number_expander.py          # Amharic number expansion
└── config/
    └── amharic_config.py            # Configuration and phoneme inventory
```

### Examples

**Example 1: Text Normalization**
```python
from amharic_tts.preprocessing.text_normalizer import AmharicTextNormalizer

normalizer = AmharicTextNormalizer()
text = normalizer.normalize("ሀሎ ዓለም")  # → "ሃሎ አለም"
```

**Example 2: G2P Conversion**
```python
from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P

g2p = AmharicG2P(backend='transphone')  # or 'epitran', 'rule-based'
phonemes = g2p.convert("ሰላም ኢትዮጵያ")  # → IPA phonemes
```

**Example 3: Number Expansion**
```python
from amharic_tts.preprocessing.number_expander import AmharicNumberExpander

expander = AmharicNumberExpander()
text = expander.expand_number("2024")  # → "ሁለት ሺህ ሃያ አራት"
```

### Troubleshooting

**Q: What if I don't have Transphone or Epitran installed?**  
A: The system automatically falls back to the rule-based backend, which works offline and requires no additional dependencies.

**Q: Should I use G2P preprocessing for Amharic?**  
A: For best results with Amharic, enabling G2P is recommended. It converts Ethiopic script to IPA phonemes, which improves pronunciation accuracy.

**Q: Which G2P backend should I choose?**  
A: 
- **Transphone** - Best accuracy, supports rare words
- **Epitran** - Fast, rule-based, good for common words
- **Rule-based** - Always available, no installation needed, good baseline

**Q: Can I fine-tune on existing Amharic models?**  
A: Yes! Use the custom model option in Tab 2 to continue training from a previously fine-tuned Amharic model.

### Documentation

For detailed information about the Amharic implementation:
- See `docs/G2P_BACKENDS_EXPLAINED.md` for G2P backend details
- See `amharic_tts/g2p/README.md` for phonological rules
- See `tests/test_amharic_integration.py` for usage examples

### Advanced Dataset Processing

Inspired by and integrated with techniques from the [dataset-maker](https://github.com/JarodMica/dataset-maker) project:
- SRT subtitle synchronization with media files
- YouTube content acquisition with transcript extraction
- RMS-based intelligent audio slicing
- Multi-format support for transcripts (SRT, VTT, JSON)

### Credits

Amharic TTS support developed with research from:
- Transphone: [github.com/xinjli/transphone](https://github.com/xinjli/transphone)
- Epitran: [github.com/dmort27/epitran](https://github.com/dmort27/epitran)
- Ethiopian script phonology research and linguistic analysis

Advanced dataset processing inspired by:
- Dataset-Maker: [github.com/JarodMica/dataset-maker](https://github.com/JarodMica/dataset-maker)

## Changes in webui

### 1 - Data processing

![image](https://github.com/daswer123/xtts-finetune-webui/assets/22278673/8f09b829-098b-48f5-9668-832e7319403b)

### 2 - Fine-tuning XTTS Encoder

![image](https://github.com/daswer123/xtts-finetune-webui/assets/22278673/897540d9-3a6b-463c-abb8-261c289cc929)

### 3 - Inference

![image](https://github.com/daswer123/xtts-finetune-webui/assets/22278673/aa05bcd4-8642-4de4-8f2f-bc0f5571af63)

## Run Remotly
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow?style=flat&logo=huggingface)](https://huggingface.co/spaces/drewThomasson/xtts-finetune-webui-gpu) [![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=flat&logo=kaggle&logoColor=white)](notebook/kaggle-xtts-finetune-webui-gradio-gui.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrewThomasson/ebook2audiobook/blob/v25/Notebooks/finetune/xtts/colab_xtts_finetune_webui.ipynb)


## 🐳 Run in Docker 
```docker
docker run -it --gpus all --pull always -p 7860:7860 --platform=linux/amd64 athomasson2/fine_tune_xtts:huggingface python app.py
```
## Run Headless

```bash
# Basic audio processing
python headlessXttsTrain.py --input_audio speaker.wav --lang en --epochs 10

# Process SRT + media file
python headlessXttsTrain.py --srt_file subtitles.srt --media_file video.mp4 --lang en --epochs 10

# Download and process YouTube video
python headlessXttsTrain.py --youtube_url "https://youtube.com/watch?v=VIDEO_ID" --lang en --epochs 10

# See all parameters
python headlessXttsTrain.py --help
```

## Install

1. Make sure you have `Cuda` installed
2. `git clone https://github.com/daswer123/xtts-finetune-webui`
3. `cd xtts-finetune-webui`
4. `pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118`
5. `pip install -r requirements.txt`

### If you're using Windows

1. First start `install.bat`
2. To start the server start `start.bat`
3. Go to the local address `127.0.0.1:5003`

### On Linux

1. Run `bash install.sh`
2. To start the server start `start.sh`
3. Go to the local address `127.0.0.1:5003`

### On Apple Silicon Mac (python 3.10 env)
1. ``` pip install --no-deps -r apple_silicon_requirements.txt ```
2. To start the server `python xtts_demo.py`
3. Go to the local address `127.0.0.1:5003`

### On Manjaro x86 (python 3.11.11 env)
1. ``` pip install --no-deps -r ManjaroX86Python3.11.11_requirements.txt ```
2. To start the server `python xtts_demo.py`
3. Go to the local address `127.0.0.1:5003`

~                                            
