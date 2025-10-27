# 🎙️ XTTS Dataset Creator

Professional audio dataset creation tool with Gradio WebUI for XTTS voice cloning.

## ✨ Features

### Multiple Input Sources
- 🎬 **YouTube Videos** - Download and process automatically
- 📁 **Audio Files** - Upload WAV, MP3, FLAC files
- 🎤 **Microphone Recording** - Record directly in browser

### Advanced Processing
- 🤖 **Automatic Transcription** - Faster Whisper with GPU support
- ✂️ **Smart Segmentation** - VAD-based audio splitting
- 🎯 **Quality Filtering** - SNR, clipping, and energy analysis
- 📊 **Real-time Statistics** - Track dataset metrics

### Export Options
- CSV format (Pandas-compatible)
- JSON format (structured data)
- LJSpeech format (with audio files)
- metadata.txt (training-ready)

## 🚀 Quick Start

### Local Installation

```bash
# Clone repository
git clone https://github.com/Diakonrobel/Amharic_XTTS-V2_TTS.git
cd Amharic_XTTS-V2_TTS/dataset_creator

# Install dependencies
pip install -r requirements.txt

# Launch app
python app.py --port 7861 --share
```

### Google Colab

1. Open `colab_dataset_creator.ipynb` in Google Colab
2. Run all cells
3. Click the public URL to access the interface

### Kaggle

Same as Colab - use the notebook in Kaggle environment.

## 📖 Usage Guide

### 1. Create a Project

```
Project Setup Tab → Enter name → Select language → Create
```

### 2. Add Data

Choose one of three methods:

**YouTube:**
- Paste video URL
- Adjust min/max duration (1-15s recommended)
- Set quality threshold (0.7 recommended)
- Click Process

**File Upload:**
- Upload audio files
- Configure segmentation options
- Click Process

**Recording:**
- Click microphone to record
- Optionally provide transcription
- Click Add Recording

### 3. Review & Export

```
Dataset Overview → Check statistics → Select format → Export
```

## ⚙️ Configuration

### Segment Duration
- **Minimum:** 0.5-5.0 seconds (default: 1.0s)
- **Maximum:** 5.0-30.0 seconds (default: 15.0s)

### Quality Threshold
- **Low (0.3-0.5):** More segments, lower quality
- **Medium (0.6-0.7):** Balanced ✅
- **High (0.8-1.0):** Fewer segments, higher quality

## 💡 Best Practices

### Audio Quality
✅ Clear voice, minimal noise
✅ Consistent volume
✅ 22050 Hz or higher sample rate
❌ Avoid music, multiple speakers, echoes

### Dataset Size
- **Testing:** 5-10 minutes
- **Good:** 30-60 minutes
- **Excellent:** 2-4 hours
- **Professional:** 10+ hours

### Language Codes
- English: `en`
- Spanish: `es`
- French: `fr`
- German: `de`
- Amharic: `am` or `amh`
- See UI for full list

## 📊 Export Formats

### CSV
```csv
audio_file,text,speaker_name,duration
wavs/segment_000001.wav,"Hello world",speaker,1.5
```

### JSON
```json
[
  {
    "audio_file": "wavs/segment_000001.wav",
    "text": "Hello world",
    "speaker_name": "speaker",
    "duration": 1.5
  }
]
```

### LJSpeech
```
LJSpeech/
├── wavs/
│   ├── segment_000001.wav
│   └── ...
├── metadata.csv
└── README.txt
```

## 🔧 Troubleshooting

### YouTube Download Fails
```bash
# Update yt-dlp
pip install -U yt-dlp
```

### Low Quality Segments
- Lower quality threshold
- Check source audio
- Adjust duration limits

### Transcription Errors
- Verify correct language
- Ensure clear audio
- Try shorter segments

### Out of Memory
- Process fewer files
- Use shorter segments
- Restart application

## 📁 Project Structure

```
dataset_creator/
├── app.py                          # Main Gradio app
├── dataset_processor.py            # Core processing logic
├── audio_recorder.py               # Microphone recording
├── utils_dataset.py                # Statistics & export
├── requirements.txt                # Dependencies
├── colab_dataset_creator.ipynb    # Colab notebook
└── README.md                       # This file
```

## 🎓 Advanced Usage

### Custom Whisper Model

```python
from dataset_processor import DatasetProcessor

processor = DatasetProcessor(
    output_dir="./my_datasets",
    whisper_model="medium"  # or "large-v3", "small"
)
```

### Programmatic API

```python
from dataset_processor import DatasetProcessor

# Initialize
processor = DatasetProcessor()
processor.set_project("./my_project")

# Process YouTube
result = processor.process_youtube_url(
    url="https://youtube.com/...",
    language="en",
    speaker_name="speaker",
    min_duration=1.0,
    max_duration=15.0
)

# Process audio file
result = processor.process_audio_file(
    audio_path="audio.wav",
    language="en",
    speaker_name="speaker"
)
```

### Custom Segmentation

```python
# Adjust VAD parameters
from utils import audio_slicer

slicer = audio_slicer.Slicer(
    sr=22050,
    threshold=-35.0,  # More sensitive
    min_length=3000,  # 3 seconds
    max_sil_kept=1000  # Keep more silence
)
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

Same as parent repository.

## 🎉 Credits

- **XTTS v2:** Coqui AI
- **Faster Whisper:** SYSTRAN
- **Gradio:** Gradio Team
- **yt-dlp:** yt-dlp Team

## 📞 Support

- GitHub Issues: [Report bugs](https://github.com/Diakonrobel/Amharic_XTTS-V2_TTS/issues)
- Documentation: See parent README
- Colab Notebook: Includes full guide

---

**⭐ Star the repo if this helps you!**

**Status:** ✅ Production Ready | **Optimized for:** Colab/Kaggle/Local
