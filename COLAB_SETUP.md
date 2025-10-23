# Google Colab Setup Guide

## 🚀 Quick Start (Direct Method - No Notebook)

This guide shows how to run the XTTS fine-tuning WebUI directly in Google Colab without using a notebook interface.

### Prerequisites

- Google account
- Access to Google Colab (free tier works!)
- GitHub repository URL

---

## 📋 Step-by-Step Setup

### 1. Open Google Colab

Go to [https://colab.research.google.com/](https://colab.research.google.com/)

### 2. Create New Notebook

Click **"New Notebook"** or **"File → New Notebook"**

### 3. Enable GPU (Optional but Recommended)

```
Runtime → Change runtime type → Hardware accelerator → GPU (T4)
```

### 4. Clone Repository

Run in a Colab cell:

```bash
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
%cd YOUR_REPO_NAME
```

Replace with your actual repository path:

```bash
!git clone https://github.com/Diakonrobel/Amharic_XTTS-V2_TTS.git
%cd Amharic_XTTS-V2_TTS
```

### 5. Install System Dependencies

```bash
!apt-get update -qq
!apt-get install -y -qq ffmpeg espeak-ng
```

### 6. Install Python Dependencies

**Use the Colab-specific requirements file:**

```bash
!pip install -q -r requirements-colab.txt
```

This installs only the necessary packages without conflicting with Colab's pre-installed PyTorch.

### 7. Download Spacy Language Model

```bash
!python -m spacy download en_core_web_sm
```

### 8. Optional: Download Japanese Model (if needed)

```bash
!python -m unidic download
```

---

## 🎯 Running the WebUI

### Option 1: Launch Gradio Interface

```python
import xtts_demo

# Launch the WebUI
demo = xtts_demo.create_demo()
demo.launch(share=True)  # share=True creates public link
```

You'll get a public URL like: `https://xxxxx.gradio.live`

### Option 2: Command Line Training

```bash
!python headlessXttsTrain.py \
  --train_csv /content/datasets/train.csv \
  --eval_csv /content/datasets/eval.csv \
  --language am \
  --num_epochs 50 \
  --batch_size 4
```

---

## 📊 YouTube Dataset Processing

### Download and Process YouTube Videos

```python
from utils import youtube_downloader, batch_processor, srt_processor

# List of YouTube URLs
urls = [
    "https://www.youtube.com/watch?v=VIDEO_ID_1",
    "https://www.youtube.com/watch?v=VIDEO_ID_2",
    "https://www.youtube.com/watch?v=VIDEO_ID_3",
]

# Process batch
train_csv, eval_csv, infos = batch_processor.process_youtube_batch(
    urls=urls,
    transcript_lang="am",  # or "en", etc.
    out_path="/content/datasets",
    youtube_downloader=youtube_downloader,
    srt_processor=srt_processor,
    cookies_from_browser=None,  # Colab doesn't have browser cookies
)

print(f"✓ Created training dataset: {train_csv}")
print(f"✓ Created evaluation dataset: {eval_csv}")
```

---

## 🔧 Colab-Specific Considerations

### Memory Management

Colab free tier has ~12GB RAM. If you run out of memory:

```python
# Clear memory
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

### Session Timeout

Colab sessions timeout after 12 hours (free tier). Save your work frequently:

```bash
# Save checkpoints to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy checkpoints
!cp -r /content/YOUR_REPO/checkpoints /content/drive/MyDrive/xtts_checkpoints
```

### File Upload/Download

```python
from google.colab import files

# Upload files
uploaded = files.upload()

# Download files
files.download('/content/YOUR_REPO/output.zip')
```

---

## 📦 What's Different in Colab?

| Feature | Regular Setup | Colab Setup |
|---------|--------------|-------------|
| PyTorch | Install from requirements | Pre-installed (skip) |
| CUDA | Manual setup | Pre-configured |
| FFmpeg | Manual install | Install via apt-get |
| Gradio | Same | Same (works with public URLs) |
| Storage | Local disk | Colab VM (12-100GB) |
| GPU | Your hardware | T4/V100/A100 |

---

## 🚨 Common Issues & Solutions

### Issue 1: "Module not found" errors

**Solution:** Make sure you used `requirements-colab.txt`:

```bash
!pip install -r requirements-colab.txt
```

### Issue 2: PyTorch version conflicts

**Solution:** Do NOT reinstall PyTorch. Use `requirements-colab.txt` which excludes it.

### Issue 3: "CUDA out of memory"

**Solutions:**
1. Reduce batch size
2. Clear cache: `torch.cuda.empty_cache()`
3. Restart runtime: `Runtime → Restart runtime`

### Issue 4: Session disconnected

**Solution:** Colab disconnects after inactivity. Use this to keep alive:

```javascript
// Run in browser console (F12)
function ClickConnect(){
    console.log("Clicking connect button"); 
    document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click();
}
setInterval(ClickConnect, 60000);
```

### Issue 5: YouTube downloads fail

**Solution:** Colab can't use browser cookies. Options:
1. Upload pre-exported cookies file
2. Use proxy (if available)
3. Pre-download videos locally and upload to Colab

---

## 💾 Persistent Storage

### Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Use Drive for storage
%cd /content/drive/MyDrive
!git clone YOUR_REPO
```

### Save Outputs

```bash
# Copy trained models to Drive
!cp -r /content/YOUR_REPO/run/training /content/drive/MyDrive/xtts_models
```

---

## 🎬 Complete Setup Script

Copy-paste this into a Colab cell for automatic setup:

```bash
%%bash
set -e

echo "🚀 Setting up XTTS Fine-tune WebUI on Colab..."

# Install system dependencies
apt-get update -qq
apt-get install -y -qq ffmpeg espeak-ng

# Clone repository
cd /content
git clone https://github.com/Diakonrobel/Amharic_XTTS-V2_TTS.git
cd Amharic_XTTS-V2_TTS

# Install Python dependencies (Colab-specific)
pip install -q -r requirements-colab.txt

# Download spacy model
python -m spacy download en_core_web_sm

echo "✅ Setup complete! Ready to use."
```

Then launch the WebUI:

```python
import sys
sys.path.append('/content/Amharic_XTTS-V2_TTS')

import xtts_demo
demo = xtts_demo.create_demo()
demo.launch(share=True)
```

---

## 📝 Batch Training Example

```python
# Complete workflow on Colab

# 1. Process YouTube videos
from utils import youtube_downloader, batch_processor, srt_processor

urls = ["https://youtube.com/watch?v=VIDEO_ID"]
train_csv, eval_csv, _ = batch_processor.process_youtube_batch(
    urls=urls,
    transcript_lang="am",
    out_path="/content/datasets",
    youtube_downloader=youtube_downloader,
    srt_processor=srt_processor,
)

# 2. Launch training
!python headlessXttsTrain.py \
  --train_csv {train_csv} \
  --eval_csv {eval_csv} \
  --language am \
  --num_epochs 50 \
  --batch_size 4 \
  --output_path /content/drive/MyDrive/xtts_output

print("✅ Training complete!")
```

---

## 🔍 Verify Installation

```python
# Test imports
import torch
import gradio
import TTS
from faster_whisper import WhisperModel

print("✅ All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

Expected output:
```
✅ All imports successful!
PyTorch version: 2.1.0+cu121
CUDA available: True
GPU: Tesla T4
```

---

## 🎯 Pro Tips

1. **Use GPU runtime** - Much faster for training/inference
2. **Save frequently** - Sessions can disconnect
3. **Use Google Drive** - Persist data across sessions
4. **Monitor resources** - `Runtime → View resources`
5. **Upgrade if needed** - Colab Pro for longer sessions & better GPUs

---

## 📚 Additional Resources

- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Colab Pro Features](https://colab.research.google.com/signup)
- [Project README](README.md)
- [YouTube Download Fix](YOUTUBE_FIX_2025.md)

---

## 🆘 Need Help?

1. Check console output for error messages
2. Verify all dependencies installed: `!pip list`
3. Check GPU availability: `torch.cuda.is_available()`
4. Restart runtime if needed: `Runtime → Restart runtime`

---

**Last Updated:** January 2025  
**Tested on:** Google Colab Free Tier, Python 3.10, PyTorch 2.1.0+cu121
