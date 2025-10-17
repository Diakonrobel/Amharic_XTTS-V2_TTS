# Audio Padding Control Guide

## 🎯 Problem Fixed

Audio segments were getting cut off at the beginning and/or end because the padding was too small (0.2s). This caused misalignment between audio and transcribed text.

## ✅ Solution

A new **Audio Padding** slider has been added to the Gradio WebUI in the SRT Processing tab!

## 📍 How to Use (Gradio WebUI)

### 1. **Launch the WebUI**
```bash
python xtts_demo.py
```

### 2. **Navigate to Processing Tab**
- Go to: **"Step 1 - Dataset Creation"** → **"📝 SRT Processing"** OR **"📹 YouTube Processing"** tab
- **Both tabs now have the same Audio Padding control!**

### 3. **Configure Audio Padding**
You'll now see a new section: **"⚙️ Segmentation Settings"**

**Audio Padding Slider:**
- **Range**: 0.1s to 1.0s
- **Default**: 0.4s (recommended)
- **Step**: 0.05s

### 4. **Choose Your Padding Level**

| Padding | Description | Use Case |
|---------|-------------|----------|
| **0.1-0.3s** | Minimal padding | ⚠️ Risk of cutoffs - not recommended |
| **0.4-0.5s** | **Recommended** | ✅ Safe, prevents cutoffs |
| **0.6-1.0s** | Maximum safety | Extra silence but guarantees full capture |

### 5. **Process Your Files**
- Upload SRT and media files as usual
- Adjust the padding slider
- Click **"▶️ Process SRT + Media"**

## 🔍 What Changed

### Default Values Updated:
- ✅ **SRT Processor**: `buffer = 0.2s → 0.4s`
- ✅ **VAD Slicer**: `speech_pad_ms = 30ms → 150ms`
- ✅ **UI Control**: New slider with 0.4s default

### YouTube Processing:
- ✅ YouTube tab now has the **same slider control** as SRT Processing
- Default: **0.4s** (same safe default)
- Fully adjustable per your needs

## 💡 Recommendations

### For Amharic (or languages with ejective consonants):
```
Audio Padding: 0.5-0.6s
```
Ejectives need extra space to capture the full burst.

### For English/Most Languages:
```
Audio Padding: 0.4s (default)
```
Balanced padding that works for 95% of cases.

### For Fast-paced Speech:
```
Audio Padding: 0.6-0.8s
```
Rapid speech benefits from more generous padding.

## 🐛 Troubleshooting

### Still getting cutoffs?
1. Increase padding to **0.6-0.8s**
2. Check your SRT timestamps are accurate
3. Use the enhanced segmentation module (see below)

### Too much silence?
1. Reduce padding to **0.3-0.4s**
2. Enable VAD Enhancement (optional)

## 🚀 Advanced: Enhanced Segmentation Module

For existing datasets with cutoff issues, use the Python API:

```python
from utils.enhanced_segmentation import fix_existing_dataset

# Fix your existing dataset
train_csv, eval_csv = fix_existing_dataset(
    dataset_dir="finetune_models/dataset",
    audio_source="original_audio.wav",
    output_dir="finetune_models/dataset_fixed",
    leading_pad_ms=300,    # 300ms before speech
    trailing_pad_ms=400,   # 400ms after speech
    use_vad_trimming=False # Keep False for safety
)
```

## 📝 UI Location

The Audio Padding control appears in **BOTH tabs**:

### 📝 SRT Processing Tab:
```
├── Upload SRT Files
├── Upload Media Files
├── Batch Mode ☑
├── VAD Enhancement ☐
├── ⚙️ Segmentation Settings ← Audio Padding slider here!
│   ├── Audio Padding: [======●==] 0.40s
│   └── 💡 Info text about padding levels
└── ⚙️ VAD Settings (Advanced)
```

### 📹 YouTube Processing Tab:
```
├── YouTube URL input
├── Transcript Language dropdown
├── Batch Mode ☑
├── Incremental Mode ☐
├── ⚙️ Segmentation Settings ← Audio Padding slider here!
│   ├── Audio Padding: [======●==] 0.40s
│   └── 💡 Info text about padding levels
├── VAD Enhancement ☐
└── ⚙️ VAD Settings (Advanced)
```

## ✨ Benefits

✅ **No more cutoffs** - Full speech captured
✅ **Better training** - Complete audio segments
✅ **Flexible control** - Adjust per your needs
✅ **Easy to use** - Just drag a slider!

## 📚 Related Files

- `xtts_demo.py` - Gradio UI with new slider
- `utils/srt_processor.py` - Updated buffer default
- `utils/vad_slicer.py` - Updated speech padding
- `utils/enhanced_segmentation.py` - Advanced re-extraction tool

---

**Last Updated**: 2025-10-17
**Version**: 1.0
