# XTTS Fine-Tuning Dataset Format

This document describes the dataset format used by the XTTS fine-tuning WebUI.

## Directory Structure

```
finetune_models/
└── dataset/
    ├── wavs/                    # Audio files directory
    │   ├── audio_000000.wav
    │   ├── audio_000001.wav
    │   ├── audio_000002.wav
    │   └── ...
    ├── metadata_train.csv       # Training metadata
    ├── metadata_eval.csv        # Evaluation metadata
    └── lang.txt                 # Language identifier
```

## File Descriptions

### 1. Audio Files (`wavs/` directory)

**Format Requirements:**
- **Format**: WAV (PCM 16-bit)
- **Sample Rate**: 22050 Hz
- **Channels**: Mono (1 channel)
- **Duration**: 0.5s to 15s per segment (recommended: 3-10s)
- **Naming**: Sequential with zero-padding (e.g., `audio_000000.wav`)

**Quality Guidelines:**
- Clear speech without background noise
- No music or sound effects overlapping speech
- Consistent volume levels
- No clipping or distortion

### 2. Metadata Files (CSV format)

**File Names:**
- `metadata_train.csv` - Training dataset (typically 85% of data)
- `metadata_eval.csv` - Evaluation dataset (typically 15% of data)

**Format Specification:**
```csv
audio_file|text|speaker_name
wavs/audio_000000.wav|This is the transcribed text.|speaker
wavs/audio_000001.wav|Another sentence here.|speaker
```

**Column Details:**

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `audio_file` | string | Yes | Relative path to audio file (e.g., `wavs/audio_000000.wav`) |
| `text` | string | Yes | Transcribed text corresponding to the audio |
| `speaker_name` | string | Yes | Speaker identifier (can be any string, e.g., "speaker", "john", "narrator") |

**Important Notes:**
- **Delimiter**: Pipe symbol `|` (not comma)
- **Encoding**: UTF-8 (supports all languages including Amharic, Arabic, Chinese, etc.)
- **Header**: First line must be `audio_file|text|speaker_name`
- **Text Format**: 
  - Clean transcription (no timestamps, no speaker labels)
  - Proper punctuation
  - Can include Unicode characters (አማርኛ, العربية, 中文, etc.)

### 3. Language File (`lang.txt`)

**Format:**
```
am
```

**Purpose:** Specifies the language of the dataset using ISO 639-1 codes

**Common Language Codes:**
- `en` - English
- `am` - Amharic (አማርኛ)
- `om` - Oromo
- `ti` - Tigrinya (ትግርኛ)
- `ar` - Arabic
- `es` - Spanish
- `fr` - French
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean

Full list: [ISO 639-1 Codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)

## Example Dataset

### Example 1: Amharic Dataset

**Directory Structure:**
```
dataset/
├── wavs/
│   ├── audiobook_000000.wav
│   ├── audiobook_000001.wav
│   └── audiobook_000002.wav
├── metadata_train.csv
├── metadata_eval.csv
└── lang.txt
```

**metadata_train.csv:**
```csv
audio_file|text|speaker_name
wavs/audiobook_000000.wav|አንድ ቀን የነበረ ንጉስ ከሀገሩ ሰዎች ጋር ይኖር ነበር።|narrator
wavs/audiobook_000001.wav|ንጉሱ በጣም ደግ እና ጠቢብ ነበር።|narrator
wavs/audiobook_000002.wav|ሁሉም ሰዎች ንጉሱን ያከብሩት ነበር።|narrator
```

**metadata_eval.csv:**
```csv
audio_file|text|speaker_name
wavs/audiobook_000003.wav|በአንድ ቀን ንጉሱ በጫካ ውስط ሄደ።|narrator
```

**lang.txt:**
```
am
```

### Example 2: English Dataset

**metadata_train.csv:**
```csv
audio_file|text|speaker_name
wavs/sample_000000.wav|Hello, how are you today?|john
wavs/sample_000001.wav|I am doing great, thank you for asking.|john
wavs/sample_000002.wav|The weather is beautiful outside.|john
```

**lang.txt:**
```
en
```

## Creating Datasets

### Method 1: Upload Audio Files (Manual)
1. Prepare audio files (WAV, MP3, or FLAC)
2. Upload via Tab 1 in the WebUI
3. System will:
   - Convert to 22050 Hz mono WAV
   - Use Whisper to transcribe
   - Generate metadata CSV files automatically

### Method 2: SRT + Media File (Timestamp-based)
1. Prepare subtitle file (SRT or VTT format)
2. Prepare corresponding audio/video file
3. Upload both via "📝 SRT + Media File Processing" accordion
4. System will:
   - Extract audio segments based on SRT timestamps
   - Use SRT text as transcriptions
   - Apply intelligent buffering for precise alignment
   - Generate metadata CSV files

### Method 3: YouTube Video Download (Automated)
1. Enter YouTube URL
2. Select language for subtitles/captions
3. Click "Download & Process YouTube"
4. System will:
   - Download audio from video
   - Download/extract subtitles in selected language
   - Process segments automatically
   - Generate dataset

### Method 4: RMS-Based Audio Slicing (Silence Detection)
1. Upload long audio file
2. Configure silence detection parameters
3. Enable auto-transcription
4. System will:
   - Detect silence and split audio
   - Use Whisper to transcribe segments
   - Generate metadata CSV files

## Dataset Quality Guidelines

### Audio Quality
✅ **Good:**
- Clear speech
- Consistent volume
- Minimal background noise
- Natural speech pace
- 3-10 second segments

❌ **Avoid:**
- Music overlapping speech
- Multiple speakers talking simultaneously
- Heavy background noise
- Distorted or clipped audio
- Too short (<0.5s) or too long (>15s) segments

### Text Quality
✅ **Good:**
- Accurate transcription
- Proper punctuation
- Consistent spelling
- Native script (e.g., አማርኛ for Amharic)

❌ **Avoid:**
- Transcription errors
- Missing punctuation
- Inconsistent romanization
- Timestamps or metadata in text

### Dataset Size
- **Minimum**: 100 segments (~5 minutes)
- **Recommended**: 500-1000 segments (30-60 minutes)
- **Optimal**: 2000+ segments (2+ hours)

## Dataset Split

The system automatically splits your data:
- **Training Set**: 85% of segments (shuffled)
- **Evaluation Set**: 15% of segments (shuffled)

This split is done randomly with `random_state=42` for reproducibility.

## Validation

Before training, verify your dataset:

1. **Check audio files exist:**
   ```bash
   ls dataset/wavs/*.wav | wc -l
   ```

2. **Check metadata format:**
   ```bash
   head -n 5 dataset/metadata_train.csv
   ```

3. **Verify language file:**
   ```bash
   cat dataset/lang.txt
   ```

4. **Count segments:**
   ```bash
   # Linux/Mac
   wc -l dataset/metadata_train.csv
   wc -l dataset/metadata_eval.csv
   
   # Windows PowerShell
   (Get-Content dataset/metadata_train.csv | Measure-Object -Line).Lines
   ```

5. **Check for encoding issues:**
   - Open CSV in text editor
   - Verify UTF-8 encoding
   - Check for special characters display correctly

## Common Issues and Solutions

### Issue 1: Audio/Text Mismatch
**Symptom:** Audio is longer than text, text cuts off
**Solution:** Update to latest version with intelligent buffering fix

### Issue 2: CSV Parse Error
**Symptom:** Training fails with CSV parsing error
**Solution:** 
- Check delimiter is `|` not `,`
- Verify UTF-8 encoding
- Remove any quotes around fields

### Issue 3: Missing Audio Files
**Symptom:** Training fails with "file not found"
**Solution:**
- Verify paths in CSV are relative: `wavs/file.wav`
- Check all WAV files exist in `wavs/` directory
- Ensure correct case (Linux is case-sensitive)

### Issue 4: Poor Training Results
**Symptom:** Model doesn't learn speaker voice
**Solution:**
- Increase dataset size (aim for 1+ hour)
- Improve audio quality (remove noise)
- Verify text transcriptions are accurate
- Check audio segments are 3-10 seconds

## Advanced: Manual Dataset Creation

If you want to create a dataset manually:

```python
import pandas as pd
import os

# Create metadata
metadata = {
    "audio_file": [],
    "text": [],
    "speaker_name": []
}

# Add your segments
metadata["audio_file"].append("wavs/audio_000000.wav")
metadata["text"].append("Your transcribed text here")
metadata["speaker_name"].append("speaker")

# Create DataFrame
df = pd.DataFrame(metadata)

# Shuffle and split
df_shuffled = df.sample(frac=1, random_state=42)
split_idx = int(len(df_shuffled) * 0.85)

train_df = df_shuffled[:split_idx]
eval_df = df_shuffled[split_idx:]

# Save with pipe delimiter
train_df.to_csv("metadata_train.csv", sep="|", index=False)
eval_df.to_csv("metadata_eval.csv", sep="|", index=False)

# Save language file
with open("lang.txt", "w", encoding="utf-8") as f:
    f.write("en\n")
```

## References

- [XTTS Documentation](https://docs.coqui.ai/en/latest/models/xtts.html)
- [ISO 639-1 Language Codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
- [SRT Format Specification](https://en.wikipedia.org/wiki/SubRip)
- [WAV Format Specification](https://en.wikipedia.org/wiki/WAV)
