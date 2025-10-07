# Headless CLI - Advanced Dataset Processing Examples

Complete command-line interface for XTTS fine-tuning with advanced dataset processing features.

---

## 📋 **Table of Contents**

1. [Basic Audio Processing](#basic-audio-processing)
2. [SRT + Media File Processing](#srt--media-file-processing)
3. [YouTube Video Download](#youtube-video-download)
4. [RMS-Based Audio Slicing](#rms-based-audio-slicing)
5. [Advanced Options](#advanced-options)
6. [Complete Examples](#complete-examples)

---

## 🎵 **Basic Audio Processing**

### Simple Training with Audio File
```bash
python headlessXttsTrain.py \
  --input_audio speaker.wav \
  --lang en \
  --epochs 10
```

### With Custom Parameters
```bash
python headlessXttsTrain.py \
  --input_audio speaker.mp3 \
  --lang en \
  --epochs 15 \
  --batch_size 4 \
  --whisper_model large-v3 \
  --output_dir_base ./my_models \
  --model_name my_speaker
```

---

## 📝 **SRT + Media File Processing**

### Process SRT with Audio File
```bash
python headlessXttsTrain.py \
  --srt_file subtitles.srt \
  --media_file audio.mp3 \
  --lang en \
  --epochs 10
```

### Process SRT with Video File
```bash
python headlessXttsTrain.py \
  --srt_file movie_subtitles.srt \
  --media_file movie.mp4 \
  --lang es \
  --epochs 12 \
  --model_name movie_voice
```

### Multiple Formats Supported
```bash
# With WAV audio
python headlessXttsTrain.py --srt_file subs.srt --media_file audio.wav --lang fr --epochs 10

# With MKV video
python headlessXttsTrain.py --srt_file subs.vtt --media_file video.mkv --lang de --epochs 10

# With AVI video
python headlessXttsTrain.py --srt_file transcript.srt --media_file lecture.avi --lang en --epochs 10
```

---

## 📹 **YouTube Video Download**

### Basic YouTube Processing
```bash
python headlessXttsTrain.py \
  --youtube_url "https://youtube.com/watch?v=VIDEO_ID" \
  --lang en \
  --epochs 10
```

### With Specific Transcript Language
```bash
python headlessXttsTrain.py \
  --youtube_url "https://youtube.com/watch?v=VIDEO_ID" \
  --youtube_transcript_lang es \
  --lang es \
  --epochs 12 \
  --model_name youtube_speaker
```

### Multiple Language Options
```bash
# English
python headlessXttsTrain.py --youtube_url "URL" --youtube_transcript_lang en --lang en --epochs 10

# Spanish
python headlessXttsTrain.py --youtube_url "URL" --youtube_transcript_lang es --lang es --epochs 10

# French
python headlessXttsTrain.py --youtube_url "URL" --youtube_transcript_lang fr --lang fr --epochs 10

# Amharic
python headlessXttsTrain.py --youtube_url "URL" --youtube_transcript_lang amh --lang amh --epochs 10
```

---

## ✂️ **RMS-Based Audio Slicing**

### Enable Audio Slicing
```bash
python headlessXttsTrain.py \
  --input_audio long_recording.wav \
  --slice_audio \
  --lang en \
  --epochs 10
```

### Custom Slicing Parameters
```bash
python headlessXttsTrain.py \
  --input_audio podcast.mp3 \
  --slice_audio \
  --slicer_threshold_db -35 \
  --slicer_min_length 3.0 \
  --slicer_min_interval 0.5 \
  --slicer_max_sil_kept 0.3 \
  --lang en \
  --epochs 12
```

### Parameter Explanations

**`--slicer_threshold_db`** (default: -40.0)
- Volume threshold for silence detection
- Lower values = more sensitive (splits on quieter audio)
- Higher values = less sensitive (only splits on louder silences)
- Examples: `-50` (very sensitive), `-35` (less sensitive)

**`--slicer_min_length`** (default: 5.0 seconds)
- Minimum segment duration
- Longer = fewer, longer segments
- Shorter = more, shorter segments
- Examples: `3.0` (shorter segments), `10.0` (longer segments)

**`--slicer_min_interval`** (default: 0.3 seconds)
- Minimum silence duration required to split
- Examples: `0.1` (split on brief pauses), `1.0` (only split on long pauses)

**`--slicer_max_sil_kept`** (default: 0.5 seconds)
- Silence padding at segment boundaries
- Examples: `0.2` (minimal padding), `1.0` (more padding)

---

## 🔧 **Advanced Options**

### Amharic G2P Preprocessing
```bash
python headlessXttsTrain.py \
  --input_audio amharic_speaker.wav \
  --lang amh \
  --use_g2p \
  --g2p_backend transphone \
  --epochs 10
```

### Custom Base Model
```bash
python headlessXttsTrain.py \
  --input_audio speaker.wav \
  --lang en \
  --custom_model path/to/custom_model.pth \
  --epochs 10
```

### Training Configuration
```bash
python headlessXttsTrain.py \
  --input_audio speaker.wav \
  --lang en \
  --epochs 20 \
  --batch_size 8 \
  --grad_acumm 2 \
  --max_audio_length 15 \
  --xtts_base_version v2.0.3 \
  --whisper_model large-v3
```

### Inference Parameters
```bash
python headlessXttsTrain.py \
  --input_audio speaker.wav \
  --lang en \
  --epochs 10 \
  --example_text "Custom text for testing the voice" \
  --temperature 0.8 \
  --length_penalty 1.2 \
  --repetition_penalty 3.0
```

---

## 🚀 **Complete Examples**

### Example 1: Movie Dubbing Dataset
```bash
python headlessXttsTrain.py \
  --srt_file movie_subtitles.srt \
  --media_file movie.mp4 \
  --lang es \
  --epochs 15 \
  --batch_size 4 \
  --model_name movie_character \
  --output_dir_base ./dubbing_models
```

### Example 2: Podcast Processing
```bash
python headlessXttsTrain.py \
  --input_audio podcast_episode.mp3 \
  --slice_audio \
  --slicer_threshold_db -35 \
  --slicer_min_length 7.0 \
  --lang en \
  --epochs 12 \
  --whisper_model large-v3 \
  --model_name podcast_host
```

### Example 3: YouTube Content Creator
```bash
python headlessXttsTrain.py \
  --youtube_url "https://youtube.com/watch?v=EXAMPLE123" \
  --youtube_transcript_lang en \
  --lang en \
  --epochs 18 \
  --batch_size 6 \
  --model_name youtube_creator \
  --example_text "Welcome back to my channel, today we're going to..."
```

### Example 4: Amharic Voice Training
```bash
python headlessXttsTrain.py \
  --input_audio amharic_speech.wav \
  --lang amh \
  --use_g2p \
  --g2p_backend transphone \
  --epochs 20 \
  --batch_size 2 \
  --model_name amharic_speaker \
  --example_text "ሰላም። እንደምን አደርክ?"
```

### Example 5: Multi-Speaker Audiobook
```bash
python headlessXttsTrain.py \
  --srt_file audiobook_chapters.srt \
  --media_file audiobook.m4a \
  --lang en \
  --slice_audio \
  --epochs 25 \
  --batch_size 8 \
  --model_name narrator_voice \
  --output_dir_base ./audiobook_voices
```

---

## 📊 **Input Method Validation**

The CLI requires **exactly ONE** input method:

✅ **Valid** (one method):
```bash
python headlessXttsTrain.py --input_audio file.wav --lang en --epochs 10
python headlessXttsTrain.py --srt_file subs.srt --media_file video.mp4 --lang en --epochs 10
python headlessXttsTrain.py --youtube_url "URL" --lang en --epochs 10
```

❌ **Invalid** (no method):
```bash
python headlessXttsTrain.py --lang en --epochs 10  # Error: No input specified
```

❌ **Invalid** (multiple methods):
```bash
python headlessXttsTrain.py --input_audio file.wav --youtube_url "URL" --lang en --epochs 10
# Error: Multiple input methods specified
```

---

## 🆘 **Help & Options**

### View All Options
```bash
python headlessXttsTrain.py --help
```

### Common Options Summary

| Option | Default | Description |
|--------|---------|-------------|
| `--input_audio` | - | Path to audio file |
| `--srt_file` | - | Path to SRT subtitle file |
| `--media_file` | - | Path to media file (with SRT) |
| `--youtube_url` | - | YouTube video URL |
| `--lang` | `en` | Dataset language |
| `--epochs` | `10` | Number of training epochs |
| `--batch_size` | `2` | Training batch size |
| `--whisper_model` | `large-v3` | Whisper model for transcription |
| `--slice_audio` | `False` | Enable audio slicing |
| `--use_g2p` | `False` | Enable Amharic G2P |
| `--output_dir_base` | `./xtts_finetuned_models` | Output directory |
| `--model_name` | (auto) | Custom model name |

---

## 🐛 **Troubleshooting**

### FFmpeg Not Found
**Error:** `ffmpeg command not found`
**Solution:** Install FFmpeg and add to PATH
```bash
# Windows: Download from ffmpeg.org
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg
```

### YouTube Download Fails
**Error:** SSL certificate or network issues
**Solution:** Update yt-dlp
```bash
pip install -U yt-dlp
```

### Out of Memory
**Error:** CUDA out of memory during training
**Solution:** Reduce batch size
```bash
python headlessXttsTrain.py --input_audio file.wav --batch_size 1 --lang en --epochs 10
```

### No Transcripts Available
**Error:** YouTube video has no subtitles
**Solution:** Use a different video or use `--input_audio` with manual transcription

---

## 📝 **Notes**

1. **Processing Time:** 
   - SRT processing: Fast (depends on media file size)
   - YouTube download: Varies by video length and network speed
   - Audio slicing + transcription: Slower (Whisper transcription required)

2. **Output Structure:**
   ```
   xtts_finetuned_models/
   └── model_name/
       ├── dataset/          # Processed dataset
       │   ├── wavs/         # Audio segments
       │   ├── metadata_train.csv
       │   ├── metadata_eval.csv
       │   └── lang.txt
       ├── run/              # Training checkpoints
       └── ready/            # Optimized model
           ├── model.pth
           ├── config.json
           ├── vocab.json
           ├── speakers_xtts.pth
           └── reference.wav
   ```

3. **YouTube ToS:** Only download content you have permission to use. For personal/educational use only.

4. **SRT Accuracy:** Subtitle timestamps should be accurate for good audio-text alignment.

---

**Version:** 2.0  
**Last Updated:** 2025-01-07  
**Status:** Production Ready
