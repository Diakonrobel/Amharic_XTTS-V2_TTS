# VAD-Enhanced Processing - Quick Start

## 🚀 What is VAD?

**Voice Activity Detection (VAD)** automatically detects when someone is speaking vs silence. This creates much cleaner audio segments for training by:

- ✂️ **Trimming silence** at the start/end of segments
- 🎯 **Finding natural pauses** for splitting long audio
- 🔊 **Detecting actual speech** boundaries (not just volume)
- 📝 **Aligning to word boundaries** (no cuts mid-word!)

## 🎬 When Should You Use VAD?

### ✅ Use VAD For:
- **SRT/VTT files** where timestamps aren't perfect
- **Podcast/interview** audio with variable pauses
- **Noisy recordings** with background sounds
- **Long audio files** needing automatic segmentation
- **Multiple speakers** with gaps between utterances

### ⚡ Skip VAD For:
- **Clean studio recordings** with perfect SRT timestamps
- **Pre-segmented files** that are already clean
- **Speed-critical processing** (VAD adds ~20% time)
- **Very short files** (<30 seconds total)

## 📝 Using VAD with SRT Files

### Command Line

```bash
# Standard processing (current method)
python utils/srt_processor.py video.srt video.mp4 output/ en

# VAD-enhanced processing (new method)
python utils/srt_processor_vad.py video.srt video.mp4 output/ en true
#                                                            ↑
#                                                       enable VAD
```

### Python API

```python
# Option 1: Standard SRT processing
from utils.srt_processor import process_srt_with_media

train_csv, eval_csv, duration = process_srt_with_media(
    srt_path="video.srt",
    media_path="video.mp4",
    output_dir="dataset/"
)

# Option 2: VAD-enhanced SRT processing
from utils.srt_processor_vad import process_srt_with_media_vad

train_csv, eval_csv, duration = process_srt_with_media_vad(
    srt_path="video.srt",
    media_path="video.mp4",
    output_dir="dataset/",
    use_vad_refinement=True,  # ← Enable VAD
    vad_threshold=0.5  # ← Sensitivity (0.3-0.7)
)
```

## 🎛️ Simple Settings Guide

### `vad_threshold` (0-1)

**Lower = More Sensitive** (detects quiet speech, may include noise)
**Higher = Less Sensitive** (only clear speech, misses quiet parts)

```python
vad_threshold=0.3  # Very sensitive - use for quiet/whispered speech
vad_threshold=0.5  # Balanced (DEFAULT) - good for most audio
vad_threshold=0.7  # Strict - only clear, loud speech
```

### When to Adjust

**If you see:**
- ❌ Segments have too much silence → Increase threshold (0.6-0.7)
- ❌ Missing quiet speech → Decrease threshold (0.3-0.4)
- ✅ Clean segments, all speech captured → Keep default (0.5)

## 🔊 Standalone Audio Slicing (No SRT)

### Basic Usage

```python
from utils.vad_slicer import slice_audio_with_vad

# Slice a long audio file automatically
segment_paths = slice_audio_with_vad(
    audio_path="long_recording.wav",
    output_dir="segments/",
    min_segment_duration=1.0,   # Min 1 second
    max_segment_duration=15.0,  # Max 15 seconds
    vad_threshold=0.5
)

print(f"Created {len(segment_paths)} segments")
# segments/long_recording_vad_0000.wav
# segments/long_recording_vad_0001.wav
# ...
```

### With Progress Tracking

```python
from utils.vad_slicer import VADSlicer
import librosa
import soundfile as sf
from tqdm import tqdm

# Load audio
audio, sr = librosa.load("audio.wav", sr=22050)

# Initialize VAD slicer
slicer = VADSlicer(
    sample_rate=sr,
    vad_threshold=0.5,
    min_segment_duration=1.0,
    max_segment_duration=15.0
)

# Slice with progress
print("Slicing audio with VAD...")
segments = slicer.slice_audio(audio)

# Save segments
print(f"Saving {len(segments)} segments...")
for i, seg in enumerate(tqdm(segments)):
    filename = f"segment_{i:04d}.wav"
    sf.write(filename, seg.audio, sr)
    print(f"  {filename}: {seg.end_time - seg.start_time:.1f}s")
```

## 📊 Comparing Results

### Before VAD (Standard RMS-based slicing)
```
segment_0001.wav: [silence] Hello world [silence]  (4.2s)
                   ↑ 0.5s    2.8s       ↑ 0.9s
```

### After VAD
```
segment_0001.wav: Hello world  (2.8s)
                  ↑ silence trimmed on both ends
```

**Result**: Cleaner segments, better training data!

## 🎯 Real-World Examples

### Example 1: Podcast Processing

```python
# Podcast has: intro music, speech, pauses, outro music
# VAD will automatically:
# - Skip intro/outro music (no speech)
# - Split at natural pauses between sentences
# - Trim silence from each segment

from utils.vad_slicer import slice_audio_with_vad

segments = slice_audio_with_vad(
    audio_path="podcast_episode.mp3",
    output_dir="podcast_segments/",
    vad_threshold=0.45,  # Slightly lower for podcasts
    min_segment_duration=2.0,  # Longer minimum for full sentences
    max_segment_duration=12.0  # Shorter max for training
)
```

### Example 2: Interview with Timestamps

```python
# Interview has SRT but timestamps include silence
# VAD will:
# - Use SRT text for each segment
# - Trim silence within each SRT segment
# - Split long answers into multiple segments

from utils.srt_processor_vad import process_srt_with_media_vad

train_csv, eval_csv, duration = process_srt_with_media_vad(
    srt_path="interview.srt",
    media_path="interview.mp4",
    output_dir="interview_dataset/",
    language="en",
    use_vad_refinement=True,
    vad_threshold=0.5
)
```

## 🔍 Checking Your Results

### 1. Visual Inspection (Recommended)

```python
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Load a segment
audio, sr = librosa.load("segments/segment_0001.wav")

# Plot waveform
plt.figure(figsize=(12, 4))
librosa.display.waveshow(audio, sr=sr)
plt.title("Segment 1")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# Check: Does speech start immediately? Is there silence at end?
```

### 2. Duration Check

```python
import librosa
import os

# Check all segments
for file in os.listdir("segments"):
    if file.endswith(".wav"):
        audio, sr = librosa.load(f"segments/{file}")
        duration = len(audio) / sr
        
        # Calculate silence at start/end
        # (Simple threshold-based check)
        threshold = 0.01
        start_silence = 0
        for i, sample in enumerate(audio):
            if abs(sample) > threshold:
                start_silence = i / sr
                break
        
        end_silence = 0
        for i, sample in enumerate(reversed(audio)):
            if abs(sample) > threshold:
                end_silence = i / sr
                break
        
        print(f"{file}: {duration:.1f}s (silence: {start_silence:.2f}s + {end_silence:.2f}s)")
```

### 3. Metadata Validation

```python
import pandas as pd

# Check dataset CSV
train_df = pd.read_csv("output/metadata_train.csv", sep='|')

print(f"Total segments: {len(train_df)}")
print(f"\nSample entries:")
print(train_df.head())

# Check for very short/long segments
durations = []
for audio_file in train_df['audio_file']:
    audio, sr = librosa.load(f"output/{audio_file}")
    durations.append(len(audio) / sr)

print(f"\nDuration stats:")
print(f"  Min: {min(durations):.2f}s")
print(f"  Max: {max(durations):.2f}s")
print(f"  Mean: {sum(durations)/len(durations):.2f}s")
```

## ⚠️ Common Issues

### Issue: "Failed to load Silero VAD"

**Solution**: Model will auto-download on first use. Ensure internet connection.

```bash
# Manual download if needed:
python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad')"
```

### Issue: Segments still have too much silence

**Solution**: Increase VAD threshold

```python
vad_threshold=0.6  # or 0.7 for very strict
```

### Issue: Missing quiet speech parts

**Solution**: Decrease VAD threshold

```python
vad_threshold=0.3  # or 0.4 for more sensitive
```

### Issue: Too many very short segments

**Solution**: Increase minimum duration

```python
min_segment_duration=2.0  # Require at least 2 seconds
```

## 🎓 Best Practices

1. **Start with defaults** (vad_threshold=0.5)
2. **Process a small sample** (first 5 minutes)
3. **Inspect visually** with waveform plots
4. **Adjust if needed** based on results
5. **Re-process full file** with tuned settings

## 📖 Next Steps

- **Full Documentation**: See `VAD_IMPLEMENTATION.md`
- **Technical Details**: See module docstrings
- **Integration**: Coming soon - UI toggle for VAD mode

## 💡 Pro Tips

### Tip 1: Batch Processing

```python
# Process multiple files with same settings
files = ["file1.srt", "file2.srt", "file3.srt"]

for srt_file in files:
    media_file = srt_file.replace(".srt", ".mp4")
    output_dir = f"datasets/{srt_file.replace('.srt', '')}"
    
    process_srt_with_media_vad(
        srt_path=srt_file,
        media_path=media_file,
        output_dir=output_dir,
        use_vad_refinement=True,
        vad_threshold=0.5
    )
```

### Tip 2: Save Settings

```python
# Save successful settings for future use
vad_config = {
    "vad_threshold": 0.5,
    "min_segment_duration": 1.0,
    "max_segment_duration": 15.0,
    "min_silence_duration_ms": 300
}

import json
with open("vad_config.json", "w") as f:
    json.dump(vad_config, f, indent=2)
```

### Tip 3: A/B Testing

```python
# Compare VAD vs standard on same file
from utils import srt_processor, srt_processor_vad

# Standard
train1, eval1, dur1 = srt_processor.process_srt_with_media(
    "test.srt", "test.mp4", "output_standard/"
)

# VAD-enhanced
train2, eval2, dur2 = srt_processor_vad.process_srt_with_media_vad(
    "test.srt", "test.mp4", "output_vad/",
    use_vad_refinement=True
)

# Compare segment counts
import pandas as pd
df1 = pd.read_csv(train1, sep='|')
df2 = pd.read_csv(train2, sep='|')

print(f"Standard: {len(df1)} segments")
print(f"VAD: {len(df2)} segments")
print(f"Difference: {len(df2) - len(df1)} more segments with VAD")
```

## 🎉 Summary

**VAD = Cleaner Segments = Better Training Data**

- 🎯 More precise speech boundaries
- ✂️ Automatic silence trimming
- 🔊 Works with noisy audio
- 📝 Respects SRT timestamps
- 🚀 Easy to use!

**Try it now** on your next SRT+media processing job!
