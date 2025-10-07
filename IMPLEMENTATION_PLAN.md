# IMPLEMENTATION PLAN: Advanced Dataset Processing Features

## Overview

This document provides a complete implementation plan for adding three major features to the XTTS Fine-Tuning WebUI:

1. **SRT + Media Processing**: Process subtitle files with audio/video
2. **YouTube Download**: Download videos with transcripts using yt-dlp
3. **Dataset-Maker Integration**: Advanced audio slicing and processing

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Dependencies](#dependencies)
3. [Module Implementation](#module-implementation)
4. [UI Integration](#ui-integration)
5. [Headless Mode](#headless-mode)
6. [Testing](#testing)
7. [Documentation Updates](#documentation-updates)
8. [Git Workflow](#git-workflow)

---

## Architecture Overview

### New Components

```
xtts-finetune-webui-fresh/
├── utils/
│   ├── srt_processor.py          ✅ CREATED (extract segments from SRT+media)
│   ├── youtube_downloader.py     ⬜ TO CREATE (download YouTube with yt-dlp)
│   ├── audio_slicer.py           ⬜ TO CREATE (RMS-based segmentation)
│   └── dataset_processor.py      ⬜ TO CREATE (unified processing pipeline)
│
├── xtts_demo.py                  ⬜ TO UPDATE (add new UI components)
├── headlessXttsTrain.py          ⬜ TO UPDATE (add CLI args)
└── requirements.txt              ⬜ TO UPDATE (add new dependencies)
```

### Data Flow

```
INPUT OPTIONS:
  1. Audio files → Faster Whisper (existing)
  2. SRT + Media → srt_processor.py (new)
  3. YouTube URL → youtube_downloader.py (new)
  4. Audio files → audio_slicer.py (new, enhanced)
                      ↓
              Dataset Creation
                      ↓
          metadata_train.csv + metadata_eval.csv
                      ↓
                  Training
```

---

## Dependencies

### Update requirements.txt

Add these dependencies:

```txt
# Existing dependencies...
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

# NEW DEPENDENCIES FOR ADVANCED FEATURES
pysrt>=1.1.2              # SRT file parsing
yt-dlp>=2024.1.0          # YouTube download
whisperx>=3.1.1           # Enhanced Whisper with alignment (optional)
librosa>=0.10.0           # Audio processing (already used)
soundfile>=0.12.1         # Audio I/O
```

### Windows Installation Commands

```powershell
# Install new dependencies
pip install pysrt yt-dlp soundfile

# Optional: Install WhisperX for enhanced processing
pip install whisperx

# Update yt-dlp (important for YouTube changes)
python -m pip install -U yt-dlp
```

---

## Module Implementation

### 1. YouTube Downloader Module

**File**: `utils/youtube_downloader.py`

```python
"""
YouTube Downloader Module
Downloads videos and subtitles from YouTube using yt-dlp with auto-update.
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import yt_dlp


def update_ytdlp():
    """
    Update yt-dlp to the latest version.
    """
    try:
        print("Updating yt-dlp...")
        subprocess.run(
            ["python", "-m", "pip", "install", "-U", "yt-dlp"],
            check=True,
            capture_output=True
        )
        print("✓ yt-dlp updated successfully")
        return True
    except Exception as e:
        print(f"Warning: Could not update yt-dlp: {e}")
        return False


def get_video_info(url: str) -> Dict:
    """
    Get video information without downloading.
    
    Args:
        url: YouTube video URL
        
    Returns:
        Dictionary with video metadata
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            'title': info.get('title', 'Unknown'),
            'duration': info.get('duration', 0),
            'uploader': info.get('uploader', 'Unknown'),
            'description': info.get('description', ''),
            'has_subtitles': bool(info.get('subtitles', {})),
            'has_automatic_captions': bool(info.get('automatic_captions', {})),
            'available_languages': list(info.get('subtitles', {}).keys())
        }


def download_youtube_video(
    url: str,
    output_dir: str,
    language: str = 'en',
    audio_only: bool = True,
    download_subtitles: bool = True,
    auto_update: bool = True
) -> Tuple[Optional[str], Optional[str], Dict]:
    """
    Download YouTube video/audio and subtitles.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save files
        language: Preferred subtitle language (ISO 639-1 code)
        audio_only: If True, download only audio
        download_subtitles: If True, attempt to download subtitles
        auto_update: If True, update yt-dlp before downloading
        
    Returns:
        Tuple of (audio_path, subtitle_path, video_info)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Auto-update yt-dlp
    if auto_update:
        update_ytdlp()
    
    # Get video info first
    print(f"Fetching video information from {url}...")
    try:
        info = get_video_info(url)
        print(f"  Title: {info['title']}")
        print(f"  Duration: {info['duration']}s")
        print(f"  Has subtitles: {info['has_subtitles']}")
        print(f"  Has auto-captions: {info['has_automatic_captions']}")
    except Exception as e:
        print(f"Warning: Could not fetch video info: {e}")
        info = {}
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best' if audio_only else 'bestvideo+bestaudio/best',
        'outtmpl': str(output_path / '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'extract_audio': audio_only,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }] if audio_only else [],
    }
    
    # Add subtitle options
    if download_subtitles:
        ydl_opts.update({
            'writesubtitles': True,
            'writeautomaticsub': True,  # Fallback to auto-generated
            'subtitleslangs': [language, 'en'],  # Preferred language + English fallback
            'subtitlesformat': 'srt',
            'skip_download': False,
        })
    
    # Download
    print(f"\nDownloading from YouTube...")
    audio_file = None
    subtitle_file = None
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            
            # Find downloaded files
            title = result.get('title', 'video')
            sanitized_title = yt_dlp.utils.sanitize_filename(title)
            
            # Look for audio file
            audio_patterns = [f"{sanitized_title}.wav", f"{sanitized_title}.mp3", f"{sanitized_title}.m4a"]
            for pattern in audio_patterns:
                audio_candidate = output_path / pattern
                if audio_candidate.exists():
                    audio_file = str(audio_candidate)
                    print(f"✓ Audio downloaded: {audio_file}")
                    break
            
            # Look for subtitle file
            if download_subtitles:
                subtitle_patterns = [
                    f"{sanitized_title}.{language}.srt",
                    f"{sanitized_title}.en.srt",
                    f"{sanitized_title}.srt"
                ]
                for pattern in subtitle_patterns:
                    subtitle_candidate = output_path / pattern
                    if subtitle_candidate.exists():
                        subtitle_file = str(subtitle_candidate)
                        print(f"✓ Subtitles downloaded: {subtitle_file}")
                        break
                
                if not subtitle_file:
                    print("⚠ No subtitles found (video may not have subtitles available)")
    
    except Exception as e:
        print(f"Error downloading from YouTube: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return audio_file, subtitle_file, info


def download_and_process_youtube(
    url: str,
    output_dir: str,
    language: str = 'en',
    use_whisper_if_no_srt: bool = True,
    auto_update: bool = True
) -> Tuple[Optional[str], Optional[str], Dict]:
    """
    Download YouTube video and prepare for dataset creation.
    If subtitles aren't available, optionally use Whisper to transcribe.
    
    Args:
        url: YouTube video URL
        output_dir: Output directory
        language: Language code
        use_whisper_if_no_srt: Use Whisper transcription if no SRT available
        auto_update: Auto-update yt-dlp
        
    Returns:
        Tuple of (audio_path, srt_path, info_dict)
    """
    # Download video/audio and subtitles
    audio_path, srt_path, info = download_youtube_video(
        url=url,
        output_dir=output_dir,
        language=language,
        audio_only=True,
        download_subtitles=True,
        auto_update=auto_update
    )
    
    if not audio_path:
        raise RuntimeError("Failed to download audio from YouTube")
    
    # If no subtitles and Whisper fallback enabled
    if not srt_path and use_whisper_if_no_srt:
        print("\n⚠ No subtitles available, using Faster Whisper to transcribe...")
        srt_path = transcribe_with_whisper(
            audio_path=audio_path,
            output_dir=output_dir,
            language=language
        )
    
    return audio_path, srt_path, info


def transcribe_with_whisper(
    audio_path: str,
    output_dir: str,
    language: str = 'en',
    model_size: str = 'large-v3'
) -> str:
    """
    Transcribe audio using Faster Whisper and save as SRT.
    
    Args:
        audio_path: Path to audio file
        output_dir: Output directory
        language: Language code
        model_size: Whisper model size
        
    Returns:
        Path to generated SRT file
    """
    from faster_whisper import WhisperModel
    import pysrt
    
    print(f"Loading Whisper model ({model_size})...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    
    print("Transcribing audio...")
    segments, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        word_timestamps=True
    )
    
    # Convert to SRT format
    srt_path = Path(output_dir) / f"{Path(audio_path).stem}.srt"
    srt_subs = pysrt.SubRipFile()
    
    for idx, segment in enumerate(segments, start=1):
        # Convert seconds to SubRipTime
        start_time = pysrt.SubRipTime(seconds=segment.start)
        end_time = pysrt.SubRipTime(seconds=segment.end)
        
        subtitle = pysrt.SubRipItem(
            index=idx,
            start=start_time,
            end=end_time,
            text=segment.text.strip()
        )
        srt_subs.append(subtitle)
    
    srt_subs.save(str(srt_path), encoding='utf-8')
    print(f"✓ Transcription saved to {srt_path}")
    
    return str(srt_path)


if __name__ == "__main__":
    # Test usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python youtube_downloader.py <youtube_url> [output_dir] [language]")
        sys.exit(1)
    
    url = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./youtube_downloads"
    language = sys.argv[3] if len(sys.argv) > 3 else "en"
    
    try:
        audio, srt, info = download_and_process_youtube(
            url=url,
            output_dir=output_dir,
            language=language,
            use_whisper_if_no_srt=True
        )
        
        print(f"\n✓ Download complete!")
        print(f"  Audio: {audio}")
        print(f"  Subtitles: {srt}")
        print(f"  Title: {info.get('title', 'N/A')}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
```

### 2. Audio Slicer Module (from dataset-maker)

**File**: `utils/audio_slicer.py`

```python
"""
Audio Slicer Module
RMS-based silence detection and segmentation (from dataset-maker/slicer2.py)
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple
import torch
import torchaudio


def get_rms(
    y,
    *,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    """
    Calculate RMS energy (from librosa).
    """
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    out_strides = y.strides + tuple([y.strides[axis]])
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(
        y, shape=out_shape, strides=out_strides
    )
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)


class AudioSlicer:
    """
    Audio slicer based on RMS energy and silence detection.
    From: https://github.com/openvpi/audio-slicer
    """
    
    def __init__(self,
                 sr: int,
                 threshold: float = -40.,
                 min_length: int = 5000,
                 min_interval: int = 300,
                 hop_size: int = 20,
                 max_sil_kept: int = 5000):
        """
        Args:
            sr: Sample rate
            threshold: RMS threshold in dB for silence detection
            min_length: Minimum length of audio chunks (ms)
            min_interval: Minimum interval for silence (ms)
            hop_size: Hop size for RMS calculation (ms)
            max_sil_kept: Maximum silence kept around chunks (ms)
        """
        if not min_length >= min_interval >= hop_size:
            raise ValueError('Must satisfy: min_length >= min_interval >= hop_size')
        if not max_sil_kept >= hop_size:
            raise ValueError('Must satisfy: max_sil_kept >= hop_size')
        
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]

    def slice(self, waveform):
        """
        Slice waveform based on silence detection.
        
        Args:
            waveform: Audio waveform (numpy array)
            
        Returns:
            List of audio chunks
        """
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        
        if (samples.shape[0] + self.hop_size - 1) // self.hop_size <= self.min_length:
            return [waveform]
        
        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            
            if silence_start is None:
                continue
            
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        
        # Handle trailing silence
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        
        # Apply slices
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]))
            if sil_tags[-1][1] < total_frames:
                chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))
            return chunks


def slice_audio_file(
    audio_path: str,
    output_dir: str,
    sr: int = 22050,
    threshold: float = -40.0,
    min_length: int = 5000,
    min_interval: int = 300,
    hop_size: int = 20,
    max_sil_kept: int = 500,
    max_segment_duration: float = 15.0,
    min_segment_duration: float = 1.0
) -> List[str]:
    """
    Slice audio file into segments based on silence detection.
    
    Args:
        audio_path: Path to audio file
        output_dir: Output directory for segments
        sr: Target sample rate
        threshold: RMS threshold in dB
        min_length: Minimum chunk length (ms)
        min_interval: Minimum silence interval (ms)
        hop_size: Hop size for RMS (ms)
        max_sil_kept: Max silence kept (ms)
        max_segment_duration: Max segment duration (seconds)
        min_segment_duration: Min segment duration (seconds)
        
    Returns:
        List of saved segment file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    print(f"Loading audio: {audio_path}")
    y, original_sr = librosa.load(str(audio_path), sr=sr, mono=False)
    
    # Initialize slicer
    slicer = AudioSlicer(
        sr=sr,
        threshold=threshold,
        min_length=min_length,
        min_interval=min_interval,
        hop_size=hop_size,
        max_sil_kept=max_sil_kept
    )
    
    # Slice audio
    print("Slicing audio based on silence detection...")
    chunks = slicer.slice(y)
    
    # Save chunks
    segment_paths = []
    base_name = Path(audio_path).stem
    
    for i, chunk in enumerate(chunks):
        # Calculate duration
        if chunk.ndim > 1:
            duration = chunk.shape[1] / sr
            chunk_to_save = chunk.T
        else:
            duration = len(chunk) / sr
            chunk_to_save = chunk
        
        # Filter by duration
        if duration < min_segment_duration or duration > max_segment_duration:
            print(f"  Skipping chunk {i}: duration {duration:.2f}s out of range")
            continue
        
        # Save segment
        segment_filename = f"{base_name}_seg{str(i).zfill(4)}.wav"
        segment_path = output_path / segment_filename
        
        sf.write(str(segment_path), chunk_to_save, sr)
        segment_paths.append(str(segment_path))
        print(f"  Saved segment {i}: {segment_filename} ({duration:.2f}s)")
    
    print(f"\n✓ Sliced into {len(segment_paths)} segments")
    return segment_paths
```

### 3. Unified Dataset Processor

**File**: `utils/dataset_processor.py`

```python
"""
Unified Dataset Processor
Combines all processing methods: Whisper, SRT, YouTube, Audio Slicer
"""

from pathlib import Path
from typing import Optional, Tuple, List
from enum import Enum

# Import our modules
from utils.formatter import format_audio_list
from utils.srt_processor import process_srt_with_media
from utils.youtube_downloader import download_and_process_youtube
from utils.audio_slicer import slice_audio_file


class ProcessingMode(Enum):
    """Dataset processing modes"""
    WHISPER = "whisper"              # Traditional Faster Whisper
    SRT_MEDIA = "srt_media"          # SRT + media file
    YOUTUBE = "youtube"              # YouTube URL
    AUDIO_SLICER = "audio_slicer"    # RMS-based slicing


def process_dataset(
    mode: ProcessingMode,
    output_dir: str,
    language: str = "en",
    speaker_name: str = "speaker",
    # Whisper mode params
    audio_files: Optional[List[str]] = None,
    whisper_model=None,
    # SRT mode params
    srt_path: Optional[str] = None,
    media_path: Optional[str] = None,
    # YouTube mode params
    youtube_url: Optional[str] = None,
    use_whisper_fallback: bool = True,
    # Audio slicer params
    use_slicer: bool = False,
    slicer_threshold: float = -40.0,
    # Common params
    min_duration: float = 0.5,
    max_duration: float = 15.0,
    gradio_progress=None
) -> Tuple[str, str, float]:
    """
    Unified dataset processing function supporting multiple input types.
    
    Args:
        mode: Processing mode (Whisper, SRT+Media, YouTube, AudioSlicer)
        output_dir: Output directory for dataset
        language: Language code
        speaker_name: Speaker identifier
        audio_files: List of audio files (for Whisper mode)
        whisper_model: Loaded Whisper model (for Whisper mode)
        srt_path: Path to SRT file (for SRT mode)
        media_path: Path to media file (for SRT mode)
        youtube_url: YouTube URL (for YouTube mode)
        use_whisper_fallback: Use Whisper if YouTube has no subtitles
        use_slicer: Use audio slicer instead of Whisper VAD
        slicer_threshold: RMS threshold for slicer
        min_duration: Minimum segment duration
        max_duration: Maximum segment duration
        gradio_progress: Gradio progress tracker
        
    Returns:
        Tuple of (train_csv, eval_csv, total_duration)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_csv = None
    eval_csv = None
    total_duration = 0.0
    
    if mode == ProcessingMode.WHISPER:
        # Traditional Faster Whisper processing
        if not audio_files or not whisper_model:
            raise ValueError("audio_files and whisper_model required for Whisper mode")
        
        print("Processing with Faster Whisper...")
        train_csv, eval_csv, total_duration = format_audio_list(
            audio_files=audio_files,
            asr_model=whisper_model,
            target_language=language,
            out_path=output_dir,
            speaker_name=speaker_name,
            gradio_progress=gradio_progress
        )
    
    elif mode == ProcessingMode.SRT_MEDIA:
        # SRT + Media file processing
        if not srt_path or not media_path:
            raise ValueError("srt_path and media_path required for SRT mode")
        
        print("Processing SRT + Media file...")
        train_csv, eval_csv, total_duration = process_srt_with_media(
            srt_path=srt_path,
            media_path=media_path,
            output_dir=output_dir,
            speaker_name=speaker_name,
            language=language,
            min_duration=min_duration,
            max_duration=max_duration,
            gradio_progress=gradio_progress
        )
    
    elif mode == ProcessingMode.YOUTUBE:
        # YouTube download and processing
        if not youtube_url:
            raise ValueError("youtube_url required for YouTube mode")
        
        print("Downloading and processing YouTube video...")
        
        # Step 1: Download
        audio_path, srt_path, info = download_and_process_youtube(
            url=youtube_url,
            output_dir=str(output_path / "youtube_download"),
            language=language,
            use_whisper_if_no_srt=use_whisper_fallback,
            auto_update=True
        )
        
        # Step 2: Process with SRT if available
        if srt_path:
            print("Processing downloaded content with SRT...")
            train_csv, eval_csv, total_duration = process_srt_with_media(
                srt_path=srt_path,
                media_path=audio_path,
                output_dir=output_dir,
                speaker_name=speaker_name,
                language=language,
                min_duration=min_duration,
                max_duration=max_duration,
                gradio_progress=gradio_progress
            )
        else:
            # Fallback to Whisper
            print("No subtitles available, using Whisper...")
            train_csv, eval_csv, total_duration = format_audio_list(
                audio_files=[audio_path],
                asr_model=whisper_model,
                target_language=language,
                out_path=output_dir,
                speaker_name=speaker_name,
                gradio_progress=gradio_progress
            )
    
    elif mode == ProcessingMode.AUDIO_SLICER:
        # Advanced audio slicing
        if not audio_files:
            raise ValueError("audio_files required for AudioSlicer mode")
        
        print("Processing with Audio Slicer...")
        
        # First, slice audio files
        all_segments = []
        for audio_file in audio_files:
            segments = slice_audio_file(
                audio_path=audio_file,
                output_dir=str(output_path / "sliced_segments"),
                threshold=slicer_threshold,
                max_segment_duration=max_duration,
                min_segment_duration=min_duration
            )
            all_segments.extend(segments)
        
        # Then transcribe with Whisper
        print(f"Transcribing {len(all_segments)} sliced segments...")
        train_csv, eval_csv, total_duration = format_audio_list(
            audio_files=all_segments,
            asr_model=whisper_model,
            target_language=language,
            out_path=output_dir,
            speaker_name=speaker_name,
            gradio_progress=gradio_progress
        )
    
    else:
        raise ValueError(f"Unknown processing mode: {mode}")
    
    print(f"\n✓ Dataset processing complete!")
    print(f"  Mode: {mode.value}")
    print(f"  Train CSV: {train_csv}")
    print(f"  Eval CSV: {eval_csv}")
    print(f"  Total duration: {total_duration:.2f}s")
    
    return train_csv, eval_csv, total_duration
```

---

## UI Integration

### Update xtts_demo.py - Tab 1 (Data Processing)

Add new processing options to the existing Tab 1. Here's the code to add:

```python
# Add after existing Tab 1 UI components (around line 400-500)

with gr.Tab("Data Processing"):
    # Existing components...
    
    # NEW: Processing Mode Selection
    with gr.Row():
        processing_mode = gr.Radio(
            choices=[
                "Whisper (Audio Files)",
                "SRT + Media File",
                "YouTube URL",
                "Audio Slicer (Advanced)"
            ],
            value="Whisper (Audio Files)",
            label="Dataset Source",
            info="Choose how to create your dataset"
        )
    
    # Existing: Whisper mode (default)
    with gr.Group(visible=True) as whisper_group:
        audio_folder_input = gr.Textbox(
            label="Audio Folder Path",
            value=str(audio_folder),
            interactive=True
        )
        # ... rest of existing Whisper UI
    
    # NEW: SRT + Media mode
    with gr.Group(visible=False) as srt_group:
        with gr.Row():
            srt_file_input = gr.File(
                label="Upload SRT Subtitle File",
                file_types=[".srt"],
                type="filepath"
            )
            media_file_input = gr.File(
                label="Upload Audio/Video File",
                file_types=[".mp4", ".mkv", ".avi", ".mov", ".wav", ".mp3", ".flac", ".m4a"],
                type="filepath"
            )
        
        with gr.Row():
            srt_min_duration = gr.Slider(
                minimum=0.1,
                maximum=5.0,
                value=0.5,
                step=0.1,
                label="Minimum Segment Duration (seconds)"
            )
            srt_max_duration = gr.Slider(
                minimum=5.0,
                maximum=30.0,
                value=15.0,
                step=1.0,
                label="Maximum Segment Duration (seconds)"
            )
        
        srt_info = gr.Markdown(
            """
            **SRT + Media Processing:**
            - Upload an SRT subtitle file and its corresponding media file
            - Segments will be extracted based on SRT timestamps
            - Supports both audio and video files
            - Automatic audio extraction from video
            """
        )
    
    # NEW: YouTube mode
    with gr.Group(visible=False) as youtube_group:
        youtube_url_input = gr.Textbox(
            label="YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            info="Paste YouTube video URL"
        )
        
        with gr.Row():
            youtube_auto_update = gr.Checkbox(
                label="Auto-update yt-dlp",
                value=True,
                info="Update yt-dlp before downloading"
            )
            youtube_use_whisper = gr.Checkbox(
                label="Use Whisper if no subtitles",
                value=True,
                info="Transcribe with Whisper if video has no subtitles"
            )
        
        youtube_info = gr.Markdown(
            """
            **YouTube Processing:**
            - Downloads video audio and subtitles automatically
            - Supports manual and auto-generated subtitles
            - Falls back to Whisper transcription if needed
            - Requires yt-dlp (installed automatically)
            """
        )
    
    # NEW: Audio Slicer mode
    with gr.Group(visible=False) as slicer_group:
        audio_slicer_folder = gr.Textbox(
            label="Audio Folder Path",
            value=str(audio_folder),
            interactive=True
        )
        
        with gr.Row():
            slicer_threshold = gr.Slider(
                minimum=-60.0,
                maximum=-20.0,
                value=-40.0,
                step=1.0,
                label="Silence Threshold (dB)",
                info="Lower = more sensitive to silence"
            )
            slicer_min_length = gr.Slider(
                minimum=1000,
                maximum=10000,
                value=5000,
                step=500,
                label="Minimum Segment Length (ms)"
            )
        
        slicer_info = gr.Markdown(
            """
            **Audio Slicer (Advanced):**
            - RMS-based silence detection and segmentation
            - More precise than VAD for clean recordings
            - Automatically splits long audio files
            - Then transcribes with Whisper
            """
        )
    
    # Add mode switching logic
    def update_processing_mode(mode):
        return {
            whisper_group: gr.update(visible=(mode == "Whisper (Audio Files)")),
            srt_group: gr.update(visible=(mode == "SRT + Media File")),
            youtube_group: gr.update(visible=(mode == "YouTube URL")),
            slicer_group: gr.update(visible=(mode == "Audio Slicer (Advanced)"))
        }
    
    processing_mode.change(
        fn=update_processing_mode,
        inputs=[processing_mode],
        outputs=[whisper_group, srt_group, youtube_group, slicer_group]
    )
    
    # Update the main processing button handler
    # (Modify existing "Step 1 - Create Dataset" button)
    def create_dataset_unified(
        mode,
        # Whisper params
        audio_folder,
        whisper_model_name,
        lang,
        # SRT params
        srt_file,
        media_file,
        srt_min_dur,
        srt_max_dur,
        # YouTube params
        youtube_url,
        youtube_auto_update,
        youtube_use_whisper,
        # Slicer params
        slicer_folder,
        slicer_thresh,
        slicer_min_len,
        # Common params
        out_path,
        speaker_name,
        progress=gr.Progress()
    ):
        try:
            from utils.dataset_processor import process_dataset, ProcessingMode
            from faster_whisper import WhisperModel
            
            # Determine processing mode
            if mode == "Whisper (Audio Files)":
                proc_mode = ProcessingMode.WHISPER
                # Load Whisper model
                whisper_model = WhisperModel(whisper_model_name, device="cuda", compute_type="float16")
                # Get audio files
                from utils.formatter import list_audios
                audio_files = list(list_audios(audio_folder))
                
                train_csv, eval_csv, duration = process_dataset(
                    mode=proc_mode,
                    output_dir=out_path,
                    language=lang,
                    speaker_name=speaker_name,
                    audio_files=audio_files,
                    whisper_model=whisper_model,
                    gradio_progress=progress
                )
            
            elif mode == "SRT + Media File":
                proc_mode = ProcessingMode.SRT_MEDIA
                
                if not srt_file or not media_file:
                    return "Please upload both SRT and media files!", "", ""
                
                train_csv, eval_csv, duration = process_dataset(
                    mode=proc_mode,
                    output_dir=out_path,
                    language=lang,
                    speaker_name=speaker_name,
                    srt_path=srt_file,
                    media_path=media_file,
                    min_duration=srt_min_dur,
                    max_duration=srt_max_dur,
                    gradio_progress=progress
                )
            
            elif mode == "YouTube URL":
                proc_mode = ProcessingMode.YOUTUBE
                
                if not youtube_url:
                    return "Please enter a YouTube URL!", "", ""
                
                # Load Whisper model for fallback
                whisper_model = WhisperModel(whisper_model_name, device="cuda", compute_type="float16")
                
                train_csv, eval_csv, duration = process_dataset(
                    mode=proc_mode,
                    output_dir=out_path,
                    language=lang,
                    speaker_name=speaker_name,
                    youtube_url=youtube_url,
                    use_whisper_fallback=youtube_use_whisper,
                    whisper_model=whisper_model,
                    gradio_progress=progress
                )
            
            elif mode == "Audio Slicer (Advanced)":
                proc_mode = ProcessingMode.AUDIO_SLICER
                # Load Whisper model
                whisper_model = WhisperModel(whisper_model_name, device="cuda", compute_type="float16")
                # Get audio files
                from utils.formatter import list_audios
                audio_files = list(list_audios(slicer_folder))
                
                train_csv, eval_csv, duration = process_dataset(
                    mode=proc_mode,
                    output_dir=out_path,
                    language=lang,
                    speaker_name=speaker_name,
                    audio_files=audio_files,
                    whisper_model=whisper_model,
                    use_slicer=True,
                    slicer_threshold=slicer_thresh,
                    gradio_progress=progress
                )
            
            return (
                f"✓ Dataset created successfully!\n"
                f"Train samples: {train_csv}\n"
                f"Eval samples: {eval_csv}\n"
                f"Total duration: {duration:.2f}s",
                train_csv,
                eval_csv
            )
        
        except Exception as e:
            import traceback
            error_msg = f"Error creating dataset: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg, "", ""
    
    # Wire up the button
    process_button.click(
        fn=create_dataset_unified,
        inputs=[
            processing_mode,
            # Whisper
            audio_folder_input,
            whisper_model_dropdown,
            lang_dropdown,
            # SRT
            srt_file_input,
            media_file_input,
            srt_min_duration,
            srt_max_duration,
            # YouTube
            youtube_url_input,
            youtube_auto_update,
            youtube_use_whisper,
            # Slicer
            audio_slicer_folder,
            slicer_threshold,
            slicer_min_length,
            # Common
            output_path,
            speaker_name_input
        ],
        outputs=[status_output, train_csv_output, eval_csv_output]
    )
```

---

## Headless Mode

### Update headlessXttsTrain.py

Add new command-line arguments:

```python
# Add to argument parser (around line 170-200)

parser.add_argument(
    "--processing_mode",
    type=str,
    choices=["whisper", "srt", "youtube", "slicer"],
    default="whisper",
    help="Dataset processing mode: whisper, srt, youtube, or slicer"
)

# SRT mode arguments
parser.add_argument(
    "--srt_file",
    type=str,
    default=None,
    help="Path to SRT subtitle file (for srt mode)"
)
parser.add_argument(
    "--media_file",
    type=str,
    default=None,
    help="Path to audio/video file (for srt mode)"
)

# YouTube mode arguments
parser.add_argument(
    "--youtube_url",
    type=str,
    default=None,
    help="YouTube video URL (for youtube mode)"
)
parser.add_argument(
    "--youtube_no_whisper_fallback",
    action="store_true",
    default=False,
    help="Don't use Whisper if YouTube video has no subtitles"
)
parser.add_argument(
    "--youtube_no_auto_update",
    action="store_true",
    default=False,
    help="Don't auto-update yt-dlp before downloading"
)

# Audio slicer arguments
parser.add_argument(
    "--use_slicer",
    action="store_true",
    default=False,
    help="Use audio slicer instead of Whisper VAD"
)
parser.add_argument(
    "--slicer_threshold",
    type=float,
    default=-40.0,
    help="RMS threshold in dB for silence detection (default: -40.0)"
)
parser.add_argument(
    "--slicer_min_length",
    type=int,
    default=5000,
    help="Minimum segment length in milliseconds (default: 5000)"
)

# Then in the main processing logic:

from utils.dataset_processor import process_dataset, ProcessingMode

# Determine mode
if args.processing_mode == "whisper":
    mode = ProcessingMode.WHISPER
    # ... existing logic
elif args.processing_mode == "srt":
    mode = ProcessingMode.SRT_MEDIA
    if not args.srt_file or not args.media_file:
        print("Error: --srt_file and --media_file required for srt mode")
        sys.exit(1)
    
    train_csv, eval_csv, duration = process_dataset(
        mode=mode,
        output_dir=output_path,
        language=args.lang,
        speaker_name=speaker_name,
        srt_path=args.srt_file,
        media_path=args.media_file
    )

elif args.processing_mode == "youtube":
    mode = ProcessingMode.YOUTUBE
    if not args.youtube_url:
        print("Error: --youtube_url required for youtube mode")
        sys.exit(1)
    
    train_csv, eval_csv, duration = process_dataset(
        mode=mode,
        output_dir=output_path,
        language=args.lang,
        speaker_name=speaker_name,
        youtube_url=args.youtube_url,
        use_whisper_fallback=not args.youtube_no_whisper_fallback,
        whisper_model=whisper_model
    )

elif args.processing_mode == "slicer":
    mode = ProcessingMode.AUDIO_SLICER
    # ... use slicer with existing audio files
```

### Headless Usage Examples

```bash
# SRT + Media
python headlessXttsTrain.py \
  --processing_mode srt \
  --srt_file subtitles.srt \
  --media_file video.mp4 \
  --lang en \
  --epochs 10

# YouTube
python headlessXttsTrain.py \
  --processing_mode youtube \
  --youtube_url "https://www.youtube.com/watch?v=..." \
  --lang en \
  --epochs 10

# Audio Slicer
python headlessXttsTrain.py \
  --processing_mode slicer \
  --input_audio long_audio.wav \
  --slicer_threshold -35 \
  --lang en \
  --epochs 10
```

---

## Testing

### Test Plan

**1. Test SRT Processing**
```powershell
# Prepare test files
# - Download a video with subtitles
# - Extract SRT file

# Test SRT processor standalone
python utils/srt_processor.py test.srt test.mp4 ./test_output en

# Check output
ls ./test_output/dataset/wavs/
cat ./test_output/dataset/metadata_train.csv
```

**2. Test YouTube Download**
```powershell
# Test YouTube downloader standalone
python utils/youtube_downloader.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" ./youtube_test en

# Check output
ls ./youtube_test/
```

**3. Test Audio Slicer**
```powershell
# Test slicer
python utils/audio_slicer.py test_audio.wav ./sliced_output
```

**4. Test Web UI**
```powershell
# Start web UI
python xtts_demo.py

# Navigate to http://127.0.0.1:5003
# Test each processing mode:
# - Upload SRT + video
# - Enter YouTube URL
# - Try audio slicer
# - Verify datasets are created
```

**5. Test Headless Mode**
```powershell
# Test SRT mode
python headlessXttsTrain.py --processing_mode srt --srt_file test.srt --media_file test.mp4 --lang en --epochs 2

# Test YouTube mode
python headlessXttsTrain.py --processing_mode youtube --youtube_url "URL" --lang en --epochs 2
```

---

## Documentation Updates

### 1. Update WARP.md

Add new section after "Essential Commands":

````markdown
## Advanced Dataset Processing

### SRT + Media Processing

Process existing subtitle files with audio/video:

```powershell
# Web UI: Select "SRT + Media File" mode in Tab 1

# Headless:
python headlessXttsTrain.py \
  --processing_mode srt \
  --srt_file subtitles.srt \
  --media_file video.mp4 \
  --lang en \
  --epochs 10
```

### YouTube Video Processing

Download and process YouTube videos automatically:

```powershell
# Web UI: Select "YouTube URL" mode in Tab 1

# Headless:
python headlessXttsTrain.py \
  --processing_mode youtube \
  --youtube_url "https://www.youtube.com/watch?v=..." \
  --lang en \
  --epochs 10
```

### Audio Slicer (Advanced)

Use RMS-based silence detection for precise segmentation:

```powershell
# Web UI: Select "Audio Slicer (Advanced)" mode in Tab 1

# Headless:
python headlessXttsTrain.py \
  --processing_mode slicer \
  --input_audio long_recording.wav \
  --slicer_threshold -35 \
  --lang en \
  --epochs 10
```

### Features Comparison

| Feature | Whisper (Default) | SRT + Media | YouTube | Audio Slicer |
|---------|-------------------|-------------|---------|--------------|
| Input | Audio files | SRT + Media | YouTube URL | Audio files |
| Transcription | Faster Whisper | Pre-made SRT | Auto/Manual subs | Faster Whisper |
| Segmentation | VAD-based | SRT timestamps | SRT timestamps | RMS-based |
| Best For | General use | Existing subtitles | Online content | Clean recordings |
| Speed | Fast | Very fast | Depends on video | Fast |
| Accuracy | High | Perfect (uses SRT) | High | High |
````

### 2. Update README.md

Add new section before "Changes in webui":

````markdown
## 🎯 Advanced Dataset Processing Features

### Multiple Input Methods

Create datasets from various sources:

#### 1️⃣ SRT + Media Files
- **Use existing subtitle files** (SRT format)
- Supports **audio and video** files
- Perfect timing from pre-made transcriptions
- Fast processing (no transcription needed)

```bash
# Example: Process video with SRT
python headlessXttsTrain.py --processing_mode srt \
  --srt_file interview.srt --media_file interview.mp4 --lang en
```

#### 2️⃣ YouTube Videos
- **Download directly from YouTube**
- Automatic subtitle extraction (manual or auto-generated)
- Falls back to Whisper if no subtitles available
- Auto-updates yt-dlp for compatibility

```bash
# Example: Train on YouTube video
python headlessXttsTrain.py --processing_mode youtube \
  --youtube_url "https://youtube.com/watch?v=..." --lang en
```

#### 3️⃣ Audio Slicer (Advanced)
- **RMS-based silence detection** from [audio-slicer](https://github.com/openvpi/audio-slicer)
- More precise than VAD for clean recordings
- Customizable silence threshold and segment length
- Integrated with Whisper transcription

```bash
# Example: Slice long audio file
python headlessXttsTrain.py --processing_mode slicer \
  --input_audio audiobook.wav --slicer_threshold -35 --lang en
```

### Feature Comparison

| Method | Best For | Speed | Accuracy |
|--------|----------|-------|----------|
| **Whisper (Default)** | General audio files | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| **SRT + Media** | Videos with subtitles | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| **YouTube** | Online content | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| **Audio Slicer** | Clean studio recordings | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
````

### 3. Update Memory Bank

Update `.warp/rules/memory-bank/context.md`:

```markdown
## Recent Work (As of January 2025)
1. Completed comprehensive WARP.md documentation
2. Created project constitution with architectural principles
3. Initialized memory bank with complete project overview
4. Established workflow integration with `.warp/` directory structure
5. **✨ Added advanced dataset processing features:**
   - SRT + media file processing
   - YouTube video download with subtitle extraction
   - RMS-based audio slicer integration
   - Unified processing pipeline
```

Update `.warp/rules/memory-bank/product.md`:

```markdown
## New Features (Advanced Dataset Processing)

### SRT + Media Processing
- Import existing subtitle files (SRT format)
- Automatic audio extraction from video
- Precise segment timing from SRT timestamps
- Supports all common video/audio formats

### YouTube Integration
- Direct download from YouTube URLs
- Automatic subtitle extraction (manual + auto-generated)
- Whisper fallback if no subtitles available
- Auto-updating yt-dlp for compatibility

### Audio Slicer
- RMS-based silence detection
- More precise than VAD for clean audio
- Customizable threshold and segment parameters
- Integrated with Whisper transcription
```

---

## Git Workflow

### Step-by-Step Commit Process

```powershell
# 1. Stage new files
git add utils/srt_processor.py
git add utils/youtube_downloader.py
git add utils/audio_slicer.py
git add utils/dataset_processor.py
git add IMPLEMENTATION_PLAN.md

# 2. Stage modified files
git add requirements.txt
git add xtts_demo.py
git add headlessXttsTrain.py
git add WARP.md
git add README.md
git add .warp/rules/memory-bank/context.md
git add .warp/rules/memory-bank/product.md

# 3. Check status
git status

# 4. Commit with detailed message
git commit -m "feat: Add advanced dataset processing features

- Add SRT + media file processing
  * Parse SRT subtitles with timestamps
  * Extract audio from video files
  * Create segments based on SRT timing
  * Module: utils/srt_processor.py

- Add YouTube download integration
  * Download videos with yt-dlp
  * Extract manual and auto-generated subtitles
  * Auto-update yt-dlp for compatibility
  * Whisper fallback for videos without subs
  * Module: utils/youtube_downloader.py

- Add RMS-based audio slicer
  * Integrate openvpi/audio-slicer algorithm
  * Silence detection and segmentation
  * Customizable threshold and parameters
  * Module: utils/audio_slicer.py

- Add unified processing pipeline
  * Support 4 processing modes (Whisper, SRT, YouTube, Slicer)
  * Consistent interface across modes
  * Module: utils/dataset_processor.py

- Update Web UI (xtts_demo.py)
  * Add mode selector in Tab 1
  * Add UI components for each mode
  * Integrated processing logic

- Update headless mode (headlessXttsTrain.py)
  * Add --processing_mode argument
  * Add mode-specific arguments
  * Support all new processing types

- Update documentation
  * Add advanced features section to WARP.md
  * Add feature comparison tables
  * Update README.md with examples
  * Update memory bank files
  * Create IMPLEMENTATION_PLAN.md guide

- Add dependencies
  * pysrt (SRT parsing)
  * yt-dlp (YouTube download)
  * soundfile (audio I/O)

Closes #[issue_number]"

# 5. Push to GitHub
git push origin main

# Or if working on a feature branch:
git checkout -b feature/advanced-dataset-processing
git push origin feature/advanced-dataset-processing
```

### Create Pull Request (if using branches)

```markdown
Title: feat: Advanced Dataset Processing (SRT, YouTube, Audio Slicer)

Description:
This PR adds three major dataset processing features:

## 🎯 Features Added

### 1. SRT + Media Processing
- Process existing subtitle files with audio/video
- Precise segment extraction based on SRT timestamps
- Supports all common video formats (MP4, MKV, AVI, etc.)
- Fast processing (no transcription needed)

### 2. YouTube Integration
- Direct download from YouTube URLs
- Automatic subtitle extraction
- Auto-updating yt-dlp
- Whisper fallback for videos without subtitles

### 3. Audio Slicer (Advanced)
- RMS-based silence detection
- Integration of openvpi/audio-slicer algorithm
- More precise than VAD for clean recordings
- Customizable parameters

## 📦 New Modules
- `utils/srt_processor.py`
- `utils/youtube_downloader.py`
- `utils/audio_slicer.py`
- `utils/dataset_processor.py`

## 🔧 Changes
- Updated Web UI with mode selector
- Updated headless mode with new arguments
- Updated documentation (WARP.md, README.md)
- Added new dependencies

## 🧪 Testing
- [x] SRT processing tested with sample videos
- [x] YouTube download tested with public videos
- [x] Audio slicer tested with long recordings
- [x] Web UI tested on Windows
- [x] Headless mode tested with all modes

## 📚 Documentation
- [x] Updated WARP.md
- [x] Updated README.md
- [x] Updated memory bank
- [x] Created IMPLEMENTATION_PLAN.md
```

---

## Implementation Checklist

Use this checklist to track implementation progress:

### Core Modules
- [x] `utils/srt_processor.py` - CREATED
- [ ] `utils/youtube_downloader.py` - Copy code from plan
- [ ] `utils/audio_slicer.py` - Copy code from plan
- [ ] `utils/dataset_processor.py` - Copy code from plan

### Dependencies
- [ ] Update `requirements.txt`
- [ ] Install new dependencies: `pip install pysrt yt-dlp soundfile`
- [ ] Test FFmpeg availability

### Web UI Updates
- [ ] Add mode selector to Tab 1
- [ ] Add SRT upload UI
- [ ] Add YouTube URL input
- [ ] Add audio slicer UI
- [ ] Update processing button handler
- [ ] Test all modes in Web UI

### Headless Mode Updates
- [ ] Add `--processing_mode` argument
- [ ] Add SRT mode arguments
- [ ] Add YouTube mode arguments
- [ ] Add slicer mode arguments
- [ ] Update main processing logic
- [ ] Test all modes in headless

### Testing
- [ ] Test SRT processing standalone
- [ ] Test YouTube download standalone
- [ ] Test audio slicer standalone
- [ ] Test Web UI (all 4 modes)
- [ ] Test headless mode (all 4 modes)
- [ ] Test on Windows
- [ ] Test with different languages
- [ ] Test error handling

### Documentation
- [ ] Update WARP.md (add advanced features section)
- [ ] Update README.md (add new features section)
- [ ] Update memory bank (context.md, product.md)
- [ ] Create usage examples
- [ ] Document new dependencies

### Git Operations
- [ ] Stage new files
- [ ] Stage modified files
- [ ] Create detailed commit message
- [ ] Push to GitHub
- [ ] (Optional) Create pull request

---

## Troubleshooting

### Common Issues

**1. FFmpeg not found**
```powershell
# Install FFmpeg on Windows
winget install FFmpeg

# Or download from: https://ffmpeg.org/download.html
# Add to PATH
```

**2. yt-dlp fails to download**
```powershell
# Update yt-dlp
python -m pip install -U yt-dlp

# Or use auto-update in the code (default)
```

**3. CUDA out of memory with WhisperX**
```python
# Use smaller Whisper model
model = WhisperModel("medium", device="cuda", compute_type="float16")

# Or use CPU
model = WhisperModel("large-v3", device="cpu", compute_type="int8")
```

**4. SRT parsing errors**
```powershell
# Check SRT file encoding (should be UTF-8)
# Re-save SRT file with UTF-8 encoding

# Or specify encoding in pysrt.open():
subs = pysrt.open(srt_path, encoding='utf-8')
```

**5. Audio slicer produces too many/few segments**
```python
# Adjust threshold (more negative = more sensitive)
slicer_threshold = -35.0  # Less sensitive
slicer_threshold = -45.0  # More sensitive

# Adjust minimum length
min_length = 7000  # Longer segments
min_length = 3000  # Shorter segments
```

---

## Next Steps

After completing this implementation:

1. **Test thoroughly** with different content types
2. **Gather user feedback** on new features
3. **Consider additions:**
   - Batch YouTube processing (playlists)
   - Custom audio filters/effects
   - Multi-language subtitle support
   - Dataset quality analysis
4. **Optimize performance:**
   - Parallel processing for multiple files
   - Caching for repeated operations
   - GPU optimization for slicing

---

## Support

For issues or questions:
- Check existing GitHub issues
- Create new issue with:
  - Processing mode used
  - Input file types
  - Error messages
  - Steps to reproduce

---

**Implementation Plan Version:** 1.0  
**Created:** January 7, 2025  
**Last Updated:** January 7, 2025  
**Status:** Ready for implementation
