"""
SRT Processor Module
Handles extraction of audio segments from media files based on SRT subtitle timestamps.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import pysrt
import torchaudio
import torch
import pandas as pd
from tqdm import tqdm

def parse_srt_file(srt_path: str) -> List[Tuple[float, float, str]]:
    """
    Parse SRT file and extract timing and text information.
    
    Args:
        srt_path: Path to SRT file
        
    Returns:
        List of tuples (start_time, end_time, text) in seconds
    """
    try:
        subs = pysrt.open(str(srt_path))
        segments = []
        
        for sub in subs:
            # Convert SubRipTime to seconds
            start = (sub.start.hours * 3600 +
                    sub.start.minutes * 60 +
                    sub.start.seconds +
                    sub.start.milliseconds / 1000.0)
            
            end = (sub.end.hours * 3600 +
                  sub.end.minutes * 60 +
                  sub.end.seconds +
                  sub.end.milliseconds / 1000.0)
            
            text = sub.text.replace('\n', ' ').strip()
            if text:  # Only include non-empty subtitles
                segments.append((start, end, text))
        
        print(f"Parsed {len(segments)} subtitle segments from {srt_path}")
        return segments
        
    except Exception as e:
        print(f"Error parsing SRT file {srt_path}: {e}")
        return []


def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
    """
    Extract audio from video file using FFmpeg.
    
    Args:
        video_path: Path to video file
        output_audio_path: Path to save extracted audio (WAV format)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", "22050",  # Sample rate for TTS
            "-ac", "1",  # Mono
            "-y",  # Overwrite
            str(output_audio_path)
        ]
        
        print(f"Extracting audio from {video_path}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
        
        print(f"Audio extracted to {output_audio_path}")
        return True
        
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg and add to PATH.")
        return False
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False


def extract_segments_from_audio(
    audio_path: str,
    srt_segments: List[Tuple[float, float, str]],
    output_dir: str,
    speaker_name: str = "speaker",
    language: str = "en",
    min_duration: float = 0.5,
    max_duration: float = 15.0,
    buffer: float = 0.2,
    gradio_progress=None
) -> Tuple[str, str]:
    """
    Extract audio segments based on SRT timestamps and create metadata CSVs.
    
    Args:
        audio_path: Path to audio file
        srt_segments: List of (start, end, text) tuples from SRT
        output_dir: Output directory for segments and metadata
        speaker_name: Speaker identifier
        min_duration: Minimum segment duration in seconds
        max_duration: Maximum segment duration in seconds
        gradio_progress: Optional Gradio progress tracker
        
    Returns:
        Tuple of (train_metadata_path, eval_metadata_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    wavs_dir = output_path / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load full audio
    print(f"Loading audio from {audio_path}...")
    wav, sr = torchaudio.load(str(audio_path))
    
    # Convert to mono if stereo
    if wav.size(0) != 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    
    wav = wav.squeeze()
    
    metadata = {
        "audio_file": [],
        "text": [],
        "speaker_name": []
    }
    
    total_segments = len(srt_segments)
    iterator = enumerate(srt_segments)
    
    if gradio_progress is not None:
        iterator = gradio_progress.tqdm(iterator, total=total_segments, desc="Extracting segments")
    else:
        iterator = tqdm(enumerate(srt_segments), total=total_segments, desc="Extracting segments")
    
    for idx, (start_time, end_time, text) in iterator:
        duration = end_time - start_time
        
        # Filter by duration
        if duration < min_duration or duration > max_duration:
            print(f"Skipping segment {idx+1}: duration {duration:.2f}s out of range")
            continue
        
        # CRITICAL FIX: Proper buffering that respects subtitle timing
        # The goal is to add a small buffer while preventing overlap between segments
        
        # Start buffer: Add buffer but don't overlap with previous segment
        if idx > 0:
            prev_end = srt_segments[idx - 1][1]
            # Ensure we don't go before previous segment ends (leave small gap)
            earliest_start = prev_end + 0.05  # 50ms gap minimum
            buffered_start = max(earliest_start, start_time - buffer)
        else:
            # First segment: can safely go back by buffer amount
            buffered_start = max(0, start_time - buffer)
        
        # End buffer: Add buffer but don't overlap with next segment
        if idx < len(srt_segments) - 1:
            next_start = srt_segments[idx + 1][0]
            # Ensure we don't go past next segment starts (leave small gap)
            latest_end = next_start - 0.05  # 50ms gap minimum
            buffered_end = min(latest_end, end_time + buffer)
        else:
            # Last segment: can safely extend by buffer amount
            buffered_end = min(len(wav) / sr, end_time + buffer)
        
        # Convert to samples
        start_sample = int(buffered_start * sr)
        end_sample = int(buffered_end * sr)
        
        # Ensure valid range
        start_sample = max(0, start_sample)
        end_sample = min(len(wav), end_sample)
        
        segment = wav[start_sample:end_sample]
        
        # Verify segment is not too short (at least 1/3 second)
        if segment.shape[0] < sr / 3:
            print(f"Skipping segment {idx+1}: extracted audio too short ({segment.shape[0]/sr:.2f}s)")
            continue
        
        # Save segment
        segment_filename = f"{Path(audio_path).stem}_{str(idx).zfill(6)}.wav"
        segment_path = wavs_dir / segment_filename
        
        torchaudio.save(
            str(segment_path),
            segment.unsqueeze(0),
            sr
        )
        
        # Add to metadata
        metadata["audio_file"].append(f"wavs/{segment_filename}")
        metadata["text"].append(text)
        metadata["speaker_name"].append(speaker_name)
    
    if not metadata["audio_file"]:
        raise ValueError("No valid segments extracted. Check duration filters and SRT content.")
    
    # Create DataFrame and save metadata
    df = pd.DataFrame(metadata)
    
    # Shuffle and split into train/eval (85/15)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * 0.85)
    
    train_df = df_shuffled[:split_idx]
    eval_df = df_shuffled[split_idx:]
    
    train_path = output_path / "metadata_train.csv"
    eval_path = output_path / "metadata_eval.csv"
    
    train_df.to_csv(train_path, sep="|", index=False)
    eval_df.to_csv(eval_path, sep="|", index=False)
    
    print(f"\nExtracted {len(metadata['audio_file'])} segments:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Evaluation: {len(eval_df)} samples")
    
    # Save language file with correct language
    lang_file = output_path / "lang.txt"
    with open(lang_file, 'w', encoding='utf-8') as f:
        f.write(f"{language}\n")
    
    return str(train_path), str(eval_path)


def process_srt_with_media(
    srt_path: str,
    media_path: str,
    output_dir: str,
    speaker_name: str = "speaker",
    language: str = "en",
    min_duration: float = 0.5,
    max_duration: float = 15.0,
    buffer: float = 0.2,
    gradio_progress=None
) -> Tuple[str, str, float]:
    """
    Complete pipeline: Process SRT file with corresponding media file.
    
    Args:
        srt_path: Path to SRT subtitle file
        media_path: Path to audio or video file
        output_dir: Output directory
        speaker_name: Speaker identifier
        language: Language code
        min_duration: Minimum segment duration
        max_duration: Maximum segment duration
        gradio_progress: Optional Gradio progress tracker
        
    Returns:
        Tuple of (train_csv_path, eval_csv_path, total_audio_duration)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse SRT file
    print("Step 1: Parsing SRT file...")
    srt_segments = parse_srt_file(srt_path)
    
    if not srt_segments:
        raise ValueError("No segments found in SRT file")
    
    # Check if media is video or audio
    media_ext = Path(media_path).suffix.lower()
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv']
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    if media_ext in video_extensions:
        # Extract audio from video
        print("Step 2: Extracting audio from video...")
        temp_audio_path = output_path / f"{Path(media_path).stem}_audio.wav"
        
        if not extract_audio_from_video(media_path, str(temp_audio_path)):
            raise RuntimeError("Failed to extract audio from video")
        
        audio_path = str(temp_audio_path)
    elif media_ext in audio_extensions:
        # Use audio directly (convert to WAV if needed)
        if media_ext != '.wav':
            print("Step 2: Converting audio to WAV format...")
            temp_audio_path = output_path / f"{Path(media_path).stem}_converted.wav"
            
            if not extract_audio_from_video(media_path, str(temp_audio_path)):
                raise RuntimeError("Failed to convert audio")
            
            audio_path = str(temp_audio_path)
        else:
            audio_path = media_path
    else:
        raise ValueError(f"Unsupported media format: {media_ext}")
    
    # Extract segments
    print("Step 3: Extracting audio segments based on SRT timestamps...")
    train_csv, eval_csv = extract_segments_from_audio(
        audio_path=audio_path,
        srt_segments=srt_segments,
        output_dir=output_dir,
        speaker_name=speaker_name,
        language=language,
        min_duration=min_duration,
        max_duration=max_duration,
        buffer=buffer,
        gradio_progress=gradio_progress
    )
    
    # Calculate total audio duration
    wav, sr = torchaudio.load(audio_path)
    total_duration = wav.shape[1] / sr
    
    print(f"\n✓ SRT processing complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Total audio duration: {total_duration:.2f} seconds")
    
    return train_csv, eval_csv, total_duration


if __name__ == "__main__":
    # Test standalone usage
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python srt_processor.py <srt_file> <media_file> <output_dir> [language]")
        sys.exit(1)
    
    srt_file = sys.argv[1]
    media_file = sys.argv[2]
    output_dir = sys.argv[3]
    language = sys.argv[4] if len(sys.argv) > 4 else "en"
    
    try:
        train_csv, eval_csv, duration = process_srt_with_media(
            srt_path=srt_file,
            media_path=media_file,
            output_dir=output_dir,
            language=language
        )
        
        print(f"\nSuccess!")
        print(f"  Train CSV: {train_csv}")
        print(f"  Eval CSV: {eval_csv}")
        print(f"  Duration: {duration:.2f}s")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
