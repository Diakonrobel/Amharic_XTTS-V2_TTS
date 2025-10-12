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
from utils.lang_norm import canonical_lang

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


def merge_short_subtitles(
    segments: List[Tuple[float, float, str]],
    min_duration: float = 1.0,
    max_duration: float = 20.0,
    max_gap: float = 1.5
) -> List[Tuple[float, float, str]]:
    """
    AGGRESSIVE merging: Combines short subtitle segments into longer, natural segments.
    
    CRITICAL: When merging across gaps, we use the EARLIEST start time of any
    merged subtitle to ensure we capture all audio. This handles YouTube subtitle
    timing issues where text might be slightly ahead/behind actual speech.
    
    Merging strategy (in priority order):
    1. ALWAYS merge if current < min_duration (unless max_duration exceeded)
    2. Merge adjacent short segments even if current is OK
    3. Tolerate larger gaps for very short segments
    4. Preserve natural speech only when hitting limits
    
    Args:
        segments: List of (start, end, text) tuples
        min_duration: Minimum target duration (merge aggressively if shorter)
        max_duration: Maximum duration for merged segments
        max_gap: Maximum gap between segments to allow merging
        
    Returns:
        List of merged segments with (earliest_start, latest_end, combined_text)
    """
    if not segments:
        return []
    
    merged = []
    current_start, current_end, current_text = segments[0]
    
    for i in range(1, len(segments)):
        next_start, next_end, next_text = segments[i]
        
        current_duration = current_end - current_start
        next_duration = next_end - next_start
        gap = next_start - current_end
        combined_duration = next_end - current_start
        
        # AGGRESSIVE merging logic
        should_merge = False
        
        # Rule 1: ALWAYS merge if current is too short (unless would exceed max)
        if current_duration < min_duration:
            if combined_duration <= max_duration:
                # Allow larger gaps for very short segments
                adjusted_max_gap = max_gap * 2 if current_duration < 1.0 else max_gap
                if gap <= adjusted_max_gap:
                    should_merge = True
        
        # Rule 2: Merge if NEXT is too short (prevent orphans)
        elif next_duration < min_duration:
            if combined_duration <= max_duration and gap <= max_gap:
                should_merge = True
        
        # Rule 3: Merge very short adjacent segments regardless
        elif current_duration < 2.0 and next_duration < 2.0:
            if combined_duration <= max_duration and gap <= max_gap * 1.5:
                should_merge = True
        
        # Rule 4: Merge if gap is tiny (continuous speech)
        elif gap < 0.5:
            if combined_duration <= max_duration:
                should_merge = True
        
        if should_merge:
            # Merge: extend current segment to include next
            current_end = next_end
            current_text = current_text + ' ' + next_text
        else:
            # Don't merge: save current and start new
            merged.append((current_start, current_end, current_text))
            current_start, current_end, current_text = next_start, next_end, next_text
    
    # Add the last segment
    merged.append((current_start, current_end, current_text))
    
    print(f"Merged subtitles: {len(segments)} → {len(merged)} segments")
    print(f"  Reduction: {len(segments) - len(merged)} segments merged ({100*(len(segments)-len(merged))/len(segments):.1f}%)")
    print(f"  Average merged duration: {sum(e-s for s,e,_ in merged) / len(merged):.2f}s")
    
    return merged


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
    min_duration: float = 1.0,
    max_duration: float = 20.0,
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
        
        # CRITICAL FIX: Don't let previous segment block current segment's start time
        # The previous merged segment might end AFTER the current merged segment starts
        # (due to out-of-order merging or gaps). We must respect the current start time!
        
        # Start buffer: Go back by buffer amount, but check for actual overlaps
        desired_start = start_time - buffer
        
        if idx > 0:
            prev_end = srt_segments[idx - 1][1]
            # Only block if prev_end is AFTER our desired start (actual overlap)
            # AND the gap is small (< 0.3s = likely continuous speech)
            if prev_end > desired_start and (prev_end - desired_start) < 0.3:
                buffered_start = max(prev_end, 0)
            else:
                buffered_start = max(desired_start, 0)
        else:
            # First segment: can safely go back by buffer amount
            buffered_start = max(desired_start, 0)
        
        # End buffer: Extend by buffer amount, but don't overlap with next
        if idx < len(srt_segments) - 1:
            next_start = srt_segments[idx + 1][0]
            # Only enforce no-overlap if segments are actually close
            buffered_end = min(next_start, end_time + buffer)
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
        
        # Verify segment is not too short (at least 0.2 second)
        if segment.shape[0] < sr / 5:
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
    # Canonicalize language for dataset artifacts
    language = canonical_lang(language)
    with open(lang_file, 'w', encoding='utf-8') as f:
        f.write(f"{language}\n")
    
    return str(train_path), str(eval_path)


def process_srt_with_media(
    srt_path: str,
    media_path: str,
    output_dir: str,
    speaker_name: str = "speaker",
    language: str = "en",
    min_duration: float = 1.0,
    max_duration: float = 20.0,
    buffer: float = 0.2,
    gradio_progress=None
) -> Tuple[str, str, float]:
    """
    Complete pipeline: Process SRT file with corresponding media file.
    
    This function:
    1. Parses the SRT subtitle file
    2. Intelligently merges short subtitle segments (< 1s) into longer ones
    3. Extracts audio segments matching the (merged) subtitle timestamps
    4. Creates train/eval split with metadata CSVs
    
    Args:
        srt_path: Path to SRT subtitle file
        media_path: Path to audio or video file
        output_dir: Output directory
        speaker_name: Speaker identifier
        language: Language code
        min_duration: Minimum segment duration (after merging)
        max_duration: Maximum segment duration
        gradio_progress: Optional Gradio progress tracker
        
    Returns:
        Tuple of (train_csv_path, eval_csv_path, total_audio_duration)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Canonicalize language for dataset artifacts
    language = canonical_lang(language)
    
    # Parse SRT file
    print("Step 1: Parsing SRT file...")
    srt_segments = parse_srt_file(srt_path)
    
    if not srt_segments:
        raise ValueError("No segments found in SRT file")
    
    # Merge short subtitles with AGGRESSIVE settings for languages like Amharic
    print("Step 1b: Merging short subtitle segments...")
    srt_segments = merge_short_subtitles(
        srt_segments,
        min_duration=3.0,  # Merge segments shorter than 3 seconds (AGGRESSIVE)
        max_duration=max_duration,
        max_gap=3.0  # Allow gaps up to 3s when merging (AGGRESSIVE)
    )
    
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
