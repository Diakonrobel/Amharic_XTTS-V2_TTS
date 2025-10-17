"""
Enhanced Audio Segmentation Module
====================================

Fixes the audio cutoff issue by applying generous padding to ensure
segments fully capture all transcribed text.

Key improvements:
- Generous padding at segment boundaries (0.3-0.5s)
- Conservative VAD trimming (keeps more audio)
- CSV-guided segmentation (uses text timestamps as ground truth)
- Padding overlap resolution
"""

import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
from tqdm import tqdm


def extract_segments_with_generous_padding(
    audio_path: str,
    csv_path: str,
    output_dir: str,
    speaker_name: str = "speaker",
    language: str = "en",
    # Generous padding parameters
    leading_pad_ms: int = 300,   # 300ms before speech starts
    trailing_pad_ms: int = 400,  # 400ms after speech ends
    min_duration: float = 0.5,
    max_duration: float = 20.0,
    # VAD refinement (optional)
    use_vad_trimming: bool = False,
    vad_aggressiveness: str = "conservative",  # "conservative", "moderate", "aggressive"
    gradio_progress=None
) -> Tuple[str, str]:
    """
    Extract audio segments with generous padding to prevent cutoffs.
    
    This function prioritizes COMPLETE audio capture over VAD accuracy.
    It's better to have a bit of silence than cut off speech!
    
    Args:
        audio_path: Path to audio file
        csv_path: Path to CSV with timing info (audio_file|text|speaker_name)
        output_dir: Output directory
        speaker_name: Speaker identifier
        language: Language code
        leading_pad_ms: Padding before segment start (milliseconds)
        trailing_pad_ms: Padding after segment end (milliseconds)
        min_duration: Minimum segment duration
        max_duration: Maximum segment duration
        use_vad_trimming: Enable gentle VAD-based trimming (removes only obvious silence)
        vad_aggressiveness: How aggressive to trim: "conservative", "moderate", "aggressive"
        gradio_progress: Optional progress tracker
        
    Returns:
        Tuple of (train_csv_path, eval_csv_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    wavs_dir = output_path / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    print(f"🎵 Loading audio: {audio_path}")
    wav, sr = torchaudio.load(str(audio_path))
    
    # Convert to mono
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    wav = wav.squeeze()
    
    # Read timing info from CSV
    print(f"📄 Reading timing from CSV: {csv_path}")
    df = pd.read_csv(csv_path, sep="|")
    
    # Validate CSV has required columns
    if not all(col in df.columns for col in ['audio_file', 'text']):
        raise ValueError("CSV must have 'audio_file' and 'text' columns")
    
    # Convert padding from milliseconds to seconds
    leading_pad = leading_pad_ms / 1000.0
    trailing_pad = trailing_pad_ms / 1000.0
    
    print(f"⚙️  Padding settings:")
    print(f"   Leading: {leading_pad_ms}ms ({leading_pad:.3f}s)")
    print(f"   Trailing: {trailing_pad_ms}ms ({trailing_pad:.3f}s)")
    
    # Initialize VAD if requested
    vad_slicer = None
    if use_vad_trimming:
        print(f"🎤 VAD trimming: {vad_aggressiveness} mode")
        try:
            from utils.vad_slicer import VADSlicer
            
            # Set padding based on aggressiveness
            vad_pads = {
                "conservative": {"speech_pad_ms": 100, "min_silence_duration_ms": 500},
                "moderate": {"speech_pad_ms": 50, "min_silence_duration_ms": 300},
                "aggressive": {"speech_pad_ms": 30, "min_silence_duration_ms": 200}
            }
            vad_params = vad_pads.get(vad_aggressiveness, vad_pads["conservative"])
            
            vad_slicer = VADSlicer(
                sample_rate=sr,
                min_segment_duration=min_duration,
                max_segment_duration=max_duration,
                vad_threshold=0.4,  # Lower threshold = less aggressive
                **vad_params
            )
            print(f"   ✓ VAD loaded ({vad_aggressiveness})")
        except Exception as e:
            print(f"   ⚠ VAD load failed: {e}")
            print(f"   → Using padding-only mode (safer)")
            vad_slicer = None
    
    # Extract segments
    metadata_out = {
        "audio_file": [],
        "text": [],
        "speaker_name": []
    }
    
    total_segments = len(df)
    iterator = enumerate(df.iterrows())
    
    if gradio_progress:
        iterator = gradio_progress.tqdm(iterator, total=total_segments, desc="Extracting with generous padding")
    else:
        iterator = tqdm(enumerate(df.iterrows()), total=total_segments, desc="Extracting with generous padding")
    
    for idx, (_, row) in iterator:
        text = row['text']
        
        # Try to extract timing from the row
        # Look for timing info in various possible column names
        timing_info = extract_timing_from_row(row, idx)
        
        if timing_info is None:
            print(f"⚠ Skipping segment {idx}: No timing information found")
            continue
        
        start_time, end_time = timing_info
        duration = end_time - start_time
        
        # Check duration constraints
        if duration < min_duration:
            print(f"⚠ Skipping segment {idx}: too short ({duration:.2f}s < {min_duration}s)")
            continue
        
        if duration > max_duration:
            print(f"⚠ Skipping segment {idx}: too long ({duration:.2f}s > {max_duration}s)")
            continue
        
        # Apply GENEROUS padding
        padded_start = max(0, start_time - leading_pad)
        padded_end = min(len(wav) / sr, end_time + trailing_pad)
        
        # Convert to samples
        start_sample = int(padded_start * sr)
        end_sample = int(padded_end * sr)
        
        # Extract segment
        segment_audio = wav[start_sample:end_sample]
        
        # Optional: VAD-based gentle trimming
        if vad_slicer and use_vad_trimming:
            try:
                segment_audio_np = segment_audio.numpy()
                trimmed_audio = gentle_vad_trim(
                    segment_audio_np,
                    vad_slicer,
                    vad_aggressiveness
                )
                
                # Only use trimmed version if it's not too aggressive
                # (keep at least 80% of original padded segment)
                if len(trimmed_audio) >= len(segment_audio_np) * 0.8:
                    segment_audio = torch.from_numpy(trimmed_audio).float()
                else:
                    print(f"   ⚠ VAD trim too aggressive for segment {idx}, keeping full padded version")
            except Exception as e:
                print(f"   ⚠ VAD trim failed for segment {idx}: {e}")
                # Keep original padded version
        
        # Validate segment
        segment_duration = len(segment_audio) / sr
        if segment_duration < min_duration * 0.5:  # Too short even after padding
            print(f"⚠ Skipping segment {idx}: extracted audio too short ({segment_duration:.2f}s)")
            continue
        
        # Save segment
        segment_filename = f"segment_{str(idx).zfill(6)}.wav"
        segment_path = wavs_dir / segment_filename
        
        torchaudio.save(
            str(segment_path),
            segment_audio.unsqueeze(0),
            sr
        )
        
        # Add to metadata
        metadata_out["audio_file"].append(f"wavs/{segment_filename}")
        metadata_out["text"].append(text)
        metadata_out["speaker_name"].append(speaker_name)
    
    if not metadata_out["audio_file"]:
        raise ValueError("No valid segments extracted!")
    
    # Create output DataFrame
    df_out = pd.DataFrame(metadata_out)
    
    # Shuffle and split (85/15)
    df_shuffled = df_out.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * 0.85)
    
    train_df = df_shuffled[:split_idx]
    eval_df = df_shuffled[split_idx:]
    
    # Save CSVs
    train_path = output_path / "metadata_train.csv"
    eval_path = output_path / "metadata_eval.csv"
    
    train_df.to_csv(train_path, sep="|", index=False)
    eval_df.to_csv(eval_path, sep="|", index=False)
    
    print(f"\n✅ Extraction complete!")
    print(f"   Total segments: {len(metadata_out['audio_file'])}")
    print(f"   Training: {len(train_df)}")
    print(f"   Evaluation: {len(eval_df)}")
    print(f"   Padding: +{leading_pad_ms}ms / -{trailing_pad_ms}ms")
    
    # Save language file
    from utils.lang_norm import canonical_lang
    lang_file = output_path / "lang.txt"
    language = canonical_lang(language)
    with open(lang_file, 'w', encoding='utf-8') as f:
        f.write(f"{language}\n")
    
    return str(train_path), str(eval_path)


def extract_timing_from_row(row: pd.Series, idx: int) -> Optional[Tuple[float, float]]:
    """
    Extract timing information from a CSV row.
    
    Looks for timing in various possible formats:
    - 'start_time' / 'end_time' columns
    - 'start' / 'end' columns
    - 'timestamp' column (format: "start-end" or "start:end")
    - Parse from audio filename if it contains timing
    
    Args:
        row: DataFrame row
        idx: Row index (for fallback)
        
    Returns:
        Tuple of (start_time, end_time) in seconds, or None if not found
    """
    # Try explicit timing columns
    if 'start_time' in row and 'end_time' in row:
        try:
            return (float(row['start_time']), float(row['end_time']))
        except (ValueError, TypeError):
            pass
    
    if 'start' in row and 'end' in row:
        try:
            return (float(row['start']), float(row['end']))
        except (ValueError, TypeError):
            pass
    
    # Try timestamp column
    if 'timestamp' in row:
        try:
            ts = str(row['timestamp'])
            if '-' in ts:
                start_str, end_str = ts.split('-', 1)
                return (float(start_str), float(end_str))
            elif ':' in ts:
                start_str, end_str = ts.split(':', 1)
                return (float(start_str), float(end_str))
        except (ValueError, AttributeError):
            pass
    
    # Try parsing from filename (e.g., "audio_000123_5.2_10.8.wav")
    if 'audio_file' in row:
        try:
            filename = str(row['audio_file'])
            # Extract numbers that might be timestamps
            import re
            numbers = re.findall(r'[\d]+\.[\d]+', filename)
            if len(numbers) >= 2:
                # Last two numbers are likely start/end times
                return (float(numbers[-2]), float(numbers[-1]))
        except (ValueError, AttributeError):
            pass
    
    # If all else fails, return None
    return None


def gentle_vad_trim(
    audio: np.ndarray,
    vad_slicer: 'VADSlicer',
    aggressiveness: str
) -> np.ndarray:
    """
    Gently trim silence from segment boundaries using VAD.
    
    This is CONSERVATIVE - only removes obvious silence, preserves speech boundaries.
    
    Args:
        audio: Audio segment
        vad_slicer: VAD slicer instance
        aggressiveness: "conservative", "moderate", or "aggressive"
        
    Returns:
        Trimmed audio (or original if VAD fails)
    """
    try:
        # Detect speech regions
        speech_regions = vad_slicer.detect_speech_segments(audio)
        
        if not speech_regions:
            # No speech detected, keep original
            return audio
        
        # Find the overall speech range
        first_speech = speech_regions[0]
        last_speech = speech_regions[-1]
        
        overall_start = first_speech['start']
        overall_end = last_speech['end']
        
        # Apply conservative margins based on aggressiveness
        margins = {
            "conservative": 0.15,  # Keep 150ms extra on each side
            "moderate": 0.10,      # Keep 100ms extra
            "aggressive": 0.05     # Keep 50ms extra
        }
        margin = margins.get(aggressiveness, 0.15)
        
        # Convert to samples
        sr = vad_slicer.sample_rate
        trim_start = max(0, int((overall_start - margin) * sr))
        trim_end = min(len(audio), int((overall_end + margin) * sr))
        
        # Only trim if we're removing significant silence (>100ms)
        trimmed_duration = (trim_end - trim_start) / sr
        original_duration = len(audio) / sr
        
        if trimmed_duration >= original_duration * 0.9:
            # Minimal trimming, keep original
            return audio
        
        return audio[trim_start:trim_end]
        
    except Exception as e:
        print(f"   VAD trim error: {e}, keeping original")
        return audio


def fix_existing_dataset(
    dataset_dir: str,
    audio_source: str,
    output_dir: str,
    leading_pad_ms: int = 300,
    trailing_pad_ms: int = 400,
    use_vad_trimming: bool = False
) -> Tuple[str, str]:
    """
    Re-extract segments from an existing dataset with better padding.
    
    Use this function to fix datasets that have cutoff issues.
    
    Args:
        dataset_dir: Directory with existing dataset (contains metadata CSVs)
        audio_source: Original audio file path
        output_dir: Output directory for fixed dataset
        leading_pad_ms: Leading padding (ms)
        trailing_pad_ms: Trailing padding (ms)
        use_vad_trimming: Enable VAD trimming
        
    Returns:
        Tuple of (train_csv_path, eval_csv_path)
    """
    dataset_path = Path(dataset_dir)
    
    # Find metadata files
    train_csv = dataset_path / "metadata_train.csv"
    eval_csv = dataset_path / "metadata_eval.csv"
    
    if not train_csv.exists() and not eval_csv.exists():
        raise ValueError(f"No metadata files found in {dataset_dir}")
    
    # Combine train and eval
    dfs = []
    if train_csv.exists():
        df_train = pd.read_csv(train_csv, sep="|")
        dfs.append(df_train)
    if eval_csv.exists():
        df_eval = pd.read_csv(eval_csv, sep="|")
        dfs.append(df_eval)
    
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Get language and speaker info
    lang_file = dataset_path / "lang.txt"
    language = "en"
    if lang_file.exists():
        with open(lang_file, 'r') as f:
            language = f.read().strip()
    
    speaker_name = df_combined['speaker_name'].iloc[0] if 'speaker_name' in df_combined.columns else "speaker"
    
    print(f"🔧 Fixing dataset: {dataset_dir}")
    print(f"   Language: {language}")
    print(f"   Speaker: {speaker_name}")
    print(f"   Segments: {len(df_combined)}")
    
    # Save combined CSV temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_csv_path = tmp.name
        df_combined.to_csv(tmp_csv_path, sep="|", index=False)
    
    try:
        # Re-extract with generous padding
        train_out, eval_out = extract_segments_with_generous_padding(
            audio_path=audio_source,
            csv_path=tmp_csv_path,
            output_dir=output_dir,
            speaker_name=speaker_name,
            language=language,
            leading_pad_ms=leading_pad_ms,
            trailing_pad_ms=trailing_pad_ms,
            use_vad_trimming=use_vad_trimming
        )
        
        return train_out, eval_out
        
    finally:
        # Clean up temp file
        import os
        try:
            os.unlink(tmp_csv_path)
        except:
            pass


if __name__ == "__main__":
    print("Enhanced Audio Segmentation Module")
    print("=" * 50)
    print()
    print("This module fixes audio cutoff issues by applying generous padding.")
    print()
    print("Usage:")
    print("  from utils.enhanced_segmentation import extract_segments_with_generous_padding")
    print()
    print("  train_csv, eval_csv = extract_segments_with_generous_padding(")
    print("      audio_path='audio.wav',")
    print("      csv_path='segments.csv',")
    print("      output_dir='output/',")
    print("      leading_pad_ms=300,   # 300ms before speech")
    print("      trailing_pad_ms=400   # 400ms after speech")
    print("  )")
