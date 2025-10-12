"""
Enhanced SRT Processor with VAD Integration
Combines SRT timestamps with VAD for more accurate segmentation.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm

from utils import srt_processor
from utils.vad_slicer import VADSlicer, AudioSegment
from utils.lang_norm import canonical_lang


def extract_segments_with_vad(
    audio_path: str,
    srt_segments: List[Tuple[float, float, str]],
    output_dir: str,
    speaker_name: str = "speaker",
    language: str = "en",
    min_duration: float = 0.5,
    max_duration: float = 15.0,
    use_vad_refinement: bool = True,
    vad_threshold: float = 0.5,
    gradio_progress=None
) -> Tuple[str, str]:
    """
    Extract audio segments using SRT timestamps with optional VAD refinement.
    
    This function improves upon basic SRT extraction by:
    1. Using VAD to detect actual speech boundaries within SRT segments
    2. Trimming silence at the start/end of segments
    3. Splitting long segments at natural pauses
    4. Merging very short adjacent segments
    
    Args:
        audio_path: Path to audio file
        srt_segments: List of (start, end, text) tuples from SRT
        output_dir: Output directory for segments and metadata
        speaker_name: Speaker identifier
        language: Language code
        min_duration: Minimum segment duration in seconds
        max_duration: Maximum segment duration in seconds
        use_vad_refinement: Enable VAD-based refinement
        vad_threshold: VAD confidence threshold
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
    audio_numpy = wav.numpy()
    
    # Initialize VAD slicer if enabled
    vad_slicer = None
    if use_vad_refinement:
        try:
            vad_slicer = VADSlicer(
                sample_rate=sr,
                min_segment_duration=min_duration,
                max_segment_duration=max_duration,
                vad_threshold=vad_threshold
            )
            print("✓ VAD refinement enabled")
        except Exception as e:
            print(f"⚠ VAD initialization failed: {e}")
            print("  Falling back to standard SRT extraction")
            vad_slicer = None
    
    metadata = {
        "audio_file": [],
        "text": [],
        "speaker_name": []
    }
    
    total_segments = len(srt_segments)
    iterator = enumerate(srt_segments)
    
    if gradio_progress is not None:
        iterator = gradio_progress.tqdm(iterator, total=total_segments, desc="Extracting segments with VAD")
    else:
        iterator = tqdm(enumerate(srt_segments), total=total_segments, desc="Extracting segments with VAD")
    
    for idx, (start_time, end_time, text) in iterator:
        duration = end_time - start_time
        
        # Filter by duration
        if duration < min_duration or duration > max_duration * 2:  # Allow 2x for VAD splitting
            continue
        
        # Extract segment with buffer
        buffer = 0.2
        buffered_start = max(0, start_time - buffer)
        buffered_end = min(len(wav) / sr, end_time + buffer)
        
        start_sample = int(buffered_start * sr)
        end_sample = int(buffered_end * sr)
        
        segment_audio = audio_numpy[start_sample:end_sample]
        
        if vad_slicer and use_vad_refinement:
            # Use VAD to refine this segment
            try:
                refined_segments = refine_segment_with_vad(
                    segment_audio=segment_audio,
                    segment_text=text,
                    segment_start_time=buffered_start,
                    vad_slicer=vad_slicer,
                    sr=sr,
                    min_duration=min_duration,
                    max_duration=max_duration
                )
                
                # Save each refined segment
                for sub_idx, refined in enumerate(refined_segments):
                    segment_filename = f"{Path(audio_path).stem}_{str(idx).zfill(6)}_{sub_idx}.wav"
                    segment_path = wavs_dir / segment_filename
                    
                    # Save audio
                    refined_tensor = torch.from_numpy(refined.audio).float()
                    torchaudio.save(
                        str(segment_path),
                        refined_tensor.unsqueeze(0),
                        sr
                    )
                    
                    # Add to metadata
                    metadata["audio_file"].append(f"wavs/{segment_filename}")
                    metadata["text"].append(refined.text if refined.text else text)
                    metadata["speaker_name"].append(speaker_name)
                
            except Exception as vad_error:
                print(f"  VAD refinement failed for segment {idx}: {vad_error}")
                # Fall back to original segment
                segment_filename = f"{Path(audio_path).stem}_{str(idx).zfill(6)}.wav"
                segment_path = wavs_dir / segment_filename
                
                segment_tensor = torch.from_numpy(segment_audio).float()
                torchaudio.save(str(segment_path), segment_tensor.unsqueeze(0), sr)
                
                metadata["audio_file"].append(f"wavs/{segment_filename}")
                metadata["text"].append(text)
                metadata["speaker_name"].append(speaker_name)
        else:
            # Standard extraction without VAD
            segment_filename = f"{Path(audio_path).stem}_{str(idx).zfill(6)}.wav"
            segment_path = wavs_dir / segment_filename
            
            segment_tensor = torch.from_numpy(segment_audio).float()
            torchaudio.save(str(segment_path), segment_tensor.unsqueeze(0), sr)
            
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
    
    print(f"\n✓ Extracted {len(metadata['audio_file'])} segments:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Evaluation: {len(eval_df)} samples")
    
# Save language file
    lang_file = output_path / "lang.txt"
    # Canonicalize language for dataset artifacts
    language = canonical_lang(language)
    with open(lang_file, 'w', encoding='utf-8') as f:
        f.write(f"{language}\n")
    
    return str(train_path), str(eval_path)


def refine_segment_with_vad(
    segment_audio: 'np.ndarray',
    segment_text: str,
    segment_start_time: float,
    vad_slicer: VADSlicer,
    sr: int,
    min_duration: float,
    max_duration: float
) -> List[AudioSegment]:
    """
    Refine a single SRT segment using VAD.
    
    This detects actual speech boundaries within the segment,
    trims silence, and optionally splits long segments.
    
    Args:
        segment_audio: Audio samples for this segment
        segment_text: Text from SRT
        segment_start_time: Start time of segment in original audio
        vad_slicer: VAD slicer instance
        sr: Sample rate
        min_duration: Minimum duration
        max_duration: Maximum duration
        
    Returns:
        List of refined AudioSegment objects
    """
    # Detect speech within this segment
    speech_regions = vad_slicer.detect_speech_segments(segment_audio)
    
    if not speech_regions:
        # No speech detected, return original
        return [AudioSegment(
            start_time=0,
            end_time=len(segment_audio) / sr,
            audio=segment_audio,
            text=segment_text,
            confidence=0.5
        )]
    
    # Merge very close regions
    merged_regions = vad_slicer.merge_short_segments(speech_regions, max_gap=0.3)
    
    # Split if too long
    if len(merged_regions) == 1 and merged_regions[0]['end'] - merged_regions[0]['start'] > max_duration:
        merged_regions = vad_slicer.split_long_segments(merged_regions, segment_audio)
    
    # Create refined segments
    refined_segments = []
    for region in merged_regions:
        start_sample = int(region['start'] * sr)
        end_sample = int(region['end'] * sr)
        
        # Ensure valid range
        start_sample = max(0, start_sample)
        end_sample = min(len(segment_audio), end_sample)
        
        region_audio = segment_audio[start_sample:end_sample]
        region_duration = (end_sample - start_sample) / sr
        
        # Filter by duration
        if region_duration < min_duration or region_duration > max_duration:
            continue
        
        refined_segments.append(AudioSegment(
            start_time=segment_start_time + region['start'],
            end_time=segment_start_time + region['end'],
            audio=region_audio,
            text=segment_text,  # Same text for all sub-segments
            confidence=region['confidence']
        ))
    
    # If no valid segments after filtering, return original
    if not refined_segments:
        return [AudioSegment(
            start_time=0,
            end_time=len(segment_audio) / sr,
            audio=segment_audio,
            text=segment_text,
            confidence=0.5
        )]
    
    return refined_segments


def process_srt_with_media_vad(
    srt_path: str,
    media_path: str,
    output_dir: str,
    speaker_name: str = "speaker",
    language: str = "en",
    min_duration: float = 0.5,
    max_duration: float = 15.0,
    use_vad_refinement: bool = True,
    vad_threshold: float = 0.5,
    gradio_progress=None
) -> Tuple[str, str, float]:
    """
    Complete pipeline: Process SRT file with VAD-enhanced extraction.
    
    Args:
        srt_path: Path to SRT subtitle file
        media_path: Path to audio or video file
        output_dir: Output directory
        speaker_name: Speaker identifier
        language: Language code
        min_duration: Minimum segment duration
        max_duration: Maximum segment duration
        use_vad_refinement: Enable VAD refinement
        vad_threshold: VAD confidence threshold
        gradio_progress: Optional Gradio progress tracker
        
    Returns:
        Tuple of (train_csv_path, eval_csv_path, total_audio_duration)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse SRT file
    print("Step 1: Parsing SRT file...")
    srt_segments = srt_processor.parse_srt_file(srt_path)
    
    if not srt_segments:
        raise ValueError("No segments found in SRT file")
    
    # Extract/convert audio (reuse existing logic)
    print("Step 2: Preparing audio...")
    media_ext = Path(media_path).suffix.lower()
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv']
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    if media_ext in video_extensions:
        temp_audio_path = output_path / f"{Path(media_path).stem}_audio.wav"
        if not srt_processor.extract_audio_from_video(media_path, str(temp_audio_path)):
            raise RuntimeError("Failed to extract audio from video")
        audio_path = str(temp_audio_path)
    elif media_ext in audio_extensions:
        if media_ext != '.wav':
            temp_audio_path = output_path / f"{Path(media_path).stem}_converted.wav"
            
            if not srt_processor.extract_audio_from_video(media_path, str(temp_audio_path)):
                raise RuntimeError("Failed to convert audio")
            audio_path = str(temp_audio_path)
        else:
            audio_path = media_path
    else:
        raise ValueError(f"Unsupported media format: {media_ext}")
    
    # Canonicalize language for dataset artifacts
    language = canonical_lang(language)
    # Extract segments with VAD
    mode_str = "VAD-enhanced" if use_vad_refinement else "standard"
    print(f"Step 3: Extracting audio segments ({mode_str})...")
    
    train_csv, eval_csv = extract_segments_with_vad(
        audio_path=audio_path,
        srt_segments=srt_segments,
        output_dir=output_dir,
        speaker_name=speaker_name,
        language=language,
        min_duration=min_duration,
        max_duration=max_duration,
        use_vad_refinement=use_vad_refinement,
        vad_threshold=vad_threshold,
        gradio_progress=gradio_progress
    )
    
    # Calculate total duration
    wav, sr = torchaudio.load(audio_path)
    total_duration = wav.shape[1] / sr
    
    print(f"\n✓ SRT processing complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Total audio duration: {total_duration:.2f} seconds")
    print(f"  VAD refinement: {'Enabled' if use_vad_refinement else 'Disabled'}")
    
    return train_csv, eval_csv, total_duration


if __name__ == "__main__":
    # Test standalone usage
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python srt_processor_vad.py <srt_file> <media_file> <output_dir> [language] [use_vad]")
        sys.exit(1)
    
    srt_file = sys.argv[1]
    media_file = sys.argv[2]
    output_dir = sys.argv[3]
    language = sys.argv[4] if len(sys.argv) > 4 else "en"
    use_vad = sys.argv[5].lower() == "true" if len(sys.argv) > 5 else True
    
    try:
        train_csv, eval_csv, duration = process_srt_with_media_vad(
            srt_path=srt_file,
            media_path=media_file,
            output_dir=output_dir,
            language=language,
            use_vad_refinement=use_vad
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
