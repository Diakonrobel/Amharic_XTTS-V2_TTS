"""
Advanced Dataset Repair - Re-match Audio to Correct SRT Text
============================================================

This script attempts to fix mismatch by re-analyzing audio segments and
matching them to the correct SRT text.

REQUIREMENTS:
- Original SRT files for each video in dataset
- SRT files named same as audio file prefix (e.g., video_xyz.srt for video_xyz_*.wav)

APPROACH:
1. Group audio files by source video (extract from filename)
2. Load corresponding SRT file
3. Analyze each audio segment's content (duration, energy pattern)
4. Re-match to best SRT entry based on timing/characteristics
5. Update metadata with corrected text

Usage:
    python repair_dataset_with_srt_rematch.py --dataset_dir <path> --srt_dir <path>
"""

import os
import argparse
import pandas as pd
import torchaudio
import pysrt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import re


def parse_srt_file(srt_path: str) -> List[Tuple[float, float, str]]:
    """Parse SRT file and extract timing and text."""
    try:
        subs = pysrt.open(str(srt_path))
        segments = []
        
        for sub in subs:
            start = (sub.start.hours * 3600 +
                    sub.start.minutes * 60 +
                    sub.start.seconds +
                    sub.start.milliseconds / 1000.0)
            
            end = (sub.end.hours * 3600 +
                  sub.end.minutes * 60 +
                  sub.end.seconds +
                  sub.end.milliseconds / 1000.0)
            
            text = sub.text.replace('\n', ' ').strip()
            if text:
                segments.append((start, end, text))
        
        return segments
    except Exception as e:
        print(f"⚠ Error parsing {srt_path}: {e}")
        return []


def extract_video_source(filename: str) -> Optional[str]:
    """
    Extract video source name from audio filename.
    
    Examples:
        "video_xyz_000123_0.wav" -> "video_xyz"
        "my_video_000045.wav" -> "my_video"
    """
    # Remove extension
    name = Path(filename).stem
    
    # Try to find pattern: basename_NNNNNN or basename_NNNNNN_N
    # Remove numeric segments from the end
    match = re.match(r'^(.+?)_\d{6}(_\d+)?$', name)
    if match:
        return match.group(1)
    
    # Fallback: just use the first part before first number
    parts = re.split(r'_\d+', name)
    if parts:
        return parts[0]
    
    return None


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        wav, sr = torchaudio.load(audio_path)
        return wav.shape[1] / sr
    except:
        return 0.0


def find_best_srt_match(
    audio_duration: float,
    srt_segments: List[Tuple[float, float, str]],
    used_indices: set,
    min_duration_match: float = 0.5
) -> Tuple[Optional[str], Optional[int]]:
    """
    Find best matching SRT segment for an audio segment based on duration.
    
    Args:
        audio_duration: Duration of audio segment
        srt_segments: List of SRT segments
        used_indices: Indices already matched (to avoid duplicates)
        min_duration_match: Minimum duration similarity ratio
        
    Returns:
        Tuple of (matched_text, srt_index) or (None, None)
    """
    best_text = None
    best_idx = None
    best_similarity = 0.0
    
    for idx, (start, end, text) in enumerate(srt_segments):
        if idx in used_indices:
            continue  # Already matched
        
        srt_duration = end - start
        
        # Calculate duration similarity
        if srt_duration == 0:
            continue
        
        duration_ratio = min(audio_duration, srt_duration) / max(audio_duration, srt_duration)
        
        # Prefer segments with similar duration
        if duration_ratio > best_similarity:
            best_similarity = duration_ratio
            best_text = text
            best_idx = idx
    
    # Only return if similarity is good enough
    if best_similarity >= min_duration_match:
        return best_text, best_idx
    
    return None, None


def repair_dataset(
    dataset_dir: str,
    srt_dir: str,
    min_duration_match: float = 0.5,
    dry_run: bool = False
):
    """
    Repair dataset by re-matching audio segments to correct SRT text.
    
    Args:
        dataset_dir: Path to dataset directory
        srt_dir: Path to directory containing SRT files
        min_duration_match: Minimum duration similarity to match
        dry_run: Preview changes without modifying files
    """
    dataset_path = Path(dataset_dir)
    srt_path = Path(srt_dir)
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    if not srt_path.exists():
        print(f"❌ SRT directory not found: {srt_dir}")
        return
    
    print("=" * 70)
    print("Advanced Dataset Repair - SRT Re-matching")
    print("=" * 70)
    print(f"\nDataset: {dataset_dir}")
    print(f"SRT files: {srt_dir}")
    print(f"Min duration match: {min_duration_match}")
    print(f"Dry run: {dry_run}\n")
    
    # Find all SRT files
    srt_files = {}
    for srt_file in srt_path.glob("*.srt"):
        video_name = srt_file.stem
        srt_files[video_name] = srt_file
    
    print(f"📄 Found {len(srt_files)} SRT files:")
    for name in sorted(srt_files.keys())[:5]:
        print(f"   - {name}.srt")
    if len(srt_files) > 5:
        print(f"   ... and {len(srt_files) - 5} more")
    
    # Process metadata files
    for csv_name in ["metadata_train.csv", "metadata_eval.csv"]:
        csv_path = dataset_path / csv_name
        
        if not csv_path.exists():
            print(f"\n⚠ Skipping {csv_name} (not found)")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {csv_name}")
        print(f"{'='*70}")
        
        # Read metadata
        df = pd.read_csv(csv_path, sep="|")
        original_count = len(df)
        print(f"Original entries: {original_count}")
        
        # Backup
        if not dry_run:
            backup_path = dataset_path / f"{csv_name}.backup_rematch_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            df.to_csv(backup_path, sep="|", index=False)
            print(f"✓ Backup created: {backup_path.name}")
        
        # Group audio files by source video
        video_groups = defaultdict(list)
        for idx, row in df.iterrows():
            audio_file = row['audio_file']
            video_source = extract_video_source(audio_file)
            if video_source:
                video_groups[video_source].append((idx, row))
        
        print(f"\n📹 Grouped into {len(video_groups)} source videos:")
        for video_name in sorted(video_groups.keys())[:5]:
            print(f"   - {video_name}: {len(video_groups[video_name])} segments")
        
        # Re-match each group
        fixed_count = 0
        removed_count = 0
        rows_to_keep = []
        
        for video_name, segments in video_groups.items():
            # Find corresponding SRT file
            srt_file = srt_files.get(video_name)
            
            if not srt_file:
                print(f"\n⚠ No SRT file for '{video_name}' - keeping original text")
                rows_to_keep.extend([row for _, row in segments])
                continue
            
            # Parse SRT
            srt_segments = parse_srt_file(srt_file)
            if not srt_segments:
                print(f"\n⚠ Could not parse {srt_file.name} - keeping original text")
                rows_to_keep.extend([row for _, row in segments])
                continue
            
            print(f"\n🔄 Re-matching {video_name} ({len(segments)} segments, {len(srt_segments)} SRT entries)")
            
            # Sort segments by filename (approximate temporal order)
            segments_sorted = sorted(segments, key=lambda x: x[1]['audio_file'])
            
            used_srt_indices = set()
            
            for idx, row in segments_sorted:
                audio_file = row['audio_file']
                audio_path = dataset_path / audio_file
                
                if not audio_path.exists():
                    removed_count += 1
                    continue
                
                # Get audio duration
                audio_duration = get_audio_duration(str(audio_path))
                
                # Find best matching SRT
                new_text, srt_idx = find_best_srt_match(
                    audio_duration,
                    srt_segments,
                    used_srt_indices,
                    min_duration_match
                )
                
                if new_text:
                    # Mark SRT as used
                    used_srt_indices.add(srt_idx)
                    
                    # Update text if different
                    if new_text != row['text']:
                        row['text'] = new_text
                        fixed_count += 1
                    
                    rows_to_keep.append(row)
                else:
                    # No good match found - remove this segment
                    removed_count += 1
                    print(f"   ⚠ No match for {audio_file} (duration={audio_duration:.2f}s)")
        
        # Create new DataFrame
        if rows_to_keep:
            fixed_df = pd.DataFrame(rows_to_keep)
            
            print(f"\n📊 Summary for {csv_name}:")
            print(f"   Original: {original_count}")
            print(f"   Fixed text: {fixed_count}")
            print(f"   Removed: {removed_count}")
            print(f"   Final: {len(fixed_df)}")
            
            # Save
            if not dry_run:
                fixed_df.to_csv(csv_path, sep="|", index=False)
                print(f"✓ Saved repaired {csv_name}")
        else:
            print(f"\n⚠ No valid entries remaining for {csv_name}")
    
    print(f"\n{'='*70}")
    print("✅ Dataset repair complete!")
    print(f"{'='*70}")
    
    if dry_run:
        print("\n⚠ DRY RUN - No files were modified")


def main():
    parser = argparse.ArgumentParser(
        description="Repair dataset by re-matching audio to correct SRT text"
    )
    
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--srt_dir",
        type=str,
        required=True,
        help="Path to directory containing original SRT files"
    )
    
    parser.add_argument(
        "--min_duration_match",
        type=float,
        default=0.5,
        help="Minimum duration similarity ratio (0-1, default: 0.5)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview changes without modifying files"
    )
    
    args = parser.parse_args()
    
    repair_dataset(
        dataset_dir=args.dataset_dir,
        srt_dir=args.srt_dir,
        min_duration_match=args.min_duration_match,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
