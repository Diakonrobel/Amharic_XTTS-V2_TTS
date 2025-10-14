"""
Dataset Statistics Calculator
Calculates total duration, segments, and other stats from dataset metadata.
"""

import os
import pandas as pd
import librosa
from pathlib import Path
from typing import Dict, Tuple


def calculate_dataset_statistics(dataset_path: str) -> Dict:
    """
    Calculate comprehensive statistics for a dataset.
    
    Args:
        dataset_path: Path to dataset directory containing metadata CSVs
        
    Returns:
        Dictionary with statistics
    """
    dataset_path = Path(dataset_path)
    
    # Check if dataset exists
    if not dataset_path.exists():
        return {
            'error': 'Dataset directory does not exist',
            'exists': False
        }
    
    train_csv = dataset_path / "metadata_train.csv"
    eval_csv = dataset_path / "metadata_eval.csv"
    lang_file = dataset_path / "lang.txt"
    wavs_dir = dataset_path / "wavs"
    
    # Check required files
    if not train_csv.exists() or not eval_csv.exists():
        return {
            'error': 'Dataset metadata files not found (metadata_train.csv or metadata_eval.csv)',
            'exists': False
        }
    
    try:
        # Load CSVs
        train_df = pd.read_csv(train_csv, sep='|')
        eval_df = pd.read_csv(eval_csv, sep='|')
        
        # Calculate segment counts
        num_train = len(train_df)
        num_eval = len(eval_df)
        num_total = num_train + num_eval
        
        # Get language
        language = "unknown"
        if lang_file.exists():
            with open(lang_file, 'r', encoding='utf-8') as f:
                language = f.read().strip()
        
        # Calculate audio duration by loading audio files
        # For speed, we can estimate from the first few files or calculate exactly
        total_duration_seconds = 0.0
        train_duration = 0.0
        eval_duration = 0.0
        
        # Fast method: Calculate from audio files
        audio_files_exist = wavs_dir.exists()
        
        if audio_files_exist:
            # Calculate train duration
            for _, row in train_df.iterrows():
                audio_file = dataset_path / row['audio_file']
                if audio_file.exists():
                    try:
                        duration = librosa.get_duration(path=str(audio_file))
                        train_duration += duration
                    except Exception as e:
                        print(f"Warning: Could not get duration for {audio_file}: {e}")
            
            # Calculate eval duration
            for _, row in eval_df.iterrows():
                audio_file = dataset_path / row['audio_file']
                if audio_file.exists():
                    try:
                        duration = librosa.get_duration(path=str(audio_file))
                        eval_duration += duration
                    except Exception as e:
                        print(f"Warning: Could not get duration for {audio_file}: {e}")
            
            total_duration_seconds = train_duration + eval_duration
        
        # Convert durations to readable formats
        total_hours = total_duration_seconds / 3600
        total_minutes = total_duration_seconds / 60
        train_hours = train_duration / 3600
        eval_hours = eval_duration / 3600
        
        # Calculate average segment duration
        avg_segment_duration = total_duration_seconds / num_total if num_total > 0 else 0
        
        # Check audio files
        num_audio_files = 0
        if wavs_dir.exists():
            num_audio_files = len(list(wavs_dir.glob("*.wav")))
        
        return {
            'exists': True,
            'language': language,
            'segments': {
                'train': num_train,
                'eval': num_eval,
                'total': num_total
            },
            'duration': {
                'total_seconds': round(total_duration_seconds, 2),
                'total_minutes': round(total_minutes, 2),
                'total_hours': round(total_hours, 2),
                'train_hours': round(train_hours, 2),
                'eval_hours': round(eval_hours, 2),
                'avg_segment_seconds': round(avg_segment_duration, 2)
            },
            'audio_files': {
                'count': num_audio_files,
                'expected': num_total,
                'matches': num_audio_files == num_total
            },
            'quality': {
                'train_eval_ratio': round(num_train / num_eval, 2) if num_eval > 0 else 0,
                'recommended_train_eval_ratio': '85:15 (5.67:1)',
                'meets_minimum_duration': total_hours >= 10.0,
                'minimum_recommended_hours': 10.0
            }
        }
    
    except Exception as e:
        return {
            'error': f'Error calculating statistics: {str(e)}',
            'exists': True
        }


def format_statistics_display(stats: Dict) -> str:
    """
    Format statistics dictionary into a human-readable display string.
    
    Args:
        stats: Statistics dictionary from calculate_dataset_statistics
        
    Returns:
        Formatted string for display
    """
    if not stats.get('exists', False):
        return f"❌ {stats.get('error', 'Dataset not found')}"
    
    if 'error' in stats:
        return f"❌ {stats['error']}"
    
    # Build formatted output
    lines = []
    lines.append("📊 Dataset Statistics")
    lines.append("=" * 60)
    lines.append("")
    
    # Language
    lines.append(f"🌍 Language: {stats['language'].upper()}")
    lines.append("")
    
    # Segments
    lines.append("📝 Segments:")
    lines.append(f"  • Training: {stats['segments']['train']:,}")
    lines.append(f"  • Evaluation: {stats['segments']['eval']:,}")
    lines.append(f"  • Total: {stats['segments']['total']:,}")
    lines.append("")
    
    # Duration
    duration = stats['duration']
    lines.append("⏱️  Duration:")
    lines.append(f"  • Total: {duration['total_hours']:.2f} hours ({duration['total_minutes']:.1f} min)")
    lines.append(f"  • Training: {duration['train_hours']:.2f} hours")
    lines.append(f"  • Evaluation: {duration['eval_hours']:.2f} hours")
    lines.append(f"  • Avg Segment: {duration['avg_segment_seconds']:.1f}s")
    lines.append("")
    
    # Quality indicators
    quality = stats['quality']
    lines.append("✅ Quality Checks:")
    
    # Check minimum duration
    if quality['meets_minimum_duration']:
        lines.append(f"  ✓ Duration: {duration['total_hours']:.2f}h / {quality['minimum_recommended_hours']:.0f}h minimum ✅")
    else:
        remaining = quality['minimum_recommended_hours'] - duration['total_hours']
        lines.append(f"  ⚠ Duration: {duration['total_hours']:.2f}h / {quality['minimum_recommended_hours']:.0f}h minimum")
        lines.append(f"    Need {remaining:.2f} more hours for optimal training")
    
    # Check train/eval ratio
    ratio = quality['train_eval_ratio']
    if 5.0 <= ratio <= 6.5:
        lines.append(f"  ✓ Train/Eval Ratio: {ratio:.2f}:1 ✅")
    else:
        lines.append(f"  ⚠ Train/Eval Ratio: {ratio:.2f}:1 (recommended: ~5.67:1)")
    
    # Check audio files
    audio = stats['audio_files']
    if audio['matches']:
        lines.append(f"  ✓ Audio Files: {audio['count']:,} / {audio['expected']:,} ✅")
    else:
        lines.append(f"  ⚠ Audio Files: {audio['count']:,} / {audio['expected']:,} (mismatch!)")
    
    lines.append("")
    lines.append("=" * 60)
    
    # Training readiness
    if quality['meets_minimum_duration'] and audio['matches']:
        lines.append("🎉 Dataset is READY for training!")
    elif quality['meets_minimum_duration']:
        lines.append("⚠️  Dataset meets duration requirement but has audio file issues")
    else:
        percent = (duration['total_hours'] / quality['minimum_recommended_hours']) * 100
        lines.append(f"⏳ Dataset is {percent:.1f}% ready (need more audio for optimal training)")
    
    return "\n".join(lines)


def get_quick_stats(dataset_path: str) -> Tuple[int, float, str]:
    """
    Get quick statistics (segments, hours, language) without detailed calculations.
    Faster than calculate_dataset_statistics for UI display.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Tuple of (total_segments, total_hours, language)
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        return 0, 0.0, "unknown"
    
    train_csv = dataset_path / "metadata_train.csv"
    eval_csv = dataset_path / "metadata_eval.csv"
    lang_file = dataset_path / "lang.txt"
    
    if not train_csv.exists() or not eval_csv.exists():
        return 0, 0.0, "unknown"
    
    try:
        # Load CSVs
        train_df = pd.read_csv(train_csv, sep='|')
        eval_df = pd.read_csv(eval_csv, sep='|')
        total_segments = len(train_df) + len(eval_df)
        
        # Get language
        language = "unknown"
        if lang_file.exists():
            with open(lang_file, 'r', encoding='utf-8') as f:
                language = f.read().strip()
        
        # Estimate duration (fast method: average 10 seconds per segment)
        # For exact calculation, use calculate_dataset_statistics
        estimated_hours = (total_segments * 10.0) / 3600
        
        return total_segments, estimated_hours, language
    
    except Exception as e:
        print(f"Error getting quick stats: {e}")
        return 0, 0.0, "unknown"


if __name__ == "__main__":
    # CLI usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset_statistics.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    stats = calculate_dataset_statistics(dataset_path)
    display = format_statistics_display(stats)
    print(display)
