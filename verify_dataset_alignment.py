"""
Dataset Alignment Verification Tool
Checks if audio segments properly match their corresponding text.
"""

import os
import sys
import pandas as pd
import torchaudio
from pathlib import Path
import argparse


def verify_dataset_alignment(dataset_dir, sample_size=None, verbose=False):
    """
    Verify that audio segments match their text metadata.
    
    Args:
        dataset_dir: Path to dataset directory containing metadata and wavs
        sample_size: Number of samples to check (None = all)
        verbose: Print detailed information for each issue
        
    Returns:
        Dictionary with verification results
    """
    dataset_path = Path(dataset_dir)
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return None
    
    # Check for metadata files
    train_csv = dataset_path / "metadata_train.csv"
    eval_csv = dataset_path / "metadata_eval.csv"
    
    if not train_csv.exists():
        print(f"❌ Training metadata not found: {train_csv}")
        return None
    
    print(f"📊 Verifying dataset: {dataset_dir}\n")
    
    # Load metadata
    train_df = pd.read_csv(train_csv, sep='|')
    eval_df = pd.read_csv(eval_csv, sep='|') if eval_csv.exists() else pd.DataFrame()
    
    all_metadata = pd.concat([train_df, eval_df], ignore_index=True)
    
    total_samples = len(all_metadata)
    samples_to_check = min(sample_size, total_samples) if sample_size else total_samples
    
    print(f"Total samples: {total_samples}")
    print(f"Checking: {samples_to_check} samples")
    print(f"{'='*60}\n")
    
    # Check samples
    issues = {
        'missing_files': [],
        'duration_mismatch': [],
        'empty_audio': [],
        'text_mismatch': []
    }
    
    checked = 0
    for idx, row in all_metadata.head(samples_to_check).iterrows():
        checked += 1
        audio_file = row['audio_file']
        text = row['text']
        
        # Build full path
        audio_path = dataset_path / audio_file
        
        # Check if file exists
        if not audio_path.exists():
            issues['missing_files'].append({
                'index': idx,
                'file': audio_file,
                'text': text
            })
            if verbose:
                print(f"⚠️  [{idx}] Missing file: {audio_file}")
            continue
        
        try:
            # Load audio
            wav, sr = torchaudio.load(str(audio_path))
            duration = wav.shape[1] / sr
            
            # Check for empty/very short audio
            if duration < 0.1:
                issues['empty_audio'].append({
                    'index': idx,
                    'file': audio_file,
                    'duration': duration,
                    'text': text
                })
                if verbose:
                    print(f"⚠️  [{idx}] Empty audio: {audio_file} ({duration:.2f}s)")
                continue
            
            # Check duration vs text length
            words = len(text.split())
            
            # Heuristics for alignment check
            # Typical speech: 2-4 words per second
            # So duration should be roughly: words / 3 ± tolerance
            expected_duration = words / 3.0  # Assume 3 words/second
            min_expected = words * 0.15  # Very fast: ~6.7 words/sec
            max_expected = words * 1.5   # Very slow: ~0.67 words/sec
            
            if duration < min_expected or duration > max_expected:
                issues['duration_mismatch'].append({
                    'index': idx,
                    'file': audio_file,
                    'text': text,
                    'duration': duration,
                    'words': words,
                    'expected_duration': expected_duration,
                    'deviation': abs(duration - expected_duration)
                })
                if verbose:
                    print(f"⚠️  [{idx}] Duration mismatch: {audio_file}")
                    print(f"     Text: \"{text[:50]}...\"")
                    print(f"     Duration: {duration:.2f}s, Words: {words}, Expected: ~{expected_duration:.2f}s")
        
        except Exception as e:
            issues['text_mismatch'].append({
                'index': idx,
                'file': audio_file,
                'error': str(e)
            })
            if verbose:
                print(f"❌ [{idx}] Error processing: {audio_file}: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📋 VERIFICATION SUMMARY")
    print(f"{'='*60}\n")
    
    total_issues = sum(len(v) for v in issues.values())
    
    print(f"Samples checked: {checked}/{total_samples}")
    print(f"Total issues found: {total_issues}\n")
    
    if total_issues == 0:
        print("✅ All checks passed! Dataset appears correctly aligned.")
    else:
        print(f"⚠️  Found issues:\n")
        
        if issues['missing_files']:
            print(f"  📁 Missing files: {len(issues['missing_files'])}")
            if verbose:
                for issue in issues['missing_files'][:5]:
                    print(f"      - {issue['file']}")
        
        if issues['empty_audio']:
            print(f"  🔇 Empty/very short audio: {len(issues['empty_audio'])}")
            if verbose:
                for issue in issues['empty_audio'][:5]:
                    print(f"      - {issue['file']} ({issue['duration']:.2f}s)")
        
        if issues['duration_mismatch']:
            print(f"  ⏱️  Duration mismatches: {len(issues['duration_mismatch'])}")
            print(f"      (May indicate audio-text misalignment!)")
            if verbose:
                print(f"\n      Top 5 worst mismatches:")
                sorted_issues = sorted(issues['duration_mismatch'], 
                                     key=lambda x: x['deviation'], 
                                     reverse=True)
                for issue in sorted_issues[:5]:
                    print(f"      - {issue['file']}")
                    print(f"        Text: \"{issue['text'][:60]}...\"")
                    print(f"        Duration: {issue['duration']:.2f}s, "
                          f"Words: {issue['words']}, "
                          f"Expected: ~{issue['expected_duration']:.2f}s")
                    print()
        
        if issues['text_mismatch']:
            print(f"  ❌ Processing errors: {len(issues['text_mismatch'])}")
    
    print(f"\n{'='*60}")
    
    # Calculate alignment score
    if checked > 0:
        alignment_score = (checked - len(issues['duration_mismatch'])) / checked * 100
        print(f"\n📊 Estimated Alignment Quality: {alignment_score:.1f}%")
        
        if alignment_score < 70:
            print("\n🔥 CRITICAL: Dataset likely has severe alignment issues!")
            print("   Recommendation: Delete and reprocess with latest code.")
        elif alignment_score < 90:
            print("\n⚠️  WARNING: Dataset may have alignment issues.")
            print("   Recommendation: Review samples and consider reprocessing.")
        else:
            print("\n✅ Dataset alignment quality looks good!")
    
    return {
        'total_samples': total_samples,
        'checked': checked,
        'issues': issues,
        'alignment_score': alignment_score if checked > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(
        description='Verify audio-text alignment in TTS datasets'
    )
    parser.add_argument(
        'dataset_dir',
        type=str,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of samples to check (default: all)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed information for each issue'
    )
    
    args = parser.parse_args()
    
    results = verify_dataset_alignment(
        args.dataset_dir,
        sample_size=args.sample_size,
        verbose=args.verbose
    )
    
    if results is None:
        sys.exit(1)
    
    # Exit with error code if alignment is poor
    if results['alignment_score'] < 70:
        sys.exit(2)
    elif results['alignment_score'] < 90:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
