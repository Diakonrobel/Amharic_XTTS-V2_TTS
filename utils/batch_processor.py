"""
Batch Processor Module
Handles batch processing of multiple YouTube URLs and files with dataset merging.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
import shutil


def parse_youtube_urls(input_text: str) -> List[str]:
    """
    Parse multiple YouTube URLs from comma-separated or line-separated input.
    
    Args:
        input_text: Input text with URLs
        
    Returns:
        List of cleaned YouTube URLs
    """
    # Split by commas, newlines, or spaces
    urls = re.split(r'[,\n\s]+', input_text.strip())
    
    # Filter out empty strings and validate YouTube URLs
    youtube_urls = []
    for url in urls:
        url = url.strip()
        if url and ('youtube.com' in url or 'youtu.be' in url):
            youtube_urls.append(url)
    
    return youtube_urls


def merge_datasets(
    dataset_paths: List[str],
    output_dir: str,
    remove_sources: bool = True
) -> Tuple[str, str, int]:
    """
    Merge multiple datasets into a single unified dataset.
    
    Args:
        dataset_paths: List of dataset directories to merge
        output_dir: Output directory for merged dataset
        remove_sources: Whether to remove source datasets after merging
        
    Returns:
        Tuple of (train_csv_path, eval_csv_path, total_segments)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    merged_wavs_dir = output_path / "wavs"
    merged_wavs_dir.mkdir(parents=True, exist_ok=True)
    
    all_train_data = []
    all_eval_data = []
    audio_counter = 0
    
    print(f"Merging {len(dataset_paths)} datasets...")
    
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            print(f"Warning: Dataset path does not exist: {dataset_path}")
            continue
        
        # Read metadata files
        train_csv = dataset_path / "metadata_train.csv"
        eval_csv = dataset_path / "metadata_eval.csv"
        
        if not train_csv.exists() or not eval_csv.exists():
            print(f"Warning: Metadata files not found in: {dataset_path}")
            continue
        
        # Load CSVs
        train_df = pd.read_csv(train_csv, sep='|')
        eval_df = pd.read_csv(eval_csv, sep='|')
        
        # Copy audio files and update paths
        wavs_dir = dataset_path / "wavs"
        if wavs_dir.exists():
            for _, row in train_df.iterrows():
                old_audio_file = dataset_path / row['audio_file']
                if old_audio_file.exists():
                    # Generate new filename
                    new_filename = f"merged_{str(audio_counter).zfill(8)}.wav"
                    new_audio_path = merged_wavs_dir / new_filename
                    
                    # Copy audio file
                    shutil.copy2(old_audio_file, new_audio_path)
                    
                    # Update row with new path
                    row['audio_file'] = f"wavs/{new_filename}"
                    all_train_data.append(row)
                    audio_counter += 1
            
            for _, row in eval_df.iterrows():
                old_audio_file = dataset_path / row['audio_file']
                if old_audio_file.exists():
                    new_filename = f"merged_{str(audio_counter).zfill(8)}.wav"
                    new_audio_path = merged_wavs_dir / new_filename
                    shutil.copy2(old_audio_file, new_audio_path)
                    row['audio_file'] = f"wavs/{new_filename}"
                    all_eval_data.append(row)
                    audio_counter += 1
        
        # Copy language file (use first one found)
        lang_file = dataset_path / "lang.txt"
        merged_lang_file = output_path / "lang.txt"
        if lang_file.exists() and not merged_lang_file.exists():
            shutil.copy2(lang_file, merged_lang_file)
    
    # Create merged DataFrames
    merged_train_df = pd.DataFrame(all_train_data)
    merged_eval_df = pd.DataFrame(all_eval_data)
    
    # Shuffle both
    merged_train_df = merged_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    merged_eval_df = merged_eval_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save merged metadata
    train_path = output_path / "metadata_train.csv"
    eval_path = output_path / "metadata_eval.csv"
    
    merged_train_df.to_csv(train_path, sep='|', index=False)
    merged_eval_df.to_csv(eval_path, sep='|', index=False)
    
    total_segments = len(merged_train_df) + len(merged_eval_df)
    
    print(f"✓ Merged dataset created:")
    print(f"  Training segments: {len(merged_train_df)}")
    print(f"  Evaluation segments: {len(merged_eval_df)}")
    print(f"  Total segments: {total_segments}")
    
    # Remove source datasets if requested
    if remove_sources:
        for dataset_path in dataset_paths:
            try:
                shutil.rmtree(dataset_path, ignore_errors=True)
                print(f"  Removed source: {dataset_path}")
            except Exception as e:
                print(f"  Warning: Could not remove {dataset_path}: {e}")
    
    return str(train_path), str(eval_path), total_segments


def process_youtube_batch(
    urls: List[str],
    transcript_lang: str,
    out_path: str,
    youtube_downloader,
    srt_processor,
    progress_callback=None
) -> Tuple[str, str, List[Dict]]:
    """
    Process multiple YouTube URLs and merge into single dataset.
    
    Args:
        urls: List of YouTube URLs
        transcript_lang: Language for transcripts
        out_path: Output base path
        youtube_downloader: YouTube downloader module
        srt_processor: SRT processor module
        progress_callback: Optional progress callback function
        
    Returns:
        Tuple of (train_csv, eval_csv, list of video info dicts)
    """
    temp_datasets = []
    video_infos = []
    
    print(f"🎬 Batch Processing {len(urls)} YouTube Videos...")
    
    for idx, url in enumerate(urls, 1):
        try:
            if progress_callback:
                progress_callback((idx - 1) / len(urls), 
                                desc=f"Processing video {idx}/{len(urls)}...")
            
            print(f"\n[{idx}/{len(urls)}] Processing: {url}")
            
            # Create temporary directory for this video
            temp_dir = tempfile.mkdtemp(prefix=f"yt_batch_{idx}_")
            
            # Download video
            audio_path, srt_path, info = youtube_downloader.download_youtube_video(
                url=url,
                output_dir=temp_dir,
                language=transcript_lang,
                audio_only=True,
                download_subtitles=True,
                auto_update=False  # Don't update for each video
            )
            
            if not audio_path or not srt_path:
                print(f"  ⚠ Skipping video {idx}: Download failed")
                continue
            
            # Process to temporary dataset
            temp_dataset_dir = os.path.join(out_path, f"temp_dataset_{idx}")
            os.makedirs(temp_dataset_dir, exist_ok=True)
            
            train_csv, eval_csv, duration = srt_processor.process_srt_with_media(
                srt_path=srt_path,
                media_path=audio_path,
                output_dir=temp_dataset_dir,
                language=transcript_lang
            )
            
            temp_datasets.append(temp_dataset_dir)
            video_infos.append({
                'url': url,
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'segments': len(pd.read_csv(train_csv, sep='|')) + len(pd.read_csv(eval_csv, sep='|'))
            })
            
            # Cleanup temp download directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
            
            print(f"  ✓ Video {idx} processed: {info.get('title', 'Unknown')}")
        
        except Exception as e:
            print(f"  ❌ Error processing video {idx}: {e}")
            continue
    
    if not temp_datasets:
        raise ValueError("No videos were successfully processed")
    
    # Merge all datasets
    if progress_callback:
        progress_callback(0.9, desc="Merging datasets...")
    
    final_dataset_dir = os.path.join(out_path, "dataset")
    train_csv, eval_csv, total_segments = merge_datasets(
        dataset_paths=temp_datasets,
        output_dir=final_dataset_dir,
        remove_sources=True
    )
    
    return train_csv, eval_csv, video_infos


def format_batch_summary(video_infos: List[Dict], total_segments: int) -> str:
    """
    Format summary of batch processing results.
    
    Args:
        video_infos: List of video info dictionaries
        total_segments: Total number of segments
        
    Returns:
        Formatted summary string
    """
    lines = ["✓ Batch Processing Complete!", "=" * 60]
    
    total_duration = sum(v['duration'] for v in video_infos)
    
    lines.append(f"\nProcessed {len(video_infos)} videos:")
    for i, info in enumerate(video_infos, 1):
        lines.append(f"\n{i}. {info['title']}")
        lines.append(f"   Duration: {info['duration']:.0f}s | Segments: {info['segments']}")
    
    lines.append("\n" + "=" * 60)
    lines.append(f"Total Videos: {len(video_infos)}")
    lines.append(f"Total Duration: {total_duration:.0f}s ({total_duration/60:.1f} minutes)")
    lines.append(f"Total Segments: {total_segments}")
    lines.append(f"Average Segments per Video: {total_segments/len(video_infos):.0f}")
    
    return "\n".join(lines)
