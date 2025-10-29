"""
Batch Processor Module
Handles batch processing of multiple YouTube URLs and files with dataset merging.
"""

import os
import re
import tempfile
import traceback
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
import shutil
from utils.lang_norm import canonical_lang
from utils.incremental_dataset_merger import merge_datasets_incremental


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
        # Strip surrounding quotes (single or double)
        url = url.strip('"\'')
        if url and ('youtube.com' in url or 'youtu.be' in url):
            youtube_urls.append(url)
    
    return youtube_urls


def parse_url_file(file_path: str) -> List[str]:
    """
    Parse YouTube URLs from uploaded text file.
    
    Args:
        file_path: Path to uploaded .txt file
        
    Returns:
        List of valid YouTube URLs
        
    Format:
        - One URL per line
        - Lines starting with # are comments (ignored)
        - Empty lines are ignored
        - Whitespace is trimmed
    """
    urls = []
    
    if not file_path:
        return urls
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Strip surrounding quotes (single or double) - handles Excel/CSV exports
                line = line.strip('"\'')
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Validate YouTube URL
                if 'youtube.com' in line or 'youtu.be' in line:
                    urls.append(line)
                else:
                    print(f"⚠ Line {line_num}: Invalid YouTube URL (skipped): {line[:50]}")
        
        print(f"✓ Loaded {len(urls)} URL(s) from file")
        return urls
        
    except Exception as e:
        print(f"❌ Error reading URL file: {e}")
        return []


def merge_datasets(
    dataset_paths: List[str],
    output_dir: str,
    remove_sources: bool = True,
    incremental: bool = False,
    check_duplicates: bool = True
) -> Tuple[str, str, int]:
    """
    Merge multiple datasets into a single unified dataset.
    
    Args:
        dataset_paths: List of dataset directories to merge
        output_dir: Output directory for merged dataset
        remove_sources: Whether to remove source datasets after merging
        incremental: If True, add to existing dataset; if False, create new dataset
        check_duplicates: If True, skip duplicate audio files (only used in incremental mode)
        
    Returns:
        Tuple of (train_csv_path, eval_csv_path, total_segments)
    """
    # Use incremental merger if requested
    if incremental:
        print(f"🔄 Using INCREMENTAL mode: Adding to existing dataset...")
        train_csv, eval_csv, total_segments, stats = merge_datasets_incremental(
            new_dataset_paths=dataset_paths,
            base_dataset_path=output_dir,
            check_duplicates=check_duplicates,
            keep_sources=not remove_sources
        )
        return train_csv, eval_csv, total_segments
    
    # Standard merge (create new dataset)
    print(f"📦 Using STANDARD mode: Creating new merged dataset...")
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
    progress_callback=None,
    incremental: bool = False,
    check_duplicates: bool = True,
    cookies_path: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,  # Deprecated: Use cookies.txt file instead
    proxy: Optional[str] = None,
    user_agent: Optional[str] = None,
    po_token: Optional[str] = None,
    visitor_data: Optional[str] = None,
    # Segmentation parameters
    buffer: float = 0.4,
    # VAD parameters
    use_vad: bool = False,
    vad_threshold: float = 0.5,
    vad_min_speech_ms: int = 250,
    vad_min_silence_ms: int = 300,
    vad_pad_ms: int = 30,
    use_enhanced_vad: bool = False,
    amharic_mode: bool = False,
    # Background music removal parameters
    remove_background_music: bool = False,
    background_removal_model: str = "htdemucs",
    background_removal_quality: str = "balanced",
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
        incremental: If True, add to existing dataset; if False, create new dataset
        check_duplicates: If True, skip duplicate audio files (only for incremental mode)
        cookies_path: Path to cookies file (Netscape format)
        cookies_from_browser: Browser name for cookies import (chrome|firefox|edge)
        proxy: Proxy URL (http(s)://user:pass@host:port)
        user_agent: Custom User-Agent string
        po_token: YouTube PO token for enhanced authentication
        visitor_data: YouTube visitor data for enhanced authentication
        use_vad: Enable VAD refinement for better quality
        vad_threshold: VAD confidence threshold (0-1)
        vad_min_speech_ms: Minimum speech duration in ms
        vad_min_silence_ms: Minimum silence duration in ms
        vad_pad_ms: Padding around speech in ms
        use_enhanced_vad: Use enhanced VAD with quality metrics
        amharic_mode: Enable Amharic-specific optimizations
        remove_background_music: Remove background music from audio before processing
        background_removal_model: Demucs model to use (htdemucs, mdx, mdx_extra)
        background_removal_quality: Quality preset (fast, balanced, best)
        
    Returns:
        Tuple of (train_csv, eval_csv, list of video info dicts)
    """
    temp_datasets = []
    video_infos = []
    failed_videos = []  # Track failed videos
    
    # Rate limiting to avoid YouTube bot detection
    # With cookies: ~2000 videos/hour = 1.8s/video minimum
    # We use 7 seconds for safety (514 videos/hour)
    RATE_LIMIT_DELAY = 7  # seconds between downloads
    
    print(f"🎬 Batch Processing {len(urls)} YouTube Videos...")
    print(f"⏱️ Rate Limiting: {RATE_LIMIT_DELAY}s delay between videos to prevent bot detection")
    print(f"📅 Estimated time: ~{(len(urls) * RATE_LIMIT_DELAY) / 60:.1f} minutes (+ download time)\n")
    
    for idx, url in enumerate(urls, 1):
        try:
            if progress_callback:
                progress_callback((idx - 1) / len(urls), 
                                desc=f"Processing video {idx}/{len(urls)}...")
            
            print(f"\n[{idx}/{len(urls)}] Processing: {url}")
            
            # Create temporary directory for this video
            temp_dir = tempfile.mkdtemp(prefix=f"yt_batch_{idx}_")
            
            # Download video with 2025 bypass methods
            audio_path, srt_path, info = youtube_downloader.download_and_process_youtube(
                url=url,
                output_dir=temp_dir,
                language=transcript_lang,
                use_whisper_if_no_srt=True,
                auto_update=False,  # Don't update for each video
                cookies_path=cookies_path,
                cookies_from_browser=cookies_from_browser,
                proxy=proxy,
                user_agent=user_agent,
                po_token=po_token,
                visitor_data=visitor_data,
                # Background music removal
                remove_background_music=remove_background_music,
                background_removal_model=background_removal_model,
                background_removal_quality=background_removal_quality,
            )
            
            if not audio_path or not srt_path:
                print(f"  ⚠ Skipping video {idx}: Download failed")
                continue
            
            # Process to temporary dataset
            temp_dataset_dir = os.path.join(out_path, f"temp_dataset_{idx}")
            os.makedirs(temp_dataset_dir, exist_ok=True)
            
            # CRITICAL: Silero VAD disabled due to text-audio mismatch bug
            # Always use standard SRT processor for reliable results
            if use_vad:
                print(f"  ⚠ VAD requested but disabled (known text-audio mismatch issue)")
                print(f"  ℹ Using standard SRT processing instead")
            
            train_csv, eval_csv, duration = srt_processor.process_srt_with_media(
                srt_path=srt_path,
                media_path=audio_path,
                output_dir=temp_dataset_dir,
                language=canonical_lang(transcript_lang),
                buffer=buffer
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
            
            # Rate limiting: Sleep between requests to avoid YouTube bot detection
            # Skip delay after last video
            if idx < len(urls):
                print(f"  ⏳ Waiting {RATE_LIMIT_DELAY}s before next video (YouTube rate limit protection)...")
                time.sleep(RATE_LIMIT_DELAY)
        
        except Exception as e:
            print(f"  ❌ Error processing video {idx}: {e}")
            print(f"  Full traceback:")
            traceback.print_exc()
            failed_videos.append({
                'index': idx,
                'url': url,
                'error': str(e)
            })
            continue
    
    # Print summary of failures if any
    if failed_videos:
        print(f"\n⚠️ {len(failed_videos)} video(s) failed to process:")
        for failed in failed_videos:
            print(f"  Video {failed['index']}: {failed['url']}")
            print(f"    Error: {failed['error']}")
    
    if not temp_datasets:
        raise ValueError("No videos were successfully processed")
    
    # Merge all datasets
    if progress_callback:
        progress_callback(0.9, desc="Merging datasets...")
    
    final_dataset_dir = os.path.join(out_path, "dataset")
    train_csv, eval_csv, total_segments = merge_datasets(
        dataset_paths=temp_datasets,
        output_dir=final_dataset_dir,
        remove_sources=True,
        incremental=incremental,
        check_duplicates=check_duplicates
    )
    
    return train_csv, eval_csv, video_infos


def pair_srt_with_media(srt_files: List[str], media_files: List[str]) -> List[Tuple[str, str]]:
    """
    Pair SRT files with media files based on filename similarity.
    
    Matches based on case-insensitive filename stems (without extension).
    Language suffixes (e.g., 'am', '2025am', 'en') are stripped before matching.
    
    Args:
        srt_files: List of SRT file paths
        media_files: List of media file paths
        
    Returns:
        List of tuples (srt_path, media_path) for matched pairs
    """
    import re
    
    pairs = []
    
    # Create lookup dict for media files by stem name
    media_lookup = {}
    for media_path in media_files:
        stem = Path(media_path).stem.lower()
        media_lookup[stem] = media_path
    
    # Match SRT files to media
    for srt_path in srt_files:
        srt_stem = Path(srt_path).stem.lower()
        
        # Try exact match first
        if srt_stem in media_lookup:
            pairs.append((srt_path, media_lookup[srt_stem]))
            print(f"✓ Paired: {Path(srt_path).name} <-> {Path(media_lookup[srt_stem]).name}")
        else:
            # Strip common language suffixes and try again
            # Matches patterns like: 2025am, am, en, 2024en, etc.
            # Strips from the end: optional year (2024-2099) + 2-3 letter language code
            stem_without_lang = re.sub(r'[._-]?(20\d{2})?(am|amh|en|eng|ar|es|fr|de|it|pt|ru|zh|ja|ko|hi|bn|ur|pa|sw|ha|yo|ig|zu|xh|so|om)$', '', srt_stem)
            
            if stem_without_lang != srt_stem and stem_without_lang in media_lookup:
                pairs.append((srt_path, media_lookup[stem_without_lang]))
                print(f"✓ Paired (language suffix stripped): {Path(srt_path).name} <-> {Path(media_lookup[stem_without_lang]).name}")
            else:
                print(f"⚠ No media file found for SRT: {Path(srt_path).name}")
    
    return pairs


def process_srt_media_batch(
    srt_files: List[str],
    media_files: List[str],
    language: str,
    out_path: str,
    srt_processor,
    progress_callback=None,
    incremental: bool = False,
    check_duplicates: bool = True,
    buffer: float = 0.4
) -> Tuple[str, str, List[Dict]]:
    """
    Process multiple SRT+media file pairs in batch mode and merge into single dataset.
    
    Args:
        srt_files: List of SRT file paths
        media_files: List of media file paths
        language: Dataset language
        out_path: Output base path
        srt_processor: SRT processor module
        progress_callback: Optional progress callback function
        incremental: If True, add to existing dataset; if False, create new dataset
        check_duplicates: If True, skip duplicate audio files (only for incremental mode)
        
    Returns:
        Tuple of (train_csv, eval_csv, list of file info dicts)
    """
    # Pair files
    pairs = pair_srt_with_media(srt_files, media_files)
    
    if not pairs:
        raise ValueError("No SRT-media pairs could be matched. Ensure filenames match (excluding extension).")
    
    temp_datasets = []
    file_infos = []
    
    print(f"📄 Batch Processing {len(pairs)} SRT-Media Pairs...")
    
    for idx, (srt_path, media_path) in enumerate(pairs, 1):
        try:
            if progress_callback:
                progress_callback((idx - 1) / len(pairs),
                                desc=f"Processing pair {idx}/{len(pairs)}...")
            
            print(f"\n[{idx}/{len(pairs)}] Processing: {Path(srt_path).name} + {Path(media_path).name}")
            
            # Process to temporary dataset
            temp_dataset_dir = os.path.join(out_path, f"temp_srt_dataset_{idx}")
            os.makedirs(temp_dataset_dir, exist_ok=True)
            
            train_csv, eval_csv, duration = srt_processor.process_srt_with_media(
                srt_path=srt_path,
                media_path=media_path,
                output_dir=temp_dataset_dir,
                language=canonical_lang(language),
                buffer=buffer
            )
            
            temp_datasets.append(temp_dataset_dir)
            
            # Count segments
            train_df = pd.read_csv(train_csv, sep='|')
            eval_df = pd.read_csv(eval_csv, sep='|')
            total_segs = len(train_df) + len(eval_df)
            
            file_infos.append({
                'srt_file': Path(srt_path).name,
                'media_file': Path(media_path).name,
                'duration': duration,
                'segments': total_segs
            })
            
            print(f"  ✓ Pair {idx} processed: {total_segs} segments, {duration:.1f}s")
        
        except Exception as e:
            print(f"  ❌ Error processing pair {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not temp_datasets:
        raise ValueError("No SRT-media pairs were successfully processed")
    
    # Merge all datasets
    if progress_callback:
        progress_callback(0.9, desc="Merging datasets...")
    
    final_dataset_dir = os.path.join(out_path, "dataset")
    train_csv, eval_csv, total_segments = merge_datasets(
        dataset_paths=temp_datasets,
        output_dir=final_dataset_dir,
        remove_sources=True,
        incremental=incremental,
        check_duplicates=check_duplicates
    )
    
    return train_csv, eval_csv, file_infos


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


def format_srt_batch_summary(file_infos: List[Dict], total_segments: int) -> str:
    """
    Format summary of SRT batch processing results.
    
    Args:
        file_infos: List of file info dictionaries
        total_segments: Total number of segments
        
    Returns:
        Formatted summary string
    """
    lines = ["✓ SRT Batch Processing Complete!", "=" * 60]
    
    total_duration = sum(f['duration'] for f in file_infos)
    
    lines.append(f"\nProcessed {len(file_infos)} SRT-Media pairs:")
    for i, info in enumerate(file_infos, 1):
        lines.append(f"\n{i}. {info['srt_file']} + {info['media_file']}")
        lines.append(f"   Duration: {info['duration']:.1f}s | Segments: {info['segments']}")
    
    lines.append("\n" + "=" * 60)
    lines.append(f"Total Pairs: {len(file_infos)}")
    lines.append(f"Total Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    lines.append(f"Total Segments: {total_segments}")
    lines.append(f"Average Segments per Pair: {total_segments/len(file_infos):.0f}")
    
    return "\n".join(lines)
