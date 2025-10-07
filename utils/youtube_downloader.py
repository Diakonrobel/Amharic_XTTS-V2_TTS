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
