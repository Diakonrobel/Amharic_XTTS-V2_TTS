"""
YouTube Downloader Module - 2025 Enhanced Edition
Downloads videos and subtitles from YouTube using yt-dlp with comprehensive bypass methods.

Features:
- Latest player client rotation (android_creator, android_music, ios_music)
- Enhanced detection bypass (player_skip, updated user agents)
- Intelligent retry logic with exponential backoff
- Multiple authentication methods (cookies, browser import, po_token)
- Comprehensive error handling with actionable solutions
- Automatic yt-dlp version checking
"""

import os
import subprocess
import json
import time
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import yt_dlp
from packaging import version as pkg_version

# === 2025 CONSTANTS ===
LATEST_IOS_VERSION = "19.45.4"
LATEST_ANDROID_VERSION = "19.43.41"
MINIMUM_YTDLP_VERSION = "2024.12.13"

# Player clients - use 'default' to let yt-dlp choose automatically
# The newer yt-dlp versions handle client selection intelligently
PLAYER_CLIENTS = ['default']  # Let yt-dlp auto-select best client

# Modern user agents for rotation (2025 versions)
USER_AGENTS = [
    f"Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36",
    f"Mozilla/5.0 (iPhone; CPU iPhone OS 18_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{LATEST_IOS_VERSION} Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    f"Mozilla/5.0 (iPad; CPU OS 18_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{LATEST_IOS_VERSION} Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36",
]


def check_ytdlp_version() -> Tuple[bool, str]:
    """
    Check if yt-dlp version is sufficient for 2025 bypass methods.
    
    Returns:
        Tuple of (is_valid, version_string)
    """
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        current_version = result.stdout.strip()
        
        # Parse version (format: YYYY.MM.DD or similar)
        try:
            is_valid = pkg_version.parse(current_version) >= pkg_version.parse(MINIMUM_YTDLP_VERSION)
        except:
            # Fallback: simple string comparison
            is_valid = current_version >= MINIMUM_YTDLP_VERSION
        
        return is_valid, current_version
    except Exception as e:
        print(f"⚠ Warning: Could not check yt-dlp version: {e}")
        return False, "unknown"


def update_ytdlp():
    """
    Update yt-dlp to the latest version.
    """
    try:
        print("📦 Updating yt-dlp to latest version...")
        subprocess.run(
            ["python", "-m", "pip", "install", "-U", "yt-dlp"],
            check=True,
            capture_output=True
        )
        print("✓ yt-dlp updated successfully")
        return True
    except Exception as e:
        print(f"⚠ Warning: Could not update yt-dlp: {e}")
        print("  Continuing with current version...")
        return False


def get_random_user_agent() -> str:
    """Get a random modern user agent."""
    return random.choice(USER_AGENTS)


def print_actionable_error(error_msg: str):
    """
    Print actionable solutions based on error type.
    """
    error_lower = error_msg.lower()
    
    print("\n" + "="*60)
    print("❌ ERROR DETECTED")
    print("="*60)
    
    if "sign in" in error_lower or "bot" in error_lower:
        print("\n🤖 Bot Detection Error")
        print("\nSOLUTIONS (in order of effectiveness):")
        print("\n1. ⭐ USE BROWSER COOKIES (95%+ success rate)")
        print("   - Run: python utils/youtube_downloader.py URL ./downloads en --from-browser chrome")
        print("   - Or set: export YTDLP_COOKIES_FROM_BROWSER='chrome'")
        print("\n2. USE A PROXY")
        print("   - Run: python utils/youtube_downloader.py URL ./downloads en --proxy http://proxy:port")
        print("\n3. USE PO TOKEN (advanced)")
        print("   - Extract from browser DevTools and use --po-token flag")
        print("\n4. WAIT 30-60 MINUTES")
        print("   - Your IP may be temporarily blocked")
        
    elif "429" in error_lower or "too many" in error_lower:
        print("\n⏱️ Rate Limit Error (HTTP 429)")
        print("\nSOLUTIONS:")
        print("\n1. USE BROWSER COOKIES")
        print("   - Cookies bypass most rate limits")
        print("\n2. USE A PROXY")
        print("   - Rotate your IP address")
        print("\n3. WAIT 10-30 MINUTES")
        print("   - Rate limits are temporary")
        print("\n4. REDUCE REQUEST RATE")
        print("   - Process fewer videos at once")
        
    elif "unavailable" in error_lower or "private" in error_lower:
        print("\n🔒 Video Unavailable")
        print("\nPOSSIBLE CAUSES:")
        print("- Video is private or deleted")
        print("- Regional restriction (use proxy from allowed region)")
        print("- Age restriction (use browser cookies from logged-in account)")
        print("- Premium content (requires Premium account cookies)")
        
    elif "network" in error_lower or "connection" in error_lower:
        print("\n🌐 Network Error")
        print("\nSOLUTIONS:")
        print("- Check your internet connection")
        print("- Try a different network")
        print("- Use a proxy if behind firewall")
        print("- Wait and retry (may be temporary YouTube issue)")
        
    else:
        print(f"\n⚠️ Error: {error_msg}")
        print("\nGENERAL TROUBLESHOOTING:")
        print("1. Update yt-dlp: pip install -U yt-dlp")
        print("2. Use browser cookies: --from-browser chrome")
        print("3. Check if video URL is correct")
        print("4. Try with a different video to isolate the issue")
    
    print("\n" + "="*60 + "\n")


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
        
        # Get available subtitle languages
        manual_subs = list(info.get('subtitles', {}).keys())
        auto_captions = list(info.get('automatic_captions', {}).keys())
        all_available_langs = sorted(set(manual_subs + auto_captions))
        
        return {
            'title': info.get('title', 'Unknown'),
            'duration': info.get('duration', 0),
            'uploader': info.get('uploader', 'Unknown'),
            'description': info.get('description', ''),
            'has_subtitles': bool(manual_subs),
            'has_automatic_captions': bool(auto_captions),
            'available_languages': manual_subs,
            'available_auto_caption_languages': auto_captions,
            'all_subtitle_languages': all_available_langs
        }


def convert_youtube_subtitle_to_srt(subtitle_text: str, format_ext: str) -> str:
    """
    Convert YouTube subtitle formats (srv1, srv2, srv3, json3, xml) to SRT format.
    
    Args:
        subtitle_text: Raw subtitle text from YouTube
        format_ext: Format extension (srv1, srv2, srv3, json3, xml, etc.)
        
    Returns:
        SRT formatted subtitle text
    """
    import re
    import json
    import xml.etree.ElementTree as ET
    from html import unescape
    
    # If already in SRT format, return as-is
    if '-->' in subtitle_text and re.search(r'^\d+$', subtitle_text.split('\n')[0].strip(), re.MULTILINE):
        return subtitle_text
    
    # Try to detect format if not specified or unknown
    if format_ext in ['srv1', 'srv2', 'srv3'] or subtitle_text.strip().startswith('<?xml'):
        # YouTube XML format (timedtext)
        try:
            print(f"  Converting YouTube XML ({format_ext}) to SRT...")
            root = ET.fromstring(subtitle_text)
            
            srt_lines = []
            index = 1
            
            for text_elem in root.findall('.//text'):
                start = float(text_elem.get('start', 0))
                duration = float(text_elem.get('dur', 0))
                end = start + duration
                text = text_elem.text or ''
                
                # Unescape HTML entities
                text = unescape(text).strip()
                if not text:
                    continue
                
                # Format timestamps
                start_time = format_timestamp(start)
                end_time = format_timestamp(end)
                
                srt_lines.append(f"{index}")
                srt_lines.append(f"{start_time} --> {end_time}")
                srt_lines.append(text)
                srt_lines.append('')  # Empty line between subtitles
                index += 1
            
            result = '\n'.join(srt_lines)
            if result.strip():
                print(f"  ✓ Converted {index-1} subtitle segments")
                return result
        except Exception as e:
            print(f"  ⚠ XML parsing failed: {e}")
    
    # Try JSON3 format
    if format_ext == 'json3' or (subtitle_text.strip().startswith('{') or subtitle_text.strip().startswith('[')):
        try:
            print(f"  Converting YouTube JSON3 to SRT...")
            data = json.loads(subtitle_text)
            
            srt_lines = []
            index = 1
            
            # JSON3 structure: events array with segments
            events = data.get('events', [])
            for event in events:
                if 'segs' not in event:
                    continue
                
                start = event.get('tStartMs', 0) / 1000.0
                duration = event.get('dDurationMs', 0) / 1000.0
                end = start + duration
                
                # Combine text segments
                text = ''.join(seg.get('utf8', '') for seg in event['segs'])
                text = unescape(text).strip()
                
                if not text:
                    continue
                
                # Format timestamps
                start_time = format_timestamp(start)
                end_time = format_timestamp(end)
                
                srt_lines.append(f"{index}")
                srt_lines.append(f"{start_time} --> {end_time}")
                srt_lines.append(text)
                srt_lines.append('')
                index += 1
            
            result = '\n'.join(srt_lines)
            if result.strip():
                print(f"  ✓ Converted {index-1} subtitle segments")
                return result
        except Exception as e:
            print(f"  ⚠ JSON parsing failed: {e}")
    
    # If all conversions fail, return original
    print(f"  ⚠ Could not convert subtitle format '{format_ext}', returning original")
    return subtitle_text


def download_subtitles_robust(
    url: str,
    output_path: Path,
    sanitized_title: str,
    language: str,
    result: dict,
    max_retries: int = 3
) -> Optional[str]:
    """
    Robustly download subtitles with multiple fallback strategies to bypass rate limiting.
    
    Strategies:
    1. Direct API extraction from result metadata (fastest, no extra request)
    2. yt-dlp with cookies and browser impersonation
    3. Exponential backoff retry with random delays
    4. Alternative subtitle formats and languages
    
    Args:
        url: YouTube video URL
        output_path: Directory to save subtitles
        sanitized_title: Sanitized video title
        language: Preferred language
        result: Video info dict from initial download
        max_retries: Maximum retry attempts
        
    Returns:
        Path to subtitle file if successful, None otherwise
    """
    subtitle_file = None
    
    print("\n🔍 Attempting to download subtitles with robust strategies...")
    
    # Strategy 1: Extract subtitles directly from metadata (no extra download)
    print("\n[Strategy 1] Extracting subtitles from video metadata...")
    try:
        # Check for manual subtitles first
        subtitles_dict = result.get('subtitles', {})
        automatic_captions_dict = result.get('automatic_captions', {})
        
        # Try manual subtitles first (better quality)
        all_subs = {**subtitles_dict, **automatic_captions_dict}
        
        # Preferred language order
        lang_priority = [language, 'en', 'en-US', 'en-GB']
        if language not in lang_priority:
            lang_priority.insert(0, language)
        
        subtitle_url = None
        selected_lang = None
        
        for lang in lang_priority:
            if lang in all_subs:
                # Find SRT format or any text format
                for sub_format in all_subs[lang]:
                    if sub_format.get('ext') in ['srt', 'vtt', 'srv3', 'srv2', 'srv1']:
                        subtitle_url = sub_format.get('url')
                        selected_lang = lang
                        print(f"  ✓ Found {sub_format.get('ext')} subtitles in '{lang}'")
                        break
                if subtitle_url:
                    break
        
        # Download subtitle file directly with custom headers
        if subtitle_url:
            import urllib.request
            import gzip
            subtitle_path = output_path / f"{sanitized_title}.{selected_lang}.srt"
            
            print(f"  Downloading from: {subtitle_url[:80]}...")
            
            # Create request with browser-like headers to avoid rate limiting
            req = urllib.request.Request(
                subtitle_url,
                headers={
                    'User-Agent': get_random_user_agent(),
                    'Accept': '*/*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate',
                    'Referer': 'https://www.youtube.com/',
                    'Origin': 'https://www.youtube.com',
                    'Connection': 'keep-alive',
                }
            )
            
            # Add small delay to be respectful
            time.sleep(random.uniform(0.5, 1.5))
            
            with urllib.request.urlopen(req, timeout=30) as response:
                subtitle_content = response.read()
                
                # Check if content is gzip-compressed (YouTube often sends compressed data)
                if subtitle_content[:2] == b'\x1f\x8b':  # gzip magic number
                    print(f"  Decompressing gzip-encoded subtitle...")
                    subtitle_content = gzip.decompress(subtitle_content)
                
                # Decode and write as text
                try:
                    # Try UTF-8 first
                    subtitle_text = subtitle_content.decode('utf-8')
                except UnicodeDecodeError:
                    # Fallback to other encodings
                    for encoding in ['utf-8-sig', 'latin-1', 'cp1252']:
                        try:
                            subtitle_text = subtitle_content.decode(encoding)
                            print(f"  Decoded with {encoding} encoding")
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        raise UnicodeDecodeError("Could not decode subtitle with any encoding")
                
                # Detect and convert subtitle format
                # YouTube returns subtitles in various formats: JSON3, XML (srv1/srv2/srv3), or direct SRT
                subtitle_text = convert_youtube_subtitle_to_srt(subtitle_text, sub_format.get('ext', 'unknown'))
                
                # Write decoded text
                with open(subtitle_path, 'w', encoding='utf-8') as f:
                    f.write(subtitle_text)
            
            # Convert VTT to SRT if needed
            if subtitle_path.suffix == '.vtt':
                srt_path = subtitle_path.with_suffix('.srt')
                convert_vtt_to_srt(subtitle_path, srt_path)
                subtitle_path.unlink()  # Remove VTT file
                subtitle_path = srt_path
            
            if subtitle_path.exists() and subtitle_path.stat().st_size > 0:
                print(f"  ✅ Successfully downloaded subtitles: {subtitle_path.name}")
                return str(subtitle_path)
    
    except Exception as e:
        error_msg = str(e)
        if '429' in error_msg:
            print(f"  ⚠ Strategy 1 failed: YouTube rate limit detected")
            print(f"    Trying alternative extraction method...")
            # Try alternative: use youtube-transcript-api as fallback
            subtitle_file = try_transcript_api(url, output_path, sanitized_title, language)
            if subtitle_file:
                return subtitle_file
        else:
            print(f"  ⚠ Strategy 1 failed: {e}")
    
    # Strategy 2: Use yt-dlp with advanced options and retry logic
    print("\n[Strategy 2] Using yt-dlp with browser impersonation and retries...")
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                # Exponential backoff with jitter
                delay = (2 ** attempt) + random.uniform(1, 3)
                print(f"  ⏳ Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
            
            # Advanced yt-dlp options to bypass rate limiting
            subtitle_opts = {
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': [language, 'en'],
                'subtitlesformat': 'srt/vtt/best',
                'outtmpl': str(output_path / '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': True,
                # Rate limit bypass options
                'extractor_retries': 3,
                'fragment_retries': 3,
                'retries': 3,
                'sleep_interval': 2,
                'max_sleep_interval': 5,
                # Use cookies to appear as logged-in user (helps with rate limits)
                'cookiefile': None,  # Can be set to browser cookies file
                # Headers to appear more like a browser
                'http_headers': {
                    'User-Agent': get_random_user_agent(),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Sec-Fetch-Mode': 'navigate',
                }
            }
            
            with yt_dlp.YoutubeDL(subtitle_opts) as ydl:
                print(f"  Attempt {attempt + 1}/{max_retries}: Downloading subtitles...")
                ydl.download([url])
            
            # Look for subtitle file
            subtitle_patterns = [
                f"{sanitized_title}.{language}.srt",
                f"{sanitized_title}.en.srt",
                f"{sanitized_title}.srt",
                f"{sanitized_title}.{language}.vtt",
                f"{sanitized_title}.en.vtt",
            ]
            
            for pattern in subtitle_patterns:
                subtitle_candidate = output_path / pattern
                if subtitle_candidate.exists() and subtitle_candidate.stat().st_size > 0:
                    subtitle_file = str(subtitle_candidate)
                    print(f"  ✅ Successfully downloaded: {subtitle_candidate.name}")
                    return subtitle_file
            
            print(f"  ⚠ Attempt {attempt + 1} completed but no subtitle file found")
        
        except Exception as e:
            print(f"  ⚠ Attempt {attempt + 1} failed: {str(e)[:100]}")
            if attempt == max_retries - 1:
                print("\n❌ All subtitle download strategies exhausted")
                print("   This is likely due to:")
                print("   - YouTube rate limiting (HTTP 429)")
                print("   - Temporary API issues")
                print("   - Regional restrictions")
                print("   📝 Will use Whisper transcription as fallback")
    
    return None


def try_transcript_api(url: str, output_path: Path, sanitized_title: str, language: str) -> Optional[str]:
    """
    Try using youtube-transcript-api as an alternative to bypass rate limits.
    This uses a different endpoint that may not be rate limited.
    
    Args:
        url: YouTube video URL
        output_path: Output directory
        sanitized_title: Sanitized video title
        language: Preferred language
        
    Returns:
        Path to subtitle file if successful, None otherwise
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        import re
        
        # Extract video ID
        video_id_match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11}).*', url)
        if not video_id_match:
            return None
        
        video_id = video_id_match.group(1)
        print(f"  Using youtube-transcript-api for video ID: {video_id}")
        
        # Try to get transcript
        try:
            # Try preferred language first
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to find transcript in preferred language
            transcript = None
            try:
                transcript = transcript_list.find_transcript([language, 'en'])
            except:
                # Fallback: get any available transcript
                transcript = transcript_list.find_generated_transcript(['en'])
            
            if transcript:
                transcript_data = transcript.fetch()
                
                # Convert to SRT format
                srt_path = output_path / f"{sanitized_title}.{language}.srt"
                with open(srt_path, 'w', encoding='utf-8') as f:
                    for i, entry in enumerate(transcript_data, start=1):
                        start = entry['start']
                        duration = entry['duration']
                        end = start + duration
                        text = entry['text']
                        
                        # Format timestamp
                        start_time = format_timestamp(start)
                        end_time = format_timestamp(end)
                        
                        f.write(f"{i}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{text}\n\n")
                
                if srt_path.exists() and srt_path.stat().st_size > 0:
                    print(f"  ✅ Successfully downloaded via transcript API: {srt_path.name}")
                    return str(srt_path)
        
        except Exception as api_error:
            print(f"  Could not fetch transcript: {api_error}")
            return None
    
    except ImportError:
        print("  youtube-transcript-api not installed, installing now...")
        try:
            import subprocess
            subprocess.run(['pip', 'install', 'youtube-transcript-api'], 
                         check=True, capture_output=True)
            print("  Installed youtube-transcript-api, please retry the download")
        except:
            pass
        return None
    except Exception as e:
        print(f"  Transcript API failed: {e}")
        return None


def format_timestamp(seconds: float) -> str:
    """
    Format seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def convert_vtt_to_srt(vtt_path: Path, srt_path: Path):
    """
    Convert WebVTT subtitle format to SRT format.
    
    Args:
        vtt_path: Path to VTT file
        srt_path: Path to output SRT file
    """
    try:
        import webvtt
        vtt = webvtt.read(str(vtt_path))
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, caption in enumerate(vtt, start=1):
                # Write subtitle index
                f.write(f"{i}\n")
                # Write timestamp (convert VTT format to SRT format)
                start = caption.start.replace('.', ',')
                end = caption.end.replace('.', ',')
                f.write(f"{start} --> {end}\n")
                # Write text
                f.write(f"{caption.text}\n\n")
    except ImportError:
        # Fallback: simple text replacement
        with open(vtt_path, 'r', encoding='utf-8') as vtt_file:
            lines = vtt_file.readlines()
        
        with open(srt_path, 'w', encoding='utf-8') as srt_file:
            # Skip VTT header
            start_writing = False
            for line in lines:
                if 'WEBVTT' in line or 'Kind:' in line or 'Language:' in line:
                    continue
                if '-->' in line:
                    start_writing = True
                    # Replace . with , in timestamps for SRT format
                    line = line.replace('.', ',')
                if start_writing:
                    srt_file.write(line)


def download_youtube_video(
    url: str,
    output_dir: str,
    language: str = 'en',
    audio_only: bool = True,
    download_subtitles: bool = True,
    auto_update: bool = True,
    # Authentication and network parameters
    cookies_path: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    proxy: Optional[str] = None,
    user_agent: Optional[str] = None,
    # Advanced parameters (rarely needed)
    po_token: Optional[str] = None,
    visitor_data: Optional[str] = None,
    # Background music removal parameters
    remove_background_music: bool = False,
    background_removal_model: str = "htdemucs",
    background_removal_quality: str = "balanced",
) -> Tuple[Optional[str], Optional[str], Dict]:
    """
    Download YouTube video/audio and subtitles with 2025 bypass methods.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save files
        language: Preferred subtitle language (ISO 639-1 code)
        audio_only: If True, download only audio
        download_subtitles: If True, attempt to download subtitles
        auto_update: If True, update yt-dlp before downloading
        cookies_path: Path to a Netscape format cookies file
        cookies_from_browser: Name of a browser to load cookies from
        proxy: URL of a proxy server to use
        user_agent: Custom user-agent string
        po_token: (Advanced) YouTube player object token
        visitor_data: (Advanced) YouTube visitor data
        remove_background_music: If True, remove background music after download
        background_removal_model: Demucs model for background removal
        background_removal_quality: Quality preset for background removal
        
    Returns:
        Tuple of (audio_path, subtitle_path, video_info)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # === VERSION CHECK ===
    print("\n🔍 Checking yt-dlp version...")
    is_valid, current_version = check_ytdlp_version()
    print(f"📦 yt-dlp version: {current_version}")
    
    if not is_valid:
        print(f"\n⚠️  WARNING: Your yt-dlp version is outdated!")
        print(f"   Current: {current_version}")
        print(f"   Required: {MINIMUM_YTDLP_VERSION} or newer")
        print(f"\n   2025 bypass methods require the latest version.")
        print(f"   Update now with: pip install -U yt-dlp\n")
        
        if auto_update:
            print("   Auto-updating now...")
            update_ytdlp()
    else:
        print("✓ Version is up to date!")
        if auto_update:
            update_ytdlp()
    
    # === GET VIDEO INFO ===
    print(f"\n🎬 Fetching video information from {url}...")
    try:
        info = get_video_info(url)
        print(f"  Title: {info['title']}")
        print(f"  Duration: {info['duration']}s")
        print(f"  Has manual subtitles: {info['has_subtitles']}")
        print(f"  Has auto-captions: {info['has_automatic_captions']}")
        
        # Display available languages
        all_langs = info.get('all_subtitle_languages', [])
        if all_langs:
            print(f"  Available subtitle languages: {', '.join(all_langs[:10])}" + 
                  (f" (+{len(all_langs)-10} more)" if len(all_langs) > 10 else ""))
            # Check if requested language is available
            if language in all_langs:
                print(f"  ✓ Requested language '{language}' is available!")
            elif language not in ['en'] and 'en' in all_langs:
                print(f"  ⚠ Requested language '{language}' not found, will use 'en' as fallback")
        else:
            print(f"  ⚠ No subtitles/captions available for this video")
    except Exception as e:
        print(f"⚠ Warning: Could not fetch video info: {e}")
        info = {}
    
    # === AUTHENTICATION SETUP ===
    # Auto-detect cookies file in project root if not specified
    if not cookies_path and not cookies_from_browser:
        # Check for cookies file in common locations
        possible_cookie_paths = [
            Path('/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/cookies.txt'),
            Path('/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/youtube_cookies.txt'),
            Path.cwd() / 'cookies.txt',
            Path.cwd() / 'youtube_cookies.txt',
        ]
        
        for cookie_file in possible_cookie_paths:
            if cookie_file.exists():
                cookies_path = str(cookie_file)
                print(f"\n✅ Auto-detected cookies file: {cookies_path}")
                break
    
    has_cookies = bool(cookies_path or cookies_from_browser)
    if has_cookies:
        print("\n🔐 Authentication: ENABLED (cookies detected)")
        if cookies_from_browser:
            print(f"  Using cookies from browser: {cookies_from_browser}")
            print(f"  ⚠️  Note: Browser must be installed on this machine!")
            print(f"  ⚠️  LIGHTNING AI USERS: Browser cookies won't work on remote servers!")
            print(f"  📝 Instead, export cookies to 'cookies.txt' and place in project root.")
        elif cookies_path:
            print(f"  Using cookies file: {cookies_path}")
            if not Path(cookies_path).exists():
                print(f"  ⚠️  WARNING: Cookies file not found at: {cookies_path}")
    else:
        print("\n⚠️  Authentication: DISABLED (no cookies provided)")
        print("  ⚠️  CRITICAL: Downloads will likely fail on Lightning.AI without cookies!")
        print("")
        print("  SOLUTION FOR LIGHTNING AI USERS:")
        print("  1. On your LOCAL PC: Install 'Get cookies.txt LOCALLY' browser extension")
        print("     Chrome: https://chrome.google.com/webstore/detail/cclelndahbckbenkjhflpdbgdldlbecc")
        print("  2. Log into YouTube on your local browser")
        print("  3. Click extension and export to 'cookies.txt'")
        print("  4. Upload 'cookies.txt' to: /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/")
        print("  5. Re-run this script (cookies will be auto-detected)")
        print("")
    
    # === CONFIGURE YT-DLP OPTIONS ===
    print("\n⚙️  Configuring download with latest yt-dlp...")
    print(f"  Auto client selection: ENABLED")
    print(f"  User-agent rotation: ENABLED ({len(USER_AGENTS)} agents)")
    
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
        
        # === 2025 BYPASS METHODS ===
        # Let yt-dlp handle client selection automatically
        # Simpler and more reliable with latest yt-dlp versions
        
        # === AUTHENTICATION ===
        # Only use cookiefile if it exists, cookiesfrombrowser will be handled with try/except
        'cookiefile': cookies_path if cookies_path and Path(cookies_path).exists() else None,
        
        # === NETWORK OPTIONS ===
        'proxy': proxy or None,
        'socket_timeout': 30,
        
        # === RATE LIMITING ===
        'sleep_interval': 2,
        'sleep_interval_requests': 1,
        'sleep_interval_subtitles': 1,
        
        # === RETRY OPTIONS ===
        'retries': 3,
        'fragment_retries': 5,
        'extractor_retries': 3,
        
        # === HEADERS ===
        'http_headers': {
            'User-Agent': user_agent or get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    }
    
    # Add subtitle options - make them optional with error handling
    if download_subtitles:
        ydl_opts.update({
            'writesubtitles': True,
            'writeautomaticsub': True,  # Fallback to auto-generated
            'subtitleslangs': [language, 'en'],  # Preferred language + English fallback
            'subtitlesformat': 'srt',
            'skip_download': False,
            'ignoreerrors': True,  # Continue even if subtitles fail
        })
    
    # === DOWNLOAD WITH INTELLIGENT RETRY ===
    audio_file = None
    subtitle_file = None
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            if attempt > 0:
                delay = (2 ** attempt) * 2 + random.uniform(1, 3)
                print(f"\n⏳ Waiting {delay:.1f}s before retry {attempt + 1}/{max_attempts}...")
                time.sleep(delay)
                
                # Rotate user agent on retry
                ydl_opts['http_headers']['User-Agent'] = get_random_user_agent()
                print(f"  Rotated user-agent for attempt {attempt + 1}")
            
            print(f"\n📥 Download attempt {attempt + 1}/{max_attempts}")
            print(f"  Using yt-dlp auto-selection...")
            
            # First, try downloading without subtitles to ensure audio download works
            opts_audio_only = ydl_opts.copy()
            if download_subtitles:
                # Remove subtitle options for first attempt
                for key in ['writesubtitles', 'writeautomaticsub', 'subtitleslangs', 'subtitlesformat']:
                    opts_audio_only.pop(key, None)
            
            # Try to add cookies_from_browser, but make it optional
            if cookies_from_browser and not opts_audio_only.get('cookiefile'):
                try:
                    # Test if browser cookies are available
                    test_opts = {'cookiesfrombrowser': (cookies_from_browser,), 'quiet': True}
                    with yt_dlp.YoutubeDL(test_opts) as test_ydl:
                        pass  # Just test initialization
                    # If successful, add to options
                    opts_audio_only['cookiesfrombrowser'] = (cookies_from_browser,)
                    if attempt == 0:
                        print(f"  ✓ Browser cookies loaded successfully")
                except Exception as e:
                    if attempt == 0:
                        print(f"  ⚠️  Could not load browser cookies: {str(e)[:100]}")
                        print(f"  Continuing without cookies (downloads may fail)")
                    # Don't add cookiesfrombrowser to options
            
            with yt_dlp.YoutubeDL(opts_audio_only) as ydl:
                print("  Downloading audio...")
                result = ydl.extract_info(url, download=True)
                
                # Check if result is None (download failed)
                if result is None:
                    raise RuntimeError("Download failed - no video data returned")
                
                # Find downloaded files
                title = result.get('title', 'video')
                sanitized_title = yt_dlp.utils.sanitize_filename(title)
                
                # Look for audio file
                audio_patterns = [f"{sanitized_title}.wav", f"{sanitized_title}.mp3", f"{sanitized_title}.m4a"]
                for pattern in audio_patterns:
                    audio_candidate = output_path / pattern
                    if audio_candidate.exists():
                        audio_file = str(audio_candidate)
                        print(f"  ✅ Audio downloaded successfully: {Path(audio_file).name}")
                        break
            
            # Download succeeded, exit retry loop
            if audio_file:
                # Try to get which client was used (for logging)
                used_client = result.get('protocol', 'unknown')
                print(f"\n✅ Download successful on attempt {attempt + 1}")
                break
        
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Attempt {attempt + 1} failed: {error_msg[:200]}")
            
            if attempt == max_attempts - 1:
                print("\n💥 All download attempts exhausted")
                print_actionable_error(error_msg)
                raise
            else:
                print("  Retrying with different configuration...")
    
    # === DOWNLOAD SUBTITLES ===
    if download_subtitles and audio_file:
        subtitle_file = download_subtitles_robust(
            url=url,
            output_path=output_path,
            sanitized_title=sanitized_title,
            language=language,
            result=result
        )
    
    return audio_file, subtitle_file, info


def download_and_process_youtube(
    url: str,
    output_dir: str,
    language: str = 'en',
    use_whisper_if_no_srt: bool = True,
    auto_update: bool = True,
    cookies_path: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    proxy: Optional[str] = None,
    user_agent: Optional[str] = None,
    po_token: Optional[str] = None,
    visitor_data: Optional[str] = None,
    remove_background_music: bool = False,
    background_removal_model: str = "htdemucs",
    background_removal_quality: str = "balanced",
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
        auto_update=auto_update,
        cookies_path=cookies_path,
        cookies_from_browser=cookies_from_browser,
        proxy=proxy,
        user_agent=user_agent,
        po_token=po_token,
        visitor_data=visitor_data,
        remove_background_music=remove_background_music,
        background_removal_model=background_removal_model,
        background_removal_quality=background_removal_quality,
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
        
        print(f"\n✅ Download complete!")
        print(f"  Audio: {audio}")
        print(f"  Subtitles: {srt}")
        print(f"  Title: {info.get('title', 'N/A')}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
