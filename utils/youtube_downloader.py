"""
YouTube Downloader Module
Downloads videos and subtitles from YouTube using yt-dlp with auto-update.
"""

import os
import subprocess
import json
import time
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import yt_dlp

# Rotatable User-Agents (2025 latest versions - updated Jan 2025)
USER_AGENTS = [
    # Chrome Windows (latest)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    # Chrome macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    # Safari iPhone (latest)
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Mobile/15E148 Safari/604.1",
    # Safari iPad
    "Mozilla/5.0 (iPad; CPU OS 18_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Mobile/15E148 Safari/604.1",
    # Android Chrome
    "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.135 Mobile Safari/537.36",
    # Firefox Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    # Edge Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
]

DEFAULT_USER_AGENT = USER_AGENTS[0]

# Latest YouTube client versions (updated Jan 2025)
LATEST_IOS_VERSION = "19.45.4"
LATEST_ANDROID_VERSION = "19.43.41"


def get_random_user_agent() -> str:
    """
    Get a random user agent from the pool to avoid fingerprinting.
    """
    return random.choice(USER_AGENTS)


def get_optimal_ytdlp_opts(
    cookies_path: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    proxy: Optional[str] = None,
    user_agent: Optional[str] = None,
    po_token: Optional[str] = None,
    visitor_data: Optional[str] = None,
    player_client: str = 'android_creator',
) -> dict:
    """
    Get optimal yt-dlp options with latest bypass techniques (2025).
    
    Args:
        cookies_path: Path to Netscape cookies file
        cookies_from_browser: Browser to extract cookies from (RECOMMENDED)
        proxy: Proxy URL
        user_agent: Custom user agent (random if None)
        po_token: YouTube PO token for authentication bypass
        visitor_data: YouTube visitor data for enhanced authentication
        player_client: Player client (android_creator, android_music, ios_music, tv_embedded)
        
    Returns:
        Optimized yt-dlp options dictionary
    """
    opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        # Latest bypass: Use creator/music clients which have highest success rates in 2025
        'extractor_args': {
            'youtube': {
                'player_client': [player_client],
                'player_skip': ['webpage', 'js', 'configs'],  # Skip all detection points
            }
        },
        # Network options
        'http_headers': {
            'User-Agent': user_agent or get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        },
        # Disable manifests for faster extraction
        'youtube_include_dash_manifest': False,
        'youtube_include_hls_manifest': False,
        # Better retry logic
        'socket_timeout': 30,
        'retries': 10,
        'fragment_retries': 10,
        'extractor_retries': 5,
        'file_access_retries': 3,
        # Sleep intervals to avoid rate limiting
        'sleep_interval': 1,
        'max_sleep_interval': 5,
        'sleep_interval_subtitles': 2,
    }
    
    # PO Token authentication (latest YouTube bypass - 2024+)
    if po_token:
        opts['extractor_args']['youtube']['po_token'] = [po_token]
    if visitor_data:
        opts['extractor_args']['youtube']['visitor_data'] = [visitor_data]
    
    # Authentication options
    if proxy:
        opts['proxy'] = proxy
    if cookies_path:
        opts['cookiefile'] = cookies_path
    elif cookies_from_browser:
        opts['cookiesfrombrowser'] = (cookies_from_browser, None, None, None)
    
    return opts


def check_ytdlp_version():
    """
    Check if yt-dlp version is recent enough for 2025 YouTube bypass.
    """
    try:
        result = subprocess.run(
            ["python", "-m", "pip", "show", "yt-dlp"],
            check=True,
            capture_output=True,
            text=True
        )
        version_line = [l for l in result.stdout.split('\n') if l.startswith('Version:')]
        if version_line:
            version = version_line[0].split(':')[1].strip()
            print(f"📦 yt-dlp version: {version}")
            # Check if version is at least 2024.12.13 (critical YouTube fixes)
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse('2024.12.13'):
                print("⚠️  WARNING: yt-dlp version is outdated for 2025 YouTube bypass!")
                print("   Please update: pip install -U yt-dlp")
                return False
            return True
    except Exception as e:
        print(f"Could not check yt-dlp version: {e}")
        return None


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


def get_video_info(
    url: str,
    cookies_path: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    proxy: Optional[str] = None,
    user_agent: Optional[str] = None,
    po_token: Optional[str] = None,
    visitor_data: Optional[str] = None,
) -> Dict:
    """
    Get video information without downloading (simplified).
    
    Args:
        url: YouTube video URL
        cookies_path: Optional path to cookies file
        cookies_from_browser: Optional browser name (only if available)
        proxy: Optional proxy URL
        user_agent: Optional user-agent string
        po_token: Optional PO token
        visitor_data: Optional visitor data
        
    Returns:
        Dictionary with video metadata
    """
    # Aggressive bypass settings - 2025 updated clients
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'extractor_args': {
            'youtube': {
                'player_client': ['android_creator', 'android_music', 'ios_music', 'tv_embedded'],
                'player_skip': ['webpage', 'js', 'configs'],  # Skip all detection points
                'skip': ['hls', 'dash'],
            }
        },
        'http_headers': {
            'User-Agent': f'com.google.ios.youtube/{LATEST_IOS_VERSION} (iPhone16,2; U; CPU iOS 18_2 like Mac OS X;)',
            'X-YouTube-Client-Name': '5',
            'X-YouTube-Client-Version': LATEST_IOS_VERSION,
        },
        'age_limit': None,
    }
    
    # Add optional parameters only if provided
    if user_agent:
        ydl_opts['http_headers']['User-Agent'] = user_agent
    if proxy:
        ydl_opts['proxy'] = proxy
    if cookies_path:
        ydl_opts['cookiefile'] = cookies_path
    elif cookies_from_browser:
        ydl_opts['cookiesfrombrowser'] = (cookies_from_browser, None, None, None)
    if po_token:
        ydl_opts['extractor_args']['youtube']['po_token'] = [po_token]
    if visitor_data:
        ydl_opts['extractor_args']['youtube']['visitor_data'] = [visitor_data]
    
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
    max_retries: int = 3,
    cookies_path: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    proxy: Optional[str] = None,
    user_agent: Optional[str] = None,
    po_token: Optional[str] = None,
    visitor_data: Optional[str] = None,
) -> Optional[str]:
    """
    Robustly download subtitles with multiple fallback strategies to bypass rate limiting.
    
    Strategies:
    1. Direct API extraction from result metadata (fastest, no extra request)
    2. yt-dlp with cookies and browser impersonation (mweb client, optional proxy)
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
'User-Agent': user_agent or DEFAULT_USER_AGENT,
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
    
    # Strategy 2: Use yt-dlp with basic approach (simplified)
    print("\n[Strategy 2] Using yt-dlp to download subtitles...")
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = 2 + random.uniform(1, 3)
                print(f"  ⏳ Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
            
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
                # Aggressive bypass settings - 2025 updated
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android_creator', 'android_music', 'ios_music', 'tv_embedded'],
                        'player_skip': ['webpage', 'js', 'configs'],
                        'skip': ['hls', 'dash'],
                    }
                },
                'http_headers': {
                    'User-Agent': f'com.google.ios.youtube/{LATEST_IOS_VERSION} (iPhone16,2; U; CPU iOS 18_2 like Mac OS X;)',
                    'X-YouTube-Client-Name': '5',
                    'X-YouTube-Client-Version': LATEST_IOS_VERSION,
                },
                'age_limit': None,
            }
            
            # Add optional parameters
            if user_agent:
                subtitle_opts['http_headers']['User-Agent'] = user_agent
            if proxy:
                subtitle_opts['proxy'] = proxy
            if cookies_path:
                subtitle_opts['cookiefile'] = cookies_path
            elif cookies_from_browser:
                subtitle_opts['cookiesfrombrowser'] = (cookies_from_browser, None, None, None)
            
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
                    print(f"  ✅ Successfully downloaded subtitles: {subtitle_candidate.name}")
                    return subtitle_file
            
            print(f"  ⚠ Attempt {attempt + 1} completed but no subtitle file found")
        
        except Exception as e:
            print(f"  ⚠ Attempt {attempt + 1} failed: {str(e)[:100]}")
    
    # If all attempts failed
    print("\n⚠ Could not download subtitles from YouTube")
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
    auto_update: bool = False,
    cookies_path: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    proxy: Optional[str] = None,
    user_agent: Optional[str] = None,
    po_token: Optional[str] = None,
    visitor_data: Optional[str] = None,
    # Background music removal parameters
    remove_background_music: bool = False,
    background_removal_model: str = "htdemucs",
    background_removal_quality: str = "balanced",
) -> Tuple[Optional[str], Optional[str], Dict]:
    """
    Download YouTube video/audio and subtitles (simplified, works without cookies).
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save files
        language: Preferred subtitle language (ISO 639-1 code)
        audio_only: If True, download only audio
        download_subtitles: If True, attempt to download subtitles
        auto_update: If True, update yt-dlp before downloading (default: False)
        cookies_path: Optional path to cookies file
        cookies_from_browser: Optional browser name for cookies (only if available)
        proxy: Optional proxy URL
        user_agent: Optional custom user agent
        po_token: Optional PO token
        visitor_data: Optional visitor data
        remove_background_music: Remove background music from audio (requires demucs)
        background_removal_model: Demucs model to use (htdemucs, mdx, mdx_extra)
        background_removal_quality: Quality preset (fast, balanced, best)
        
    Returns:
        Tuple of (audio_path, subtitle_path, video_info)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Only use env variables if explicitly set (don't force cookies)
    if cookies_path is None:
        cookies_path = os.getenv("YTDLP_COOKIES")
    if cookies_from_browser is None:
        cookies_from_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER")
    if proxy is None:
        proxy = os.getenv("YTDLP_PROXY")
    if user_agent is None:
        user_agent = os.getenv("YTDLP_UA")
    if po_token is None:
        po_token = os.getenv("YTDLP_PO_TOKEN")
    if visitor_data is None:
        visitor_data = os.getenv("YTDLP_VISITOR_DATA")
    
    # Check yt-dlp version
    check_ytdlp_version()
    
    # Auto-update yt-dlp
    if auto_update:
        update_ytdlp()
    
    # Get video info first
    print(f"Fetching video information from {url}...")
    try:
        info = get_video_info(
            url,
            cookies_path=cookies_path,
            cookies_from_browser=cookies_from_browser,
            proxy=proxy,
            user_agent=user_agent,
            po_token=po_token,
            visitor_data=visitor_data,
        )
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
        print(f"Warning: Could not fetch video info: {e}")
        info = {}
    
    # Aggressive yt-dlp options - 2025 UPDATED with best bypass methods
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
        # CRITICAL: Use latest working clients (android_creator has highest success rate)
        'extractor_args': {
            'youtube': {
                'player_client': ['android_creator', 'android_music', 'ios_music', 'tv_embedded', 'android_vr'],
                'player_skip': ['webpage', 'js', 'configs'],  # Skip ALL detection points
                'skip': ['hls', 'dash'],
            }
        },
        # Latest iOS YouTube app headers
        'http_headers': {
            'User-Agent': f'com.google.ios.youtube/{LATEST_IOS_VERSION} (iPhone16,2; U; CPU iOS 18_2 like Mac OS X;)',
            'X-YouTube-Client-Name': '5',
            'X-YouTube-Client-Version': LATEST_IOS_VERSION,
        },
        'age_limit': None,
        # Retry logic
        'retries': 10,
        'fragment_retries': 10,
        'socket_timeout': 30,
    }
    
    # Add optional authentication/networking (only if provided)
    if user_agent:
        ydl_opts['http_headers']['User-Agent'] = user_agent
    if proxy:
        ydl_opts['proxy'] = proxy
    if cookies_path:
        ydl_opts['cookiefile'] = cookies_path
    elif cookies_from_browser:
        # Only add if explicitly provided
        ydl_opts['cookiesfrombrowser'] = (cookies_from_browser, None, None, None)
    if po_token:
        ydl_opts['extractor_args']['youtube']['po_token'] = [po_token]
    if visitor_data:
        ydl_opts['extractor_args']['youtube']['visitor_data'] = [visitor_data]
    
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
    
    # Download (simplified - works without cookies)
    print(f"\nDownloading from YouTube...")
    audio_file = None
    subtitle_file = None
    result = None
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Downloading audio...")
            result = ydl.extract_info(url, download=True)
            
            # Check if extraction failed
            if result is None:
                raise RuntimeError("Failed to extract video information")
            
            # Find downloaded files
            title = result.get('title', 'video')
            sanitized_title = yt_dlp.utils.sanitize_filename(title)
            
            # Look for audio file
            audio_patterns = [f"{sanitized_title}.wav", f"{sanitized_title}.mp3", f"{sanitized_title}.m4a"]
            for pattern in audio_patterns:
                audio_candidate = output_path / pattern
                if audio_candidate.exists():
                    audio_file = str(audio_candidate)
                    print(f"✓ Audio downloaded successfully: {audio_file}")
                    break
            
            if not audio_file:
                raise RuntimeError("Audio file not found after download")
            
            # Remove background music if requested
            if remove_background_music:
                try:
                    from utils.audio_background_remover import remove_background_music as remove_bg, is_available
                    
                    if not is_available():
                        print("\n⚠ Background music removal requested but Demucs not installed")
                        print("   Install with: pip install demucs")
                        print("   Continuing without background removal...")
                    else:
                        print("\n🎵 Removing background music from downloaded audio...")
                        print(f"   Model: {background_removal_model}")
                        print(f"   Quality: {background_removal_quality}")
                        
                        # Process in-place (replace original audio file)
                        audio_file = remove_bg(
                            input_audio=audio_file,
                            output_audio=None,  # In-place processing
                            model=background_removal_model,
                            quality=background_removal_quality,
                            verbose=True
                        )
                        
                        print(f"\n✅ Background removed! Clean audio ready for processing.")
                except Exception as bg_error:
                    print(f"\n⚠ Background removal failed: {bg_error}")
                    print("   Continuing with original audio...")
    
    except Exception as e:
        error_str = str(e)
        print(f"\n❌ Error downloading from YouTube: {error_str}")
        print(f"\nTroubleshooting:")
        print(f"  1. Make sure yt-dlp is updated: pip install -U yt-dlp")
        print(f"  2. Try with cookies: --from-browser chrome")
        print(f"  3. Try with a proxy: --proxy http://proxy:port")
        print(f"  4. Check if the video is available and not region-locked")
        import traceback
        traceback.print_exc()
        raise
    
    # Now try to download subtitles using robust strategies
    if download_subtitles:
        subtitle_file = download_subtitles_robust(
            url=url,
            output_path=output_path,
            sanitized_title=sanitized_title,
            language=language,
            result=result,
            cookies_path=cookies_path,
            cookies_from_browser=cookies_from_browser,
            proxy=proxy,
            user_agent=user_agent,
            po_token=po_token,
            visitor_data=visitor_data,
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
    # Background music removal parameters
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
        remove_background_music: Remove background music from audio (requires demucs)
        background_removal_model: Demucs model to use (htdemucs, mdx, mdx_extra)
        background_removal_quality: Quality preset (fast, balanced, best)
        
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
    )
    
    if not audio_path:
        raise RuntimeError("Failed to download audio from YouTube")
    
    # Remove background music if requested
    if remove_background_music:
        try:
            from utils.audio_background_remover import remove_background_music as remove_bg, is_available
            
            if not is_available():
                print("\n⚠ Background music removal requested but Demucs not installed")
                print("   Install with: pip install demucs")
                print("   Continuing without background removal...")
            else:
                print("\n🎵 Removing background music from downloaded audio...")
                print(f"   Model: {background_removal_model}")
                print(f"   Quality: {background_removal_quality}")
                
                # Process in-place (replace original audio file)
                audio_path = remove_bg(
                    input_audio=audio_path,
                    output_audio=None,  # In-place processing
                    model=background_removal_model,
                    quality=background_removal_quality,
                    verbose=True
                )
                
                print(f"\n✅ Background removed! Clean audio ready for processing.")
        except Exception as e:
            print(f"\n⚠ Background removal failed: {e}")
            print("   Continuing with original audio...")
    
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
    import torch
    
    print(f"Loading Whisper model ({model_size})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "float32"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
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
    # CLI usage
    import argparse

    parser = argparse.ArgumentParser(description="YouTube audio+subtitle downloader")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("output_dir", nargs="?", default="./youtube_downloads", help="Output directory")
    parser.add_argument("language", nargs="?", default="en", help="Subtitle language (ISO 639-1)")
    parser.add_argument("--cookies", dest="cookies_path", default=os.getenv("YTDLP_COOKIES"), help="Path to cookies file (Netscape format)")
    parser.add_argument("--from-browser", dest="cookies_from_browser", default=os.getenv("YTDLP_COOKIES_FROM_BROWSER"), help="Import cookies from browser (chrome|firefox|edge)")
    parser.add_argument("--proxy", dest="proxy", default=os.getenv("YTDLP_PROXY"), help="Proxy URL, e.g. http://user:pass@host:port")
    parser.add_argument("--ua", dest="user_agent", default=os.getenv("YTDLP_UA"), help="Custom User-Agent")
    parser.add_argument("--po-token", dest="po_token", default=os.getenv("YTDLP_PO_TOKEN"), help="YouTube PO token for enhanced authentication")
    parser.add_argument("--visitor-data", dest="visitor_data", default=os.getenv("YTDLP_VISITOR_DATA"), help="YouTube visitor data for enhanced authentication")

    args = parser.parse_args()

    try:
        audio, srt, info = download_and_process_youtube(
            url=args.url,
            output_dir=args.output_dir,
            language=args.language,
            use_whisper_if_no_srt=True,
            cookies_path=args.cookies_path,
            cookies_from_browser=args.cookies_from_browser,
            proxy=args.proxy,
            user_agent=args.user_agent,
            po_token=args.po_token,
            visitor_data=args.visitor_data,
        )

        print(f"\n✓ Download complete!")
        print(f"  Audio: {audio}")
        print(f"  Subtitles: {srt}")
        print(f"  Title: {info.get('title', 'N/A')}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise
