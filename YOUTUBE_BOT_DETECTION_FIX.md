# YouTube Bot Detection Fix

## Problem

YouTube is blocking download requests with the error:
```
ERROR: [youtube] Sign in to confirm you're not a bot
```

This happens when YouTube detects automated access and requires browser cookies for authentication.

## Solution Options

### Option 1: Use Browser Cookies (Recommended)

Export cookies from your browser and use them with yt-dlp:

#### Step 1: Export Cookies

**Chrome/Edge:**
1. Install extension: [Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
2. Go to YouTube and sign in
3. Click the extension icon and download cookies
4. Save as `youtube_cookies.txt`

**Firefox:**
1. Install extension: [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)
2. Go to YouTube and sign in  
3. Click the extension icon and export
4. Save as `youtube_cookies.txt`

#### Step 2: Place Cookies File

Put the cookies file in your project directory:
```bash
/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/youtube_cookies.txt
```

#### Step 3: Update Code to Use Cookies

The code already supports cookies via the `--cookies` option in yt-dlp, but we need to enable it.

### Option 2: Use --cookies-from-browser (Easier)

Instead of exporting cookies manually, yt-dlp can extract them directly from your browser:

```python
# In youtube_downloader.py, add to ydl_opts:
'cookiesfrombrowser': ('chrome',),  # or 'firefox', 'edge', 'safari'
```

### Option 3: Temporary Workaround - Retry with Delays

Add exponential backoff and user-agent rotation to reduce bot detection.

## Quick Fix Implementation

Here's the code fix to add cookie support:

### File: `utils/youtube_downloader.py`

```python
def download_youtube_video(
    url: str,
    output_dir: str,
    language: str = 'en',
    audio_only: bool = True,
    download_subtitles: bool = True,
    auto_update: bool = True,
    cookies_file: str = None  # NEW PARAMETER
) -> Tuple[Optional[str], Optional[str], Dict]:
    """
    Download YouTube video/audio and subtitles.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save files
        language: Preferred subtitle language
        audio_only: If True, download only audio
        download_subtitles: If True, attempt to download subtitles
        auto_update: If True, update yt-dlp before downloading
        cookies_file: Path to cookies file (to bypass bot detection)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Auto-update yt-dlp
    if auto_update:
        update_ytdlp()
    
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
        # Anti-bot detection measures
        'extractor_retries': 5,
        'fragment_retries': 5,
        'retries': 10,
        'sleep_interval': 3,
        'max_sleep_interval': 8,
    }
    
    # Add cookies if provided
    if cookies_file and os.path.exists(cookies_file):
        print(f"Using cookies from: {cookies_file}")
        ydl_opts['cookiefile'] = cookies_file
    else:
        # Try to use browser cookies as fallback
        print("Attempting to use browser cookies...")
        try:
            # Try Chrome first, then Firefox
            for browser in ['chrome', 'firefox', 'edge']:
                try:
                    ydl_opts['cookiesfrombrowser'] = (browser,)
                    print(f"  Trying {browser} cookies...")
                    break
                except:
                    continue
        except Exception as e:
            print(f"  Could not access browser cookies: {e}")
    
    # ... rest of the function
```

### File: `xtts_demo.py` - Update UI

Add cookies file input:

```python
with gr.TabItem("📹 YouTube Processing"):
    gr.Markdown("**Download videos and extract transcripts automatically**")
    youtube_url = gr.Textbox(...)
    
    # NEW: Cookie file input
    youtube_cookies_file = gr.Textbox(
        label="🍪 Cookies File (Optional)",
        placeholder="Path to youtube_cookies.txt (to bypass bot detection)",
        value="youtube_cookies.txt",
        info="Export from browser if you get 'Sign in to confirm you're not a bot' error"
    )
    
    # ... rest of UI
```

## Implementation Steps

1. **Update `youtube_downloader.py`** to accept cookies parameter
2. **Update `batch_processor.py`** to pass cookies to downloader
3. **Update `xtts_demo.py`** UI to accept cookies file path
4. **Document** cookie export process for users

## Alternative: Use yt-dlp Directly in Terminal

As a quick test, try downloading directly with cookies:

```bash
# With cookie file
yt-dlp --cookies youtube_cookies.txt "https://www.youtube.com/watch?v=qXFRHxF3rAM"

# Or with browser cookies
yt-dlp --cookies-from-browser chrome "https://www.youtube.com/watch?v=qXFRHxF3rAM"
```

## Error Handling Improvement

Also fix the `NoneType` error by adding null check:

```python
try:
    with yt_dlp.YoutubeDL(opts_audio_only) as ydl:
        print("Downloading audio...")
        result = ydl.extract_info(url, download=True)
        
        # FIX: Check if result is None
        if result is None:
            raise RuntimeError("Failed to extract video information. YouTube may be blocking requests.")
        
        # Find downloaded files
        title = result.get('title', 'video')
        sanitized_title = yt_dlp.utils.sanitize_filename(title)
        # ...
```

## Testing

After implementing the fix, test with:

1. **Single video**: Use one YouTube URL with cookies
2. **Batch mode**: Try multiple URLs
3. **Without cookies**: Verify graceful fallback/error message

## Long-term Solution

Consider implementing **multiple download backends** as fallback:
1. yt-dlp with cookies (primary)
2. pytube (alternative library)
3. Manual instruction for users to download and provide audio files

---

Would you like me to implement this fix now?
