# YouTube Download Bypass Guide (2025)

## 🚀 Overview

This guide covers the latest methods to bypass YouTube restrictions when downloading videos and subtitles for dataset creation. The system now includes multiple advanced bypass techniques to handle:

- **Rate limiting (HTTP 429 errors)**
- **Regional restrictions**
- **Bot detection**
- **IP blocking**
- **Captcha challenges**

## 🎯 Quick Start

### Basic Usage (No Configuration)

The system will automatically use the latest bypass methods with default settings:

```bash
python utils/youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID" ./downloads en
```

### Recommended Configuration

For best results, use cookies from your browser:

```bash
python utils/youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID" ./downloads en --from-browser chrome
```

---

## 🛠️ Bypass Methods

### 1. Player Client Rotation (Automatic)

The system automatically tries multiple player clients in order of success rate:

1. **iOS** - Best success rate (2025)
2. **Android** - Good fallback
3. **TV Embedded** - For restricted content
4. **MediaConnect** - Alternative method
5. **Mobile Web** - Last resort

**No configuration needed** - This happens automatically!

---

### 2. Browser Cookies (Recommended)

Using cookies from your logged-in browser session dramatically improves success rates.

#### Method A: Import from Browser (Easiest)

```bash
# Chrome
python utils/youtube_downloader.py URL ./downloads en --from-browser chrome

# Firefox
python utils/youtube_downloader.py URL ./downloads en --from-browser firefox

# Edge
python utils/youtube_downloader.py URL ./downloads en --from-browser edge
```

**Environment Variable:**
```powershell
$env:YTDLP_COOKIES_FROM_BROWSER = "chrome"
```

#### Method B: Export Cookies File

1. **Install Browser Extension:**
   - Chrome/Edge: [Get cookies.txt](https://chrome.google.com/webstore/detail/get-cookiestxt/bgaddhkoddajcdgocldbbfleckgcbcid)
   - Firefox: [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)

2. **Export Cookies:**
   - Go to YouTube and log in
   - Click extension icon → Export cookies for youtube.com
   - Save as `youtube_cookies.txt`

3. **Use Cookies:**
```bash
python utils/youtube_downloader.py URL ./downloads en --cookies youtube_cookies.txt
```

**Environment Variable:**
```powershell
$env:YTDLP_COOKIES = "D:\path\to\youtube_cookies.txt"
```

---

### 3. Proxy Support

Use proxies to bypass IP-based rate limiting and regional restrictions.

#### Single Proxy

```bash
python utils/youtube_downloader.py URL ./downloads en --proxy http://proxy:port
```

**With Authentication:**
```bash
python utils/youtube_downloader.py URL ./downloads en --proxy http://user:pass@proxy:port
```

**Environment Variable:**
```powershell
$env:YTDLP_PROXY = "http://proxy:port"
```

#### Proxy Types Supported

- **HTTP/HTTPS:** `http://proxy:port`
- **SOCKS5:** `socks5://proxy:port`
- **SOCKS4:** `socks4://proxy:port`

#### Proxy Rotation (Advanced)

For batch processing, use proxy rotation to avoid rate limits:

**Option 1: Proxy List File**

Create `proxies.txt`:
```
http://proxy1:8080
http://user:pass@proxy2:3128
socks5://proxy3:1080
```

Set environment variable:
```powershell
$env:YTDLP_PROXY_FILE = "D:\path\to\proxies.txt"
```

**Option 2: Comma-Separated List**
```powershell
$env:YTDLP_PROXY_LIST = "http://proxy1:8080,http://proxy2:3128,socks5://proxy3:1080"
```

The system will automatically rotate through proxies on failures.

---

### 4. PO Token Authentication (2024+ Method)

YouTube now uses PO tokens for enhanced authentication. This is the **latest bypass method**.

#### What is a PO Token?

- YouTube's latest anti-bot mechanism (introduced 2024)
- Required for some restricted content
- Changes periodically

#### How to Get PO Token

**Method 1: Browser Developer Tools**

1. Open YouTube in your browser
2. Press `F12` to open DevTools
3. Go to **Network** tab
4. Play any video
5. Filter for `player` requests
6. Look for `X-YouTube-PO-Token` or `poToken` in headers/params
7. Copy the token value

**Method 2: Use yt-dlp's extractor**
```bash
yt-dlp --print "%(requested_formats)s" --cookies-from-browser chrome "URL"
```

#### Using PO Token

```bash
python utils/youtube_downloader.py URL ./downloads en --po-token YOUR_PO_TOKEN --visitor-data YOUR_VISITOR_DATA
```

**Environment Variables:**
```powershell
$env:YTDLP_PO_TOKEN = "your_po_token_here"
$env:YTDLP_VISITOR_DATA = "your_visitor_data_here"
```

**Note:** PO tokens expire after some time (hours to days). You'll need to extract new ones periodically.

---

### 5. Custom User-Agent

Rotate user agents to avoid fingerprinting.

```bash
python utils/youtube_downloader.py URL ./downloads en --ua "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)..."
```

**Environment Variable:**
```powershell
$env:YTDLP_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)..."
```

**Built-in Rotation:** The system automatically rotates between 7 different modern user agents!

---

## 📊 Environment Variables Summary

Set these in PowerShell for persistent configuration:

```powershell
# Browser cookies (easiest method)
$env:YTDLP_COOKIES_FROM_BROWSER = "chrome"

# Or use cookies file
$env:YTDLP_COOKIES = "D:\path\to\cookies.txt"

# Proxy configuration
$env:YTDLP_PROXY = "http://proxy:port"
$env:YTDLP_PROXY_FILE = "D:\path\to\proxies.txt"
$env:YTDLP_PROXY_LIST = "http://p1:8080,http://p2:3128"

# Advanced authentication
$env:YTDLP_PO_TOKEN = "your_token"
$env:YTDLP_VISITOR_DATA = "your_visitor_data"

# User agent
$env:YTDLP_UA = "Mozilla/5.0..."
```

To make permanent (Windows):
```powershell
[System.Environment]::SetEnvironmentVariable('YTDLP_COOKIES_FROM_BROWSER', 'chrome', 'User')
```

---

## 🎬 Batch Processing

When processing multiple videos, the system automatically:

1. **Retries with different player clients** on each failure
2. **Uses exponential backoff** to avoid triggering rate limits
3. **Tracks and reports failures** at the end
4. **Continues processing** even if some videos fail

Example:
```python
from utils import youtube_downloader, batch_processor, srt_processor

urls = [
    "https://youtube.com/watch?v=VIDEO1",
    "https://youtube.com/watch?v=VIDEO2",
    "https://youtube.com/watch?v=VIDEO3",
]

train_csv, eval_csv, infos = batch_processor.process_youtube_batch(
    urls=urls,
    transcript_lang="en",
    out_path="./datasets",
    youtube_downloader=youtube_downloader,
    srt_processor=srt_processor,
    cookies_from_browser="chrome",  # Use your browser cookies
    # proxy="http://proxy:port",     # Optional proxy
    # po_token="your_token",          # Optional PO token
)
```

---

## 🔧 Troubleshooting

### Error: "Sign in to confirm you're not a bot"

**Solution:** Use browser cookies
```bash
--from-browser chrome
```

### Error: "HTTP Error 429: Too Many Requests"

**Solutions (in order):**
1. Use browser cookies: `--from-browser chrome`
2. Use a proxy: `--proxy http://proxy:port`
3. Wait 10-30 minutes before retrying
4. Use a VPN or different network
5. Get a PO token: `--po-token TOKEN`

### Error: "Video unavailable"

**Possible causes:**
- Video is private/deleted
- Regional restriction
- Age restriction

**Solutions:**
1. Use cookies from logged-in account: `--from-browser chrome`
2. Use proxy from allowed region
3. Verify the URL is correct

### Error: "Extraction failed with all player clients"

**This means YouTube is blocking your IP/session.**

**Solutions:**
1. ✅ Use browser cookies (most important)
2. ✅ Use a proxy or VPN
3. ✅ Get fresh PO token + visitor data
4. Wait 30-60 minutes
5. Try from different network

### Subtitles Not Available

**The system automatically falls back to Whisper transcription!**

If subtitles aren't available:
1. System tries multiple extraction methods
2. Uses youtube-transcript-api as fallback
3. If all fails, uses Faster Whisper to transcribe audio
4. You get transcripts either way! 🎉

---

## 🏆 Best Practices

### For Maximum Success Rate

1. **Always use browser cookies** (`--from-browser chrome`)
2. **Enable auto-update** (default) to get latest yt-dlp
3. **Use delays between requests** (automatic)
4. **Start with fewer videos** to test configuration
5. **Keep yt-dlp updated** regularly

### For Batch Processing

1. **Use browser cookies** - Essential for reliability
2. **Consider proxy rotation** for large batches (>10 videos)
3. **Monitor console output** for failures
4. **Process in smaller batches** (5-10 videos) if you hit limits
5. **Save failed URLs** and retry later with different method

### For Restricted Content

1. **Log into YouTube** in your browser first
2. **Use cookies from that browser** (`--from-browser chrome`)
3. **Get fresh PO token** if available
4. **Use residential proxy** if regionally restricted

---

## 🌐 Proxy Recommendations

### Free Proxies
- Generally unreliable
- High failure rate
- Good for testing only
- See `utils/proxy_manager.py` for free sources

### Paid Proxy Services (Recommended)
- **Bright Data** - Best for YouTube (residential IPs)
- **Smartproxy** - Good balance of price/performance
- **Oxylabs** - Enterprise-grade
- **IPRoyal** - Budget-friendly residential proxies

### Proxy Types for YouTube
- **Residential Proxies** - Best success rate (look like real users)
- **Datacenter Proxies** - Cheaper but lower success rate
- **Mobile Proxies** - Excellent for mobile content

---

## 📱 Platform Support

### Windows (PowerShell)
```powershell
$env:YTDLP_COOKIES_FROM_BROWSER = "chrome"
python utils/youtube_downloader.py URL ./downloads en
```

### Linux/Mac (Bash)
```bash
export YTDLP_COOKIES_FROM_BROWSER="chrome"
python utils/youtube_downloader.py URL ./downloads en
```

---

## 🔄 Keeping Updated

YouTube constantly updates their anti-bot measures. Keep your tools updated:

### Update yt-dlp (Automatic)
The system auto-updates yt-dlp before each download by default.

### Manual Update
```bash
pip install -U yt-dlp
```

### Update This Project
Pull latest changes to get newest bypass methods:
```bash
git pull origin main
```

---

## ⚡ Performance Tips

1. **Parallel downloads:** Process multiple videos, but not too many at once (5-10 max)
2. **Use SSD storage:** Faster audio processing
3. **Enable Whisper fallback:** Always get transcripts
4. **Monitor rate limits:** If you hit 429 errors, slow down or use proxy
5. **Keep cookies fresh:** Re-export cookies weekly

---

## 📞 Getting Help

If you're still having issues:

1. **Check console output** - Error messages are detailed
2. **Try with single video first** - Test configuration
3. **Verify prerequisites:**
   - yt-dlp is installed and updated
   - FFmpeg is installed
   - Python packages are up to date
4. **Check YouTube status** - Sometimes YouTube itself has issues

---

## 🎉 Success Indicators

You'll know it's working when you see:
- ✓ Video information fetched
- ✓ Audio downloaded successfully with [client] client
- ✓ Successfully downloaded subtitles
- No "HTTP 429" errors
- No "Sign in to confirm" messages

---

## 📚 Additional Resources

- [yt-dlp documentation](https://github.com/yt-dlp/yt-dlp)
- [Cookies.txt format](https://curl.se/docs/http-cookies.html)
- [Proxy setup guide](https://github.com/yt-dlp/yt-dlp#network-options)

---

**Last Updated:** October 2025  
**Compatibility:** yt-dlp 2024.10+, Python 3.8+
