# YouTube Downloads on Lightning AI - Bot Detection Issue

## Problem

YouTube is blocking downloads from Lightning AI with the error:
```
ERROR: Sign in to confirm you're not a bot
```

**Root Cause:** Lightning AI uses shared cloud IP addresses that YouTube has flagged for bot activity. This affects **all** client types (web, mobile, Android, iOS).

## Why Android Clients Also Fail

Contrary to earlier assumptions, even Android clients fail on Lightning AI because:
1. YouTube's bot detection is **IP-based**, not client-based
2. Lightning AI's shared IPs are flagged
3. Without authentication (cookies/OAuth), all requests are blocked

## Workarounds

### Option 1: Manual Download + Upload (RECOMMENDED)

On your **local machine** where you have browser access:

```bash
# Download using your browser's cookies
yt-dlp --cookies-from-browser chrome https://www.youtube.com/watch?v=VIDEO_ID

# Or with a cookies file
yt-dlp --cookies cookies.txt https://www.youtube.com/watch?v=VIDEO_ID
```

Then upload the audio files to Lightning AI and process them directly.

### Option 2: Use a Residential Proxy

Purchase a residential proxy service and configure:

```python
# In your code
proxy = "http://username:password@proxy-server:port"

youtube_downloader.download_youtube_video(
    url=video_url,
    output_dir="./downloads",
    proxy=proxy
)
```

**Proxy Services:**
- Bright Data (formerly Luminati)
- Smartproxy
- Oxylabs
- NetNut

Cost: ~$50-300/month depending on volume.

### Option 3: Use Cookies (If Available)

If you can somehow get fresh YouTube cookies into Lightning AI:

1. Export cookies from your browser using "Get cookies.txt" extension
2. Upload `cookies.txt` to Lightning AI
3. Update code to use cookies

**Note:** This is difficult on Lightning AI since you don't have a logged-in browser there.

## Why Can't We Fix This in Code?

YouTube's bot detection is **server-side** and IP-based:
- No amount of client switching will help
- User-agent rotation won't work
- Rate limiting won't help (it's instant blocking)
- The only solutions require either:
  - Authenticated requests (cookies/OAuth)
  - Different IP address (proxy/VPN)
  - Manual download from non-flagged network

## Recommended Workflow for Lightning AI

1. **Batch download videos on your local machine:**
   ```bash
   # Create a download script locally
   while read url; do
       yt-dlp --cookies-from-browser chrome \
              --extract-audio \
              --audio-format wav \
              "$url"
       sleep 5  # Rate limiting
   done < video_urls.txt
   ```

2. **Upload audio files to Lightning AI:**
   - Use Lightning AI's file upload interface
   - Or use `rsync`/`scp` if you have SSH access

3. **Process uploaded files on Lightning AI:**
   - Skip YouTube download step
   - Process audio files directly
   - Generate transcripts with Whisper if needed

## Long-term Solution

For production use, consider:
1. **Self-hosted environment** with residential IP
2. **YouTube Data API** (requires OAuth, has quotas)
3. **Pre-downloaded dataset** that you maintain
4. **Proxy service** integrated into your workflow

## Status

- ✅ Code works correctly (client selection, error handling)
- ❌ YouTube blocks Lightning AI's IPs
- ✅ Workarounds available (manual download + upload)
- ⚠️  Automated YouTube downloads on Lightning AI are **not possible** without cookies or proxy

## See Also

- [yt-dlp FAQ - Passing Cookies](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp)
- [YouTube Cookie Export Guide](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies)
- [Lightning AI File Upload Docs](https://lightning.ai/docs)
