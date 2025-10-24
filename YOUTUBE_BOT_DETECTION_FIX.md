# YouTube Bot Detection Fix (January 2025)

## Problem
When batch processing multiple YouTube videos (e.g., 328 videos), downloads fail with:
```
ERROR: Sign in to confirm you're not a bot
WARNING: Unable to fetch PO Token: Missing required Data Sync ID
```

Even with cookies loaded, rapid-fire requests trigger YouTube's enhanced bot detection.

## Root Causes

### 1. **Rate Limiting Violation**
YouTube enforces strict rate limits:
- **Guest sessions**: ~300 videos/hour (~1000 requests/hour)
- **Authenticated (with cookies)**: ~2000 videos/hour (~4000 requests/hour)

Batch processing without delays exceeds these limits instantly.

### 2. **Suboptimal Player Client Selection**
Using `tv,web_safari,web` clients with cookies is not ideal. The `mweb` (mobile web) client works better with authentication according to yt-dlp 2024 documentation.

### 3. **Missing Authentication Tokens**
YouTube now requires:
- **PO Token** (Proof of Origin Token)
- **Data Sync ID** (Account identifier)

However, these are optional if proper rate limiting and client selection are used.

## Solutions Implemented

### ✅ 1. Rate Limiting (CRITICAL FIX)

**File**: `utils/batch_processor.py`

Added automatic rate limiting between video downloads:

```python
# Rate limiting to avoid YouTube bot detection
RATE_LIMIT_DELAY = 7  # seconds between downloads

# With cookies: ~2000 videos/hour = 1.8s/video minimum
# We use 7 seconds for safety (514 videos/hour)
```

**Benefits**:
- Stays well below YouTube's rate limits (514 videos/hour vs 2000 limit)
- Prevents bot detection triggers
- Shows progress messages with estimated time
- Automatically skips delay after last video

**Output**:
```
🎬 Batch Processing 328 YouTube Videos...
⏱️ Rate Limiting: 7s delay between videos to prevent bot detection
📅 Estimated time: ~38.3 minutes (+ download time)

[1/328] Processing: https://www.youtube.com/watch?v=...
  ✓ Video 1 processed: Example Video Title
  ⏳ Waiting 7s before next video (YouTube rate limit protection)...
```

### ✅ 2. Optimized Player Client Selection

**File**: `utils/youtube_downloader.py`

Changed player client strategy:

**Before**:
```python
'player_client': ['tv', 'web_safari', 'web']  # With cookies
```

**After**:
```python
'player_client': ['mweb', 'tv']  # With cookies - mweb works best
```

**Why `mweb`?**
- Mobile web client has better cookie authentication support
- Less aggressive bot detection
- Recommended by yt-dlp maintainers for authenticated requests (2024)

### ✅ 3. Quote Stripping Fix

**File**: `utils/batch_processor.py`

Fixed URL parsing to handle Excel/CSV exports:

```python
# Strip surrounding quotes (single or double)
url = url.strip('"\'')
```

This prevents errors from URLs like `"https://www.youtube.com/..."`

## Usage

### For Batch Processing (328 videos)

1. **Export Fresh Cookies** (IMPORTANT!)
   ```bash
   # Use incognito/private window method (see YOUTUBE_FIX_2025.md)
   # This ensures cookies don't get rotated by YouTube
   ```

2. **Upload URL List**
   - Create `urls.txt` with one URL per line
   - Upload via new File upload component
   - Or paste URLs in text box

3. **Enable Batch Mode**
   - Check "🎬 Batch Mode" checkbox
   - Set output path
   - Click "Process"

4. **Monitor Progress**
   ```
   [18/328] Processing: https://www.youtube.com/watch?v=...
   ⏳ Waiting 7s before next video...
   ```

5. **Expected Time**
   - With 328 videos + 7s delay: ~38 minutes delay time
   - Plus actual download/processing time per video
   - Total estimate: 1-3 hours depending on video lengths

### For Single/Small Batches

For <10 videos, rate limiting is less critical but still applied for safety.

## Configuration

### Adjust Rate Limit (Advanced)

Edit `utils/batch_processor.py` line ~291:

```python
RATE_LIMIT_DELAY = 7  # Change this value

# Recommended values:
# 3 seconds  = 1200 videos/hour (aggressive, may trigger detection)
# 7 seconds  = 514 videos/hour (safe, recommended)
# 10 seconds = 360 videos/hour (very safe)
```

### Use Different Player Clients

Edit `utils/youtube_downloader.py` line ~856:

```python
'player_client': ['mweb', 'tv']  # Current (recommended)
# Alternatives:
# ['android', 'mweb']  # Android client + mobile web
# ['tv_simply', 'tv']  # TV clients (no auth)
```

## Troubleshooting

### Still Getting Bot Detection?

**1. Increase Rate Limit Delay**
```python
RATE_LIMIT_DELAY = 10  # Slower but safer
```

**2. Export Fresh Cookies**
- Use private/incognito window method
- Navigate to youtube.com/robots.txt FIRST
- Then export cookies immediately
- Never reuse the private window

**3. Use Proxy (Optional)**
```python
youtube_proxy = "http://proxy:port"
```

**4. Try Different Client**
```python
'player_client': ['android', 'mweb']
```

### "This content isn't available" Error

This means you've already hit rate limit. Wait 1 hour, then:
- Use slower rate limit (10s)
- Use fresh cookies
- Consider splitting batch into smaller groups

## Technical Details

### Why 7 Seconds?

YouTube authenticated rate limit: 2000 videos/hour = 1.8s/video minimum

We use 7 seconds (4x the minimum) because:
1. Each video requires multiple requests (video info, download, subtitles)
2. Provides buffer for occasional slower downloads
3. Accounts for processing time between requests
4. Reduces risk of false positives

### Player Client Comparison

| Client | Auth Support | Bot Detection | Speed | Notes |
|--------|-------------|---------------|-------|-------|
| `mweb` | ✅ Excellent | Low | Fast | Recommended with cookies |
| `tv` | ⚠️ Limited | Medium | Fast | Good fallback |
| `web` | ✅ Good | High | Medium | Triggers detection easily |
| `android` | ✅ Good | Low | Fast | Alternative to mweb |
| `tv_simply` | ❌ None | Low | Fast | Guest access only |

### Rate Limit Calculation

```python
# With 7s delay:
videos_per_hour = 3600 / 7 ≈ 514 videos/hour

# YouTube limit: 2000 videos/hour
# Our usage: 514 videos/hour (25.7% of limit)
# Safety margin: 74.3%
```

## Related Files

- `utils/batch_processor.py` - Batch processing with rate limiting
- `utils/youtube_downloader.py` - Download logic with client selection
- `YOUTUBE_FIX_2025.md` - General YouTube download troubleshooting
- `COLAB_SETUP.md` - Google Colab specific setup

## References

- [yt-dlp PO Token Guide](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide)
- [yt-dlp Extractors Wiki](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies)
- [YouTube Rate Limits](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#this-content-isnt-available-try-again-later)

---

**Last Updated**: January 24, 2025  
**yt-dlp Version**: 2024.12.23  
**Status**: ✅ Tested and Working
