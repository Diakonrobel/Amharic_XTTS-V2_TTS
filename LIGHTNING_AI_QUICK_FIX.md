# Lightning.AI YouTube Fix - Quick Guide

## ❌ Problem

You're seeing this error on Lightning.AI:
```
ERROR: could not find chrome cookies database
failed to load cookies
```

**Why:** Lightning.AI doesn't have Chrome installed, so `cookies_from_browser: 'chrome'` fails.

## ✅ Solution (5 minutes)

### Step 1: Export Cookies from Your Local PC

1. **On your Windows/Mac PC**, install browser extension:
   - Chrome: [Get cookies.txt](https://chrome.google.com/webstore/detail/get-cookiestxt/bgaddhkoddajcdgocldbbfleckgcbcid)
   - Firefox: [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)

2. **Log into YouTube** on your local browser

3. **Export cookies**:
   - Click extension icon
   - Select "Export for youtube.com"
   - Save as `youtube_cookies.txt`

### Step 2: Upload to Lightning.AI

1. In Lightning.AI web interface, click **Upload Files**
2. Upload `youtube_cookies.txt` to your project directory
3. Note the full path (e.g., `/teamspace/studios/this_studio/youtube_cookies.txt`)

### Step 3: Use in Your Code

**In the WebUI:**
- Go to "Download/Subtitles Options"
- Set **Cookies file (optional)** to: `/teamspace/studios/this_studio/youtube_cookies.txt`
- Leave **Cookies from browser** EMPTY
- Click "Process All Links"

**In Python code:**
```python
from utils import youtube_downloader, batch_processor, srt_processor

urls = [
    "https://www.youtube.com/watch?v=VIDEO1",
    "https://www.youtube.com/watch?v=VIDEO2",
]

train_csv, eval_csv, video_infos = batch_processor.process_youtube_batch(
    urls=urls,
    transcript_lang="en",
    out_path="./datasets",
    youtube_downloader=youtube_downloader,
    srt_processor=srt_processor,
    # USE THIS:
    cookies_path="/teamspace/studios/this_studio/youtube_cookies.txt",
    # DO NOT USE THIS ON LIGHTNING.AI:
    # cookies_from_browser="chrome",  # ❌ This will fail!
)
```

## 📊 Expected Results

**Before fix:**
```
ERROR: could not find chrome cookies database
failed to load cookies
```

**After fix:**
```
🔐 Authentication: ENABLED (cookies detected)
  Using cookies file: /teamspace/studios/this_studio/youtube_cookies.txt

📥 Download attempt 1/3
  ✓ Browser cookies loaded successfully
  Downloading audio...
  ✅ Audio downloaded successfully

✅ Download successful on attempt 1
```

## ⚠️ Important Notes

1. **Cookies expire** - Re-export every 1-2 weeks
2. **Full path required** - Use absolute path on Lightning.AI
3. **Don't use cookies_from_browser** - Only works on machines with installed browsers
4. **Success rate**: 95%+ with valid cookies

## 🔧 Troubleshooting

### "File not found"
- Check the path is correct
- Use absolute path: `/teamspace/studios/this_studio/youtube_cookies.txt`
- Make sure file was uploaded successfully

### "Invalid cookies"
- Cookies expired - export fresh ones
- Make sure you were logged into YouTube when exporting
- Try logging out and back into YouTube, then export again

### Still failing?
- Check if YouTube is blocking your Lightning.AI IP
- Try using a proxy (see main guide)
- Verify cookies file is not empty

## 📚 More Info

For complete documentation, see:
- `YOUTUBE_2025_COMPLETE_GUIDE.md` - Full setup guide
- `LIGHTNING_AI_YOUTUBE_LIMITATION.md` - Technical explanation

---

**Quick Summary:**
1. Export cookies from local browser
2. Upload to Lightning.AI  
3. Use `cookies_path` instead of `cookies_from_browser`
4. Done! 🎉
