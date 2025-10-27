# 🍪 YouTube Cookies Setup for Lightning AI

## Problem
YouTube detects bot-like behavior and blocks downloads on remote servers like Lightning AI with the error:
```
ERROR: [youtube] Sign in to confirm you're not a bot
```

## Solution
Export YouTube cookies from your **local browser** and upload them to Lightning AI.

---

## 📋 Step-by-Step Instructions

### Step 1: Export Cookies from Your Local Browser

#### Method A: Using Browser Extension (Recommended ⭐)

1. **Install the Extension:**
   - **Chrome/Edge**: [Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
   - **Firefox**: [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)

2. **Export Cookies:**
   - Open YouTube.com and make sure you're **logged in**
   - Click the extension icon in your browser toolbar
   - Click **"Export"** or **"Export As"**
   - Save the file as `cookies.txt`

#### Method B: Using yt-dlp (Alternative)

On your **local machine** (where Chrome is installed):
```bash
yt-dlp --cookies-from-browser chrome --cookies cookies.txt --skip-download https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

---

### Step 2: Upload Cookies to Lightning AI

1. **Open Lightning AI Studio** in your browser

2. **Navigate to your project:**
   ```
   /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/
   ```

3. **Upload the file:**
   - Click the **Upload** button in the file browser
   - Select your `cookies.txt` file
   - Upload it to the project root directory

   **OR** use the Lightning AI terminal:
   ```bash
   # If you have the file on your local machine, use scp or the web interface
   # The file should be at:
   /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/cookies.txt
   ```

---

### Step 3: Verify Setup

The code will **automatically detect** the cookies file in these locations:
1. `/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/cookies.txt`
2. `/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/youtube_cookies.txt`
3. Current working directory + `cookies.txt`
4. Current working directory + `youtube_cookies.txt`

When you run the downloader, you should see:
```
✅ Auto-detected cookies file: /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/cookies.txt
🔐 Authentication: ENABLED (cookies detected)
  Using cookies file: /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/cookies.txt
```

---

### Step 4: Test Download

Run your YouTube download script. The downloads should now work!

---

## ⚠️ Important Notes

### Security
- **DO NOT commit `cookies.txt` to Git!** It contains sensitive authentication tokens.
- Add to `.gitignore`:
  ```gitignore
  cookies.txt
  youtube_cookies.txt
  ```

### Cookie Expiration
- YouTube cookies expire after some time (usually weeks/months)
- If downloads start failing again with "Sign in" errors, **re-export fresh cookies**

### Privacy
- Cookies contain your YouTube session information
- Only use cookies from **your own account**
- Don't share your cookies file with others

---

## 🐛 Troubleshooting

### "Cookies file not found"
- Verify the file is uploaded to: `/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/cookies.txt`
- Check file permissions: `ls -la cookies.txt`
- Ensure the filename is exactly `cookies.txt` (not `cookies.txt.txt`)

### Still getting "Sign in" errors
1. **Re-export cookies** - they may have expired
2. Make sure you're **logged into YouTube** when exporting
3. Try logging out and back into YouTube, then re-export
4. Check if your YouTube account has restrictions or is suspended

### "Format is not available"
This is different from authentication errors. Means:
- Video is region-locked
- Video quality/format not available
- Try different videos to isolate the issue

---

## 🔧 Manual Configuration (Advanced)

If auto-detection doesn't work, you can manually specify the cookies path:

```python
# In your code
youtube_downloader.download_and_process_youtube(
    url="...",
    cookies_path="/path/to/your/cookies.txt",  # Manual path
    cookies_from_browser=None,  # Don't use browser cookies
)
```

Or in `xtts_demo.py` when calling batch processing:
```python
process_youtube_batch(
    urls=urls,
    cookies_path="/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/cookies.txt",
)
```

---

## ✅ Success Indicators

When everything is working correctly, you'll see:
- ✅ Auto-detected cookies file
- 🔐 Authentication: ENABLED
- ✓ Browser cookies loaded successfully (or) Using cookies file
- ✅ Audio downloaded successfully

---

## 📚 Additional Resources

- [yt-dlp Authentication Documentation](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp)
- [YouTube Cookie Export Guide](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies)
- [Get cookies.txt Extension](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)

---

## 🎯 Quick Reference

```bash
# Lightning AI file location
/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/cookies.txt

# Check if file exists
ls -la /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/cookies.txt

# Verify file format (should start with "# Netscape HTTP Cookie File")
head -n 1 /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/cookies.txt
```

---

**Last Updated:** 2025-01-27  
**Applies to:** Lightning AI remote environments where browsers are not installed
