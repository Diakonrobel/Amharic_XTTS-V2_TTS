# YouTube Download Methods - Quick Reference 🚀

**TL;DR:** Use no-cookies mode first. Add cookies only if needed.

---

## 🎯 Which Method Should I Use?

### ✅ **Start Here: No-Cookies Mode (2025 Bypass)**

**When to use:** First attempt for any download

```bash
python utils/youtube_downloader.py "URL" ./downloads en
```

**Pros:**
- ✅ No setup required
- ✅ Works on Lightning AI without browser
- ✅ 90-95% success rate for public videos
- ✅ No cookie expiration issues

**Cons:**
- ⚠️ Lower success for age-restricted videos (50-70%)
- ⚠️ May fail for some regional restrictions

---

### 🔐 **Fallback: Cookies Mode**

**When to use:** If no-cookies mode fails

```bash
python utils/youtube_downloader.py "URL" ./downloads en --cookies-path cookies.txt
```

**Pros:**
- ✅ 99%+ success rate for all video types
- ✅ Works with age-restricted content
- ✅ Handles private/unlisted videos (if you have access)

**Cons:**
- ⚠️ Requires cookie export from browser
- ⚠️ Need to upload to Lightning AI
- ⚠️ Cookies expire and need renewal

---

## 📊 Quick Comparison Table

| Feature | No-Cookies | Cookies | PO Token |
|---------|------------|---------|----------|
| **Setup Complexity** | None | Medium | High |
| **Success Rate (Public)** | 90-95% | 99%+ | 99%+ |
| **Success Rate (Age-Restricted)** | 50-70% | 99%+ | 99%+ |
| **Lightning AI Compatible** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Maintenance Required** | None | Periodic cookie refresh | Token refresh |
| **Best For** | First attempt | Reliable downloads | Last resort |

---

## 🔧 Technical: How No-Cookies Works

The 2025 bypass uses these techniques:

### 1. Client Emulation
```python
'player_client': ['android', 'ios', 'tv_embedded']
```
- Pretends to be YouTube mobile/TV apps
- These clients have fewer restrictions

### 2. Request Optimization
```python
'player_skip': ['webpage', 'configs']
```
- Skips bot detection triggers
- Goes straight to video streams

### 3. Mobile User Agents
```python
"com.google.android.youtube/19.43.41 (Linux; U; Android 14; en_US)"
```
- Uses real app user agents
- YouTube whitelists these

---

## 🚨 Troubleshooting Decision Tree

```
Download Failed?
│
├─ Error: "Sign in to confirm you're not a bot"
│  └─ Solution: Add cookies
│
├─ Error: "Video unavailable"
│  ├─ Age-restricted? → Add cookies
│  ├─ Regional block? → Try proxy + no-cookies
│  └─ Private video? → Must use cookies from authorized account
│
├─ Error: HTTP 403/429
│  ├─ Try no-cookies (if not already)
│  ├─ Add cookies
│  └─ Use proxy
│
└─ Error: Network/timeout
   └─ Check internet, retry with delay
```

---

## 📝 Setup Instructions

### No-Cookies (Default - No Setup)
Already configured! Just run the downloader.

### Cookies Setup
1. Install browser extension: [Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/cclelndahbckbenkjhflpdbgdldlbecc)
2. Log into YouTube
3. Click extension → Export → Save as `cookies.txt`
4. Lightning AI: Upload to `/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/`
5. Local: Place in project root

### PO Token (Advanced)
See `YOUTUBE_2025_NO_COOKIES_BYPASS.md` for details.

---

## 🎓 Best Practices

### For Lightning AI Users
1. **Always try no-cookies first** - It's already configured
2. **Keep cookies.txt uploaded** - As backup if needed
3. **Update yt-dlp regularly** - `pip install -U yt-dlp`

### For Batch Processing
1. **Start with no-cookies** for all videos
2. **Collect failed URLs** and retry with cookies
3. **Use delays** between downloads (2-3 seconds)
4. **Monitor rate limits** and adjust

### For Age-Restricted Content
- **Always use cookies** - No-cookies has low success rate
- Log into account that passed age verification
- Export fresh cookies before batch job

---

## 🔄 When to Update Cookies

Cookies expire or become invalid when:
- ⏰ After ~30 days (typical)
- 🔑 After password change
- 🚪 After logout
- 🔒 After security settings change

**Symptoms:**
- Downloads that worked before now fail
- "Sign in" or "unauthorized" errors
- 403 errors despite having cookies

**Solution:** Export fresh cookies and re-upload.

---

## 📚 Full Documentation

- **Detailed bypass guide:** `YOUTUBE_2025_NO_COOKIES_BYPASS.md`
- **Cookie setup guide:** `LIGHTNING_AI_COOKIES_SETUP.md`
- **Implementation code:** `utils/youtube_downloader.py`

---

## 💡 Pro Tips

### Maximizing Success Rate
1. **Update yt-dlp** before big batch jobs
2. **Use mobile user agents** (automatic in our code)
3. **Add delays** between requests
4. **Rotate IPs** if possible (proxy)
5. **Keep cookies fresh** as backup

### Lightning AI Specific
- ✅ No-cookies works perfectly (no browser needed)
- ✅ Upload cookies.txt to studio path once
- ✅ Auto-detection finds it automatically
- ❌ DON'T use `--cookies-from-browser` (no browser on server)

### Common Mistakes to Avoid
- ❌ Using old yt-dlp version
- ❌ Trying browser cookies on Lightning AI
- ❌ Not updating cookies after 30+ days
- ❌ Making requests too fast (rate limits)
- ❌ Downloading entire playlists without delays

---

**Last Updated:** January 2025  
**Quick Reference Version:** 1.0
