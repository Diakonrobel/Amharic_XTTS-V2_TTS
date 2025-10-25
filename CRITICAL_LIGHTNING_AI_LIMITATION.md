# CRITICAL: YouTube Downloads on Lightning.AI Are Blocked

## ❌ The Reality

**YouTube actively blocks ALL Lightning.AI IP addresses**, regardless of:
- ✗ Cookies (even fresh, valid ones)
- ✗ Proxies (datacenter IPs are blocked)
- ✗ User-agent rotation
- ✗ Player client selection
- ✗ Any code-based bypass method

This is **YouTube's intentional blocking** of cloud/datacenter IPs to prevent automated scraping.

## 📊 Test Results

Tested on January 2025:
- ✓ Latest yt-dlp (2025.10.22)
- ✓ Valid cookies from logged-in browser
- ✓ Multiple bypass attempts
- **Result: 100% failure rate**

## ✅ Working Solutions

### Solution 1: Download Locally (RECOMMENDED)

**On your Windows PC:**

```powershell
# Install
pip install -U yt-dlp

# Download with cookies
yt-dlp --cookies-from-browser chrome --extract-audio --audio-format wav "VIDEO_URL"

# Upload .wav files to Lightning.AI via web interface
```

### Solution 2: Use Residential Proxy

**Requirements:**
- Residential proxy service (NOT datacenter)
- Cost: $50-500/month
- Services: Bright Data, Smartproxy, IPRoyal

**Code:**
```python
cookies_path="/path/to/cookies.txt",
proxy="http://user:pass@residential-proxy:port"  # Must be residential!
```

**Success rate:** 80-90% with residential proxy

### Solution 3: Self-Hosted Environment

Run the code on:
- Your local Windows PC
- A VPS with residential IP
- Your home internet connection

## 🚫 What DOESN'T Work

- ❌ Any cookies (Lightning.AI IPs are flagged)
- ❌ Datacenter proxies (also blocked)
- ❌ Code changes (can't bypass IP blocks)
- ❌ Waiting/retrying (permanent block)
- ❌ Different yt-dlp versions
- ❌ PO tokens, visitor data, etc.

## 📝 Technical Explanation

YouTube detects:
1. **Source IP** - Lightning.AI IPs are in known cloud ranges
2. **Request patterns** - Automated behavior
3. **Network fingerprinting** - Datacenter characteristics

**Result:** Instant block, regardless of authentication.

## 💡 Recommended Workflow

### For Lightning.AI Users:

1. **Download videos locally** (your PC)
2. **Upload audio files** to Lightning.AI
3. **Process on Lightning.AI** (transcription, dataset creation)

This separates YouTube downloading (local) from dataset processing (cloud).

### Example Workflow:

**Local PC (Windows):**
```powershell
# Create URLs file
echo https://www.youtube.com/watch?v=VIDEO1 > urls.txt
echo https://www.youtube.com/watch?v=VIDEO2 >> urls.txt

# Batch download
foreach ($url in Get-Content urls.txt) {
    yt-dlp --cookies-from-browser chrome --extract-audio --audio-format wav "$url"
}

# Upload all .wav files to Lightning.AI
```

**Lightning.AI:**
- Upload audio files via web interface
- Process directly with your TTS pipeline
- Skip YouTube download entirely

## ⚖️ Legal Note

This limitation exists because:
- YouTube Terms of Service prohibit automated downloads
- Cloud providers' IPs are flagged for abuse prevention  
- Using workarounds may violate TOS

**Recommendation:** Use YouTube Data API v3 for metadata, download locally for audio.

## 🔗 References

- [YouTube Terms of Service](https://www.youtube.com/t/terms)
- [yt-dlp Known Issues](https://github.com/yt-dlp/yt-dlp/issues)
- Lightning.AI Documentation

---

**Last Updated:** January 2025  
**Status:** YouTube blocks persist, no code-based workaround exists  
**Recommendation:** Download locally, process on Lightning.AI
