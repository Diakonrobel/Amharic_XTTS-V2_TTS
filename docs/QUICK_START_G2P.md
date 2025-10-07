# Quick Start: Amharic G2P Setup

## 🚀 Recommended Setup (1 Minute)

For the **best Amharic TTS quality**, install Transphone:

### Option 1: Quick Install
```bash
python setup_amharic_g2p.py --transphone-only --auto
```

### Option 2: Interactive Setup
```bash
python setup_amharic_g2p.py
```

### Option 3: Manual Install
```bash
pip install transphone
```

That's it! The system will automatically use Transphone for state-of-the-art G2P quality.

---

## 📊 Check Backend Status

```bash
python setup_amharic_g2p.py --check-only
```

Output:
```
======================================================================
📊 G2P Backend Status
======================================================================
Transphone (Primary)           ✅ Available         ⭐ Best accuracy
Epitran (Backup)               ❌ Not available     Good accuracy
Rule-Based (Fallback)          ✅ Available         Always available
======================================================================
```

---

## 🎯 Usage Examples

### With Transphone (Recommended)

```python
from amharic_tts.g2p.amharic_g2p_enhanced import EnhancedAmharicG2P

# Initialize (will use Transphone automatically if installed)
g2p = EnhancedAmharicG2P()

# Convert Amharic text to phonemes
phonemes = g2p.convert("ሰላም ዓለም")
print(phonemes)  # Output: səlamɨ ʔələmɨ
```

### Without Transphone (Still Works!)

```python
from amharic_tts.g2p.amharic_g2p_enhanced import EnhancedAmharicG2P

# Initialize (will use rule-based backend)
g2p = EnhancedAmharicG2P()

# Convert Amharic text to phonemes
phonemes = g2p.convert("ሰላም ዓለም")
print(phonemes)  # Output: səlamɨ ʔələmɨ (same output!)
```

**Note:** The rule-based backend is production-ready and provides high-quality output even without Transphone!

---

## 🏗️ Architecture Overview

```
┌─────────────────────────┐
│   First Use             │
│   Amharic G2P           │
└───────────┬─────────────┘
            │
            ▼
    ┌───────────────┐
    │ Is Transphone │
    │  installed?   │
    └───────┬───────┘
            │
      ┌─────┴─────┐
      │           │
     YES         NO
      │           │
      ▼           ▼
┌──────────┐  ┌──────────────────┐
│ Use      │  │ Show friendly    │
│ Transphone│  │ install message  │
│ (⭐ Best)│  │                  │
└──────────┘  │ Offer to install │
              │                  │
              └────────┬─────────┘
                       │
                ┌──────┴──────┐
                │ User        │
                │ installs?   │
                └──────┬──────┘
                       │
                 ┌─────┴─────┐
                 │           │
                YES         NO
                 │           │
                 ▼           ▼
            ┌────────┐  ┌──────────┐
            │ Use    │  │ Use      │
            │Transphone││Rule-Based│
            └────────┘  └──────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Still produces│
                    │ high-quality  │
                    │ output! ✅    │
                    └───────────────┘
```

---

## 🔧 Setup Options

### 1. **Auto-Install Everything** (Easiest)
```bash
python setup_amharic_g2p.py --auto
```
- Installs Transphone + Epitran
- No prompts
- Best for automation

### 2. **Interactive Setup** (Recommended)
```bash
python setup_amharic_g2p.py
```
- Prompts for each backend
- Lets you choose what to install
- Best for first-time setup

### 3. **Transphone Only** (Minimal)
```bash
python setup_amharic_g2p.py --transphone-only
```
- Only installs Transphone
- Fastest setup
- Recommended for most users

### 4. **No Installation** (Zero Dependencies)
```bash
# Just use the system as-is!
# Rule-based backend always works
```
- No setup needed
- Rule-based G2P is production-ready
- Perfect for air-gapped systems

---

## 📦 What Gets Installed?

### Transphone (~100MB)
```bash
pip install transphone
```

**What it does:**
- Downloads pretrained G2P model on first use (~100MB)
- Provides state-of-the-art accuracy
- Zero-shot learning for rare words
- Supports 7,546 languages including Amharic

**Disk space:** ~150MB total

### Epitran (Optional, ~20MB)
```bash
pip install epitran
```

**What it does:**
- Rule-based G2P system
- Fast and accurate
- Good backup to Transphone

**Disk space:** ~50MB total

---

## 🎓 Understanding the Backends

### Transphone (Primary) ⭐⭐⭐⭐⭐
- **Accuracy**: Best (state-of-the-art)
- **Speed**: Medium
- **Setup**: `pip install transphone`
- **Use case**: Best quality TTS

### Epitran (Backup) ⭐⭐⭐⭐
- **Accuracy**: Good
- **Speed**: Fast
- **Setup**: `pip install epitran`
- **Use case**: Backup when Transphone unavailable

### Rule-Based (Fallback) ⭐⭐⭐⭐
- **Accuracy**: High (259 mappings, IPA-compliant)
- **Speed**: Very fast
- **Setup**: None (always available)
- **Use case**: Always works, zero dependencies

**Important:** All three backends produce high-quality output. Transphone is just slightly better for rare words!

---

## 💡 FAQs

### Q: Is Transphone required?
**A:** No! The rule-based backend is production-ready and always works. Transphone just provides a slight quality improvement.

### Q: What if I don't want to install anything?
**A:** That's fine! The system works perfectly with the built-in rule-based backend.

### Q: How much better is Transphone?
**A:** Transphone excels at rare/unseen words. For common Amharic text, the difference is minimal.

### Q: Can I install Transphone later?
**A:** Yes! Just run `pip install transphone` anytime.

### Q: Will the system prompt me every time?
**A:** No. The install prompt only shows once on first use. After that, it just uses whatever's available.

### Q: Does Transphone require internet?
**A:** Only for the initial model download (~100MB). After that, it works offline.

---

## 🔍 Verification

After installation, verify everything works:

```python
from amharic_tts.g2p.amharic_g2p_enhanced import EnhancedAmharicG2P

# Test G2P
g2p = EnhancedAmharicG2P()

# Try some Amharic text
test_words = ["ሰላም", "ኢትዮጵያ", "አማርኛ", "ቋንቋ"]

for word in test_words:
    phonemes = g2p.convert(word)
    print(f"{word:15} → {phonemes}")

# Check which backend is being used
config = g2p.get_config() if hasattr(g2p, 'get_config') else {}
print(f"\nActive backends: {config}")
```

Expected output:
```
ሰላም             → səlamɨ
ኢትዮጵያ           → ʔitɨjopʼɨja
አማርኛ            → ʔəmarɨɲa
ቋንቋ             → qʷanɨqʷa
```

---

## 🚨 Troubleshooting

### Transphone not detected
```bash
# Check if installed
pip list | grep transphone

# If not found, install
pip install transphone

# Verify
python -c "import transphone; print('✅ Transphone available')"
```

### Model download fails
```bash
# Transphone will try to download model on first use
# If it fails, check internet connection or try:
python -m transphone.download --lang amh
```

### ImportError
```bash
# Make sure you're in the project directory
cd D:\FINETUNE-XTTS-WEBUI-LIGHTNING\xtts-finetune-webui-fresh

# Run setup
python setup_amharic_g2p.py
```

---

## 📝 Summary

| Setup Method | Time | Best For |
|-------------|------|----------|
| `python setup_amharic_g2p.py --transphone-only --auto` | 1 min | **Recommended** - Quick & best quality |
| `python setup_amharic_g2p.py` | 2 min | Interactive first-time setup |
| `pip install transphone` | 1 min | Manual installation |
| No installation | 0 sec | Air-gapped systems, testing |

**Bottom line:** Install Transphone for best quality, but the system works great even without it!

---

## 🔗 Related Documentation

- [Complete G2P Backend Guide](./G2P_BACKENDS_EXPLAINED.md)
- [Phase 3 Technical Documentation](./AMHARIC_G2P_PHASE3.md)
- [Configuration Options](../amharic_tts/config/amharic_config.py)

---

**Need help?** Run `python setup_amharic_g2p.py --help`
