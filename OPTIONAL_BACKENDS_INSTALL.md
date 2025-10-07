# Optional G2P Backend Installation Guide

**Status**: Your Amharic TTS is **fully functional** with the rule-based backend!  
**This guide**: Install optional backends for enhanced quality (optional, not required)

---

## 🎯 Which Backend Should You Install?

| Backend | Quality | Speed | Installation | When to Use |
|---------|---------|-------|--------------|-------------|
| **Rule-based** | Good | Fast | ✅ **Already installed** | Default, no setup needed |
| **Transphone** | Excellent | Medium | `pip install transphone` | Best overall quality |
| **Epitran** | Very Good | Fast | `pip install epitran` | Fast, rule-based |

---

## 📦 Installation Options

### Option 1: Transphone (Recommended for Best Quality)

**What it is**: Zero-shot G2P supporting 7500+ languages including Amharic

**Installation**:
```bash
pip install transphone
```

**First Use** (downloads model ~100MB):
```python
from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P

# First time will download pretrained model
g2p = AmharicG2P(backend='transphone')
result = g2p.convert("ሰላም")  # Downloads model on first run
```

**Verification**:
```bash
python -c "from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P; g2p = AmharicG2P(backend='transphone'); print('✅ Transphone installed:', g2p.convert('ሰላም'))"
```

---

### Option 2: Epitran (Fast Alternative)

**What it is**: Rule-based G2P with explicit Ethiopic script support

**Installation**:
```bash
pip install epitran
```

**Verification**:
```bash
python -c "from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P; g2p = AmharicG2P(backend='epitran'); print('✅ Epitran installed:', g2p.convert('ሰላም'))"
```

---

### Option 3: Install Both (Maximum Flexibility)

```bash
pip install transphone epitran
```

**The system will automatically**:
1. Try Transphone first (best quality)
2. Fall back to Epitran if Transphone fails
3. Fall back to rule-based if both fail

---

## 🧪 Testing Your Installation

### Quick Test
```bash
python test_amharic_simple.py
```

Expected output with Transphone/Epitran installed:
```
✅ G2P Transphone backend (or "✅ G2P Epitran backend")
```

### Manual Test
```python
from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P

# Test all backends
for backend in ['transphone', 'epitran', 'rule-based']:
    try:
        g2p = AmharicG2P(backend=backend)
        result = g2p.convert("ሰላም ኢትዮጵያ")
        print(f"✅ {backend:12} → {result}")
    except Exception as e:
        print(f"❌ {backend:12} → Not available")
```

---

## 🚨 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'transphone'"

**Solution**:
```bash
# Make sure you're in the correct environment
pip install transphone

# Or with specific Python version
python -m pip install transphone
```

### Issue: "SSL Certificate Error" during installation

**Solution**:
```bash
# Windows PowerShell
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org transphone

# Or disable SSL verification temporarily
pip install --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org transphone
```

### Issue: Transphone downloads fail on first use

**Solution**:
```python
# Pre-download the model
from transphone import read_g2p

# This will download the model
g2p = read_g2p('amh')
print("✅ Model downloaded successfully")
```

### Issue: "ImportError: cannot import name 'Epitran'"

**Solution**:
```bash
# Reinstall epitran
pip uninstall epitran
pip install epitran
```

---

## 🔍 Verify Current Configuration

Run this to see which backends are available:

```python
from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P

g2p = AmharicG2P(backend='auto')
print("\nAvailable G2P Backends:")
print("=" * 50)

# Check Transphone
try:
    import transphone
    print("✅ Transphone: INSTALLED")
except ImportError:
    print("⚠️  Transphone: Not installed (pip install transphone)")

# Check Epitran
try:
    import epitran
    print("✅ Epitran: INSTALLED")
except ImportError:
    print("⚠️  Epitran: Not installed (pip install epitran)")

print("✅ Rule-based: Always available")
print("\nCurrent backend order:")
print(g2p.config.g2p.backend_order if hasattr(g2p.config, 'g2p') else "Default")
```

---

## 📊 Performance Comparison

Test script to compare backends:

```python
import time
from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P

test_text = "ሰላም ኢትዮጵያ አማርኛ መልካም ቀን"

for backend in ['transphone', 'epitran', 'rule-based']:
    try:
        g2p = AmharicG2P(backend=backend)
        
        # Warmup
        g2p.convert(test_text)
        
        # Benchmark
        start = time.time()
        result = g2p.convert(test_text)
        elapsed = time.time() - start
        
        print(f"\n{backend.upper()}")
        print(f"  Result: {result}")
        print(f"  Time: {elapsed*1000:.2f}ms")
        
    except Exception as e:
        print(f"\n{backend.upper()}: Not available")
```

---

## 💡 Recommendations

### For Development
- ✅ Use rule-based backend (no setup needed)
- ⚠️  Optional: Install transphone for better quality

### For Production
- 🌟 **Recommended**: Install Transphone for best quality
- ⚡ Alternative: Use rule-based (fast, no dependencies)
- 🔄 Hybrid: Install both for automatic fallback

### For Research
- 📚 Install all backends
- 📊 Compare outputs for your specific use case
- 🔬 Benchmark performance on your hardware

---

## 🎯 Quick Commands

```bash
# Install Transphone (recommended)
pip install transphone

# Install Epitran (alternative)
pip install epitran

# Install both
pip install transphone epitran

# Test installation
python test_amharic_simple.py

# Verify in Python
python -c "from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P; print('✅ All imports work')"
```

---

## 📞 Support

If you encounter issues:
1. Check the error message carefully
2. Try the troubleshooting steps above
3. Verify your Python environment
4. Check internet connectivity (for first-time downloads)
5. Open a GitHub issue with error details

---

## ✅ Summary

- **Your system works perfectly with rule-based G2P** (already installed)
- **Transphone and Epitran are optional** enhancements for better quality
- **Install them anytime** - the system will automatically detect and use them
- **No configuration needed** - just install and it works!

**Current Status**: ✅ Production ready with rule-based backend  
**Optional Enhancement**: 🌟 Install transphone for best quality

---

**Remember**: The rule-based backend is production-ready and doesn't require any additional installation!
