# 🌩️ Lightning AI Testing Guide

## Step-by-Step Testing on Lightning AI

### **Step 1: Pull Latest Changes from GitHub**

```bash
# SSH into your Lightning AI workspace
ssh s_01k7x54qcrv1atww40z8bxf9a3@ssh.lightning.ai

# Navigate to your project
cd /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/

# Pull latest changes
git pull origin main
```

**Expected Output:**
```
Updating ca8d1d5..ce9bc0c
Fast-forward
 HYBRID_G2P_SYSTEM.md                                  | 402 +++++++++++++++++++
 LIGHTNING_AI_TESTING_GUIDE.md                         | 250 ++++++++++++
 amharic_tts/g2p/hybrid_g2p.py                        | 651 ++++++++++++++++++++++++++++
 amharic_tts/preprocessing/ethiopian_numeral_expander.py | 406 +++++++++++++++++++
 amharic_tts/preprocessing/prosody_handler.py          | 432 +++++++++++++++++++
 test_hybrid_g2p_system.py                             | 261 +++++++++++
 utils/g2p_backend_selector.py                         |   6 +-
 xtts_demo.py                                          |  36 +-
 8 files changed, 2432 insertions(+), 12 deletions(-)
```

---

### **Step 2: Run Comprehensive Test Suite**

```bash
# Run the complete test suite
python test_hybrid_g2p_system.py
```

**Expected Output:**
```
🚀 HYBRID G2P SYSTEM - COMPREHENSIVE TEST SUITE
================================================================================

✅ PASS     - Ethiopian Numerals
✅ PASS     - Prosody Handler
✅ PASS     - Hybrid G2P
✅ PASS     - Backend Selector
✅ PASS     - Integration

Total: 5/5 tests passed (100%)

🎉 ALL TESTS PASSED! System is ready for production.
```

---

### **Step 3: Test Individual Components**

#### **A. Test Ethiopian Numeral Expander**

```bash
cd /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/amharic_tts/preprocessing
python ethiopian_numeral_expander.py
```

**What to Check:**
- ✅ All test cases pass (17+/18 tests)
- ✅ Ethiopian numerals convert to Arabic numbers correctly
- ✅ Text expansion works: `፲፰፻፹፯` → Arabic number
- ✅ Amharic word expansion (if number_expander available)

#### **B. Test Prosody Handler**

```bash
cd /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/amharic_tts/preprocessing
python prosody_handler.py
```

**What to Check:**
- ✅ Pause detection works (short, medium, long)
- ✅ Intonation markers added (rising, falling, emphatic)
- ✅ Code-switching boundaries detected
- ✅ Statistics calculated correctly

#### **C. Test Hybrid G2P System**

```bash
cd /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/amharic_tts/g2p
python hybrid_g2p.py
```

**What to Check:**
- ✅ All backends load (Preprocessing, Epitran, Rule-based)
- ✅ Conversions work for pure Amharic
- ✅ Conversions work for mixed Am+En text
- ✅ Code-switching detected (language: "mixed")
- ✅ Cache statistics show hits/misses
- ✅ Performance metrics look reasonable

---

### **Step 4: Test in WebUI (Safe Simulation)**

#### **A. Start Gradio UI**

```bash
cd /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/
python xtts_demo.py
```

#### **B. Open WebUI**

Lightning AI will provide a URL like:
```
https://xxxxx-8000.lightning-staging.ai
```

#### **C. Test Data Processing Tab**

1. Go to **📁 Data Processing** tab
2. Check **G2P Backend dropdown**:
   - ✅ Should show: `hybrid, epitran, transphone, rule_based`
   - ✅ Default should be: `hybrid`
   - ✅ Info text should say: "hybrid = BEST (epitran+rule_based+preprocessing)"

3. **DO NOT process real dataset yet** - just verify the UI elements exist

#### **D. Test Fine-tuning Tab**

1. Go to **🔧 Fine-tuning** tab
2. Scroll to **🇪🇹 Amharic G2P Options**
3. Check **G2P Backend dropdown**:
   - ✅ Should show: `hybrid, epitran, transphone, rule_based`
   - ✅ Default should be: `hybrid`
   - ✅ Info text should say: "hybrid = BEST (all features + code-switching)"

---

### **Step 5: Test with Small Sample Data** (RECOMMENDED)

Create a tiny test dataset to verify everything works end-to-end without risking your main data.

#### **A. Create Test Dataset**

```bash
# Create test directory
mkdir -p /teamspace/studios/this_studio/test_hybrid_g2p
cd /teamspace/studios/this_studio/test_hybrid_g2p

# Create test audio file (1 second silence)
ffmpeg -f lavfi -i anullsrc=r=22050:cl=mono -t 1 test_audio.wav

# Create test transcription
echo "ሰላም ዓለም። በ፲፰፻፹፯ ዓ.ም ተወለደ። Hello World!" > test_transcript.txt
```

#### **B. Process Test Dataset in WebUI**

1. Go to **Data Processing** tab
2. Upload `test_audio.wav`
3. Manual transcription: Copy from `test_transcript.txt`
4. Language: **amh**
5. Enable G2P: **✓ Checked**
6. G2P Backend: **hybrid**
7. Output path: `/teamspace/studios/this_studio/test_hybrid_g2p/output`
8. Click **Create Dataset**

**Expected Behavior:**
- ✅ Processing starts without errors
- ✅ Console shows: "Initializing Hybrid G2P"
- ✅ Console shows: "✅ Hybrid G2P system initialized successfully"
- ✅ Ethiopian numerals expand: `፲፰፻፹፯` → Amharic words
- ✅ Abbreviations expand: `ዓ.ም` → "ዓመተ ምህረት"
- ✅ Amharic converts to IPA: `ሰላም` → `səlam`
- ✅ English preserved or converted (based on config)
- ✅ Dataset files created in output folder

#### **C. Inspect Output**

```bash
# Check generated dataset
ls -lh /teamspace/studios/this_studio/test_hybrid_g2p/output/dataset/

# View metadata CSV
head /teamspace/studios/this_studio/test_hybrid_g2p/output/dataset/metadata_train.csv

# Check if text is converted to phonemes
cat /teamspace/studios/this_studio/test_hybrid_g2p/output/dataset/metadata_train.csv | cut -d'|' -f2
```

**What to Check:**
- ✅ `metadata_train.csv` exists
- ✅ `metadata_eval.csv` exists
- ✅ Text column contains IPA phonemes (not raw Amharic)
- ✅ Ethiopian numerals are expanded
- ✅ Abbreviations are expanded

---

### **Step 6: Performance Benchmarks** (OPTIONAL)

Test the system performance on Lightning AI hardware.

```bash
cd /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/

# Create benchmark script
python -c "
from amharic_tts.g2p.hybrid_g2p import HybridAmharicG2P
import time

g2p = HybridAmharicG2P()

test_texts = [
    'ሰላም ዓለም',
    'በ፲፰፻፹፯ ዓ.ም ተወለደ',
    'ዶ.ር አብርሃም በ 2025 ዓ.ም ተመረቁ',
    'ከዕለታት አንድ ቀን። When they reached a tree.',
] * 250  # 1000 conversions total

print('Running 1000 conversions...')
start = time.time()
results = g2p.convert_batch(test_texts, show_progress=True)
elapsed = time.time() - start

print(f'Total time: {elapsed:.2f}s')
print(f'Speed: {len(test_texts)/elapsed:.1f} conversions/second')

stats = g2p.get_statistics()
print(f'Cache hit rate: {stats['cache_hit_rate']:.1%}')
"
```

**Expected Performance on Lightning AI:**
- First run (cold cache): ~10-20 conversions/second
- With caching: ~100-200 conversions/second
- Cache hit rate: >75% on repeated text

---

### **Step 7: Cleanup Test Data**

```bash
# Remove test directory after verification
rm -rf /teamspace/studios/this_studio/test_hybrid_g2p
```

---

## 🚨 Common Issues & Solutions

### Issue: "Module not found: amharic_tts"

**Solution:**
```bash
# Ensure you're in the correct directory
cd /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS"

# Or run from project root
python -m amharic_tts.g2p.hybrid_g2p
```

### Issue: "Epitran not available"

**Solution:**
```bash
# Install epitran
pip install epitran

# Verify installation
python -c "import epitran; print('Epitran OK')"
```

### Issue: "Preprocessing modules not fully available"

**This is OK!** The system will fall back to epitran-only mode, which still works well.

**To fix (optional):**
```bash
# Ensure all dependencies installed
pip install -r requirements.txt
```

### Issue: "Ethiopian numerals not expanding to words"

**Check:**
```python
from amharic_tts.preprocessing.ethiopian_numeral_expander import EthiopianNumeralExpander

expander = EthiopianNumeralExpander()
print(expander.expand("፲፰፻፹፯"))  # Should print Arabic number or Amharic words
```

---

## ✅ Success Criteria

Your system is ready for production if:

✅ **All 5 tests pass** in comprehensive test suite  
✅ **Hybrid backend** appears in WebUI dropdowns  
✅ **Hybrid is default** in both Data Processing and Training tabs  
✅ **Test dataset processing** completes without errors  
✅ **Output CSV** contains IPA phonemes (not raw Amharic)  
✅ **Ethiopian numerals** expand correctly  
✅ **Code-switching** works (Am+En in same text)  
✅ **Performance** is acceptable (>5 conversions/second)  

---

## 🎯 Next Steps After Successful Testing

### 1. **Process Your Real Dataset**

Now that everything is tested, use hybrid G2P on your actual dataset:

```bash
# In WebUI: Data Processing tab
- Upload your audio files
- Language: amh
- Enable G2P: ✓
- Backend: hybrid
- Process!
```

### 2. **Train Your Model**

```bash
# In WebUI: Fine-tuning tab
- Load parameters
- Enable G2P: ✓
- Backend: hybrid
- Start training!
```

### 3. **Monitor Performance**

Keep an eye on:
- Training loss convergence
- Validation metrics
- Generated audio quality
- G2P cache hit rate (in console logs)

---

## 📊 Expected Behavior Summary

| Component | Expected Behavior |
|-----------|------------------|
| **Ethiopian Numerals** | ፲፰፻፹፯ → "አሥራ ስምንት መቶ..." |
| **Abbreviations** | ዶ.ር → "ዶክተር", ዓ.ም → "ዓመተ ምህረት" |
| **Amharic Text** | ሰላም → səlam (IPA) |
| **Code-Switching** | "ሰላም Hello" → "səlam hello" |
| **Prosody** | "ሰላም!" → Pause+emphatic markers |
| **Caching** | 2nd+ conversions 10-20x faster |
| **Fallback** | If epitran fails → rule_based |

---

## 🔧 Debugging Commands

```bash
# Check Git status
git status
git log --oneline -5

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Test imports
python -c "from amharic_tts.g2p.hybrid_g2p import HybridAmharicG2P; print('OK')"

# Check epitran
python -c "import epitran; e = epitran.Epitran('amh-Ethi'); print(e.transliterate('ሰላም'))"

# View system info
python test_hybrid_g2p_system.py --full
```

---

**Ready to test?** Start with **Step 1** and work through each step! 🚀

**Questions?** Check `HYBRID_G2P_SYSTEM.md` for detailed API reference and troubleshooting.
