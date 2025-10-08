# Troubleshooting Guide - Amharic XTTS Fine-tuning WebUI

This document contains solutions to common issues encountered while running the Amharic XTTS Fine-tuning WebUI.

---

## 📋 Table of Contents

1. [Language Code Issues](#language-code-issues)
2. [Slider Validation Errors](#slider-validation-errors)
3. [PyTorch 2.6 Compatibility](#pytorch-26-compatibility)
4. [Gradio Interface Issues](#gradio-interface-issues)
5. [Training Issues](#training-issues)

---

## 🌍 Language Code Issues

### Issue 1: "Value: am is not in the list of choices"

**Error Message:**
```
gradio.exceptions.Error: "Value: am is not in the list of choices: 
['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh', 'hu', 'ko', 'ja', 'amh']"
```

**Cause:**
The Amharic language has two standard ISO codes:
- ISO 639-1: `am` (2-letter code)
- ISO 639-3: `amh` (3-letter code)

Some UI dropdowns only included `amh`, causing errors when `am` was used.

**Solution:**
✅ **Fixed in commit `d66d843`**

All language dropdowns now support both codes:
- Data Processing tab → Dataset Language dropdown
- Training tab → G2P activation check
- Inference tab → TTS Language dropdown
- Headless script → `--lang` argument

**Files Modified:**
- `xtts_demo.py` (lines 259, 1023, 1155)
- `headlessXttsTrain.py` (line 689)

**Code Example:**
```python
# Before (caused error):
choices=["en", "es", ..., "amh"]  # Missing "am"

# After (fixed):
choices=["en", "es", ..., "am", "amh"]  # Both codes supported
info="Use 'am' or 'amh' for Amharic"
```

---

## 📊 Slider Validation Errors

### Issue 2: "Value 1 is less than minimum value 2"

**Error Message:**
```
gradio.exceptions.Error: 'Value 1 is less than minimum value 2.'
```

**Cause:**
The **Gradient Accumulation** slider had a minimum value of `2`, but the default argument value was `1`:
- Command-line arg: `--grad_acumm` default=`1` (line 216)
- Slider setting: `minimum=2` (line 948)

When Gradio tried to initialize the slider with `value=1`, it failed validation.

**Solution:**
✅ **Fixed in commit `aca4904`**

Changed slider minimum from `2` to `1`:

```python
# Before (caused error):
grad_acumm = gr.Slider(
    label="Grad Accumulation",
    minimum=2,  # ← Problem!
    maximum=128,
    value=args.grad_acumm  # = 1
)

# After (fixed):
grad_acumm = gr.Slider(
    label="Grad Accumulation",
    minimum=1,  # ← Fixed!
    maximum=128,
    value=args.grad_acumm
)
```

**What This Means:**
- Gradient accumulation can now be set from **1 to 128**
- Setting to `1` = no accumulation (normal batch processing)
- Higher values (2, 4, 8) = effectively multiply batch size with less GPU memory

---

## 🔧 PyTorch 2.6 Compatibility

### Issue 3: Weights-only Load Failed / UnpicklingError

**Error Message:**
```
_pickle.UnpicklingError: Weights only load failed. 
This file can still be loaded, to do so you have two options...
PyTorch 2.6 changed the default value of the `weights_only` argument 
in `torch.load` from `False` to `True`.

WeightsUnpickler error: Unsupported global: 
GLOBAL TTS.tts.configs.xtts_config.XttsConfig was not an allowed global by default.
```

**Cause:**
PyTorch 2.6+ introduced a security change:
- **Before PyTorch 2.6**: `torch.load(..., weights_only=False)` (default)
- **PyTorch 2.6+**: `torch.load(..., weights_only=True)` (default)

XTTS model checkpoints contain custom classes (`XttsConfig`, etc.) that require `weights_only=False` to load properly.

**Solution:**
✅ **Fixed in commit `3d10cfa`**

Created automatic compatibility patch that monkey-patches `torch.load`:

**New File:** `utils/pytorch26_patch.py`

**How It Works:**
1. Detects PyTorch version on startup
2. If PyTorch 2.6+, patches `trainer.io.load_fsspec` function
3. Adds `weights_only=False` parameter to `torch.load` calls
4. Only applies to trusted XTTS checkpoints from official sources

**Integration:**
```python
# xtts_demo.py (line 10-15)
try:
    from utils.pytorch26_patch import apply_pytorch26_compatibility_patches
    apply_pytorch26_compatibility_patches()
except Exception as e:
    print(f"Warning: Could not apply PyTorch 2.6 patches: {e}")
```

**Security Note:**
⚠️ Only use this patch with checkpoints from **trusted sources** (official XTTS models from Coqui). The `weights_only=False` parameter allows executing arbitrary Python code during unpickling, which could be dangerous with untrusted checkpoints.

**Alternative Manual Fix:**
If the patch doesn't work, you can manually edit the TTS library:

1. Find: `site-packages/trainer/io.py` (line ~83)
2. Change:
   ```python
   # Before:
   return torch.load(f, map_location=map_location, **kwargs)
   
   # After:
   return torch.load(f, map_location=map_location, weights_only=False, **kwargs)
   ```

---

## 🖥️ Gradio Interface Issues

### Issue 4: UI Not Loading / Blank Screen

**Symptoms:**
- WebUI starts but shows blank screen
- Console shows no errors
- Browser console shows JavaScript errors

**Possible Causes:**
1. **Port conflict**: Another process using port 5003
2. **Gradio version mismatch**: Old/incompatible Gradio version
3. **Browser cache**: Stale cached assets

**Solutions:**

**A. Change Port:**
```bash
python xtts_demo.py --port 5004
```

**B. Clear Browser Cache:**
- Chrome/Edge: `Ctrl+Shift+Delete` → Clear cached images and files
- Firefox: `Ctrl+Shift+Delete` → Cache
- Or use Incognito/Private mode

**C. Update Gradio:**
```bash
pip install --upgrade gradio
```

**D. Check Firewall:**
Ensure port 5003 (or your chosen port) is not blocked by firewall.

---

### Issue 5: "Function didn't return enough output values"

**Error Message:**
```
ValueError: A function (train_model) didn't return enough output values 
(needed: 6, returned: 5).
```

**Cause:**
The `train_model` function's return statement doesn't match the expected number of output components defined in Gradio.

**Solution:**
Check the function signature and ensure it returns exactly 6 values:

```python
# Correct return format:
return (
    "Training complete!",     # 1. Status message (Label)
    config_path,              # 2. Config path (Textbox)
    vocab_file,               # 3. Vocab path (Textbox)
    ft_xtts_checkpoint,       # 4. Checkpoint path (Textbox)
    speaker_xtts_path,        # 5. Speaker path (Textbox)
    speaker_reference_path    # 6. Reference audio path (Textbox)
)

# Error handling must also return 6 values:
return (
    "Error message",          # 1. Error status
    "", "", "", "", ""        # 2-6. Empty strings for remaining outputs
)
```

---

## 🎓 Training Issues

### Issue 6: "Dataset too short" / "Audio total len is less than 2 minutes"

**Error Message:**
```
The sum of the duration of the audios that you provided should be at least 2 minutes!
```

**Cause:**
XTTS requires a minimum of **2 minutes** of total audio for training.

**Solutions:**

**A. Add More Audio:**
Upload more audio files or use longer recordings.

**B. Use Advanced Processing:**
- **YouTube Processing**: Download and process YouTube videos with Amharic audio
- **SRT Processing**: Extract audio from subtitle-synced media
- **Audio Slicer**: Split long recordings into training segments

**C. Check Audio Quality:**
Ensure audio files are:
- Valid format (WAV, MP3, FLAC)
- Not corrupted
- Contain actual audio (not silence)

---

### Issue 7: Low Dataset Segment Count

**Symptoms:**
- Processing YouTube videos produces very few segments
- Expected 1000+ segments, got only 25-50

**Causes:**
1. **Subtitle filtering too strict**: Duration filters rejecting segments
2. **Short subtitle segments**: Amharic subtitles often have very short segments
3. **Large gaps**: Subtitle timing has large gaps between segments

**Solutions:**

**A. Enable Aggressive Merging:**
When processing SRT/YouTube, the system now has intelligent segment merging:
- Merges short adjacent segments
- Bridges small gaps
- Produces longer, more natural audio-text pairs

**B. Adjust Parameters:**
In future versions, you can configure:
```python
min_duration = 3.0    # Minimum segment length (seconds)
max_duration = 15.0   # Maximum segment length (seconds)
max_gap = 3.0         # Maximum gap to bridge (seconds)
```

**C. Verify Subtitles:**
Check that subtitle files:
- Have proper timing information
- Contain text (not just timestamps)
- Match the audio language

---

### Issue 8: Amharic G2P Not Working

**Symptoms:**
- G2P checkbox enabled but no phoneme conversion
- Training uses raw Ethiopic characters

**Diagnostic Steps:**

**1. Check G2P Availability:**
```python
from amharic_tts.g2p.amharic_g2p_enhanced import EnhancedAmharicG2P
g2p = EnhancedAmharicG2P()
result = g2p.convert("ሰላም")
print(result)  # Should output: "səlamɨ"
```

**2. Verify Language Code:**
Ensure you're using `am` or `amh` (not `en` or other codes).

**3. Check Backend:**
Available G2P backends (with fallback):
- `transphone` (best quality, requires internet)
- `epitran` (good quality, local)
- `rule_based` (always available, built-in)

**4. Review Logs:**
Look for messages like:
```
> Amharic G2P mode enabled: Text will be converted to IPA phonemes
> Creating Amharic tokenizer with G2P preprocessing...
```

---

### Issue 9: TTS Tokenizer "Language 'am' is not supported"

**Error Message:**
```
NotImplementedError: Language 'am' is not supported.
```

**Cause:**
The TTS library's tokenizer only recognizes `"amh"` (ISO 639-3), not `"am"` (ISO 639-1).

This error occurs during training when the dataset metadata contains `"am"` but the XTTS tokenizer expects `"amh"`.

**Solution:**
✅ **Fixed in commit `b91ab17`**

Added automatic language code normalization:

```python
def normalize_xtts_lang(lang: str) -> str:
    """Normalize user language code to XTTS-supported code."""
    if lang in ("am", "amh"):
        return "amh"  # XTTS requires 'amh'
    return lang
```

**Where Applied:**
- Training: `train_gpt()` function automatically converts `"am"` → `"amh"`
- Inference: `run_tts()` function automatically converts `"am"` → `"amh"`
- G2P: Accepts both `"am"` and `"amh"` for activation

**User Experience:**
- ✅ Users can select **"am"** in the UI
- ✅ Internally normalized to **"amh"** for XTTS
- ✅ Both codes work seamlessly

**Manual Fix (if needed):**
If you created datasets before this fix, update the language in `lang.txt`:
```bash
echo "amh" > output/dataset/lang.txt
```

---

## 🔍 General Debugging Tips

### Enable Verbose Logging

**Modify log level:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check System Requirements

**Minimum:**
- Python 3.8+
- PyTorch 1.13+
- CUDA 11.8+ (for GPU training)
- 8GB RAM (16GB recommended)
- 10GB disk space

**Check versions:**
```bash
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import gradio; print(f'Gradio: {gradio.__version__}')"
nvidia-smi  # Check GPU availability
```

### Clear Cache and Restart

**GPU Cache:**
```python
import torch
torch.cuda.empty_cache()
```

**Restart Application:**
```bash
# Stop current process (Ctrl+C)
# Clear temporary files
rm -rf /tmp/gradio_*

# Restart
./launch.sh
```

---

## 📝 Quick Reference: All Fixes

| Issue | Commit | Files Changed | Status |
|-------|--------|---------------|--------|
| Missing "am" in language dropdowns | `d66d843` | `xtts_demo.py`, `headlessXttsTrain.py` | ✅ Fixed |
| Grad accumulation slider minimum | `aca4904` | `xtts_demo.py` | ✅ Fixed |
| PyTorch 2.6 weights_only error | `3d10cfa` | `pytorch26_patch.py`, `xtts_demo.py`, `gpt_train.py` | ✅ Fixed |
| TTS tokenizer "am" not supported | `b91ab17` | `xtts_demo.py`, `gpt_train.py` | ✅ Fixed |

---

## 🚀 After Updating

To apply all fixes on your remote server:

```bash
cd Amharic_XTTS-V2_TTS
git pull origin main
./launch.sh
```

Expected startup messages:
```
Applying PyTorch 2.6 compatibility patches...
✅ Successfully patched trainer.io.load_fsspec for PyTorch 2.6+ compatibility
Starting Amharic XTTS Fine-Tuning WebUI...
* Running on local URL:  http://127.0.0.1:5003
```

---

## 📞 Support

For additional help:
- Check documentation: `docs/` directory
- Review test files: `tests/` directory
- See architecture docs: `docs/AMHARIC_TOKENIZER_ARCHITECTURE.md`
- Amharic G2P guide: `amharic_tts/g2p/README.md`

---

## 🔄 Version History

- **v1.3** (2025-10-08): Added PyTorch 2.6 compatibility patch
- **v1.2** (2025-10-08): Fixed gradient accumulation slider
- **v1.1** (2025-10-08): Added support for both "am" and "amh" codes
- **v1.0** (2025-10-08): Initial release with Amharic support

---

**Last Updated:** 2025-10-08  
**Status:** All known issues resolved ✅
