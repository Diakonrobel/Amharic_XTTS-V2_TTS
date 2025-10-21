# Quick Fix Summary: Amharic BPE-Only Training

## 🎯 Problem
Training crashes with `NotImplementedError: Language 'amh' is not supported` when using BPE-only mode (no G2P).

## ✅ Solution Applied
Created **global class-level monkey-patch** that runs before any tokenizers are created.

## 📁 Files Added/Modified

### NEW Files
1. `utils/amharic_bpe_tokenizer_patch.py` - The patch module
2. `test_amharic_bpe_fix.py` - Verification script
3. `AMHARIC_BPE_FIX.md` - Full documentation

### MODIFIED Files
1. `utils/gpt_train.py` - Lines 22-27 (applies patch)
2. `xtts_demo.py` - Lines 36-41 (applies patch)

## 🧪 Test the Fix

```bash
python test_amharic_bpe_fix.py
```

Expected: All tests pass ✅

## 🚀 How to Use

### In Gradio WebUI
1. Language: Select `amh` or `am`
2. **Uncheck** "Use Amharic G2P" 
3. Click "Train"

### Result
- No G2P dependencies needed
- Uses raw Ethiopic characters
- BPE tokenization works directly

## 🔧 Technical Summary

**What it does:**
- Patches `VoiceBpeTokenizer` class (before instances created)
- Adds `'am'` and `'amh'` to supported languages
- Maps Amharic → English preprocessing (returns raw text)

**Why it works:**
- English preprocessing = no transformation (perfect for BPE)
- Class-level = all instances inherit the fix
- Applied early = before dataset loading

**Robust because:**
- ✅ No system files modified
- ✅ Version control friendly
- ✅ Works for both G2P and BPE-only modes
- ✅ Idempotent (safe to call multiple times)

## 📊 Comparison

| Method | Works? | Portable? | Robust? |
|--------|--------|-----------|---------|
| Instance patch (old) | ❌ Too late | ✅ Yes | ❌ No |
| Edit system files | ✅ Yes | ❌ No | ❌ No |
| **Global class patch** | ✅ Yes | ✅ Yes | ✅ Yes |

## 🎓 Credits

Developed using MCP servers as required:
- `clear-thought` - Root cause analysis & solution design
- `context7-mcp` - XTTS documentation research

See `AMHARIC_BPE_FIX.md` for full details.
