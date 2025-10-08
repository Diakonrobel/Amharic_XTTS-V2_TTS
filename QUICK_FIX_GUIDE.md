# Quick Fix Guide: Audio-Text Misalignment

## 🚨 If You Have Existing YouTube Datasets

### Step 1: Stop Everything
```bash
# Stop any running training
# Your datasets are likely corrupted and need to be recreated
```

### Step 2: Verify Your Dataset (Optional but Recommended)
```bash
# Check if your dataset has alignment issues
python verify_dataset_alignment.py "path/to/your/dataset" --verbose

# Example:
python verify_dataset_alignment.py "finetune_models/my_dataset/dataset" -v
```

**What to look for:**
- Alignment Quality < 70% = **CRITICAL** - Dataset is corrupted
- Alignment Quality 70-90% = **WARNING** - May have issues
- Alignment Quality > 90% = **OK** - Proceed with caution

### Step 3: Delete Old Datasets
```bash
# Delete all YouTube-generated datasets
# They were created with the bug and are unusable
```

### Step 4: Reprocess Your Videos
1. Open the WebUI
2. Go to "Prepare Dataset" → "YouTube Download"
3. Enter your YouTube URLs again
4. Enable "Batch Mode" if processing multiple videos
5. Click "Download and Process"

### Step 5: Verify New Dataset
```bash
# Verify the new dataset is correctly aligned
python verify_dataset_alignment.py "path/to/new/dataset" --verbose
```

**Expected result:** Alignment Quality > 95%

---

## ✅ For New Users (No Existing Datasets)

You're all set! The fix is already applied. Just use the tool normally:

1. Go to "Prepare Dataset" → "YouTube Download"
2. Enter YouTube URL(s)
3. Select language
4. Process and train

Your datasets will be correctly aligned from the start.

---

## 🔍 How to Manually Check Alignment

### Method 1: Listen to Random Samples

```bash
# Pick 5-10 random audio files from dataset/wavs/
# Play each one and check if audio matches the text in metadata CSV
```

**Signs of misalignment:**
- Audio says different words than text
- Audio is cut off mid-word
- Audio starts/ends at wrong points
- Multiple words blended together

### Method 2: Use the Verification Script

```bash
# Quick check (first 100 samples)
python verify_dataset_alignment.py "dataset/path" --sample-size 100

# Full check (all samples, detailed output)
python verify_dataset_alignment.py "dataset/path" -v

# Just check if alignment is OK (exit code tells you)
python verify_dataset_alignment.py "dataset/path" && echo "✅ OK" || echo "❌ ISSUES"
```

---

## 📊 What Changed

### The Bug
Audio extraction used wrong math that shifted timestamps:
```python
# WRONG (old code)
buffered_start = max(start_time - buffer, (prev_end + start_time) / 2)
```

### The Fix
Now respects subtitle boundaries properly:
```python
# CORRECT (new code)
earliest_start = prev_end + 0.05  # 50ms gap
buffered_start = max(earliest_start, start_time - buffer)
```

---

## ❓ FAQ

### Q: Do I need to reprocess ALL my datasets?
**A:** Only YouTube datasets. Datasets from:
- Direct audio+SRT upload: **May need reprocessing** (same bug)
- Whisper transcription: **OK** (different code path)
- Pre-existing datasets: **OK** (not affected)

### Q: How can I tell if my model was trained on bad data?
**A:** If you notice:
- Poor pronunciation
- Words in wrong order
- Mumbled/unclear speech
- Random artifacts
- Model doesn't match reference audio style

→ Your training data was likely misaligned

### Q: Will retraining with fixed data improve my model?
**A:** YES! Significantly. Clean, aligned data is critical for TTS quality.

### Q: How long does reprocessing take?
**A:** About the same as original processing:
- Download: 1-5 min per video
- Processing: 1-2 min per video
- Total: ~5-10 min per video

### Q: Can I use my old training checkpoints?
**A:** Not recommended. Start fresh with clean data for best results.

---

## 🎯 Success Checklist

Before starting new training:

- [ ] Latest code pulled/updated
- [ ] Old YouTube datasets deleted
- [ ] Videos reprocessed with fixed code
- [ ] Verification script shows >95% alignment
- [ ] Random sample spot-check passed (listen to 5-10 files)
- [ ] Dataset history cleared (or new output directory used)

---

## 📞 Still Having Issues?

1. Check console output for error messages
2. Run verification script with `-v` flag
3. Manually check 5-10 random audio files
4. Compare old vs new dataset alignment scores
5. Report any persistent issues with:
   - Verification script output
   - Sample audio files
   - YouTube URLs used
   - Console logs

---

**Last Updated:** 2025-10-08  
**Fix Version:** v1.0.0
