# Batch SRT Processing - Quick Start Guide

## 🎯 What's New?

You can now process **multiple SRT+media file pairs** in a single operation, and they will be **automatically merged** into one unified dataset!

## 🚀 How to Use

### Step 1: Prepare Your Files
Make sure your SRT and media files have **matching names**:

```
✅ GOOD:
   - video1.srt → video1.mp4
   - podcast_ep2.srt → podcast_ep2.wav
   - Interview_3.SRT → interview_3.MP4 (case-insensitive!)

❌ BAD:
   - audio_A.srt → video_B.mp4 (names don't match)
```

### Step 2: Upload Files
1. Navigate to **"1 - Data processing"** tab
2. Open **"📝 SRT + Media File Processing"** accordion
3. Click **"SRT/VTT Subtitle File(s)"** → Upload all SRT files
4. Click **"Media File(s)"** → Upload all matching media files

### Step 3: Enable Batch Mode
✅ Check the **"🎬 Batch Mode"** checkbox

### Step 4: Process
Click **"Process SRT + Media"** button

## 📊 What Happens?

1. **Pairing**: System matches SRT files with media files by name
   ```
   ✓ Paired: video1.srt <-> video1.mp4
   ✓ Paired: video2.srt <-> video2.wav
   ⚠ No media file found for SRT: orphan.srt
   ```

2. **Processing**: Each pair is processed individually
   ```
   [1/2] Processing: video1.srt + video1.mp4
     ✓ Pair 1 processed: 45 segments, 320.5s
   
   [2/2] Processing: video2.srt + video2.wav
     ✓ Pair 2 processed: 32 segments, 215.0s
   ```

3. **Merging**: All datasets combined into one
   ```
   ✓ Merged dataset created:
     Training segments: 65
     Evaluation segments: 12
     Total segments: 77
   ```

4. **Summary**: You get a detailed report
   ```
   ✓ SRT Batch Processing Complete!
   ============================================================
   
   Processed 2 SRT-Media pairs:
   
   1. video1.srt + video1.mp4
      Duration: 320.5s | Segments: 45
   
   2. video2.srt + video2.wav
      Duration: 215.0s | Segments: 32
   
   ============================================================
   Total Pairs: 2
   Total Duration: 535.5s (8.9 minutes)
   Total Segments: 77
   Average Segments per Pair: 39
   ```

## 📁 Output Structure

Everything goes into **one unified dataset**:

```
finetune_models/
└── dataset/
    ├── metadata_train.csv    ← Combined from all pairs
    ├── metadata_eval.csv     ← Combined from all pairs
    ├── lang.txt
    └── wavs/
        ├── merged_00000000.wav
        ├── merged_00000001.wav
        ├── merged_00000002.wav
        └── ... (all audio from all pairs)
```

## 💡 Tips

### Single File? No Problem!
- You can still upload just **one SRT** and **one media** file
- Batch mode works for single files too
- Or just **leave batch mode unchecked** for single-file processing

### Name Matching Rules
- ✅ Extensions are **ignored**: `video.srt` matches `video.mp4`
- ✅ Case is **ignored**: `Video.SRT` matches `video.mp4`
- ❌ Base name must **match exactly**: `vid1.srt` won't match `video1.mp4`

### Multiple Pairs
- Upload 2, 3, 10, or more pairs at once
- Processing is sequential (one after another)
- Failed pairs won't stop the batch
- Final dataset includes only successful pairs

## 🔧 Troubleshooting

### "No SRT-media pairs could be matched"
**Problem**: File names don't match

**Solution**: Rename files so SRT and media have the same base name
```
Before:
  audio1.srt, video2.mp4 ❌

After:
  audio1.srt, audio1.mp4 ✅
```

### "⚠ No media file found for SRT: file.srt"
**Problem**: One SRT file has no matching media

**Solution**: Either:
- Upload the matching media file
- Remove the orphan SRT file
- Continue anyway (it will be skipped)

### Only Some Pairs Processed
**Problem**: Some pairs failed during processing

**Solution**: Check the error messages for specific pairs and fix those issues (corrupted files, unsupported formats, etc.)

## 🎬 Example Workflow

### Scenario: Process 3 podcast episodes

**Files**:
```
podcast_ep1.srt (15 KB)
podcast_ep1.mp3 (25 MB)

podcast_ep2.srt (18 KB)
podcast_ep2.mp3 (30 MB)

podcast_ep3.srt (20 KB)
podcast_ep3.mp3 (28 MB)
```

**Steps**:
1. Upload all 3 `.srt` files → SRT/VTT Subtitle File(s)
2. Upload all 3 `.mp3` files → Media File(s)
3. ✅ Check "Batch Mode"
4. Select language (e.g., "en")
5. Click "Process SRT + Media"

**Result**:
```
✓ SRT Batch Processing Complete!
Processed 3 SRT-Media pairs:
Total Segments: 135
Ready for training!
```

## 🆚 Batch vs Single Mode

| Feature | Single Mode | Batch Mode |
|---------|-------------|------------|
| Files | 1 SRT + 1 Media | Multiple pairs |
| Processing | Immediate | Sequential |
| Output | 1 dataset | 1 merged dataset |
| Checkbox | Unchecked | ✅ Checked |
| Use Case | Quick test | Full project |

## 🎉 Summary

**Before**: Process one SRT file at a time → multiple datasets → manual merging

**Now**: Upload all SRT+media pairs → automatic pairing → automatic processing → **one unified dataset** ready for training!

---

**Need Help?** Check `BATCH_SRT_IMPLEMENTATION.md` for detailed technical documentation.
