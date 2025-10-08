# Batch YouTube Processing - User Guide

## ✅ Feature Complete!

The batch YouTube processing feature is now fully implemented and ready to use!

## 🎬 How to Use

### Single Video (Normal Mode)
1. Go to **Tab 1 - Data processing**
2. Open **"📹 YouTube Video Download"** accordion
3. Enter a single URL
4. **Leave "Batch Mode" unchecked**
5. Select language (e.g., "Amharic (አማርኛ)")
6. Click **"Download & Process YouTube"**

### Multiple Videos (Batch Mode)
1. Go to **Tab 1 - Data processing**
2. Open **"📹 YouTube Video Download"** accordion
3. Enter multiple URLs (see formats below)
4. **✓ Check "🎬 Batch Mode"** checkbox
5. Select language (e.g., "Amharic (አማርኛ)")
6. Click **"Download & Process YouTube"**

## 📝 URL Input Formats

### Comma-Separated:
```
https://youtube.com/watch?v=VIDEO1, https://youtube.com/watch?v=VIDEO2, https://youtube.com/watch?v=VIDEO3
```

### Newline-Separated:
```
https://youtube.com/watch?v=VIDEO1
https://youtube.com/watch?v=VIDEO2
https://youtube.com/watch?v=VIDEO3
```

### Mixed (Comma + Newline):
```
https://youtube.com/watch?v=VIDEO1,
https://youtube.com/watch?v=VIDEO2,
https://youtube.com/watch?v=VIDEO3
```

All formats work! The system automatically parses and validates URLs.

## 🔍 What Happens

### Single Video Mode:
1. Download video audio
2. Download subtitles in selected language
3. Process to dataset
4. Track in history
5. Show results

### Batch Mode (Multiple Videos):
1. ✅ Parse all URLs
2. ✅ Download each video sequentially
3. ✅ Process each to temporary dataset
4. ✅ **Merge all datasets into ONE unified dataset**
5. ✅ Track batch in history
6. ✅ Show detailed summary with all videos

## 📊 Batch Output Example

```
✓ Batch Processing Complete!
============================================================

Processed 3 videos:

1. ትረካ ፡ ከቸርችል ጎዳና ወደ ቸርችል ደመና
   Duration: 2400s | Segments: 577

2. ትረካ ፡ አፍቃሪው ንጉስ - ኤድዋርድ 8ኛ
   Duration: 2745s | Segments: 579

3. ትረካ ፡ የሄሌን ትሮይ ታሪክ
   Duration: 1980s | Segments: 445

============================================================
Total Videos: 3
Total Duration: 7125s (118.8 minutes)
Total Segments: 1601
Average Segments per Video: 534

ℹ This batch dataset has been saved to history.
```

## 🎯 Benefits

### Single Dataset for Training:
- All videos combined into one dataset
- Consistent file naming (`merged_00000001.wav`, etc.)
- Single `metadata_train.csv` and `metadata_eval.csv`
- Properly shuffled and split (85/15)

### Time Saving:
- Process multiple videos at once
- No manual merging needed
- Automatic cleanup of temporary files

### Organization:
- One dataset directory
- Clean file structure
- Easy to manage

## 📁 Output Structure

```
finetune_models/
└── dataset/
    ├── wavs/
    │   ├── merged_00000000.wav  ← From video 1
    │   ├── merged_00000001.wav  ← From video 1
    │   ├── merged_00000002.wav  ← From video 2
    │   ├── merged_00000003.wav  ← From video 2
    │   ├── merged_00000004.wav  ← From video 3
    │   └── ...
    ├── metadata_train.csv       ← Combined & shuffled
    ├── metadata_eval.csv        ← Combined & shuffled
    ├── lang.txt                 ← Language (e.g., "am")
    └── dataset_history.json     ← Tracking info
```

## 🔄 Duplicate Detection

The system automatically detects:
- **Single Mode**: Checks if video already processed
- **Batch Mode**: Each video checked individually during processing

If a duplicate is found during batch processing, it's skipped with a warning.

## 💡 Tips

### For Best Results:
1. Use videos from same speaker/narrator
2. Select correct language for all videos
3. Ensure good internet connection
4. Have enough disk space (~50MB per 30min video)

### Recommended Use Cases:
- **Multiple audiobook chapters** → One dataset
- **Podcast series** → One dataset
- **Lecture series** → One dataset
- **Interview series** → One dataset

### When to Use Single Mode:
- Testing individual videos
- Different speakers/languages
- Need separate datasets
- Single video processing

### When to Use Batch Mode:
- Same speaker across videos
- Want unified dataset
- Series of related content
- Batch downloading

## ⚠️ Important Notes

1. **Same Language**: All videos processed with the same language setting
2. **Sequential Processing**: Videos processed one at a time (not parallel)
3. **Memory**: Large batches may take time and disk space
4. **Errors**: If one video fails, others continue processing
5. **Tracking**: Batch tracked as single entry in history

## 🐛 Troubleshooting

### "No valid YouTube URLs found"
- Check URL format
- Ensure URLs contain `youtube.com` or `youtu.be`
- Remove extra spaces or characters

### "Failed to download video"
- Check internet connection
- Verify video is not private/deleted
- Try different video

### "No subtitles available"
- Video may not have captions
- Try different language
- System will fall back to Whisper transcription

### Batch stops midway:
- Check console output for specific error
- Individual video failures are logged but don't stop batch
- Check disk space

## 📈 Performance

### Approximate Times:
- **Download**: ~30-60s per video (depends on length/connection)
- **Processing**: ~1-2min per video (depends on CPU/GPU)
- **Merging**: ~10-30s for batch

### Example: 3 videos, 30min each:
- Download: ~3 minutes
- Processing: ~6 minutes
- Merging: ~15 seconds
- **Total**: ~9-10 minutes

## 🎓 Example Workflow

### Creating Amharic Audiobook Dataset:

1. **Find 3-5 audiobook chapters on YouTube**
2. **Copy all URLs**
3. **Paste into URL box** (one per line or comma-separated)
4. **Check "Batch Mode"**
5. **Select "Amharic (አማርኛ)"**
6. **Click "Download & Process"**
7. **Wait for completion** (~10-15 minutes for 5 videos)
8. **Result**: Single unified dataset ready for training!

## ✅ Success Indicators

You'll know it worked when you see:
- ✅ "Batch Processing Complete!" message
- ✅ List of all processed videos
- ✅ Total segments count
- ✅ Combined duration
- ✅ "saved to history" message
- ✅ Single dataset directory with all audio files

## 🚀 Next Steps After Batch Processing

1. **Verify Dataset**: Check `finetune_models/dataset/`
2. **Proceed to Training**: Go to Tab 2
3. **Load Parameters**: Click "Step 2.1"
4. **Start Training**: Click "Step 2 - Train"

---

## 🎉 Feature Status: PRODUCTION READY!

This feature is fully tested and ready for production use. Enjoy batch processing your Amharic audiobooks! 🇪🇹
