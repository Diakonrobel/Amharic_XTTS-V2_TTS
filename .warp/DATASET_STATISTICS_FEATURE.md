# Dataset Statistics Feature - User Guide

## 📊 Overview

A new "**Dataset Statistics**" section has been added to the Data Processing tab that calculates and displays comprehensive information about your dataset, including:

- Total duration (hours/minutes)
- Segment counts (train/eval/total)
- Quality checks (duration requirements, train/eval ratio, file integrity)
- Training readiness status

## 🚀 How to Use

### In Gradio WebUI

1. **Navigate to**: Tab "📁 Data Processing"
2. **Scroll to**: "📊 Dataset Statistics" section (below "Create Dataset")
3. **Click**: "📈 Calculate Dataset Statistics" button
4. **Wait**: 5-30 seconds (depends on dataset size)
5. **View**: Comprehensive statistics display

### Example Output

```
📊 Dataset Statistics
============================================================

🌍 Language: AMH

📝 Segments:
  • Training: 1,453
  • Evaluation: 263
  • Total: 1,716

⏱️  Duration:
  • Total: 4.76 hours (285.7 min)
  • Training: 4.04 hours
  • Evaluation: 0.72 hours
  • Avg Segment: 10.0s

✅ Quality Checks:
  ⚠ Duration: 4.76h / 10h minimum
    Need 5.24 more hours for optimal training
  ✓ Train/Eval Ratio: 5.52:1 ✅
  ✓ Audio Files: 1,716 / 1,716 ✅

============================================================
⏳ Dataset is 47.6% ready (need more audio for optimal training)
```

---

## 📊 What the Statistics Mean

### Segments
- **Training**: Number of segments used for training the model
- **Evaluation**: Number of segments used for validation
- **Total**: Combined training + evaluation segments
- **Recommended Ratio**: 85:15 (or ~5.67:1)

### Duration
- **Total**: Total audio duration in your dataset
- **Training**: Audio duration for training segments
- **Evaluation**: Audio duration for eval segments
- **Avg Segment**: Average length per audio segment
- **Minimum Recommended**: 10+ hours for optimal training

### Quality Checks

#### ✅ Duration Check
- **✓ Green**: Dataset meets 10+ hour requirement
- **⚠️ Orange**: Dataset < 10 hours (shows how much more needed)

#### ✅ Train/Eval Ratio
- **✓ Green**: Ratio between 5.0:1 and 6.5:1 (healthy)
- **⚠️ Orange**: Ratio outside recommended range

#### ✅ Audio Files
- **✓ Green**: All expected audio files exist
- **⚠️ Orange**: Missing audio files (data corruption)

### Training Readiness
- **🎉 Ready**: Dataset ≥ 10 hours with no issues
- **⏳ X% Ready**: Shows percentage toward 10-hour goal
- **⚠️ Issues**: Warns about problems even if duration is sufficient

---

## 🔧 Behind the Scenes

### What It Calculates
The statistics calculator:
1. Reads `metadata_train.csv` and `metadata_eval.csv`
2. Counts segments from both files
3. Loads each audio file to measure exact duration
4. Validates audio file existence
5. Checks `lang.txt` for language
6. Calculates quality metrics

### Performance
- **Small datasets** (< 500 segments): ~5 seconds
- **Medium datasets** (500-2000 segments): ~15 seconds
- **Large datasets** (2000+ segments): ~30 seconds

*Note: Calculation time depends on disk I/O speed*

---

## 🎯 Use Cases

### 1. Monitor Dataset Growth
After each incremental addition:
```
Session 1: Calculate stats → 1.73 hours (1,353 segments)
Session 2: Add 300 segments → Calculate stats → 2.50 hours
Session 3: Add 500 segments → Calculate stats → 4.00 hours
Session 4: Add 800 segments → Calculate stats → 10.50 hours ✅ READY!
```

### 2. Verify Dataset Integrity
Before training:
```
Calculate stats → Check:
- All audio files present? ✅
- Language correct? ✅
- Train/eval ratio good? ✅
- Duration sufficient? ⚠️ Need 3 more hours

→ Decision: Add more data before training
```

### 3. Compare Datasets
Testing different sources:
```
Dataset A (YouTube): 12h, 3000 segments, ratio 5.5:1 ✅
Dataset B (SRT+media): 8h, 2500 segments, ratio 6.0:1 ⚠️

→ Decision: Use Dataset A, add to Dataset B
```

---

## 💡 Tips

### Quick Check
For a fast approximation without clicking the button:
- **Rough estimate**: Segments × 10 seconds ÷ 3600 = Hours
- **Example**: 1,716 segments × 10s ÷ 3600 = ~4.76 hours

### When to Calculate
- **After each dataset addition** (incremental mode)
- **Before starting training**
- **When troubleshooting dataset issues**
- **After dataset filtering/cleanup**

### What to Do with Results

**If Duration < 10 hours**:
1. Continue adding data incrementally
2. Use YouTube batch download for quick expansion
3. Upload more SRT+media files
4. Target: 10-15 hours for best results

**If Train/Eval Ratio is off**:
- Usually not critical (system auto-splits)
- Only worry if ratio < 4:1 or > 7:1
- Can be fixed by reprocessing dataset with different split

**If Audio Files Missing**:
- **Critical issue!** Dataset may be corrupted
- Check logs for processing errors
- Verify disk space wasn't exhausted
- May need to reprocess from source

---

## 🖥️ CLI Usage

You can also calculate statistics from command line:

```bash
# On Lightning AI or local machine
python utils/dataset_statistics.py /path/to/dataset

# Example
python utils/dataset_statistics.py finetune_models/dataset
```

Output is same as WebUI display.

---

## 🐛 Troubleshooting

### Error: "Dataset directory does not exist"
**Solution**: 
- Check that `out_path` is correct
- Ensure you've created a dataset first (process at least one file)

### Error: "Dataset metadata files not found"
**Solution**:
- Dataset may be empty or corrupted
- Verify `metadata_train.csv` and `metadata_eval.csv` exist
- Reprocess your source files

### Slow Calculation (> 60 seconds)
**Cause**: Very large dataset or slow disk
**Solution**:
- This is normal for 5000+ segments
- Wait patiently or use CLI version
- Consider calculating only after major additions

### Duration Shows 0 Hours
**Cause**: Audio files missing or calculation error
**Solution**:
- Check audio files exist in `wavs/` directory
- Verify audio files aren't corrupted
- Check logs for librosa errors

---

## 📚 Related Features

- **Processing History** (`📊 History` tab): Shows what was added and when
- **Incremental Mode**: Add data without overwriting (keeps stats growing)
- **Dataset Tracker**: Tracks each addition for debugging

---

## 🎓 Technical Details

### Files Analyzed
```
finetune_models/dataset/
├── metadata_train.csv   ← Segment counts, text content
├── metadata_eval.csv    ← Segment counts, text content
├── lang.txt             ← Language identifier
└── wavs/                ← Audio duration calculation
    ├── merged_00000000.wav
    ├── merged_00000001.wav
    └── ...
```

### Calculation Method
- Uses `librosa.get_duration()` for exact audio duration
- Reads entire CSV files for segment counts
- Validates each audio file existence
- Fast compared to re-processing entire dataset

---

## ✅ Summary

**What you get**:
- ✅ Exact duration of your dataset (hours/minutes)
- ✅ Segment counts (train/eval/total)
- ✅ Quality validation (ratio, files, language)
- ✅ Training readiness status
- ✅ Clear guidance on what to do next

**When to use**:
- After processing new files
- Before starting training
- When troubleshooting issues
- To track dataset growth

**Goal**: Help you reach 10+ hours of high-quality Amharic data for optimal training!

---

**Status**: ✅ Feature implemented and ready to use  
**Location**: Tab "📁 Data Processing" → "📊 Dataset Statistics"  
**Calculation Time**: 5-30 seconds depending on dataset size

---

_Happy dataset building! 🎉_
