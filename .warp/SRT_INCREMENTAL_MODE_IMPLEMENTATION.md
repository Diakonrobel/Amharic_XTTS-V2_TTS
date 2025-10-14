# SRT Upload Incremental Mode Implementation Guide

## 📋 Overview

Your webUI already has **incremental dataset mode for YouTube downloads**, but the **SRT + media upload tab lacks this feature**. This guide shows how to add incremental mode to the SRT processing tab so you can upload local SRT+audio/video files and expand your dataset on Lightning AI without overwriting existing data.

## ✅ What Already Works

### YouTube Tab ✓
- ✅ Incremental mode checkbox
- ✅ Skip duplicates checkbox  
- ✅ Backend fully supports incremental merging
- ✅ Proper UI feedback

### SRT Upload Tab ❌
- ❌ No incremental mode option
- ❌ No duplicate detection option
- ❌ Always creates new dataset (overwrites existing)
- ✅ Backend already supports incremental (just not exposed in UI)

## 🎯 What Needs to Be Done

**Good news**: The backend functions `batch_processor.process_srt_media_batch()` and `merge_datasets()` **already have full incremental support**! You just need to:

1. Add 2 checkboxes to the SRT tab UI
2. Pass 2 additional parameters to the processing function
3. Done!

---

## 🔧 Implementation Steps

### Step 1: Add UI Checkboxes to SRT Tab

**File**: `xtts_demo.py`  
**Location**: Lines ~698-710 (in the "📝 SRT Processing" tab)

**Current code**:
```python path=D:/FINETUNE-XTTS-WEBUI-LIGHTNING/xtts-finetune-webui-fresh/xtts_demo.py start=698
with gr.Row():
    srt_batch_mode = gr.Checkbox(
        label="🎬 Batch Mode",
        value=False,
        info="Process multiple file pairs as one dataset",
        scale=1
    )
    use_vad_refinement = gr.Checkbox(
        label="🎤 VAD Enhancement",
        value=False,
        info="AI-powered speech detection (+20% time)",
        scale=1
    )
```

**Add after line 710** (after `use_vad_refinement`):
```python path=null start=null
# Add these two new rows for incremental mode
with gr.Row():
    srt_incremental_mode = gr.Checkbox(
        label="➕ Incremental Mode",
        value=False,
        info="Add to existing dataset (no overwrite)",
        scale=1
    )
    srt_check_duplicates = gr.Checkbox(
        label="🔍 Skip Duplicates",
        value=True,
        info="Auto-detect and skip duplicate audio files",
        scale=1
    )

gr.Markdown("""
💡 **Tip**: Use *Incremental Mode* to grow your dataset over time by adding new SRT+media files without losing existing data.
Perfect for uploading batches of locally prepared subtitle files!
""")
```

---

### Step 2: Update Backend Function Call

**File**: `xtts_demo.py`  
**Location**: Lines ~879-926 (`process_srt_media_batch_handler` function)

**Current code**:
```python path=D:/FINETUNE-XTTS-WEBUI-LIGHTNING/xtts-finetune-webui-fresh/xtts_demo.py start=887
# Process all pairs
train_csv, eval_csv, file_infos = batch_processor.process_srt_media_batch(
    srt_files=srt_files_list,
    media_files=media_files_list,
    language=language,
    out_path=out_path,
    srt_processor=srt_processor,
    progress_callback=lambda p, desc: progress(p, desc=desc)
)
```

**Update to include incremental parameters**:
```python path=null start=null
# Process all pairs
train_csv, eval_csv, file_infos = batch_processor.process_srt_media_batch(
    srt_files=srt_files_list,
    media_files=media_files_list,
    language=language,
    out_path=out_path,
    srt_processor=srt_processor,
    progress_callback=lambda p, desc: progress(p, desc=desc),
    incremental=incremental,          # ← ADD THIS
    check_duplicates=check_duplicates # ← ADD THIS
)
```

---

### Step 3: Update Function Signature

**File**: `xtts_demo.py`  
**Location**: Lines ~928-940 (`process_srt_media` function signature)

**Current signature**:
```python path=D:/FINETUNE-XTTS-WEBUI-LIGHTNING/xtts-finetune-webui-fresh/xtts_demo.py start=928
def process_srt_media(
    srt_file_input, 
    media_file_input, 
    language, 
    out_path, 
    batch_mode, 
    use_vad, 
    vad_threshold_val=0.5,
    vad_min_speech_ms=250,
    vad_min_silence_ms=300,
    vad_pad_ms=30,
    progress=gr.Progress(track_tqdm=True)
):
```

**Update to**:
```python path=null start=null
def process_srt_media(
    srt_file_input, 
    media_file_input, 
    language, 
    out_path, 
    batch_mode, 
    use_vad, 
    vad_threshold_val=0.5,
    vad_min_speech_ms=250,
    vad_min_silence_ms=300,
    vad_pad_ms=30,
    incremental=False,          # ← ADD THIS
    check_duplicates=True,      # ← ADD THIS
    progress=gr.Progress(track_tqdm=True)
):
```

Then **inside the function**, when calling `process_srt_media_batch_handler`, pass these parameters:

Find this line (around line 884):
```python path=null start=null
# Process all pairs
train_csv, eval_csv, file_infos = batch_processor.process_srt_media_batch(
    ...
```

And make sure it includes:
```python path=null start=null
train_csv, eval_csv, file_infos = batch_processor.process_srt_media_batch(
    srt_files=srt_files_list,
    media_files=media_files_list,
    language=language,
    out_path=out_path,
    srt_processor=srt_processor,
    progress_callback=lambda p, desc: progress(p, desc=desc),
    incremental=incremental,          # ← Pass from function params
    check_duplicates=check_duplicates # ← Pass from function params
)
```

---

### Step 4: Wire Up Gradio Button

**File**: `xtts_demo.py`  
**Location**: Lines ~1833-1848 (button click handler)

**Current code**:
```python path=D:/FINETUNE-XTTS-WEBUI-LIGHTNING/xtts-finetune-webui-fresh/xtts_demo.py start=1833
process_srt_btn.click(
    fn=process_srt_media,
    inputs=[
        srt_files,
        media_files,
        lang,
        out_path,
        srt_batch_mode,
        use_vad_refinement,  # VAD enable/disable
        vad_threshold,  # VAD threshold
        vad_min_speech_duration,  # Min speech duration
        vad_min_silence_duration,  # Min silence duration
        vad_speech_pad,  # Speech padding
    ],
    outputs=[srt_status],
)
```

**Update to include new checkboxes**:
```python path=null start=null
process_srt_btn.click(
    fn=process_srt_media,
    inputs=[
        srt_files,
        media_files,
        lang,
        out_path,
        srt_batch_mode,
        use_vad_refinement,  # VAD enable/disable
        vad_threshold,  # VAD threshold
        vad_min_speech_duration,  # Min speech duration
        vad_min_silence_duration,  # Min silence duration
        vad_speech_pad,  # Speech padding
        srt_incremental_mode,  # ← ADD THIS
        srt_check_duplicates,  # ← ADD THIS
    ],
    outputs=[srt_status],
)
```

---

## 📝 Complete Diff Summary

Here's what you're changing:

### 1. UI Addition (after line 710):
```diff
+ with gr.Row():
+     srt_incremental_mode = gr.Checkbox(
+         label="➕ Incremental Mode",
+         value=False,
+         info="Add to existing dataset (no overwrite)",
+         scale=1
+     )
+     srt_check_duplicates = gr.Checkbox(
+         label="🔍 Skip Duplicates",
+         value=True,
+         info="Auto-detect and skip duplicate audio files",
+         scale=1
+     )
+ 
+ gr.Markdown("""
+ 💡 **Tip**: Use *Incremental Mode* to grow your dataset over time.
+ """)
```

### 2. Function Signature Update (line 928):
```diff
  def process_srt_media(
      srt_file_input, 
      media_file_input, 
      language, 
      out_path, 
      batch_mode, 
      use_vad, 
      vad_threshold_val=0.5,
      vad_min_speech_ms=250,
      vad_min_silence_ms=300,
      vad_pad_ms=30,
+     incremental=False,
+     check_duplicates=True,
      progress=gr.Progress(track_tqdm=True)
  ):
```

### 3. Backend Call Update (line 887):
```diff
  train_csv, eval_csv, file_infos = batch_processor.process_srt_media_batch(
      srt_files=srt_files_list,
      media_files=media_files_list,
      language=language,
      out_path=out_path,
      srt_processor=srt_processor,
      progress_callback=lambda p, desc: progress(p, desc=desc),
+     incremental=incremental,
+     check_duplicates=check_duplicates
  )
```

### 4. Button Wiring Update (line 1833):
```diff
  process_srt_btn.click(
      fn=process_srt_media,
      inputs=[
          srt_files,
          media_files,
          lang,
          out_path,
          srt_batch_mode,
          use_vad_refinement,
          vad_threshold,
          vad_min_speech_duration,
          vad_min_silence_duration,
          vad_speech_pad,
+         srt_incremental_mode,
+         srt_check_duplicates,
      ],
      outputs=[srt_status],
  )
```

---

## 🚀 Usage After Implementation

### On Local Machine (Windows)

1. **Prepare SRT + Media Files**
   - Export your subtitle files from editing software (.srt format)
   - Ensure matching filenames (e.g., `video1.srt` + `video1.mp4`)

2. **Upload to Lightning AI**
   - Use Lightning AI file upload or sync via GitHub
   - Place files in accessible directory on Lightning AI instance

3. **Process via Gradio UI**
   ```
   Navigate to: http://localhost:7860 (or your Lightning AI URL)
   
   Tab: "Step 1: Dataset Preparation"
   Sub-tab: "📝 SRT Processing"
   
   Actions:
   1. Upload SRT files (multiple)
   2. Upload matching media files (multiple)
   3. ✅ Check "🎬 Batch Mode"
   4. ✅ Check "➕ Incremental Mode"  ← NEW!
   5. ✅ Check "🔍 Skip Duplicates"   ← NEW!
   6. Click "▶️ Process SRT + Media"
   ```

### On Lightning AI (Remote)

**Workflow for Growing Dataset**:

```bash path=null start=null
# Session 1: Create initial dataset
Upload: batch1/*.srt + batch1/*.mp4
Settings: Batch=Yes, Incremental=No
Result: 500 segments created

# Session 2: Add more data
Upload: batch2/*.srt + batch2/*.mp4
Settings: Batch=Yes, Incremental=Yes ✓, Skip Duplicates=Yes ✓
Result: 300 new segments added → 800 total

# Session 3: Add even more
Upload: batch3/*.srt + batch3/*.mp4
Settings: Batch=Yes, Incremental=Yes ✓, Skip Duplicates=Yes ✓
Result: 250 new segments added → 1050 total
```

---

## 📊 Expected Behavior

### Standard Mode (Incremental = OFF)
```
Before: dataset/ with 500 segments
Upload: 300 new segments
After:  dataset/ with 300 segments (REPLACED)
```

### Incremental Mode (Incremental = ON)
```
Before: dataset/ with 500 segments
Upload: 300 new segments
After:  dataset/ with 800 segments (MERGED)
```

### With Duplicate Detection (Skip Duplicates = ON)
```
Before: dataset/ with 500 segments
Upload: 300 segments (50 duplicates)
After:  dataset/ with 750 segments (250 added, 50 skipped)
```

---

## ⚠️ Important Notes

### 1. Language Consistency
- **Must use same language** for all incremental additions
- System validates `lang.txt` automatically
- Prevents accidental mixing of languages

### 2. File Naming
- SRT and media files must have **matching stems**
  - ✅ `episode1.srt` + `episode1.mp4`
  - ✅ `clip_002.srt` + `clip_002.wav`
  - ❌ `episode1.srt` + `episode1_final.mp4` (no match)

### 3. Duplicate Detection
- Hashes audio content (first + last chunks)
- Fast comparison (~0.1ms per file)
- Skips exact duplicates automatically

### 4. Dataset Structure
```
finetune_models/
└── dataset/
    ├── metadata_train.csv   ← Grows with each addition
    ├── metadata_eval.csv    ← Grows with each addition
    ├── lang.txt             ← Must match for incremental
    └── wavs/
        ├── merged_00000000.wav
        ├── merged_00000001.wav
        ...
        └── merged_00000999.wav  ← Sequential numbering continues
```

---

## 🧪 Testing Plan

### Test 1: Basic Incremental Addition
```python path=null start=null
# Initial dataset: 10 segments
# Upload: 5 new SRT+media pairs
# Expected: 15 segments total
# Verify: No overwrites, sequential file numbering
```

### Test 2: Duplicate Detection
```python path=null start=null
# Initial dataset: 10 segments
# Upload: 5 pairs (2 are duplicates from initial)
# Expected: 13 segments total (3 new added, 2 skipped)
# Verify: Duplicate message in logs
```

### Test 3: Language Validation
```python path=null start=null
# Initial dataset: Amharic (lang.txt = "amh")
# Upload: English SRT+media with lang=en
# Expected: Error "Language mismatch"
# Verify: Dataset unchanged
```

### Test 4: Standard Mode (No Incremental)
```python path=null start=null
# Initial dataset: 10 segments
# Upload: 5 pairs with Incremental=OFF
# Expected: 5 segments total (original replaced)
# Verify: Old files removed
```

---

## 🔒 Safety Features

### Already Implemented in Backend
- ✅ Language compatibility validation
- ✅ Dataset structure validation
- ✅ Audio file existence checks
- ✅ Duplicate detection with hashing
- ✅ Error handling (continues on individual file errors)
- ✅ Atomic operations (temp datasets merged only on success)

### No Additional Safety Needed
The existing `batch_processor.py` and `incremental_dataset_merger.py` already handle all safety concerns robustly!

---

## 📈 Performance Considerations

### Speed
- **Upload**: Limited by network bandwidth to Lightning AI
- **Processing**: Same as current (per-file SRT processing)
- **Merging**: ~0.1ms per file for duplicate check
- **Total**: Approximately same time as non-incremental

### Memory
- Loads CSVs into memory (typically < 10MB even for large datasets)
- Doesn't load audio into memory
- Memory footprint: < 100MB

### Disk Space on Lightning AI
- Keeps original files until merge completes
- Removes temp datasets after successful merge
- Temporary space needed: ~2x size of new upload

---

## 🎯 Benefits of This Feature

### For You
1. **No data loss** - Existing dataset preserved
2. **Flexible growth** - Add data across multiple sessions
3. **Efficient use of time** - Process batches when convenient
4. **No manual merging** - Automatic deduplication and merging

### For Lightning AI Workflow
1. **Upload files via web UI** - No command-line needed
2. **Process remotely** - Use GPU for heavy processing
3. **Grow dataset incrementally** - Perfect for limited upload time
4. **Consistent with YouTube tab** - Same UI/UX patterns

---

## 🔄 Migration Path

### Existing Datasets
Your current datasets are **fully compatible**! The system will:
- Detect existing `dataset/` folder
- Continue sequential numbering from last file
- Preserve all existing segments
- Validate language consistency

### No Breaking Changes
- Standard mode still works (Incremental=OFF by default)
- Existing scripts/workflows unaffected
- Backward compatible with all existing datasets

---

## 📚 Related Documentation

- **Backend Implementation**: `utils/batch_processor.py` lines 328-422
- **Incremental Merger**: `utils/incremental_dataset_merger.py`
- **Existing Guide**: `INCREMENTAL_DATASET_ADDITION.md`
- **YouTube Example**: Lines 736-788 in `xtts_demo.py` (reference UI)

---

## ✅ Implementation Checklist

- [ ] Step 1: Add checkboxes to SRT tab UI
- [ ] Step 2: Update `process_srt_media_batch_handler` call
- [ ] Step 3: Update `process_srt_media` function signature
- [ ] Step 4: Wire up button inputs
- [ ] Step 5: Test with small dataset (5 files)
- [ ] Step 6: Test incremental addition
- [ ] Step 7: Test duplicate detection
- [ ] Step 8: Test language validation
- [ ] Step 9: Commit and push to GitHub
- [ ] Step 10: Pull on Lightning AI and test remotely

---

## 🎉 Summary

**What you get**:
- ✅ Upload SRT + audio/video files directly via Gradio UI
- ✅ Expand existing dataset without overwriting
- ✅ Automatic duplicate detection
- ✅ Language consistency validation
- ✅ Same workflow as YouTube tab
- ✅ Works on Lightning AI remote GPU

**Effort required**:
- ~20 lines of code added to `xtts_demo.py`
- ~15 minutes to implement and test
- No backend changes needed (already supported!)

**Perfect for**:
- Uploading locally edited subtitle files
- Growing dataset across multiple sessions
- Working with professional subtitle exports
- Maintaining dataset quality with deduplication

---

_Generated for XTTS Amharic TTS Project - Ready for immediate implementation_
