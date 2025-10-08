# Batch SRT+Media Processing Implementation

## Overview
This document describes the implementation of batch processing for SRT subtitle files with corresponding media files.

## Features Implemented

### 1. File Pairing (`batch_processor.py`)
- **Function**: `pair_srt_with_media(srt_files, media_files)`
- **Logic**: Matches SRT files with media files based on filename stems (case-insensitive)
- **Example**: 
  - `video1.srt` ↔ `video1.mp4`
  - `podcast_ep2.srt` ↔ `podcast_ep2.wav`
  - Mismatches are reported with warnings

### 2. Batch Processing (`batch_processor.py`)
- **Function**: `process_srt_media_batch(srt_files, media_files, language, out_path, srt_processor, progress_callback)`
- **Workflow**:
  1. Pair SRT files with media files
  2. Process each pair individually to temporary datasets
  3. Merge all temporary datasets into one unified dataset
  4. Clean up temporary datasets
  5. Return combined train/eval CSV paths and file info

### 3. Summary Formatting (`batch_processor.py`)
- **Function**: `format_srt_batch_summary(file_infos, total_segments)`
- **Output**: Detailed summary showing:
  - List of processed SRT-Media pairs
  - Duration and segment count per pair
  - Total statistics (pairs, duration, segments)

### 4. UI Integration (`xtts_demo.py`)
- **Updated Function**: `process_srt_media()`
  - Now accepts multiple files (list or single)
  - Detects batch mode from checkbox
  - Routes to batch handler or single-file handler accordingly
- **Backward Compatible**: Single file processing still works as before
- **Button Handler**: Updated to pass `srt_files`, `media_files`, and `srt_batch_mode` parameters

## Usage

### Single File Mode (Default)
1. Upload one SRT file
2. Upload one media file
3. Leave "Batch Mode" checkbox unchecked
4. Click "Process SRT + Media"

### Batch Mode
1. Upload multiple SRT files
2. Upload multiple media files (with matching names)
3. Check "🎬 Batch Mode" checkbox
4. Click "Process SRT + Media"

**Important**: Files are matched by filename stem:
- `episode1.srt` + `episode1.mp4` ✓
- `Episode1.SRT` + `episode1.MP4` ✓ (case-insensitive)
- `part_A.srt` + `part_B.mp4` ✗ (names don't match)

## File Naming Convention
For batch processing to work correctly:
- SRT and media files must have the same base filename
- Extensions are ignored
- Case is ignored
- Example valid pairs:
  ```
  video1.srt → video1.mp4
  Video2.SRT → video2.wav
  podcast_ep3.srt → PODCAST_EP3.mp3
  ```

## Output

### Batch Processing Output
```
✓ SRT Batch Processing Complete!
============================================================

Processed 3 SRT-Media pairs:

1. video1.srt + video1.mp4
   Duration: 320.5s | Segments: 45

2. video2.srt + video2.wav
   Duration: 215.0s | Segments: 32

3. podcast_ep3.srt + podcast_ep3.mp3
   Duration: 480.2s | Segments: 58

============================================================
Total Pairs: 3
Total Duration: 1015.7s (16.9 minutes)
Total Segments: 135
Average Segments per Pair: 45

ℹ This batch dataset has been saved to history.
```

### Dataset Structure
All processed files are merged into a single unified dataset:
```
finetune_models/
└── dataset/
    ├── metadata_train.csv
    ├── metadata_eval.csv
    ├── lang.txt
    └── wavs/
        ├── merged_00000000.wav
        ├── merged_00000001.wav
        ├── merged_00000002.wav
        └── ...
```

Audio files are renamed with sequential numbering to avoid conflicts.

## Code Architecture

### Module Dependencies
```
xtts_demo.py
├── process_srt_media() → Main entry point
│   ├── process_srt_media_batch_handler() → Batch orchestrator
│   │   └── batch_processor.process_srt_media_batch()
│   │       ├── pair_srt_with_media() → File matching
│   │       ├── srt_processor.process_srt_with_media() → Per-file processing
│   │       └── merge_datasets() → Dataset consolidation
│   └── srt_processor.process_srt_with_media() → Single file (legacy)
```

### Key Functions

#### `pair_srt_with_media()`
- **Input**: Lists of SRT and media file paths
- **Output**: List of tuples `(srt_path, media_path)`
- **Logic**: Case-insensitive stem matching

#### `process_srt_media_batch()`
- **Input**: File lists, language, output path, processors
- **Output**: Train CSV, eval CSV, file info list
- **Logic**: Iterate → Process → Merge → Clean

#### `merge_datasets()`
- **Input**: List of dataset directories
- **Output**: Train CSV, eval CSV, segment count
- **Logic**: Copy all audio files with renaming, concatenate CSVs, shuffle

## Error Handling
- **No matching pairs**: Clear error message
- **Individual pair failure**: Logged, continues with remaining pairs
- **All pairs fail**: Error message with details
- **File not found**: Warning in pairing stage

## Testing Recommendations

### Test Case 1: Basic Batch (2 files)
- Upload: `test1.srt`, `test2.srt`
- Upload: `test1.mp4`, `test2.mp4`
- Enable batch mode
- Expected: 2 pairs processed, single merged dataset

### Test Case 2: Name Mismatch
- Upload: `video_a.srt`, `video_b.srt`
- Upload: `audio_x.mp3`, `audio_y.mp3`
- Enable batch mode
- Expected: Warning about no matching pairs

### Test Case 3: Single File in Batch Mode
- Upload: `single.srt`
- Upload: `single.mp4`
- Enable batch mode
- Expected: Works as single-file processing

### Test Case 4: Mixed Case Names
- Upload: `Episode1.SRT`, `EPISODE2.srt`
- Upload: `episode1.MP4`, `Episode2.wav`
- Enable batch mode
- Expected: Both pairs matched correctly

## Future Enhancements
- [ ] Fuzzy name matching (similarity threshold)
- [ ] Manual pair assignment interface
- [ ] Preview paired files before processing
- [ ] Support for archive uploads (zip/tar with matched pairs)
- [ ] Progress bar per-file during batch processing
