# VAD-SRT Text-Audio Mismatch Fix

## ⚠️ **UPDATE: Silero VAD Removed (2025-10-17)**

Due to critical text-audio mismatching issues, **Silero VAD has been completely removed** from YouTube/SRT processing.

**Current Status:**
- ✅ Standard SRT-based extraction is now the only method
- ✅ Simple, reliable, and proven to work
- ❌ VAD functionality removed from UI and backend

---

## Problem Description (Historical)

### Critical Issue
When processing YouTube videos with SRT subtitles using Silero VAD, the system was **incorrectly pairing audio segments with text**. This caused severe misalignment in the training dataset.

### Root Cause
1. **Silero VAD** detects speech boundaries and splits audio into multiple segments
2. **Previous behavior**: ALL VAD segments from the same SRT timeframe got the SAME text
3. **Result**: Text-audio mismatch where:
   - Segment 1: Gets "Hello world" → ✓ Correct
   - Segment 2: Gets "Hello world" → ❌ Wrong (should be silent/different text)
   - Segment 3: Gets "Hello world" → ❌ Wrong (should be silent/different text)

### Example Scenario
```
SRT Entry:
  Time: 10.0s - 15.0s
  Text: "Hello world"

VAD Detection (with 0.2s buffer):
  Segment A: 9.8s - 12.5s  (overlaps 80% with SRT)
  Segment B: 13.0s - 15.2s (overlaps 60% with SRT)
  Segment C: 15.5s - 17.0s (overlaps 0% with SRT - noise/next sentence)

OLD BEHAVIOR (WRONG):
  A → "Hello world" ✓
  B → "Hello world" ❌ (should be empty or different)
  C → "Hello world" ❌ (should be empty or different)

NEW BEHAVIOR (CORRECT):
  A → "Hello world" ✓ (best overlap)
  B → "" (skipped - no text)
  C → "" (skipped - no text)
```

---

## Solution Implemented

### 1. **Fixed `refine_segment_with_vad()` Function**
Location: `utils/srt_processor_vad.py` (lines 225-334)

#### Key Changes:
- Added parameters: `original_srt_start` and `original_srt_end`
- **Overlap calculation**: For each VAD segment, calculate overlap with original SRT timing
- **Best match selection**: Only the segment with maximum overlap gets the text
- **Validation**: Segments without text are skipped during dataset creation

```python
# NEW: Calculate overlap for each VAD segment
for idx, region in enumerate(merged_regions):
    region_abs_start = segment_start_time + region['start']
    region_abs_end = segment_start_time + region['end']
    
    # Find overlap with original SRT
    overlap_start = max(region_abs_start, original_srt_start)
    overlap_end = min(region_abs_end, original_srt_end)
    overlap = max(0, overlap_end - overlap_start)
    
    # Track best match
    if overlap > best_overlap:
        best_overlap = overlap
        best_match_idx = idx

# Only assign text to best matching segment
assigned_text = segment_text if idx == best_match_idx else ""
```

### 2. **Updated Segment Extraction Logic**
Location: `utils/srt_processor_vad.py` (lines 146-166)

#### Key Changes:
- **Skip empty segments**: Segments with no text are not saved
- **Text validation**: Only segments with valid text are added to metadata
- **Prevents dataset pollution**: No more mismatched audio-text pairs

```python
# NEW: Skip segments without text
for sub_idx, refined in enumerate(refined_segments):
    if not refined.text or refined.text.strip() == "":
        continue  # Skip VAD artifacts
    
    # Only save segments with valid text
    metadata["audio_file"].append(f"wavs/{segment_filename}")
    metadata["text"].append(refined.text.strip())
    metadata["speaker_name"].append(speaker_name)
```

### 3. **Dataset Repair Script**
Location: `fix_vad_srt_mismatch.py`

A utility script to fix existing datasets created with the old buggy code.

**Usage:**
```bash
python fix_vad_srt_mismatch.py \
    --dataset_dir "path/to/dataset" \
    --srt_file "path/to/original.srt" \
    --min_overlap 0.3
```

**Features:**
- Parses original SRT file
- Re-validates audio-text pairs
- Removes segments with no/invalid text
- Creates backup of original CSVs
- Cleans up mismatched entries

---

## Impact & Benefits

### ✅ Fixed Issues
1. **Correct text-audio alignment**: Each audio segment now has its matching text
2. **No duplicate text**: Same text is no longer assigned to multiple segments
3. **Clean dataset**: Only valid pairs are included in training/eval sets
4. **Better training quality**: Model learns correct audio-text correspondence

### 📊 Expected Results
- **Fewer segments**: Invalid/duplicate segments are removed
- **Higher quality**: Only segments with proper overlap are kept
- **Correct timestamps**: Audio boundaries match SRT timing accurately

### 🎯 For Amharic & Multilingual Content
- Works with enhanced Silero VAD (Amharic mode)
- Handles ejective consonants and complex phonetics correctly
- Respects language-specific timing patterns

---

## Testing & Verification

### Verify Fixed Dataset
```python
import pandas as pd

# Load metadata
df = pd.read_csv("dataset/metadata_train.csv", sep="|")

# Check for empty text
empty_text = df[df['text'].str.strip() == ""]
print(f"Segments with empty text: {len(empty_text)}")  # Should be 0

# Check for duplicate audio files with same text
duplicates = df[df.duplicated(subset=['audio_file', 'text'], keep=False)]
print(f"Duplicate entries: {len(duplicates)}")  # Should be 0
```

### Manual Verification
1. Load a few audio segments from dataset
2. Compare audio content with text
3. Verify they match semantically
4. Check timing alignment

---

## Migration Guide

### For New Datasets
✅ **No action needed** - New processing automatically uses the fixed code

### For Existing Datasets

**Option 1: Reprocess from scratch (Recommended)**
```bash
# Re-run the YouTube downloader/SRT processor
python your_processing_script.py \
    --srt_file video.srt \
    --media_file video.mp4 \
    --output_dir dataset_fixed \
    --use_vad_refinement \
    --use_enhanced_vad
```

**Option 2: Fix existing dataset**
```bash
# Use the repair script
python fix_vad_srt_mismatch.py \
    --dataset_dir dataset_old \
    --srt_file original_video.srt
```

**Option 3: Manual cleanup**
1. Load metadata CSV
2. Remove rows where `text` is empty or very short (< 3 chars)
3. Check for obvious mismatches
4. Re-save CSV

---

## Technical Details

### Overlap Calculation Formula
```python
overlap_start = max(audio_start, srt_start)
overlap_end = min(audio_end, srt_end)
overlap_duration = max(0, overlap_end - overlap_start)
overlap_ratio = overlap_duration / audio_duration
```

### Matching Threshold
- **Minimum overlap**: 30% of audio segment must overlap with SRT
- **Best match wins**: Segment with highest overlap gets the text
- **Others discarded**: Segments below threshold are skipped

### Timing References
- `segment_start_time`: Buffered start (includes 0.2s padding)
- `original_srt_start`: Actual SRT start time (no buffer)
- `original_srt_end`: Actual SRT end time (no buffer)

This ensures accurate overlap calculation even when buffer is added.

---

## Files Modified

1. **`utils/srt_processor_vad.py`**
   - Modified `refine_segment_with_vad()` function
   - Modified `extract_segments_with_vad()` function
   - Added overlap-based text assignment

2. **`fix_vad_srt_mismatch.py`** (NEW)
   - Dataset repair utility
   - SRT re-matching logic
   - Backup and validation

---

## Future Enhancements

### Potential Improvements
1. **Store timestamps**: Save original SRT start/end in metadata for verification
2. **Confidence scores**: Include overlap ratio as quality metric
3. **Multi-SRT matching**: Handle cases where audio spans multiple SRT entries
4. **Audio analysis**: Use speech recognition to validate text-audio match

### Monitoring
- Track rejected segments count
- Log overlap ratios for debugging
- Report quality metrics per dataset

---

## Support

If you encounter issues:

1. **Check logs**: Look for overlap calculation details
2. **Verify SRT file**: Ensure SRT timings are accurate
3. **Test with small dataset**: Process 1-2 videos first
4. **Compare before/after**: Use repair script on test data

For questions, check the codebase or create an issue.

---

**Status**: ✅ FIXED (2025-10-17)  
**Priority**: CRITICAL  
**Impact**: High - Affects all VAD-based YouTube dataset processing
