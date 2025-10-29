# SRT Processing Text-Audio Mismatch Fix

## Problem

**Critical Issue**: When processing SRT+media files in batch mode, the text transcriptions in the metadata CSV were misaligned with their corresponding audio WAV segments, causing corrupted training data.

### Symptoms
- Metadata CSV text doesn't match actual audio content
- Training produces poor quality or nonsensical outputs
- VAD refinement made the issue worse

## Root Cause

The `batch_processor.process_srt_media_batch()` function was calling `srt_processor.process_srt_with_media()` **without critical parameters**:

1. **Missing `speaker_name`** - Defaulted to wrong speaker identifier
2. **Missing `min_duration`** - Used hardcoded 1.0s instead of UI value
3. **Missing `max_duration`** - Used hardcoded 20.0s instead of UI value  
4. **Ignored `use_vad_refinement`** - Always used basic processor, even when VAD was enabled

This caused the SRT processor to use different segment boundaries than expected, resulting in text-audio misalignment.

### Why YouTube Processing Worked

YouTube batch processing in `batch_processor.process_youtube_batch()` (line 395) **intentionally disabled VAD** with a comment:

```python
# CRITICAL: Silero VAD disabled due to text-audio mismatch bug
# Always use standard SRT processor for reliable results
```

This avoided the bug by always using the basic processor, but it masked the underlying parameter issue.

## Solution

### Changes Made

1. **Updated `batch_processor.process_srt_media_batch()`** (`utils/batch_processor.py`):
   - Added parameters: `speaker_name`, `min_duration`, `max_duration`, `use_vad_refinement`
   - Routes to correct processor based on `use_vad_refinement` flag:
     - If enabled: calls `srt_processor_vad.process_srt_with_media_vad()`
     - If disabled: calls `srt_processor.process_srt_with_media()`
   - Properly forwards all parameters to the processor

2. **Updated `xtts_demo.py`** (line 1310):
   - Passes UI parameters to `batch_processor.process_srt_media_batch()`:
     - `speaker_name` from UI
     - `min_seg_duration` from UI
     - `max_seg_duration` from UI
     - `use_vad_refinement` from UI checkbox

3. **Added module-level import** (`utils/batch_processor.py`):
   - Import `srt_processor_vad` at top to avoid repeated imports in loop

### Code Flow (After Fix)

```
UI Input (xtts_demo.py)
  ↓ speaker_name, min_duration, max_duration, use_vad_refinement
batch_processor.process_srt_media_batch()
  ↓
  ├─ if use_vad_refinement:
  │    srt_processor_vad.process_srt_with_media_vad(...)
  │      → Correct VAD processing with all parameters
  │
  └─ else:
       srt_processor.process_srt_with_media(...)
         → Standard processing with all parameters
```

## Verification

After applying the fix:

1. **Check metadata alignment**:
   ```python
   import pandas as pd
   import torchaudio
   
   # Read metadata
   df = pd.read_csv('dataset/metadata_train.csv', sep='|')
   
   # Check a few samples
   for idx in range(3):
       audio_file = df.iloc[idx]['audio_file']
       text = df.iloc[idx]['text']
       
       # Load audio
       wav, sr = torchaudio.load(f'dataset/{audio_file}')
       duration = wav.shape[1] / sr
       
       print(f"Audio: {audio_file}")
       print(f"Duration: {duration:.2f}s")
       print(f"Text: {text}")
       print()
   ```

2. **Verify parameters are honored**:
   - Check that all segments are within `min_duration` to `max_duration` range
   - Verify speaker name matches UI input
   - Confirm VAD refinement produces different boundaries when enabled

## Impact

- **Training Quality**: Fixed corrupted training data that caused poor model outputs
- **VAD Support**: Can now safely use VAD refinement in SRT batch processing
- **Consistency**: SRT processing now matches YouTube processing behavior

## Related Files

- `utils/batch_processor.py` - Main fix location
- `xtts_demo.py` - UI parameter passing
- `utils/srt_processor.py` - Standard SRT processor
- `utils/srt_processor_vad.py` - VAD-enhanced processor
- `knowledge.md` - Updated with fix information

## Prevention

**Always pass these critical parameters** when calling SRT processing functions:
- `speaker_name` - Don't rely on defaults
- `min_duration` / `max_duration` - Explicitly set boundaries
- `use_vad_refinement` - Choose processor explicitly
- `gradio_progress` - For UI feedback

**Verify text-audio alignment** before training:
- Listen to a few WAV files
- Check that text in metadata matches audio content
- Verify segment durations are within expected range
