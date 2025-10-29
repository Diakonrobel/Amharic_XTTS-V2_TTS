# SRT and YouTube Processing Unification

## Problem

**Critical Issue**: SRT batch processing had text-audio cutoff problems - segments were cut off at start/end, and text didn't match the audio properly. Meanwhile, YouTube batch processing worked perfectly.

### Symptoms
- Audio segments cut off at beginning or end
- Text in metadata doesn't precisely match audio content
- Inconsistent quality between YouTube and SRT processing
- Training data corruption

## Root Cause

The codebase had **TWO DIFFERENT CODE PATHS** for processing media+SRT:

1. **YouTube Batch Processing** (working perfectly):
   - Location: `batch_processor.process_youtube_batch()` line 395
   - **Always uses basic `srt_processor.process_srt_with_media()`**
   - **VAD intentionally disabled** with comment: "CRITICAL: Silero VAD disabled due to text-audio mismatch bug"
   - Result: Perfect, reliable processing

2. **SRT Batch Processing** (had cutoff issues):
   - Location: `batch_processor.process_srt_media_batch()` line 510
   - Attempted to use `srt_processor_vad.process_srt_with_media_vad()` when VAD enabled
   - VAD processing causes text-audio cutoffs and misalignment
   - Result: Corrupted segments with cutoffs

## Solution

### Unified Code Path

**Make SRT batch processing use the EXACT SAME logic as YouTube batch processing:**

```python
# BEFORE (SRT batch - buggy)
if use_vad_refinement:
    # Use VAD processor (causes cutoffs)
    train_csv, eval_csv, duration, _quality_stats = srt_processor_vad.process_srt_with_media_vad(...)
else:
    # Use basic processor
    train_csv, eval_csv, duration = srt_processor.process_srt_with_media(...)

# AFTER (SRT batch - fixed, matches YouTube)
if use_vad_refinement:
    print("⚠ VAD disabled (known cutoff issue) - using standard processing")

# Always use basic processor (same as YouTube)
train_csv, eval_csv, duration = srt_processor.process_srt_with_media(...)
```

### Key Changes

1. **Removed VAD Code Path** from `batch_processor.process_srt_media_batch()`
   - No longer calls `srt_processor_vad.process_srt_with_media_vad()`
   - Always uses `srt_processor.process_srt_with_media()`
   - Same behavior as YouTube batch processing

2. **Consistent Parameters**:
   - `speaker_name`: Properly forwarded
   - `min_duration`: Properly forwarded  
   - `max_duration`: Properly forwarded
   - `buffer`: Audio padding to prevent cutoffs (0.4s default)
   - `language`: Canonicalized language code

3. **User Feedback**:
   - If VAD is requested, show warning that it's disabled
   - Explain it's using same reliable path as YouTube

## Why VAD Causes Problems

VAD (Voice Activity Detection) attempts to refine segment boundaries by detecting speech vs silence. However:

1. **Boundary Detection Issues**: VAD can cut speech too early or too late
2. **Text-Audio Mismatch**: When VAD changes boundaries, text may not match new audio
3. **Ejective Consonants**: Languages like Amharic have sounds VAD misinterprets as silence
4. **Complexity**: Additional processing introduces more failure points

The basic SRT processor is **simpler and more reliable**:
- Uses exact SRT timestamps (with buffer padding)
- No boundary modifications
- Text always matches audio because timestamps are preserved
- Proven to work perfectly in YouTube batch processing

## Code Paths Now Unified

### Before (2 different paths)
```
YouTube Batch ──→ srt_processor.process_srt_with_media() ──→ ✓ Works perfectly

SRT Batch ──┬──→ if VAD: srt_processor_vad.process_srt_with_media_vad() ──→ ✗ Cutoffs
            └──→ else: srt_processor.process_srt_with_media() ──→ ✓ Works
```

### After (1 unified path)
```
YouTube Batch ──┐
                ├──→ srt_processor.process_srt_with_media() ──→ ✓ Works perfectly
SRT Batch ──────┘
```

## Impact

✅ **Fixed Issues**:
- No more text-audio cutoffs in SRT processing
- Consistent quality between YouTube and SRT processing
- Reliable, predictable results
- Training data integrity guaranteed

✅ **Code Quality**:
- Eliminated duplicate/divergent code paths
- Reduced complexity
- Easier to maintain (one path instead of two)
- Consistent behavior across all processing modes

## Verification

After applying the fix:

1. **Process SRT+media files** and check that:
   - No audio cutoffs at segment boundaries
   - Text in metadata precisely matches audio content
   - Segment durations respect min/max settings
   - Buffer padding prevents speech cutoffs

2. **Compare with YouTube processing**:
   - Both should produce identical quality
   - Same processing messages in console
   - Same segment boundary handling

3. **Training quality**:
   - Model should train properly on SRT-processed data
   - No nonsensical outputs from misaligned segments

## Related Files

- `utils/batch_processor.py` - Unified processing logic
- `utils/srt_processor.py` - Basic processor (now used for everything)
- `utils/srt_processor_vad.py` - VAD processor (no longer used in batch)
- `knowledge.md` - Updated with fix information

## Future Considerations

**VAD Processing**:
- VAD remains available for single-file processing if users want to try it
- Not recommended due to known cutoff issues
- May be improved in future with better boundary detection
- For now, basic processing is the reliable choice

**Buffer Padding**:
- Default 0.4s padding prevents most cutoffs
- Users can adjust via UI if needed (0.2s - 1.0s)
- Higher padding = safer but may include extra silence

**Consistency is Key**:
- Always keep YouTube and SRT batch processing in sync
- Both should use identical code paths
- Any future improvements should apply to both
