# YouTube Processing Workflow Enhancement

## Current Issues

1. **No subtitle check before download** - Wastes bandwidth downloading videos without subtitles
2. **Audio separation during download** - Background removal happens inline (lines 935-960), should be post-processing
3. **No validation** - Doesn't verify both audio AND subtitles succeeded before proceeding

## Enhanced Workflow

### Phase 1: Pre-flight Check
```
1. Fetch video info
2. Check subtitle availability:
   - ✓ Has manual subtitles: {language}
   - ✓ Has auto-captions: {language}  
   - ⚠ No subtitles - Whisper fallback available
   - ❌ No subtitles - Whisper disabled (FAIL FAST)
3. Display status to user BEFORE downloading
```

### Phase 2: Download
```
4. Download audio (with progress)
5. Download SRT (with progress)
6. Validate both files exist and are non-empty
```

### Phase 3: Post-Processing (NEW)
```
7. IF remove_background_music requested:
   - Check audio file exists ✓
   - Check SRT file exists ✓
   - Apply Demucs separation
   - Replace audio file with clean version
   - Show progress: "🎵 Removing background music..."
8. Return paths
```

## Benefits

✅ **Fail fast** - Know if video has subtitles BEFORE downloading  
✅ **Clear status** - User sees what's available upfront  
✅ **Efficient** - Don't waste GPU on videos without subtitles  
✅ **Safe** - Audio separation only after confirming both media+SRT exist  
✅ **Predictable** - Each step has clear success/failure state  

## Implementation

### Key Changes

1. **Move background removal** out of download loop (lines 935-960)
2. **Add subtitle pre-check** with clear messaging
3. **Separate post-processing function**:
   ```python
   def apply_post_processing(
       audio_path: str,
       srt_path: Optional[str],
       remove_background_music: bool,
       model: str,
       quality: str
   ) -> str:
       """Apply post-processing after successful download"""
   ```

4. **Enhanced info display**:
   ```
   📹 Video: "Title Here" (5:32)
   ✓ Subtitles available: manual (am), auto (en)
   ✓ Will download with background removal
   
   Downloading...
   ✓ Audio: video.wav (12.3 MB)
   ✓ Subtitles: video.am.srt (45 KB)
   
   🎵 Removing background music...
   ✓ Clean audio ready!
   ```

### Pseudo-code

```python
def enhanced_download_youtube_video(...):
    # Phase 1: Pre-flight
    info = get_video_info(url, ...)
    subtitle_status = check_subtitle_availability(info, language)
    
    if not subtitle_status['available'] and not use_whisper:
        raise ValueError(f"❌ No subtitles for '{language}' and Whisper disabled")
    
    print_availability_report(info, subtitle_status)
    
    # Phase 2: Download
    audio_path = download_audio(url, ...)
    srt_path = download_subtitles(url, language, ...) or None
    
    validate_download(audio_path, srt_path, use_whisper)
    
    # Phase 3: Post-process (ONLY if download succeeded)
    if remove_background_music and audio_path:
        print("\n🎵 Post-processing: Removing background music...")
        audio_path = apply_audio_separation(audio_path, model, quality)
        print("✓ Audio separation complete!")
    
    return audio_path, srt_path, info
```

## Migration Path

1. ✅ First: Add debug logging (DONE - cookies/proxy logging added)
2. ✅ Next: Fix client selection (DONE - tv,web_safari,web for cookies)
3. **TODO**: Add subtitle pre-check with clear status
4. **TODO**: Move background removal to post-processing
5. **TODO**: Add validation between phases

## User Experience

### Before (Current)
```
Downloading from YouTube...
Downloading audio...
[long yt-dlp output]
⚠ Background music removal requested...
[demucs processing]
❌ Error: No subtitles found
```
^ Wasted time on background removal for video without subtitles!

### After (Enhanced)
```
📹 Checking video: "Title" (5:32)
❌ No subtitles available for 'am'
⚠ Whisper transcription will be used (slower)

Continue? [Y/n]

Downloading audio... ✓
Transcribing with Whisper... ✓
🎵 Removing background music... ✓

✅ Ready for dataset creation!
```

^ Clear, predictable, efficient!

## Files to Update

1. `utils/youtube_downloader.py`
   - Add `check_subtitle_availability()`
   - Move background removal to `apply_post_processing()`
   - Add validation steps

2. `utils/batch_processor.py`
   - Update to use enhanced workflow
   - Show per-video subtitle status

3. `xtts_demo.py`
   - Update progress messages to show phases
   - Add subtitle status to UI

## Status

- ✅ Debug logging added
- ✅ Client selection fixed
- ⏳ Workflow enhancement (THIS DOCUMENT)
- ⏸️ Awaiting implementation

## Next Steps

1. Implement `check_subtitle_availability()` function
2. Refactor background removal into separate function  
3. Add validation checkpoints
4. Update UI messages
5. Test with various video types

---

**Priority:** Medium  
**Impact:** High (better UX, saves time/resources)  
**Effort:** ~2 hours  

Last updated: January 24, 2025
