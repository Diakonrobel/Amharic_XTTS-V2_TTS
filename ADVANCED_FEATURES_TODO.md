# Advanced Features Implementation Status

## ✅ Completed Features

### 1. Dataset Tracking System
**Status:** ✅ COMPLETE
- Module: `utils/dataset_tracker.py`
- JSON-based tracking with `dataset_history.json`
- YouTube duplicate detection by video ID + language
- File duplicate detection by hash + filename  
- Integrated into YouTube downloader and SRT processor
- Automatic duplicate detection with user-friendly messages

### 2. Batch YouTube Processing
**Status:** ✅ COMPLETE
- Module: `utils/batch_processor.py`
- Parse comma/newline/space-separated URLs
- Process multiple videos sequentially
- Merge datasets into single unified output
- Smart file renaming (`merged_00000001.wav`)
- Automatic cleanup of temporary datasets

### 3. Dataset Merging
**Status:** ✅ COMPLETE
- Function: `merge_datasets()` in `batch_processor.py`
- Combines multiple dataset directories
- Preserves all audio files with new naming
- Shuffles and splits train/eval properly
- Language file preservation

### 4. Core Improvements (Previous)
- ✅ YouTube subtitle download with rate limit bypass
- ✅ Multi-strategy subtitle extraction (XML, JSON3, srv1/2/3)
- ✅ Gzip decompression for YouTube responses
- ✅ Intelligent audio buffering (0.2s with midpoint calculation)
- ✅ Correct language detection and lang.txt writing
- ✅ Ethiopian languages support (Amharic, Oromo, Tigrinya, etc.)

---

## 🚧 In Progress / TODO

### 1. UI Integration for Batch Processing
**Status:** 🚧 TODO
**Location:** `xtts_demo.py` - YouTube accordion

**Implementation Plan:**
```python
# Add to YouTube accordion UI
youtube_url = gr.Textbox(
    label="YouTube URL(s)",
    placeholder="Enter single URL or multiple URLs separated by commas/newlines",
    lines=3  # Multi-line for batch input
)

youtube_batch_mode = gr.Checkbox(
    label="Batch Mode (Process multiple URLs)",
    value=False,
    info="Enable to process comma-separated URLs as single dataset"
)

# Update download function
def download_youtube_video(url, transcript_lang, language, out_path, batch_mode, progress):
    if batch_mode:
        urls = batch_processor.parse_youtube_urls(url)
        if len(urls) > 1:
            # Use batch processor
            train_csv, eval_csv, video_infos = batch_processor.process_youtube_batch(
                urls=urls,
                transcript_lang=transcript_lang,
                out_path=out_path,
                youtube_downloader=youtube_downloader,
                srt_processor=srt_processor,
                progress_callback=lambda p, desc: progress(p, desc=desc)
            )
            
            # Track batch dataset
            total_segments = len(pd.read_csv(train_csv, sep='|')) + len(pd.read_csv(eval_csv, sep='|'))
            
            # Format summary
            summary = batch_processor.format_batch_summary(video_infos, total_segments)
            return summary
    
    # Existing single video processing...
```

### 2. VAD-Enhanced Segmentation
**Status:** 🚧 TODO - HIGH PRIORITY
**Goal:** Detect word boundaries while respecting SRT timestamps

**Implementation Plan:**

**A. Install Silero VAD:**
```bash
pip install silero-vad
```

**B. Create `utils/vad_segmentation.py`:**
```python
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

def refine_segment_boundaries(
    audio_tensor: torch.Tensor,
    start_time: float,
    end_time: float,
    sr: int,
    buffer: float = 0.2
) -> Tuple[float, float]:
    """
    Use VAD to refine segment boundaries while respecting SRT timestamps.
    
    Args:
        audio_tensor: Audio tensor
        start_time: Original start time from SRT
        end_time: Original end time from SRT
        sr: Sample rate
        buffer: Maximum buffer to extend (seconds)
        
    Returns:
        Tuple of (refined_start, refined_end)
    """
    # Load Silero VAD model
    model = load_silero_vad()
    
    # Extract segment with buffer
    buffered_start = max(0, start_time - buffer)
    buffered_end = end_time + buffer
    
    start_sample = int(buffered_start * sr)
    end_sample = int(buffered_end * sr)
    segment = audio_tensor[start_sample:end_sample]
    
    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        segment, 
        model,
        sampling_rate=sr,
        threshold=0.5
    )
    
    if not speech_timestamps:
        # No speech detected, use SRT timestamps
        return start_time, end_time
    
    # Find first and last speech segments
    first_speech = speech_timestamps[0]['start'] / sr + buffered_start
    last_speech = speech_timestamps[-1]['end'] / sr + buffered_start
    
    # Constrain to buffer limits
    refined_start = max(start_time - buffer, min(start_time, first_speech))
    refined_end = min(end_time + buffer, max(end_time, last_speech))
    
    return refined_start, refined_end
```

**C. Integrate into `srt_processor.py`:**
```python
from utils import vad_segmentation

def extract_segments_from_audio(
    ...
    use_vad_refinement: bool = True  # NEW parameter
):
    ...
    
    for idx, (start_time, end_time, text) in iterator:
        ...
        
        # Existing intelligent buffering
        buffered_start = ...
        buffered_end = ...
        
        # Optional VAD refinement
        if use_vad_refinement:
            buffered_start, buffered_end = vad_segmentation.refine_segment_boundaries(
                audio_tensor=wav,
                start_time=buffered_start,
                end_time=buffered_end,
                sr=sr,
                buffer=buffer
            )
        
        ...
```

**Benefits:**
- ✅ More accurate word boundaries
- ✅ Removes leading/trailing silence
- ✅ Respects SRT timing constraints
- ✅ Better training data quality

### 3. Multiple SRT+Media File Upload
**Status:** 🚧 TODO
**Location:** `xtts_demo.py` - SRT accordion

**Implementation Plan:**
```python
# Update SRT accordion UI
srt_files = gr.File(
    file_count="multiple",  # Changed from "single"
    label="SRT/VTT Subtitle Files",
    file_types=[".srt", ".vtt"],
)

media_files = gr.File(
    file_count="multiple",  # Changed from "single"
    label="Media Files (Audio or Video)",
    file_types=[".mp4", ".mkv", ".avi", ".wav", ".mp3", ".flac"],
)

srt_batch_mode = gr.Checkbox(
    label="Batch Mode (Process multiple pairs)",
    value=False,
    info="Pairs SRT files with media files by matching filenames"
)

def process_srt_media(srt_files, media_files, language, out_path, batch_mode, progress):
    if batch_mode and len(srt_files) > 1:
        # Process each pair
        temp_datasets = []
        
        for i, srt_file in enumerate(srt_files):
            # Find matching media file (by stem name)
            media_file = find_matching_media(srt_file, media_files)
            if not media_file:
                print(f"Warning: No matching media for {srt_file}")
                continue
            
            # Process to temp dataset
            temp_dir = os.path.join(out_path, f"temp_dataset_{i}")
            train_csv, eval_csv, duration = srt_processor.process_srt_with_media(
                srt_path=srt_file,
                media_path=media_file,
                output_dir=temp_dir,
                language=language
            )
            temp_datasets.append(temp_dir)
        
        # Merge all datasets
        final_dir = os.path.join(out_path, "dataset")
        train_csv, eval_csv, total = batch_processor.merge_datasets(
            dataset_paths=temp_datasets,
            output_dir=final_dir
        )
        
        return f"✓ Batch SRT Processing Complete! Processed {len(temp_datasets)} pairs, {total} total segments"
    
    # Existing single file processing...
```

### 4. Dataset History Viewer UI
**Status:** 🚧 TODO - NICE TO HAVE
**Location:** New accordion in Tab 1

**Implementation Plan:**
```python
with gr.Accordion("📊 Dataset History", open=False):
    history_display = gr.Textbox(
        label="Processing History",
        lines=15,
        interactive=False
    )
    
    with gr.Row():
        refresh_history_btn = gr.Button("Refresh History")
        clear_history_btn = gr.Button("Clear History", variant="stop")
    
    def show_history(out_path):
        tracker = dataset_tracker.get_tracker(
            os.path.join(out_path, "dataset_history.json")
        )
        return tracker.format_history_display(limit=20)
    
    refresh_history_btn.click(
        fn=show_history,
        inputs=[out_path],
        outputs=[history_display]
    )
```

---

## 📋 Priority Order

1. **HIGH**: VAD-Enhanced Segmentation
   - Most impactful for quality
   - Relatively straightforward implementation
   - Improves all dataset creation methods

2. **MEDIUM**: Batch YouTube UI Integration
   - High user value
   - Already have backend complete
   - Just needs UI wiring

3. **MEDIUM**: Multiple SRT File Upload  
   - Useful for large projects
   - Backend logic straightforward
   - Requires file pairing logic

4. **LOW**: Dataset History Viewer
   - Nice-to-have feature
   - History already tracked
   - Just needs display UI

---

## 🔧 Implementation Commands

### Quick Setup for VAD:
```bash
pip install silero-vad
```

### Test Batch Processing:
```python
from utils import batch_processor, youtube_downloader, srt_processor

urls = batch_processor.parse_youtube_urls("""
https://youtube.com/watch?v=VIDEO1,
https://youtube.com/watch?v=VIDEO2,
https://youtube.com/watch?v=VIDEO3
""")

train_csv, eval_csv, infos = batch_processor.process_youtube_batch(
    urls=urls,
    transcript_lang="am",
    out_path="./finetune_models",
    youtube_downloader=youtube_downloader,
    srt_processor=srt_processor
)

summary = batch_processor.format_batch_summary(infos, total_segments)
print(summary)
```

---

## 📝 Notes

- All core functionality for batch processing is COMPLETE
- VAD enhancement will significantly improve dataset quality
- UI integration is straightforward - just wiring existing functions
- Consider adding VAD as optional parameter (default enabled)
- Batch mode should track as single "batch" entry in history

---

## 🎯 Next Session Goals

1. Implement VAD-enhanced segmentation
2. Wire batch YouTube processing to UI
3. Test with real Amharic videos
4. Verify audio quality improvements
