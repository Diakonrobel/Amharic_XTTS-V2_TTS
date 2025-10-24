# YouTube URL File Upload Feature

## Feature Request
Add support for uploading `.txt` files containing multiple YouTube URLs (one per line) in the Gradio WebUI.

## Current State
- Users manually paste URLs into text box
- Multiple URLs separated by newlines/spaces
- No file upload support

## Proposed Enhancement

### 1. Add File Upload Component

In `xtts_demo.py`, add a File component in the YouTube tab:

```python
with gr.Row():
    youtube_url = gr.Textbox(
        label="YouTube URL(s)",
        placeholder="Enter one or more YouTube URLs (one per line) or upload a file below...",
        lines=3
    )
    
with gr.Row():
    url_file_upload = gr.File(
        label="Or upload URL list (.txt file, one URL per line)",
        file_types=[".txt"],
        type="filepath"
    )
```

### 2. Create URL Parser Function

Add to `utils/batch_processor.py`:

```python
def parse_url_file(file_path: str) -> List[str]:
    """
    Parse YouTube URLs from uploaded text file.
    
    Args:
        file_path: Path to uploaded .txt file
        
    Returns:
        List of valid YouTube URLs
        
    Format:
        - One URL per line
        - Lines starting with # are comments (ignored)
        - Empty lines are ignored
        - Whitespace is trimmed
    """
    urls = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Validate YouTube URL
                if 'youtube.com/watch' in line or 'youtu.be/' in line:
                    urls.append(line)
                else:
                    print(f"⚠ Line {line_num}: Invalid YouTube URL: {line[:50]}")
        
        print(f"✓ Loaded {len(urls)} URLs from file")
        return urls
        
    except Exception as e:
        print(f"❌ Error reading URL file: {e}")
        return []
```

### 3. Update Processing Function

Modify `download_youtube_video()` in `xtts_demo.py` to handle both text input and file:

```python
def download_youtube_video(url, url_file, transcript_lang, ...):
    """Download YouTube video(s) - supports text input and file upload"""
    
    # Collect URLs from both sources
    urls_from_text = []
    urls_from_file = []
    
    # Parse text input
    if url and url.strip():
        urls_from_text = batch_processor.parse_youtube_urls(url)
    
    # Parse uploaded file
    if url_file:
        urls_from_file = batch_processor.parse_url_file(url_file)
    
    # Combine and deduplicate
    all_urls = list(set(urls_from_text + urls_from_file))
    
    if not all_urls:
        return "❌ No valid YouTube URLs found in text or file!"
    
    # Show summary
    summary = f"📋 Found URLs:\n"
    if urls_from_text:
        summary += f"  - {len(urls_from_text)} from text input\n"
    if urls_from_file:
        summary += f"  - {len(urls_from_file)} from uploaded file\n"
    summary += f"  - {len(all_urls)} total (after removing duplicates)\n\n"
    
    print(summary)
    
    # Continue with existing processing...
    if batch_mode and len(all_urls) > 1:
        return process_youtube_batch_urls(all_urls, ...)
    # ... rest of code
```

### 4. Update UI Component Linking

Update the `.click()` handler to include the new file parameter:

```python
youtube_btn.click(
    fn=download_youtube_video,
    inputs=[
        youtube_url,          # Existing text input
        url_file_upload,      # NEW: File upload
        transcript_lang,
        language,
        # ... other parameters
    ],
    outputs=youtube_output
)
```

## Example URL File Format

Users create a file like `youtube_urls.txt`:

```
# My YouTube video list for training
# Lines starting with # are comments

https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2
https://www.youtube.com/watch?v=VIDEO_ID_3

# Can add more later:
https://youtu.be/SHORT_ID_4
https://www.youtube.com/watch?v=VIDEO_ID_5
```

## User Workflow

### Option 1: Manual Entry (Current)
1. Paste URLs in textbox
2. Click Download & Process

### Option 2: File Upload (New!)
1. Create `urls.txt` with one URL per line
2. Click "Upload URL list"
3. Select file
4. Click Download & Process

### Option 3: Both!
1. Paste some URLs in textbox
2. Also upload a file with more URLs
3. System combines both, removes duplicates
4. Processes all URLs

## Benefits

✅ **Easier bulk processing** - Save URL lists for reuse  
✅ **Better organization** - Keep URL collections in files  
✅ **Comments support** - Document which videos are which  
✅ **Less error-prone** - No copy-paste mistakes  
✅ **Reusable** - Same file can be used multiple times  

## Implementation Files

1. **`utils/batch_processor.py`** - Add `parse_url_file()` function
2. **`xtts_demo.py`** - Add File component & update processing logic

## Code Location

In `xtts_demo.py`, around line 2700-2800 (YouTube Processing tab):

```python
# CURRENT (around line 2719):
youtube_url = gr.Textbox(...)

# ADD AFTER:
url_file_upload = gr.File(
    label="📎 Or upload URL list (.txt)",
    file_types=[".txt"],
    type="filepath"
)
```

## Testing

Test cases:
1. ✓ Upload file only
2. ✓ Text input only (existing)
3. ✓ Both file and text (should combine)
4. ✓ File with comments and empty lines
5. ✓ File with invalid URLs (should warn but continue)
6. ✓ Duplicate URLs (should deduplicate)

## Status

- ⏸️ **Not yet implemented**
- Priority: **Medium**
- Effort: **~30 minutes**
- Impact: **High** (much better UX for batch users)

## Next Steps

1. Add `parse_url_file()` to `batch_processor.py`
2. Add File component to YouTube tab in `xtts_demo.py`
3. Update `download_youtube_video()` to merge URLs from both sources
4. Test with sample URL file
5. Update documentation

---

**Ready to implement!** This will significantly improve the batch processing workflow.

Last updated: January 24, 2025
