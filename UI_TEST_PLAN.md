# Web UI Testing Plan - Advanced Dataset Processing Features

## ✅ Pre-Test Checklist

### Environment Setup
- [ ] Python 3.9+ installed
- [ ] All dependencies from `requirements.txt` installed
- [ ] FFmpeg installed and available in PATH
- [ ] PyTorch 2.1+ installed (CPU or CUDA)

### Dependency Installation
```powershell
# Windows
.\install.bat

# Or manual installation
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Launch Web UI
```powershell
# Windows
.\start.bat

# Or directly
python xtts_demo.py --port 5003
```

Access at: `http://127.0.0.1:5003`

---

## 🎯 Test Scenarios

### **Test 1: SRT + Media File Processing**

#### Setup:
1. Prepare a short video file (MP4/MKV) or audio file (WAV/MP3)
2. Create or obtain an SRT subtitle file matching the media

#### Test Steps:
1. Launch the Web UI
2. Navigate to **Tab 1: Data Processing**
3. Scroll to "📝 SRT + Media File Processing" accordion
4. Click to expand the accordion
5. Upload SRT file using the "SRT/VTT Subtitle File" component
6. Upload media file using the "Media File (Audio or Video)" component
7. Select target language from the dropdown
8. Click **"Process SRT + Media"** button

#### Expected Results:
- ✅ Progress indicator shows processing stages:
  - "Initializing SRT processor..."
  - "Processing SRT and extracting audio segments..."
  - "SRT processing complete!"
- ✅ Status textbox displays:
  - ✓ Success indicator
  - Number of segments processed
  - Total audio duration
  - Dataset output path
- ✅ Files created in `output/dataset/`:
  - `wavs/` directory with audio segments
  - `metadata_train.csv`
  - `metadata_eval.csv`
  - `lang.txt`

#### Error Cases to Test:
- Upload SRT without media file → Error message
- Upload media without SRT file → Error message
- Upload mismatched file types → Graceful error handling
- Malformed SRT file → Clear error message

---

### **Test 2: YouTube Video Download**

#### Setup:
1. Find a YouTube video with subtitles/transcripts available
2. Copy the video URL

#### Test Steps:
1. Navigate to **Tab 1: Data Processing**
2. Scroll to "📹 YouTube Video Download" accordion
3. Click to expand the accordion
4. Paste YouTube URL in the text field
5. Select preferred transcript language
6. Click **"Download & Process YouTube"** button

#### Expected Results:
- ✅ Progress indicator shows:
  - "Initializing YouTube downloader..."
  - "Downloading video and subtitles..."
  - "Processing transcript and audio..."
  - "YouTube processing complete!"
- ✅ Status textbox displays:
  - ✓ Success indicator
  - Video title
  - Video duration
  - Number of segments processed
  - Dataset output path
- ✅ Files created in `output/dataset/`
- ✅ yt-dlp auto-update occurs

#### Error Cases to Test:
- Invalid YouTube URL → Clear error message
- Video without subtitles → Warning about no transcripts
- Age-restricted video → Appropriate error handling
- Network issues → Connection error message

---

### **Test 3: RMS-Based Audio Slicing**

#### Setup:
1. Prepare a long audio file (WAV/MP3/FLAC) with speech and pauses

#### Test Steps:
1. Navigate to **Tab 1: Data Processing**
2. Scroll to "✂️ RMS-Based Audio Slicing" accordion
3. Click to expand the accordion
4. Upload audio file
5. Adjust slicing parameters:
   - **Silence Threshold**: Try -40 dB (default)
   - **Min Segment Length**: Try 5 seconds (default)
   - **Min Silence Interval**: Try 0.3 seconds (default)
   - **Silence Padding**: Try 0.5 seconds (default)
6. Check "Auto-transcribe with Whisper" checkbox
7. Select Whisper model size
8. Click **"Slice Audio"** button

#### Expected Results:
- ✅ Progress indicator shows:
  - "Initializing audio slicer..."
  - "Loading audio..."
  - "Slicing audio..."
  - "Saving X segments..."
  - "Transcribing segments with Whisper..." (if auto-transcribe enabled)
  - "Creating metadata..."
  - "Audio slicing complete!"
- ✅ Status textbox displays:
  - ✓ Success indicator
  - Number of segments created
  - Segments save path
  - Auto-transcription status
- ✅ Files created:
  - Individual segment files in `output/dataset/wavs/`
  - If auto-transcribe: `metadata_train.csv` and `metadata_eval.csv`

#### Parameter Testing:
Test different parameter combinations:
- **Lower threshold** (-50 dB) → More sensitive to quiet audio
- **Higher threshold** (-30 dB) → Only splits on louder silences
- **Longer min length** (10s) → Fewer, longer segments
- **Shorter min length** (2s) → More, shorter segments

#### Error Cases to Test:
- Upload without file → Error message
- Too aggressive slicing parameters → No segments created warning
- Very large audio file → Appropriate handling/progress

---

## 🔍 Integration Testing

### Test 4: Sequential Processing
1. Process SRT + Media → Dataset created
2. Verify dataset files exist
3. Process YouTube → Same dataset directory
4. Verify new segments added to existing dataset
5. Check that segments don't conflict (unique filenames)

### Test 5: Different Languages
1. Test with English content
2. Test with Amharic content (if available)
3. Test with other supported languages
4. Verify `lang.txt` is updated correctly

### Test 6: Tab Integration
After processing in Tab 1:
1. Navigate to **Tab 2: Fine-tuning**
2. Click "Load Params from output folder"
3. Verify train/eval CSV paths are loaded
4. Verify language is detected correctly
5. Proceed with a short training run (1 epoch)

---

## 🎨 UI/UX Testing

### Visual Elements:
- [ ] Accordions expand/collapse properly
- [ ] Buttons have correct styling (variant='secondary')
- [ ] Status textboxes are read-only
- [ ] Progress bars show during processing
- [ ] Unicode symbols display correctly (✓, ❌, 📝, 📹, ✂️)
- [ ] Markdown headers render properly
- [ ] Sliders show current values
- [ ] Dropdowns populate correctly

### Responsiveness:
- [ ] Test on different screen sizes
- [ ] Components don't overlap
- [ ] Scrolling works smoothly
- [ ] Accordions don't interfere with each other

### Error Handling:
- [ ] Error messages are user-friendly
- [ ] Console shows detailed tracebacks
- [ ] UI doesn't freeze on errors
- [ ] Can recover from errors without restart

---

## 📝 Performance Testing

### Small Dataset (< 5 minutes audio):
- [ ] SRT processing completes in reasonable time
- [ ] YouTube download doesn't timeout
- [ ] Audio slicing is responsive

### Medium Dataset (5-30 minutes audio):
- [ ] Progress indicators update regularly
- [ ] UI remains responsive
- [ ] Memory usage is acceptable

### Large Dataset (> 30 minutes audio):
- [ ] Long operations don't timeout
- [ ] Progress tracking works throughout
- [ ] Segments are created incrementally

---

## ✅ Success Criteria

### Functionality:
- ✅ All 3 processing methods work correctly
- ✅ Datasets are created with proper structure
- ✅ Metadata CSVs are properly formatted
- ✅ Audio segments are playable and correct quality
- ✅ Error handling is robust

### User Experience:
- ✅ Instructions are clear
- ✅ Feedback is immediate and informative
- ✅ Progress is visible for long operations
- ✅ Errors don't crash the application
- ✅ UI is intuitive and easy to navigate

### Integration:
- ✅ Processed datasets work with Tab 2 (Fine-tuning)
- ✅ Multiple processing methods can be used together
- ✅ Existing datasets are handled correctly
- ✅ Language detection works across tabs

---

## 🐛 Known Limitations

1. **YouTube downloads require internet connection**
2. **FFmpeg must be installed for media processing**
3. **Large files may take significant time to process**
4. **SRT timestamps must be accurate for good alignment**
5. **Auto-transcription with Whisper is slower than pre-transcribed content**

---

## 📊 Test Results Template

```
Test Date: _____________
Tester: _____________
Environment: _____________

| Test # | Feature | Status | Notes |
|--------|---------|--------|-------|
| 1      | SRT + Media | ⬜ Pass / ⬜ Fail | |
| 2      | YouTube Download | ⬜ Pass / ⬜ Fail | |
| 3      | Audio Slicing | ⬜ Pass / ⬜ Fail | |
| 4      | Integration | ⬜ Pass / ⬜ Fail | |
| 5      | Languages | ⬜ Pass / ⬜ Fail | |
| 6      | Tab Integration | ⬜ Pass / ⬜ Fail | |
| 7      | UI/UX | ⬜ Pass / ⬜ Fail | |
| 8      | Performance | ⬜ Pass / ⬜ Fail | |

Overall Result: ⬜ PASS / ⬜ FAIL

Issues Found:
1. _____________
2. _____________
3. _____________
```

---

## 🔧 Troubleshooting

### Issue: UI won't launch
**Solution:** Check that all dependencies are installed, especially `gradio` and `coqui-tts`

### Issue: FFmpeg not found
**Solution:** Install FFmpeg and add to PATH. Test with `ffmpeg -version`

### Issue: Out of memory during processing
**Solution:** Process smaller chunks, or increase system RAM/swap

### Issue: Whisper transcription is very slow
**Solution:** Use smaller Whisper model size, or skip auto-transcription

### Issue: YouTube download fails
**Solution:** Check internet connection, update yt-dlp: `pip install -U yt-dlp`

---

## 📚 Additional Resources

- **WARP.md** - Technical documentation
- **README.md** - User guide with examples
- **IMPLEMENTATION_PLAN.md** - Development roadmap
- **test_advanced_features.py** - Automated tests

---

**Generated:** 2025-01-07  
**Version:** 1.0  
**Status:** Ready for Testing
