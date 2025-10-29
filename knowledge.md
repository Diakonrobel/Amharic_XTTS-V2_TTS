# XTTS Fine-tuning WebUI Project Knowledge

## Project Overview
XTTS (Coqui TTS) fine-tuning web interface for training custom voice models, with special support for Amharic and other languages.

## Key Components

### Training Pipeline
- Main training script: `headlessXttsTrain.py`
- Training utilities in `utils/` directory
- Checkpoint management via `utils/checkpoint_manager.py`
- Dataset validation through `utils/dataset_validator.py`

### Amharic Language Support
- Custom G2P (grapheme-to-phoneme) in `amharic_tts/g2p/`
- Hybrid tokenization system in `amharic_tts/tokenizer/`
- Text normalization and preprocessing in `amharic_tts/preprocessing/`
- Multiple G2P backends: TransPhone, espeak-ng, BPE-only mode

### Web Interface
- Dataset merger UI: `webui/dataset_merger_ui.py`
- YouTube processing: `webui/youtube_processing_alt.py`
- Dataset creator with audio recorder in `dataset_creator/`

### Audio Processing
- VAD-based segmentation: `utils/vad_slicer.py` and `utils/silero_vad_enhanced.py`
- SRT subtitle processing: `utils/srt_processor.py` and `utils/srt_processor_vad.py`
  - Now includes background music removal via Demucs (optional)
  - Retry logic with exponential backoff for FFmpeg operations
  - Quality metrics computation (SNR, speech probability, energy stability)
  - Enhanced error messages with troubleshooting steps
  - **Breaking change**: `process_srt_with_media_vad()` now returns 4 values (train, eval, duration, quality_stats)
- Background music removal: `utils/audio_background_remover.py`
- Audio augmentation: `utils/audio_augmentation.py`

## Important Patterns

### Dataset Format
- CSV with columns: audio_file, text, speaker_name
- Audio files should be WAV format, 22050 Hz recommended
- See DATASET_FORMAT.md for details

### Training Workflow
1. Prepare dataset (audio + transcriptions)
2. Validate dataset with `dataset_validator.py`
3. Run training via `headlessXttsTrain.py`
4. Monitor with TensorBoard
5. Test checkpoints for inference

### Platform-Specific Notes
- **Windows**: Use PowerShell scripts (`.ps1`) or batch files (`.bat`)
- **Lightning.ai**: Cookies-based YouTube download, see LIGHTNING_AI_COOKIES_SETUP.md
- **Colab/Kaggle**: Use notebook versions in `notebooks/` and `dataset_creator/`

## Common Tasks

### Adding New Language Support
1. Create G2P backend in `amharic_tts/g2p/`
2. Add tokenizer wrapper if needed
3. Update `utils/g2p_backend_selector.py`
4. Test with small dataset first

### Resuming Training
- Use checkpoint manager to select checkpoint
- See CHECKPOINT_RESUMPTION_FIX.md and RESUME_TRAINING_GUIDE.md
- Ensure vocab consistency when switching between G2P modes

### Dataset Management
- Merge datasets: `merge_datasets_auto.py` or `merge_datasets_simple.py`
- Incremental addition: `utils/incremental_dataset_merger.py`
- Statistics: `utils/dataset_statistics.py`

## Style Preferences
- Use Gradio for web interfaces
- Keep utilities modular in `utils/` directory
- Comprehensive error handling with actionable error messages
- Retry logic with exponential backoff for network/IO operations
- Quality metrics computation and reporting
- Progress feedback via gradio_progress parameter
- Document fixes in dedicated markdown files

## Known Issues & Fixes
- Amharic tokenization: See AMHARIC_BPE_FIX.md, AMHARIC_KEYERROR_FIX.md
- YouTube download 2025: See YOUTUBE_FIX_2025.md, YOUTUBE_2025_NO_COOKIES_BYPASS.md
- Training overfitting: See OVERFITTING_FIX_V2_AGGRESSIVE.md
- Checkpoint resume: See CHECKPOINT_RESUMPTION_FIX.md

## Testing
- Run validation scripts before training
- Test inference with `xtts_demo.py`
- Monitor logs in `logs/` directory (if configured)

## External Dependencies
- PyTorch (CPU or CUDA)
- TTS library (Coqui)
- Gradio for web UI
- yt-dlp for YouTube downloads
- Various audio processing libraries (see requirements.txt)