# XTTS Fine-Tuning WebUI Constitution

## Core Principles

### I. Modularity & Separation of Concerns
Core functionality organized into independent, focused modules:
- `utils/formatter.py` - Dataset preparation (audio processing, transcription, metadata)
- `utils/gpt_train.py` - Training orchestration (model setup, training loop)
- `utils/tokenizer.py` - Language-specific text processing
- `amharic_tts/*` - Self-contained Amharic language support with zero hard dependencies on main codebase

Each module must be independently testable and have clear, documented responsibilities.

### II. Graceful Degradation & Fallback Systems
Critical features must work with or without optional dependencies:
- **Amharic G2P:** Transphone → Epitran → Rule-based (always works)
- **Audio processing:** FFmpeg conversion with fallback error handling
- **Training resumption:** State preserved in files, UI restart-safe

Never fail completely when a fallback option exists.

### III. Language-Agnostic Core with Language-Specific Extensions
Core training pipeline supports any XTTS-compatible language:
- Base implementation language-neutral
- Language-specific features in separate modules (e.g., `amharic_tts/`)
- Language selection via ISO 639-3 codes
- Special handling documented and isolated (e.g., Japanese `num_workers=0`)

### IV. File-Based State Management
State persisted to filesystem, not in-memory:
- Training progress in checkpoint files
- Dataset metadata in CSV files
- Language settings in `lang.txt`
- Model configuration in JSON files

This enables:
- Safe UI restarts without data loss
- Incremental dataset updates
- Reproducible training runs
- Easy debugging and inspection

### V. Path Safety & Platform Compatibility
- Always use `pathlib.Path` for cross-platform compatibility
- Never hardcode absolute paths
- Support Windows (PowerShell), Linux (bash), macOS
- Use `os.path.join()` or Path operations for path construction

## Technical Standards

### Audio Processing
- **Input formats:** WAV, MP3, FLAC (auto-conversion via FFmpeg)
- **Sample rate:** 22.05kHz for training, 24kHz for output
- **Channels:** Mono preferred (auto-convert stereo)
- **Duration limits:** Enforce max length (default 11.6s per segment) to prevent OOM

### Dataset Quality
- **VAD filtering:** Always enabled in Faster Whisper
- **Sentence segmentation:** Split on punctuation with configurable buffer
- **Text cleaning:** Apply `multilingual_cleaners` from XTTS tokenizer
- **Metadata format:** Pipe-separated CSV (audio_file|text|speaker_name)
- **Train/eval split:** 15% evaluation by default

### Training Configuration
- **Checkpointing:** Save every 1000 steps, keep only best model
- **Batch size:** User-configurable (2-4 for consumer GPUs)
- **Learning rate:** 5e-06 (XTTS-optimized)
- **Max audio length:** 255995 frames (~11.6 seconds)
- **Base models:** Cache in `base_models/{version}/` to avoid re-downloading

### Code Quality
- **Error handling:** Graceful failures with user-friendly messages
- **Progress indication:** Use tqdm/Gradio progress bars for long operations
- **Logging:** Print to console (UI doesn't capture logs)
- **Documentation:** Docstrings for public functions, inline comments for complex logic

## Development Workflow

### Adding New Language Support
1. Test with existing XTTS tokenizer first
2. If special processing needed, create module in `amharic_tts/` pattern
3. Update `utils/tokenizer.py` for tokenization overrides
4. Add language-specific handling in `utils/gpt_train.py` if needed
5. Test both Web UI and headless modes
6. Document in `WARP.md` and README

### Adding New Features
1. Preserve backward compatibility with existing datasets/models
2. Make features opt-in via UI toggles or CLI flags
3. Provide sensible defaults that work for most users
4. Test on both CPU and GPU (where applicable)
5. Update both `xtts_demo.py` and `headlessXttsTrain.py` if feature applies to both

### Modifying Training Pipeline
1. Changes to `utils/gpt_train.py` must maintain compatibility with existing checkpoints
2. Test with at least 2 languages (English + one other)
3. Verify model optimization still works
4. Check inference after training completes
5. Document new parameters in code and `WARP.md`

## Testing Requirements

### Manual Testing Checklist (Minimum)
- [ ] Audio upload and transcription
- [ ] Dataset creation (train + eval metadata)
- [ ] Training runs for at least 2 epochs
- [ ] Model optimization creates `ready/` folder
- [ ] Inference loads model and generates audio
- [ ] Headless mode with same parameters works

### Automated Testing (When Available)
- Run Amharic tests: `python tests/test_amharic_integration.py`
- Run G2P tests: `python tests/test_amharic_g2p_comprehensive.py`
- Test new language modules with pytest

## Governance

This constitution defines the architectural principles for the XTTS Fine-Tuning WebUI project. When adding features or making changes:

1. **Backward Compatibility:** Existing datasets and models must continue to work
2. **Documentation:** Update WARP.md and README.md for user-facing changes
3. **Multi-Modal Support:** Test both Web UI and headless CLI
4. **Platform Testing:** At minimum, verify Windows + one other platform
5. **Memory Constraints:** Be mindful of VRAM limits (6-12GB target range)

When in doubt, prioritize:
1. User experience (clear errors, progress indication)
2. Robustness (fallbacks, error recovery)
3. Simplicity (avoid over-engineering)

**Version**: 1.0.0 | **Ratified**: 2025-01-07 | **Last Amended**: 2025-01-07
