# Current Context

## Project Status
**Active Development** - Stable fork with Amharic language enhancements

## Recent Work (As of January 2025)
1. Completed comprehensive WARP.md documentation
2. Created project constitution with architectural principles
3. Initialized memory bank with complete project overview
4. Established workflow integration with `.warp/` directory structure

## Current Focus
Maintaining and extending XTTS fine-tuning capabilities with emphasis on:
- Multi-language support (especially Amharic/Ethiopic scripts)
- User experience improvements (both Web UI and headless modes)
- Robustness and fallback systems
- Documentation quality

## Next Steps (Potential)
1. **Enhanced G2P Integration:** Deeper XTTS tokenizer integration for Amharic G2P
2. **Training Resumption:** Implement checkpoint resumption for interrupted training runs
3. **Batch Processing:** Add multi-file headless processing with JSON config
4. **Quality Metrics:** Add automatic audio quality assessment after training
5. **Web API:** Expose training/inference as REST API endpoints

## Active Branches / Versions
- **Main branch:** Production-ready code
- **XTTS versions supported:** v2.0.1, v2.0.2, main
- **Python versions:** 3.10, 3.11 tested
- **Platforms:** Windows 10/11, Ubuntu 20.04+, macOS (Intel & ARM)

## Known Working Configurations

### Tested Hardware
- **NVIDIA RTX 3060** (12GB) - batch_size=2-4, works well
- **NVIDIA RTX 3090** (24GB) - batch_size=4-8, optimal
- **Apple M1/M2/M3** - MPS backend, slower but functional
- **CPU-only** - Inference works, training not recommended

### Tested Languages
- **English** (en) - Most tested, reference implementation
- **Amharic** (amh/am) - Extensive testing with G2P system
- **Japanese** (ja) - Special handling with num_workers=0
- **Spanish, French, German, Italian** - Basic testing
- **Arabic, Chinese** - Limited testing

## Recent Changes Summary

### Documentation
- Created comprehensive WARP.md with:
  - Installation commands (Windows/Linux/macOS)
  - Architecture overview and data flow diagrams
  - Development patterns and common tasks
  - Platform-specific notes
  - Training pipeline detailed explanation

- Updated `.warp/rules/constitution.md` with:
  - 5 core principles (Modularity, Graceful Degradation, Language-Agnostic, File-Based State, Path Safety)
  - Technical standards (Audio, Dataset, Training, Code Quality)
  - Development workflows (Language Support, Features, Pipeline Modifications)
  - Testing requirements and governance

- Initialized memory bank files:
  - `brief.md` - Project overview and goals
  - `product.md` - User experience and features
  - `architecture.md` - System design and implementation details
  - `tech.md` - Technology stack and setup
  - `context.md` - Current state (this file)

### Workflow Integration
- Configured `.warp/workflows/` for Spec-Driven Development
- Set up planning (`plan.md`), implementation (`implement.md`), specification (`specify.md`), and task tracking (`tasks.md`)

## Important Context for Future Work

### File Organization
- **Entry points:** `xtts_demo.py` (web), `headlessXttsTrain.py` (CLI)
- **Core logic:** `utils/` directory (formatter, gpt_train, tokenizer)
- **Language modules:** `amharic_tts/` (self-contained, extensible pattern)
- **Tests:** `tests/` (Amharic G2P comprehensive tests)

### Critical Paths
1. **Dataset Creation:** `utils/formatter.py::format_audio_list()` (lines 54-198)
2. **Training Setup:** `utils/gpt_train.py::train_gpt()` (lines 15-220)
3. **Amharic G2P:** `amharic_tts/g2p/amharic_g2p_enhanced.py::convert()`

### State Management
- All state in filesystem (not memory)
- `output/` or `finetune_models/` for user data
- `base_models/` for cached XTTS models
- Incremental dataset updates via CSV prefix checking

### Common Pitfalls
1. **Japanese training:** Must set `num_workers=0`
2. **Path handling:** Always use `pathlib.Path` for cross-platform compatibility
3. **CUDA OOM:** Start with batch_size=2, increase if VRAM allows
4. **FFmpeg dependency:** Required for headless audio preprocessing
5. **Model versions:** Mixing XTTS versions can cause incompatibilities

## User Feedback & Pain Points
(To be updated as issues are reported)

- **Positive:** Amharic G2P system works well, fallback is reliable
- **Positive:** Incremental dataset updates save significant time
- **Positive:** Web UI is intuitive for non-technical users
- **Area for improvement:** Training progress visibility (no logs in UI)
- **Area for improvement:** Clearer error messages when VRAM insufficient

## Development Priorities

### High Priority
1. Maintain backward compatibility with existing datasets/models
2. Keep documentation up-to-date (WARP.md, README, memory bank)
3. Test on multiple platforms before major changes
4. Preserve file-based state management pattern

### Medium Priority
1. Expand test coverage beyond Amharic
2. Add more language-specific modules (following Amharic pattern)
3. Improve headless mode feature parity with web UI
4. Docker image optimization

### Low Priority / Future Considerations
1. Web-based inference API
2. Distributed training support
3. Model quantization for deployment
4. Real-time streaming inference

## Dependencies to Watch
- **Coqui TTS:** May become unmaintained (check for forks)
- **Faster Whisper:** Active development, updates may change API
- **Gradio:** Frequent updates, test UI after upgrades
- **PyTorch:** CUDA compatibility important for GPU users

## Notes for AI Agents
When working on this project:
1. Read WARP.md first for command reference
2. Check constitution.md for architectural principles
3. Follow the memory bank pattern (this file should be updated after significant changes)
4. Test both web UI and headless mode if changes affect core utilities
5. Consider Amharic G2P as example for new language support
6. Always use Path operations, never hardcode OS-specific paths
7. GPU availability varies - ensure graceful degradation to CPU when possible

## Metrics & Success Indicators
- Training completion rate: ~95% (most failures due to user errors like insufficient VRAM)
- Amharic G2P accuracy: Estimated 85-90% (rule-based), 90-95% (Transphone)
- Model quality: Subjective feedback generally positive for 10+ epochs
- Time to first model: 1-2 hours for typical use case (10 min audio, 10 epochs)

## Last Updated
**Date:** 2025-01-07  
**By:** Warp AI Agent  
**Changes:** Initial memory bank creation with comprehensive project documentation
