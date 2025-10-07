# Project Brief

## Project Name
XTTS Fine-Tuning WebUI (Fresh Fork with Amharic Support)

## Purpose
A user-friendly web interface and CLI tool for fine-tuning Coqui XTTS v2 text-to-speech models on custom voice datasets. Enables anyone to create high-quality, personalized TTS models without deep ML expertise.

## Core Goals
1. **Accessibility:** Make XTTS fine-tuning approachable through intuitive Gradio web UI
2. **Multilingual:** Support 20+ languages with special focus on Amharic (Ethiopian) TTS
3. **Flexibility:** Provide both interactive (web UI) and automated (headless CLI) workflows
4. **Quality:** Ensure production-ready model output with optimization and deployment features
5. **Robustness:** Handle edge cases gracefully with fallback mechanisms

## Key Differentiators
- **Comprehensive Amharic Support:** Advanced G2P (Grapheme-to-Phoneme) system with multi-backend architecture
- **Incremental Dataset Updates:** Skip previously processed files, add new data seamlessly
- **Headless Training:** Full CLI automation for server/batch deployments
- **Model Chaining:** Fine-tune already fine-tuned models
- **Deployment Ready:** One-click model optimization to ready-to-use format

## Primary Users
- Voice actors and content creators wanting custom TTS voices
- Language researchers working on under-resourced languages (especially Ethiopic scripts)
- Developers building TTS applications
- ML practitioners experimenting with voice cloning

## Success Criteria
- Users can fine-tune a working TTS model from raw audio in < 2 hours
- Amharic G2P system works without manual phoneme annotation
- Models produce natural-sounding speech matching speaker's voice characteristics
- Training process is stable across different GPU configurations (6-24GB VRAM)
- Both web UI and headless modes produce identical results

## Technical Foundation
- **Base:** Coqui TTS XTTS v2 (multi-lingual, cross-lingual TTS)
- **UI Framework:** Gradio 4.x
- **Audio Processing:** Faster Whisper (automatic transcription with VAD)
- **Training:** PyTorch with CUDA acceleration
- **Languages:** Python 3.10+, supports Windows/Linux/macOS

## Constraints
- CUDA-capable GPU required for training (CPU inference only)
- Minimum 6GB VRAM for small batch sizes
- Audio input limited to ~40 minutes (auto-trimmed to prevent OOM)
- Tied to XTTS v2 architecture (breaking changes if upstream updates)
