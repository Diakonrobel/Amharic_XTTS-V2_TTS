# Current Context

## Project Status
**Phase**: Active Implementation - Amharic TTS Core Components  
**Last Updated**: 2025-10-07  
**Current Focus**: Testing and Integration

## Recent Changes
1. ✅ Created complete `.warp/` directory structure with memory bank
2. ✅ Implemented full Amharic G2P module (`amharic_tts/g2p/amharic_g2p.py`)
3. ✅ Created text preprocessing modules (normalizer, number expander)
4. ✅ Extended `utils/tokenizer.py` with Amharic support
5. ✅ Added "amh" language to UI dropdowns (web and headless)
6. ✅ Tested number expansion - works correctly for Amharic numbers

## What We're Working On Now

### ✅ Completed Tasks (Phase 1 Complete)
1. **Memory Bank Setup**: All memory bank files created
   - ✅ brief.md - Project mission and goals
   - ✅ product.md - Product vision and user experience
   - ✅ context.md - Current state (this file)
   - ✅ architecture.md - System architecture documentation
   - ✅ tech.md - Technology stack and setup

2. **Amharic Module Structure**: Created complete directory structure
   ```
   amharic_tts/
   ├── __init__.py
   ├── g2p/
   │   ├── __init__.py
   │   └── amharic_g2p.py      ✅ Multi-backend G2P with phonological rules
   ├── tokenizer/
   │   └── __init__.py
   ├── preprocessing/
   │   ├── __init__.py
   │   ├── text_normalizer.py  ✅ Character normalization & cleaning
   │   └── number_expander.py  ✅ Amharic number word expansion (tested)
   └── config/
       ├── __init__.py
       └── amharic_config.py   ✅ Complete configuration module
   ```

3. **Core Integration**:
   - ✅ Extended `utils/tokenizer.py` with Amharic support
   - ✅ Added Amharic number expansion to multilingual_cleaners
   - ✅ Integrated text normalization pipeline
   - ✅ Added "amh" to `xtts_demo.py` language dropdowns
   - ✅ Added "amh" to `headlessXttsTrain.py` CLI choices

4. **GitHub Integration**:
   - ✅ Git LFS configured for training files
   - ✅ Google Colab notebook with LFS persistence
   - ✅ Helper scripts for save/load operations
   - ✅ Comprehensive LFS workflow documentation

5. **Advanced Configuration** (NEW):
   - ✅ Created `amharic_tts/config/amharic_config.py`
   - ✅ G2P backend ordering (Transphone → Epitran → Rule-based)
   - ✅ Quality thresholds for G2P output validation
   - ✅ Tokenizer modes (Raw BPE vs Hybrid G2P+BPE)
   - ✅ Preset configurations (default, fast, quality, research)
   - ✅ Complete Amharic phoneme inventory

### 🔄 Current Focus (Phase 2: Enhanced G2P)
1. **Enhanced G2P with Quality Heuristics**:
   - Upgrading `amharic_g2p.py` with intelligent backend selection
   - Adding quality validation for G2P outputs
   - Implementing fallback logic based on quality scores
   - Supporting vowel ratio, Ethiopic ratio, IPA ratio checks

2. **Next: Enrich Rule-Based G2P** (Phase 3):
   - Expand grapheme-to-phoneme tables for all Ethiopic orders (1-7)
   - Add complete consonant mappings (231 total)
   - Improve epenthesis and gemination rules

3. **Next: Hybrid G2P+BPE Tokenizer** (Phase 4):
   - Create `amharic_tts/tokenizer/hybrid_tokenizer.py`
   - Phoneme-aware BPE training
   - Better linguistic representation for Amharic

### Next Immediate Steps
1. Implement Amharic tokenizer extension
2. Create configuration files with phoneme mappings
3. Write comprehensive unit tests:
   - G2P conversion with diverse samples
   - Text normalization edge cases
   - Number expansion boundary conditions
4. Integration testing:
   - Dataset creation with Amharic audio
   - Training run validation
   - Inference quality check
5. Documentation:
   - Add Amharic usage guide to README
   - Document G2P backend selection
   - Provide sample datasets and commands

## Key Decisions Made
1. **Modular Approach**: Create separate amharic_tts/ module rather than inline modifications
2. **Backward Compatible**: All changes must preserve existing 16-language support
3. **G2P Strategy**: Use transphone as primary, epitran as fallback, custom rules as last resort
4. **Phased Rollout**: 
   - Phase 1: Core Amharic support (G2P, tokenizer, text processing)
   - Phase 2: Training pipeline integration
   - Phase 3: UI enhancements
   - Phase 4 (Future): API, streaming, advanced features

## Current Blockers
None - greenfield development phase

## Questions to Resolve
1. Should we bundle transphone/epitran dependencies or make them optional?
2. What's the minimum Amharic dataset size to validate the implementation?
3. Should G2P conversion happen during data processing or training?
4. How to handle mixed Amharic-English text (code-switching)?

## Recent Discoveries
- ✅ Existing `tokenizer.py` already has extensive multilingual support structure - successfully extended
- ✅ `formatter.py` uses `multilingual_cleaners()` function - integrated Amharic normalization here
- ✅ Training pipeline in `gpt_train.py` is language-agnostic - no changes needed
- ✅ Whisper ASR supports Amharic - dataset creation should work seamlessly
- ✅ Amharic number expansion works correctly - tested with values 0-999999
- ✅ Multi-backend G2P strategy provides robust fallback chain
- ✅ Ethiopic script normalization handles variant forms (e.g., ዓ vs አ)

## Active Experiments
- Testing G2P backends (transphone vs epitran vs rule-based) for quality comparison
- Evaluating optimal character limits for Amharic text inputs
- Exploring epenthesis rule application timing (preprocessing vs runtime)

## Important Notes
- The `enhancement_ideas/Guidance.md` file provides comprehensive Amharic TTS specifications
- All Amharic additions maintain backward compatibility with existing 16 languages
- Focus on maintaining simplicity while adding sophisticated features
- All Amharic-specific phonological rules are documented in `amharic_g2p.py`
- Next critical path: Amharic tokenizer + config files + comprehensive testing
- Test with real Amharic data early and often before production deployment
