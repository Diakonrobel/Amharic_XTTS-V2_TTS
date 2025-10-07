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

### ✅ Completed Tasks
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
       └── __init__.py
   ```

3. **Core Integration**:
   - ✅ Extended `utils/tokenizer.py` with Amharic support
   - ✅ Added Amharic number expansion to multilingual_cleaners
   - ✅ Integrated text normalization pipeline
   - ✅ Added "amh" to `xtts_demo.py` language dropdowns
   - ✅ Added "amh" to `headlessXttsTrain.py` CLI choices

### 🔄 Current Focus
1. **Amharic-Specific Tokenizer**:
   - Create `amharic_tts/tokenizer/amharic_tokenizer.py`
   - Support Ethiopic script (Unicode U+1200–U+137F)
   - Handle Amharic phoneme mappings

2. **Configuration Files**:
   - Create `amharic_tts/config/amharic_config.py`
   - Define Amharic phoneme inventory
   - Set character limits and special tokens

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
