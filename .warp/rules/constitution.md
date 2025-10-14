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

#### Amharic Language Support (First-Class Requirement)
**Amharic must work seamlessly across all pipeline stages:**

**Dataset Creation:**
- Ethiopic script (U+1200-U+137F) properly handled in transcriptions
- UTF-8 encoding enforced throughout
- Character normalization applied (ሥ→ስ, ዕ→እ, ሀ→ሃ, ዓ→አ)
- Language code `amh` (ISO 639-3) consistently used

**Text Preprocessing:**
- Multi-backend G2P system (Transphone → Epitran → Rule-based)
- Quality validation on G2P output (vowel ratio, IPA presence)
- Graceful fallback to rule-based if external backends unavailable
- Text normalization before G2P (character variants, punctuation)

**Training:**
- Vocabulary extension with Ethiopic characters + IPA phonemes (~7500 vs 6152 standard)
- On-the-fly or pre-processed G2P conversion
- Dataset detection (Ethiopic script vs IPA phonemes)
- Language code switching (amh → en after G2P conversion)
- No "UNK token" errors during training

**Inference:**
- Automatic Amharic detection from input text
- G2P preprocessing applied transparently
- Vocabulary size mismatch handling (extended vocab models)
- Language code normalization (am/AM/Amharic → amh)

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

## Amharic Development Guidelines

### Language Code Consistency
**CRITICAL**: Use ISO 639-3 code `amh` throughout pipeline:
- **Dataset creation**: `lang.txt` contains `amh`
- **Training config**: `BaseDatasetConfig(language="amh")` or `"en"` (if phonemes)
- **Inference**: User input `am`/`AM`/`Amharic` → normalized to `amh`
- **WebUI**: Display "Amharic (amh)" in dropdowns

**Normalization Function**: `utils/lang_norm.py::canonical_lang()`
- Accepts: `am`, `amh`, `am-ET`, `AM`, `Amharic`, `አማርኛ`
- Returns: `amh` (for Coqui/training/dataset)

### G2P Pipeline Requirements

**Backend Selection Priority**:
1. **Transphone** (primary): ML-based, 95%+ accuracy, requires `pip install transphone`
2. **Epitran** (secondary): Rule-based, 85-90% accuracy, requires `pip install epitran`
3. **Rule-based** (fallback): Always available, 80-85% accuracy, no dependencies

**Implementation**: `amharic_tts/g2p/amharic_g2p_enhanced.py::EnhancedAmharicG2P`

**Quality Validation Thresholds**:
- Minimum vowel ratio: 25% (Amharic is vowel-rich)
- Maximum Ethiopic char ratio in output: 10% (should be converted)
- Minimum IPA char presence: 50% (output should use IPA)
- Minimum length ratio: 50% (output shouldn't collapse)

**Usage in Pipeline**:
```python
# Option A: Pre-training preprocessing
python preprocess_amharic_dataset.py \
    --input-train metadata_train.csv \
    --input-eval metadata_eval.csv \
    --g2p-backend transphone

# Option B: On-the-fly during training
train_gpt(..., use_amharic_g2p=True, ...)
# Automatically applies G2P in amharic_g2p_dataset_wrapper.py
```

### Vocabulary Extension Requirements

**When to Extend**:
- **ALWAYS** when training with Amharic (either script or phonemes)
- Extends from 6152 tokens → ~7500 tokens (+22%)

**What Gets Added**:
1. **Ethiopic characters**: 384 chars (U+1200-U+137F, U+1380-U+139F)
2. **Amharic IPA phonemes**: 45 phonemes (tʼ, kʼ, pʼ, kʷ, gʷ, qʷ, ʕ, ʔ, ɨ, ə)
3. **Common subword units**: 77 units (səl, lam, amɨ, kʼɨ, etc.)
4. **Dataset-specific tokens**: Top 500 frequent n-grams (min_freq=5)

**Implementation**: `utils/vocab_extension.py::create_extended_vocab_for_training()`

**File Location**: `output/ready/vocab_extended_amharic.json`

### Character Normalization Rules

**Applied in**: `amharic_tts/preprocessing/text_normalizer.py::AmharicTextNormalizer`

**Normalization Table**:
| Input | Output | Reason |
|-------|--------|--------|
| ሥ | ስ | Variant of sin |
| ዕ | እ | Variant of glottal |
| ህ | ሕ | Variant of he |
| ዝ | ክ | Variant of ke |
| ሐ | ሃ | Variant of ha |
| ዓ | አ | Variant of a |
| ኣ | አ | Tigrinya variant |
| ተ | ፕ | Rare variant |

**Punctuation Handling**:
- Ethiopic word space (፡) → space
- Ethiopic punctuation (።፣፤፥፦፧) preserved by default
- Optional conversion to ASCII punctuation (disabled)

### Dataset Detection Logic

**Function**: `utils/amharic_g2p_dataset_wrapper.py::is_dataset_already_preprocessed()`

**Algorithm**:
1. Sample first 10 rows of CSV
2. Check text column for Ethiopic script (U+1200-U+137F)
3. Calculate Ethiopic character ratio
4. If ratio > 50% → dataset needs preprocessing
5. If ratio < 50% → dataset already preprocessed (phonemes)

**Impact on Training**:
- Ethiopic script detected → apply G2P, use `language="amh"`
- Phonemes detected → skip G2P, use `language="en"`

### Training Pipeline Integration Points

**`gpt_train.py` Flow**:
```python
1. Normalize language code: am/AM/Amharic → amh
2. Check if G2P enabled: use_amharic_g2p=True
3. Detect dataset state: Ethiopic script vs phonemes
4. If Ethiopic:
   a. Extend vocabulary (Ethiopic + IPA tokens)
   b. Load training samples
   c. Apply G2P on-the-fly (amharic_g2p_dataset_wrapper)
   d. Switch language code: amh → en
5. If phonemes:
   a. Extend vocabulary (IPA tokens only)
   b. Load samples as-is
   c. Use language code: en
6. Initialize GPTTrainer with extended vocab
7. Train (no UNK token errors)
```

### Inference Pipeline Requirements

**`xtts_demo.py::run_tts()` Flow**:
```python
1. Detect language: is_amharic(text) → check for Ethiopic chars
2. Normalize code: am/AM → amh
3. Load model:
   a. Check vocab size: checkpoint vs vocab.json
   b. If mismatch: find matching vocab or expand embeddings
4. Preprocess text:
   a. If Amharic detected: apply G2P conversion
   b. Text: ሰላም ዓልም → salam ʕaləm
5. Generate speech:
   a. Use phoneme text as input
   b. Language code: en (phonemes are Latin)
6. Return audio
```

### Model Vocabulary Mismatch Handling

**Problem**: Extended vocab training creates 7536-token models, but base vocab has 6152 tokens

**Solution in `xtts_demo.py::load_model()`**:
1. Load checkpoint, check `gpt.text_embedding.weight.shape[0]`
2. Load vocab.json, check `len(vocab['model']['vocab'])`
3. If mismatch:
   - Search `ready/` directory for matching vocab file
   - If found: use that vocab
   - If not found: manually expand embedding layers
     - Copy existing weights [0:checkpoint_size]
     - Random init new weights [checkpoint_size:vocab_size]
4. Load model with correct vocab

### Testing Requirements for Amharic Changes

**Minimum Test Coverage**:
1. ✅ Text normalization: `tests/test_language_normalization_fix.py`
2. ✅ G2P conversion (all backends): `tests/test_amharic_g2p_comprehensive.py`
3. ✅ Quality validation: Check thresholds in G2P output
4. ✅ Vocabulary extension: Verify ~7500 tokens created
5. ✅ Dataset detection: Ethiopic vs phonemes
6. ✅ Training integration: No UNK errors
7. ✅ Inference: Amharic input → audio output

**Manual Testing Checklist**:
- [ ] Upload Amharic audio → transcription in Ethiopic script
- [ ] Enable "Amharic G2P" in training → training completes
- [ ] Check `ready/vocab_extended_amharic.json` exists (~7500 tokens)
- [ ] Load trained model → no vocab size errors
- [ ] Input Amharic text → generates speech
- [ ] Check logs: language code consistency (amh or en)

### Common Pitfalls & Solutions

**Pitfall 1**: Training fails with "AssertionError: assert not torch.any(tokens == 1)"
- **Cause**: Tokenizer encounters unknown Ethiopic characters
- **Solution**: Enable Amharic G2P (`use_amharic_g2p=True`) or pre-preprocess dataset

**Pitfall 2**: Vocab size mismatch error during inference
- **Cause**: Model trained with extended vocab (7536), loading with standard vocab (6152)
- **Solution**: Use `vocab_extended_amharic.json` or enable auto-expansion in `load_model()`

**Pitfall 3**: Language code inconsistency (am vs amh vs en)
- **Cause**: Different parts of pipeline use different codes
- **Solution**: Always use `canonical_lang()` from `utils/lang_norm.py`

**Pitfall 4**: Poor G2P quality (gibberish output)
- **Cause**: Backend failing quality checks
- **Solution**: Check G2P backend installation, try different backend, verify input text normalization

**Pitfall 5**: Dataset detected as "already preprocessed" when it's not
- **Cause**: Mixed Ethiopic + Latin text in dataset (e.g., names, numbers)
- **Solution**: Increase Ethiopic ratio threshold or manually preprocess

### Documentation Requirements

**When Adding Amharic Features**:
1. Update `README.md` with user-facing changes
2. Update `WARP.md` with technical details
3. Document in `amharic_tts/` module docstrings
4. Add examples to relevant test files
5. Update `.warp/rules/constitution.md` if architectural changes

**When Modifying G2P Pipeline**:
1. Update `docs/G2P_BACKENDS_EXPLAINED.md`
2. Document quality threshold changes
3. Add test cases for edge cases
4. Update `amharic_tts/config/amharic_config.py` if config changes

---

**Version**: 2.0.0 | **Ratified**: 2025-01-13 | **Last Amended**: 2025-01-13 (Amharic Guidelines Added)
