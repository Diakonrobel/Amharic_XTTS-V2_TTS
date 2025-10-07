# System Architecture

## Overview
The XTTS Fine-tuning WebUI is built as a Python application with Gradio frontend, PyTorch backend, and modular utilities. The Amharic enhancement adds a new layer of language-specific processing while maintaining the existing architecture.

## Source Code Structure

```
xtts-finetune-webui-fresh/
├── xtts_demo.py              # Main Gradio web application (3 tabs)
├── headlessXttsTrain.py      # CLI training interface
├── utils/                     # Core processing utilities
│   ├── formatter.py          # Audio processing & dataset creation
│   ├── gpt_train.py          # XTTS training logic
│   └── tokenizer.py          # Multilingual text processing
├── amharic_tts/              # NEW: Amharic-specific modules
│   ├── g2p/                  # G2P conversion
│   ├── tokenizer/            # Amharic tokenizer extensions
│   ├── preprocessing/        # Text normalization
│   └── config/               # Amharic configurations
├── enhancement_ideas/         # Design specifications
│   └── Guidance.md           # Amharic TTS specification
├── .warp/                    # Project meta-information
│   ├── rules/                # Memory bank and constitution
│   └── workflows/            # Development workflows
└── WARP.md                   # Project documentation
```

## Component Architecture

### Existing Components

#### 1. Web Interface (`xtts_demo.py`)
**Purpose**: Gradio-based UI for TTS fine-tuning workflow

**Key Functions**:
- `preprocess_dataset()`: Tab 1 - Creates training dataset from audio
- `train_model()`: Tab 2 - Fine-tunes XTTS model
- `optimize_model()`: Tab 2.5 - Reduces model size
- `load_model()` + `run_tts()`: Tab 3 - Inference

**Flow**:
```
Audio Files → Whisper Transcription → Dataset CSV → 
  Training → Best Model → Optimization → Ready Model → 
  Inference
```

#### 2. Audio Processing (`utils/formatter.py`)
**Purpose**: Convert raw audio to training-ready datasets

**Key Functions**:
- `format_audio_list()`: Main dataset creation pipeline
  - Loads audio with torchaudio
  - Transcribes with Whisper ASR (supports VAD filtering)
  - Segments at sentence boundaries
  - Applies `multilingual_cleaners()` to text
  - Creates metadata CSVs (train/eval split)

**Integration Point for Amharic**: 
- `multilingual_cleaners()` call (line 139) - Add Amharic text processing here
- Language parameter passed to Whisper and cleaners

#### 3. Training (`utils/gpt_train.py`)
**Purpose**: Fine-tune XTTS model on custom dataset

**Key Features**:
- Downloads base XTTS model if needed
- Configures GPTTrainerConfig with audio/training params
- Handles speaker embedding generation
- Manages checkpointing and optimization

**Amharic Integration**: Minimal changes needed - already language-agnostic

#### 4. Tokenizer (`utils/tokenizer.py`)
**Purpose**: Text → Token IDs for model input

**Existing Support**: 16 languages (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, hu, ko, ja)

**Key Components**:
- `VoiceBpeTokenizer`: Base BPE tokenizer
- `multilingual_cleaners()`: Text normalization per language
- `expand_numbers_multilingual()`: Number → words
- Language-specific abbreviation/symbol expansion

**Amharic Integration**: Add 'amh' to all language dictionaries

### New Components (Amharic Module)

#### 1. G2P Converter (`amharic_tts/g2p/amharic_g2p.py`)
**Purpose**: Ethiopic script → IPA phonemes

**Architecture**:
```python
AmharicG2P
├── Backends
│   ├── Transphone (primary): read_g2p('amh')
│   ├── Epitran (fallback): Epitran('amh-Ethi')
│   └── Custom Rules (offline): Rule-based mapping
├── Phonological Rules
│   ├── Epenthesis: C_C → CɨC insertion
│   ├── Gemination: gem marker → CC doubling
│   └── Labiovelars: Special consonants (ቋ→qʷa)
└── Text Preprocessing
    ├── Normalization: Variant forms (ሥ→ስ)
    └── Punctuation: Ethiopic → Standard
```

**Usage**:
```python
g2p = AmharicG2P(use_transphone=True)
phonemes = g2p.convert("ሰላም ዓለም")  # → "səlam ʕaləm"
```

#### 2. Amharic Tokenizer (`amharic_tts/tokenizer/amharic_tokenizer.py`)
**Purpose**: Extend VoiceBpeTokenizer for Ethiopic script

**Features**:
- Ethiopic Unicode range support (U+1200 - U+137F)
- Phoneme vocabulary (IPA symbols)
- Sub-word BPE for common syllables
- Special tokens for Amharic

**Integration**: Used alongside existing tokenizer

#### 3. Text Preprocessor (`amharic_tts/preprocessing/`)
**Purpose**: Normalize Amharic text before G2P

**Components**:
- `text_normalizer.py`: Clean and standardize text
  - Variant character normalization
  - Whitespace handling
  - Punctuation conversion
- `number_expander.py`: Numbers → Amharic words
  - አንድ (1), ሁለት (2), ሶስት (3), etc.

#### 4. Configuration (`amharic_tts/config/`)
**Purpose**: Amharic-specific settings and mappings

**Files**:
- `amharic_config.yaml`: Model and processing config
- `phoneme_mapping.json`: Grapheme → Phoneme rules

## Data Flow

### Training Flow (with Amharic)
```
1. Audio Input (.wav/.mp3/.flac)
   ↓
2. Whisper Transcription (supports Amharic)
   ↓ (Ethiopic script text)
3. Text Preprocessing
   ├─ Normalize variants (ሥ→ስ)
   ├─ Expand numbers (123→አንድ መቶ ሃያ ሶስት)
   └─ Handle punctuation (።→.)
   ↓
4. G2P Conversion (NEW for Amharic)
   └─ Ethiopic → IPA phonemes
   ↓
5. Tokenization
   ├─ Ethiopic character vocab (NEW)
   ├─ Phoneme vocab (NEW)
   └─ BPE encoding
   ↓
6. Dataset Creation
   └─ metadata_train.csv, metadata_eval.csv
   ↓
7. XTTS Fine-tuning
   └─ Transfer learning from v2.0.3 base
   ↓
8. Model Optimization
   └─ Remove optimizer states, DVAE weights
   ↓
9. Ready Model
   └─ ready/model.pth, config.json, vocab.json
```

### Inference Flow (with Amharic)
```
1. Amharic Text Input (Ethiopic script)
   ↓
2. Text Preprocessing (normalize, expand numbers)
   ↓
3. G2P Conversion → IPA phonemes
   ↓
4. Tokenization → Token IDs
   ↓
5. XTTS Model → Mel Spectrogram
   ↓
6. Vocoder → Audio Waveform
   ↓
7. Output .wav file
```

## Integration Points

### Where Amharic Code Connects

1. **Language Selection UI**:
   - `xtts_demo.py` line 257-278: Add "amh" to `lang` dropdown
   - `headlessXttsTrain.py` line 686: Add "amh" to choices

2. **Text Cleaning**:
   - `utils/formatter.py` line 139: `multilingual_cleaners(sentence, target_language)`
   - `utils/tokenizer.py` line 577: `multilingual_cleaners()` function
   - Add Amharic branch: `if lang == "amh": return amharic_clean(text)`

3. **Tokenizer**:
   - `utils/tokenizer.py`: Add Amharic to language dictionaries
   - Import and use `AmharicXTTSTokenizer` when lang="amh"

4. **Number Expansion**:
   - `utils/tokenizer.py` line 548: `expand_numbers_multilingual()`
   - Add Amharic case with custom number→word logic

5. **Dataset Creation**:
   - `utils/formatter.py`: Pass Amharic text through G2P before storage
   - Store both original text and phonemes in metadata

## Critical Implementation Paths

### Path 1: Basic Amharic Support (Minimal)
1. Add "amh" to language dropdowns
2. Extend `multilingual_cleaners()` with Amharic text normalization
3. Add Amharic number expansion to `expand_numbers_multilingual()`
4. Test with Amharic audio → should work with basic phoneme matching

### Path 2: G2P Integration (Enhanced)
1. Create `amharic_tts/g2p/` module
2. Integrate G2P in `formatter.py` during dataset creation
3. Store phonemes in metadata CSV
4. Modify tokenizer to use phoneme vocabulary for Amharic

### Path 3: Full Amharic Pipeline (Complete)
1. Implement all Amharic modules (G2P, tokenizer, preprocessing)
2. Create Amharic test dataset
3. Train and validate Amharic model
4. Document usage and examples
5. Update WARP.md with Amharic instructions

## Design Patterns

### 1. Backend Strategy Pattern
- G2P converter supports multiple backends (transphone, epitran, custom)
- Graceful degradation: primary → fallback → offline

### 2. Language Plugin Architecture
- Amharic code isolated in `amharic_tts/` module
- Other Ethiopian languages (Tigrinya, Oromo) can follow same pattern
- Minimal changes to core codebase

### 3. Configuration-Driven Processing
- Language-specific rules in JSON/YAML files
- Easy to tune without code changes
- Phoneme mappings externalized

## Component Relationships

```
xtts_demo.py
    ├─ calls → utils/formatter.py
    │            ├─ uses → WhisperModel (transcription)
    │            └─ calls → multilingual_cleaners()
    │                         └─ NEW: amharic_tts.preprocessing
    ├─ calls → utils/gpt_train.py
    │            └─ uses → TTS.tts.layers.xtts
    └─ calls → load_model() + run_tts()
                 └─ uses → Xtts model

headlessXttsTrain.py
    ├─ calls → prepare_audio() (FFmpeg integration)
    ├─ calls → preprocess_dataset_headless()
    │            └─ same as above
    └─ calls → train_model_headless()

amharic_tts/ (NEW)
    ├─ g2p/amharic_g2p.py
    │    └─ imports → transphone, epitran
    ├─ tokenizer/amharic_tokenizer.py
    │    └─ extends → VoiceBpeTokenizer
    └─ preprocessing/
         └─ used by → multilingual_cleaners()
```

## Key Technical Decisions

1. **G2P Timing**: Run G2P during dataset creation (not inference)
   - Rationale: Pre-compute phonemes, store in CSV
   - Benefit: Faster training, consistent phoneme representations

2. **Tokenizer Strategy**: Dual-mode tokenizer (text OR phonemes)
   - Rationale: Support both Ethiopic script and IPA input
   - Benefit: Flexibility for research and production

3. **Module Isolation**: Separate `amharic_tts/` directory
   - Rationale: Clean separation, easy to maintain
   - Benefit: Can be packaged independently

4. **Backward Compatibility**: No breaking changes to existing languages
   - Rationale: Preserve existing functionality
   - Benefit: Amharic is additive, not disruptive

## Performance Considerations

- G2P conversion adds ~50-100ms per sentence
- Ethiopic character encoding requires UTF-8 support
- Phoneme vocabulary increases tokenizer size slightly
- Training time should be similar to other languages
- Inference speed unaffected (phonemes pre-computed)
