# Architecture

## System Overview

```
┌─────────────────────────────────────────────────┐
│           User Interfaces                        │
├─────────────────────────────────────────────────┤
│  xtts_demo.py (Gradio)  │  headlessXttsTrain.py │
│  - 3 tabs (Web UI)      │  - CLI automation     │
└───────────┬──────────────┴─────────┬─────────────┘
            │                        │
            ├────────────────────────┘
            ▼
┌─────────────────────────────────────────────────┐
│              Core Utilities                      │
├─────────────────────────────────────────────────┤
│  utils/formatter.py   │ Audio → Dataset         │
│  utils/gpt_train.py   │ Training orchestration  │
│  utils/tokenizer.py   │ Language processing     │
└───────────┬──────────────────────────┬───────────┘
            │                          │
            ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐
│  External Libraries  │    │  Language Modules   │
├─────────────────────┤    ├─────────────────────┤
│ • Faster Whisper    │    │  amharic_tts/       │
│ • TTS (Coqui)       │    │  ├─ g2p/            │
│ • PyTorch           │    │  ├─ tokenizer/      │
│ • Gradio            │    │  ├─ preprocessing/  │
│ • torchaudio        │    │  ├─ config/         │
│ • librosa           │    │  └─ utils/          │
└─────────────────────┘    └─────────────────────┘
```

## Source Code Structure

### Entry Points

**`xtts_demo.py`** (Lines: ~1500)
- Main Gradio application
- Defines 3-tab interface layout
- Orchestrates workflow: Audio → Dataset → Training → Inference
- Key functions:
  - `format_audio_list()` → calls `utils/formatter.py`
  - `train_gpt()` → calls `utils/gpt_train.py`
  - `load_model()`, `run_tts()` → inference logic
  - `load_params_tts()` → loads model from `ready/`

**`headlessXttsTrain.py`** (Lines: ~800)
- CLI-based automation
- Parallels web UI workflow without Gradio
- FFmpeg integration for audio preprocessing
- Argument parsing via argparse
- Key functions:
  - `prepare_audio()` → conversion + trimming
  - `run_ffmpeg()` → subprocess FFmpeg execution
  - `get_audio_duration()` → torchaudio.info + fallback

**`smart_install.py`** (Lines: ~200)
- Platform-specific dependency installation
- CUDA detection and PyTorch version selection
- Used by `install.bat` / `install.sh`

### Core Utilities (`utils/`)

**`formatter.py`** - Dataset Preparation (Lines: ~198)
- **Purpose:** Audio files → Training-ready CSV metadata
- **Key Functions:**
  - `format_audio_list()` (main)
    - Input: Audio files, Whisper model, language, output path
    - Process: Transcribe → Segment → Clean → Save
    - Output: metadata_train.csv, metadata_eval.csv
  - `list_audios()` → finds audio files recursively
  - `find_latest_best_model()` → locates best checkpoint
- **Dependencies:** 
  - Faster Whisper (transcription)
  - torchaudio (audio I/O)
  - pandas (CSV metadata)
  - TTS.tokenizer (multilingual_cleaners)
- **Critical Path:** 
  ```python
  for audio in files:
    wav, sr = torchaudio.load(audio)
    segments = whisper.transcribe(audio, vad_filter=True)
    for sentence in segment_by_punctuation(segments):
      save_segment(sentence)
      append_to_csv(metadata)
  ```

**`gpt_train.py`** - Training Orchestration (Lines: ~220)
- **Purpose:** Configure and execute XTTS fine-tuning
- **Key Function:** `train_gpt()`
  - Parameters: custom_model, version, language, epochs, batch_size, ...
  - Downloads base models (XTTS, DVAE, vocab, config)
  - Configures `GPTTrainer` with `GPTArgs` and `GPTTrainerConfig`
  - Loads train/eval samples from CSV
  - Executes training loop (managed by Trainer class)
  - Checkpoints saved to `{output_path}/run/training/GPT_XTTS_FT-*/`
- **Critical Paths:**
  - Base model download: `ModelManager._download_model_files()`
  - Amharic G2P: `if use_amharic_g2p and language == "am":`
  - Training: `trainer.fit()` (from TTS library)
- **Dependencies:**
  - TTS library (GPTTrainer, GPTArgs, load_tts_samples)
  - Trainer (from trainer package)
  - torch (PyTorch backend)

**`tokenizer.py`** - Language-Specific Text Processing (Lines: ~150)
- **Purpose:** Extend XTTS tokenizer with additional languages
- **Japanese Support:**
  - cutlet (Romaji conversion)
  - fugashi (morphological analysis)
  - Requires `num_workers=0` in training
- **Pattern:** Override `multilingual_cleaners()` for custom languages

### Amharic Language Module (`amharic_tts/`)

**Architecture Pattern:** Self-contained, zero hard dependencies on main codebase

**`g2p/amharic_g2p_enhanced.py`** - Multi-Backend G2P
- **Class:** `EnhancedAmharicG2P`
- **Backends:** Transphone → Epitran → Rule-based (fallback order)
- **Key Methods:**
  - `convert(text)` → tries each backend until success
  - `_validate_quality(output)` → ensures phoneme accuracy
  - Phonological rules: epenthesis, gemination
- **Fallback Logic:**
  ```python
  for backend in [Transphone, Epitran, RuleBased]:
    try:
      result = backend.convert(text)
      if validate_quality(result):
        return result
    except:
      continue  # try next backend
  ```

**`g2p/ethiopic_g2p_table.py`** - Character Mappings
- **Data:** `COMPLETE_G2P_TABLE` dict (259 entries)
- **Coverage:** 
  - 231 base consonant-vowel combos (33 consonants × 7 orders)
  - 20 labiovelar variants (qʷ, kʷ, gʷ, xʷ with 5 vowels)
  - 8 punctuation marks (።፣፤፥, etc.)
- **IPA Compliance:** All phonemes in standard IPA notation

**`preprocessing/text_normalizer.py`** - Character Normalization
- **Class:** `AmharicTextNormalizer`
- **Functions:**
  - Variant normalization (ሥ→ስ, ዕ→እ)
  - Unicode range validation (U+1200-U+137F)
  - Punctuation handling

**`preprocessing/number_expander.py`** - Number to Words
- **Class:** `AmharicNumberExpander`
- **Examples:**
  - 123 → አንድ መቶ ሃያ ሶስት
  - 2024 → ሁለት ሺህ ሃያ አራት

**`tokenizer/hybrid_tokenizer.py`** - G2P + BPE
- **Class:** `HybridAmharicTokenizer`
- **Strategy:** Convert to IPA phonemes, then apply BPE tokenization
- **Use Case:** Training with phoneme-based representations

**`tokenizer/xtts_tokenizer_wrapper.py`** - XTTS Integration
- **Function:** `create_xtts_tokenizer()`
- **Purpose:** Wrap hybrid tokenizer to match XTTS interface
- **Note:** Current integration is basic, can be deepened

**`config/amharic_config.py`** - Configuration
- **Classes:**
  - `G2PBackend` (enum): TRANSPHONE, EPITRAN, RULE_BASED
  - `G2PQualityThresholds`: min_vowel_ratio, max_ethiopic_ratio, etc.
  - `AmharicTTSConfig`: Backend order, phoneme inventory
- **Default Backend Order:**
  ```python
  [G2PBackend.TRANSPHONE, 
   G2PBackend.EPITRAN, 
   G2PBackend.RULE_BASED]
  ```

## Training Pipeline (Detailed)

### Phase 1: Dataset Creation
```
Input: Audio files (WAV/MP3/FLAC)
  ↓
[formatter.py::format_audio_list()]
  ├─ Load audio with torchaudio
  ├─ Convert to mono if stereo
  ├─ Transcribe with Faster Whisper (VAD enabled)
  │  └─ Returns segments with word-level timestamps
  ├─ For each sentence-ending word (.?!):
  │  ├─ Extract audio segment (with buffer)
  │  ├─ Clean text (multilingual_cleaners)
  │  ├─ Save WAV to output/dataset/wavs/
  │  └─ Append to metadata CSV
  ├─ Shuffle combined dataset
  └─ Split into train (85%) / eval (15%)
  ↓
Output: metadata_train.csv, metadata_eval.csv
```

### Phase 2: Training
```
Input: CSV metadata, language, hyperparameters
  ↓
[gpt_train.py::train_gpt()]
  ├─ Download base XTTS model (if not cached)
  │  ├─ base_models/{version}/model.pth
  │  ├─ base_models/{version}/config.json
  │  ├─ base_models/{version}/vocab.json
  │  └─ base_models/{version}/dvae.pth
  ├─ Copy vocab, config, speakers to ready/
  ├─ Configure GPTTrainer
  │  ├─ Model args (max_wav_length, tokenizer, etc.)
  │  ├─ Audio config (22050Hz, DVAE settings)
  │  └─ Training config (lr, batch_size, epochs)
  ├─ Load samples from CSV (train + eval)
  ├─ Initialize model from base checkpoint
  └─ Run training loop (Trainer.fit)
     ├─ Save checkpoint every 1000 steps
     ├─ Track validation loss
     └─ Keep only best model
  ↓
Output: output/run/training/GPT_XTTS_FT-*/best_model.pth
```

### Phase 3: Optimization
```
Input: Training folder with best_model.pth
  ↓
[xtts_demo.py::optimize button]
  ├─ Find best checkpoint (formatter.find_latest_best_model)
  ├─ Copy to output/ready/model.pth
  ├─ Copy example audio to output/ready/reference.wav
  └─ Optional: Delete training folders
  ↓
Output: Deployment-ready model in ready/
```

### Phase 4: Inference
```
Input: ready/ folder, text, reference audio
  ↓
[xtts_demo.py::run_tts()]
  ├─ Load XTTS model from ready/
  │  ├─ config.json
  │  ├─ model.pth
  │  ├─ vocab.json
  │  └─ speakers_xtts.pth
  ├─ Extract conditioning latents from reference audio
  │  └─ XTTS.get_conditioning_latents()
  ├─ Run inference
  │  └─ XTTS.inference(text, language, latents, params)
  └─ Save output WAV
  ↓
Output: Synthesized speech audio
```

## Data Flow

### File System Structure
```
project_root/
├── xtts_demo.py             # Main entry point
├── headlessXttsTrain.py     # CLI entry point
├── utils/
│   ├── formatter.py         # Dataset creation
│   ├── gpt_train.py         # Training logic
│   └── tokenizer.py         # Language processing
├── amharic_tts/             # Amharic module (self-contained)
│   ├── g2p/                 # G2P conversion
│   ├── tokenizer/           # Amharic tokenization
│   ├── preprocessing/       # Text normalization
│   └── config/              # Configuration
├── tests/                   # Test suite
│   ├── test_amharic_integration.py
│   └── test_amharic_g2p_comprehensive.py
├── base_models/             # Cached XTTS models
│   ├── v2.0.1/
│   ├── v2.0.2/
│   └── main/
└── output/ (or finetune_models/)  # User data
    ├── dataset/             # Processed training data
    │   ├── wavs/            # Audio segments
    │   ├── metadata_train.csv
    │   ├── metadata_eval.csv
    │   └── lang.txt
    ├── run/training/        # Training checkpoints
    │   └── GPT_XTTS_FT-*/
    │       └── best_model.pth
    └── ready/               # Optimized model
        ├── model.pth
        ├── config.json
        ├── vocab.json
        ├── speakers_xtts.pth
        └── reference.wav
```

## Key Design Decisions

### 1. File-Based State
**Decision:** All state persisted to filesystem, not memory
**Rationale:** 
- Enables UI restart without data loss
- Simplifies debugging (inspect files directly)
- Supports incremental workflows (add audio over time)
**Trade-off:** More I/O, but negligible vs. training time

### 2. Multi-Backend G2P
**Decision:** Try multiple backends with fallback
**Rationale:**
- Transphone best quality but optional dependency
- Rule-based always works (zero dependencies)
- Users choose accuracy vs. installation complexity
**Trade-off:** More code complexity, but better UX

### 3. CSV Metadata Format
**Decision:** Pipe-separated CSV (not JSON or database)
**Rationale:**
- Compatible with Coqui TTS data loaders
- Human-readable and editable
- Efficient for 1000s of samples
**Trade-off:** Less flexible than JSON, but standard in TTS community

### 4. Separate Headless Script
**Decision:** Duplicate logic in headlessXttsTrain.py vs. reusing xtts_demo.py
**Rationale:**
- Different UX needs (CLI vs. callbacks)
- Gradio dependency optional for headless
- FFmpeg preprocessing only in headless
**Trade-off:** Code duplication, but cleaner separation

### 5. Language Module Pattern
**Decision:** Self-contained `amharic_tts/` with zero hard deps
**Rationale:**
- Extensible to other languages
- Optional installation (transphone, epitran)
- Clear API boundaries
**Trade-off:** More directory structure, but better maintainability

## Critical Implementation Paths

### Path 1: Audio → Metadata (formatter.py)
```python
# Lines 89-176 in formatter.py
for audio_path in audio_files:
    # Check if already processed (incremental)
    if any(metadata contains audio_file_name prefix):
        skip_processing = True
        continue
    
    # Load and convert to mono
    wav, sr = torchaudio.load(audio_path)
    if wav.size(0) != 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    
    # Transcribe with Faster Whisper
    segments = whisper.transcribe(
        audio_path, 
        vad_filter=True,      # Voice activity detection
        word_timestamps=True,  # Word-level timing
        language=target_language
    )
    
    # Segment by sentence-ending punctuation
    for word in words_list:
        sentence += word.word
        if word.word[-1] in ["!", "。", ".", "?"]:
            # Clean text
            sentence = multilingual_cleaners(sentence, language)
            
            # Extract audio segment
            audio_segment = wav[start_sample:end_sample]
            
            # Save if duration >= 0.33 seconds
            if audio_segment.duration >= sr/3:
                torchaudio.save(wav_path, audio_segment, sr)
                metadata.append(wav_path, sentence, speaker)
```

### Path 2: Training Configuration (gpt_train.py)
```python
# Lines 133-186 in gpt_train.py
model_args = GPTArgs(
    max_conditioning_length=132300,  # 6 seconds
    max_wav_length=255995,          # ~11.6 seconds
    max_text_length=200,
    mel_norm_file=MEL_NORM_FILE,
    dvae_checkpoint=DVAE_CHECKPOINT,
    xtts_checkpoint=XTTS_CHECKPOINT,  # Base or custom
    tokenizer_file=TOKENIZER_FILE,
    # ... GPT-specific args
)

config = GPTTrainerConfig(
    epochs=num_epochs,
    batch_size=BATCH_SIZE,
    lr=5e-06,  # Critical: XTTS-optimized learning rate
    save_step=1000,
    save_n_checkpoints=1,  # Keep only best
    # ... other training params
)

model = GPTTrainer.init_from_config(config)
trainer = Trainer(...)
trainer.fit()  # Start training
```

### Path 3: Amharic G2P Conversion (amharic_g2p_enhanced.py)
```python
# Lines 120-180 in amharic_g2p_enhanced.py
def convert(self, text: str) -> str:
    # Try each backend in order
    for backend in self.config.g2p.backend_order:
        try:
            if backend == G2PBackend.TRANSPHONE:
                result = self._convert_transphone(text)
            elif backend == G2PBackend.EPITRAN:
                result = self._convert_epitran(text)
            elif backend == G2PBackend.RULE_BASED:
                result = self._convert_rule_based(text)
            
            # Validate quality
            if self._validate_quality(result, text):
                self._log_backend_success(backend)
                return result
        except Exception as e:
            self._log_backend_failure(backend, e)
            continue  # Try next backend
    
    # All backends failed - should never happen (rule-based is failsafe)
    raise RuntimeError("All G2P backends failed")
```

## Component Relationships

```
xtts_demo.py (Gradio UI)
  ↓ imports
  ├─ utils.formatter (audio processing)
  │  ↓ uses
  │  ├─ faster_whisper.WhisperModel
  │  ├─ torchaudio (audio I/O)
  │  └─ TTS.tokenizer.multilingual_cleaners
  │
  ├─ utils.gpt_train (training)
  │  ↓ uses
  │  ├─ TTS.tts.layers.xtts.trainer.GPTTrainer
  │  ├─ TTS.tts.datasets.load_tts_samples
  │  └─ trainer.Trainer
  │
  └─ TTS.tts.models.xtts.Xtts (inference)
      ↓ uses
      └─ ready/ folder (model files)

amharic_tts/ (independent module)
  ├─ g2p.amharic_g2p_enhanced
  │  ↓ optional imports
  │  ├─ transphone (if installed)
  │  └─ epitran (if installed)
  │
  └─ config.amharic_config
      ↓ defines
      └─ Backend order, thresholds, phonemes
```
