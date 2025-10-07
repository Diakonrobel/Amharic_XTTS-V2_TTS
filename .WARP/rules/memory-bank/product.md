# Product: XTTS Fine-tuning WebUI with Amharic Support

## Why This Project Exists

### Problem Statement
1. **Lack of Amharic TTS Options**: Limited high-quality Amharic text-to-speech systems exist
2. **Complex Script Handling**: Ethiopic script (340+ characters) requires specialized processing
3. **Phonological Complexity**: Amharic has unique phonological features (epenthesis, gemination, labiovelars) not handled by standard TTS systems
4. **Voice Cloning for Amharic**: No accessible tools for zero-shot Amharic voice cloning
5. **Accessibility**: Current TTS tools are technical and hard to use for non-developers

### Who This Helps
- **Amharic Content Creators**: Podcasters, video creators, audiobook producers
- **Accessibility Services**: Screen readers for visually impaired Amharic speakers
- **Educational Technology**: E-learning platforms for Amharic language education
- **Voice Assistant Developers**: Building Amharic voice interfaces
- **Researchers**: Studying Amharic speech synthesis and phonology

## How It Should Work

### User Experience Goals

#### For Web Interface Users
1. **Simple Training Flow**:
   - Upload Amharic audio files (with Ethiopic script transcriptions)
   - System automatically handles G2P conversion and phoneme extraction
   - Configure training parameters via intuitive sliders
   - One-click model optimization

2. **Amharic-Aware Processing**:
   - Automatic script detection (Ethiopic)
   - Proper handling of Amharic punctuation (።፣፤፥ etc.)
   - Number-to-word expansion in Amharic
   - Abbreviation handling (ዓ.ም, ክ.ክ, etc.)

3. **Quality Inference**:
   - Natural-sounding speech generation
   - Control over speed, pitch, emotion
   - Zero-shot voice cloning from reference audio
   - Support for long-form text synthesis

#### For Headless/CLI Users
1. **Scriptable Workflow**:
   ```bash
   python headlessXttsTrain.py --input_audio amharic_speaker.wav \
     --lang amh --epochs 10 --use_g2p --g2p_backend transphone
   ```

2. **Batch Processing**:
   - Process multiple speakers sequentially
   - Automatic dataset organization
   - Reproducible training configurations

### Core Capabilities

#### 1. Amharic Text Processing
- **Script Normalization**: Handle variant forms (ሥ→ስ, ዕ→እ, etc.)
- **G2P Conversion**: 
  - Transphone backend (primary)
  - Epitran backend (fallback)
  - Custom rule-based system (offline fallback)
- **Phonological Rules**:
  - Epenthetic vowel insertion: `[k] + [t] → [kɨt]`
  - Gemination: `[t_gem] → [tt]`
  - Labiovelar mapping: `ቋ → [qʷa]`

#### 2. Model Training
- **Dataset Creation**: Whisper transcription + Amharic text alignment
- **Fine-tuning**: Transfer learning from XTTS v2 base model
- **Optimization**: Reduced model size for deployment
- **Validation**: Automatic quality checks during training

#### 3. Inference
- **Text-to-Speech**: Ethiopic text → natural speech
- **Voice Cloning**: Clone any Amharic speaker from short audio sample
- **Prosody Control**: Adjust speed, pitch, emphasis
- **Batch Generation**: Process multiple texts efficiently

## Product Principles

### 1. Language-First Design
- Amharic is a first-class citizen, not an afterthought
- Script-aware UI elements
- Culturally appropriate default configurations

### 2. Scientific Accuracy
- Linguistically informed G2P rules
- Validated phoneme mappings
- Proper handling of Amharic phonology

### 3. Accessibility
- Both GUI and CLI interfaces
- Clear documentation with Amharic examples
- Helpful error messages in context

### 4. Extensibility
- Modular G2P backends
- Pluggable tokenizer extensions
- Easy addition of new Ethiopian languages (Tigrinya, Oromo)

## Success Metrics

### Technical Quality
- MOS Score ≥ 4.0 (near-human quality)
- WER < 10% (high intelligibility)
- RTF < 0.3 (faster than real-time)
- Speaker similarity > 0.85 (accurate cloning)

### Usability
- Users can train a model in < 1 hour with 20 minutes of audio
- Zero-shot cloning works with < 10 seconds reference audio
- 95% of users complete training without errors
- Clear documentation reduces support requests by 80%

### Impact
- 1000+ Amharic TTS models trained in first year
- Adopted by major Ethiopian educational platforms
- Enables new Amharic voice assistant applications
- Community contributions for other Ethiopian languages
