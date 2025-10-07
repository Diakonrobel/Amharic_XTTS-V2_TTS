# Project Brief: XTTS Fine-tuning WebUI with Amharic TTS Enhancement

## Core Mission
Transform the existing XTTS fine-tuning web interface into a comprehensive multilingual TTS platform with native support for Amharic language, incorporating advanced G2P (Grapheme-to-Phoneme) conversion, custom tokenization, and specialized fine-tuning capabilities for Ethiopic script.

## Primary Goals

### 1. Maintain Existing Functionality
- Preserve all current XTTS v2 fine-tuning capabilities
- Keep the 3-tab Gradio interface intact (Data Processing, Fine-tuning, Inference)
- Maintain support for 16 existing languages
- Ensure headless training mode continues to work

### 2. Add Amharic Language Support
- Implement comprehensive Ethiopic script support (340+ characters)
- Integrate G2P conversion using transphone/epitran backends
- Handle Amharic-specific phonological rules:
  - Epenthetic vowel insertion
  - Gemination handling
  - Labiovelar consonant processing
- Support both Amharic text and IPA phoneme input

### 3. Quality Targets
- Mean Opinion Score (MOS): ≥ 4.0
- Word Error Rate (WER): < 10%
- Real-Time Factor (RTF): < 0.3
- Speaker similarity: > 0.85

## Technical Scope

### In Scope
- Amharic G2P module with multiple backends
- Extended tokenizer with Ethiopic script support
- Amharic text preprocessing (numbers, abbreviations, punctuation)
- Custom phoneme mapping for Amharic
- Integration with existing XTTS v2 training pipeline
- Amharic-specific dataset preparation tools

### Out of Scope (Phase 1)
- Complete API deployment infrastructure
- Real-time streaming synthesis
- Multi-speaker emotion control
- Production monitoring system
- ONNX optimization

## Success Criteria
1. Successfully train an XTTS model on Amharic dataset
2. Generate natural-sounding Amharic speech
3. Zero-shot voice cloning works for Amharic speakers
4. Headless training supports Amharic with `--lang amh`
5. WebUI allows selection of Amharic language
6. G2P conversion produces correct IPA phonemes for Amharic text

## Constraints
- Must use existing XTTS v2.0.3 base model
- GPU memory usage must stay under 8GB
- No breaking changes to existing language support
- Windows, Linux, and Mac compatibility maintained
