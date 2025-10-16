# Silero VAD Integration Guide

## 🎯 Overview

This project uses **Silero VAD** (Voice Activity Detection) for intelligent audio segmentation. Silero VAD is a state-of-the-art, language-agnostic VAD system that supports **6000+ languages** including Amharic.

---

## ✨ Features

### Core Capabilities
- ✅ **Language-Agnostic**: Works with Amharic, English, and 6000+ languages
- ✅ **High Accuracy**: State-of-the-art VAD from Silero.ai
- ✅ **GPU Acceleration**: CUDA support for fast processing
- ✅ **ONNX Runtime**: Optional ONNX for even faster inference
- ✅ **Quality Metrics**: SNR estimation, confidence scores, silence detection

### Amharic-Specific Optimizations
- 🇪🇹 **Ejective Consonants**: Optimized padding to capture sharp consonantal bursts
- 🇪🇹 **Consonant Clusters**: Adjusted silence detection for complex clusters
- 🇪🇹 **Variable Syllables**: Adaptive thresholding for varying syllable durations
- 🇪🇹 **Natural Pauses**: Preserves phrase boundaries common in Amharic speech

---

## 📚 Usage Guide

### Method 1: Using Gradio WebUI (Easiest)

1. **Navigate to "Dataset Preparation" tab**
2. **Choose "SRT Processing" sub-tab**
3. **Enable VAD refinement checkbox**
4. **Upload your SRT + audio files**
5. **Adjust VAD settings if needed:**
   - Sensitivity: 0.5 (default, works well for most cases)
   - Min Speech: 250ms
   - Min Silence: 300ms
   - Padding: 30ms
6. **Process!**

### Method 2: Programmatic Usage

```python
from utils.silero_vad_enhanced import SileroVADEnhanced, process_audio_with_silero

# Quick processing (Amharic audio)
segments = process_audio_with_silero(
    audio_path="amharic_audio.wav",
    output_dir="segments/",
    language="am",  # Enables Amharic optimizations
    vad_threshold=0.5,
    amharic_mode=True  # Explicitly enable Amharic mode
)

print(f"Extracted {len(segments)} segments")
for seg in segments:
    print(f"  {seg.start:.2f}s - {seg.end:.2f}s (SNR: {seg.snr_estimate:.1f}dB)")
```

### Method 3: Advanced Custom Processing

```python
import torch
import torchaudio
from utils.silero_vad_enhanced import SileroVADEnhanced

# Load audio
audio, sr = torchaudio.load("my_audio.wav")
audio = audio.squeeze()  # Ensure 1D

# Initialize VAD with custom settings
vad = SileroVADEnhanced(
    sample_rate=sr,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=300,
    amharic_mode=True,  # Enable for Amharic
    adaptive_threshold=True,  # Auto-adjust threshold
    use_onnx=False,  # Set True for faster inference
    device='cuda'  # or 'cpu'
)

# Detect speech segments
segments = vad.detect_speech_timestamps(audio)

# Filter by quality
high_quality = vad.filter_by_quality(
    segments,
    min_confidence=0.6,
    min_snr=15.0,  # Minimum 15dB SNR
    min_duration=1.0,
    max_duration=15.0
)

# Merge close segments
final_segments = vad.merge_close_segments(high_quality, max_gap=0.5)

print(f"Final: {len(final_segments)} high-quality segments")
```

---

## 🇪🇹 Amharic Mode

### What it does:
When `amharic_mode=True`, the VAD applies these optimizations:

| Parameter | Default | Amharic Mode | Reason |
|-----------|---------|--------------|--------|
| `min_silence_duration_ms` | 300ms | 250ms | Faster speech patterns |
| `speech_pad_ms` | 30ms | 50ms | Capture ejective consonants |
| `threshold` | 0.5 | 0.4 | Better ejective detection |

### Amharic Phonetic Characteristics:
- **Ejective consonants** (ጥ, ጭ, ቅ, etc.): Sharp bursts need extra padding
- **Gemination** (doubled consonants): Need careful silence detection
- **Syllabic structure**: CV patterns with varying duration
- **Phrase breaks**: Natural pauses between clauses

### When to use:
- ✅ Processing Amharic speech
- ✅ Mixed Amharic/English speech
- ✅ Amharic with background noise
- ❌ Pure English/other languages (use default mode)

---

## 🎛️ Parameter Tuning Guide

### `threshold` (0.0 - 1.0)
**What it does:** Confidence threshold for speech detection  
**Default:** 0.5  
**Tuning:**
- **Lower (0.3-0.4):** More sensitive, captures whispers/quiet speech, may include noise
- **Higher (0.6-0.8):** Less sensitive, only clear speech, may miss soft sounds
- **Amharic:** 0.4-0.5 (captures ejectives)

### `min_speech_duration_ms`
**What it does:** Minimum length of speech segment  
**Default:** 250ms  
**Tuning:**
- **Shorter (100-200ms):** Captures brief utterances, more segments
- **Longer (300-500ms):** Only substantial speech, fewer segments
- **Amharic:** 200-250ms (syllables can be brief)

### `min_silence_duration_ms`
**What it does:** Minimum silence to split segments  
**Default:** 300ms  
**Tuning:**
- **Shorter (200-250ms):** Splits at brief pauses, more segments
- **Longer (400-600ms):** Only splits at major pauses, longer segments
- **Amharic:** 250-300ms (natural phrase breaks)

### `speech_pad_ms`
**What it does:** Padding around detected speech  
**Default:** 30ms  
**Tuning:**
- **Shorter (10-20ms):** Tight boundaries, may clip edges
- **Longer (50-100ms):** Generous boundaries, includes more context
- **Amharic:** 50ms (captures ejective bursts)

---

## 📊 Quality Metrics

Each segment includes quality metrics:

### `confidence`
VAD confidence score (0-1)  
**Good:** > 0.6  
**Excellent:** > 0.8

### `snr_estimate`
Estimated Signal-to-Noise Ratio (dB)  
**Poor:** < 10 dB  
**Good:** 15-25 dB  
**Excellent:** > 25 dB

### `speech_prob_mean`
Average speech probability (0-1)  
**Good:** > 0.6  
**Excellent:** > 0.8

### `has_silence_padding`
Whether segment has natural silence at boundaries  
**True:** Clean segment with pauses  
**False:** May be clipped or merged

---

## 🚀 Performance Tips

### Speed Optimization

1. **Use ONNX Runtime** (2-3x faster):
   ```python
   vad = SileroVADEnhanced(use_onnx=True)
   ```

2. **Use GPU** (5-10x faster for long audio):
   ```python
   vad = SileroVADEnhanced(device='cuda')
   ```

3. **Batch Processing**: Process multiple files in parallel

### Quality Optimization

1. **Enable adaptive threshold**:
   ```python
   vad = SileroVADEnhanced(adaptive_threshold=True)
   ```

2. **Filter by quality**:
   ```python
   segments = vad.filter_by_quality(segments, min_snr=15.0)
   ```

3. **Merge close segments**:
   ```python
   segments = vad.merge_close_segments(segments, max_gap=0.5)
   ```

---

## 🔧 Troubleshooting

### Issue: VAD model fails to load
**Solution:**
```bash
# Reinstall torch and torchaudio
pip install --upgrade torch torchaudio

# Clear torch hub cache
rm -rf ~/.cache/torch/hub/snakers4_silero-vad_master
```

### Issue: Too many small segments
**Solution:**
- Increase `min_speech_duration_ms` to 400-500ms
- Decrease `threshold` to 0.6-0.7
- Use `merge_close_segments()` with `max_gap=1.0`

### Issue: Speech is being cut off
**Solution:**
- Increase `speech_pad_ms` to 50-100ms
- Decrease `threshold` to 0.3-0.4
- Check if audio has clipping or distortion

### Issue: Too much noise in segments
**Solution:**
- Increase `threshold` to 0.6-0.7
- Use `filter_by_quality()` with higher `min_snr`
- Pre-process audio with noise reduction

### Issue: Amharic ejectives are cut
**Solution:**
- Enable `amharic_mode=True`
- Increase `speech_pad_ms` to 60-80ms
- Decrease `threshold` to 0.35-0.4

---

## 📖 References

- **Silero VAD GitHub**: https://github.com/snakers4/silero-vad
- **Paper**: Silero VAD: pre-trained enterprise-grade Voice Activity Detector
- **Supported Languages**: 6000+ (language-agnostic model)
- **Model License**: MIT

---

## 🆘 Support

For issues specific to this integration:
1. Check the GitHub Issues page
2. Review the troubleshooting section above
3. Test with the example scripts

For Silero VAD questions:
- Visit: https://github.com/snakers4/silero-vad/discussions

---

**Last Updated**: 2025-10-16  
**Version**: 2.0 (Enhanced with Amharic optimizations)
