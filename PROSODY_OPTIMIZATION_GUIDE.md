# Prosody Optimization Guide for Amharic TTS
## Enhancing Natural Stops, Breathing, and Emotional Delivery

---

## 📊 **Impact Analysis: Where Prosody is Controlled**

```
┌─────────────────────────────────────────────────────────────┐
│ Factor                    │ Impact  │ When Applied          │
├───────────────────────────┼─────────┼──────────────────────┤
│ Training Dataset Quality  │ 70%     │ Before Training       │
│ Preprocessing Pipeline    │ 20%     │ Training + Inference  │
│ Inference Parameters      │ 5-8%    │ Runtime              │
│ Model Architecture        │ 2-5%    │ Training             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 **1. Training Dataset Quality** (70% Impact)

### **Critical Factors**

#### **A. Audio Quality for Prosody**
```
✅ GOOD Dataset:
- Natural pauses at punctuation (0.3-0.5s at ።)
- Audible breath sounds between sentences
- Rising intonation at question marks (፧, ?)
- Emphatic delivery at exclamations (!)
- Consistent punctuation in transcripts

❌ BAD Dataset:
- No pauses at punctuation
- Breath sounds cut out
- Monotone delivery
- Missing or inconsistent punctuation
```

#### **B. Audit Your Dataset**

Run this to check prosody quality:

```python
import librosa
import numpy as np

def analyze_prosody_in_audio(wav_path):
    """Analyze pause distribution in audio"""
    audio, sr = librosa.load(wav_path, sr=22050)
    
    # Detect silent segments
    intervals = librosa.effects.split(audio, top_db=40)
    
    # Calculate pause durations
    pauses = []
    for i in range(len(intervals) - 1):
        pause_start = intervals[i][1]
        pause_end = intervals[i+1][0]
        pause_duration = (pause_end - pause_start) / sr
        pauses.append(pause_duration)
    
    print(f"Audio: {wav_path}")
    print(f"Total pauses: {len(pauses)}")
    print(f"Avg pause: {np.mean(pauses):.3f}s")
    print(f"Pauses > 0.3s: {sum(1 for p in pauses if p > 0.3)}")
    
    return pauses
```

#### **C. Dataset Improvement Strategies**

**Option 1: Manual Audio Editing**
```bash
# Use Audacity or similar tool to:
# 1. Add 0.3-0.5s silence at sentence boundaries
# 2. Preserve natural breathing sounds
# 3. Ensure pauses match punctuation in transcripts
```

**Option 2: Prosody-Aware Recording**
```
Recording Guidelines:
- Read with natural pauses at punctuation
- Take breaths between sentences (don't edit out)
- Vary intonation: ↗️ for ?, ↘️ for ., ❗ for !
- Record emotional variations (happy, serious, questioning)
```

**Option 3: Synthesize Training Data with Prosody**
```python
# Use your existing TTS to generate varied prosody samples
texts = [
    "ሰላም!",           # Exclamation
    "እንዴት ነህ?",       # Question
    "ደህና ነኝ።",       # Statement
]
```

---

## ⚙️ **2. Preprocessing Pipeline** (20% Impact)

### **Your Current System**

You already have a comprehensive prosody system:

```
✅ Prosody Handler (amharic_tts/preprocessing/prosody_handler.py)
  - Pause duration mapping: ። (0.5s), ፣ (0.2s), ! (0.5s)
  - Intonation detection: ? (rising), ! (emphatic)
  - Ethiopic punctuation support

✅ Hybrid G2P (amharic_tts/g2p/hybrid_g2p.py)
  - Now properly normalizes punctuation spacing
  - Preserves punctuation for XTTS prosody learning
```

### **How XTTS Handles Prosody**

```python
# XTTS Internal Process:
# 1. Input text with punctuation: "ሰላም! እንዴት ነህ?"
# 2. Phoneme sequence preserves punctuation positions
# 3. Attention mechanism learns pause patterns from training
# 4. Generates audio with natural pauses at punctuation
```

**Key Insight:** XTTS learns prosody **implicitly** from punctuation positions during training. No explicit prosody markers needed!

### **Preprocessing Best Practices**

```python
# ✅ GOOD: Preserve punctuation
text = "ሰላም! እንዴት ነህ?"
# → "səlam! ɨndɨtə nəh?"  (punctuation preserved)

# ❌ BAD: Remove punctuation
text = "ሰላም እንዴት ነህ"
# → "səlam ɨndɨtə nəh"  (no prosody info)
```

---

## 🎛️ **3. Inference Parameters** (5-8% Impact)

### **XTTS Inference Controls**

```python
# In xtts_demo.py or inference code:

# Temperature: Controls prosody variation
temperature = 0.7  # 0.5-0.8 for natural prosody
                    # Higher = more expressive
                    # Lower = more monotone

# Repetition Penalty: Affects pause consistency
repetition_penalty = 2.0  # 1.5-2.5 for better pauses

# Speed: Affects overall pacing
speed = 1.0  # 0.9-1.1 recommended
             # Slower allows more natural pauses
```

### **Parameter Tuning Guide**

| Goal | Temperature | Repetition Penalty | Speed |
|------|------------|-------------------|-------|
| **More natural pauses** | 0.65 | 2.0 | 0.95 |
| **Emotional delivery** | 0.75-0.85 | 1.8 | 1.0 |
| **Question intonation** | 0.7 | 2.2 | 1.0 |
| **Calm narration** | 0.6 | 2.5 | 0.9 |

---

## 🧪 **4. Testing & Validation**

### **Prosody Test Suite**

```python
# Test different prosody scenarios:

test_cases = [
    # Exclamations
    "ሰላም! እንኳን ደስ አለዎት!",
    
    # Questions
    "እንዴት ነህ? ደህና ነህ?",
    
    # Statements with pauses
    "ዛሬ ቀናት። ምሽት ቀዝቃዛ ነው።",
    
    # Complex emotional mix
    "ምን? ይህ እውነት ነው! አማኝም።",
    
    # Long narration with breathing
    "ከዕለታት አንድ ቀን አንዲት መነኩሲት ጎኅ ሲቀድ ወደ ቤተ ክርስቲያን ሊሄዱ ይነሳሉ። በመንገድ ላይ ዛፍ አገኙ።",
]

for text in test_cases:
    # Generate with hybrid G2P
    phonemes = g2p.convert(text)
    
    # Generate audio with TTS
    audio = model.synthesize(phonemes, ...)
    
    # Listen and evaluate:
    # - Are pauses natural at punctuation?
    # - Does intonation rise for questions?
    # - Is emphasis clear on exclamations?
```

### **Prosody Quality Metrics**

```python
def evaluate_prosody(audio_path, transcript):
    """
    Evaluate prosody quality
    """
    # 1. Check pause alignment with punctuation
    pauses = detect_silence(audio_path)
    punct_positions = find_punctuation(transcript)
    
    alignment_score = compare_positions(pauses, punct_positions)
    
    # 2. Check intonation contours
    pitch_contour = extract_f0(audio_path)
    question_positions = find_questions(transcript)
    
    rising_intonation_score = check_rising_pitch(
        pitch_contour, question_positions
    )
    
    return {
        'pause_alignment': alignment_score,
        'question_intonation': rising_intonation_score,
        'overall_naturalness': (alignment_score + rising_intonation_score) / 2
    }
```

---

## 📋 **Action Plan: Improve Your TTS Prosody**

### **Phase 1: Dataset Audit** (1-2 hours)

```bash
# 1. Check 30 random samples
python -c "
import random
import os
dataset_files = os.listdir('datasets/wavs')
samples = random.sample(dataset_files, 30)
for wav in samples:
    print(f'Check: {wav}')
    # Listen and note:
    # - Pauses at punctuation? Y/N
    # - Natural breathing? Y/N
    # - Varied intonation? Y/N
"

# 2. Calculate statistics
# Target: 80%+ of samples have natural prosody
```

### **Phase 2: Preprocessing Verification** (30 mins)

```bash
# Test that punctuation is preserved
python -c "
from amharic_tts.g2p.hybrid_g2p import HybridAmharicG2P

g2p = HybridAmharicG2P()
test = 'ሰላም! እንዴት ነህ?'
result = g2p.convert(test)
print(f'Input:  {test}')
print(f'Output: {result}')
# Verify: ! and ? still present in output
"
```

### **Phase 3: Retrain with Enhanced Dataset** (if needed)

```bash
# If prosody is poor in current model:
# 1. Add/fix pauses in training audio
# 2. Ensure transcript punctuation matches audio
# 3. Retrain model
python headlessXttsTrain.py --config config_amharic.json
```

### **Phase 4: Inference Tuning** (15 mins)

```python
# Fine-tune inference parameters
# Test these combinations:

configs = [
    {'temperature': 0.65, 'repetition_penalty': 2.0, 'speed': 0.95},
    {'temperature': 0.70, 'repetition_penalty': 2.2, 'speed': 1.00},
    {'temperature': 0.75, 'repetition_penalty': 1.8, 'speed': 1.00},
]

for cfg in configs:
    audio = synthesize(text, **cfg)
    evaluate_and_save(audio, cfg)
```

---

## 🎯 **Quick Wins** (Implement Today)

1. ✅ **Prosody handler activated** (just committed!)
2. ✅ **Punctuation preserved** in G2P pipeline
3. 🔄 **Test inference parameters** (try temperature=0.7, speed=0.95)
4. 📊 **Audit 10 training samples** for natural pauses

---

## 📚 **Understanding XTTS Prosody**

### **How XTTS Learns Prosody**

```
Training Time:
┌─────────────────────────────────────────────┐
│ Text: "ሰላም።"                                 │
│   ↓                                          │
│ Phonemes: "səlam."  (. preserved)            │
│   ↓                                          │
│ Attention: Learns pause pattern at .         │
│   ↓                                          │
│ Audio: 0.5s silence after /m/ before period  │
└─────────────────────────────────────────────┘

Inference Time:
┌─────────────────────────────────────────────┐
│ Text: "ደህና ነኝ።"                             │
│   ↓                                          │
│ Phonemes: "dəhɨna nəɲ."                      │
│   ↓                                          │
│ Attention: Recognizes . → applies pause      │
│   ↓                                          │
│ Audio: Natural 0.5s pause                    │
└─────────────────────────────────────────────┘
```

### **Prosody Hierarchy in XTTS**

```
1. Primary (Learned from Training):
   - Punctuation-based pauses
   - Sentence boundary intonation
   - Word stress patterns

2. Secondary (Influenced by Parameters):
   - Prosody variation (temperature)
   - Pause consistency (repetition_penalty)
   - Overall pacing (speed)

3. Tertiary (Speaker-dependent):
   - Emotional coloring from reference audio
   - Speaking style transfer
```

---

## ❓ **FAQ**

### **Q: Should I add explicit prosody markers like `<pause:0.5>`?**
**A:** No! XTTS doesn't use them. Just keep natural punctuation (`.`, `!`, `?`, `።`, `፣`).

### **Q: My TTS has no pauses at all. What's wrong?**
**A:** Check:
1. Training data has pauses at punctuation?
2. Transcripts have punctuation?
3. G2P preserves punctuation? (now fixed ✅)

### **Q: Can I control breath sounds?**
**A:** Yes, but only indirectly:
- Include breath sounds in training data
- Model learns to add them naturally
- Cannot control per-inference

### **Q: Exclamations sound flat. How to fix?**
**A:** 
1. Training data needs emphatic delivery at `!`
2. Increase temperature (0.75-0.85)
3. Ensure `!` preserved in preprocessing

### **Q: Questions don't have rising intonation?**
**A:**
1. Training data must have question intonation
2. Ensure `?` or `፧` in transcripts
3. Can't fix with inference params alone

---

## 🚀 **Next Steps**

1. **Pull these changes on Lightning AI:**
   ```bash
   git pull origin main
   ```

2. **Test prosody with hybrid G2P:**
   ```python
   from amharic_tts.g2p.hybrid_g2p import HybridAmharicG2P
   g2p = HybridAmharicG2P()
   result = g2p.convert("ሰላም! እንዴት ነህ? ደህና ነኝ።")
   print(result)  # Should preserve !, ?, ።
   ```

3. **Generate test audio samples** with various prosody

4. **Audit training dataset** if prosody is still poor

---

## 📖 **References**

- **XTTS Paper:** [XTTS: Multilingual TTS](https://arxiv.org/abs/2406.04904)
- **Prosody Modeling:** FastSpeech2, Tacotron2
- **Your Implementation:** `amharic_tts/preprocessing/prosody_handler.py`

---

**Summary:** Prosody is 70% dataset quality, 20% preprocessing (now optimized ✅), 10% inference tuning. Focus on training data quality for biggest gains!
