#!/usr/bin/env python3
"""
Test Amharic Inference with Different Modes
============================================

This script helps you test which combination works for your trained model:
1. G2P enabled vs disabled
2. Different language codes ('am' vs 'amh' vs 'en')
3. Direct Amharic text vs phonemes

Run this on your remote server where the model is loaded.
"""

# Sample Amharic text for testing
TEST_TEXT = "ሰላም"  # "Hello" in Amharic

print("""
╔══════════════════════════════════════════════════════════════════════╗
║          AMHARIC TTS INFERENCE MODE TESTER                           ║
╚══════════════════════════════════════════════════════════════════════╝

This script will test your model with different configurations to find
which one produces correct pronunciation.

TEST TEXT: "ሰላም" (Hello)

The script will test these combinations:
1. Amharic text + NO G2P + lang='amh'
2. Amharic text + NO G2P + lang='am'  
3. Amharic text + NO G2P + lang='en'
4. Amharic text + G2P enabled + lang='amh'
5. Amharic text + G2P enabled + lang='en' (phoneme mode)

After testing, listen to each output and identify which sounds correct.
""")

# Paste this into your inference code or Gradio interface:
print("""
═══════════════════════════════════════════════════════════════════════
TESTING INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════

Option A: Test via Gradio UI
------------------------------
1. In the Gradio interface, enter: ሰላም
2. Try these combinations and note which sounds best:

   Test 1: 
   - Language: amh
   - Enable Amharic G2P: UNCHECKED
   - Generate and listen → Note: "Test1_amh_noG2P.wav"

   Test 2:
   - Language: am  
   - Enable Amharic G2P: UNCHECKED
   - Generate and listen → Note: "Test2_am_noG2P.wav"

   Test 3:
   - Language: en
   - Enable Amharic G2P: UNCHECKED
   - Generate and listen → Note: "Test3_en_noG2P.wav"

   Test 4:
   - Language: amh
   - Enable Amharic G2P: CHECKED
   - Generate and listen → Note: "Test4_amh_withG2P.wav"

   Test 5:
   - Language: en
   - Enable Amharic G2P: CHECKED
   - Generate and listen → Note: "Test5_en_withG2P.wav"


Option B: Test Programmatically  
--------------------------------
Run this Python code in your server environment:
""")

test_code = '''
from TTS.tts.models.xtts import Xtts
import torch

# Assume model is already loaded as XTTS_MODEL
# and speaker latents are available as gpt_cond_latent, speaker_embedding

test_text = "ሰላም"
test_configs = [
    {"lang": "amh", "use_g2p": False, "name": "Test1_amh_noG2P"},
    {"lang": "am", "use_g2p": False, "name": "Test2_am_noG2P"},
    {"lang": "en", "use_g2p": False, "name": "Test3_en_noG2P"},
    {"lang": "amh", "use_g2p": True, "name": "Test4_amh_withG2P"},
    {"lang": "en", "use_g2p": True, "name": "Test5_en_withG2P"},
]

for config in test_configs:
    print(f"\\nTesting: {config['name']}")
    
    # Apply G2P if needed
    if config["use_g2p"]:
        try:
            from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
            g2p = AmharicG2P(backend='rule_based')  # Use rule-based for offline
            processed_text = g2p.convert(test_text)
            print(f"  G2P: {test_text} → {processed_text}")
        except Exception as e:
            print(f"  G2P failed: {e}")
            processed_text = test_text
    else:
        processed_text = test_text
        print(f"  Raw text: {processed_text}")
    
    try:
        # Run inference
        out = XTTS_MODEL.inference(
            text=processed_text,
            language=config["lang"],
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.7,
        )
        
        # Save audio
        import torchaudio
        torchaudio.save(
            f"{config['name']}.wav",
            torch.tensor(out["wav"]).unsqueeze(0),
            24000
        )
        print(f"  ✅ Saved: {config['name']}.wav")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\\n✅ Testing complete! Listen to each file and identify which sounds correct.")
'''

print(test_code)

print("""
═══════════════════════════════════════════════════════════════════════
INTERPRETING RESULTS
═══════════════════════════════════════════════════════════════════════

After testing, identify which test produced correct pronunciation:

If Test1 (amh, no G2P) works:
  → Your model was trained WITHOUT G2P on Amharic characters
  → Solution: Always use lang='amh' and disable G2P at inference

If Test2 (am, no G2P) works:
  → Your model was trained with 'am' language code
  → Solution: Always use lang='am' and disable G2P at inference

If Test3 (en, no G2P) works:
  → Your model treats Amharic as if it were English characters
  → Solution: Use lang='en' but this is suboptimal - retrain recommended

If Test4 (amh, with G2P) works:
  → Your model was trained WITH G2P phoneme preprocessing
  → Solution: Always enable G2P at inference with lang='amh'

If Test5 (en, with G2P) works:
  → Your model expects phonemes in 'en' phoneme mode
  → Solution: Enable G2P and use lang='en' (phoneme language)

If NONE work correctly:
  → Critical issue: vocab size mismatch between training and inference
  → Solution: Run diagnose_amharic_issue.py to confirm, then retrain

═══════════════════════════════════════════════════════════════════════
NEXT STEPS AFTER IDENTIFYING THE WORKING MODE
═══════════════════════════════════════════════════════════════════════

Once you know which mode works, you can:

1. Update the default settings in xtts_demo.py
2. Document the correct mode for your specific checkpoint
3. If no mode works, prepare to retrain with proper configuration

For retraining guidance, see:
- AMHARIC_TRAINING_SOLUTION.md
- VOCAB_EXTENSION_GUIDE.md
- QUICK_START_G2P.md
""")
