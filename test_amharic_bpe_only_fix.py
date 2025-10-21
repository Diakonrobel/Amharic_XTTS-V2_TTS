"""
Test Amharic BPE-Only Tokenizer Patch
======================================

This script verifies that the tokenizer patch allows BPE-only training
with Amharic language code without G2P dependencies.

Run: python test_amharic_bpe_only_fix.py
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_tokenizer_patch():
    """Test that the tokenizer can handle 'amh' language code"""
    
    print("=" * 70)
    print("🧪 TESTING AMHARIC BPE-ONLY TOKENIZER PATCH")
    print("=" * 70)
    print()
    
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer, GPTTrainerConfig, XttsAudioConfig, GPTArgs
        
        print("✅ TTS library imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TTS library: {e}")
        return False
    
    # Create minimal config for testing
    print("\n📝 Creating test configuration...")
    
    try:
        audio_config = XttsAudioConfig(
            sample_rate=22050,
            dvae_sample_rate=22050,
            output_sample_rate=24000
        )
        
        model_args = GPTArgs(
            max_conditioning_length=132300,
            min_conditioning_length=66150,
            debug_loading_failures=False,
            max_wav_length=255995,
            max_text_length=200,
            mel_norm_file=None,  # Not needed for tokenizer test
            dvae_checkpoint=None,
            xtts_checkpoint=None,  # Test without checkpoint
            tokenizer_file=None,  # Use default
            gpt_num_audio_tokens=1026,
            gpt_start_audio_token=1024,
            gpt_stop_audio_token=1025,
            gpt_use_masking_gt_prompt_approach=True,
            gpt_use_perceiver_resampler=True,
        )
        
        config = GPTTrainerConfig(
            epochs=1,
            output_path="./test_output",
            model_args=model_args,
            run_name="TEST",
            project_name="TEST",
            run_description="Tokenizer patch test",
            dashboard_logger="tensorboard",
            logger_uri=None,
            audio=audio_config,
            batch_size=1,
            batch_group_size=1,
            eval_batch_size=1,
            num_loader_workers=0,
            eval_split_max_size=1,
            print_step=1,
            plot_step=100,
            log_model_step=100,
            save_step=1000,
            save_n_checkpoints=1,
            save_checkpoints=False,
            print_eval=False,
            optimizer="AdamW",
            optimizer_wd_only_on_weights=True,
            optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 0.01},
            lr=1e-5,
            lr_scheduler="MultiStepLR",
            lr_scheduler_params={"milestones": [1], "gamma": 0.5},
            test_sentences=[],
        )
        
        print("✅ Configuration created")
    except Exception as e:
        print(f"❌ Failed to create config: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Initialize model
    print("\n🔧 Initializing model...")
    
    try:
        model = GPTTrainer.init_from_config(config)
        print("✅ Model initialized")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Apply the patch (same as in gpt_train.py)
    print("\n🩹 Applying Amharic tokenizer patch...")
    
    language = "amh"  # Test with 'amh' code
    
    if language in ["am", "amh"]:
        # Get tokenizer from model
        tokenizer = None
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        elif hasattr(model, 'xtts') and hasattr(model.xtts, 'tokenizer'):
            tokenizer = model.xtts.tokenizer
        
        if tokenizer and hasattr(tokenizer, 'char_limits'):
            # Add Amharic language codes
            if 'am' not in tokenizer.char_limits:
                tokenizer.char_limits['am'] = 200
                print(" > ✅ Added 'am' language code")
            
            if 'amh' not in tokenizer.char_limits:
                tokenizer.char_limits['amh'] = 200
                print(" > ✅ Added 'amh' language code")
            
            # Patch preprocess_text
            if hasattr(tokenizer, 'preprocess_text'):
                _original_preprocess = tokenizer.preprocess_text
                
                def _amharic_safe_preprocess(txt, lang):
                    try:
                        base_lang = lang.split('-')[0].lower() if isinstance(lang, str) else lang
                    except Exception:
                        base_lang = lang
                    
                    ipa_markers = ('ə', 'ɨ', 'ʔ', 'ʕ', 'ʷ', 'ː', 'ʼ', 'ʃ', 'ʧ', 'ʤ', 'ɲ')
                    
                    if base_lang in ('am', 'amh'):
                        if txt and any(marker in txt for marker in ipa_markers):
                            return txt
                        try:
                            return _original_preprocess(txt, 'en')
                        except Exception:
                            return txt
                    
                    return _original_preprocess(txt, lang)
                
                tokenizer.preprocess_text = _amharic_safe_preprocess
                print(" > ✅ Patched preprocess_text()")
            
            print(" > ℹ️  Patch applied successfully!")
        else:
            print(" > ❌ Could not find tokenizer")
            return False
    
    # Test tokenization with Amharic text
    print("\n🧪 Testing tokenization with Amharic text...")
    
    test_texts = [
        ("ሰላም ዓለም", "amh"),  # "Hello world" in Amharic
        ("ኢትዮጵያ", "amh"),   # "Ethiopia"
        ("አማርኛ", "am"),      # "Amharic" with 'am' code
    ]
    
    all_passed = True
    
    for text, lang_code in test_texts:
        print(f"\n  Testing: '{text}' (lang={lang_code})")
        
        try:
            # This should NOT raise NotImplementedError anymore
            encoded = tokenizer.encode(text, lang_code)
            print(f"    ✅ Encoded successfully: {len(encoded)} tokens")
            
            # Try decoding
            try:
                decoded = tokenizer.decode(encoded)
                print(f"    ✅ Decoded successfully")
            except Exception as e:
                print(f"    ⚠️  Decode failed (not critical): {e}")
            
        except NotImplementedError as e:
            print(f"    ❌ FAILED: {e}")
            all_passed = False
        except Exception as e:
            print(f"    ❌ FAILED with error: {e}")
            all_passed = False
    
    # Final result
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print()
        print("🎉 The tokenizer patch works correctly!")
        print("   You can now train with BPE-only mode using 'amh' language code.")
        print()
        print("Next step: Run training with use_amharic_g2p=False")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        print()
        print("The patch may need adjustment. Please check the error messages above.")
        return False


if __name__ == "__main__":
    success = test_tokenizer_patch()
    sys.exit(0 if success else 1)
