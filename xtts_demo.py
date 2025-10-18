import argparse
import os
import sys
import tempfile
from pathlib import Path

import shutil
import glob

# Apply PyTorch 2.6 compatibility patches BEFORE importing TTS/trainer
try:
    from utils.pytorch26_patch import apply_pytorch26_compatibility_patches
    apply_pytorch26_compatibility_patches()
except Exception as e:
    print(f"Warning: Could not apply PyTorch 2.6 patches: {e}")

import gradio as gr
import librosa.display
import numpy as np

import torch
import torchaudio
import traceback
from utils.formatter import format_audio_list,find_latest_best_model, list_audios
from utils.gpt_train import train_gpt
from utils import srt_processor
from utils import youtube_downloader, srt_processor, audio_slicer, dataset_tracker, batch_processor, dataset_statistics, checkpoint_manager
from utils import audio_slicer

from faster_whisper import WhisperModel

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import requests
from utils.lang_norm import canonical_lang, is_amharic
try:
    from tokenizers import Tokenizer as HFTokenizer
except Exception:
    HFTokenizer = None

def download_file(url, destination):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file to {destination}")
        return destination
    except Exception as e:
        print(f"Failed to download the file: {e}")
        return None

# Clear logs
def remove_log_file(file_path):
     log_file = Path(file_path)

     if log_file.exists() and log_file.is_file():
         log_file.unlink()

# remove_log_file(str(Path.cwd() / "log.out"))

def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

XTTS_MODEL = None

def create_zip(folder_path, zip_name):
    zip_path = os.path.join(tempfile.gettempdir(), f"{zip_name}.zip")
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)
    return zip_path

def get_model_zip(out_path):
    ready_folder = os.path.join(out_path, "ready")
    if os.path.exists(ready_folder):
        return create_zip(ready_folder, "optimized_model")
    return None

def get_dataset_zip(out_path):
    dataset_folder = os.path.join(out_path, "dataset")
    if os.path.exists(dataset_folder):
        return create_zip(dataset_folder, "dataset")
    return None

def normalize_xtts_lang(lang: str) -> str:
    """Normalize user-provided language to the canonical code for Coqui/XTTS.
    
    Rules:
    - Amharic: accept 'am', 'amh', 'am-ET', etc. → return 'amh' (ISO 639-3)
      This avoids NotImplementedError in upstream tokenizer and ensures
      dataset/lang.txt, training, and inference are consistent.
    - Chinese: 'zh' → 'zh-cn' (as expected in some XTTS paths)
    - Others: lowercased base code.
    """
    return canonical_lang(lang, purpose="coqui") or lang


def load_model(xtts_checkpoint, xtts_config, xtts_vocab,xtts_speaker):
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to run the previous steps or manually set the `XTTS checkpoint path`, `XTTS config path`, and `XTTS vocab path` fields !!"
    
    config = XttsConfig()
    config.load_json(xtts_config)
    
    print("Loading XTTS model! ")
    
    # Check if checkpoint has extended vocabulary
    try:
        checkpoint = torch.load(xtts_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_vocab_size = checkpoint["model"]["gpt.text_embedding.weight"].shape[0]
        
        # Load vocabulary to check size
        import json
        with open(xtts_vocab, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            vocab_size = len(vocab_data['model']['vocab'])
        
        print(f" > Checkpoint vocabulary size: {checkpoint_vocab_size}")
        print(f" > Vocab file size: {vocab_size}")
        
        if checkpoint_vocab_size != vocab_size:
            print(f" > ⚠️  SIZE MISMATCH DETECTED!")
            print(f" > Checkpoint was trained with extended vocabulary ({checkpoint_vocab_size} tokens)")
            print(f" > But vocab file has {vocab_size} tokens")

            # Try to find a vocab file in ready/ that matches checkpoint size
            try:
                from pathlib import Path as _Path
                ready_dir = _Path(xtts_vocab).parent
                import json as _json
                candidate = None
                for p in sorted(ready_dir.glob("vocab*.json")):
                    try:
                        with open(p, 'r', encoding='utf-8') as _f:
                            _v = _json.load(_f)
                        _sz = len(_v.get('model', {}).get('vocab', []))
                        if _sz == checkpoint_vocab_size:
                            candidate = str(p)
                            break
                    except Exception:
                        continue
                if candidate and candidate != xtts_vocab:
                    print(f" > Found matching vocab for checkpoint: {candidate}")
                    xtts_vocab = candidate
                    vocab_size = checkpoint_vocab_size
            except Exception as _e:
                print(f" > Warning: vocab scan failed: {_e}")

            # Initialize model
            XTTS_MODEL = Xtts.init_from_config(config)

            # If still mismatched after scan, expand embeddings to vocab file size and load weights manually
            if vocab_size != checkpoint_vocab_size and hasattr(XTTS_MODEL, 'gpt'):
                print(f" > Resizing embeddings to vocab size ({vocab_size}) and loading weights manually")
                state_dict = checkpoint["model"]
                embed_dim = state_dict['gpt.text_embedding.weight'].shape[1]
                # Build new layers with vocab_size
                new_text_embedding = torch.nn.Embedding(vocab_size, embed_dim)
                new_text_embedding.weight.data[:checkpoint_vocab_size] = state_dict['gpt.text_embedding.weight']
                new_text_embedding.weight.data[checkpoint_vocab_size:] = torch.randn(vocab_size - checkpoint_vocab_size, embed_dim) * 0.02

                new_text_head = torch.nn.Linear(embed_dim, vocab_size)
                new_text_head.weight.data[:checkpoint_vocab_size] = state_dict['gpt.text_head.weight']
                new_text_head.weight.data[checkpoint_vocab_size:] = torch.randn(vocab_size - checkpoint_vocab_size, embed_dim) * 0.02
                new_text_head.bias.data[:checkpoint_vocab_size] = state_dict['gpt.text_head.bias']
                new_text_head.bias.data[checkpoint_vocab_size:] = torch.zeros(vocab_size - checkpoint_vocab_size)

            # Load non-text layers
                filtered_state = {k: v for k, v in state_dict.items() if 'text_embedding' not in k and 'text_head' not in k}
                _loader_mod = getattr(XTTS_MODEL, 'xtts', XTTS_MODEL)
                _loader_mod.load_state_dict(filtered_state, strict=False)

                # Replace layers
                XTTS_MODEL.gpt.text_embedding = new_text_embedding
                XTTS_MODEL.gpt.text_head = new_text_head
                # Ensure internal tokenizer is initialized from vocab
                try:
                    if HFTokenizer and hasattr(XTTS_MODEL, 'tokenizer') and getattr(XTTS_MODEL.tokenizer, 'tokenizer', None) is None:
                        XTTS_MODEL.tokenizer.tokenizer = HFTokenizer.from_file(xtts_vocab)
                        print(" > ✅ Initialized internal tokenizer from vocab file")
                except Exception as _e:
                    print(f" > ⚠️ Could not init internal tokenizer: {_e}")
                # Ensure gpt_inference compatibility proxy exists
                try:
                    if hasattr(XTTS_MODEL, 'gpt') and not hasattr(XTTS_MODEL.gpt, 'gpt_inference'):
                        # For legacy checkpoints, gpt_inference should just point to the main GPT model itself
                        # The GPT wrapper class in XTTS already has the generate() method
                        XTTS_MODEL.gpt.gpt_inference = XTTS_MODEL.gpt
                        print(" > ✅ Created compatibility proxy for gpt_inference (aliased to main gpt)")
                except Exception as _e:
                    print(f" > ⚠️ Could not create gpt_inference proxy: {_e}")
                print(f" > ✅ Checkpoint loaded (manual) and embeddings expanded to {vocab_size} tokens")
            else:
                # Sizes now match; proceed with standard load
                XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, speaker_file_path=xtts_speaker, use_deepspeed=False, eval=True)
        else:
            # Normal loading - sizes match
            XTTS_MODEL = Xtts.init_from_config(config)
            XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, speaker_file_path=xtts_speaker, use_deepspeed=False)
        
    except Exception as e:
        print(f" > Detected legacy checkpoint layout; will try standard load then robust fallback ({e})")
        # Try standard load first (works if sizes are actually compatible)
        XTTS_MODEL = Xtts.init_from_config(config)
        try:
            XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, speaker_file_path=xtts_speaker, use_deepspeed=False)
        except Exception as std_err:
            print(f" > Standard load failed: {std_err}")
            print(f" > Attempting robust manual loading...")
            # Fallback path: initialize model and load checkpoint manually, ignoring text embedding/head mismatches
            XTTS_MODEL = Xtts.init_from_config(config)
            try:
                # Load checkpoint and vocab sizes
                import json as _json
                checkpoint = torch.load(xtts_checkpoint, map_location="cpu", weights_only=False)
                state_dict = checkpoint.get("model", checkpoint)
                with open(xtts_vocab, 'r', encoding='utf-8') as _f:
                    vocab_data = _json.load(_f)
                vocab_size = len(vocab_data.get('model', {}).get('vocab', []))

                # Find candidate keys for embeddings and head in checkpoint
                embed_key = None
                head_w_key = None
                head_b_key = None
                for k, v in state_dict.items():
                    if embed_key is None and isinstance(v, torch.Tensor) and v.ndim == 2 and k.endswith("text_embedding.weight"):
                        embed_key = k
                    if embed_key is None and isinstance(v, torch.Tensor) and v.ndim == 2 and (".wte.weight" in k or k.endswith("embeddings.weight")):
                        embed_key = k
                    if head_w_key is None and isinstance(v, torch.Tensor) and v.ndim == 2 and (k.endswith("text_head.weight") or ".lm_head.0.weight" in k):
                        head_w_key = k
                    if head_b_key is None and isinstance(v, torch.Tensor) and v.ndim == 1 and (k.endswith("text_head.bias") or ".lm_head.0.bias" in k):
                        head_b_key = k
                # Determine checkpoint vocab size if possible
                checkpoint_vocab_size = None
                if embed_key is not None:
                    checkpoint_vocab_size = state_dict[embed_key].shape[0]

                # Build new embedding/head sized to vocab_size and copy if possible
                if hasattr(XTTS_MODEL, 'gpt'):
                    embed_dim = XTTS_MODEL.gpt.text_embedding.weight.shape[1]
                    new_text_embedding = torch.nn.Embedding(vocab_size, embed_dim)
                    if embed_key is not None:
                        ckpt_E = state_dict[embed_key]
                        ncopy = min(ckpt_E.shape[0], vocab_size)
                        new_text_embedding.weight.data[:ncopy] = ckpt_E[:ncopy]
                        if ncopy < vocab_size:
                            new_text_embedding.weight.data[ncopy:] = torch.randn(vocab_size - ncopy, embed_dim) * 0.02
                    else:
                        new_text_embedding.weight.data = torch.randn(vocab_size, embed_dim) * 0.02

                    new_text_head = torch.nn.Linear(embed_dim, vocab_size)
                    if head_w_key is not None and head_b_key is not None:
                        ckpt_W = state_dict[head_w_key]
                        ckpt_b = state_dict[head_b_key]
                        ncopy = min(ckpt_W.shape[0], vocab_size)
                        new_text_head.weight.data[:ncopy] = ckpt_W[:ncopy]
                        if ncopy < vocab_size:
                            new_text_head.weight.data[ncopy:] = torch.randn(vocab_size - ncopy, embed_dim) * 0.02
                        new_text_head.bias.data[:ncopy] = ckpt_b[:ncopy]
                        if ncopy < vocab_size:
                            new_text_head.bias.data[ncopy:] = torch.zeros(vocab_size - ncopy)
                    else:
                        # Initialize randomly if no head in checkpoint
                        torch.nn.init.normal_(new_text_head.weight, mean=0.0, std=0.02)
                        torch.nn.init.zeros_(new_text_head.bias)

                    # Filter out embedding/head keys from checkpoint
                    def _skip_key(name: str) -> bool:
                        return (
                            name.endswith("text_embedding.weight")
                            or name.endswith("text_head.weight")
                            or name.endswith("text_head.bias")
                            or name.endswith("embeddings.weight")
                            or name.endswith(".wte.weight")
                            or ".lm_head." in name
                        )

                    filtered_state = {k: v for k, v in state_dict.items() if not _skip_key(k)}
                    _loader_mod = getattr(XTTS_MODEL, 'xtts', XTTS_MODEL)
                    _loader_mod.load_state_dict(filtered_state, strict=False)
                    # Replace layers on model
                    XTTS_MODEL.gpt.text_embedding = new_text_embedding
                    XTTS_MODEL.gpt.text_head = new_text_head
                    # Ensure internal tokenizer is initialized from vocab
                    try:
                        if HFTokenizer and hasattr(XTTS_MODEL, 'tokenizer') and getattr(XTTS_MODEL.tokenizer, 'tokenizer', None) is None:
                            XTTS_MODEL.tokenizer.tokenizer = HFTokenizer.from_file(xtts_vocab)
                            print(" > ✅ Initialized internal tokenizer from vocab file")
                    except Exception as _e:
                        print(f" > ⚠️ Could not init internal tokenizer: {_e}")
                    # Ensure gpt_inference compatibility proxy exists
                    try:
                        if hasattr(XTTS_MODEL, 'gpt') and not hasattr(XTTS_MODEL.gpt, 'gpt_inference'):
                            # For legacy checkpoints, gpt_inference should just point to the main GPT model itself
                            # The GPT wrapper class in XTTS already has the generate() method
                            XTTS_MODEL.gpt.gpt_inference = XTTS_MODEL.gpt
                            print(" > ✅ Created compatibility proxy for gpt_inference (aliased to main gpt)")
                    except Exception as _e:
                        print(f" > ⚠️ Could not create gpt_inference proxy: {_e}")
                    print(f" > ✅ Manual load successful with vocab size {vocab_size} (ckpt tokens: {checkpoint_vocab_size or 'unknown'})")
                else:
                    print(" > ⚠️ Model does not expose gpt module; manual embedding resize skipped")
            except Exception as load_error:
                return (
                    "VOCABULARY SIZE MISMATCH ERROR\n\n"
                    "Your model was trained with Amharic G2P (extended vocabulary) but the vocab file doesn't match.\n\n"
                    "Please check if you have 'vocab_extended.json' in your model folder and use that instead of 'vocab.json'.\n\n"
                    f"Error: {str(load_error)}"
                )
    
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()
    
    # Patch tokenizer to support Amharic language codes
    if hasattr(XTTS_MODEL, 'tokenizer') and hasattr(XTTS_MODEL.tokenizer, 'char_limits'):
        if 'am' not in XTTS_MODEL.tokenizer.char_limits:
            # Add support for ISO 639-1 Amharic code
            XTTS_MODEL.tokenizer.char_limits['am'] = 200  # Amharic (ISO 639-1)
            print(" > ✅ Patched tokenizer to support 'am' language code")
        if 'amh' not in XTTS_MODEL.tokenizer.char_limits:
            # Add support for ISO 639-3 Amharic code
            XTTS_MODEL.tokenizer.char_limits['amh'] = 200  # Amharic (ISO 639-3)
            print(" > ✅ Patched tokenizer to support 'amh' language code")

        # Patch tokenizer preprocessing to protect IPA for Amharic phoneme mode
        try:
            if hasattr(XTTS_MODEL.tokenizer, 'preprocess_text'):
                _orig_preprocess = XTTS_MODEL.tokenizer.preprocess_text

                def _preprocess_text_ipa_safe(txt, lang):
                    # Normalize to base code without region (e.g., zh-cn -> zh)
                    try:
                        base_lang = lang.split('-')[0].lower() if isinstance(lang, str) else lang
                    except Exception:
                        base_lang = lang

                    # IPA markers to detect phoneme strings
                    ipa_markers = ('ə', 'ɨ', 'ʔ', 'ʕ', 'ʷ', 'ː', 'ʼ', 'ʃ', 'ʧ', 'ʤ', 'ɲ')

                    # Treat Amharic codes ('am','amh') and 'en' specially
                    if base_lang in ('am', 'amh', 'en'):
                        # If looks like IPA, return unchanged (already phonemized)
                        try:
                            if txt and any(marker in txt for marker in ipa_markers):
                                return txt
                        except Exception:
                            pass
                        # Fallback: use English cleaner to avoid NotImplementedError for 'amh'/'am'
                        try:
                            return _orig_preprocess(txt, 'en')
                        except Exception:
                            return txt

                    # Default behavior for all other cases
                    return _orig_preprocess(txt, lang)

                XTTS_MODEL.tokenizer.preprocess_text = _preprocess_text_ipa_safe
                print(" > ✅ Patched tokenizer preprocessing to preserve IPA (for 'am' and phoneme 'en')")
        except Exception as e:
            print(f" > ⚠️ Could not patch tokenizer preprocess_text IPA protection: {e}")

    print("Model Loaded!")
    return "Model Loaded!"

def run_tts(lang, tts_text, speaker_audio_file, temperature, length_penalty,repetition_penalty,top_k,top_p,sentence_split,use_config,use_g2p_inference=False, g2p_backend_infer="auto"):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model !!", None, None

    # Canonicalize language early (ensures 'amh')
    lang = normalize_xtts_lang(lang)

    # Apply G2P preprocessing if enabled for Amharic text
    g2p_active = False
    if use_g2p_inference and lang in ["am", "amh"]:
        try:
            # Choose backend: UI prefer, otherwise training meta if available, else transphone
            resolved_backend = g2p_backend_infer
            if resolved_backend == "auto":
                try:
                    # Derive ready dir from the speaker reference path
                    from pathlib import Path as _Path
                    ready_dir = _Path(speaker_audio_file).parent if speaker_audio_file else None
                    meta_path = ready_dir / "training_meta.json" if ready_dir else None
                    if meta_path and meta_path.exists():
                        import json as _json
                        with open(meta_path, 'r', encoding='utf-8') as _f:
                            _meta = _json.load(_f)
                            _am_meta = _meta.get('amharic', {})
                            resolved_backend = _am_meta.get('g2p_backend', 'transphone') or 'transphone'
                    else:
                        resolved_backend = 'transphone'
                except Exception:
                    resolved_backend = 'transphone'

            from amharic_tts.tokenizer.xtts_tokenizer_wrapper import create_xtts_tokenizer
            print(f" > 🇪🇹 Amharic G2P enabled for inference (backend: {resolved_backend})")
            print(f" > Original text: {tts_text[:50]}{'...' if len(tts_text) > 50 else ''}")
            tokenizer = create_xtts_tokenizer(use_phonemes=True, g2p_backend=resolved_backend)
            original_text = tts_text
            _g2p_lang = 'am' if lang in ('am', 'amh') else lang
            tts_text = tokenizer.preprocess_text(tts_text, lang=_g2p_lang)
            print(f" > Converted to phonemes: {tts_text[:100]}{'...' if len(tts_text) > 100 else ''}")
            
            # Validate G2P conversion worked
            if tts_text == original_text:
                print(f" > ⚠️  Warning: G2P conversion may not have worked (text unchanged)")
                print(f" > ⚠️  This could happen if G2P backends are not properly installed")
            else:
                print(f" > ✅ G2P conversion successful")
                g2p_active = True
                
        except Exception as e:
            print(f" > ❌ Error: G2P preprocessing failed: {e}")
            print(f" > 🔄 Falling back to original text")
            print(f" > 💡 Tip: For best results, install Transphone: pip install transphone")

# Normalize language code for XTTS (already canonicalized above)
    lang_norm = normalize_xtts_lang(lang)
    
    # FIXED: Don't override Amharic to English for G2P
    # This was causing the pronunciation issues by making the model
    # interpret Amharic phonemes as English
    if g2p_active:
        print(f" > Using language: {lang_norm} with phoneme mode")
    elif lang != lang_norm:
        print(f" > Language normalization: {lang} → {lang_norm}")
    else:
        print(f" > Using language: {lang_norm}")

    # Use 'am' for XTTS inference API when Amharic selected to avoid upstream NotImplementedError
    _inference_lang = 'am' if lang_norm in ('am', 'amh') else lang_norm

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs
    )

    if use_config:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=_inference_lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=XTTS_MODEL.config.temperature, # Add custom parameters here
            length_penalty=XTTS_MODEL.config.length_penalty,
            repetition_penalty=XTTS_MODEL.config.repetition_penalty,
            top_k=XTTS_MODEL.config.top_k,
            top_p=XTTS_MODEL.config.top_p,
            enable_text_splitting=True
        )
    else:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=_inference_lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature, # Add custom parameters here
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting=sentence_split
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return "Speech generated !", out_path, speaker_audio_file


def load_params_tts(out_path,version):
    
    out_path = Path(out_path)

    # base_model_path = Path.cwd() / "models" / version 

    # if not base_model_path.exists():
    #     return "Base model not found !","","",""

    ready_model_path = out_path / "ready" 

    # Check for extended vocabulary first (Amharic G2P)
    # Support both legacy and current extended vocab filenames
    vocab_extended_path_legacy = ready_model_path / "vocab_extended.json"
    vocab_extended_path_amharic = ready_model_path / "vocab_extended_amharic.json"
    vocab_path = ready_model_path / "vocab.json"

    extended_vocab_found = False
    if vocab_extended_path_amharic.exists():
        vocab_path = vocab_extended_path_amharic
        extended_vocab_found = True
        print(" > Found extended vocabulary (Amharic): vocab_extended_amharic.json")
    elif vocab_extended_path_legacy.exists():
        vocab_path = vocab_extended_path_legacy
        extended_vocab_found = True
        print(" > Found extended vocabulary (legacy): vocab_extended.json")
    elif vocab_path.exists():
        print(" > Using standard vocabulary")
    else:
        return "Vocabulary file not found", "", "", "", "", ""
    
    # Ensure a minimal training_meta.json exists (for inference auto settings)
    try:
        import json as _json
        training_meta_path = ready_model_path / "training_meta.json"
        if not training_meta_path.exists():
            amharic_meta = {
                "g2p_training_enabled": bool(extended_vocab_found),
                "g2p_backend": "transphone" if extended_vocab_found else None,
                # If G2P was used, tokenizer language during training was effectively 'en'
                # Otherwise, default to 'amh' for Amharic context in this WebUI
                "effective_language": "en" if extended_vocab_found else "amh",
                "vocab_used": "extended" if extended_vocab_found else "standard",
            }
            with open(training_meta_path, "w", encoding="utf-8") as _f:
                _json.dump({"amharic": amharic_meta}, _f, indent=2, ensure_ascii=False)
            print(" > Created training metadata at ready/training_meta.json (inferred)")
    except Exception as _e:
        print(f" > Warning: Could not create training metadata: {_e}")

    config_path = ready_model_path / "config.json"
    speaker_path =  ready_model_path / "speakers_xtts.pth"
    reference_path  = ready_model_path / "reference.wav"

    model_path = ready_model_path / "model.pth"

    if not model_path.exists():
        model_path = ready_model_path / "unoptimize_model.pth"
        if not model_path.exists():
          return "Params for TTS not found", "", "", "", "", ""         

    # Ensure 'amh' appears in config.json languages for clarity (does not affect runtime behavior)
    try:
        import json as _json
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as _f:
                _cfg = _json.load(_f)
            langs = _cfg.get("languages")
            if isinstance(langs, list) and "amh" not in langs:
                langs.append("amh")
                _cfg["languages"] = langs
                with open(config_path, "w", encoding="utf-8") as _f:
                    _json.dump(_cfg, _f, indent=2, ensure_ascii=False)
                print(" > Appended 'amh' to languages in ready/config.json")
    except Exception as _e:
        print(f" > Warning: Could not update languages in config.json: {_e}")

    return "Params for TTS loaded", model_path, config_path, vocab_path,speaker_path, reference_path
     

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""XTTS fine-tuning demo\n\n"""
        """
        Example runs:
        python3 TTS/demos/xtts_ft_demo/xtts_demo.py --port 
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        help="Name of the whisper model selected by default (Optional) Choices are: ['large-v3','large-v2', 'large', 'medium', 'small']   Default Value: 'large-v3'",
        default="large-v3",
    )
    parser.add_argument(
        "--audio_folder_path",
        type=str,
        help="Path to the folder with audio files (optional)",
        default="",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Enable sharing of the Gradio interface via public link.",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to run the gradio demo. Default: 5003",
        default=5003,
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="Output path (where data and checkpoints will be saved) Default: output/",
        default=str(Path.cwd() / "finetune_models"),
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train. Default: 15 (optimal for 30-40hr datasets)",
        default=15,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size. Default: 2 (memory-safe for 15GB GPU)",
        default=2,
    )
    parser.add_argument(
        "--grad_acumm",
        type=int,
        help="Grad accumulation steps. Default: 8 (effective batch = 16)",
        default=8,
    )
    parser.add_argument(
        "--max_audio_length",
        type=int,
        help="Max permitted audio size in seconds. Default: 11 (optimized for Amharic)",
        default=11,
    )

    args = parser.parse_args()

    with gr.Blocks(title=os.environ.get("APP_NAME", "Amharic XTTS Fine-tuning WebUI"), theme=gr.themes.Soft(), css="""
        .gradio-container {max-width: 1400px !important;}
        .tabs {border-radius: 8px;}
        .tab-nav button {font-weight: 500;}
        h1 {text-align: center; margin-bottom: 1em;}
        .compact-row {gap: 0.5em !important;}
    """) as demo:
        gr.Markdown("# 🎙️ Amharic XTTS Fine-tuning WebUI", elem_classes=["text-center"])
        gr.Markdown("### Professional Voice Cloning System with Advanced Dataset Processing", elem_classes=["text-center"])
        
        with gr.Tab("📁 Data Processing"):
            gr.Markdown("## Dataset Creation & Management")
            
            with gr.Group():
                gr.Markdown("### 🎯 **Configuration**")
                with gr.Row():
                    out_path = gr.Textbox(
                        label="Output Directory",
                        value=args.out_path,
                        placeholder="Path where datasets and models will be saved",
                        scale=3
                    )
                    whisper_model = gr.Dropdown(
                        label="Whisper Model",
                        value=args.whisper_model,
                        choices=["large-v3", "large-v2", "large", "medium", "small"],
                        scale=1
                    )
                with gr.Row():
                    lang = gr.Dropdown(
                        label="Dataset Language",
                        value="en",
                        choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "hu", "ko", "ja", "am", "amh"],
                        scale=1,
                        info="Use 'am' or 'amh' for Amharic"
                    )
                    audio_folder_path = gr.Textbox(
                        label="Audio Folder Path (Optional)",
                        value=args.audio_folder_path,
                        placeholder="Leave empty to use uploaded files",
                        scale=2
                    )
            
            with gr.Tabs():
                with gr.TabItem("📤 Upload Audio Files"):
                    upload_file = gr.File(
                        file_count="multiple",
                        label="Audio Files (WAV, MP3, FLAC)",
                        file_types=[".wav", ".mp3", ".flac"],
                        height=150
                    )
                    gr.Markdown("*Upload one or more audio files for automatic transcription and dataset creation*")
                
                with gr.TabItem("📝 SRT Processing"):
                    gr.Markdown("**Process subtitle files with media for timestamp-accurate datasets**")
                    with gr.Row():
                        srt_files = gr.File(
                            file_count="multiple",
                            label="📄 Subtitle Files (SRT/VTT)",
                            file_types=[".srt", ".vtt"],
                            scale=1
                        )
                        media_files = gr.File(
                            file_count="multiple",
                            label="🎬 Media Files (Audio/Video)",
                            file_types=[".mp4", ".mkv", ".avi", ".wav", ".mp3", ".flac"],
                            scale=1
                        )
                    
                    with gr.Row():
                        srt_batch_mode = gr.Checkbox(
                            label="🎬 Batch Mode",
                            value=False,
                            info="Process multiple file pairs as one dataset",
                            scale=1
                        )
                        use_vad_refinement = gr.Checkbox(
                            label="🎤 VAD Enhancement",
                            value=False,
                            info="AI-powered speech detection (+20% time)",
                            scale=1
                        )
                    
                    with gr.Row():
                        srt_incremental_mode = gr.Checkbox(
                            label="➕ Incremental Mode",
                            value=False,
                            info="Add to existing dataset (no overwrite)",
                            scale=1
                        )
                        srt_check_duplicates = gr.Checkbox(
                            label="🔍 Skip Duplicates",
                            value=True,
                            info="Auto-detect and skip duplicate audio files",
                            scale=1
                        )
                    
                    gr.Markdown("""
                    💡 **Tip**: Use *Incremental Mode* to grow your dataset over time by adding new SRT+media files without losing existing data.
                    Perfect for uploading batches of locally prepared subtitle files!
                    """)
                    
                    with gr.Accordion("⚙️ Segmentation Settings", open=True):
                        srt_buffer_padding = gr.Slider(
                            label="Audio Padding (seconds)",
                            minimum=0.1, maximum=1.0, step=0.05, value=0.4,
                            info="Extra audio before/after each segment to prevent cutoffs. Higher = safer (0.4s recommended)"
                        )
                        gr.Markdown("""
                        💡 **Audio Padding**: Adds extra audio around each segment to prevent speech cutoffs.  
                        - **0.2-0.3s**: Minimal padding (risk of cutoffs)  
                        - **0.4-0.5s**: Recommended (safe, prevents cutoffs)  
                        - **0.6-1.0s**: Maximum safety (may include extra silence)
                        """)
                    
                    with gr.Accordion("⚙️ VAD Settings (Advanced)", open=False):
                        with gr.Row():
                            use_enhanced_vad_option = gr.Checkbox(
                                label="✨ Enhanced VAD",
                                value=False,
                                info="Advanced VAD with quality metrics & adaptive threshold",
                                scale=1
                            )
                            amharic_mode_option = gr.Checkbox(
                                label="🇪🇹 Amharic Mode",
                                value=False,
                                info="Optimize for Amharic ejective consonants (auto for 'am' language)",
                                scale=1
                            )
                        with gr.Row():
                            vad_threshold = gr.Slider(
                                label="Sensitivity",
                                minimum=0.1, maximum=0.9, step=0.05, value=0.5,
                                info="Higher = stricter"
                            )
                            vad_min_speech_duration = gr.Slider(
                                label="Min Speech (ms)",
                                minimum=100, maximum=1000, step=50, value=250
                            )
                        with gr.Row():
                            vad_min_silence_duration = gr.Slider(
                                label="Min Silence (ms)",
                                minimum=100, maximum=1000, step=50, value=300
                            )
                            vad_speech_pad = gr.Slider(
                                label="Padding (ms)",
                                minimum=0, maximum=200, step=10, value=30
                            )
                        gr.Markdown("""
                        💡 **Enhanced VAD**: Better quality with adaptive threshold, SNR estimation, speech prob metrics.  
                        🇪🇹 **Amharic Mode**: Tuned for Amharic ejectives (ጥ, ጭ, ቅ) with extra padding. Auto-enabled for Amharic language.
                        """)
                    
                    process_srt_btn = gr.Button(value="▶️ Process SRT + Media", variant="primary", size="lg")
                    srt_status = gr.Textbox(label="Status", interactive=False, lines=6, show_label=False)
                
                with gr.TabItem("📹 YouTube Processing"):
                    gr.Markdown("**Download videos and extract transcripts automatically**")
                    youtube_url = gr.Textbox(
                        label="🔗 YouTube URL(s)",
                        placeholder="https://youtube.com/watch?v=... (comma or newline separated for batch)",
                        lines=2,
                        max_lines=5
                    )
                    with gr.Row():
                        youtube_transcript_lang = gr.Dropdown(
                            label="🌐 Transcript Language",
                            value="en",
                            choices=[
                                ("English", "en"), ("Spanish", "es"), ("French", "fr"), ("German", "de"),
                                ("Italian", "it"), ("Portuguese", "pt"), ("Russian", "ru"), ("Chinese", "zh"),
                                ("Japanese", "ja"), ("Korean", "ko"), ("Arabic", "ar"),
                                ("Amharic (አማርኛ)", "am"), ("Oromo (Oromoo)", "om"), ("Tigrinya (ትግርኛ)", "ti"),
                                ("Somali (Soomaali)", "so"), ("Swahili", "sw"), ("Hausa", "ha"),
                                ("Hindi", "hi"), ("Bengali", "bn"), ("Vietnamese", "vi"), ("Thai", "th"),
                                ("Indonesian", "id"), ("Filipino", "fil"), ("Polish", "pl"), ("Turkish", "tr"),
                                ("Dutch", "nl"), ("Czech", "cs"), ("Hungarian", "hu"), ("Ukrainian", "uk"),
                            ],
                            allow_custom_value=True,
                            scale=2
                        )
                        youtube_batch_mode = gr.Checkbox(
                            label="🎬 Batch Mode",
                            value=False,
                            info="Merge multiple videos",
                            scale=1
                        )
                    
                    with gr.Row():
                        youtube_incremental_mode = gr.Checkbox(
                            label="➕ Incremental Mode",
                            value=False,
                            info="Add to existing dataset (no overwrite)",
                            scale=1
                        )
                        youtube_check_duplicates = gr.Checkbox(
                            label="🔍 Skip Duplicates",
                            value=True,
                            info="Auto-detect and skip duplicate audio files",
                            scale=1
                        )
                    
                    gr.Markdown("""
                    💡 **Tip**: Use *Incremental Mode* to grow your dataset over time by adding new videos without losing existing data.
                    Perfect for building large datasets across multiple sessions!
                    """)
                    
                    with gr.Accordion("⚙️ Segmentation Settings", open=True):
                        youtube_buffer_padding = gr.Slider(
                            label="Audio Padding (seconds)",
                            minimum=0.1, maximum=1.0, step=0.05, value=0.4,
                            info="Extra audio before/after each segment to prevent cutoffs. Higher = safer (0.4s recommended)"
                        )
                        gr.Markdown("""
                        💡 **Audio Padding**: Adds extra audio around each segment to prevent speech cutoffs.  
                        - **0.2-0.3s**: Minimal padding (risk of cutoffs)  
                        - **0.4-0.5s**: Recommended (safe, prevents cutoffs)  
                        - **0.6-1.0s**: Maximum safety (may include extra silence)
                        """)
                    
                    with gr.Row():
                        youtube_use_vad = gr.Checkbox(
                            label="🎤 VAD Enhancement",
                            value=False,
                            info="AI-powered speech detection (+20% time per video)",
                            scale=1
                        )
                    
                    with gr.Accordion("⚙️ VAD Settings", open=False):
                        with gr.Row():
                            youtube_use_enhanced_vad = gr.Checkbox(
                                label="✨ Enhanced VAD",
                                value=False,
                                info="Advanced VAD with quality metrics",
                                scale=1
                            )
                            youtube_amharic_mode = gr.Checkbox(
                                label="🇪🇹 Amharic Mode",
                                value=False,
                                info="Optimize for Amharic ejectives (auto for 'am' language)",
                                scale=1
                            )
                        with gr.Row():
                            youtube_vad_threshold = gr.Slider(
                                label="Sensitivity",
                                minimum=0.1, maximum=0.9, step=0.05, value=0.5,
                                info="Higher = stricter"
                            )
                            youtube_vad_min_speech = gr.Slider(
                                label="Min Speech (ms)",
                                minimum=100, maximum=1000, step=50, value=250
                            )
                        with gr.Row():
                            youtube_vad_min_silence = gr.Slider(
                                label="Min Silence (ms)",
                                minimum=100, maximum=1000, step=50, value=300
                            )
                            youtube_vad_speech_pad = gr.Slider(
                                label="Padding (ms)",
                                minimum=0, maximum=200, step=10, value=30
                            )
                        gr.Markdown("""
                        💡 **Enhanced VAD**: Better quality with adaptive threshold, SNR estimation.  
                        🇪🇹 **Amharic Mode**: Tuned for Amharic ejectives (ጥ, ጭ, ቅ) with extra padding.
                        """)
                    
                    with gr.Accordion("🎵 Background Music Removal (Optional)", open=False):
                        gr.Markdown("""
                        **Remove background music from downloaded audio using AI (Demucs):**
                        - Extracts clean vocals for better TTS training
                        - Improves voice quality and reduces artifacts
                        - Requires `pip install demucs` (will be skipped if not installed)
                        """)
                        with gr.Row():
                            youtube_remove_bg = gr.Checkbox(
                                label="🎵 Remove Background Music",
                                value=False,
                                info="AI-powered vocal separation (adds ~2-5min per video)",
                                scale=1
                            )
                            youtube_bg_quality = gr.Radio(
                                label="Quality",
                                choices=[("⚡ Fast (1-2min)", "fast"), ("⚖️ Balanced (3-5min)", "balanced"), ("✨ Best (10-15min)", "best")],
                                value="balanced",
                                info="Higher quality = slower processing",
                                scale=2
                            )
                        youtube_bg_model = gr.Dropdown(
                            label="Model (Advanced)",
                            choices=[("htdemucs (Best Quality)", "htdemucs"), ("mdx_extra (Balanced)", "mdx_extra"), ("mdx (Fast)", "mdx")],
                            value="htdemucs",
                            info="Select Demucs model for separation",
                            visible=False  # Hidden by default, can be shown for advanced users
                        )
                        gr.Markdown("""
                        💡 **Recommended:** Use "Balanced" quality for most cases. "Fast" for quick tests, "Best" for final production datasets.  
                        ⚠️ **Note:** Background removal adds processing time but significantly improves TTS training quality.
                        """)
                    
                    with gr.Accordion("🔐 YouTube Authentication & Bypass (Optional)", open=False):
                        gr.Markdown("""
                        **Fix YouTube bot detection / sign-in requirements:**
                        - Export browser cookies or use browser cookie import
                        - Uses mobile web client (mweb) to bypass restrictions
                        """)
                        with gr.Row():
                            youtube_cookies_file = gr.Textbox(
                                label="🍪 Cookies File Path",
                                placeholder="C:\\path\\to\\cookies.txt (Netscape format)",
                                info="Export cookies from YouTube using browser extension",
                                scale=2
                            )
                            youtube_cookies_browser = gr.Dropdown(
                                label="📱 Import from Browser",
                                choices=["", "chrome", "firefox", "edge", "brave", "opera"],
                                value="",
                                info="Auto-import cookies from browser",
                                scale=1
                            )
                        with gr.Row():
                            youtube_proxy = gr.Textbox(
                                label="🌐 Proxy URL (Optional)",
                                placeholder="http://user:pass@host:port or socks5://host:port",
                                info="Optional proxy for IP rotation",
                                scale=2
                            )
                            youtube_user_agent = gr.Textbox(
                                label="🤖 Custom User-Agent (Optional)",
                                placeholder="Leave empty for default",
                                info="Custom browser user-agent string",
                                scale=2
                            )
                        gr.Markdown("""
                        **💡 How to export cookies:**
                        1. Install a browser extension like "Get cookies.txt LOCALLY" (Chrome/Firefox)
                        2. Visit youtube.com and sign in
                        3. Export cookies as Netscape format (.txt)
                        4. Paste the file path above OR use browser import
                        """)
                    
                    download_youtube_btn = gr.Button(value="▶️ Download & Process", variant="primary", size="lg")
                    youtube_status = gr.Textbox(label="Status", interactive=False, lines=6, show_label=False)
                
                with gr.TabItem("✂️ Audio Slicer"):
                    gr.Markdown("**Intelligently segment audio based on silence detection**")
                    slicer_audio_file = gr.File(
                        file_count="single",
                        label="🎵 Audio File",
                        file_types=[".wav", ".mp3", ".flac"],
                    )
                    with gr.Row():
                        slicer_threshold_db = gr.Slider(
                            label="Silence Threshold (dB)",
                            minimum=-60, maximum=-10, step=1, value=-40,
                            info="Volume for silence"
                        )
                        slicer_min_length = gr.Slider(
                            label="Min Length (sec)",
                            minimum=1.0, maximum=20.0, step=0.5, value=5.0,
                            info="Min segment duration"
                        )
                    with gr.Row():
                        slicer_min_interval = gr.Slider(
                            label="Min Silence (sec)",
                            minimum=0.1, maximum=2.0, step=0.1, value=0.3,
                            info="Min silence to split"
                        )
                        slicer_max_sil_kept = gr.Slider(
                            label="Padding (sec)",
                            minimum=0.0, maximum=2.0, step=0.1, value=0.5,
                            info="Silence padding"
                        )
                    slicer_auto_transcribe = gr.Checkbox(
                        label="🎤 Auto-transcribe with Whisper",
                        value=True,
                        info="Generate transcriptions automatically"
                    )
                    slice_audio_btn = gr.Button(value="▶️ Slice Audio", variant="primary", size="lg")
                    slicer_status = gr.Textbox(label="Status", interactive=False, lines=6, show_label=False)
                
                with gr.TabItem("📊 History"):
                    gr.Markdown("**View and manage dataset processing history**")
                    history_display = gr.Textbox(
                        label="Processing History",
                        lines=12,
                        interactive=False,
                        max_lines=20,
                        show_label=False
                    )
                    with gr.Row():
                        refresh_history_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1)
                        clear_history_btn = gr.Button("🗑️ Clear All", variant="stop", scale=1)
            
            with gr.Group():
                gr.Markdown("### 🎯 **Amharic G2P Options** (for 'amh' language)")
                with gr.Row():
                    use_amharic_g2p_preprocessing = gr.Checkbox(
                        label="Enable G2P Preprocessing",
                        value=False,
                        info="Convert Amharic text to phonemes",
                        scale=2
                    )
                    g2p_backend_selection = gr.Dropdown(
                        label="G2P Backend",
                        value="transphone",
                        choices=["transphone", "epitran", "rule_based"],
                        info="Auto-fallback if unavailable",
                        scale=1
                    )
            
            with gr.Group():
                gr.Markdown("### 🚀 **Create Dataset**")
                progress_data = gr.Label(label="Status", value="Ready")
                prompt_compute_btn = gr.Button(value="▶️ Step 1 - Create Dataset", variant="primary", size="lg")
            
            with gr.Group():
                gr.Markdown("### 📊 **Dataset Statistics**")
                with gr.Row():
                    calculate_stats_btn = gr.Button(value="📈 Calculate Dataset Statistics", variant="secondary", size="lg")
                dataset_stats_display = gr.Textbox(
                    label="Statistics",
                    lines=20,
                    interactive=False,
                    max_lines=25,
                    show_label=False,
                    placeholder="Click 'Calculate Dataset Statistics' to see your dataset info (segments, duration, quality checks, etc.)"
                )
        
            # Advanced processing functions
            def show_dataset_statistics(out_path):
                """Calculate and display dataset statistics"""
                try:
                    dataset_path = os.path.join(out_path, "dataset")
                    stats = dataset_statistics.calculate_dataset_statistics(dataset_path)
                    display = dataset_statistics.format_statistics_display(stats)
                    return display
                except Exception as e:
                    traceback.print_exc()
                    return f"❌ Error calculating statistics: {str(e)}"
            def show_dataset_history(out_path):
                """Display dataset processing history"""
                try:
                    tracker = dataset_tracker.get_tracker(os.path.join(out_path, "dataset_history.json"))
                    return tracker.format_history_display(limit=20)
                except Exception as e:
                    return f"❌ Error loading history: {str(e)}"
            
            def clear_dataset_history(out_path):
                """Clear all dataset history"""
                try:
                    tracker = dataset_tracker.get_tracker(os.path.join(out_path, "dataset_history.json"))
                    result = tracker.clear_history()
                    return result
                except Exception as e:
                    return f"❌ Error clearing history: {str(e)}"
            def process_srt_media_batch_handler(srt_files_list, media_files_list, language, out_path, buffer_padding, incremental, check_duplicates, progress):
                """Handle batch processing of multiple SRT+media pairs"""
                try:
                    # Canonicalize language for dataset artifacts
                    language = normalize_xtts_lang(language)
                    mode_desc = "INCREMENTAL (adding to existing)" if incremental else "STANDARD (new dataset)"
                    progress(0, desc=f"Initializing batch processing ({mode_desc}) for {len(srt_files_list)} pairs...")
                    
                    # Process all pairs
                    train_csv, eval_csv, file_infos = batch_processor.process_srt_media_batch(
                        srt_files=srt_files_list,
                        media_files=media_files_list,
                        language=language,
                        out_path=out_path,
                        srt_processor=srt_processor,
                        progress_callback=lambda p, desc: progress(p, desc=desc),
                        incremental=incremental,
                        check_duplicates=check_duplicates,
                        buffer=buffer_padding
                    )
                    
                    # Count total segments
                    import pandas as pd
                    train_df = pd.read_csv(train_csv, sep='|')
                    eval_df = pd.read_csv(eval_csv, sep='|')
                    total_segments = len(train_df) + len(eval_df)
                    
                    # Track batch as single entry
                    tracker = dataset_tracker.get_tracker(os.path.join(out_path, "dataset_history.json"))
                    
                    # Add batch entry (using first file info as representative)
                    if file_infos:
                        first_info = file_infos[0]
                        tracker.add_file_dataset(
                            file_path=f"BATCH: {len(file_infos)} SRT files",
                            file_type="srt_batch",
                            language=language,
                            num_segments=total_segments,
                            output_path=os.path.join(out_path, "dataset"),
                            media_file=f"BATCH: {len(file_infos)} media files"
                        )
                    
                    # Format summary
                    summary = batch_processor.format_srt_batch_summary(file_infos, total_segments)
                    if incremental:
                        summary += "\n\n✅ INCREMENTAL MODE: New data added to existing dataset!"
                    summary += "\n\nℹ This batch dataset has been saved to history."
                    
                    progress(1.0, desc="Batch processing complete!")
                    return summary
                    
                except Exception as e:
                    traceback.print_exc()
                    return f"❌ Error in batch processing: {str(e)}"
            
            def process_srt_media(
                srt_file_input, 
                media_file_input, 
                language, 
                out_path, 
                batch_mode, 
                use_vad, 
                buffer_padding=0.4,
                vad_threshold_val=0.5,
                vad_min_speech_ms=250,
                vad_min_silence_ms=300,
                vad_pad_ms=30,
                use_enhanced_vad=False,
                amharic_mode=False,
                incremental=False,
                check_duplicates=True,
                progress=gr.Progress(track_tqdm=True)
            ):
                """Process SRT subtitle file(s) with corresponding media file(s)"""
                try:
                    # Canonicalize language for dataset artifacts
                    language = normalize_xtts_lang(language)
                    # Handle file inputs - can be None, single file path, or list of file paths
                    srt_files_list = []
                    media_files_list = []
                    
                    if srt_file_input:
                        if isinstance(srt_file_input, list):
                            srt_files_list = srt_file_input
                        else:
                            srt_files_list = [srt_file_input]
                    
                    if media_file_input:
                        if isinstance(media_file_input, list):
                            media_files_list = media_file_input
                        else:
                            media_files_list = [media_file_input]
                    
                    if not srt_files_list or not media_files_list:
                        return "Please upload both SRT file(s) and media file(s)!"
                    
                    # Check if batch mode and multiple files
                    if batch_mode and (len(srt_files_list) > 1 or len(media_files_list) > 1):
                        return process_srt_media_batch_handler(srt_files_list, media_files_list, language, out_path, buffer_padding, incremental, check_duplicates, progress)
                    
                    # Single file processing
                    srt_file_path = srt_files_list[0]
                    media_file_path = media_files_list[0]
                    
                    # Check if already processed
                    tracker = dataset_tracker.get_tracker(os.path.join(out_path, "dataset_history.json"))
                    is_processed, existing_dataset = tracker.is_file_processed(srt_file_path, "srt")
                    
                    if is_processed:
                        date = existing_dataset.get("processed_at", "unknown")[:19].replace("T", " ")
                        return f"⚠ SRT File Already Processed!\n\nThis file was already processed:\n" \
                               f"File: {existing_dataset.get('file_name', 'Unknown')}\n" \
                               f"Language: {existing_dataset.get('language', '?')}\n" \
                               f"Segments: {existing_dataset.get('num_segments', 0)}\n" \
                               f"Processed: {date}\n\n" \
                               f"ℹ If you want to reprocess, use a different output directory."
                    
                    output_path = os.path.join(out_path, "dataset")
                    os.makedirs(output_path, exist_ok=True)
                    
                    # CRITICAL: Silero VAD disabled due to text-audio mismatch bug
                    # Always use standard SRT processor for reliable results
                    mode_desc = "standard"
                    if use_vad:
                        progress(0, desc="⚠ VAD disabled (known bug) - using standard SRT processing...")
                    else:
                        progress(0, desc=f"Initializing {mode_desc} SRT processor...")
                    
                    progress(0.3, desc="Processing SRT and extracting audio segments...")
                    
                    train_csv, eval_csv, duration = srt_processor.process_srt_with_media(
                        srt_path=srt_file_path,
                        media_path=media_file_path,
                        output_dir=output_path,
                        language=language,
                        buffer=buffer_padding,
                        gradio_progress=progress
                    )
                    
                    # Count segments from train CSV
                    import pandas as pd
                    train_df = pd.read_csv(train_csv, sep='|')
                    eval_df = pd.read_csv(eval_csv, sep='|')
                    num_segments = len(train_df) + len(eval_df)
                    
                    # Track this dataset
                    tracker.add_file_dataset(
                        file_path=srt_file_path,
                        file_type="srt",
                        language=language,
                        num_segments=num_segments,
                        output_path=output_path,
                        media_file=media_file_path
                    )
                    
                    progress(1.0, desc="SRT processing complete!")
                    vad_warning = ""
                    if use_vad:
                        vad_warning = "\n\n⚠ Note: VAD was requested but disabled (text-audio mismatch bug). Used standard SRT processing instead."
                    return f"✓ SRT Processing Complete!\nProcessed {num_segments} segments\nTotal audio: {duration:.1f}s\nDataset created at: {output_path}\nMode: Standard (SRT-based){vad_warning}\n\nℹ This dataset has been saved to history and won't be reprocessed."
                    
                except Exception as e:
                    traceback.print_exc()
                    return f"❌ Error processing SRT: {str(e)}"
            
            def process_youtube_batch_urls(urls, transcript_lang, out_path, incremental, check_duplicates, cookies_path, cookies_from_browser, proxy, user_agent, buffer_padding, use_vad, vad_threshold, vad_min_speech, vad_min_silence, vad_pad, use_enhanced_vad, amharic_mode, remove_bg, bg_quality, bg_model, progress):
                """Process multiple YouTube URLs in batch mode"""
                try:
                    mode_desc = "INCREMENTAL (adding to existing)" if incremental else "STANDARD (new dataset)"
                    vad_desc = " + VAD" if use_vad else ""
                    progress(0, desc=f"Initializing batch processing ({mode_desc}{vad_desc}) for {len(urls)} videos...")
                    
                    # Prepare auth parameters (empty strings -> None)
                    cookies_file = cookies_path.strip() if cookies_path and cookies_path.strip() else None
                    cookies_browser = cookies_from_browser.strip() if cookies_from_browser and cookies_from_browser.strip() else None
                    proxy_url = proxy.strip() if proxy and proxy.strip() else None
                    ua = user_agent.strip() if user_agent and user_agent.strip() else None
                    
                    # Process all videos
                    train_csv, eval_csv, video_infos = batch_processor.process_youtube_batch(
                        urls=urls,
                        transcript_lang=transcript_lang,
                        out_path=out_path,
                        youtube_downloader=youtube_downloader,
                        srt_processor=srt_processor,
                        progress_callback=lambda p, desc: progress(p, desc=desc),
                        incremental=incremental,
                        check_duplicates=check_duplicates,
                        cookies_path=cookies_file,
                        cookies_from_browser=cookies_browser,
                        proxy=proxy_url,
                        user_agent=ua,
                        buffer=buffer_padding,
                        use_vad=use_vad,
                        vad_threshold=vad_threshold,
                        vad_min_speech_ms=int(vad_min_speech),
                        vad_min_silence_ms=int(vad_min_silence),
                        vad_pad_ms=int(vad_pad),
                        use_enhanced_vad=use_enhanced_vad,
                        amharic_mode=amharic_mode,
                        # Background music removal
                        remove_background_music=remove_bg,
                        background_removal_model=bg_model,
                        background_removal_quality=bg_quality,
                    )
                    
                    # Count total segments
                    import pandas as pd
                    train_df = pd.read_csv(train_csv, sep='|')
                    eval_df = pd.read_csv(eval_csv, sep='|')
                    total_segments = len(train_df) + len(eval_df)
                    
                    # Track batch as single entry
                    tracker = dataset_tracker.get_tracker(os.path.join(out_path, "dataset_history.json"))
                    
                    # Add batch entry (using first video info as representative)
                    if video_infos:
                        first_video = video_infos[0]
                        video_id = tracker._extract_youtube_id(first_video['url'])
                        if video_id:
                            tracker.add_youtube_dataset(
                                url=f"BATCH: {len(video_infos)} videos",
                                video_id=f"batch_{video_id}",
                                title=f"Batch: {', '.join([v['title'][:30] for v in video_infos[:3]])}{'...' if len(video_infos) > 3 else ''}",
                                language=transcript_lang,
                                duration=sum(v['duration'] for v in video_infos),
                                num_segments=total_segments,
                                output_path=os.path.join(out_path, "dataset")
                            )
                    
                    # Format summary
                    summary = batch_processor.format_batch_summary(video_infos, total_segments)
                    if incremental:
                        summary += "\n\n✅ INCREMENTAL MODE: New data added to existing dataset!"
                        summary += "\nℹ This batch dataset has been saved to history."
                    else:
                        summary += "\n\nℹ This batch dataset has been saved to history."
                    
                    progress(1.0, desc="Batch processing complete!")
                    return summary
                    
                except Exception as e:
                    traceback.print_exc()
                    return f"❌ Error in batch processing: {str(e)}"
            
            def download_youtube_video(url, transcript_lang, language, out_path, batch_mode, incremental_mode, check_duplicates, cookies_path, cookies_from_browser, proxy, user_agent, buffer_padding, use_vad, vad_threshold, vad_min_speech, vad_min_silence, vad_pad, use_enhanced_vad, amharic_mode, remove_bg, bg_quality, bg_model, progress=gr.Progress(track_tqdm=True)):
                """Download YouTube video(s) and extract transcripts"""
                try:
                    if not url:
                        return "Please enter a YouTube URL!"
                    
                    # Parse URLs
                    urls = batch_processor.parse_youtube_urls(url)
                    
                    if not urls:
                        return "❌ No valid YouTube URLs found. Please check your input."
                    
                    # Prepare auth parameters (empty strings -> None)
                    cookies_file = cookies_path.strip() if cookies_path and cookies_path.strip() else None
                    cookies_browser = cookies_from_browser.strip() if cookies_from_browser and cookies_from_browser.strip() else None
                    proxy_url = proxy.strip() if proxy and proxy.strip() else None
                    ua = user_agent.strip() if user_agent and user_agent.strip() else None
                    
                    # Check if batch mode and multiple URLs
                    if batch_mode and len(urls) > 1:
                        return process_youtube_batch_urls(urls, transcript_lang, out_path, incremental_mode, check_duplicates, cookies_file, cookies_browser, proxy_url, ua, buffer_padding, use_vad, vad_threshold, vad_min_speech, vad_min_silence, vad_pad, use_enhanced_vad, amharic_mode, remove_bg, bg_quality, bg_model, progress)
                    
                    # Single URL processing (existing logic)
                    url = urls[0]  # Use first URL
                    
                    # Check if already processed
                    tracker = dataset_tracker.get_tracker(os.path.join(out_path, "dataset_history.json"))
                    is_processed, existing_dataset = tracker.is_youtube_processed(url, transcript_lang)
                    
                    if is_processed:
                        date = existing_dataset.get("processed_at", "unknown")[:19].replace("T", " ")
                        return f"⚠ Video Already Processed!\n\nThis video was already downloaded and processed:\n" \
                               f"Title: {existing_dataset.get('title', 'Unknown')}\n" \
                               f"Language: {existing_dataset.get('language', '?')}\n" \
                               f"Segments: {existing_dataset.get('num_segments', 0)}\n" \
                               f"Processed: {date}\n" \
                               f"Output: {existing_dataset.get('output_path', 'Unknown')}\n\n" \
                               f"ℹ If you want to reprocess, please delete the dataset first or use a different output directory."
                    
                    progress(0, desc="Initializing YouTube downloader...")
                    temp_dir = tempfile.mkdtemp()
                    
                    progress(0.2, desc="Downloading video and subtitles...")
                    audio_path, srt_path, info = youtube_downloader.download_youtube_video(
                        url=url,
                        output_dir=temp_dir,
                        language=transcript_lang,
                        audio_only=True,
                        download_subtitles=True,
                        auto_update=True,
                        cookies_path=cookies_file,
                        cookies_from_browser=cookies_browser,
                        proxy=proxy_url,
                        user_agent=ua,
                        # Background music removal
                        remove_background_music=remove_bg,
                        background_removal_model=bg_model,
                        background_removal_quality=bg_quality,
                    )
                    
                    if not audio_path:
                        return "❌ Failed to download YouTube video. Check URL and try again."
                    
                    if not srt_path:
                        return "❌ No transcripts/subtitles available for this video. Try a different video or language."
                    
                    progress(0.6, desc="Processing transcript and audio...")
                    output_path = os.path.join(out_path, "dataset")
                    os.makedirs(output_path, exist_ok=True)
                    
# Use transcript language as dataset language (transcript_lang is the actual content language)
                    dataset_language = normalize_xtts_lang(transcript_lang)
                    print(f"Setting dataset language to '{dataset_language}' (from YouTube transcript language)")
                    
                    # CRITICAL: Silero VAD disabled due to text-audio mismatch bug
                    # Always use standard SRT processor for reliable results
                    if use_vad:
                        progress(0.6, desc="⚠ VAD disabled (known bug) - using standard processing...")
                    else:
                        progress(0.6, desc="Processing transcript and audio...")
                    
                    train_csv, eval_csv, duration = srt_processor.process_srt_with_media(
                        srt_path=srt_path,
                        media_path=audio_path,
                        output_dir=output_path,
                        language=dataset_language,
                        buffer=buffer_padding,  # Use user-configured padding
                        gradio_progress=progress
                    )
                    
                    # Count segments
                    import pandas as pd
                    train_df = pd.read_csv(train_csv, sep='|')
                    eval_df = pd.read_csv(eval_csv, sep='|')
                    num_segments = len(train_df) + len(eval_df)
                    
                    # Track this dataset
                    video_id = tracker._extract_youtube_id(url)
                    if video_id:
                        tracker.add_youtube_dataset(
                            url=url,
                            video_id=video_id,
                            title=info.get('title', 'Unknown'),
                            language=dataset_language,
                            duration=info.get('duration', 0),
                            num_segments=num_segments,
                            output_path=output_path
                        )
                    
                    # Cleanup temp directory
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except:
                        pass
                    
                    progress(1.0, desc="YouTube processing complete!")
                    vad_warning = ""
                    if use_vad:
                        vad_warning = "\n\n⚠ Note: VAD was requested but disabled (text-audio mismatch bug). Used standard SRT processing instead."
                    return f"✓ YouTube Processing Complete!\nTitle: {info.get('title', 'Unknown')}\nDuration: {info.get('duration', 0):.0f}s\nProcessed {num_segments} segments\nDataset created at: {output_path}{vad_warning}\n\nℹ This dataset has been saved to history and won't be reprocessed."
                    
                except Exception as e:
                    traceback.print_exc()
                    return f"❌ Error downloading YouTube video: {str(e)}"
            
            def slice_audio_file(audio_file_path, threshold_db, min_length, min_interval, max_sil_kept, auto_transcribe, whisper_model, language, out_path, progress=gr.Progress(track_tqdm=True)):
                """Slice audio file using RMS-based silence detection"""
                try:
                    if not audio_file_path:
                        return "Please upload an audio file to slice!"
                    
                    # Canonicalize dataset language for metadata writing
                    language = normalize_xtts_lang(language)

                    progress(0, desc="Initializing audio slicer...")
                    
                    output_path = os.path.join(out_path, "dataset", "wavs")
                    os.makedirs(output_path, exist_ok=True)
                    
                    # Load audio
                    import librosa
                    import soundfile as sf
                    progress(0.1, desc="Loading audio...")
                    audio, sr = librosa.load(audio_file_path, sr=22050, mono=True)
                    
                    # Initialize slicer
                    progress(0.2, desc="Slicing audio...")
                    slicer = audio_slicer.Slicer(
                        sr=sr,
                        threshold=threshold_db,
                        min_length=int(min_length * 1000),  # Convert to milliseconds
                        min_interval=int(min_interval * 1000),
                        hop_size=10,
                        max_sil_kept=int(max_sil_kept * 1000)
                    )
                    
                    # Slice audio
                    chunks = slicer.slice(audio)
                    
                    if not chunks:
                        return "❌ No segments created. Try adjusting the slicing parameters."
                    
                    progress(0.4, desc=f"Saving {len(chunks)} segments...")
                    
                    # Save segments
                    segment_files = []
                    for i, chunk in enumerate(chunks):
                        segment_filename = f"{Path(audio_file_path).stem}_segment_{str(i).zfill(4)}.wav"
                        segment_path = os.path.join(output_path, segment_filename)
                        sf.write(segment_path, chunk, sr)
                        segment_files.append(segment_path)
                    
                    # Auto-transcribe if requested
                    transcription_status = "Disabled"
                    if auto_transcribe:
                        progress(0.6, desc="Transcribing segments with Whisper...")
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        compute_type = "float16" if torch.cuda.is_available() else "float32"
                        asr_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
                        
                        # Create metadata by transcribing segments
                        # This processes all segments and creates train/eval CSVs
                        try:
                            train_meta, eval_meta, total_size = format_audio_list(
                                segment_files,
                                asr_model=asr_model,
                                target_language=language,
                                out_path=os.path.join(out_path, "dataset"),
                                gradio_progress=progress
                            )
                            transcription_status = "Enabled - Metadata created"
                        except Exception as e:
                            print(f"Warning: Could not create metadata: {e}")
                            transcription_status = "Enabled - Segments created, metadata pending"
                    
                    progress(1.0, desc="Audio slicing complete!")
                    return f"✓ Audio Slicing Complete!\nCreated {len(chunks)} segments\nSegments saved to: {output_path}\nAuto-transcription: {transcription_status}"
                    
                except Exception as e:
                    traceback.print_exc()
                    return f"❌ Error slicing audio: {str(e)}"
        
            def preprocess_dataset(audio_path, audio_folder_path, language, whisper_model, out_path, train_csv, eval_csv, use_g2p_preprocessing=False, g2p_backend="transphone", progress=gr.Progress(track_tqdm=True)):
                clear_gpu_cache()
            
                train_csv = ""
                eval_csv = ""
            
                # Canonicalize dataset language (ensure 'amh' for Amharic)
                language = normalize_xtts_lang(language)

                out_path = os.path.join(out_path, "dataset")
                os.makedirs(out_path, exist_ok=True)
                
                # Configure G2P if Amharic preprocessing is enabled
                g2p_converter = None
                if use_g2p_preprocessing and language == "amh":
                    try:
                        from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
                        print(f"Initializing Amharic G2P with backend: {g2p_backend}")
                        g2p_converter = AmharicG2P(backend=g2p_backend)
                        print("G2P converter initialized successfully")
                    except ImportError as e:
                        print(f"Warning: Could not load Amharic G2P: {e}")
                        print("Dataset will be created without G2P preprocessing")
                    except Exception as e:
                        print(f"Error initializing G2P: {e}")
                        print("Dataset will be created without G2P preprocessing")
            
                if audio_folder_path:
                    audio_files = list(list_audios(audio_folder_path))
                else:
                    audio_files = audio_path
            
                if not audio_files:
                    return "No audio files found! Please provide files via Gradio or specify a folder path.", "", ""
                else:
                    try:
                        # Loading Whisper
                        device = "cuda" if torch.cuda.is_available() else "cpu" 
                        
                        # Detect compute type 
                        if torch.cuda.is_available():
                            compute_type = "float16"
                        else:
                            compute_type = "float32"
                        
                        asr_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
                        
                        # Apply G2P preprocessing if converter is available
                        # Note: This would require modifications to format_audio_list to accept g2p_converter
                        # For now, we'll note this and users can preprocess separately
                        if g2p_converter:
                            print("Note: G2P preprocessing will be applied during training")
                            print("For dataset-level G2P, use the preprocessing utility")
                        
                        train_meta, eval_meta, audio_total_size = format_audio_list(audio_files, asr_model=asr_model, target_language=language, out_path=out_path, gradio_progress=progress)
                    except:
                        traceback.print_exc()
                        error = traceback.format_exc()
                        return f"The data processing was interrupted due an error !! Please check the console to verify the full error message! \n Error summary: {error}", "", ""
            
                # clear_gpu_cache()
            
                # if audio total len is less than 2 minutes raise an error
                if audio_total_size < 120:
                    message = "The sum of the duration of the audios that you provided should be at least 2 minutes!"
                    print(message)
                    return message, "", ""
            
                print("Dataset Processed!")
                return "Dataset Processed!", train_meta, eval_meta


        with gr.Tab("🔧 Fine-tuning"):
            gr.Markdown("## Model Training Configuration")
            
            with gr.Group():
                gr.Markdown("### 📂 **Dataset Configuration**")
                load_params_btn = gr.Button(value="📥 Load Parameters from Output Folder", variant="secondary")
                with gr.Row():
                    train_csv = gr.Textbox(label="Train CSV Path", placeholder="Auto-filled after loading", scale=2)
                    eval_csv = gr.Textbox(label="Eval CSV Path", placeholder="Auto-filled after loading", scale=2)
                with gr.Row():
                    lang = gr.Dropdown(
                        label="🌍 Dataset Language",
                        value="amh",
                        choices=["ar", "cs", "de", "en", "es", "fr", "hu", "it", "ja", "ko", "nl", "pl", "pt", "ru", "tr", "zh-cn", "amh", "am"],
                        info="Use 'amh' for Amharic (auto-loaded from dataset)",
                        scale=2
                    )
                    version = gr.Dropdown(
                        label="XTTS Version",
                        value="v2.0.2",
                        choices=["v2.0.3", "v2.0.2", "v2.0.1", "v2.0.0", "main"],
                        scale=1
                    )
            
            with gr.Group():
                gr.Markdown("### ⚙️ **Training Parameters**")
                custom_model = gr.Textbox(
                    label="Custom Model Path (Optional)",
                    placeholder="Leave blank to use base model or enter URL/path",
                    value="",
                )
                with gr.Row():
                    num_epochs = gr.Slider(
                        label="Epochs",
                        minimum=1, maximum=100, step=1, value=args.num_epochs,
                        info="15-20 recommended for 30-40hr datasets"
                    )
                    batch_size = gr.Slider(
                        label="Batch Size",
                        minimum=2, maximum=512, step=1, value=args.batch_size,
                        info="2 = safe for 15GB GPU, 4-8 for 24GB+"
                    )
                with gr.Row():
                    grad_acumm = gr.Slider(
                        label="Grad Accumulation",
                        minimum=1, maximum=128, step=1, value=args.grad_acumm,
                        info="8 = effective batch of 16 (recommended)"
                    )
                    max_audio_length = gr.Slider(
                        label="Max Audio (sec)",
                        minimum=2, maximum=20, step=1, value=args.max_audio_length,
                        info="11 sec optimal for Amharic (256KB samples)"
                    )
                
                with gr.Row():
                    save_step = gr.Slider(
                        label="Checkpoint Save Frequency (steps)",
                        minimum=100, maximum=5000, step=100, value=1000,
                        info="1000 steps recommended for 30-40hr data"
                    )
                    save_n_checkpoints = gr.Slider(
                        label="Keep N Checkpoints",
                        minimum=1, maximum=10, step=1, value=3,
                        info="Keep top 3 checkpoints (saves disk space)"
                    )
                
                clear_train_data = gr.Dropdown(
                    label="Cleanup After Training",
                    value="none",
                    choices=["none", "run", "dataset", "all"],
                    info="Delete training data after optimization"
                )
            
            with gr.Group():
                gr.Markdown("### 🛡️ **Anti-Overfitting Configuration** (Prevent Memorization)")
                gr.Markdown("""
                **Critical for ~60hr datasets**: These settings prevent your model from memorizing the training data (overfitting after 2 epochs).
                Recommended for Amharic and limited-diversity datasets.
                """)
                
                with gr.Accordion("🔒 Layer Freezing (Recommended)", open=True):
                    gr.Markdown("""
                    **Freeze layers** to prevent the model from changing too much. Only train the essential parts.
                    """)
                    with gr.Row():
                        freeze_encoder = gr.Checkbox(
                            label="Freeze Encoder (mel_encoder + dvae)",
                            value=True,
                            info="✅ Recommended: Freeze audio encoder to preserve pretrained features",
                            scale=2
                        )
                        freeze_n_gpt_layers = gr.Slider(
                            label="Freeze First N GPT Layers",
                            minimum=0, maximum=30, step=1, value=28,
                            info="28 = freeze 28/30 layers (only train last 2) - HIGHLY recommended for 60hr",
                            scale=3
                        )
                    gr.Markdown("""
                    💡 **Why freeze?** XTTS v2 has 30 GPT layers. Freezing 28 means only the last 2 layers adapt to your voice,
                    while preserving the pretrained language knowledge. This is **critical** for preventing overfitting with limited speaker diversity.
                    """)
                
                with gr.Accordion("📉 Learning Rate & Regularization", open=True):
                    gr.Markdown("""
                    **Lower learning rate** = slower, more careful updates. **Higher weight decay** = stronger regularization.
                    """)
                    with gr.Row():
                        learning_rate_custom = gr.Number(
                            label="Learning Rate Override",
                            value=2e-6,
                            precision=0,
                            info="2e-6 recommended (5x lower than default). Use 1e-6 for even safer training.",
                            scale=1
                        )
                        weight_decay_custom = gr.Slider(
                            label="Weight Decay (Regularization)",
                            minimum=0.01, maximum=0.2, step=0.01, value=0.05,
                            info="0.05 = 5x stronger than default (prevents overfitting)",
                            scale=1
                        )
                    gr.Markdown("""
                    💡 **For your case**: LR=2e-6 and WD=0.05 will make training much more conservative,
                    preventing the rapid overfitting you experienced in 2 epochs.
                    """)
                
                with gr.Accordion("⏹️ Early Stopping", open=True):
                    gr.Markdown("""
                    **Stop automatically** when validation loss stops improving (prevents overfitting).
                    """)
                    enable_early_stopping = gr.Checkbox(
                        label="Enable Early Stopping",
                        value=True,
                        info="✅ Strongly recommended: Auto-stop when model starts overfitting"
                    )
                    early_stop_patience = gr.Slider(
                        label="Patience (epochs)",
                        minimum=1, maximum=10, step=1, value=2,
                        info="Stop if eval loss doesn't improve for 2 epochs"
                    )
                    gr.Markdown("""
                    💡 **Your scenario**: With patience=2, training will stop automatically after 2 epochs of no improvement,
                    catching overfitting before it ruins your model.
                    """)
            
            with gr.Group():
                gr.Markdown("### ⚡ **Training Optimizations** (Speed & Memory)")
                gr.Markdown("_Enable optimizations for faster training with less memory_")
                with gr.Row():
                    enable_grad_checkpoint = gr.Checkbox(
                        label="Gradient Checkpointing",
                        value=True,
                        info="✅ Enabled: 20-30% memory reduction"
                    )
                    enable_sdpa = gr.Checkbox(
                        label="Fast Attention (SDPA)",
                        value=True,
                        info="✅ Enabled: 1.3-1.5x speed + 30-40% memory saving"
                    )
                    enable_mixed_precision = gr.Checkbox(
                        label="Mixed Precision (FP16/BF16)",
                        value=False,  # Will auto-enable if modern GPU detected
                        info="✅ AUTO-ENABLED on modern GPUs (RTX 30xx/40xx). 2x faster training!"
                    )
            
            
            with gr.Group():
                gr.Markdown("### 🌟 **Advanced Training Enhancements** (Quality & Stability)")
                gr.Markdown("""
                **Latest techniques** for better model quality and stable training (all risk-free, proven effective).
                """)
                
                with gr.Row():
                    use_ema = gr.Checkbox(
                        label="🌟 EMA (Exponential Moving Average)",
                        value=True,
                        info="✅ RECOMMENDED: Smoothed model weights = 5-10% better quality",
                        scale=2
                    )
                    lr_warmup_steps = gr.Slider(
                        label="LR Warmup Steps",
                        minimum=0, maximum=2000, step=100, value=500,
                        info="500 steps = gradual LR increase (prevents initial instability)",
                        scale=2
                    )
                
                gr.Markdown("""
                💡 **EMA**: Maintains a "smoothed" version of your model that's often 5-10% better quality. Used in all modern TTS systems.  
                💡 **LR Warmup**: Gradually increases learning rate from 0 → target over first 500 steps. Essential for stable training with frozen layers.
                """)
            
            with gr.Group():
                gr.Markdown("### 🔄 **Resume Training** (Continue from Checkpoint)")
                gr.Markdown("""
                💡 **Continue training** from a previous checkpoint instead of starting fresh.
                Useful for adding more epochs or resuming interrupted training.
                """)
                with gr.Row():
                    resume_from_checkpoint = gr.Checkbox(
                        label="Resume from Checkpoint",
                        value=False,
                        info="Continue training instead of starting fresh",
                        scale=1
                    )
                    checkpoint_selector = gr.Dropdown(
                        label="Select Checkpoint",
                        choices=[("Click 🔄 to load checkpoints", "")],
                        value="",
                        interactive=True,
                        info="Checkpoints appear here after training starts",
                        scale=2
                    )
                    refresh_checkpoints_btn = gr.Button(
                        value="🔄",
                        size="sm",
                        scale=0,
                        min_width=40
                    )
                gr.Markdown("""
                🔍 **Tip**: Checkpoints are saved in `output/run/training/` during training.  
                Click 🔄 to refresh the list after training completes.
                """)
            
            with gr.Group():
                gr.Markdown("### 🇪🇹 **Amharic G2P Options** (for 'amh' language)")
                gr.Markdown("""
                ⚠️ **Important**: Enable this if your dataset contains **raw Amharic text** (not phonemes).
                The training will convert text to phonemes automatically.
                """)
                with gr.Row():
                    enable_amharic_g2p = gr.Checkbox(
                        label="Enable G2P for Training",
                        value=True,  # Default to True for Amharic!
                        info="Convert Amharic → IPA phonemes during training",
                        scale=2
                    )
                    g2p_backend_train = gr.Dropdown(
                        label="G2P Backend",
                        value="rule_based",  # rule_based is most reliable
                        choices=["rule_based", "transphone", "epitran"],
                        info="rule_based = offline, no dependencies",
                        scale=1
                    )
                
                # Add vocab info display
                vocab_info_display = gr.Textbox(
                    label="📚 Vocabulary Information",
                    value="Click 'Check Vocab' to verify vocabulary consistency",
                    interactive=False,
                    lines=3
                )
                check_vocab_btn = gr.Button(value="🔍 Check Vocab & Dataset", size="sm", variant="secondary")
            
            with gr.Group():
                gr.Markdown("### 🚀 **Execute Training**")
                progress_train = gr.Label(label="Status", value="Ready")
                with gr.Row():
                    train_btn = gr.Button(value="▶️ Step 2 - Train Model", variant="primary", size="lg", scale=2)
                    optimize_model_btn = gr.Button(value="⚡ Step 2.5 - Optimize Model", variant="primary", size="lg", scale=1)
            
            with gr.Group():
                gr.Markdown("### 📦 **Checkpoint Manager** (Monitor & Cleanup)")
                gr.Markdown("_Manage training checkpoints: monitor, analyze, and cleanup_")
                
                with gr.Row():
                    refresh_train_checkpoints_btn = gr.Button(value="🔄 Refresh Checkpoints", variant="secondary", scale=1, size="sm")
                    analyze_train_overfitting_btn = gr.Button(value="📊 Analyze Training", variant="secondary", scale=1, size="sm")
                    cleanup_checkpoints_btn = gr.Button(value="🗑️ Cleanup Old", variant="secondary", scale=1, size="sm")
                
                train_checkpoint_display = gr.Textbox(
                    label="Checkpoint Status",
                    lines=15,
                    interactive=False,
                    placeholder="Click 'Refresh Checkpoints' to see saved checkpoints from your training...",
                    show_label=False
                )
                
                with gr.Accordion("⚙️ Checkpoint Actions", open=False):
                    gr.Markdown("**Select checkpoints to manage:**")
                    
                    train_checkpoint_selector = gr.CheckboxGroup(
                        label="Available Checkpoints",
                        choices=[],
                        value=[],
                        interactive=True
                    )
                    
                    with gr.Row():
                        delete_selected_btn = gr.Button(value="🗑️ Delete Selected", variant="stop", size="sm")
                        copy_to_ready_btn = gr.Button(value="📋 Copy Best to Ready", variant="primary", size="sm")
                        export_analysis_btn = gr.Button(value="📄 Export Analysis", variant="secondary", size="sm")
                    
                    checkpoint_action_status = gr.Textbox(
                        label="Action Status",
                        lines=3,
                        interactive=False
                    )
            
            import os
            import shutil
            from pathlib import Path
            import traceback
            
            def load_available_checkpoints(output_path):
                """Load list of available checkpoints from training directory"""
                try:
                    training_dir = Path(output_path) / "run" / "training"
                    if not training_dir.exists():
                        return []
                    
                    # Find all checkpoint files RECURSIVELY in subdirectories
                    checkpoints = list(training_dir.rglob("checkpoint_*.pth"))
                    checkpoints += list(training_dir.rglob("best_model*.pth"))  # best_model*.pth to match best_model_4141.pth etc
                    
                    if not checkpoints:
                        print(f"No checkpoints found in {training_dir}")
                        # Debug: list what's actually there
                        for subdir in training_dir.glob("*"):
                            if subdir.is_dir():
                                print(f"  Found training run: {subdir.name}")
                                ckpts_in_subdir = list(subdir.glob("*.pth"))
                                print(f"    Checkpoints: {[c.name for c in ckpts_in_subdir]}")
                        return []
                    
                    # Sort by modification time (newest first)
                    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    
                    # Return relative paths for display
                    return [str(cp.relative_to(Path(output_path))) for cp in checkpoints]
                except Exception as e:
                    print(f"Error loading checkpoints: {e}")
                    import traceback
                    traceback.print_exc()
                    return []
            
            def refresh_checkpoint_list(output_path):
                """Refresh the checkpoint dropdown"""
                checkpoints = load_available_checkpoints(output_path)
                if not checkpoints:
                    # Return helpful message when no checkpoints exist
                    return gr.Dropdown(
                        choices=[("⚠️ No checkpoints found - Complete training first", "")],
                        value="",
                        info="Checkpoints are saved during training to: output/run/training/"
                    )
                return gr.Dropdown(
                    choices=checkpoints,
                    value=checkpoints[0] if checkpoints else None,
                    info=f"Found {len(checkpoints)} checkpoint(s) - Select one to resume training"
                )
            
            def check_vocab_and_dataset(output_path, train_csv):
                """Check vocab consistency and dataset info"""
                try:
                    import json
                    import pandas as pd
                    
                    result = []
                    result.append("✅ **Vocabulary & Dataset Check**")
                    result.append("="*50)
                    
                    # Check vocab files
                    ready_dir = Path(output_path) / "ready"
                    vocab_files = list(ready_dir.glob("vocab*.json"))
                    
                    if not vocab_files:
                        return "❌ No vocab files found in ready/ directory!\nPlease create dataset first."
                    
                    vocab_sizes = {}
                    for vf in vocab_files:
                        try:
                            with open(vf, 'r', encoding='utf-8') as f:
                                vocab_data = json.load(f)
                                size = len(vocab_data['model']['vocab'])
                                vocab_sizes[vf.name] = size
                        except:
                            vocab_sizes[vf.name] = "Error reading"
                    
                    result.append("\n📚 **Vocabulary Files:**")
                    for name, size in vocab_sizes.items():
                        result.append(f"  • {name}: {size} tokens")
                    
                    # Check if sizes match
                    unique_sizes = set(s for s in vocab_sizes.values() if isinstance(s, int))
                    if len(unique_sizes) == 1:
                        result.append(f"\n✅ All vocab files have matching size: {unique_sizes.pop()} tokens")
                    else:
                        result.append(f"\n⚠️  WARNING: Vocab files have different sizes! This will cause errors!")
                        result.append(f"   Sizes found: {unique_sizes}")
                    
                    # Check dataset
                    if train_csv and os.path.exists(train_csv):
                        try:
                            df = pd.read_csv(train_csv, sep='|')
                            result.append(f"\n🎙️ **Dataset Information:**")
                            result.append(f"  • Training samples: {len(df)}")
                            
                            # Check if text is Amharic or phonemes
                            if len(df) > 0:
                                sample_text = df.iloc[0]['text'] if 'text' in df.columns else ""
                                has_amharic = any(ord(c) >= 0x1200 and ord(c) <= 0x137F for c in sample_text[:50])
                                has_ipa = any(c in sample_text for c in ['ə', 'ɨ', 'ʔ', 'ː'])
                                
                                if has_amharic and not has_ipa:
                                    result.append(f"  • Text format: Raw Amharic characters")
                                    result.append(f"    ➡️ ENABLE G2P for training!")
                                elif has_ipa:
                                    result.append(f"  • Text format: IPA phonemes")
                                    result.append(f"    ➡️ G2P already applied, disable for training")
                                else:
                                    result.append(f"  • Text format: Unknown/Other")
                        except Exception as e:
                            result.append(f"\n⚠️  Error reading dataset: {e}")
                    else:
                        result.append(f"\n⚠️  Train CSV not found or not specified")
                    
                    result.append("\n" + "="*50)
                    result.append("💡 **Recommendation:**")
                    if len(unique_sizes) > 1:
                        result.append("  ⚠️  Fix vocab mismatch before training!")
                        result.append("  ➡️ Use: python create_7536_vocab.py (if needed)")
                    else:
                        result.append("  ✅ Ready to train with current configuration!")
                    
                    return "\n".join(result)
                    
                except Exception as e:
                    return f"❌ Error checking vocab: {str(e)}"
            
            def train_model(custom_model, version, language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length, save_step=1000, save_n_checkpoints=1, enable_grad_checkpoint=False, enable_sdpa=False, enable_mixed_precision=False, enable_amharic_g2p=False, g2p_backend_train="transphone", resume_from_checkpoint_flag=False, checkpoint_path=None, freeze_encoder_flag=True, freeze_n_gpt_layers_val=0, learning_rate_val=None, weight_decay_val=None, enable_early_stop=False, early_stop_patience_val=None, use_ema_flag=True, lr_warmup_steps_val=500):
                clear_gpu_cache()
                
                # Strip whitespace from paths to prevent accidental spaces
                train_csv = train_csv.strip() if train_csv else ""
                eval_csv = eval_csv.strip() if eval_csv else ""
                custom_model = custom_model.strip() if custom_model else ""
          
                # Check if `custom_model` is a URL and download it if true.
                if custom_model.startswith("http"):
                    print("Downloading custom model from URL...")
                    custom_model = download_file(custom_model, "custom_model.pth")
                    if not custom_model:
                        return "Failed to download the custom model.", "", "", "", ""
            
                run_dir = Path(output_path) / "run"
            
                # Handle checkpoint resumption
                restore_checkpoint_path = None
                if resume_from_checkpoint_flag and checkpoint_path:
                    # Construct full path
                    restore_checkpoint_path = Path(output_path) / checkpoint_path
                    if not restore_checkpoint_path.exists():
                        return f"❌ Checkpoint not found: {restore_checkpoint_path}", "", "", "", "", ""
                    print(f"🔄 Resuming training from checkpoint: {restore_checkpoint_path}")
                else:
                    # Remove train dir only if starting fresh
                    if run_dir.exists():
                        shutil.rmtree(run_dir)
                
                # Check if the dataset language matches the language you specified 
                lang_file_path = Path(output_path) / "dataset" / "lang.txt"
            
                # Check if lang.txt already exists and contains a different language
                current_language = None
                if lang_file_path.exists():
                    with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
                        current_language = existing_lang_file.read().strip()
                        if current_language != language:
                            print("The language that was prepared for the dataset does not match the specified language. Change the language to the one specified in the dataset")
                            language = current_language
                        
                if not train_csv or not eval_csv:
                    return "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !", "", "", "", ""
                # Configure Amharic G2P for training (accept both 'am' and 'amh')
                use_amharic_g2p = enable_amharic_g2p and language in ["am", "amh"]
                if use_amharic_g2p:
                    print(f"Amharic G2P enabled with backend: {g2p_backend_train}")
                    # Set backend order based on user selection
                    try:
                        from amharic_tts.config.amharic_config import G2PBackend
                        # This will be used in the training pipeline
                    except ImportError:
                        print("Warning: Amharic G2P module not available")
                        use_amharic_g2p = False
                
                try:
                    # convert seconds to waveform frames
                    max_audio_length = int(max_audio_length * 22050)

# Normalize language code for XTTS internals
                    language_norm = normalize_xtts_lang(language)

                    # Use restore checkpoint path if resuming, otherwise use custom_model
                    model_to_load = str(restore_checkpoint_path) if restore_checkpoint_path else custom_model

                    speaker_xtts_path, config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(
                        model_to_load, version, language_norm, num_epochs, batch_size, grad_acumm,
                        train_csv, eval_csv, output_path=output_path, max_audio_length=max_audio_length,
                        save_step=save_step, save_n_checkpoints=save_n_checkpoints,
                        use_amharic_g2p=use_amharic_g2p,
                        enable_grad_checkpoint=enable_grad_checkpoint,
                        enable_sdpa=enable_sdpa,
                        enable_mixed_precision=enable_mixed_precision,
                        freeze_encoder=freeze_encoder_flag,
                        freeze_first_n_gpt_layers=int(freeze_n_gpt_layers_val) if freeze_n_gpt_layers_val else 0,
                        learning_rate_override=float(learning_rate_val) if learning_rate_val else None,
                        weight_decay_override=float(weight_decay_val) if weight_decay_val else None,
                        early_stopping_patience=int(early_stop_patience_val) if (enable_early_stop and early_stop_patience_val) else None,
                        use_ema=bool(use_ema_flag),
                        lr_warmup_steps=int(lr_warmup_steps_val) if lr_warmup_steps_val else 500
                    )
                except:
                    traceback.print_exc()
                    error = traceback.format_exc()
                    return f"The training was interrupted due to an error !! Please check the console to check the full error message! \n Error summary: {error}", "", "", "", ""
            
                ready_dir = Path(output_path) / "ready"
            
                ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
            
                shutil.copy(ft_xtts_checkpoint, ready_dir / "unoptimize_model.pth")
            
                ft_xtts_checkpoint = os.path.join(ready_dir, "unoptimize_model.pth")
            
                # Move reference audio to output folder and rename it
                speaker_reference_path = Path(speaker_wav)
                speaker_reference_new_path = ready_dir / "reference.wav"
                shutil.copy(speaker_reference_path, speaker_reference_new_path)
            
                print("Model training done!")

                # Write training metadata for inference alignment
                try:
                    import json as _json
                    amharic_meta = {
                        "g2p_training_enabled": bool(enable_amharic_g2p and language in ["am", "amh"]),
                        "g2p_backend": g2p_backend_train if (enable_amharic_g2p and language in ["am", "amh"]) else None,
                        # When G2P is applied, training pipeline switches tokenizer language to 'en'
                        "effective_language": "en" if (enable_amharic_g2p and language in ["am", "amh"]) else normalize_xtts_lang(language),
                        "vocab_used": "extended" if (ready_dir / "vocab_extended_amharic.json").exists() or (ready_dir / "vocab_extended.json").exists() else "standard",
                    }
                    meta = {"amharic": amharic_meta}
                    with open(ready_dir / "training_meta.json", "w", encoding="utf-8") as _f:
                        _json.dump(meta, _f, indent=2, ensure_ascii=False)
                    print(" > Saved training metadata to ready/training_meta.json")
                except Exception as _e:
                    print(f" > Warning: Could not write training metadata: {_e}")

                return "Model training done!", config_path, vocab_file, ft_xtts_checkpoint, speaker_xtts_path, speaker_reference_new_path

            def optimize_model(out_path, clear_train_data):
                # print(out_path)
                out_path = Path(out_path)  # Ensure that out_path is a Path object.
            
                ready_dir = out_path / "ready"
                run_dir = out_path / "run"
                dataset_dir = out_path / "dataset"
            
                # Clear specified training data directories.
                if clear_train_data in {"run", "all"} and run_dir.exists():
                    try:
                        shutil.rmtree(run_dir)
                    except PermissionError as e:
                        print(f"An error occurred while deleting {run_dir}: {e}")
            
                if clear_train_data in {"dataset", "all"} and dataset_dir.exists():
                    try:
                        shutil.rmtree(dataset_dir)
                    except PermissionError as e:
                        print(f"An error occurred while deleting {dataset_dir}: {e}")
            
                # Get full path to model
                model_path = ready_dir / "unoptimize_model.pth"

                if not model_path.is_file():
                    return "Unoptimized model not found in ready folder", ""
            
                # Load the checkpoint and remove unnecessary parts.
                checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
                del checkpoint["optimizer"]

                for key in list(checkpoint["model"].keys()):
                    if "dvae" in key:
                        del checkpoint["model"][key]

                # Make sure out_path is a Path object or convert it to Path
                os.remove(model_path)

                  # Save the optimized model.
                optimized_model_file_name="model.pth"
                optimized_model=ready_dir/optimized_model_file_name
            
                torch.save(checkpoint, optimized_model)
                ft_xtts_checkpoint=str(optimized_model)

                clear_gpu_cache()
        
                return f"Model optimized and saved at {ft_xtts_checkpoint}!", ft_xtts_checkpoint

            def load_params(out_path):
                path_output = Path(out_path)
                
                dataset_path = path_output / "dataset"

                if not dataset_path.exists():
                    return "The output folder does not exist!", "", ""

                eval_train = dataset_path / "metadata_train.csv"
                eval_csv = dataset_path / "metadata_eval.csv"

                # Write the target language to lang.txt in the output directory
                lang_file_path =  dataset_path / "lang.txt"

                # Check if lang.txt already exists and contains a different language
                current_language = None
                if os.path.exists(lang_file_path):
                    with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
                        current_language = existing_lang_file.read().strip()

                clear_gpu_cache()

                print(current_language)
                return "The data has been updated", eval_train, eval_csv, current_language

        with gr.Tab("🎤 Inference"):
            gr.Markdown("## Text-to-Speech Generation")
            
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 📥 **Model Loading**")
                        load_params_tts_btn = gr.Button(value="📂 Load from Output Folder", variant="secondary")
                        xtts_checkpoint = gr.Textbox(label="Checkpoint Path", placeholder="Auto-filled")
                        xtts_config = gr.Textbox(label="Config Path", placeholder="Auto-filled")
                        xtts_vocab = gr.Textbox(label="Vocab Path", placeholder="Auto-filled")
                        xtts_speaker = gr.Textbox(label="Speaker Path", placeholder="Auto-filled")
                        progress_load = gr.Label(label="Status", value="Not Loaded")
                        load_btn = gr.Button(value="▶️ Step 3 - Load Model", variant="primary", size="lg")
                    
                    with gr.Group():
                        gr.Markdown("### 🔄 **Checkpoint Selection** (Advanced)")
                        gr.Markdown("_Select a specific training checkpoint instead of using the default best model_")
                        with gr.Row():
                            refresh_checkpoints_btn = gr.Button(value="🔍 Scan Checkpoints", variant="secondary", scale=1, size="sm")
                            analyze_overfitting_btn = gr.Button(value="📊 Analyze Overfitting", variant="secondary", scale=1, size="sm")
                        
                        checkpoint_selector = gr.Dropdown(
                            label="Available Checkpoints",
                            choices=[("Click 'Scan Checkpoints' to load", "")],
                            value="",
                            interactive=True,
                            info="Choose a checkpoint from your latest training run"
                        )
                        
                        checkpoint_info_display = gr.Textbox(
                            label="Checkpoint Information",
                            lines=15,
                            interactive=False,
                            placeholder="Click 'Scan Checkpoints' to see available checkpoints from your latest training",
                            show_label=False
                        )
                        
                        use_selected_checkpoint_btn = gr.Button(
                            value="✅ Use Selected Checkpoint",
                            variant="primary",
                            size="lg"
                        )
                        
                        gr.Markdown("---")
                        gr.Markdown("### 🎤 **Quick Inference Test** (Compare Checkpoints)")
                        gr.Markdown("_Test speech generation directly from any checkpoint without loading_")
                        
                        checkpoint_test_text = gr.Textbox(
                            label="Test Text",
                            placeholder="Enter text to test with selected checkpoint...",
                            lines=2,
                            value="This is a test of the checkpoint."
                        )
                        
                        with gr.Row():
                            checkpoint_test_language = gr.Dropdown(
                                label="Language",
                                value="en",
                                choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "hu", "ko", "ja", "am", "amh"],
                                scale=1
                            )
                            checkpoint_use_g2p = gr.Checkbox(
                                label="🇪🇹 Use G2P (Amharic)",
                                value=True,
                                scale=1
                            )
                        
                        test_checkpoint_btn = gr.Button(
                            value="🎙️ Test Selected Checkpoint",
                            variant="secondary",
                            size="lg"
                        )
                        
                        checkpoint_test_output = gr.Audio(
                            label="Checkpoint Test Output",
                            type="filepath"
                        )
                        
                        checkpoint_test_status = gr.Textbox(
                            label="Test Status",
                            lines=3,
                            interactive=False
                        )

                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 🎙️ **Generation Settings**")
                        speaker_reference_audio = gr.Textbox(label="Reference Audio Path", placeholder="Auto-filled")
                        tts_language = gr.Dropdown(
                            label="Language",
                            value="en",
                            choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "hu", "ko", "ja", "am", "amh"],
                            info="Use 'am' or 'amh' for Amharic"
                        )
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            placeholder="Enter the text you want to convert to speech...",
                            lines=4,
                            value="This model sounds really good and above all, it's reasonably fast.",
                        )
                        
                        # G2P option for Amharic
                        use_g2p_inference = gr.Checkbox(
                            label="🇪🇹 Enable Amharic G2P",
                            value=True,
                            info="Convert Amharic text to phonemes (required if model trained with G2P)"
                        )
                        g2p_backend_infer = gr.Dropdown(
                            label="G2P Backend (Inference)",
                            value="auto",
                            choices=["auto", "transphone", "epitran", "rule_based"],
                            info="Auto uses the backend from training_meta.json if available; otherwise transphone"
                        )
                        
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            with gr.Row():
                                temperature = gr.Slider(label="Temperature", minimum=0, maximum=1, step=0.05, value=0.75)
                                length_penalty = gr.Slider(label="Length Penalty", minimum=-10.0, maximum=10.0, step=0.5, value=1)
                            with gr.Row():
                                repetition_penalty = gr.Slider(label="Repetition Penalty", minimum=1, maximum=10, step=0.5, value=5)
                                top_k = gr.Slider(label="Top K", minimum=1, maximum=100, step=1, value=50)
                            with gr.Row():
                                top_p = gr.Slider(label="Top P", minimum=0, maximum=1, step=0.05, value=0.85)
                                sentence_split = gr.Checkbox(label="Text Splitting", value=True)
                            use_config = gr.Checkbox(label="Use Config Settings", value=False, info="Override above with config values")
                        
                        tts_btn = gr.Button(value="▶️ Step 4 - Generate Speech", variant="primary", size="lg")
                    
                    with gr.Group():
                        gr.Markdown("### 📦 **Export**")
                        with gr.Row():
                            model_download_btn = gr.Button("📥 Download Model", variant="secondary", scale=1)
                            dataset_download_btn = gr.Button("📥 Download Dataset", variant="secondary", scale=1)
                        model_zip_file = gr.File(label="Model ZIP", interactive=False, visible=False)
                        dataset_zip_file = gr.File(label="Dataset ZIP", interactive=False, visible=False)

                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 🔊 **Output**")
                        progress_gen = gr.Label(label="Status", value="Ready")
                        tts_output_audio = gr.Audio(label="Generated Audio", type="filepath")
                        reference_audio = gr.Audio(label="Reference Audio Used", type="filepath")

            prompt_compute_btn.click(
                fn=preprocess_dataset,
                inputs=[
                    upload_file,
                    audio_folder_path,
                    lang,
                    whisper_model,
                    out_path,
                    train_csv,
                    eval_csv,
                    use_amharic_g2p_preprocessing,
                    g2p_backend_selection,
                ],
                outputs=[
                    progress_data,
                    train_csv,
                    eval_csv,
                ],
            )
            
            # Advanced features button handlers
            process_srt_btn.click(
                fn=process_srt_media,
                inputs=[
                    srt_files,
                    media_files,
                    lang,
                    out_path,
                    srt_batch_mode,
                    use_vad_refinement,  # VAD enable/disable
                    srt_buffer_padding,  # Audio padding (seconds) to prevent cutoffs
                    vad_threshold,  # VAD threshold
                    vad_min_speech_duration,  # Min speech duration
                    vad_min_silence_duration,  # Min silence duration
                    vad_speech_pad,  # Speech padding
                    use_enhanced_vad_option,  # Enhanced VAD with quality metrics
                    amharic_mode_option,  # Amharic-specific optimizations
                    srt_incremental_mode,  # Incremental mode
                    srt_check_duplicates,  # Skip duplicates
                ],
                outputs=[srt_status],
            )
            
            download_youtube_btn.click(
                fn=download_youtube_video,
                inputs=[
                    youtube_url,
                    youtube_transcript_lang,
                    lang,
                    out_path,
                    youtube_batch_mode,
                    youtube_incremental_mode,  # Incremental mode
                    youtube_check_duplicates,  # Skip duplicates
                    youtube_cookies_file,  # Cookies file path
                    youtube_cookies_browser,  # Browser for cookie import
                    youtube_proxy,  # Proxy URL
                    youtube_user_agent,  # Custom User-Agent
                    youtube_buffer_padding,  # Audio padding (seconds) to prevent cutoffs
                    youtube_use_vad,  # VAD enable/disable
                    youtube_vad_threshold,  # VAD threshold
                    youtube_vad_min_speech,  # Min speech duration
                    youtube_vad_min_silence,  # Min silence duration
                    youtube_vad_speech_pad,  # Speech padding
                    youtube_use_enhanced_vad,  # Enhanced VAD
                    youtube_amharic_mode,  # Amharic mode
                    youtube_remove_bg,  # Background music removal enable/disable
                    youtube_bg_quality,  # Background removal quality
                    youtube_bg_model,  # Background removal model
                ],
                outputs=[youtube_status],
            )
            
            slice_audio_btn.click(
                fn=slice_audio_file,
                inputs=[
                    slicer_audio_file,
                    slicer_threshold_db,
                    slicer_min_length,
                    slicer_min_interval,
                    slicer_max_sil_kept,
                    slicer_auto_transcribe,
                    whisper_model,
                    lang,
                    out_path,
                ],
                outputs=[slicer_status],
            )
            
            # History viewer handlers
            refresh_history_btn.click(
                fn=show_dataset_history,
                inputs=[out_path],
                outputs=[history_display],
            )
            
            clear_history_btn.click(
                fn=clear_dataset_history,
                inputs=[out_path],
                outputs=[history_display],
            )
            
            # Dataset statistics handler
            calculate_stats_btn.click(
                fn=show_dataset_statistics,
                inputs=[out_path],
                outputs=[dataset_stats_display],
            )


            load_params_btn.click(
                fn=load_params,
                inputs=[out_path],
                outputs=[
                    progress_train,
                    train_csv,
                    eval_csv,
                    lang
                ]
            )
            
            check_vocab_btn.click(
                fn=check_vocab_and_dataset,
                inputs=[out_path, train_csv],
                outputs=[vocab_info_display]
            )
            
            # Wire up resume training checkpoint refresh button
            refresh_checkpoints_btn.click(
                fn=refresh_checkpoint_list,
                inputs=[out_path],
                outputs=[checkpoint_selector]  # This is the training tab dropdown at line 1685
            )
            
            # Fine-tuning tab checkpoint manager handlers
            def refresh_training_checkpoints(output_path):
                """Refresh and display checkpoints from training folder"""
                try:
                    run_dir, checkpoints = checkpoint_manager.get_latest_training_run_checkpoints(output_path)
                    
                    if not checkpoints:
                        return {
                            train_checkpoint_display: "❌ No checkpoints found.\n\nComplete a training run first, or checkpoints will appear here after training starts.",
                            train_checkpoint_selector: gr.CheckboxGroup(choices=[], value=[])
                        }
                    
                    # Get recommendation
                    recommended = checkpoint_manager.recommend_best_checkpoint(checkpoints)
                    
                    # Build display text
                    lines = []
                    lines.append("📦 **Training Checkpoints**")
                    lines.append("=" * 70)
                    lines.append("")
                    
                    if run_dir:
                        lines.append(f"Training Run: {run_dir.name}")
                        lines.append("")
                    
                    if recommended:
                        lines.append(f"🏆 **BEST CHECKPOINT**: {recommended.display_name()}")
                        lines.append(f"   Reason: {'Lowest eval loss' if recommended.eval_loss else 'Early checkpoint (avoids overfitting)'}")
                        lines.append("")
                    
                    lines.append(f"**Total Checkpoints**: {len(checkpoints)}")
                    lines.append("")
                    
                    # Group by status
                    early = [c for c in checkpoints if c.epoch is not None and c.epoch <= 5]
                    mid = [c for c in checkpoints if c.epoch is not None and 5 < c.epoch <= 20]
                    late = [c for c in checkpoints if c.epoch is not None and c.epoch > 20]
                    
                    if early:
                        lines.append(f"✅ **Early Checkpoints** (Epoch 0-5): {len(early)} - Recommended")
                    if mid:
                        lines.append(f"⚠️ **Mid Checkpoints** (Epoch 6-20): {len(mid)} - Use with caution")
                    if late:
                        lines.append(f"❌ **Late Checkpoints** (Epoch 21+): {len(late)} - Likely overfitted")
                    
                    lines.append("")
                    lines.append("**All Checkpoints:**")
                    lines.append("")
                    
                    for i, ckpt in enumerate(checkpoints, 1):
                        prefix = "  ➡" if ckpt == recommended else "   "
                        lines.append(f"{prefix} {i}. {ckpt.display_name()}")
                    
                    lines.append("")
                    lines.append("=" * 70)
                    lines.append("💡 **Actions**: Use buttons above to analyze, cleanup, or manage checkpoints")
                    
                    display_text = "\n".join(lines)
                    
                    # Generate checkbox choices (display_name, path)
                    checkbox_choices = [(ckpt.display_name(), ckpt.path) for ckpt in checkpoints]
                    
                    return {
                        train_checkpoint_display: display_text,
                        train_checkpoint_selector: gr.CheckboxGroup(choices=checkbox_choices, value=[])
                    }
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return {
                        train_checkpoint_display: f"❌ Error loading checkpoints: {str(e)}",
                        train_checkpoint_selector: gr.CheckboxGroup(choices=[], value=[])
                    }
            
            def analyze_training_checkpoints(output_path):
                """Analyze checkpoints for overfitting"""
                try:
                    run_dir, checkpoints = checkpoint_manager.get_latest_training_run_checkpoints(output_path)
                    
                    if not checkpoints:
                        return "❌ No checkpoints available for analysis."
                    
                    analysis = checkpoint_manager.analyze_checkpoints_for_overfitting(checkpoints)
                    
                    lines = []
                    lines.append("📊 **Training Analysis Report**")
                    lines.append("=" * 70)
                    lines.append("")
                    
                    lines.append("**Overfitting Status:**")
                    lines.append(analysis["warning_message"])
                    lines.append("")
                    
                    if analysis["eval_loss_trend"]:
                        lines.append("**Evaluation Loss Progression:**")
                        lines.append("")
                        for item in analysis["eval_loss_trend"]:
                            epoch = item.get('epoch', '?')
                            step = item.get('step', '?')
                            loss = item.get('eval_loss', 0)
                            lines.append(f"  Epoch {epoch:2} | Step {step:5} | Loss: {loss:.4f}")
                        lines.append("")
                    
                    if analysis["safe_checkpoint"]:
                        lines.append("**Recommended Safe Checkpoint:**")
                        lines.append(f"  🏆 {analysis['safe_checkpoint'].display_name()}")
                        lines.append("")
                    
                    lines.append("**Training Statistics:**")
                    lines.append(f"  Total Checkpoints: {len(checkpoints)}")
                    checkpoints_with_loss = [c for c in checkpoints if c.eval_loss is not None]
                    if checkpoints_with_loss:
                        lines.append(f"  With Eval Loss: {len(checkpoints_with_loss)}")
                        lines.append(f"  Best Loss: {min(c.eval_loss for c in checkpoints_with_loss):.4f}")
                        lines.append(f"  Worst Loss: {max(c.eval_loss for c in checkpoints_with_loss):.4f}")
                    
                    lines.append("")
                    lines.append("=" * 70)
                    lines.append("💡 **Recommendation**: Use the checkpoint manager in the Inference tab to test and select the best checkpoint")
                    
                    return "\n".join(lines)
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"❌ Error analyzing checkpoints: {str(e)}"
            
            def cleanup_old_checkpoints(output_path):
                """Remove checkpoints from late epochs (likely overfitted)"""
                try:
                    run_dir, checkpoints = checkpoint_manager.get_latest_training_run_checkpoints(output_path)
                    
                    if not checkpoints:
                        return {
                            train_checkpoint_display: "❌ No checkpoints found to cleanup.",
                            train_checkpoint_selector: gr.CheckboxGroup(choices=[], value=[])
                        }
                    
                    # Find checkpoints to remove (epoch > 20 or high loss)
                    to_remove = []
                    for ckpt in checkpoints:
                        # Keep if best model
                        if ckpt.is_best:
                            continue
                        # Remove if late epoch
                        if ckpt.epoch is not None and ckpt.epoch > 20:
                            to_remove.append(ckpt)
                            continue
                        # Remove if eval loss is high compared to min
                        checkpoints_with_loss = [c for c in checkpoints if c.eval_loss is not None]
                        if checkpoints_with_loss and ckpt.eval_loss:
                            min_loss = min(c.eval_loss for c in checkpoints_with_loss)
                            if ckpt.eval_loss > min_loss * 1.5:  # 50% worse than best
                                to_remove.append(ckpt)
                    
                    if not to_remove:
                        return {
                            train_checkpoint_display: "✅ No cleanup needed!\n\nAll checkpoints are from early epochs or have good evaluation loss.\nYour training looks healthy!",
                            train_checkpoint_selector: gr.CheckboxGroup(choices=[(c.display_name(), c.path) for c in checkpoints], value=[])
                        }
                    
                    # Delete the checkpoints
                    deleted = []
                    failed = []
                    for ckpt in to_remove:
                        try:
                            import os
                            os.remove(ckpt.path)
                            deleted.append(ckpt)
                        except Exception as e:
                            failed.append((ckpt, str(e)))
                    
                    # Build result message
                    lines = []
                    lines.append("🧹 **Cleanup Complete**")
                    lines.append("=" * 70)
                    lines.append("")
                    
                    if deleted:
                        lines.append(f"✅ **Deleted {len(deleted)} checkpoint(s):**")
                        for ckpt in deleted:
                            lines.append(f"  ❌ {ckpt.display_name()}")
                        lines.append("")
                    
                    if failed:
                        lines.append(f"❌ **Failed to delete {len(failed)} checkpoint(s):**")
                        for ckpt, err in failed:
                            lines.append(f"  ⚠️ {ckpt.display_name()} - {err}")
                        lines.append("")
                    
                    # Refresh remaining checkpoints
                    _, remaining = checkpoint_manager.get_latest_training_run_checkpoints(output_path)
                    lines.append(f"**Remaining Checkpoints**: {len(remaining)}")
                    lines.append("")
                    
                    for ckpt in remaining:
                        lines.append(f"  ✅ {ckpt.display_name()}")
                    
                    lines.append("")
                    lines.append("=" * 70)
                    lines.append("💡 **Tip**: Click 'Refresh Checkpoints' to update the list")
                    
                    display_text = "\n".join(lines)
                    checkbox_choices = [(c.display_name(), c.path) for c in remaining]
                    
                    return {
                        train_checkpoint_display: display_text,
                        train_checkpoint_selector: gr.CheckboxGroup(choices=checkbox_choices, value=[])
                    }
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return {
                        train_checkpoint_display: f"❌ Error during cleanup: {str(e)}",
                        train_checkpoint_selector: gr.CheckboxGroup(choices=[], value=[])
                    }
            
            def delete_selected_checkpoints(selected_paths, output_path):
                """Delete user-selected checkpoints"""
                try:
                    if not selected_paths:
                        return "❌ No checkpoints selected. Please select checkpoints to delete."
                    
                    deleted = []
                    failed = []
                    
                    for path in selected_paths:
                        try:
                            import os
                            os.remove(path)
                            deleted.append(os.path.basename(path))
                        except Exception as e:
                            failed.append((os.path.basename(path), str(e)))
                    
                    lines = []
                    if deleted:
                        lines.append(f"✅ Successfully deleted {len(deleted)} checkpoint(s):")
                        for name in deleted:
                            lines.append(f"  ❌ {name}")
                    
                    if failed:
                        lines.append(f"\n❌ Failed to delete {len(failed)} checkpoint(s):")
                        for name, err in failed:
                            lines.append(f"  ⚠️ {name}: {err}")
                    
                    lines.append("\n💡 Click 'Refresh Checkpoints' to update the list")
                    return "\n".join(lines)
                    
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            def copy_best_checkpoint_to_ready(output_path):
                """Copy the best checkpoint to ready folder"""
                try:
                    run_dir, checkpoints = checkpoint_manager.get_latest_training_run_checkpoints(output_path)
                    
                    if not checkpoints:
                        return "❌ No checkpoints available."
                    
                    recommended = checkpoint_manager.recommend_best_checkpoint(checkpoints)
                    
                    if not recommended:
                        return "❌ Could not determine best checkpoint."
                    
                    success, message = checkpoint_manager.copy_checkpoint_to_ready(
                        checkpoint_path=recommended.path,
                        output_path=output_path,
                        as_name="model.pth"
                    )
                    
                    if success:
                        return f"✅ Best checkpoint copied to ready/model.pth!\n\n{recommended.display_name()}\n\n{message}\n\n💡 Go to Inference tab to load and test it!"
                    else:
                        return message
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"❌ Error: {str(e)}"
            
            def export_checkpoint_analysis(output_path):
                """Export checkpoint analysis to file"""
                try:
                    run_dir, checkpoints = checkpoint_manager.get_latest_training_run_checkpoints(output_path)
                    
                    if not checkpoints:
                        return "❌ No checkpoints to export."
                    
                    # Generate report
                    import json
                    from datetime import datetime
                    
                    report = {
                        "export_date": datetime.now().isoformat(),
                        "training_run": str(run_dir) if run_dir else "Unknown",
                        "total_checkpoints": len(checkpoints),
                        "checkpoints": [ckpt.to_dict() for ckpt in checkpoints]
                    }
                    
                    # Add analysis
                    analysis = checkpoint_manager.analyze_checkpoints_for_overfitting(checkpoints)
                    report["analysis"] = {
                        "overfitting_detected": analysis["overfitting_detected"],
                        "warning_message": analysis["warning_message"],
                        "eval_loss_trend": analysis["eval_loss_trend"]
                    }
                    
                    # Save to file
                    export_path = os.path.join(output_path, "checkpoint_analysis_report.json")
                    with open(export_path, 'w', encoding='utf-8') as f:
                        json.dump(report, f, indent=2, ensure_ascii=False)
                    
                    return f"✅ Analysis exported!\n\nSaved to: {export_path}\n\nYou can open this file to see detailed checkpoint information."
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"❌ Export failed: {str(e)}"
            
            # Wire up Fine-tuning tab checkpoint manager handlers
            refresh_train_checkpoints_btn.click(
                fn=refresh_training_checkpoints,
                inputs=[out_path],
                outputs=[train_checkpoint_display, train_checkpoint_selector]
            )
            
            analyze_train_overfitting_btn.click(
                fn=analyze_training_checkpoints,
                inputs=[out_path],
                outputs=[train_checkpoint_display]
            )
            
            cleanup_checkpoints_btn.click(
                fn=cleanup_old_checkpoints,
                inputs=[out_path],
                outputs=[train_checkpoint_display, train_checkpoint_selector]
            )
            
            delete_selected_btn.click(
                fn=delete_selected_checkpoints,
                inputs=[train_checkpoint_selector, out_path],
                outputs=[checkpoint_action_status]
            )
            
            copy_to_ready_btn.click(
                fn=copy_best_checkpoint_to_ready,
                inputs=[out_path],
                outputs=[checkpoint_action_status]
            )
            
            export_analysis_btn.click(
                fn=export_checkpoint_analysis,
                inputs=[out_path],
                outputs=[checkpoint_action_status]
            )


            train_btn.click(
                fn=train_model,
                inputs=[
                    custom_model,
                    version,
                    lang,
                    train_csv,
                    eval_csv,
                    num_epochs,
                    batch_size,
                    grad_acumm,
                    out_path,
                    max_audio_length,
                    save_step,
                    save_n_checkpoints,
                    enable_grad_checkpoint,
                    enable_sdpa,
                    enable_mixed_precision,
                    enable_amharic_g2p,
                    g2p_backend_train,
                    resume_from_checkpoint,
                    checkpoint_selector,
                    freeze_encoder,
                    freeze_n_gpt_layers,
                    learning_rate_custom,
                    weight_decay_custom,
                    enable_early_stopping,
                    early_stop_patience,
                    use_ema,
                    lr_warmup_steps,
                ],
                outputs=[progress_train, xtts_config, xtts_vocab, xtts_checkpoint,xtts_speaker, speaker_reference_audio],
            )

            optimize_model_btn.click(
                fn=optimize_model,
                inputs=[
                    out_path,
                    clear_train_data
                ],
                outputs=[progress_train,xtts_checkpoint],
            )
            
            load_btn.click(
                fn=load_model,
                inputs=[
                    xtts_checkpoint,
                    xtts_config,
                    xtts_vocab,
                    xtts_speaker
                ],
                outputs=[progress_load],
            )

            tts_btn.click(
                fn=run_tts,
                inputs=[
                    tts_language,
                    tts_text,
                    speaker_reference_audio,
                    temperature,
                    length_penalty,
                    repetition_penalty,
                    top_k,
                    top_p,
                    sentence_split,
                    use_config,
                    use_g2p_inference,
                    g2p_backend_infer,
                ],
                outputs=[progress_gen, tts_output_audio,reference_audio],
            )

            load_params_tts_btn.click(
                fn=load_params_tts,
                inputs=[
                    out_path,
                    version
                    ],
                outputs=[progress_load,xtts_checkpoint,xtts_config,xtts_vocab,xtts_speaker,speaker_reference_audio],
            )
             
            model_download_btn.click(
                fn=get_model_zip,
                inputs=[out_path],
                outputs=[model_zip_file]
            )
            
            dataset_download_btn.click(
                fn=get_dataset_zip,
                inputs=[out_path],
                outputs=[dataset_zip_file]
            )
            
            # Checkpoint selection handlers
            def scan_and_list_checkpoints(output_path):
                """Scan for available checkpoints and display info"""
                try:
                    run_dir, checkpoints = checkpoint_manager.get_latest_training_run_checkpoints(output_path)
                    
                    if not checkpoints:
                        return {
                            checkpoint_selector: gr.Dropdown(choices=[("No checkpoints found", "")], value=""),
                            checkpoint_info_display: "❌ No checkpoints found.\n\nPlease complete a training run first."
                        }
                    
                    # Get recommendation
                    recommended = checkpoint_manager.recommend_best_checkpoint(checkpoints)
                    
                    # Format display
                    display_text = checkpoint_manager.format_checkpoint_list_for_display(checkpoints, recommended)
                    
                    # Generate dropdown choices
                    choices = checkpoint_manager.get_checkpoint_dropdown_choices(checkpoints)
                    
                    # Set recommended as default value
                    default_value = recommended.path if recommended else (choices[0][1] if choices else "")
                    
                    return {
                        checkpoint_selector: gr.Dropdown(choices=choices, value=default_value),
                        checkpoint_info_display: display_text
                    }
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return {
                        checkpoint_selector: gr.Dropdown(choices=[("Error scanning", "")], value=""),
                        checkpoint_info_display: f"❌ Error scanning checkpoints: {str(e)}"
                    }
            
            def analyze_checkpoint_overfitting(output_path):
                """Analyze checkpoints for overfitting patterns"""
                try:
                    run_dir, checkpoints = checkpoint_manager.get_latest_training_run_checkpoints(output_path)
                    
                    if not checkpoints:
                        return "❌ No checkpoints found for analysis."
                    
                    analysis = checkpoint_manager.analyze_checkpoints_for_overfitting(checkpoints)
                    
                    # Format analysis display
                    lines = []
                    lines.append("📊 **Overfitting Analysis Report**")
                    lines.append("=" * 70)
                    lines.append("")
                    lines.append(analysis["warning_message"])
                    lines.append("")
                    
                    if analysis["eval_loss_trend"]:
                        lines.append("**Evaluation Loss Trend:**")
                        lines.append("")
                        for item in analysis["eval_loss_trend"]:
                            lines.append(f"  Epoch {item['epoch']:2d} | Step {item['step']:5d} | Loss: {item['eval_loss']:.4f}")
                        lines.append("")
                    
                    if analysis["safe_checkpoint"]:
                        lines.append("**Recommended Safe Checkpoint:**")
                        lines.append(f"  {analysis['safe_checkpoint'].display_name()}")
                    
                    lines.append("")
                    lines.append("=" * 70)
                    
                    return "\n".join(lines)
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"❌ Error analyzing checkpoints: {str(e)}"
            
            def use_selected_checkpoint(selected_checkpoint_path, output_path, xtts_config_path, xtts_vocab_path, xtts_speaker_path):
                """Copy selected checkpoint to ready folder and update paths"""
                try:
                    if not selected_checkpoint_path or selected_checkpoint_path == "":
                        return {
                            progress_load: "❌ No checkpoint selected. Please select a checkpoint first.",
                            xtts_checkpoint: xtts_checkpoint,
                        }
                    
                    # Copy checkpoint to ready folder
                    success, message = checkpoint_manager.copy_checkpoint_to_ready(
                        checkpoint_path=selected_checkpoint_path,
                        output_path=output_path,
                        as_name="model.pth"
                    )
                    
                    if not success:
                        return {
                            progress_load: message,
                            xtts_checkpoint: xtts_checkpoint,
                        }
                    
                    # Update checkpoint path
                    from pathlib import Path
                    new_checkpoint_path = str(Path(output_path) / "ready" / "model.pth")
                    
                    status_message = (
                        f"✅ Checkpoint Selected Successfully!\n\n"
                        f"{message}\n\n"
                        f"📍 New checkpoint path: {new_checkpoint_path}\n\n"
                        f"💡 Next: Click 'Step 3 - Load Model' to load this checkpoint"
                    )
                    
                    return {
                        progress_load: status_message,
                        xtts_checkpoint: new_checkpoint_path,
                    }
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return {
                        progress_load: f"❌ Error using checkpoint: {str(e)}",
                        xtts_checkpoint: xtts_checkpoint,
                    }
            
            # Wire up checkpoint selection handlers
            refresh_checkpoints_btn.click(
                fn=scan_and_list_checkpoints,
                inputs=[out_path],
                outputs=[checkpoint_selector, checkpoint_info_display]
            )
            
            analyze_overfitting_btn.click(
                fn=analyze_checkpoint_overfitting,
                inputs=[out_path],
                outputs=[checkpoint_info_display]
            )
            
            use_selected_checkpoint_btn.click(
                fn=use_selected_checkpoint,
                inputs=[checkpoint_selector, out_path, xtts_config, xtts_vocab, xtts_speaker],
                outputs=[progress_load, xtts_checkpoint]
            )
            
            def test_checkpoint_inference(selected_checkpoint_path, test_text, test_lang, use_g2p, 
                                         out_path, speaker_ref_path):
                """Generate speech directly from a checkpoint without loading it globally"""
                try:
                    if not selected_checkpoint_path or selected_checkpoint_path == "":
                        return None, "❌ No checkpoint selected. Please select a checkpoint first."
                    
                    if not test_text or test_text.strip() == "":
                        return None, "❌ Please enter test text."
                    
                    import tempfile
                    from pathlib import Path
                    
                    print(f"\n🎙️ Testing checkpoint: {selected_checkpoint_path}")
                    print(f"📝 Test text: {test_text}")
                    print(f"🌍 Language: {test_lang}")
                    
                    # Canonicalize language
                    test_lang = normalize_xtts_lang(test_lang)
                    
                    # Get paths from ready folder (vocab, config, speaker)
                    ready_dir = Path(out_path) / "ready"
                    
                    # Find vocab - prefer extended for Amharic
                    vocab_extended_amharic = ready_dir / "vocab_extended_amharic.json"
                    vocab_extended = ready_dir / "vocab_extended.json"
                    vocab_standard = ready_dir / "vocab.json"
                    
                    if vocab_extended_amharic.exists():
                        vocab_path = str(vocab_extended_amharic)
                    elif vocab_extended.exists():
                        vocab_path = str(vocab_extended)
                    else:
                        vocab_path = str(vocab_standard)
                    
                    config_path = str(ready_dir / "config.json")
                    speaker_path = str(ready_dir / "speakers_xtts.pth")
                    
                    # Use provided speaker reference or find one
                    if not speaker_ref_path or not os.path.exists(speaker_ref_path):
                        ref_path = ready_dir / "reference.wav"
                        if ref_path.exists():
                            speaker_ref_path = str(ref_path)
                        else:
                            return None, "❌ No speaker reference found. Please load model parameters first."
                    
                    # Load config
                    config = XttsConfig()
                    config.load_json(config_path)
                    
                    # Initialize model
                    print(" > Initializing model...")
                    test_model = Xtts.init_from_config(config)
                    
                    # Load checkpoint
                    print(f" > Loading checkpoint: {os.path.basename(selected_checkpoint_path)}")
                    test_model.load_checkpoint(
                        config,
                        checkpoint_path=selected_checkpoint_path,
                        vocab_path=vocab_path,
                        speaker_file_path=speaker_path,
                        use_deepspeed=False,
                        eval=True
                    )
                    
                    if torch.cuda.is_available():
                        test_model.cuda()
                    
                    # Apply G2P if needed
                    original_text = test_text
                    if use_g2p and test_lang in ["am", "amh"]:
                        try:
                            from amharic_tts.tokenizer.xtts_tokenizer_wrapper import create_xtts_tokenizer
                            print(" > 🇪🇹 Applying Amharic G2P...")
                            tokenizer = create_xtts_tokenizer(use_phonemes=True, g2p_backend="rule_based")
                            test_text = tokenizer.preprocess_text(test_text, lang='am')
                            print(f" > Converted: {original_text[:50]}... → {test_text[:50]}...")
                        except Exception as e:
                            print(f" > ⚠️ G2P failed: {e}, using original text")
                    
                    # Get conditioning latents
                    print(" > Extracting speaker embedding...")
                    gpt_cond_latent, speaker_embedding = test_model.get_conditioning_latents(
                        audio_path=speaker_ref_path,
                        gpt_cond_len=test_model.config.gpt_cond_len,
                        max_ref_length=test_model.config.max_ref_len,
                        sound_norm_refs=test_model.config.sound_norm_refs
                    )
                    
                    # Generate speech
                    print(" > Generating speech...")
                    _inference_lang = 'am' if test_lang in ('am', 'amh') else test_lang
                    
                    out = test_model.inference(
                        text=test_text,
                        language=_inference_lang,
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        temperature=0.75,
                        length_penalty=1.0,
                        repetition_penalty=5.0,
                        top_k=50,
                        top_p=0.85,
                        enable_text_splitting=True
                    )
                    
                    # Save output
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
                        out_path_audio = fp.name
                        torchaudio.save(out_path_audio, out["wav"], 24000)
                    
                    # Cleanup
                    del test_model
                    clear_gpu_cache()
                    
                    checkpoint_name = os.path.basename(selected_checkpoint_path)
                    status = (
                        f"✅ Test Complete!\n\n"
                        f"Checkpoint: {checkpoint_name}\n"
                        f"Text: {original_text[:50]}{'...' if len(original_text) > 50 else ''}\n"
                        f"Language: {test_lang}\n"
                        f"G2P: {'Enabled' if use_g2p and test_lang in ['am', 'amh'] else 'Disabled'}"
                    )
                    
                    print(" > ✅ Inference complete!")
                    return out_path_audio, status
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    error_msg = (
                        f"❌ Error testing checkpoint:\n\n"
                        f"{str(e)}\n\n"
                        f"Tip: Make sure you've loaded parameters (vocab, config, speaker) first."
                    )
                    return None, error_msg
            
            test_checkpoint_btn.click(
                fn=test_checkpoint_inference,
                inputs=[
                    checkpoint_selector,
                    checkpoint_test_text,
                    checkpoint_test_language,
                    checkpoint_use_g2p,
                    out_path,
                    speaker_reference_audio
                ],
                outputs=[checkpoint_test_output, checkpoint_test_status]
            )

    demo.launch(
        share=args.share,
        debug=False,
        server_port=args.port,
        # inweb=True,
        # server_name="localhost"
    )
