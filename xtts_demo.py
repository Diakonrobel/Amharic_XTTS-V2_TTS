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
from utils import youtube_downloader, srt_processor, audio_slicer, dataset_tracker, batch_processor
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
                        tgt = getattr(XTTS_MODEL.gpt, 'gpt', XTTS_MODEL.gpt)
                        class _InferenceProxy:
                            def __init__(self, target):
                                self._target = target
                                self.prefix_emb = None
                            def store_prefix_emb(self, emb):
                                self.prefix_emb = emb
                            def __getattr__(self, name):
                                return getattr(self._target, name)
                        XTTS_MODEL.gpt.gpt_inference = _InferenceProxy(tgt)
                        print(" > ✅ Created compatibility proxy for gpt_inference")
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
                            tgt = getattr(XTTS_MODEL.gpt, 'gpt', XTTS_MODEL.gpt)
                            class _InferenceProxy:
                                def __init__(self, target):
                                    self._target = target
                                    self.prefix_emb = None
                                def store_prefix_emb(self, emb):
                                    self.prefix_emb = emb
                                def __getattr__(self, name):
                                    return getattr(self._target, name)
                            XTTS_MODEL.gpt.gpt_inference = _InferenceProxy(tgt)
                            print(" > ✅ Created compatibility proxy for gpt_inference")
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
        help="Number of epochs to train. Default: 6",
        default=6,
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
        help="Max permitted audio size in seconds. Default: 20",
        default=20,
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
                    
                    with gr.Accordion("⚙️ VAD Settings", open=False):
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
        
            # Advanced processing functions
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
            def process_srt_media_batch_handler(srt_files_list, media_files_list, language, out_path, progress):
                """Handle batch processing of multiple SRT+media pairs"""
                try:
                    # Canonicalize language for dataset artifacts
                    language = normalize_xtts_lang(language)
                    progress(0, desc=f"Initializing batch processing for {len(srt_files_list)} pairs...")
                    
                    # Process all pairs
                    train_csv, eval_csv, file_infos = batch_processor.process_srt_media_batch(
                        srt_files=srt_files_list,
                        media_files=media_files_list,
                        language=language,
                        out_path=out_path,
                        srt_processor=srt_processor,
                        progress_callback=lambda p, desc: progress(p, desc=desc)
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
                vad_threshold_val=0.5,
                vad_min_speech_ms=250,
                vad_min_silence_ms=300,
                vad_pad_ms=30,
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
                        return process_srt_media_batch_handler(srt_files_list, media_files_list, language, out_path, progress)
                    
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
                    
                    # Choose processor based on VAD setting
                    if use_vad:
                        from utils import srt_processor_vad
                        
                        mode_desc = "VAD-enhanced"
                        progress(0, desc=f"Initializing {mode_desc} SRT processor...")
                        progress(0.2, desc="Loading VAD model...")
                        progress(0.3, desc="Processing SRT with VAD refinement...")
                        
                        train_csv, eval_csv, duration = srt_processor_vad.process_srt_with_media_vad(
                            srt_path=srt_file_path,
                            media_path=media_file_path,
                            output_dir=output_path,
                            language=language,
                            use_vad_refinement=True,
                            vad_threshold=vad_threshold_val,
                            gradio_progress=progress
                        )
                    else:
                        mode_desc = "standard"
                        progress(0, desc=f"Initializing {mode_desc} SRT processor...")
                        progress(0.3, desc="Processing SRT and extracting audio segments...")
                        
                        train_csv, eval_csv, duration = srt_processor.process_srt_with_media(
                            srt_path=srt_file_path,
                            media_path=media_file_path,
                            output_dir=output_path,
                            language=language,
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
                    vad_info = f" (VAD-enhanced)" if use_vad else ""
                    return f"✓ SRT Processing Complete{vad_info}!\nProcessed {num_segments} segments\nTotal audio: {duration:.1f}s\nDataset created at: {output_path}\nMode: {mode_desc.capitalize()}\n\nℹ This dataset has been saved to history and won't be reprocessed."
                    
                except Exception as e:
                    traceback.print_exc()
                    return f"❌ Error processing SRT: {str(e)}"
            
            def process_youtube_batch_urls(urls, transcript_lang, out_path, progress):
                """Process multiple YouTube URLs in batch mode"""
                try:
                    progress(0, desc=f"Initializing batch processing for {len(urls)} videos...")
                    
                    # Process all videos
                    train_csv, eval_csv, video_infos = batch_processor.process_youtube_batch(
                        urls=urls,
                        transcript_lang=transcript_lang,
                        out_path=out_path,
                        youtube_downloader=youtube_downloader,
                        srt_processor=srt_processor,
                        progress_callback=lambda p, desc: progress(p, desc=desc)
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
                    summary += "\n\nℹ This batch dataset has been saved to history."
                    
                    progress(1.0, desc="Batch processing complete!")
                    return summary
                    
                except Exception as e:
                    traceback.print_exc()
                    return f"❌ Error in batch processing: {str(e)}"
            
            def download_youtube_video(url, transcript_lang, language, out_path, batch_mode, progress=gr.Progress(track_tqdm=True)):
                """Download YouTube video(s) and extract transcripts"""
                try:
                    if not url:
                        return "Please enter a YouTube URL!"
                    
                    # Parse URLs
                    urls = batch_processor.parse_youtube_urls(url)
                    
                    if not urls:
                        return "❌ No valid YouTube URLs found. Please check your input."
                    
                    # Check if batch mode and multiple URLs
                    if batch_mode and len(urls) > 1:
                        return process_youtube_batch_urls(urls, transcript_lang, out_path, progress)
                    
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
                        auto_update=True
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
                    
                    train_csv, eval_csv, duration = srt_processor.process_srt_with_media(
                        srt_path=srt_path,
                        media_path=audio_path,
                        output_dir=output_path,
                        language=dataset_language,
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
                    return f"✓ YouTube Processing Complete!\nTitle: {info.get('title', 'Unknown')}\nDuration: {info.get('duration', 0):.0f}s\nProcessed {num_segments} segments\nDataset created at: {output_path}\n\nℹ This dataset has been saved to history and won't be reprocessed."
                    
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
                        info="Training iterations"
                    )
                    batch_size = gr.Slider(
                        label="Batch Size",
                        minimum=2, maximum=512, step=1, value=args.batch_size,
                        info="Samples per batch"
                    )
                with gr.Row():
                    grad_acumm = gr.Slider(
                        label="Grad Accumulation",
                        minimum=1, maximum=128, step=1, value=args.grad_acumm,
                        info="Gradient accumulation steps"
                    )
                    max_audio_length = gr.Slider(
                        label="Max Audio (sec)",
                        minimum=2, maximum=20, step=1, value=args.max_audio_length,
                        info="Max permitted audio length"
                    )
                
                with gr.Row():
                    save_step = gr.Slider(
                        label="Checkpoint Save Frequency (steps)",
                        minimum=100, maximum=5000, step=100, value=1000,
                        info="Save checkpoint every N steps"
                    )
                    save_n_checkpoints = gr.Slider(
                        label="Keep N Checkpoints",
                        minimum=1, maximum=10, step=1, value=1,
                        info="Number of checkpoints to retain"
                    )
                
                clear_train_data = gr.Dropdown(
                    label="Cleanup After Training",
                    value="none",
                    choices=["none", "run", "dataset", "all"],
                    info="Delete training data after optimization"
                )
            
            with gr.Group():
                gr.Markdown("### ⚡ **Training Optimizations** (Speed & Memory)")
                gr.Markdown("_Enable optimizations for faster training with less memory_")
                with gr.Row():
                    enable_grad_checkpoint = gr.Checkbox(
                        label="Gradient Checkpointing",
                        value=False,
                        info="20-30% memory reduction (minimal speed impact)"
                    )
                    enable_sdpa = gr.Checkbox(
                        label="Fast Attention (SDPA)",
                        value=False,
                        info="1.3-1.5x speed + 30-40% memory reduction"
                    )
                    enable_mixed_precision = gr.Checkbox(
                        label="Mixed Precision (FP16/BF16)",
                        value=False,
                        info="Additional speedup (Ampere+ GPUs)"
                    )
            
            with gr.Group():
                gr.Markdown("### 🇪🇹 **Amharic G2P Options** (for 'amh' language)")
                with gr.Row():
                    enable_amharic_g2p = gr.Checkbox(
                        label="Enable G2P for Training",
                        value=False,
                        info="Use phoneme tokenization",
                        scale=2
                    )
                    g2p_backend_train = gr.Dropdown(
                        label="G2P Backend",
                        value="transphone",
                        choices=["transphone", "epitran", "rule_based"],
                        info="G2P conversion backend",
                        scale=1
                    )
            
            with gr.Group():
                gr.Markdown("### 🚀 **Execute Training**")
                progress_train = gr.Label(label="Status", value="Ready")
                with gr.Row():
                    train_btn = gr.Button(value="▶️ Step 2 - Train Model", variant="primary", size="lg", scale=2)
                    optimize_model_btn = gr.Button(value="⚡ Step 2.5 - Optimize Model", variant="primary", size="lg", scale=1)
            
            import os
            import shutil
            from pathlib import Path
            import traceback
            
            def train_model(custom_model, version, language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length, save_step=1000, save_n_checkpoints=1, enable_grad_checkpoint=False, enable_sdpa=False, enable_mixed_precision=False, enable_amharic_g2p=False, g2p_backend_train="transphone"):
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
            
                # Remove train dir
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

                    speaker_xtts_path, config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(
                        custom_model, version, language_norm, num_epochs, batch_size, grad_acumm,
                        train_csv, eval_csv, output_path=output_path, max_audio_length=max_audio_length,
                        save_step=save_step, save_n_checkpoints=save_n_checkpoints,
                        use_amharic_g2p=use_amharic_g2p,
                        enable_grad_checkpoint=enable_grad_checkpoint,
                        enable_sdpa=enable_sdpa,
                        enable_mixed_precision=enable_mixed_precision
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
                    vad_threshold,  # VAD threshold
                    vad_min_speech_duration,  # Min speech duration
                    vad_min_silence_duration,  # Min silence duration
                    vad_speech_pad,  # Speech padding
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
                    youtube_batch_mode,  # Add batch mode parameter
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

    demo.launch(
        share=args.share,
        debug=False,
        server_port=args.port,
        # inweb=True,
        # server_name="localhost"
    )
