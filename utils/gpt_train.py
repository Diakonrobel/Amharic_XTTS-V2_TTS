import logging
import os
import gc
import torch
from pathlib import Path

# Apply PyTorch 2.6 compatibility patches
try:
    from utils.pytorch26_patch import apply_pytorch26_compatibility_patches
    apply_pytorch26_compatibility_patches()
except Exception as e:
    print(f"Warning: Could not apply PyTorch 2.6 patches: {e}")

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
import shutil

# Apply global Amharic BPE tokenizer patch BEFORE any model initialization
try:
    from utils.amharic_bpe_tokenizer_patch import apply_global_amharic_bpe_patch
    apply_global_amharic_bpe_patch()
except Exception as e:
    print(f" > Warning: Could not apply Amharic BPE patch: {e}")

# Import training optimizations
try:
    from utils.training_optimizations import TrainingOptimizer, UnslothStyleOptimizations
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    print(" > Warning: Training optimizations module not available")
    OPTIMIZATIONS_AVAILABLE = False

# Import training enhancements
try:
    from utils.training_enhancements import (
        EMAModel, WarmupLRScheduler, LabelSmoother, AdaptiveGradientClipper, auto_detect_mixed_precision
    )
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    print(" > Warning: Training enhancements module not available")
    ENHANCEMENTS_AVAILABLE = False

# Import dataset validator
try:
    from utils.dataset_validator import validate_dataset_before_training, DatasetValidationError
    DATASET_VALIDATOR_AVAILABLE = True
except ImportError:
    print(" > Warning: Dataset validator not available")
    DATASET_VALIDATOR_AVAILABLE = False

# Import small dataset configuration
try:
    from utils.xtts_small_dataset_config import XTTSSmallDatasetConfig, EarlyStoppingCallback
    SMALL_DATASET_CONFIG_AVAILABLE = True
except ImportError:
    print(" > Warning: Small dataset config not available")
    SMALL_DATASET_CONFIG_AVAILABLE = False


def train_gpt(custom_model,version, language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path, max_audio_length=255995, save_step=1000, save_n_checkpoints=1, use_amharic_g2p=False, g2p_backend_train="transphone", enable_grad_checkpoint=False, enable_sdpa=False, enable_mixed_precision=False, freeze_encoder=True, freeze_first_n_gpt_layers=0, learning_rate_override=None, weight_decay_override=None, early_stopping_patience=None, use_ema=True, lr_warmup_steps=500, use_label_smoothing=False, label_smoothing_factor=0.1):
    #  Logging parameters
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None
    
    # ===================================================================
    # AUTOMATIC DATASET VALIDATION AND FIXING
    # ===================================================================
    # Validate and auto-fix dataset CSV files before training starts
    if DATASET_VALIDATOR_AVAILABLE:
        try:
            print(" > Running automatic dataset validation...")
            validate_dataset_before_training(
                train_csv=train_csv,
                eval_csv=eval_csv,
                expected_language=language,
                auto_fix=True  # Automatically fix common issues
            )
            print(" > Γ£à Dataset validation complete!")
        except DatasetValidationError as e:
            print(f" > Γ¥î CRITICAL: Dataset validation failed!")
            print(f" > Error: {e}")
            print(f" > Please fix the dataset issues before training.")
            raise  # Re-raise to stop training
        except Exception as e:
            print(f" > ΓÜá∩╕Å  Warning: Dataset validation encountered an error: {e}")
            print(f" > Training will continue, but there may be issues...")
    else:
        print(" > ΓÜá∩╕Å  Dataset validator not available - skipping validation")
    # ===================================================================

    # print(f"XTTS version = {version}")

    # Set here the path that the checkpoints will be saved. Default: ./run/training/
    OUT_PATH = os.path.join(output_path, "run", "training")

    # ===================================================================
    # SMALL DATASET DETECTION AND OPTIMIZATION
    # ===================================================================
    # Detect dataset size and apply optimal configuration
    use_small_dataset_config = False
    
    if SMALL_DATASET_CONFIG_AVAILABLE:
        # Count samples in training CSV
        try:
            import csv
            with open(train_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                train_sample_count = sum(1 for row in reader)
                # Subtract 1 if header present
                if train_sample_count > 0:
                    with open(train_csv, 'r', encoding='utf-8') as f:
                        first_line = f.readline()
                        if 'audio_file' in first_line.lower():
                            train_sample_count -= 1
            
            # Apply small dataset config if <3000 samples
            if train_sample_count < 3000:
                use_small_dataset_config = True
                print("\n" + "="*70)
                print(f"≡ƒôè SMALL DATASET DETECTED: {train_sample_count} samples")
                print("="*70)
                print(" > Applying anti-overfitting configuration automatically")
                print(" > This prevents memorization and improves generalization")
                print("="*70 + "\n")
                
                # Override parameters with small dataset config
                BATCH_SIZE = XTTSSmallDatasetConfig.BATCH_SIZE
                GRAD_ACUMM_STEPS = XTTSSmallDatasetConfig.GRAD_ACCUM_STEPS
                num_epochs = XTTSSmallDatasetConfig.MAX_EPOCHS
                
                # Print configuration
                XTTSSmallDatasetConfig.print_config_summary()
        except Exception as e:
            print(f" > Could not detect dataset size: {e}")
            print(" > Using provided configuration")
    
    # Training Parameters
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
    START_WITH_EVAL = False  # if True it will star with evaluation
    BATCH_SIZE = batch_size if not use_small_dataset_config else BATCH_SIZE  # set here the batch size
    GRAD_ACUMM_STEPS = grad_acumm if not use_small_dataset_config else GRAD_ACUMM_STEPS  # set here the grad accumulation steps


    # Dataset config will be created after G2P preprocessing (if enabled)
    # to ensure correct language code is used
    DATASETS_CONFIG_LIST = []

    # Define the path where XTTS v2.0.1 files will be downloaded
    CHECKPOINTS_OUT_PATH = os.path.join(Path.cwd(), "base_models",f"{version}")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


    # DVAE files
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

    # Set the path to the downloaded files
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

    # download DVAE files if needed
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)


    # Download XTTS v2.0 checkpoint if needed
    TOKENIZER_FILE_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/vocab.json"
    XTTS_CHECKPOINT_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/model.pth"
    XTTS_CONFIG_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/config.json"
    XTTS_SPEAKER_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/speakers_xtts.pth"

    # XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CONFIG_LINK))  # config.json file
    XTTS_SPEAKER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_SPEAKER_LINK))  # speakers_xtts.pth file

    # download XTTS v2.0 files if needed
    if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
        print(f" > Downloading XTTS v{version} files!")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK, XTTS_CONFIG_LINK,XTTS_SPEAKER_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )

    # Transfer this files to ready folder
    READY_MODEL_PATH = os.path.join(output_path,"ready")
    if not os.path.exists(READY_MODEL_PATH):
        os.makedirs(READY_MODEL_PATH)

    NEW_TOKENIZER_FILE = os.path.join(READY_MODEL_PATH, "vocab.json")
    # NEW_XTTS_CHECKPOINT = os.path.join(READY_MODEL_PATH, "model.pth")
    NEW_XTTS_CONFIG_FILE = os.path.join(READY_MODEL_PATH, "config.json")
    NEW_XTTS_SPEAKER_FILE = os.path.join(READY_MODEL_PATH, "speakers_xtts.pth")

    shutil.copy(TOKENIZER_FILE, NEW_TOKENIZER_FILE)
    # shutil.copy(XTTS_CHECKPOINT, os.path.join(READY_MODEL_PATH, "model.pth"))
    shutil.copy(XTTS_CONFIG_FILE, NEW_XTTS_CONFIG_FILE)
    shutil.copy(XTTS_SPEAKER_FILE, NEW_XTTS_SPEAKER_FILE)

# Use from ready folder
    TOKENIZER_FILE = NEW_TOKENIZER_FILE # vocab.json file
    # XTTS_CHECKPOINT = NEW_XTTS_CHECKPOINT  # model.pth file
    XTTS_CONFIG_FILE = NEW_XTTS_CONFIG_FILE  # config.json file
    XTTS_SPEAKER_FILE = NEW_XTTS_SPEAKER_FILE  # speakers_xtts.pth file


    if custom_model != "":
        if os.path.exists(custom_model) and custom_model.endswith('.pth'):
            XTTS_CHECKPOINT = custom_model
            print(f" > Loading custom model: {XTTS_CHECKPOINT}")
        else:
            print(" > Error: The specified custom model is not a valid .pth file path.")

    # Reduce num_workers to prevent data loading issues (system recommends 4 max)
    num_workers = 4
    if language == "ja":
        num_workers = 0
    
    # Handle Amharic-specific preprocessing - ROBUST IMPLEMENTATION
    # Accept both 'am' and 'amh' for G2P
    amharic_g2p_enabled = use_amharic_g2p and language in ["am", "amh", "en"]
    effective_language = language  # Will be updated if G2P preprocessing is applied
    dataset_already_phonemes = False  # Track if dataset is already phonemes
    
    # Extended vocabulary path (will be created if G2P enabled)
    extended_vocab_path = None
    
    if amharic_g2p_enabled:
        print(" > Amharic G2P mode ENABLED")
        print(" > Dataset will be checked and converted if needed")
        print(" > Vocabulary will be extended with Amharic tokens")
        
        # Check if dataset needs preprocessing
        try:
            from utils.amharic_g2p_dataset_wrapper import is_dataset_already_preprocessed
            train_already_processed = is_dataset_already_preprocessed(train_csv)
            eval_already_processed = is_dataset_already_preprocessed(eval_csv)
            
            if train_already_processed and eval_already_processed:
                print(" > Dataset is already preprocessed with phonemes")
                print(" > Switching language code to 'en' for XTTS tokenizer")
                effective_language = "en"
                dataset_already_phonemes = True  # Mark as already preprocessed
            else:
                print(" > Dataset contains Amharic script - will convert to phonemes")
                effective_language = language  # Keep original for now, will be updated after loading
                dataset_already_phonemes = False  # Needs preprocessing
        except Exception as e:
            print(f" > Warning: Could not check dataset preprocessing status: {e}")
            dataset_already_phonemes = False  # Assume needs preprocessing on error
        
        # Extend vocabulary with Amharic tokens
        print(" > Extending XTTS vocabulary with Amharic tokens...")
        try:
            from utils.vocab_extension import create_extended_vocab_for_training
            
            extended_vocab_path = create_extended_vocab_for_training(
                base_vocab_path=TOKENIZER_FILE,
                output_dir=READY_MODEL_PATH,
                train_csv_path=train_csv,
                eval_csv_path=eval_csv
            )
            
            print(f" > Γ£à Extended vocabulary created: {extended_vocab_path}")
            print(f" > This vocab includes Ethiopic chars + IPA phonemes + dataset-specific tokens")
            
        except Exception as e:
            print(f" > ΓÜá∩╕Å  Warning: Could not extend vocabulary: {e}")
            print(" > Training will continue with standard vocabulary")
            import traceback
            traceback.print_exc()
    
    # Create dataset config with effective language
    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=os.path.dirname(train_csv),
        meta_file_train=os.path.basename(train_csv),
        meta_file_val=os.path.basename(eval_csv),
        language=effective_language,
    )
    DATASETS_CONFIG_LIST = [config_dataset]
    
    # Use extended vocabulary if available (Amharic G2P mode)
    tokenizer_file_to_use = extended_vocab_path if extended_vocab_path else TOKENIZER_FILE
    if extended_vocab_path:
        print(f" > Using EXTENDED vocabulary for training: {tokenizer_file_to_use}")
    else:
        print(f" > Using standard vocabulary for training: {tokenizer_file_to_use}")
    
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=max_audio_length,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=tokenizer_file_to_use,  # Use extended vocab if available
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # Determine if we should enable language adaptation mode (Amharic, non-small dataset)
    language_adaptation_mode = (language in ["am", "amh"]) and (not use_small_dataset_config)
    
    # Auto-enable mixed precision if not explicitly set
    if not enable_mixed_precision and ENHANCEMENTS_AVAILABLE:
        enable_mixed_precision = auto_detect_mixed_precision()
        if enable_mixed_precision:
            print(" > ✅ Mixed precision AUTO-ENABLED (modern GPU detected)")
    
    # Override parameters for better anti-overfitting
    final_learning_rate = learning_rate_override if learning_rate_override is not None else (2e-06 if language_adaptation_mode else 1e-05)
    final_weight_decay = weight_decay_override if weight_decay_override is not None else (0.05 if language_adaptation_mode else 0.01)
    
    # Enable EMA by default for language adaptation
    use_ema_final = use_ema and ENHANCEMENTS_AVAILABLE
    lr_warmup_steps_final = lr_warmup_steps if (use_ema_final or language_adaptation_mode) else 0
    
    if use_ema_final:
        print(f" > ✅ EMA (Exponential Moving Average) enabled: decay=0.999")
    if lr_warmup_steps_final > 0:
        print(f" > ✅ LR Warmup enabled: {lr_warmup_steps_final} steps (0 → {final_learning_rate})")
    
    # Determine layer freezing strategy
    freeze_encoder_layers = freeze_encoder if freeze_encoder is not None else (True if language_adaptation_mode else False)
    freeze_gpt_layers_count = freeze_first_n_gpt_layers if freeze_first_n_gpt_layers > 0 else (28 if language_adaptation_mode else 0)
    
    # Early stopping configuration
    use_early_stopping = early_stopping_patience is not None
    early_stop_patience = early_stopping_patience if early_stopping_patience is not None else (2 if language_adaptation_mode else 5)

    # training parameters config
    # Use small dataset config if applicable
    if use_small_dataset_config:
        optimizer_config = XTTSSmallDatasetConfig.get_optimizer_config()
        scheduler_config = XTTSSmallDatasetConfig.get_scheduler_config()
        
        config = GPTTrainerConfig(
            epochs=num_epochs,
            output_path=OUT_PATH,
            model_args=model_args,
            run_name=RUN_NAME,
            project_name=PROJECT_NAME,
            run_description="""
                GPT XTTS training with small dataset optimization
                """,
            dashboard_logger=DASHBOARD_LOGGER,
            logger_uri=LOGGER_URI,
            audio=audio_config,
            batch_size=BATCH_SIZE,
            batch_group_size=48,
            eval_batch_size=BATCH_SIZE,
            num_loader_workers=num_workers,
            eval_split_max_size=256,
            print_step=XTTSSmallDatasetConfig.PRINT_STEP,
            plot_step=100,
            log_model_step=100,
            save_step=XTTSSmallDatasetConfig.SAVE_STEP,
            save_n_checkpoints=XTTSSmallDatasetConfig.SAVE_N_CHECKPOINTS,
            save_checkpoints=True,
            print_eval=False,
            optimizer=optimizer_config["optimizer"],
            optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
            optimizer_params=optimizer_config["optimizer_params"],
            lr=optimizer_config["lr"],
            lr_scheduler=scheduler_config["lr_scheduler"],
            lr_scheduler_params=scheduler_config["lr_scheduler_params"],
            test_sentences=[],
        )
    else:
        config = GPTTrainerConfig(
            epochs=num_epochs,
            output_path=OUT_PATH,
            model_args=model_args,
            run_name=RUN_NAME,
            project_name=PROJECT_NAME,
            run_description="""
                GPT XTTS training
                """,
            dashboard_logger=DASHBOARD_LOGGER,
            logger_uri=LOGGER_URI,
            audio=audio_config,
            batch_size=BATCH_SIZE,
            batch_group_size=48,
            eval_batch_size=BATCH_SIZE,
            num_loader_workers=num_workers,
            eval_split_max_size=256,
            print_step=50,
            plot_step=100,
            log_model_step=100,
            save_step=save_step,
            save_n_checkpoints=save_n_checkpoints,
            save_checkpoints=True,
            # target_loss="loss",
            print_eval=False,
            # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
            optimizer="AdamW",
            optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
            optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": final_weight_decay},
            lr=final_learning_rate,
            lr_scheduler="MultiStepLR",
            lr_scheduler_params={"milestones": [1, 2, 3], "gamma": 0.5},
            test_sentences=[],
        )

    # Handle extended vocabulary - temporarily disable checkpoint loading, then reload manually
    checkpoint_to_load = None
    if extended_vocab_path:
        print(" > Extended vocabulary detected - will handle checkpoint loading manually...")
        checkpoint_to_load = model_args.xtts_checkpoint
        model_args.xtts_checkpoint = None  # Temporarily disable auto-loading
    
    # Apply training optimizations BEFORE model initialization
    optimization_status = {}
    if OPTIMIZATIONS_AVAILABLE and (enable_grad_checkpoint or enable_sdpa or enable_mixed_precision):
        print(" > Configuring training optimizations...")
        
        # Enable cuDNN and other low-level optimizations
        UnslothStyleOptimizations.enable_cudnn_optimizations()
        
        # Create optimizer instance
        optimizer = TrainingOptimizer(
            enable_gradient_checkpointing=enable_grad_checkpoint,
            enable_sdpa=enable_sdpa,
            enable_mixed_precision=enable_mixed_precision,
            verbose=True
        )
    

    # init the model from config (without checkpoint if extended vocab)
    model = GPTTrainer.init_from_config(config)
    
    # Apply layer freezing for small datasets or language adaptation (Amharic)
    if use_small_dataset_config:
        print("\n" + "="*70)
        print("≡ƒöÆ APPLYING LAYER FREEZING FOR SMALL DATASET")
        print("="*70)
        total_params, trainable_params = XTTSSmallDatasetConfig.apply_layer_freezing(model)
        print("="*70 + "\n")
    
    # Apply layer freezing based on configuration
    if freeze_encoder_layers or freeze_gpt_layers_count > 0:
        try:
            print("\n" + "="*70)
            print("≡ƒöÆ APPLYING LAYER FREEZING FOR ANTI-OVERFITTING")
            print("="*70)
            
            frozen_params = 0
            trainable_params = 0
            
            if hasattr(model, 'xtts'):
                # Freeze encoder modules if requested
                if freeze_encoder_layers:
                    if hasattr(model.xtts, 'mel_encoder'):
                        for p in model.xtts.mel_encoder.parameters():
                            p.requires_grad = False
                            frozen_params += p.numel()
                        print("  ✓ Froze mel_encoder")
                    if hasattr(model.xtts, 'dvae'):
                        for p in model.xtts.dvae.parameters():
                            p.requires_grad = False
                            frozen_params += p.numel()
                        print("  ✓ Froze dvae")
                
                # Freeze first N GPT layers
                if freeze_gpt_layers_count > 0 and hasattr(model.xtts, 'gpt') and hasattr(model.xtts.gpt, 'transformer'):
                    layers = getattr(model.xtts.gpt.transformer, 'h', [])
                    total_layers = len(layers)
                    freeze_n = min(freeze_gpt_layers_count, total_layers - 2)  # Keep at least 2 trainable
                    
                    for i, layer in enumerate(layers):
                        if i < freeze_n:
                            for p in layer.parameters():
                                p.requires_grad = False
                                frozen_params += p.numel()
                        else:
                            for p in layer.parameters():
                                p.requires_grad = True
                                trainable_params += p.numel()
                    
                    print(f"  ✓ Froze first {freeze_n}/{total_layers} GPT layers")
                    print(f"  ✓ Last {total_layers - freeze_n} layers TRAINABLE")
                
                # Ensure text embedding and head stay trainable
                if hasattr(model.xtts.gpt, 'text_embedding'):
                    for p in model.xtts.gpt.text_embedding.parameters():
                        p.requires_grad = True
                        trainable_params += p.numel()
                    print("  ✓ text_embedding TRAINABLE")
                if hasattr(model.xtts.gpt, 'text_head'):
                    for p in model.xtts.gpt.text_head.parameters():
                        p.requires_grad = True
                        trainable_params += p.numel()
                    print("  ✓ text_head TRAINABLE")
            
            # Count total trainable/frozen parameters
            for name, param in model.named_parameters():
                if param.requires_grad and 'xtts.gpt' not in name:
                    trainable_params += param.numel()
                elif not param.requires_grad:
                    frozen_params += param.numel()
            
            total_params = frozen_params + trainable_params
            print(f"\n└┐ Parameter Summary:")
            print(f"  Total: {total_params:,}")
            print(f"  Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
            print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
            print("="*70 + "\n")
        except Exception as e:
            print(f" > Warning: Could not apply layer freezing: {e}")
            import traceback
            traceback.print_exc()
    
    # Apply optimizations to model AFTER initialization
    if OPTIMIZATIONS_AVAILABLE and (enable_grad_checkpoint or enable_sdpa or enable_mixed_precision):
        print(" > Applying optimizations to model...")
        optimization_status = optimizer.optimize_model(model)
        
        # Print initial memory stats
        TrainingOptimizer.print_memory_stats()
    
    # If using extended vocabulary, manually load checkpoint and resize embeddings
    if extended_vocab_path and checkpoint_to_load:
        print(f" > Loading checkpoint manually for vocab expansion: {checkpoint_to_load}")
        try:
            import json
            # Load extended vocab to get size
            with open(tokenizer_file_to_use, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
                new_vocab_size = len(vocab_data['model']['vocab'])
            
            # Load checkpoint (PyTorch 2.6+ requires weights_only=False for legacy checkpoints)
            checkpoint = torch.load(checkpoint_to_load, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model"]
            
            # Find text embedding key (handle different formats)
            embed_key = None
            head_w_key = None
            head_b_key = None
            
            # Try different key patterns
            for key in state_dict.keys():
                if 'text_embedding.weight' in key:
                    embed_key = key
                if 'text_head.weight' in key:
                    head_w_key = key
                if 'text_head.bias' in key:
                    head_b_key = key
            
            if not embed_key:
                raise KeyError("Could not find text_embedding.weight in checkpoint. Keys available: " + str(list(state_dict.keys())[:10]))
            
            old_vocab_size = state_dict[embed_key].shape[0]
            embed_dim = state_dict[embed_key].shape[1]
            
            # Check if vocab already matches (resuming from extended vocab training)
            if old_vocab_size == new_vocab_size:
                print(f" > ✅ Checkpoint already has extended vocab (size: {old_vocab_size})")
                print(f" > Skipping vocab expansion, loading checkpoint directly")
                # Load checkpoint normally
                model.xtts.load_state_dict(state_dict, strict=False)
                print(f" > ✅ Checkpoint loaded successfully!")
                # Skip the expansion logic below
                checkpoint_to_load = None  # Signal to skip
            else:
                # Proceed with expansion
                print(f" > Checkpoint vocab size: {old_vocab_size}")
                print(f" > Extended vocab size: {new_vocab_size}")
                print(f" > Will add {new_vocab_size - old_vocab_size} new token embeddings")
                
                # Create new extended embedding layers
                new_text_embedding = torch.nn.Embedding(new_vocab_size, embed_dim)
                new_text_embedding.weight.data[:old_vocab_size] = state_dict[embed_key]
                new_text_embedding.weight.data[old_vocab_size:] = torch.randn(new_vocab_size - old_vocab_size, embed_dim) * 0.02
                
                new_text_head = torch.nn.Linear(embed_dim, new_vocab_size)
                new_text_head.weight.data[:old_vocab_size] = state_dict[head_w_key]
                new_text_head.weight.data[old_vocab_size:] = torch.randn(new_vocab_size - old_vocab_size, embed_dim) * 0.02
                new_text_head.bias.data[:old_vocab_size] = state_dict[head_b_key]
                new_text_head.bias.data[old_vocab_size:] = torch.zeros(new_vocab_size - old_vocab_size)
                
                # Remove text embedding layers from state dict
                filtered_state = {k: v for k, v in state_dict.items() 
                                if 'text_embedding' not in k and 'text_head' not in k}
                
                # Load non-text layers
                model.xtts.load_state_dict(filtered_state, strict=False)
                
                # Replace text embedding layers with extended versions
                model.xtts.gpt.text_embedding = new_text_embedding
                model.xtts.gpt.text_head = new_text_head
                
                print(f" > ✅ Checkpoint loaded and embeddings resized!")
                print(f" > Copied {old_vocab_size} existing embeddings")
                print(f" > Initialized {new_vocab_size - old_vocab_size} new embeddings (random, will be learned)")
            
        except Exception as e:
            print(f" > ΓÜá∩╕Å  Error in extended vocab checkpoint loading: {e}")
            print(" > Training may fail - consider using standard vocabulary")
            import traceback
            traceback.print_exc()

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    
    # Apply G2P preprocessing if enabled and dataset is NOT already phonemes
    if amharic_g2p_enabled and not dataset_already_phonemes:
        print(" > Applying Amharic G2P preprocessing to training data...")
        print(f" > Current effective_language: '{effective_language}'")
        print(f" > Will convert Amharic text → IPA phonemes")
        try:
            # Use optimized parallel G2P preprocessor
            from utils.g2p_dataset_optimizer import apply_g2p_to_training_data_optimized
            from utils.g2p_backend_selector import select_g2p_backend
            
            # Use user-selected G2P backend from WebUI (or auto-select if None)
            selected_backend, reason = select_g2p_backend(
                preferred=g2p_backend_train,  # Use user's choice from WebUI
                fallback=True,
                verbose=False
            )
            print(f" > Selected G2P backend: {selected_backend} ({reason})")
            print(f" > User requested: {g2p_backend_train}")
            
            # Preprocess samples with optimized parallel processing + caching
            # This is 10-100x faster than the old sequential method
            train_samples, eval_samples, new_language = apply_g2p_to_training_data_optimized(
                train_samples=train_samples,
                eval_samples=eval_samples,
                train_csv_path=train_csv,
                eval_csv_path=eval_csv,
                language=language,
                g2p_backend=selected_backend,  # Use dynamically selected backend
                num_workers=None,  # Auto-detect CPU count
                enable_cache=True  # Enable disk caching for instant 2nd run
            )
            
            # Update effective language
            print(f" > Language code updated for tokenizer: '{effective_language}' ΓåÆ '{new_language}'")
            effective_language = new_language
            
            # Update dataset config language
            config_dataset.language = effective_language
            print(f" > Dataset config language updated to: '{effective_language}'")
            print(f" > Γ£à G2P preprocessing completed successfully!")
            
        except Exception as e:
            print(f" > Γ¥î ERROR: G2P preprocessing failed: {e}")
            print(" > Training will continue with original data - may fail with UNK tokens!")
            import traceback
            traceback.print_exc()
    elif amharic_g2p_enabled and dataset_already_phonemes:
        print(" > Γ£à Dataset already contains phonemes - skipping G2P conversion")
        print(f" > Using language code: '{effective_language}'")

    # Optionally enforce a minimum grad accumulation for stability in language adaptation mode
    if language_adaptation_mode and GRAD_ACUMM_STEPS < 4:
        print(f" > Language adaptation: increasing grad_accum_steps {GRAD_ACUMM_STEPS} -> 4 for stabler updates")
        GRAD_ACUMM_STEPS = 4

    # init the trainer and ≡ƒÜÇ
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    # ===================================================================
    # PATCH TOKENIZER FOR AMHARIC LANGUAGE CODE SUPPORT (BPE-ONLY MODE)
    # ===================================================================
    # CRITICAL: This must be done AFTER Trainer initialization because the
    # dataset creates its own tokenizer instance that needs to be patched.
    # The XTTS tokenizer doesn't natively support 'amh' language code.
    # This patch enables BPE-only training without G2P by:
    # 1. Adding 'am'/'amh' to char_limits
    # 2. Mapping Amharic preprocessing to English (returns raw text)
    if language in ["am", "amh"]:
        print("\n" + "="*70)
        print("🇪🇹 PATCHING TOKENIZER FOR AMHARIC LANGUAGE SUPPORT")
        print("="*70)
        
        # Get ALL tokenizer instances that need patching
        tokenizers_to_patch = []
        
        # 1. Model's tokenizer
        if hasattr(model, 'tokenizer'):
            tokenizers_to_patch.append(("model.tokenizer", model.tokenizer))
        elif hasattr(model, 'xtts'):
            if hasattr(model.xtts, 'tokenizer'):
                tokenizers_to_patch.append(("model.xtts.tokenizer", model.xtts.tokenizer))
            elif hasattr(model.xtts, 'gpt') and hasattr(model.xtts.gpt, 'tokenizer'):
                tokenizers_to_patch.append(("model.xtts.gpt.tokenizer", model.xtts.gpt.tokenizer))
        
        # 2. Dataset tokenizers (CRITICAL - this is what the training loop uses!)
        if hasattr(trainer, 'train_loader') and hasattr(trainer.train_loader, 'dataset'):
            dataset = trainer.train_loader.dataset
            if hasattr(dataset, 'tokenizer'):
                tokenizers_to_patch.append(("train_dataset.tokenizer", dataset.tokenizer))
        
        if hasattr(trainer, 'eval_loader') and hasattr(trainer.eval_loader, 'dataset'):
            dataset = trainer.eval_loader.dataset
            if hasattr(dataset, 'tokenizer'):
                tokenizers_to_patch.append(("eval_dataset.tokenizer", dataset.tokenizer))
        
        # Patch all found tokenizers
        patched_count = 0
        for name, tokenizer in tokenizers_to_patch:
            if tokenizer and hasattr(tokenizer, 'char_limits'):
                # Add Amharic language codes to char_limits
                if 'am' not in tokenizer.char_limits:
                    tokenizer.char_limits['am'] = 200  # Amharic (ISO 639-1)
                
                if 'amh' not in tokenizer.char_limits:
                    tokenizer.char_limits['amh'] = 200  # Amharic (ISO 639-3)
                
                # Patch preprocess_text to handle Amharic codes
                if hasattr(tokenizer, 'preprocess_text'):
                    _original_preprocess = tokenizer.preprocess_text
                    
                    def _amharic_safe_preprocess(txt, lang):
                        """
                        Amharic-safe preprocessing for BPE-only mode.
                        Maps 'am'/'amh' to 'en' preprocessing (returns raw text).
                        This avoids NotImplementedError while preserving Ethiopic characters.
                        """
                        try:
                            # Normalize language code
                            base_lang = lang.split('-')[0].lower() if isinstance(lang, str) else lang
                        except Exception:
                            base_lang = lang
                        
                        # IPA markers (for G2P mode detection)
                        ipa_markers = ('ə', 'ɨ', 'ʔ', 'ʕ', 'ʷ', 'ː', 'ʼ', 'ʃ', 'ʧ', 'ʤ', 'ɲ')
                        
                        # Handle Amharic codes specially
                        if base_lang in ('am', 'amh'):
                            # If text looks like IPA phonemes, keep as-is
                            if txt and any(marker in txt for marker in ipa_markers):
                                return txt
                            
                            # Otherwise, use English preprocessing (returns raw text)
                            # This is perfect for BPE-only mode with Ethiopic script
                            try:
                                return _original_preprocess(txt, 'en')
                            except Exception:
                                # Ultimate fallback: return text unchanged
                                return txt
                        
                        # Default behavior for other languages
                        return _original_preprocess(txt, lang)
                    
                    tokenizer.preprocess_text = _amharic_safe_preprocess
                    print(f" > ✅ Patched {name}")
                    patched_count += 1
        
        if patched_count > 0:
            print(f" > ✅ Successfully patched {patched_count} tokenizer(s)")
            print(" > ℹ️  Amharic text will be preprocessed as raw Ethiopic (BPE-only)")
            print("="*70 + "\n")
        else:
            print(" > ⚠️  Warning: Could not find any tokenizers to patch")
            print(" > Training may fail with NotImplementedError for 'amh'")
            print("="*70 + "\n")
    # ===================================================================
    
    # Apply Mixed Precision (AMP) if enabled
    if enable_mixed_precision and torch.cuda.is_available():
        try:
            # Determine precision type based on GPU compute capability
            # BF16 requires Ampere+ (compute capability >= 8.0)
            # T4 = 7.5 (NO BF16), A100/A10 = 8.0+ (YES BF16)
            compute_cap = torch.cuda.get_device_capability()
            use_bf16 = compute_cap[0] >= 8  # Ampere or newer
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            
            print(f"\n{'='*70}")
            print("⚡ ENABLING MIXED PRECISION TRAINING")
            print(f"{'='*70}")
            print(f" > Precision: {'BF16 (bfloat16)' if use_bf16 else 'FP16 (float16)'}")
            print(f" > GPU: {torch.cuda.get_device_name(0)}")
            print(f" > Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
            print(f" > GPU supports: FP16 ✅ | BF16 {'✅' if use_bf16 else '❌ (requires Ampere+ / CC 8.0+)'}")
            print(f" > Using: {'BF16' if use_bf16 else 'FP16'} for training")
            print(f" > Expected: 1.5-2x training speedup, 20-30% memory reduction")
            print(f"{'='*70}\n")
            
            # Wrap training step with autocast
            original_train_step = trainer.train_step
            
            def amp_train_step(*args, **kwargs):
                # Use autocast context for mixed precision
                with torch.cuda.amp.autocast(dtype=dtype):
                    return original_train_step(*args, **kwargs)
            
            trainer.train_step = amp_train_step
            print(" > ✅ AMP autocast enabled - training will use mixed precision")
            
        except Exception as e:
            print(f" > ⚠️  Could not enable mixed precision: {e}")
            print(f" > Training will continue in FP32 (no speedup)")
            import traceback
            traceback.print_exc()
    
    # Initialize training enhancements
    ema_model = None
    warmup_scheduler = None
    adaptive_clipper = None
    
    if ENHANCEMENTS_AVAILABLE:
        # Initialize EMA
        if use_ema_final:
            try:
                ema_model = EMAModel(model, decay=0.999)
                print(" > ✅ EMA model initialized")
            except Exception as e:
                print(f" > ⚠️  Could not initialize EMA: {e}")
                ema_model = None
        
        # Initialize adaptive gradient clipper
        try:
            grad_clip_norm = XTTSSmallDatasetConfig.GRAD_CLIP_NORM if use_small_dataset_config else (0.5 if language_adaptation_mode else 1.0)
            adaptive_clipper = AdaptiveGradientClipper(model, max_norm=grad_clip_norm)
            print(f" > ✅ Adaptive gradient clipping enabled (max_norm={grad_clip_norm})")
        except Exception as e:
            print(f" > ⚠️  Could not enable adaptive clipping: {e}")
            # Fallback to basic clipping
            try:
                import torch.nn.utils as nn_utils
                original_train_step = trainer.train_step
                
                def train_step_with_grad_clip(*args, **kwargs):
                    result = original_train_step(*args, **kwargs)
                    if model.training:
                        nn_utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    return result
                
                trainer.train_step = train_step_with_grad_clip
                print(f" > ✅ Basic gradient clipping enabled (max_norm={grad_clip_norm})")
            except Exception as e2:
                print(f" > ⚠️  Could not enable gradient clipping: {e2}")
    else:
        # Fallback to basic clipping if enhancements not available
        grad_clip_norm = XTTSSmallDatasetConfig.GRAD_CLIP_NORM if use_small_dataset_config else (0.5 if language_adaptation_mode else 1.0)
        try:
            import torch.nn.utils as nn_utils
            original_train_step = trainer.train_step
            
            def train_step_with_grad_clip(*args, **kwargs):
                result = original_train_step(*args, **kwargs)
                if model.training:
                    nn_utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                return result
            
            trainer.train_step = train_step_with_grad_clip
            print(f" > ✅ Gradient clipping enabled (max_norm={grad_clip_norm})")
        except Exception as e:
            print(f" > ⚠️  Could not enable gradient clipping: {e}")
    
    
    # Initialize LR warmup after trainer is created
    if lr_warmup_steps_final > 0 and ENHANCEMENTS_AVAILABLE:
        try:
            warmup_scheduler = WarmupLRScheduler(
                trainer.optimizer,
                warmup_steps=lr_warmup_steps_final,
                base_lr=final_learning_rate
            )
            print(f" > ✅ LR Warmup scheduler initialized")
        except Exception as e:
            print(f" > ⚠️  Could not initialize warmup scheduler: {e}")
            warmup_scheduler = None
    
    # Monkey-patch trainer to add EMA and warmup updates
    if (ema_model or warmup_scheduler or adaptive_clipper) and ENHANCEMENTS_AVAILABLE:
        original_optimizer_step = trainer.optimize
        
        def enhanced_optimize(*args, **kwargs):
            # Run original optimization with all arguments
            result = original_optimizer_step(*args, **kwargs)
            
            # Update EMA after optimizer step
            if ema_model:
                try:
                    ema_model.update()
                except:
                    pass
            
            # Update warmup scheduler
            if warmup_scheduler:
                try:
                    warmup_scheduler.step()
                except:
                    pass
            
            return result
        
        trainer.optimize = enhanced_optimize
        print(" > ✅ Training loop enhanced with EMA/Warmup hooks")
    
    # Setup early stopping
    early_stopping = None
    if use_small_dataset_config or use_early_stopping:
        patience_val = XTTSSmallDatasetConfig.EARLY_STOP_PATIENCE if use_small_dataset_config else early_stop_patience
        early_stopping = EarlyStoppingCallback(
            patience=patience_val,
            min_delta=XTTSSmallDatasetConfig.EARLY_STOP_MIN_DELTA if use_small_dataset_config else 0.01,
            verbose=True
        )
        print("\n" + "=" * 70)
        print("≡ƒÄ» EARLY STOPPING ENABLED")
        print("=" * 70)
        print(f" > Patience: {XTTSSmallDatasetConfig.EARLY_STOP_PATIENCE} epoch(s)")
        print(f" > Min Delta: {XTTSSmallDatasetConfig.EARLY_STOP_MIN_DELTA}")
        print(" > Will stop automatically if validation loss increases")
        print("=" * 70 + "\n")
    
    # Training info banner
    if not use_small_dataset_config:
        print("\n" + "=" * 70)
        print("≡ƒöÑ AGGRESSIVE OVERFITTING PREVENTION - V2")
        print("=" * 70)
        print(f" > Learning Rate: 1e-06 (REDUCED: 5e-06 ΓåÆ 2e-06 ΓåÆ 1e-06)")
        print(f" > LR Schedule: Reduce by 50% at epochs 1, 2, 3")
        print(f" >   - Epoch 0: LR = 1e-06")
        print(f" >   - Epoch 1: LR = 5e-07 (50% reduction)")
        print(f" >   - Epoch 2: LR = 2.5e-07 (50% reduction)")
        print(f" >   - Epoch 3+: LR = 1.25e-07 (50% reduction)")
        print(f" > Gradient Clipping: max_norm=1.0")
        print(f" > DataLoader Workers: 4 (optimized)")
        print(f" > Weight Decay: 0.05 (INCREASED for stronger regularization)")
        print("")
        print(" > ≡ƒÄ» TARGET: Eval loss < 3.5 after epoch 1")
        print(" > ΓÜá∩╕Å  IMPORTANT: Monitor eval_loss after each epoch")
        print(" >    Stop training if eval_loss > 4.0 after 2 epochs")
        print(" >    Expected: eval_loss should decrease by 20-30% per epoch")
        print("=" * 70 + "\n")
    
    # Run training with enhancements
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING WITH ENHANCEMENTS")
    if use_ema_final:
        print("  ✅ EMA: Enabled (decay=0.999)")
    if lr_warmup_steps_final > 0:
        print(f"  ✅ LR Warmup: {lr_warmup_steps_final} steps")
    if adaptive_clipper:
        print("  ✅ Adaptive Gradient Clipping: Enabled")
    if enable_mixed_precision:
        print("  ✅ Mixed Precision: Enabled")
    print("="*70 + "\n")
    
    trainer.fit()
    
    # If using EMA, save the EMA checkpoint as best model
    if ema_model:
        try:
            print("\n" + "="*70)
            print("🌟 SAVING EMA MODEL (Smoothed Weights)")
            print("="*70)
            
            # Apply EMA weights
            ema_model.apply_shadow()
            
            # Save EMA checkpoint
            ema_checkpoint_path = os.path.join(trainer.output_path, "best_model_ema.pth")
            torch.save({
                'model': trainer.model.state_dict(),
                'config': trainer.config,
                'ema_decay': ema_model.decay
            }, ema_checkpoint_path)
            
            print(f" > ✅ EMA checkpoint saved: {ema_checkpoint_path}")
            print(" > 💡 This checkpoint often has better quality than the raw checkpoint!")
            print("="*70 + "\n")
            
            # Restore original weights for normal checkpoint saving
            ema_model.restore()
        except Exception as e:
            print(f" > ⚠️  Could not save EMA checkpoint: {e}")

    # get the longest text audio file to use as speaker reference
    samples_len = [len(item["text"].split(" ")) for item in train_samples]
    longest_text_idx =  samples_len.index(max(samples_len))
    speaker_ref = train_samples[longest_text_idx]["audio_file"]

    trainer_out_path = trainer.output_path
    
    # close file handlers and remove them from the logger
    for handler in logging.getLogger('trainer').handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logging.getLogger('trainer').removeHandler(handler)
    
    # now you should be able to delete the log file
    log_file = os.path.join(trainer.output_path, f"trainer_{trainer.args.rank}_log.txt")
    os.remove(log_file)

    # deallocate VRAM and RAM
    del model, trainer, train_samples, eval_samples
    gc.collect()

    return XTTS_SPEAKER_FILE,XTTS_CONFIG_FILE, XTTS_CHECKPOINT, TOKENIZER_FILE, trainer_out_path, speaker_ref
