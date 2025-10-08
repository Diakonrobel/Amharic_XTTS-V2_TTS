import logging
import os
import gc
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


def train_gpt(custom_model,version, language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path, max_audio_length=255995, use_amharic_g2p=False):
    #  Logging parameters
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    # print(f"XTTS version = {version}")

    # Set here the path that the checkpoints will be saved. Default: ./run/training/
    OUT_PATH = os.path.join(output_path, "run", "training")

    # Training Parameters
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
    START_WITH_EVAL = False  # if True it will star with evaluation
    BATCH_SIZE = batch_size  # set here the batch size
    GRAD_ACUMM_STEPS = grad_acumm  # set here the grad accumulation steps


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

    num_workers = 8
    if language == "ja":
        num_workers = 0
    
    # Handle Amharic-specific preprocessing - ROBUST IMPLEMENTATION
    # Accept both 'am' and 'amh' for G2P
    amharic_g2p_enabled = use_amharic_g2p and language in ["am", "amh", "en"]
    effective_language = language  # Will be updated if G2P preprocessing is applied
    
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
            else:
                print(" > Dataset contains Amharic script - will convert to phonemes")
                effective_language = language  # Keep original for now, will be updated after loading
        except Exception as e:
            print(f" > Warning: Could not check dataset preprocessing status: {e}")
        
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
            
            print(f" > ✅ Extended vocabulary created: {extended_vocab_path}")
            print(f" > This vocab includes Ethiopic chars + IPA phonemes + dataset-specific tokens")
            
        except Exception as e:
            print(f" > ⚠️  Warning: Could not extend vocabulary: {e}")
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
    # training parameters config
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
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
    )

    # Handle extended vocabulary - temporarily disable checkpoint loading, then reload manually
    checkpoint_to_load = None
    if extended_vocab_path:
        print(" > Extended vocabulary detected - will handle checkpoint loading manually...")
        checkpoint_to_load = model_args.xtts_checkpoint
        model_args.xtts_checkpoint = None  # Temporarily disable auto-loading
    
    # init the model from config (without checkpoint if extended vocab)
    model = GPTTrainer.init_from_config(config)
    
    # If using extended vocabulary, manually load checkpoint and resize embeddings
    if extended_vocab_path and checkpoint_to_load:
        print(f" > Loading checkpoint manually for vocab expansion: {checkpoint_to_load}")
        try:
            import json
            # Load extended vocab to get size
            with open(tokenizer_file_to_use, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
                new_vocab_size = len(vocab_data['model']['vocab'])
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_to_load, map_location="cpu")
            state_dict = checkpoint["model"]
            
            old_vocab_size = state_dict['gpt.text_embedding.weight'].shape[0]
            embed_dim = state_dict['gpt.text_embedding.weight'].shape[1]
            
            print(f" > Checkpoint vocab size: {old_vocab_size}")
            print(f" > Extended vocab size: {new_vocab_size}")
            print(f" > Will add {new_vocab_size - old_vocab_size} new token embeddings")
            
            # Create new extended embedding layers
            new_text_embedding = torch.nn.Embedding(new_vocab_size, embed_dim)
            new_text_embedding.weight.data[:old_vocab_size] = state_dict['gpt.text_embedding.weight']
            new_text_embedding.weight.data[old_vocab_size:] = torch.randn(new_vocab_size - old_vocab_size, embed_dim) * 0.02
            
            new_text_head = torch.nn.Linear(embed_dim, new_vocab_size)
            new_text_head.weight.data[:old_vocab_size] = state_dict['gpt.text_head.weight']
            new_text_head.weight.data[old_vocab_size:] = torch.randn(new_vocab_size - old_vocab_size, embed_dim) * 0.02
            new_text_head.bias.data[:old_vocab_size] = state_dict['gpt.text_head.bias']
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
            print(f" > ⚠️  Error in extended vocab checkpoint loading: {e}")
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
    
    # Apply G2P preprocessing if enabled and needed
    if amharic_g2p_enabled and effective_language != "en":
        print(" > Applying Amharic G2P preprocessing to training data...")
        try:
            from utils.amharic_g2p_dataset_wrapper import apply_g2p_to_training_data
            
            # Preprocess samples and get effective language
            train_samples, eval_samples, new_language = apply_g2p_to_training_data(
                train_samples=train_samples,
                eval_samples=eval_samples,
                train_csv_path=train_csv,
                eval_csv_path=eval_csv,
                language=language,
                g2p_backend="transphone"
            )
            
            # Update effective language
            print(f" > Language code updated for tokenizer: '{effective_language}' → '{new_language}'")
            effective_language = new_language
            
            # Update dataset config language
            config_dataset.language = effective_language
            print(f" > Dataset config language updated to: '{effective_language}'")
            
        except Exception as e:
            print(f" > ERROR: G2P preprocessing failed: {e}")
            print(" > Training will continue with original data - may fail with UNK tokens!")
            import traceback
            traceback.print_exc()

    # init the trainer and 🚀
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
    trainer.fit()

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
