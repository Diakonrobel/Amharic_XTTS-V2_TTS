"""
Small Dataset Training Configuration V2
Aggressive layer freezing + early stopping for datasets < 20 hours

This config freezes 95% of the model and only trains:
- Last 2 GPT layers
- Speaker embeddings
- Text embeddings (lightly)

Designed specifically for datasets under 20 hours.
"""

from trainer import TrainerArgs


def get_aggressive_small_dataset_config(
    output_path: str,
    num_epochs: int = 10,  # Can train longer with freezing
    batch_size: int = 2,
    grad_acumm: int = 2,  # Effective batch size = 4
    lr: float = 5e-6,  # Higher LR for unfrozen layers
):
    """
    Get training configuration optimized for small datasets (< 20 hours).
    
    Key features:
    - Freezes first 10/12 GPT layers (83% of model)
    - Only trains last 2 layers + embeddings
    - Early stopping with patience=2
    - Aggressive regularization
    - Gradient clipping
    
    Args:
        output_path: Where to save checkpoints
        num_epochs: Max epochs (early stopping will likely stop sooner)
        batch_size: Batch size per GPU
        grad_acumm: Gradient accumulation steps
        lr: Learning rate for trainable layers
    
    Returns:
        dict: Training configuration
    """
    
    return {
        "output_path": output_path,
        "run_name": "small_dataset_v2",
        
        # Dataset settings
        "batch_size": batch_size,
        "eval_batch_size": batch_size,
        "num_loader_workers": 4,
        "compute_input_seq_cache": True,
        "use_noise_augment": True,  # CRITICAL for small datasets
        
        # Training schedule
        "epochs": num_epochs,
        "grad_clip": 1.0,  # Prevent gradient explosion
        "gradient_accumulation_steps": grad_acumm,
        
        # Learning rate
        "lr": lr,
        "lr_scheduler": "MultiStepLR",
        "lr_scheduler_params": {
            "milestones": [int(num_epochs * 0.6), int(num_epochs * 0.8)],
            "gamma": 0.5  # Reduce LR at 60% and 80% of training
        },
        
        # Optimizer
        "optimizer": "AdamW",
        "optimizer_params": {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01  # L2 regularization
        },
        
        # Mixed precision (faster training)
        "mixed_precision": True,
        
        # Checkpointing (save frequently to catch best model)
        "save_step": 500,  # Save every 500 steps
        "save_n_checkpoints": 3,  # Keep last 3 checkpoints
        "save_best_after": 1000,  # Start saving best after 1000 steps
        
        # Evaluation
        "eval_split_size": 0.1,  # 10% for validation
        "print_step": 50,
        "plot_step": 500,
        "log_model_step": 1000,
        
        # CRITICAL: Early stopping
        "early_stopping": {
            "enabled": True,
            "patience": 2,  # Stop if eval loss doesn't improve for 2 epochs
            "min_delta": 0.001,  # Minimum improvement to count
            "monitor": "eval_loss",  # Watch validation loss
            "mode": "min",  # Lower is better
        },
        
        # CRITICAL: Layer freezing
        "freeze_layers": [
            # Freeze first 10 of 12 GPT layers (83% of model)
            "gpt.gpt.transformer.h.0",
            "gpt.gpt.transformer.h.1",
            "gpt.gpt.transformer.h.2",
            "gpt.gpt.transformer.h.3",
            "gpt.gpt.transformer.h.4",
            "gpt.gpt.transformer.h.5",
            "gpt.gpt.transformer.h.6",
            "gpt.gpt.transformer.h.7",
            "gpt.gpt.transformer.h.8",
            "gpt.gpt.transformer.h.9",
            # Train only layers 10, 11 (last 2 layers)
        ],
        
        # Model-specific settings
        "gpt_use_masking_gt_prompt_approach": True,
        "gpt_use_perceiver_resampler": True,
        
        # Audio settings
        "mel_norm_file": None,  # Will use base model's normalization
        
        # Test sentences for monitoring
        "test_sentences": [
            {"text": "Hello, this is a test.", "speaker_wav": None},
            {"text": "The model is being fine-tuned.", "speaker_wav": None},
        ],
    }


def apply_layer_freezing_callback():
    """
    Callback to freeze layers during training initialization.
    Call this in your training script AFTER model initialization.
    
    Returns:
        Callback function that freezes specified layers
    """
    def freeze_callback(trainer):
        print("\n" + "="*60)
        print("APPLYING AGGRESSIVE LAYER FREEZING")
        print("="*60)
        
        frozen_params = 0
        trainable_params = 0
        
        # Get the model
        model = trainer.model
        
        # Freeze embeddings partially (allow fine-tuning but with small gradients)
        if hasattr(model, 'gpt'):
            gpt = model.gpt
            
            # Freeze first 10 GPT transformer layers
            for i in range(10):
                layer_name = f"gpt.transformer.h.{i}"
                if hasattr(gpt, 'gpt') and hasattr(gpt.gpt, 'transformer'):
                    try:
                        layer = gpt.gpt.transformer.h[i]
                        for param in layer.parameters():
                            param.requires_grad = False
                            frozen_params += param.numel()
                        print(f"  ✓ Frozen layer {i}")
                    except (AttributeError, IndexError):
                        print(f"  ⚠ Could not freeze layer {i}")
            
            # Keep last 2 layers trainable (layers 10, 11)
            for i in range(10, 12):
                try:
                    layer = gpt.gpt.transformer.h[i]
                    for param in layer.parameters():
                        param.requires_grad = True
                        trainable_params += param.numel()
                    print(f"  ✓ Layer {i} TRAINABLE")
                except (AttributeError, IndexError):
                    print(f"  ⚠ Layer {i} not found")
            
            # Keep embeddings trainable but with lower learning rate
            if hasattr(gpt, 'text_embedding'):
                for param in gpt.text_embedding.parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()
                print("  ✓ Text embeddings TRAINABLE")
            
            if hasattr(gpt, 'text_head'):
                for param in gpt.text_head.parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()
                print("  ✓ Text head TRAINABLE")
        
        total_params = frozen_params + trainable_params
        print(f"\n📊 Parameter Summary:")
        print(f"  Total: {total_params:,}")
        print(f"  Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print("="*60 + "\n")
        
        return trainer
    
    return freeze_callback


def get_optimizer_params_with_different_lr(model, base_lr=5e-6, embedding_lr=1e-6):
    """
    Create parameter groups with different learning rates.
    
    - Unfrozen GPT layers: base_lr (5e-6)
    - Embeddings: embedding_lr (1e-6, more conservative)
    
    Args:
        model: The XTTS model
        base_lr: Learning rate for main trainable layers
        embedding_lr: Learning rate for embeddings (lower to prevent drift)
    
    Returns:
        List of parameter groups for optimizer
    """
    param_groups = []
    
    # Collect parameters by type
    embedding_params = []
    layer_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'embedding' in name.lower() or 'text_head' in name:
            embedding_params.append(param)
        elif 'transformer.h.1' in name:  # Last 2 layers (10, 11)
            layer_params.append(param)
        else:
            other_params.append(param)
    
    # Create parameter groups
    if embedding_params:
        param_groups.append({
            'params': embedding_params,
            'lr': embedding_lr,
            'name': 'embeddings'
        })
        print(f"  📌 Embedding params: LR={embedding_lr}")
    
    if layer_params:
        param_groups.append({
            'params': layer_params,
            'lr': base_lr,
            'name': 'layers'
        })
        print(f"  📌 Layer params: LR={base_lr}")
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'name': 'other'
        })
        print(f"  📌 Other params: LR={base_lr}")
    
    return param_groups


if __name__ == "__main__":
    # Example usage
    config = get_aggressive_small_dataset_config(
        output_path="./finetune_models",
        num_epochs=10,
        batch_size=2,
        grad_acumm=2,
        lr=5e-6
    )
    
    print("Configuration for small dataset (<20 hours):")
    print(f"  Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  Max epochs: {config['epochs']}")
    print(f"  Early stopping: {config['early_stopping']['patience']} epochs")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Frozen layers: {len(config['freeze_layers'])}/12 (83%)")
