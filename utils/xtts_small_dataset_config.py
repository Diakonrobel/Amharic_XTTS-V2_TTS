#!/usr/bin/env python3
"""
XTTS v2 Small Dataset Training Configuration
=============================================

Optimized configuration for fine-tuning XTTS v2 with small datasets (1-3 hours of audio).
Based on best practices from Coqui TTS community and successful small-dataset fine-tuning experiments.

Key Techniques:
1. Layer Freezing - Only train last few layers
2. Very Low Learning Rate - 5e-7 to 1e-7
3. Minimal Epochs - 2-3 epochs max
4. Early Stopping - Stop on validation loss increase
5. Data Augmentation - Pitch/time/noise variations
6. High Dropout - Prevent memorization
7. Low Batch Size - Better gradients for small data
8. High Gradient Accumulation - Stable updates

References:
- https://github.com/coqui-ai/TTS/discussions/2827
- https://github.com/daswer123/xtts-finetune-webui
- Community best practices for <5k samples
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class XTTSSmallDatasetConfig:
    """
    Configuration optimized for training XTTS v2 with small datasets.
    
    Designed for datasets with:
    - 1,000 - 3,000 samples
    - 1-3 hours of audio
    - Single or few speakers
    """
    
    # ====================================================================
    # CORE TRAINING PARAMETERS - OPTIMIZED FOR SMALL DATASETS
    # ====================================================================
    
    # Batch size - CRITICAL: Use 1-2 for small datasets
    BATCH_SIZE = 1  # Smaller batch = better gradients for limited data
    GRAD_ACCUM_STEPS = 16  # Effective batch size = 1 * 16 = 16
    
    # Learning rate - CRITICAL: Much lower than default
    LEARNING_RATE = 5e-7  # 10x lower than standard (5e-6)
    
    # Epochs - CRITICAL: Very few epochs to prevent overfitting
    MAX_EPOCHS = 2  # Only 2 epochs for <2000 samples
    
    # Early stopping - Stop if validation loss increases
    EARLY_STOP_PATIENCE = 1  # Stop after 1 epoch of val loss increase
    EARLY_STOP_MIN_DELTA = 0.01  # Minimum improvement threshold
    
    # ====================================================================
    # LAYER FREEZING - ONLY TRAIN WHAT'S NEEDED
    # ====================================================================
    
    # Freeze most layers, only train:
    # - Text embedding layers (for new language tokens)
    # - Last 2-3 GPT layers (for adaptation)
    # - Text head (output layer)
    FREEZE_ENCODER = True  # Freeze audio encoder completely
    FREEZE_FIRST_N_GPT_LAYERS = 28  # Freeze first 28 of 30 layers (only train last 2)
    TRAIN_TEXT_EMBEDDING = True  # Always train text embeddings
    TRAIN_TEXT_HEAD = True  # Always train output layer
    
    # ====================================================================
    # REGULARIZATION - PREVENT OVERFITTING
    # ====================================================================
    
    # Weight decay - L2 regularization
    WEIGHT_DECAY = 0.1  # Higher than default (0.01) for stronger regularization
    
    # Dropout rates - Increase to prevent memorization
    GPT_DROPOUT = 0.2  # Higher dropout in GPT layers
    EMBEDDING_DROPOUT = 0.1  # Dropout after embeddings
    
    # Gradient clipping - Prevent exploding gradients
    GRAD_CLIP_NORM = 0.5  # Lower than default (1.0) for stability
    
    # ====================================================================
    # DATA AUGMENTATION
    # ====================================================================
    
    # Audio augmentation probabilities
    AUGMENT_PITCH_SHIFT = True
    PITCH_SHIFT_RANGE = (-2, 2)  # Semitones
    PITCH_SHIFT_PROB = 0.3  # 30% of samples
    
    AUGMENT_TIME_STRETCH = True
    TIME_STRETCH_RANGE = (0.9, 1.1)  # 90%-110% speed
    TIME_STRETCH_PROB = 0.3
    
    AUGMENT_ADD_NOISE = True
    NOISE_LEVEL_RANGE = (0.001, 0.01)  # Very subtle noise
    ADD_NOISE_PROB = 0.2  # 20% of samples
    
    # ====================================================================
    # LEARNING RATE SCHEDULE
    # ====================================================================
    
    # Use cosine annealing with warm restarts for better convergence
    LR_SCHEDULER = "CosineAnnealingWarmRestarts"
    LR_SCHEDULER_PARAMS = {
        "T_0": 500,  # Restart every 500 steps (~half epoch)
        "T_mult": 2,  # Double the restart period each time
        "eta_min": 1e-8,  # Minimum learning rate
    }
    
    # Alternative: MultiStepLR (original aggressive schedule)
    # LR_SCHEDULER = "MultiStepLR"
    # LR_SCHEDULER_PARAMS = {
    #     "milestones": [1000],  # Reduce at end of epoch 1
    #     "gamma": 0.5,  # Cut LR in half
    # }
    
    # ====================================================================
    # CHECKPOINT SAVING
    # ====================================================================
    
    SAVE_STEP = 500  # Save every 500 steps
    SAVE_N_CHECKPOINTS = 3  # Keep best 3 checkpoints
    SAVE_BEST_ONLY = True  # Only save if validation improves
    
    # ====================================================================
    # MONITORING
    # ====================================================================
    
    PRINT_STEP = 50  # Print every 50 steps
    EVAL_STEP = 500  # Evaluate every 500 steps (end of epoch for ~1000 samples)
    LOG_STEP = 25  # Log to tensorboard every 25 steps
    
    @classmethod
    def apply_layer_freezing(cls, model) -> Tuple[int, int]:
        """
        Freeze layers according to configuration.
        
        Returns:
            Tuple of (total_params, trainable_params)
        """
        total_params = 0
        trainable_params = 0
        
        # Freeze encoder if specified
        if cls.FREEZE_ENCODER and hasattr(model, 'xtts'):
            if hasattr(model.xtts, 'mel_encoder'):
                for param in model.xtts.mel_encoder.parameters():
                    param.requires_grad = False
                logger.info("✓ Froze mel_encoder")
            
            if hasattr(model.xtts, 'dvae'):
                for param in model.xtts.dvae.parameters():
                    param.requires_grad = False
                logger.info("✓ Froze dvae")
        
        # Freeze first N GPT layers
        if cls.FREEZE_FIRST_N_GPT_LAYERS > 0 and hasattr(model, 'xtts'):
            if hasattr(model.xtts, 'gpt'):
                gpt_model = model.xtts.gpt
                
                # Freeze transformer layers
                if hasattr(gpt_model, 'transformer'):
                    for i, layer in enumerate(gpt_model.transformer.h):
                        if i < cls.FREEZE_FIRST_N_GPT_LAYERS:
                            for param in layer.parameters():
                                param.requires_grad = False
                    logger.info(f"✓ Froze first {cls.FREEZE_FIRST_N_GPT_LAYERS} GPT layers")
                
                # Always train text embeddings and head (needed for new tokens)
                if hasattr(gpt_model, 'text_embedding'):
                    for param in gpt_model.text_embedding.parameters():
                        param.requires_grad = True
                    logger.info("✓ Enabled training for text_embedding")
                
                if hasattr(gpt_model, 'text_head'):
                    for param in gpt_model.text_head.parameters():
                        param.requires_grad = True
                    logger.info("✓ Enabled training for text_head")
        
        # Count parameters
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"LAYER FREEZING SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total parameters:      {total_params:,}")
        logger.info(f"Trainable parameters:  {trainable_params:,}")
        logger.info(f"Frozen parameters:     {total_params - trainable_params:,}")
        logger.info(f"Training only:         {100 * trainable_params / total_params:.2f}% of model")
        logger.info(f"{'='*70}\n")
        
        return total_params, trainable_params
    
    @classmethod
    def get_optimizer_config(cls) -> Dict:
        """Get optimizer configuration."""
        return {
            "optimizer": "AdamW",
            "optimizer_params": {
                "betas": [0.9, 0.96],
                "eps": 1e-8,
                "weight_decay": cls.WEIGHT_DECAY,
            },
            "lr": cls.LEARNING_RATE,
        }
    
    @classmethod
    def get_scheduler_config(cls) -> Dict:
        """Get learning rate scheduler configuration."""
        return {
            "lr_scheduler": cls.LR_SCHEDULER,
            "lr_scheduler_params": cls.LR_SCHEDULER_PARAMS,
        }
    
    @classmethod
    def print_config_summary(cls):
        """Print configuration summary."""
        print("\n" + "="*70)
        print("🎯 XTTS v2 SMALL DATASET TRAINING CONFIGURATION")
        print("="*70)
        print(f"Batch Size:              {cls.BATCH_SIZE}")
        print(f"Gradient Accumulation:   {cls.GRAD_ACCUM_STEPS}")
        print(f"Effective Batch Size:    {cls.BATCH_SIZE * cls.GRAD_ACCUM_STEPS}")
        print(f"Learning Rate:           {cls.LEARNING_RATE}")
        print(f"Max Epochs:              {cls.MAX_EPOCHS}")
        print(f"Weight Decay:            {cls.WEIGHT_DECAY}")
        print(f"Gradient Clip:           {cls.GRAD_CLIP_NORM}")
        print()
        print("Layer Freezing:")
        print(f"  - Freeze Encoder:      {cls.FREEZE_ENCODER}")
        print(f"  - Freeze GPT Layers:   First {cls.FREEZE_FIRST_N_GPT_LAYERS}")
        print(f"  - Train Embeddings:    {cls.TRAIN_TEXT_EMBEDDING}")
        print(f"  - Train Text Head:     {cls.TRAIN_TEXT_HEAD}")
        print()
        print("Data Augmentation:")
        print(f"  - Pitch Shift:         {cls.AUGMENT_PITCH_SHIFT} (prob={cls.PITCH_SHIFT_PROB})")
        print(f"  - Time Stretch:        {cls.AUGMENT_TIME_STRETCH} (prob={cls.TIME_STRETCH_PROB})")
        print(f"  - Add Noise:           {cls.AUGMENT_ADD_NOISE} (prob={cls.ADD_NOISE_PROB})")
        print()
        print("Early Stopping:")
        print(f"  - Patience:            {cls.EARLY_STOP_PATIENCE} epoch(s)")
        print(f"  - Min Delta:           {cls.EARLY_STOP_MIN_DELTA}")
        print("="*70 + "\n")


class EarlyStoppingCallback:
    """
    Early stopping callback to prevent overfitting.
    
    Stops training if validation loss doesn't improve for patience epochs.
    """
    
    def __init__(self, patience: int = 1, min_delta: float = 0.01, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.should_stop = False
    
    def __call__(self, current_loss: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_loss: Current validation loss
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if current_loss < (self.best_loss - self.min_delta):
            # Improvement
            self.best_loss = current_loss
            self.epochs_without_improvement = 0
            if self.verbose:
                print(f"\n✅ Validation loss improved to {current_loss:.4f} (best so far)")
        else:
            # No improvement
            self.epochs_without_improvement += 1
            if self.verbose:
                print(f"\n⚠️  No improvement in validation loss (epoch {epoch})")
                print(f"   Current: {current_loss:.4f} | Best: {self.best_loss:.4f}")
                print(f"   Epochs without improvement: {self.epochs_without_improvement}/{self.patience}")
            
            if self.epochs_without_improvement >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"\n🛑 EARLY STOPPING TRIGGERED")
                    print(f"   Validation loss has not improved for {self.patience} epoch(s)")
                    print(f"   Best validation loss: {self.best_loss:.4f}")
                    print(f"   Stopping training to prevent overfitting...")
                return True
        
        return False
    
    def reset(self):
        """Reset the callback state."""
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.should_stop = False
