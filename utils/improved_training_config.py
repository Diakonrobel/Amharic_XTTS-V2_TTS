"""
Improved Training Configuration for XTTS
Fixes overfitting issues with early stopping, LR scheduling, and regularization

Based on diagnosis: .warp/training_diagnosis.md
Implementation guide: .warp/training_fixes.md
"""

import torch
from typing import Dict, Optional


class ImprovedTrainingConfig:
    """
    Enhanced training configuration to prevent overfitting
    and improve model generalization
    """
    
    def __init__(self):
        # Early Stopping Configuration
        self.early_stopping = {
            'enabled': True,
            'monitor': 'eval_loss',  # Watch validation loss
            'patience': 10,           # Stop after 10 epochs without improvement
            'min_delta': 0.001,       # Minimum change to qualify as improvement
            'mode': 'min',            # Minimize loss
            'verbose': True,
            'restore_best_weights': True
        }
        
        # Learning Rate Scheduler Configuration
        self.lr_scheduler = {
            'type': 'ReduceLROnPlateau',  # Simpler and more reliable
            'monitor': 'eval_loss',
            'factor': 0.5,                 # Reduce LR by 50%
            'patience': 5,                 # After 5 epochs without improvement
            'min_lr': 1e-07,              # Minimum learning rate
            'verbose': True,
            'mode': 'min'
        }
        
        # Alternative: Cosine Annealing (for more advanced use)
        self.lr_scheduler_cosine = {
            'type': 'CosineAnnealingWarmRestarts',
            'T_0': 10,                    # Restart every 10 epochs
            'T_mult': 2,                  # Double period after each restart
            'eta_min': 1e-07,            # Minimum LR
        }
        
        # Gradient Clipping
        self.gradient_clipping = {
            'enabled': True,
            'max_norm': 1.0,              # Clip to max norm of 1.0
            'norm_type': 2.0              # L2 norm
        }
        
        # Regularization
        self.regularization = {
            'weight_decay': 0.01,         # L2 regularization (already in config)
            'dropout': 0.1,               # Will need model architecture changes
            'attention_dropout': 0.1,     # For attention layers
            'label_smoothing': 0.1,       # Smooth hard targets
        }
        
        # Checkpoint Strategy
        self.checkpoint = {
            'monitor': 'eval_loss',       # Use validation loss for best model
            'save_top_k': 3,              # Keep top 3 checkpoints
            'mode': 'min',
            'filename': 'best-epoch{epoch:02d}-valloss{val_loss:.4f}',
            'save_best_after_epoch': 0    # Start tracking from epoch 0
        }
        
        # Training Monitoring
        self.monitoring = {
            'check_val_every_n_epoch': 1,  # Validate every epoch
            'log_every_n_steps': 50,       # Log frequently
            'track_grad_norm': True,       # Monitor gradient norm
            'val_check_interval': 1.0,     # Validate at end of each epoch
        }
    
    def get_lr_scheduler_params(self, use_cosine=False):
        """Get LR scheduler parameters for GPTTrainerConfig"""
        if use_cosine:
            return {
                'scheduler': 'CosineAnnealingWarmRestarts',
                'T_0': self.lr_scheduler_cosine['T_0'],
                'T_mult': self.lr_scheduler_cosine['T_mult'],
                'eta_min': self.lr_scheduler_cosine['eta_min'],
            }
        else:
            # Use ReduceLROnPlateau (default, more reliable)
            return {
                'scheduler': 'ReduceLROnPlateau',
                'factor': self.lr_scheduler['factor'],
                'patience': self.lr_scheduler['patience'],
                'min_lr': self.lr_scheduler['min_lr'],
                'mode': self.lr_scheduler['mode'],
            }
    
    def apply_gradient_clipping_to_trainer(self, trainer):
        """
        Apply gradient clipping to trainer
        Call this after trainer initialization
        """
        if not self.gradient_clipping['enabled']:
            return
        
        # Store original optimizer step
        original_step = trainer.optimizer.step
        
        def step_with_clipping(*args, **kwargs):
            # Clip gradients before optimizer step
            torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(),
                max_norm=self.gradient_clipping['max_norm'],
                norm_type=self.gradient_clipping['norm_type']
            )
            return original_step(*args, **kwargs)
        
        trainer.optimizer.step = step_with_clipping
        print(f" > ✅ Gradient clipping enabled (max_norm={self.gradient_clipping['max_norm']})")


class EarlyStoppingCallback:
    """
    Early stopping callback to prevent overfitting
    Monitors validation loss and stops training when it stops improving
    """
    
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, current_loss, current_epoch):
        """
        Check if training should stop
        
        Args:
            current_loss: Current validation loss
            current_epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = current_loss
            self.best_epoch = current_epoch
            if self.verbose:
                print(f" > Early Stopping: Baseline set at epoch {current_epoch}, loss={current_loss:.4f}")
            return False
        
        # Check if loss improved by at least min_delta
        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.best_epoch = current_epoch
            self.counter = 0
            if self.verbose:
                print(f" > Early Stopping: Improvement detected at epoch {current_epoch}, loss={current_loss:.4f}")
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f" > Early Stopping: No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f" > ⛔ EARLY STOPPING TRIGGERED at epoch {current_epoch}")
                    print(f" > Best epoch was {self.best_epoch} with loss={self.best_loss:.4f}")
                    print(f" > Training stopped to prevent further overfitting")
                return True
            
            return False
    
    def get_best_info(self):
        """Get information about best epoch"""
        return {
            'best_epoch': self.best_epoch,
            'best_loss': self.best_loss,
            'stopped_early': self.early_stop
        }


class TrainingMetricsMonitor:
    """
    Monitor training metrics and detect overfitting patterns
    """
    
    def __init__(self, train_val_gap_threshold=3.0):
        self.train_losses = []
        self.val_losses = []
        self.train_val_gap_threshold = train_val_gap_threshold
        self.warnings = []
    
    def add_metrics(self, train_loss, val_loss, epoch):
        """Add metrics for an epoch"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Check for overfitting patterns
        self._check_overfitting_patterns(train_loss, val_loss, epoch)
    
    def _check_overfitting_patterns(self, train_loss, val_loss, epoch):
        """Detect overfitting patterns and issue warnings"""
        
        # Check train/val gap
        if train_loss > 0:
            gap = val_loss / train_loss
            if gap > self.train_val_gap_threshold:
                warning = f"🚩 OVERFITTING WARNING at epoch {epoch}: Train/Val gap = {gap:.2f}x (threshold: {self.train_val_gap_threshold}x)"
                self.warnings.append(warning)
                print(f" > {warning}")
        
        # Check if val loss is increasing while train loss decreases
        if len(self.train_losses) >= 3:
            recent_train = self.train_losses[-3:]
            recent_val = self.val_losses[-3:]
            
            train_decreasing = all(recent_train[i] > recent_train[i+1] for i in range(len(recent_train)-1))
            val_increasing = all(recent_val[i] < recent_val[i+1] for i in range(len(recent_val)-1))
            
            if train_decreasing and val_increasing:
                warning = f"🚩 DIVERGENCE WARNING at epoch {epoch}: Train loss decreasing but Val loss increasing"
                self.warnings.append(warning)
                print(f" > {warning}")
    
    def get_summary(self):
        """Get training summary"""
        if not self.val_losses:
            return "No metrics recorded"
        
        best_val_epoch = self.val_losses.index(min(self.val_losses))
        final_gap = self.val_losses[-1] / self.train_losses[-1] if self.train_losses[-1] > 0 else 0
        
        summary = {
            'best_val_epoch': best_val_epoch,
            'best_val_loss': self.val_losses[best_val_epoch],
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_train_val_gap': final_gap,
            'total_warnings': len(self.warnings),
            'warnings': self.warnings
        }
        
        return summary


def create_improved_gpt_config(
    base_config_params: Dict,
    use_early_stopping: bool = True,
    use_lr_scheduling: bool = True,
    use_gradient_clipping: bool = True,
    use_cosine_lr: bool = False
):
    """
    Create improved GPTTrainerConfig with overfitting prevention
    
    Args:
        base_config_params: Dictionary of base config parameters
        use_early_stopping: Enable early stopping
        use_lr_scheduling: Enable LR scheduling
        use_gradient_clipping: Enable gradient clipping
        use_cosine_lr: Use cosine annealing instead of ReduceLROnPlateau
    
    Returns:
        Tuple of (config_params, improved_config, early_stopping_callback)
    """
    improved_config = ImprovedTrainingConfig()
    
    # Update LR scheduler
    if use_lr_scheduling:
        if use_cosine_lr:
            base_config_params['lr_scheduler'] = 'CosineAnnealingWarmRestarts'
            base_config_params['lr_scheduler_params'] = {
                'T_0': improved_config.lr_scheduler_cosine['T_0'],
                'T_mult': improved_config.lr_scheduler_cosine['T_mult'],
                'eta_min': improved_config.lr_scheduler_cosine['eta_min'],
            }
        else:
            base_config_params['lr_scheduler'] = 'ReduceLROnPlateau'
            base_config_params['lr_scheduler_params'] = {
                'factor': improved_config.lr_scheduler['factor'],
                'patience': improved_config.lr_scheduler['patience'],
                'min_lr': improved_config.lr_scheduler['min_lr'],
                'mode': improved_config.lr_scheduler['mode'],
            }
        print(" > ✅ Learning rate scheduling enabled")
    
    # Create early stopping callback
    early_stopping_callback = None
    if use_early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            patience=improved_config.early_stopping['patience'],
            min_delta=improved_config.early_stopping['min_delta'],
            verbose=improved_config.early_stopping['verbose']
        )
        print(" > ✅ Early stopping enabled (patience=10 epochs)")
    
    if use_gradient_clipping:
        print(" > ✅ Gradient clipping will be enabled (max_norm=1.0)")
    
    return base_config_params, improved_config, early_stopping_callback


# Example usage documentation
if __name__ == "__main__":
    print("=" * 60)
    print("Improved Training Configuration for XTTS")
    print("=" * 60)
    
    config = ImprovedTrainingConfig()
    
    print("\n📋 Configuration Summary:\n")
    print(f"Early Stopping:")
    print(f"  - Enabled: {config.early_stopping['enabled']}")
    print(f"  - Patience: {config.early_stopping['patience']} epochs")
    print(f"  - Min Delta: {config.early_stopping['min_delta']}")
    
    print(f"\nLearning Rate Scheduler:")
    print(f"  - Type: {config.lr_scheduler['type']}")
    print(f"  - Patience: {config.lr_scheduler['patience']} epochs")
    print(f"  - Factor: {config.lr_scheduler['factor']}")
    print(f"  - Min LR: {config.lr_scheduler['min_lr']}")
    
    print(f"\nGradient Clipping:")
    print(f"  - Enabled: {config.gradient_clipping['enabled']}")
    print(f"  - Max Norm: {config.gradient_clipping['max_norm']}")
    
    print(f"\nRegularization:")
    print(f"  - Weight Decay: {config.regularization['weight_decay']}")
    print(f"  - Dropout: {config.regularization['dropout']}")
    
    print("\n" + "=" * 60)
    print("✅ Configuration loaded successfully!")
    print("=" * 60)
