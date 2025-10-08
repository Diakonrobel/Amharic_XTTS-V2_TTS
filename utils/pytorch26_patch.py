"""
PyTorch 2.6 Compatibility Patch
================================

PyTorch 2.6+ changed the default value of `weights_only` parameter in torch.load
from False to True for security reasons. However, XTTS model checkpoints contain
custom classes that require `weights_only=False`.

This patch monkey-patches the trainer.io.load_fsspec function to add 
`weights_only=False` for trusted XTTS checkpoints.

SECURITY NOTE: Only use this with checkpoints from trusted sources (official XTTS models).
"""

import sys
import torch
import logging

logger = logging.getLogger(__name__)


def patch_torch_load_for_xtts():
    """
    Patch trainer.io.load_fsspec to work with PyTorch 2.6+
    
    This adds weights_only=False to torch.load calls in the TTS library
    to allow loading XTTS model checkpoints that contain custom classes.
    """
    try:
        import trainer.io
        
        # Store original function
        original_load_fsspec = trainer.io.load_fsspec
        
        def patched_load_fsspec(path, map_location=None, cache=False, **kwargs):
            """
            Patched version that adds weights_only=False for PyTorch 2.6+
            """
            # Check PyTorch version
            torch_version = torch.__version__.split('+')[0]  # Remove +cu118 etc
            major, minor = map(int, torch_version.split('.')[:2])
            
            # PyTorch 2.6+ requires weights_only parameter
            if major > 2 or (major == 2 and minor >= 6):
                # Add weights_only=False for trusted XTTS checkpoints
                kwargs['weights_only'] = False
                logger.info(f"PyTorch {torch_version} detected: Loading checkpoint with weights_only=False")
            
            # Call original function with updated kwargs
            return original_load_fsspec(path, map_location=map_location, cache=cache, **kwargs)
        
        # Replace the function
        trainer.io.load_fsspec = patched_load_fsspec
        
        logger.info("✅ Successfully patched trainer.io.load_fsspec for PyTorch 2.6+ compatibility")
        return True
        
    except ImportError:
        logger.warning("trainer.io module not found - patch not applied")
        return False
    except Exception as e:
        logger.error(f"Failed to patch trainer.io.load_fsspec: {e}")
        return False


def patch_torch_load_globally():
    """
    Alternative: Monkey-patch torch.load directly
    
    This is more aggressive but ensures all torch.load calls work with XTTS.
    Use only if the trainer.io patch doesn't work.
    """
    try:
        original_torch_load = torch.load
        
        def patched_torch_load(f, map_location=None, pickle_module=None, 
                               weights_only=None, mmap=None, **kwargs):
            """
            Patched torch.load that defaults to weights_only=False if not specified
            """
            # Check PyTorch version
            torch_version = torch.__version__.split('+')[0]
            major, minor = map(int, torch_version.split('.')[:2])
            
            # PyTorch 2.6+ defaults to weights_only=True
            if major > 2 or (major == 2 and minor >= 6):
                # If weights_only not specified, set to False for compatibility
                if weights_only is None:
                    weights_only = False
                    logger.debug("Setting weights_only=False for checkpoint loading")
            
            # Call original torch.load
            return original_torch_load(
                f=f,
                map_location=map_location,
                pickle_module=pickle_module,
                weights_only=weights_only,
                mmap=mmap,
                **kwargs
            )
        
        # Replace torch.load
        torch.load = patched_torch_load
        
        logger.info("✅ Successfully patched torch.load globally for PyTorch 2.6+ compatibility")
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch torch.load: {e}")
        return False


def apply_pytorch26_compatibility_patches():
    """
    Apply all necessary PyTorch 2.6 compatibility patches
    
    Returns:
        bool: True if patches applied successfully
    """
    logger.info("Applying PyTorch 2.6 compatibility patches...")
    
    # Try trainer.io patch first (more targeted)
    if patch_torch_load_for_xtts():
        return True
    
    # Fallback to global torch.load patch
    logger.info("Falling back to global torch.load patch...")
    return patch_torch_load_globally()


# Auto-apply patches on import if PyTorch 2.6+ detected
def _auto_patch():
    """Auto-apply patches when this module is imported"""
    try:
        torch_version = torch.__version__.split('+')[0]
        major, minor = map(int, torch_version.split('.')[:2])
        
        if major > 2 or (major == 2 and minor >= 6):
            logger.info(f"PyTorch {torch_version} detected - applying compatibility patches")
            apply_pytorch26_compatibility_patches()
        else:
            logger.info(f"PyTorch {torch_version} - no patches needed")
    except Exception as e:
        logger.warning(f"Could not determine PyTorch version or apply patches: {e}")


# Apply patches on module import
_auto_patch()
