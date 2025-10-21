"""
Pure BPE Training Configuration for Amharic TTS
================================================

This configuration explicitly disables G2P and uses only BPE tokenization
on raw Ethiopic script. This is useful when:
- G2P backends are not available
- You want faster training without phoneme preprocessing
- Your dataset already has good pronunciation coverage
- You prefer the model to learn character-level representations

Benefits of BPE-only:
- Faster preprocessing (no G2P conversion)
- No external dependencies (transphone, epitran)
- Direct learning from Ethiopic characters
- Works well for languages with syllabic scripts

Trade-offs:
- May have less accurate pronunciation for rare words
- Larger vocabulary size for Ethiopic characters
- Model must learn phonological patterns implicitly
"""

from dataclasses import dataclass, field
from typing import Optional
from .amharic_config import (
    AmharicTTSConfig,
    TokenizerMode,
    TokenizerConfiguration,
    G2PConfiguration,
    G2PBackend
)


@dataclass
class BPEOnlyConfiguration:
    """
    Configuration for pure BPE training without G2P preprocessing
    
    This explicitly disables all G2P backends and uses raw Ethiopic text
    with standard BPE tokenization.
    """
    
    # Disable G2P completely
    use_g2p: bool = False
    
    # Tokenizer settings for raw BPE
    tokenizer_mode: TokenizerMode = TokenizerMode.RAW_BPE
    vocab_size: int = 1024  # Standard BPE vocab size
    
    # Ensure no G2P backends are attempted
    g2p_backends_disabled: bool = True
    
    # Training parameters optimized for character-level learning
    learning_rate: float = 5e-6  # Slightly higher for character learning
    batch_size: int = 2
    grad_accum_steps: int = 84
    
    # Text preprocessing (non-G2P)
    normalize_ethiopic_variants: bool = True  # ሥ→ስ, ዕ→እ normalization
    preserve_punctuation: bool = True
    
    def to_amharic_config(self) -> AmharicTTSConfig:
        """Convert to full AmharicTTSConfig with G2P disabled"""
        return AmharicTTSConfig(
            g2p=G2PConfiguration(
                backend_order=[],  # No G2P backends
                enable_quality_check=False,
                enable_epenthesis=False,
                enable_gemination=False,
                enable_labiovelars=False,
                fallback_to_rules=False  # Don't even use rule-based
            ),
            tokenizer=TokenizerConfiguration(
                mode=self.tokenizer_mode,
                vocab_size=self.vocab_size,
                use_g2p_preprocessing=False,  # Explicitly disable
                add_word_boundaries=True,
                preserve_syllable_structure=True  # Ethiopic is syllabic
            ),
            enable_hybrid_tokenizer_for_training=False  # Use standard tokenizer
        )


# Preset configurations
BPE_ONLY_FAST = BPEOnlyConfiguration(
    vocab_size=512,  # Smaller vocab for faster training
    batch_size=2,
    grad_accum_steps=84
)

BPE_ONLY_QUALITY = BPEOnlyConfiguration(
    vocab_size=2048,  # Larger vocab for better representation
    batch_size=2,
    grad_accum_steps=84,
    normalize_ethiopic_variants=True
)

BPE_ONLY_MINIMAL = BPEOnlyConfiguration(
    vocab_size=256,  # Minimal vocab for testing
    batch_size=1,
    grad_accum_steps=168
)


def create_bpe_only_config(
    preset: str = "default",
    vocab_size: Optional[int] = None,
    normalize_variants: bool = True
) -> BPEOnlyConfiguration:
    """
    Create a BPE-only configuration
    
    Args:
        preset: Configuration preset ("fast", "quality", "minimal", "default")
        vocab_size: Override vocabulary size
        normalize_variants: Normalize Ethiopic character variants
        
    Returns:
        BPEOnlyConfiguration instance
    """
    presets = {
        "fast": BPE_ONLY_FAST,
        "quality": BPE_ONLY_QUALITY,
        "minimal": BPE_ONLY_MINIMAL,
        "default": BPEOnlyConfiguration()
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    config = presets[preset]
    
    # Apply overrides
    if vocab_size is not None:
        config.vocab_size = vocab_size
    if normalize_variants is not None:
        config.normalize_ethiopic_variants = normalize_variants
    
    return config


def validate_bpe_only_environment():
    """
    Validate that BPE-only training can proceed
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    checks = []
    
    # Check that we're not accidentally using G2P
    try:
        from ..tokenizer.hybrid_tokenizer import HybridAmharicTokenizer
        tokenizer = HybridAmharicTokenizer(use_phonemes=False)
        checks.append(("Tokenizer", True, "BPE mode confirmed"))
    except Exception as e:
        checks.append(("Tokenizer", False, f"Tokenizer initialization failed: {e}"))
    
    # Check XTTS TTS library
    try:
        from TTS.tts.models.xtts import Xtts
        checks.append(("TTS Library", True, "XTTS available"))
    except ImportError as e:
        checks.append(("TTS Library", False, f"TTS not found: {e}"))
    
    # Check PyTorch
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        checks.append(("PyTorch", True, f"CUDA: {cuda_available}"))
    except ImportError:
        checks.append(("PyTorch", False, "PyTorch not found"))
    
    # Print validation results
    print("\n" + "=" * 70)
    print("🔍 BPE-ONLY TRAINING ENVIRONMENT VALIDATION")
    print("=" * 70)
    
    all_passed = True
    for name, passed, message in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {name:20s}: {message}")
        if not passed:
            all_passed = False
    
    print("=" * 70 + "\n")
    
    if all_passed:
        return True, "✅ All checks passed - BPE-only training ready!"
    else:
        return False, "❌ Some checks failed - see above for details"


if __name__ == "__main__":
    # Test configuration
    print("=" * 70)
    print("BPE-ONLY CONFIGURATION TEST")
    print("=" * 70)
    
    # Test default config
    config = create_bpe_only_config("default")
    print(f"\nDefault Configuration:")
    print(f"  Use G2P: {config.use_g2p}")
    print(f"  Tokenizer Mode: {config.tokenizer_mode}")
    print(f"  Vocab Size: {config.vocab_size}")
    print(f"  Normalize Variants: {config.normalize_ethiopic_variants}")
    
    # Convert to full config
    full_config = config.to_amharic_config()
    print(f"\nFull AmharicTTSConfig:")
    print(f"  G2P Backends: {full_config.g2p.backend_order}")
    print(f"  Tokenizer Mode: {full_config.tokenizer.mode}")
    print(f"  Use G2P Preprocessing: {full_config.tokenizer.use_g2p_preprocessing}")
    
    # Validate environment
    print()
    success, message = validate_bpe_only_environment()
    print(f"\n{message}")
