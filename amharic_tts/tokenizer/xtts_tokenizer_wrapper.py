"""
XTTS-Compatible Tokenizer Wrapper for Amharic

This module wraps the hybrid Amharic tokenizer to be compatible with
the XTTS training pipeline, maintaining the same API as the standard
XTTS tokenizer while adding G2P preprocessing support.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Union
from pathlib import Path

try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    logging.warning("tokenizers library not available")

try:
    from .hybrid_tokenizer import HybridAmharicTokenizer
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from amharic_tts.tokenizer.hybrid_tokenizer import HybridAmharicTokenizer

logger = logging.getLogger(__name__)


class XTTSAmharicTokenizer:
    """
    XTTS-compatible wrapper for Amharic hybrid tokenizer
    
    This wrapper:
    - Maintains XTTS tokenizer API compatibility
    - Adds optional G2P preprocessing for Amharic
    - Falls back gracefully to standard tokenization
    - Supports both raw text and phoneme modes
    
    Usage:
        # Create wrapper with phoneme mode
        tokenizer = XTTSAmharicTokenizer(
            vocab_file="vocab.json",
            use_phonemes=True
        )
        
        # Use like standard XTTS tokenizer
        ids = tokenizer.encode("ሰላም ዓለም", lang="am")
        text = tokenizer.decode(ids)
    """
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        use_phonemes: bool = False,
        config: Optional = None,
        **kwargs
    ):
        """
        Initialize XTTS-compatible Amharic tokenizer
        
        Args:
            vocab_file: Path to vocab.json file
            use_phonemes: Whether to use G2P phoneme preprocessing
            config: Amharic configuration object
            **kwargs: Additional arguments for base tokenizer
        """
        self.vocab_file = vocab_file
        self.use_phonemes = use_phonemes
        self.config = config
        
        # Load base tokenizer if vocab file provided
        self.base_tokenizer = None
        if vocab_file and os.path.exists(vocab_file):
            self._load_base_tokenizer(vocab_file)
        
        # Initialize hybrid tokenizer for Amharic
        self.hybrid_tokenizer = HybridAmharicTokenizer(
            vocab_file=vocab_file,
            use_phonemes=use_phonemes,
            config=config,
            base_tokenizer=self.base_tokenizer
        )
        
        logger.info(f"Initialized XTTSAmharicTokenizer (phoneme_mode={use_phonemes})")
    
    def _load_base_tokenizer(self, vocab_file: str):
        """Load the base XTTS tokenizer"""
        if not TOKENIZERS_AVAILABLE:
            logger.warning("tokenizers library not available, using fallback")
            return
        
        try:
            self.base_tokenizer = Tokenizer.from_file(vocab_file)
            logger.info(f"Loaded base tokenizer from {vocab_file}")
        except Exception as e:
            logger.warning(f"Could not load base tokenizer: {e}")
            self.base_tokenizer = None
    
    def is_amharic(self, text: str) -> bool:
        """Check if text contains Amharic characters"""
        return self.hybrid_tokenizer.is_amharic_text(text)
    
    def preprocess_text(self, text: str, lang: str = None) -> str:
        """
        Preprocess text for tokenization
        
        For Amharic: applies G2P if phoneme mode enabled
        For other languages: returns as-is
        
        Args:
            text: Input text
            lang: Language code (optional)
            
        Returns:
            Preprocessed text
        """
        # Detect language if not specified
        if lang is None:
            lang = "am" if self.is_amharic(text) else "en"
        
        # Use hybrid tokenizer preprocessing
        return self.hybrid_tokenizer.preprocess_text(text, lang=lang)
    
    def encode(
        self,
        text: str,
        lang: str = None,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], 'torch.Tensor']:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            lang: Language code (am for Amharic)
            return_tensors: Return format ("pt" for PyTorch)
            
        Returns:
            Token IDs as list or tensor
        """
        # Detect language if not specified
        if lang is None:
            lang = "am" if self.is_amharic(text) else "en"
        
        # Use hybrid tokenizer
        return self.hybrid_tokenizer.encode(
            text,
            lang=lang,
            return_tensors=return_tensors
        )
    
    def decode(
        self,
        token_ids: Union[List[int], 'torch.Tensor'],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.hybrid_tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
    
    def tokenize(self, text: str, lang: str = None) -> List[str]:
        """
        Tokenize text into tokens
        
        Args:
            text: Input text
            lang: Language code
            
        Returns:
            List of tokens
        """
        if lang is None:
            lang = "am" if self.is_amharic(text) else "en"
        
        return self.hybrid_tokenizer.tokenize(text, lang=lang)
    
    def batch_encode(
        self,
        texts: List[str],
        lang: str = None,
        return_tensors: Optional[str] = None,
        padding: bool = True,
        max_length: Optional[int] = None
    ) -> Dict:
        """
        Batch encode multiple texts
        
        Args:
            texts: List of input texts
            lang: Language code
            return_tensors: Return format
            padding: Whether to pad sequences
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if lang is None:
            # Check first text for language
            lang = "am" if texts and self.is_amharic(texts[0]) else "en"
        
        return self.hybrid_tokenizer.batch_encode(
            texts,
            lang=lang,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_length
        )
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.hybrid_tokenizer.get_vocab_size()
    
    def get_vocab(self) -> Dict:
        """Get vocabulary dictionary"""
        if self.base_tokenizer and hasattr(self.base_tokenizer, 'get_vocab'):
            return self.base_tokenizer.get_vocab()
        return {}
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary size property"""
        return self.get_vocab_size()
    
    def clear_cache(self):
        """Clear preprocessing cache"""
        self.hybrid_tokenizer.clear_cache()
    
    def get_config(self) -> Dict:
        """Get tokenizer configuration"""
        config = self.hybrid_tokenizer.get_config()
        config['vocab_file'] = self.vocab_file
        config['wrapper'] = 'XTTSAmharicTokenizer'
        return config
    
    def save(self, output_path: str):
        """
        Save tokenizer configuration
        
        Args:
            output_path: Directory to save configuration
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Save config
        config_path = os.path.join(output_path, "amharic_tokenizer_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_config(), f, indent=2)
        
        logger.info(f"Saved tokenizer config to {config_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        use_phonemes: bool = None,
        **kwargs
    ):
        """
        Load tokenizer from pretrained model directory
        
        Args:
            model_path: Path to model directory
            use_phonemes: Override phoneme mode setting
            **kwargs: Additional arguments
            
        Returns:
            XTTSAmharicTokenizer instance
        """
        # Look for vocab file
        vocab_file = os.path.join(model_path, "vocab.json")
        if not os.path.exists(vocab_file):
            logger.warning(f"vocab.json not found in {model_path}")
            vocab_file = None
        
        # Look for config file
        config_file = os.path.join(model_path, "amharic_tokenizer_config.json")
        config_use_phonemes = False
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    config_use_phonemes = config_data.get('use_phonemes', False)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")
        
        # Override with provided value
        if use_phonemes is None:
            use_phonemes = config_use_phonemes
        
        return cls(
            vocab_file=vocab_file,
            use_phonemes=use_phonemes,
            **kwargs
        )
    
    def __call__(self, text: str, lang: str = None, **kwargs):
        """Make tokenizer callable like standard tokenizers"""
        return self.encode(text, lang=lang, **kwargs)
    
    def __repr__(self) -> str:
        return (
            f"XTTSAmharicTokenizer("
            f"phoneme_mode={self.use_phonemes}, "
            f"vocab_file={self.vocab_file is not None}, "
            f"base_tokenizer={self.base_tokenizer is not None}"
            f")"
        )


def create_xtts_tokenizer(
    vocab_path: Optional[str] = None,
    vocab_file: Optional[str] = None,  # Alias for compatibility
    use_phonemes: bool = False,
    use_g2p: bool = False,  # Alias for use_phonemes
    g2p_backend: str = 'auto',  # Backend selection (ignored for now, uses config)
    config = None
) -> XTTSAmharicTokenizer:
    """
    Factory function to create XTTS-compatible Amharic tokenizer
    
    Args:
        vocab_path: Path to vocab.json (preferred name)
        vocab_file: Path to vocab.json (alias for compatibility)
        use_phonemes: Enable G2P phoneme mode
        use_g2p: Enable G2P phoneme mode (alias)
        g2p_backend: G2P backend selection (auto/transphone/epitran/rule-based)
        config: Configuration object
        
    Returns:
        XTTSAmharicTokenizer instance
    """
    # Handle parameter aliases
    vocab = vocab_path or vocab_file
    enable_phonemes = use_phonemes or use_g2p
    
    # Create config with backend if specified
    if g2p_backend and g2p_backend != 'auto' and config is None:
        try:
            from ..config.amharic_config import G2PConfiguration, G2PBackend
            from ..config import AmharicTTSConfig
            
            # Create config with specific backend
            backend_map = {
                'transphone': G2PBackend.TRANSPHONE,
                'epitran': G2PBackend.EPITRAN,
                'rule-based': G2PBackend.RULE_BASED,
                'rule_based': G2PBackend.RULE_BASED,
            }
            
            if g2p_backend in backend_map:
                g2p_config = G2PConfiguration(
                    backend_order=[backend_map[g2p_backend], G2PBackend.RULE_BASED]
                )
                config = AmharicTTSConfig(g2p=g2p_config)
        except ImportError:
            logger.warning(f"Could not load config for backend {g2p_backend}")
    
    return XTTSAmharicTokenizer(
        vocab_file=vocab,
        use_phonemes=enable_phonemes,
        config=config
    )


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    print("=" * 80)
    print("XTTS Amharic Tokenizer Wrapper - Demo")
    print("=" * 80)
    print()
    
    # Test with phoneme mode
    print("🔧 Creating tokenizer with phoneme mode...")
    tokenizer = XTTSAmharicTokenizer(use_phonemes=True)
    print(f"✅ {tokenizer}")
    print()
    
    # Test encoding
    test_texts = [
        "ሰላም ዓለም",
        "Hello world",
        "ኢትዮጵያ አማርኛ",
    ]
    
    print("📝 Encoding Tests:")
    print("-" * 80)
    for text in test_texts:
        is_amharic = tokenizer.is_amharic(text)
        preprocessed = tokenizer.preprocess_text(text)
        token_ids = tokenizer.encode(text)
        
        print(f"\nInput:        {text}")
        print(f"Language:     {'Amharic' if is_amharic else 'Other'}")
        print(f"Preprocessed: {preprocessed}")
        print(f"Token IDs:    {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}")
    
    # Test configuration
    print("\n" + "=" * 80)
    print("📊 Tokenizer Configuration:")
    print("-" * 80)
    config = tokenizer.get_config()
    for key, value in config.items():
        print(f"{key:20}: {value}")
    
    print("\n" + "=" * 80)
    print("✅ Demo complete!")
