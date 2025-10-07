"""
Hybrid G2P+BPE Tokenizer for Amharic TTS

This module provides a phoneme-aware tokenizer that combines the benefits of:
1. G2P (Grapheme-to-Phoneme) conversion for accurate pronunciation
2. BPE (Byte Pair Encoding) tokenization for efficient representation

Architecture:
- Converts Amharic text to IPA phonemes using enhanced G2P
- Applies BPE tokenization to phoneme sequences
- Maintains compatibility with multilingual XTTS tokenizer
- Supports both phoneme-based and raw text modes

Benefits:
- Better pronunciation modeling for rare words
- Improved phonetic consistency
- Enhanced handling of ejectives and labiovelars
- Preserved multilingual capabilities
"""

import os
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

# Import G2P converter
try:
    from ..g2p.amharic_g2p_enhanced import EnhancedAmharicG2P
    G2P_AVAILABLE = True
except ImportError:
    G2P_AVAILABLE = False
    logging.warning("Amharic G2P not available. Hybrid tokenizer will fallback to raw text.")

# Import configuration
try:
    from ..config.amharic_config import AmharicTTSConfig, DEFAULT_CONFIG
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)


class HybridAmharicTokenizer:
    """
    Hybrid tokenizer for Amharic TTS combining G2P and BPE
    
    Features:
    - Phoneme-aware tokenization using G2P conversion
    - BPE encoding on phoneme sequences
    - Configurable phoneme/text mode
    - Multilingual compatibility
    
    Usage:
        # Initialize with phoneme mode
        tokenizer = HybridAmharicTokenizer(use_phonemes=True)
        
        # Tokenize Amharic text
        tokens = tokenizer.tokenize("ሰላም ዓለም")
        
        # Encode for model input
        token_ids = tokenizer.encode("ሰላም ዓለም")
    """
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        use_phonemes: bool = True,
        config: Optional = None,
        base_tokenizer = None
    ):
        """
        Initialize hybrid tokenizer
        
        Args:
            vocab_file: Path to BPE vocabulary (optional)
            use_phonemes: Whether to use G2P phoneme conversion
            config: AmharicTTSConfig instance (optional)
            base_tokenizer: Existing XTTS tokenizer to wrap (optional)
        """
        self.use_phonemes = use_phonemes and G2P_AVAILABLE
        self.vocab_file = vocab_file
        self.base_tokenizer = base_tokenizer
        
        # Load configuration
        if CONFIG_AVAILABLE and config is None:
            self.config = DEFAULT_CONFIG
        else:
            self.config = config or self._get_fallback_config()
        
        # Initialize G2P converter if phoneme mode enabled
        self.g2p = None
        if self.use_phonemes:
            self._initialize_g2p()
        
        # Cache for tokenization
        self._phoneme_cache = {}
        
        logger.info(f"Initialized HybridAmharicTokenizer (phoneme_mode={self.use_phonemes})")
    
    def _get_fallback_config(self):
        """Create basic config if not available"""
        class FallbackConfig:
            class g2p:
                backend_order = []
                enable_quality_check = False
                enable_epenthesis = True
                enable_gemination = True
                enable_labiovelars = True
        return FallbackConfig()
    
    def _initialize_g2p(self):
        """Initialize G2P converter"""
        if not G2P_AVAILABLE:
            logger.warning("G2P not available, falling back to raw text mode")
            self.use_phonemes = False
            return
        
        try:
            self.g2p = EnhancedAmharicG2P(config=self.config)
            logger.info("✅ G2P converter initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize G2P: {e}")
            self.use_phonemes = False
            self.g2p = None
    
    def is_amharic_text(self, text: str) -> bool:
        """
        Check if text contains Amharic (Ethiopic) characters
        
        Args:
            text: Input text
            
        Returns:
            True if text contains Ethiopic characters
        """
        # Ethiopic Unicode range: U+1200–U+137F
        for char in text:
            if 0x1200 <= ord(char) <= 0x137F:
                return True
        return False
    
    def preprocess_text(self, text: str, lang: str = "am") -> str:
        """
        Preprocess text for tokenization
        
        For Amharic text:
        - Applies G2P conversion if phoneme mode enabled
        - Normalizes text otherwise
        
        For non-Amharic text:
        - Returns as-is (handled by base tokenizer)
        
        Args:
            text: Input text
            lang: Language code (default: "am" for Amharic)
            
        Returns:
            Preprocessed text (phonemes or normalized text)
        """
        if not text or not text.strip():
            return ""
        
        # Check if Amharic text
        is_amharic = (lang == "am") or self.is_amharic_text(text)
        
        if is_amharic and self.use_phonemes and self.g2p:
            # Apply G2P conversion
            try:
                # Check cache first
                if text in self._phoneme_cache:
                    return self._phoneme_cache[text]
                
                # Convert to phonemes
                phonemes = self.g2p.convert(text)
                
                # Cache result
                if len(self._phoneme_cache) < 10000:  # Prevent unbounded growth
                    self._phoneme_cache[text] = phonemes
                
                logger.debug(f"G2P: {text[:50]}... → {phonemes[:50]}...")
                return phonemes
                
            except Exception as e:
                logger.warning(f"G2P conversion failed: {e}, using raw text")
                return text
        else:
            # Return as-is for non-Amharic or raw text mode
            return text
    
    def tokenize(self, text: str, lang: str = "am") -> List[str]:
        """
        Tokenize text into subword tokens
        
        Args:
            text: Input text
            lang: Language code
            
        Returns:
            List of tokens
        """
        # Preprocess (apply G2P if needed)
        preprocessed = self.preprocess_text(text, lang=lang)
        
        # If base tokenizer available, use it
        if self.base_tokenizer is not None:
            if hasattr(self.base_tokenizer, 'tokenize'):
                return self.base_tokenizer.tokenize(preprocessed)
            elif hasattr(self.base_tokenizer, 'encode'):
                # Some tokenizers only have encode, not tokenize
                token_ids = self.base_tokenizer.encode(preprocessed)
                return [self.base_tokenizer.decode([tid]) for tid in token_ids]
        
        # Fallback: simple whitespace tokenization
        return preprocessed.split()
    
    def encode(
        self,
        text: str,
        lang: str = "am",
        return_tensors: Optional[str] = None
    ) -> Union[List[int], 'torch.Tensor']:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            lang: Language code
            return_tensors: Return format ("pt" for PyTorch, None for list)
            
        Returns:
            Token IDs as list or tensor
        """
        # Preprocess
        preprocessed = self.preprocess_text(text, lang=lang)
        
        # If base tokenizer available, use it
        if self.base_tokenizer is not None:
            if hasattr(self.base_tokenizer, 'encode'):
                result = self.base_tokenizer.encode(preprocessed)
                
                # Convert to tensor if requested
                if return_tensors == "pt":
                    import torch
                    result = torch.tensor(result)
                
                return result
        
        # Fallback: return character codes
        token_ids = [ord(c) for c in preprocessed]
        
        if return_tensors == "pt":
            import torch
            return torch.tensor(token_ids)
        
        return token_ids
    
    def decode(
        self,
        token_ids: Union[List[int], 'torch.Tensor'],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: List of token IDs or tensor
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        # Convert tensor to list if needed
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        # If base tokenizer available, use it
        if self.base_tokenizer is not None:
            if hasattr(self.base_tokenizer, 'decode'):
                return self.base_tokenizer.decode(
                    token_ids,
                    skip_special_tokens=skip_special_tokens
                )
        
        # Fallback: convert from character codes
        try:
            return ''.join([chr(tid) for tid in token_ids])
        except (ValueError, OverflowError):
            return ""
    
    def batch_encode(
        self,
        texts: List[str],
        lang: str = "am",
        return_tensors: Optional[str] = None,
        padding: bool = True,
        max_length: Optional[int] = None
    ) -> Dict:
        """
        Batch encode multiple texts
        
        Args:
            texts: List of input texts
            lang: Language code
            return_tensors: Return format ("pt" for PyTorch)
            padding: Whether to pad sequences
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Encode each text
        all_ids = [self.encode(text, lang=lang) for text in texts]
        
        # Determine max length
        if max_length is None:
            max_length = max(len(ids) for ids in all_ids) if all_ids else 0
        
        # Pad sequences if needed
        if padding:
            padded_ids = []
            attention_masks = []
            
            for ids in all_ids:
                # Truncate if too long
                if len(ids) > max_length:
                    ids = ids[:max_length]
                
                # Pad if too short
                padding_length = max_length - len(ids)
                padded = ids + [0] * padding_length
                mask = [1] * len(ids) + [0] * padding_length
                
                padded_ids.append(padded)
                attention_masks.append(mask)
            
            result = {
                'input_ids': padded_ids,
                'attention_mask': attention_masks
            }
        else:
            result = {'input_ids': all_ids}
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            result = {k: torch.tensor(v) for k, v in result.items()}
        
        return result
    
    def get_vocab_size(self) -> int:
        """
        Get vocabulary size
        
        Returns:
            Size of vocabulary
        """
        if self.base_tokenizer is not None:
            if hasattr(self.base_tokenizer, 'vocab_size'):
                return self.base_tokenizer.vocab_size
            elif hasattr(self.base_tokenizer, 'get_vocab'):
                return len(self.base_tokenizer.get_vocab())
        
        # Fallback: Unicode BMP size
        return 65536
    
    def clear_cache(self):
        """Clear phoneme conversion cache"""
        self._phoneme_cache.clear()
        logger.info("Phoneme cache cleared")
    
    def get_config(self) -> Dict:
        """
        Get tokenizer configuration
        
        Returns:
            Configuration dictionary
        """
        return {
            'use_phonemes': self.use_phonemes,
            'vocab_file': self.vocab_file,
            'g2p_available': G2P_AVAILABLE,
            'cache_size': len(self._phoneme_cache)
        }
    
    def __repr__(self) -> str:
        return (
            f"HybridAmharicTokenizer("
            f"phoneme_mode={self.use_phonemes}, "
            f"g2p_available={self.g2p is not None}, "
            f"base_tokenizer={type(self.base_tokenizer).__name__ if self.base_tokenizer else None}"
            f")"
        )


def create_hybrid_tokenizer(
    base_tokenizer=None,
    use_phonemes: bool = True,
    config=None
) -> HybridAmharicTokenizer:
    """
    Factory function to create hybrid tokenizer
    
    Args:
        base_tokenizer: Existing XTTS tokenizer to wrap
        use_phonemes: Whether to enable phoneme mode
        config: Configuration object
        
    Returns:
        HybridAmharicTokenizer instance
    """
    return HybridAmharicTokenizer(
        use_phonemes=use_phonemes,
        config=config,
        base_tokenizer=base_tokenizer
    )


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    print("=" * 80)
    print("HYBRID AMHARIC TOKENIZER - DEMO")
    print("=" * 80)
    print()
    
    # Test with phoneme mode
    print("🔧 Initializing tokenizer with phoneme mode...")
    tokenizer = HybridAmharicTokenizer(use_phonemes=True)
    print(f"✅ {tokenizer}")
    print()
    
    # Test examples
    examples = [
        "ሰላም ዓለም",
        "ኢትዮጵያ",
        "አማርኛ",
        "ቋንቋ",
    ]
    
    print("📝 Tokenization Examples:")
    print("-" * 80)
    for text in examples:
        preprocessed = tokenizer.preprocess_text(text)
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        
        print(f"\nInput:         {text}")
        print(f"Preprocessed:  {preprocessed}")
        print(f"Tokens:        {tokens[:10]}..." if len(tokens) > 10 else f"Tokens:        {tokens}")
        print(f"Token IDs:     {token_ids[:20]}..." if len(token_ids) > 20 else f"Token IDs:     {token_ids}")
    
    print("\n" + "=" * 80)
    print("✅ Demo complete!")
