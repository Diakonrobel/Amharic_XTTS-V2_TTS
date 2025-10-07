"""
Amharic Text Normalizer

Handles:
- Character variant normalization
- Punctuation conversion
- Whitespace normalization
- Abbreviation expansion
"""

import re
from typing import Dict


class AmharicTextNormalizer:
    """Normalize Amharic text for TTS processing"""
    
    def __init__(self):
        self._load_rules()
        
    def _load_rules(self):
        """Load normalization rules"""
        
        # Character variant normalization
        self.char_variants = {
            'ሥ': 'ስ',  # ሥ → ስ
            'ዕ': 'እ',  # ዕ → እ
            'ፕ': '፝',  # ፕ → ፝
            'ኅ': 'ሕ',  # ኅ → ሕ
            'ኽ': 'ክ',  # ኽ → ክ
            'ሕ': 'ሕ',  # ሕ → ሕ
            'ዐ': 'አ',  # ዐ → አ
            'ኣ': 'አ',  # ኣ → አ
            'ሀ': 'ሃ',  # ሀ → ሃ (test requirement)
            'ዓ': 'አ',  # ዓ → አ (test requirement)
        }
        
        # Punctuation conversion (Ethiopic → Standard)
        # Note: Set convert_punctuation to False to preserve Ethiopic punctuation
        self.convert_punctuation_enabled = False  # Preserve Ethiopic by default
        self.punctuation_map = {
            # Disabled by default - uncomment to convert to standard punctuation
            # '።': '.',  # Amharic full stop
            # '፣': ',',  # Amharic comma
            # '፤': ';',  # Amharic semicolon
            # '፥': ':',  # Amharic colon
            # '፦': ':',  # Amharic preface colon
            # '፧': '?',  # Amharic question mark
            # '፨': '¶',  # Amharic paragraph separator
            '፡': ' ',  # Amharic word space (always convert to space)
        }
        
        # Common Amharic abbreviations
        self.abbreviations = {
            'ዓ.ም': 'ዓመተ ምህረት',  # Year of Grace (Ethiopian calendar)
            'ዓ.ዓ': 'ዓመተ ዓለም',     # Year of the World
            'ክ.ክ': 'ክፍለ ከተማ',     # Sub-city
            'ት.ቤት': 'ትምህርት ቤት',  # School
            'ት.ት': 'ትምህርት ተቋም',   # Educational institution
            'ኢ.ፌ.ዲ.ሪ': 'የኢትዮጵያ ፌደራላዊ ዲሞክራሲያዊ ሪፐብሊክ',
            'ም.ም': 'መምህር',         # Teacher (title)
            'ዶ.ር': 'ዶክተር',         # Doctor
            'ፕ.ር': 'ፕሮፌሰር',       # Professor
        }
        
    def normalize(self, text: str) -> str:
        """
        Normalize Amharic text
        
        Args:
            text: Input Amharic text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
            
        # Step 1: Normalize character variants
        text = self._normalize_characters(text)
        
        # Step 2: Expand abbreviations
        text = self._expand_abbreviations(text)
        
        # Step 3: Convert punctuation
        text = self._convert_punctuation(text)
        
        # Step 4: Normalize whitespace
        text = self._normalize_whitespace(text)
        
        return text
        
    def _normalize_characters(self, text: str) -> str:
        """Normalize variant character forms"""
        for old, new in self.char_variants.items():
            text = text.replace(old, new)
        return text
        
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common Amharic abbreviations"""
        for abbr, expansion in self.abbreviations.items():
            text = text.replace(abbr, expansion)
        return text
        
    def _convert_punctuation(self, text: str) -> str:
        """Convert Ethiopic punctuation to standard"""
        for ethiopic, standard in self.punctuation_map.items():
            text = text.replace(ethiopic, standard)
        return text
        
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text


# Convenience function
def normalize_amharic_text(text: str) -> str:
    """
    Normalize Amharic text
    
    Args:
        text: Input Amharic text
        
    Returns:
        Normalized text
    """
    normalizer = AmharicTextNormalizer()
    return normalizer.normalize(text)
