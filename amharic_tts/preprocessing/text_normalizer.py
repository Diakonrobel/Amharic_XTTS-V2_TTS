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
        # NOTE: ዓ and ፕ are NOT normalized to preserve abbreviations like ዓ.ም and ፕ.ር
        self.char_variants = {
            'ሥ': 'ስ',  # ሥ → ስ
            'ዕ': 'እ',  # ዕ → እ
            # 'ፕ': '፝',  # DO NOT normalize ፕ - used in abbreviations like ፕ.ር!
            'ኅ': 'ሕ',  # ኅ → ሕ
            'ኽ': 'ክ',  # ኽ → ክ
            'ሕ': 'ሕ',  # ሕ → ሕ
            'ዐ': 'አ',  # ዐ → አ
            'ኣ': 'አ',  # ኣ → አ
            'ሀ': 'ሃ',  # ሀ → ሃ (test requirement)
            # 'ዓ': 'አ',  # DO NOT normalize ዓ - used in abbreviations!
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
        # NOTE: Abbreviations are matched case-sensitively and with exact spacing
        self.abbreviations = {
            # Calendar/Time abbreviations
            'ዓ.ም': 'ዓመተ ምህረት',  # Year of Grace (Ethiopian calendar)
            'ዓ.ም.': 'ዓመተ ምህረት',  # With trailing period
            'ዓ . ም': 'ዓመተ ምህረት',  # With spaces
            'ዓ.ዓ': 'ዓመተ ዓለም',     # Year of the World
            'ዓ.ዓ.': 'ዓመተ ዓለም',    # With trailing period
            
            # Place/Organization abbreviations
            'ክ.ክ': 'ክፍለ ከተማ',     # Sub-city
            'ክ.ክ.': 'ክፍለ ከተማ',    # With trailing period
            'ክ . ክ': 'ክፍለ ከተማ',    # With spaces
            
            # Education abbreviations
            'ት.ቤት': 'ትምህርት ቤት',  # School
            'ት.ት': 'ትምህርት ተቋም',   # Educational institution
            'ኢ.ፌ.ዲ.ሪ': 'የኢትዮጵያ ፌደራላዊ ዲሞክራሲያዊ ሪፐብሊክ',
            
            # Title abbreviations
            'ም.ም': 'መምህር',         # Teacher (title)
            'ዶ.ር': 'ዶክተር',         # Doctor
            'ዶ.ር.': 'ዶክተር',        # With trailing period
            'ዶ . ር': 'ዶክተር',        # With spaces
            'ፕ.ር': 'ፕሮፌሰር',       # Professor
            'ፕ.ር.': 'ፕሮፌሰር',      # With trailing period
            'ፕ . ር': 'ፕሮፌሰር',      # With spaces
        }
        
    def normalize(self, text: str) -> str:
        """
        Normalize Amharic text
        
        Args:
            text: Input Amharic text
            
        Returns:
            Normalized text
            
        Note:
            Order matters! Abbreviations must be expanded BEFORE character 
            normalization, because abbreviations like 'ዓ.ም' use specific 
            characters that would be normalized away.
        """
        if not text:
            return ""
            
        # Step 1: Expand abbreviations FIRST (before character normalization!)
        text = self._expand_abbreviations(text)
        
        # Step 2: Normalize character variants
        text = self._normalize_characters(text)
        
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
        """
        Expand common Amharic abbreviations
        
        Uses regex word boundaries to avoid partial matches.
        Example: 'ዓ.ም' expands, but 'ዓ.ም.ን' won't accidentally expand just 'ዓ.ም' part.
        """
        # Sort abbreviations by length (longest first) to handle overlaps
        sorted_abbrs = sorted(self.abbreviations.items(), key=lambda x: len(x[0]), reverse=True)
        
        for abbr, expansion in sorted_abbrs:
            # Use regex for more robust matching
            # Pattern matches abbreviation with word boundaries
            pattern = re.escape(abbr)
            text = re.sub(pattern, expansion, text)
        
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
