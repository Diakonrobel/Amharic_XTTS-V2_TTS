"""
Symbol Expander for Amharic TTS

Expands mathematical operators, currency symbols, and other special characters
to Amharic words for natural speech synthesis.

Features:
- Mathematical operators: +, -, ×, ÷, =, %, etc.
- Currency symbols: $, €, £, ¥, ብር
- Common symbols: &, @, #, °C, °F
- Context-aware expansion (prefix/suffix/infix)
"""

import re
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SymbolExpander:
    """
    Expand symbols and mathematical operators to Amharic words
    
    Usage:
        expander = SymbolExpander()
        text = expander.expand("5 + 3 = 8")
        # → "5 መደመር 3 እኩል 8"
        
        text = expander.expand("$50")
        # → "50 ዶላር"
    """
    
    def __init__(self, target_language: str = 'am'):
        """
        Initialize symbol expander
        
        Args:
            target_language: Target language code ('am' for Amharic, 'en' for English)
        """
        self.target_language = target_language
        self._load_symbol_mappings()
    
    def _load_symbol_mappings(self):
        """Load symbol to word mappings"""
        
        # Mathematical operators (infix - between numbers)
        self.math_operators = {
            '+': 'መደመር',      # plus/add
            '−': 'መቀነስ',       # minus/subtract (Unicode minus)
            '-': 'መቀነስ',       # minus/subtract (hyphen-minus)
            '×': 'ማባዛት',      # times/multiply
            'x': 'ማባዛት',      # times (letter x)
            'X': 'ማባዛത',      # times (capital X)
            '÷': 'መከፋፈል',     # divided by
            '/': 'በ',          # divided by / per
            '=': 'እኩል',        # equals
            '≈': 'ይህ ያህል',    # approximately
            '≠': 'አይመስልም',   # not equal
            '<': 'ያነሰ',        # less than
            '>': 'ይበልጣል',     # greater than
            '≤': 'ያነሰ ወይም እኩል',  # less than or equal
            '≥': 'ይበልጣል ወይም እኩል',  # greater than or equal
        }
        
        # Percent and per symbols (suffix)
        self.suffix_symbols = {
            '%': 'ፐርሰንት',     # percent
            '‰': 'በሺ',         # per mille
            '°': 'ዲግሪ',        # degree
        }
        
        # Temperature units
        self.temperature_units = {
            '°C': 'ዲግሪ ሴልሲየስ',
            '°F': 'ዲግሪ ፋራናይት',
            '°K': 'ኬልቪን',
        }
        
        # Currency symbols (prefix or suffix)
        self.currency_symbols = {
            '$': 'ዶላር',        # dollar
            '€': 'ዩሮ',         # euro
            '£': 'ፓውንድ',       # pound
            '¥': 'የን',         # yen
            '₹': 'ሩፒ',         # rupee
            'USD': 'ዶላር',
            'EUR': 'ዩሮ',
            'GBP': 'ፓውንድ',
            'ETB': 'ብር',       # Ethiopian Birr
            'birr': 'ብር',
            'Birr': 'ብር',
        }
        
        # Common symbols
        self.common_symbols = {
            '&': 'እና',         # and
            '@': 'በ',          # at
            '#': 'ቁጥር',        # number/hash
            '©': 'የቅጂ መብት',   # copyright
            '®': 'የተመዘገበ',    # registered
            '™': 'የንግድ ምልክት', # trademark
            '§': 'ክፍል',        # section
            '†': 'መስቀል',       # dagger
            '‡': 'ድርብ መስቀል',  # double dagger
        }
        
        # Fallback symbol names (Amharic pronunciation for any unrecognized symbol)
        # These are spoken when a symbol is not in the primary mappings above
        self.fallback_symbol_names = {
            # Slash and related
            '/': 'ሀዝባር',           # forward slash (hazbar)
            '\\': 'ተቃራኒ ሀዝባር',   # backslash (reverse hazbar)
            '|': 'ቀጥተኛ መስመር',     # vertical bar/pipe
            
            # Brackets and parentheses
            '(': 'ክፍት ቅንፍ',       # open parenthesis
            ')': 'ዝግ ቅንፍ',         # close parenthesis
            '[': 'ክፍት አራት ማዕዘን',  # open square bracket
            ']': 'ዝግ አራት ማዕዘን',    # close square bracket
            '{': 'ክፍት ሰንደቅ',      # open curly brace
            '}': 'ዝግ ሰንደቅ',        # close curly brace
            
            # Quotes
            '"': 'ድርብ ጥቅስ',       # double quote
            "'": 'ነጠላ ጥቅስ',       # single quote/apostrophe
            '`': 'የኋላ ጥቅስ',        # backtick
            
            # Punctuation (fallback names)
            '!': 'መለያ ምልክት',      # exclamation mark
            '?': 'ጥያቄ ምልክት',      # question mark
            '.': 'ነጥብ',            # period/dot
            ',': 'ኮማ',             # comma
            ';': 'ሴሚኮሎን',         # semicolon
            ':': 'ኮሎን',            # colon
            '…': 'ሶስት ነጥብ',        # ellipsis
            
            # Math symbols (fallback names)
            '*': 'ኮከብ',            # asterisk/star
            '^': 'ሃይል',            # caret/power
            '~': 'ሞገድ',            # tilde
            '_': 'ስር መስመር',        # underscore
        }
    
    def expand(self, text: str) -> str:
        """
        Expand all symbols in text to Amharic words
        
        Args:
            text: Input text with symbols
            
        Returns:
            Text with symbols expanded to words
            
        Examples:
            "5 + 3" → "5 መደመር 3"
            "$50" → "50 ዶላር"
            "25%" → "25 ፐርሰንት"
            "5 x 3 = 15" → "5 ማባዛት 3 እኩል 15"
            "ዓ/ም" → "ዓ ሀዝባር ም" (speaks slash as 'hazbar')
        """
        if not text:
            return ""
        
        result = text
        
        # Step 1: Temperature units (must come before degree symbol)
        result = self._expand_temperature_units(result)
        
        # Step 2: Currency symbols
        result = self._expand_currency(result)
        
        # Step 3: Percent and suffix symbols
        result = self._expand_suffix_symbols(result)
        
        # Step 4: Mathematical operators (between numbers/words)
        result = self._expand_math_operators(result)
        
        # Step 5: Common symbols
        result = self._expand_common_symbols(result)
        
        # Step 6: Fallback for any remaining unrecognized symbols
        result = self._expand_fallback_symbols(result)
        
        return result
    
    def _expand_temperature_units(self, text: str) -> str:
        """Expand temperature units like °C, °F"""
        for symbol, word in self.temperature_units.items():
            # Match temperature like "25°C" or "98.6°F"
            pattern = r'(\d+(?:\.\d+)?)\s*' + re.escape(symbol)
            replacement = r'\1 ' + word
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _expand_currency(self, text: str) -> str:
        """
        Expand currency symbols
        
        Handles:
        - Prefix: $50 → 50 ዶላር
        - Suffix: 50€ → 50 ዩሮ
        - ISO codes: 50 USD → 50 ዶላር
        """
        for symbol, word in self.currency_symbols.items():
            # Prefix currency: $50, $1,000
            if len(symbol) == 1:  # Single character currencies
                # Match currency symbol before number
                pattern = re.escape(symbol) + r'\s*(\d[\d,]*(?:\.\d+)?)'
                replacement = r'\1 ' + word
                text = re.sub(pattern, replacement, text)
                
                # Match currency symbol after number
                pattern = r'(\d[\d,]*(?:\.\d+)?)\s*' + re.escape(symbol)
                replacement = r'\1 ' + word
                text = re.sub(pattern, replacement, text)
            else:  # Multi-character ISO codes
                # Match ISO code after number: 50 USD
                pattern = r'(\d[\d,]*(?:\.\d+)?)\s+' + re.escape(symbol) + r'\b'
                replacement = r'\1 ' + word
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _expand_suffix_symbols(self, text: str) -> str:
        """Expand suffix symbols like % (percent)"""
        for symbol, word in self.suffix_symbols.items():
            # Match number + symbol: 25%, 3.14%, 1,000%
            pattern = r'(\d[\d,]*(?:\.\d+)?)\s*' + re.escape(symbol)
            replacement = r'\1 ' + word
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _expand_math_operators(self, text: str) -> str:
        """
        Expand mathematical operators
        
        Handles:
        - Infix: 5 + 3 → 5 መደመር 3
        - With spacing: 5+3 or 5 + 3
        
        Note: Special handling for / (slash) - only expand in clear math context
        Other contexts (like / in ዓ/ም) are handled by fallback expansion
        """
        for symbol, word in self.math_operators.items():
            # Special handling for /, x, X - only expand in clear math context
            if symbol in ['/', 'x', 'X']:
                # Only match when there are numbers/spaces around: 10 / 2, 5 x 3
                # This prevents matching 'x' in 'text', 'example', etc.
                pattern = r'(\d[\d,]*(?:\.\d+)?)\s*' + re.escape(symbol) + r'\s*(\d[\d,]*(?:\.\d+)?)'
                replacement = r'\1 ' + word + r' \2'
                text = re.sub(pattern, replacement, text)
            else:
                # For other operators, match between any non-space characters
                # Pattern: (number/word) [spaces] operator [spaces] (number/word)
                pattern = r'(\S+)\s*' + re.escape(symbol) + r'\s*(\S+)'
                replacement = r'\1 ' + word + r' \2'
                text = re.sub(pattern, replacement, text)
        
        return text
    
    def _expand_common_symbols(self, text: str) -> str:
        """Expand common symbols like &, @, #"""
        for symbol, word in self.common_symbols.items():
            # Simple replacement with word boundaries
            pattern = r'\s*' + re.escape(symbol) + r'\s*'
            replacement = ' ' + word + ' '
            text = re.sub(pattern, replacement, text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _expand_fallback_symbols(self, text: str) -> str:
        """
        Expand any remaining unrecognized symbols using fallback names
        
        This catches symbols that weren't handled by specific expansion methods
        and speaks their Amharic name instead of leaving them as-is.
        
        Example:
            "ዓ/ም" → "ዓ ሀዝባር ም" (speaks '/' as 'hazbar')
        """
        for symbol, word in self.fallback_symbol_names.items():
            if symbol in text:
                # Surround symbol with spaces for proper word separation
                pattern = re.escape(symbol)
                replacement = ' ' + word + ' '
                text = re.sub(pattern, replacement, text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def get_supported_symbols(self) -> Dict[str, str]:
        """
        Get all supported symbols and their expansions
        
        Returns:
            Dictionary of symbol -> word mappings (includes fallback names)
        """
        all_symbols = {}
        all_symbols.update(self.math_operators)
        all_symbols.update(self.suffix_symbols)
        all_symbols.update(self.currency_symbols)
        all_symbols.update(self.common_symbols)
        all_symbols.update(self.temperature_units)
        all_symbols.update(self.fallback_symbol_names)  # Include fallback names
        
        return all_symbols


# Convenience function
def expand_symbols(text: str) -> str:
    """
    Quick function to expand symbols in text
    
    Args:
        text: Input text with symbols
        
    Returns:
        Text with symbols expanded
        
    Example:
        >>> expand_symbols("5 + 3 = 8, $50, 25%")
        "5 መደመር 3 እኩል 8, 50 ዶላር, 25 ፐርሰንት"
    """
    expander = SymbolExpander()
    return expander.expand(text)


# Example usage and tests
if __name__ == "__main__":
    print("=" * 80)
    print("SYMBOL EXPANDER - DEMONSTRATION")
    print("=" * 80)
    print()
    
    expander = SymbolExpander()
    
    # Test cases
    test_cases = [
        # Mathematics
        "5 + 3",
        "10 - 2",
        "4 x 5",
        "20 ÷ 4",
        "5 + 3 = 8",
        "2 x 3 = 6",
        
        # Currency
        "$50",
        "€100",
        "£25.50",
        "50 USD",
        "1,000 birr",
        
        # Percent
        "25%",
        "3.14%",
        "100%",
        
        # Temperature
        "25°C",
        "98.6°F",
        
        # Common symbols
        "Ahmed & Sara",
        "contact@example.com",
        "#1 winner",
        
        # Complex combinations
        "$50 + $25 = $75",
        "Price: $1,500 (15% off)",
        "5 x 3 + 2 = 17",
        "Temperature: 25°C to 30°C",
    ]
    
    print("📝 Symbol Expansion Examples:")
    print("-" * 80)
    
    for text in test_cases:
        expanded = expander.expand(text)
        print(f"In:  {text:40} → {expanded}")
    
    print()
    print("=" * 80)
    print("📋 Supported Symbols:")
    print("-" * 80)
    
    symbols = expander.get_supported_symbols()
    for symbol, word in sorted(symbols.items(), key=lambda x: x[0]):
        print(f"{symbol:10} → {word}")
    
    print()
    print("=" * 80)
    print("✅ Demonstration complete!")
    print("=" * 80)
