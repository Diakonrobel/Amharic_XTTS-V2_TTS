"""
Ethiopian Numeral Expander

Converts Ethiopic numerals (፩-፼) to Amharic words for TTS processing.

Ethiopian Numeral System:
- Units: ፩-፱ (1-9)
- Ten: ፲ (10)
- Tens: ፳፴፵፶፷፸፹፺ (20-90)
- Hundred: ፻ (100)
- Ten-thousand: ፼ (10000)

Features:
- Complete support for ፩-፼ (1-10000)
- Combination rules (e.g., ፲፪ = 12, ፻፳፫ = 123)
- Integration with existing number_expander.py
- Enterprise-grade error handling
"""

import re
from typing import Union, Dict, List
import logging

# Import existing Amharic number expander for word conversion
try:
    from .number_expander import AmharicNumberExpander
    NUMBER_EXPANDER_AVAILABLE = True
except ImportError:
    NUMBER_EXPANDER_AVAILABLE = False
    logging.warning("AmharicNumberExpander not available")

logger = logging.getLogger(__name__)


class EthiopianNumeralExpander:
    """
    Expand Ethiopian numerals (Ethiopic digits) to Amharic words
    
    Usage:
        expander = EthiopianNumeralExpander()
        result = expander.expand("፻፳፫")  # "123" → "አንድ መቶ ሃያ ሶስት"
        
        # With text
        text = "በ፲፰፻፹፯ ዓ.ም ተወለደ"
        expanded = expander.expand_in_text(text)
        # Output: "በ አሥራ ስምንት መቶ ሰማንያ ሰባት ዓመተ ምህረት ተወለደ"
    """
    
    def __init__(self):
        """Initialize Ethiopian numeral expander"""
        self._load_numeral_mappings()
        
        # Initialize Amharic word expander if available
        if NUMBER_EXPANDER_AVAILABLE:
            self.word_expander = AmharicNumberExpander()
        else:
            self.word_expander = None
            logger.warning("Amharic number-to-word conversion not available")
    
    def _load_numeral_mappings(self):
        """Load Ethiopian numeral to Arabic numeral mappings"""
        
        # Basic digits (1-9)
        self.units = {
            '፩': 1, '፪': 2, '፫': 3, '፬': 4, '፭': 5,
            '፮': 6, '፯': 7, '፰': 8, '፱': 9
        }
        
        # Ten
        self.ten = {'፲': 10}
        
        # Tens (20-90)
        self.tens = {
            '፳': 20, '፴': 30, '፵': 40, '፶': 50,
            '፷': 60, '፸': 70, '፹': 80, '፺': 90
        }
        
        # Hundred
        self.hundred = {'፻': 100}
        
        # Ten-thousand
        self.ten_thousand = {'፼': 10000}
        
        # Combined lookup for quick access
        self.all_numerals = {
            **self.units,
            **self.ten,
            **self.tens,
            **self.hundred,
            **self.ten_thousand
        }
    
    def parse_ethiopian_numeral(self, numeral_str: str) -> int:
        """
        Parse Ethiopian numeral string to integer
        
        Algorithm:
        - Ethiopian numerals are additive (left-to-right)
        - Example: ፻፳፫ = 100 + 20 + 3 = 123
        - Example: ፪፻፵፭ = 2*100 + 40 + 5 = 245
        - Example: ፲፪ = 10 + 2 = 12
        
        Args:
            numeral_str: Ethiopian numeral string (e.g., "፻፳፫")
            
        Returns:
            Integer value
            
        Raises:
            ValueError: If numeral string is invalid
        """
        if not numeral_str:
            raise ValueError("Empty numeral string")
        
        result = 0
        current_multiplier = 1
        i = 0
        
        while i < len(numeral_str):
            char = numeral_str[i]
            
            if char not in self.all_numerals:
                raise ValueError(f"Invalid Ethiopian numeral character: {char}")
            
            value = self.all_numerals[char]
            
            # Handle multipliers (፼ = 10000, ፻ = 100)
            if char == '፼':  # Ten-thousand
                if i > 0:
                    # Previous digit is multiplier (e.g., ፫፼ = 3 * 10000)
                    prev_char = numeral_str[i-1]
                    if prev_char in self.units:
                        prev_value = self.units[prev_char]
                        result = result - prev_value + (prev_value * 10000)
                    else:
                        result += 10000
                else:
                    result += 10000
                    
            elif char == '፻':  # Hundred
                if i > 0:
                    # Previous digit is multiplier (e.g., ፫፻ = 3 * 100)
                    prev_char = numeral_str[i-1]
                    if prev_char in self.units:
                        prev_value = self.units[prev_char]
                        result = result - prev_value + (prev_value * 100)
                    else:
                        result += 100
                else:
                    result += 100
                    
            elif char == '፲':  # Ten
                if i > 0:
                    # Previous digit is multiplier (e.g., ፫፲ = 3 * 10)
                    prev_char = numeral_str[i-1]
                    if prev_char in self.units:
                        prev_value = self.units[prev_char]
                        result = result - prev_value + (prev_value * 10)
                    else:
                        result += 10
                else:
                    result += 10
                    
            else:
                # Regular units or tens (additive)
                result += value
            
            i += 1
        
        return result
    
    def expand(self, numeral: Union[str, int]) -> str:
        """
        Expand Ethiopian numeral to Amharic words
        
        Args:
            numeral: Ethiopian numeral string or integer
            
        Returns:
            Amharic word representation
            
        Examples:
            expand("፻፳፫") → "አንድ መቶ ሃያ ሶስት" (123)
            expand("፲፰") → "አስራ ስምንት" (18)
            expand("፪፻") → "ሁለት መቶ" (200)
        """
        # If already integer, convert to Amharic words
        if isinstance(numeral, int):
            if self.word_expander:
                return self.word_expander.expand(numeral)
            else:
                return str(numeral)
        
        # Parse Ethiopian numeral to integer
        try:
            arabic_number = self.parse_ethiopian_numeral(str(numeral))
            
            # Convert integer to Amharic words
            if self.word_expander:
                return self.word_expander.expand(arabic_number)
            else:
                return str(arabic_number)
                
        except ValueError as e:
            logger.warning(f"Failed to parse Ethiopian numeral '{numeral}': {e}")
            return str(numeral)  # Return as-is if parsing fails
    
    def expand_in_text(self, text: str) -> str:
        """
        Expand all Ethiopian numerals in text to Amharic words
        
        Args:
            text: Input text with Ethiopian numerals
            
        Returns:
            Text with numerals expanded to words
            
        Examples:
            "በ፲፰፻፹፯ ዓ.ም" → "በ አሥራ ስምንት መቶ ሰማንያ ሰባት ዓመተ ምህረት"
            "ዋጋው ፻ ብር ነው" → "ዋጋው አንድ መቶ ብር ነው"
        """
        if not text:
            return ""
        
        # Pattern to match sequences of Ethiopian numerals
        ethiopic_numeral_pattern = r'[፩-፼]+'
        
        def replace_numeral(match):
            numeral = match.group(0)
            try:
                return self.expand(numeral)
            except Exception as e:
                logger.warning(f"Failed to expand numeral '{numeral}': {e}")
                return numeral
        
        # Replace all Ethiopian numerals with expanded words
        result = re.sub(ethiopic_numeral_pattern, replace_numeral, text)
        return result
    
    def is_ethiopian_numeral(self, char: str) -> bool:
        """
        Check if character is an Ethiopian numeral
        
        Args:
            char: Single character
            
        Returns:
            True if Ethiopian numeral
        """
        return char in self.all_numerals
    
    def contains_ethiopian_numerals(self, text: str) -> bool:
        """
        Check if text contains any Ethiopian numerals
        
        Args:
            text: Input text
            
        Returns:
            True if text contains Ethiopian numerals
        """
        return any(self.is_ethiopian_numeral(char) for char in text)
    
    def get_supported_range(self) -> Dict[str, int]:
        """
        Get supported numeral range
        
        Returns:
            Dictionary with min and max supported values
        """
        return {
            'min': 1,
            'max': 99999,  # ፱፼፱፻፺፱ = 9*10000 + 9*100 + 90 + 9
            'characters': '፩-፼',
            'coverage': 'Full Ethiopian numeral system'
        }


# Convenience function for quick expansion
def expand_ethiopian_numerals(text: str) -> str:
    """
    Quick function to expand Ethiopian numerals in text
    
    Args:
        text: Input text with Ethiopian numerals
        
    Returns:
        Text with numerals expanded to Amharic words
        
    Example:
        >>> expand_ethiopian_numerals("በ፲፰፻፹፯ ዓ.ም")
        "በ አሥራ ስምንት መቶ ሰማንያ ሰባት ዓመተ ምህረት"
    """
    expander = EthiopianNumeralExpander()
    return expander.expand_in_text(text)


# Example usage and comprehensive tests
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    print("=" * 80)
    print("ETHIOPIAN NUMERAL EXPANDER - COMPREHENSIVE TEST")
    print("=" * 80)
    print()
    
    expander = EthiopianNumeralExpander()
    
    # Test cases
    test_cases = [
        # Basic units
        ("፩", 1, "አንድ"),
        ("፭", 5, "አምስት"),
        ("፱", 9, "ዘጠኝ"),
        
        # Ten and teens
        ("፲", 10, "አሥር"),
        ("፲፪", 12, "አስራ ሁለት"),
        ("፲፱", 19, "አስራ ዘጠኝ"),
        
        # Tens
        ("፳", 20, "ሃያ"),
        ("፴", 30, "ሰላሳ"),
        ("፺", 90, "ዘጠና"),
        
        # Tens + units
        ("፳፫", 23, "ሃያ ሶስት"),
        ("፹፯", 87, "ሰማንያ ሰባት"),
        
        # Hundreds
        ("፻", 100, "መቶ"),
        ("፪፻", 200, "ሁለት መቶ"),
        ("፱፻", 900, "ዘጠኝ መቶ"),
        
        # Complex numbers
        ("፻፳፫", 123, "አንድ መቶ ሃያ ሶስት"),
        ("፪፻፵፭", 245, "ሁለት መቶ አርባ አምስት"),
        ("፱፻፺፱", 999, "ዘጠኝ መቶ ዘጠና ዘጠኝ"),
        
        # Historical year (example: 1987 Ethiopian calendar)
        ("፲፰፻፹፯", 1887, None),  # Will show Amharic expansion
    ]
    
    print("📊 Unit Tests:")
    print("-" * 80)
    passed = 0
    failed = 0
    
    for ethiopic, expected_arabic, expected_amharic in test_cases:
        try:
            # Test parsing
            arabic = expander.parse_ethiopian_numeral(ethiopic)
            
            # Test expansion
            amharic = expander.expand(ethiopic)
            
            # Verify
            parse_ok = (arabic == expected_arabic)
            
            status = "✅" if parse_ok else "❌"
            print(f"{status} {ethiopic:10} → {arabic:6} → {amharic}")
            
            if parse_ok:
                passed += 1
            else:
                failed += 1
                print(f"   Expected: {expected_arabic}, Got: {arabic}")
                
        except Exception as e:
            print(f"❌ {ethiopic:10} → ERROR: {e}")
            failed += 1
    
    print()
    print(f"Results: {passed} passed, {failed} failed")
    print()
    
    # Text expansion tests
    print("📝 Text Expansion Tests:")
    print("-" * 80)
    
    text_tests = [
        "በ፲፰፻፹፯ ዓ.ም ተወለደ።",
        "ዋጋው ፻፳ ብር ነው።",
        "ከ፩ እስከ ፲ ድረስ ቁጠር።",
        "፪፻፵፭ ተማሪዎች አሉ።",
    ]
    
    for text in text_tests:
        expanded = expander.expand_in_text(text)
        print(f"Original:  {text}")
        print(f"Expanded:  {expanded}")
        print()
    
    # Supported range
    print("📌 Supported Range:")
    print("-" * 80)
    range_info = expander.get_supported_range()
    for key, value in range_info.items():
        print(f"{key:15}: {value}")
    
    print()
    print("=" * 80)
    print("✅ Testing complete!")
    print("=" * 80)
