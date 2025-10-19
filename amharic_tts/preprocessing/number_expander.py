"""
Amharic Number Expander

Converts Arabic numerals and Ethiopic numerals to Amharic words.
"""

import re
from typing import Union


class AmharicNumberExpander:
    """Expand numbers to Amharic words"""
    
    def __init__(self):
        self._load_number_vocabulary()
        
    def _load_number_vocabulary(self):
        """Load Amharic number vocabulary"""
        
        # Basic numbers 0-19
        self.ones = {
            0: '',
            1: 'አንድ',
            2: 'ሁለት',
            3: 'ሶስት',
            4: 'አራት',
            5: 'አምስት',
            6: 'ስድስት',
            7: 'ሰባት',
            8: 'ስምንት',
            9: 'ዘጠኝ',
            10: 'አሥር',
            11: 'አስራ አንድ',
            12: 'አስራ ሁለት',
            13: 'አስራ ሶስት',
            14: 'አስራ አራት',
            15: 'አስራ አምስት',
            16: 'አስራ ስድስት',
            17: 'አስራ ሰባት',
            18: 'አስራ ስምንት',
            19: 'አስራ ዘጠኝ',
        }
        
        # Tens 20-90
        self.tens = {
            20: 'ሃያ',
            30: 'ሰላሳ',
            40: 'አርባ',
            50: 'ሃምሳ',
            60: 'ስልሳ',
            70: 'ሰባ',
            80: 'ሰማንያ',
            90: 'ዘጠና',
        }
        
        # Hundreds
        self.hundreds_word = 'መቶ'
        
        # Thousands
        self.thousands_word = 'ሺህ'
        
        # Millions
        self.millions_word = 'ሚሊዮን'
        
        # Billions
        self.billions_word = 'ቢሊዮን'
        
    def expand(self, number: Union[int, str]) -> str:
        """
        Expand a number to Amharic words
        
        Args:
            number: Integer or string representation of number
                    Supports comma-separated numbers (e.g., "15,000")
            
        Returns:
            Amharic word representation
        """
        try:
            # Remove commas for numbers like 15,000 or 1,000,000
            num_str = str(number).strip().replace(',', '')
            num = int(num_str)
        except ValueError:
            return str(number)  # Return as-is if not a valid number
            
        if num == 0:
            return 'ዜሮ'
            
        if num < 0:
            return 'መቀነስ ' + self.expand(abs(num))
            
        return self._expand_number(num).strip()
    
    def expand_number(self, number: Union[int, str]) -> str:
        """
        Alias for expand() method
        Expand a number to Amharic words
        
        Args:
            number: Integer or string representation of number
            
        Returns:
            Amharic word representation
        """
        return self.expand(number)
        
    def _expand_number(self, num: int) -> str:
        """Recursively expand a number"""
        
        if num < 20:
            return self.ones.get(num, '')
            
        if num < 100:
            tens_digit = (num // 10) * 10
            ones_digit = num % 10
            result = self.tens.get(tens_digit, '')
            if ones_digit > 0:
                result += ' ' + self.ones[ones_digit]
            return result
            
        if num < 1000:
            hundreds_digit = num // 100
            remainder = num % 100
            
            if hundreds_digit == 1:
                result = self.hundreds_word
            else:
                result = self.ones[hundreds_digit] + ' ' + self.hundreds_word
                
            if remainder > 0:
                result += ' ' + self._expand_number(remainder)
            return result
            
        if num < 1000000:
            thousands_digit = num // 1000
            remainder = num % 1000
            
            # Special case: 1000 = "ሽህ" not "አንድ ሽህ"
            if thousands_digit == 1:
                result = self.thousands_word
            else:
                result = self._expand_number(thousands_digit) + ' ' + self.thousands_word
            
            if remainder > 0:
                result += ' ' + self._expand_number(remainder)
            return result
            
        if num < 1000000000:
            millions_digit = num // 1000000
            remainder = num % 1000000
            
            result = self._expand_number(millions_digit) + ' ' + self.millions_word
            
            if remainder > 0:
                result += ' ' + self._expand_number(remainder)
            return result
            
        # Billions and above
        billions_digit = num // 1000000000
        remainder = num % 1000000000
        
        result = self._expand_number(billions_digit) + ' ' + self.billions_word
        
        if remainder > 0:
            result += ' ' + self._expand_number(remainder)
        return result


def expand_amharic_numbers(text: str) -> str:
    """
    Expand all numbers in Amharic text to words
    
    Args:
        text: Input text with numbers
        
    Returns:
        Text with numbers expanded to words
    """
    expander = AmharicNumberExpander()
    
    # Find all numbers in the text
    def replace_number(match):
        number = match.group(0)
        return expander.expand(number)
    
    # Replace Arabic numerals
    text = re.sub(r'\d+', replace_number, text)
    
    # TODO: Handle Ethiopic numerals (፩፪፫... etc.) if needed
    
    return text


# Example usage and test
if __name__ == "__main__":
    expander = AmharicNumberExpander()
    
    test_numbers = [0, 1, 5, 10, 15, 20, 25, 50, 99, 100, 123, 500, 1000, 2025, 10000, 1000000]
    
    print("Amharic Number Expansion Tests:")
    print("=" * 50)
    for num in test_numbers:
        print(f"{num:>10} → {expander.expand(num)}")
    
    print("\n" + "=" * 50)
    print("Text expansion test:")
    test_text = "ዛሬ 25 የካቲት 2025 ነው። በ 100 ሺህ ብር ገዛሁት።"
    print(f"Original: {test_text}")
    print(f"Expanded: {expand_amharic_numbers(test_text)}")
