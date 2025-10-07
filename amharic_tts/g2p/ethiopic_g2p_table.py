"""
Complete Ethiopic Grapheme-to-Phoneme Mapping Table

This module provides comprehensive G2P mappings for Ethiopic script (Fidäl).
Covers all 33 base consonants across 7 orders (vowel modifications).

Ethiopic Orders:
- 1st order (ə): ሀ ለ ሐ መ... (base schwa)
- 2nd order (u): ሁ ሉ ሑ ሙ...
- 3rd order (i): ሂ ሊ ሒ ሚ...
- 4th order (a): ሃ ላ ሓ ማ...
- 5th order (e): ሄ ሌ ሔ ሜ...
- 6th order (ɨ): ህ ል ሕ ም... (bare consonant + ɨ)
- 7th order (o): ሆ ሎ ሖ ሞ...

Total: 231 mappings (33 consonants × 7 orders)
"""

# Complete Ethiopic G2P table
# Format: 'grapheme': 'IPA_phoneme'
ETHIOPIC_G2P_TABLE = {
    # ሀ-series (h) - Glottal fricative
    'ሀ': 'hə', 'ሁ': 'hu', 'ሂ': 'hi', 'ሃ': 'ha', 'ሄ': 'he', 'ህ': 'hɨ', 'ሆ': 'ho',
    
    # ለ-series (l) - Alveolar lateral
    'ለ': 'lə', 'ሉ': 'lu', 'ሊ': 'li', 'ላ': 'la', 'ሌ': 'le', 'ል': 'lɨ', 'ሎ': 'lo',
    
    # ሐ-series (ḥ) - Pharyngeal fricative (often merges with h)
    'ሐ': 'ħə', 'ሑ': 'ħu', 'ሒ': 'ħi', 'ሓ': 'ħa', 'ሔ': 'ħe', 'ሕ': 'ħɨ', 'ሖ': 'ħo',
    
    # መ-series (m) - Bilabial nasal
    'መ': 'mə', 'ሙ': 'mu', 'ሚ': 'mi', 'ማ': 'ma', 'ሜ': 'me', 'ም': 'mɨ', 'ሞ': 'mo',
    
    # ሠ-series (ś) - Alveolar fricative (merges with s)
    'ሠ': 'sə', 'ሡ': 'su', 'ሢ': 'si', 'ሣ': 'sa', 'ሤ': 'se', 'ሥ': 'sɨ', 'ሦ': 'so',
    
    # ረ-series (r) - Alveolar trill
    'ረ': 'rə', 'ሩ': 'ru', 'ሪ': 'ri', 'ራ': 'ra', 'ሬ': 're', 'ር': 'rɨ', 'ሮ': 'ro',
    
    # ሰ-series (s) - Voiceless alveolar fricative
    'ሰ': 'sə', 'ሱ': 'su', 'ሲ': 'si', 'ሳ': 'sa', 'ሴ': 'se', 'ስ': 'sɨ', 'ሶ': 'so',
    
    # ሸ-series (š) - Voiceless postalveolar fricative
    'ሸ': 'ʃə', 'ሹ': 'ʃu', 'ሺ': 'ʃi', 'ሻ': 'ʃa', 'ሼ': 'ʃe', 'ሽ': 'ʃɨ', 'ሾ': 'ʃo',
    
    # ቀ-series (q) - Uvular stop
    'ቀ': 'qə', 'ቁ': 'qu', 'ቂ': 'qi', 'ቃ': 'qa', 'ቄ': 'qe', 'ቅ': 'qɨ', 'ቆ': 'qo',
    
    # በ-series (b) - Voiced bilabial stop
    'በ': 'bə', 'ቡ': 'bu', 'ቢ': 'bi', 'ባ': 'ba', 'ቤ': 'be', 'ብ': 'bɨ', 'ቦ': 'bo',
    
    # ተ-series (t) - Voiceless alveolar stop
    'ተ': 'tə', 'ቱ': 'tu', 'ቲ': 'ti', 'ታ': 'ta', 'ቴ': 'te', 'ት': 'tɨ', 'ቶ': 'to',
    
    # ቸ-series (č) - Voiceless postalveolar affricate
    'ቸ': 'tʃə', 'ቹ': 'tʃu', 'ቺ': 'tʃi', 'ቻ': 'tʃa', 'ቼ': 'tʃe', 'ች': 'tʃɨ', 'ቾ': 'tʃo',
    
    # ኀ-series (ḫ) - Voiceless velar fricative (often merges with h)
    'ኀ': 'xə', 'ኁ': 'xu', 'ኂ': 'xi', 'ኃ': 'xa', 'ኄ': 'xe', 'ኅ': 'xɨ', 'ኆ': 'xo',
    
    # ነ-series (n) - Alveolar nasal
    'ነ': 'nə', 'ኑ': 'nu', 'ኒ': 'ni', 'ና': 'na', 'ኔ': 'ne', 'ን': 'nɨ', 'ኖ': 'no',
    
    # ኘ-series (ñ) - Palatal nasal
    'ኘ': 'ɲə', 'ኙ': 'ɲu', 'ኚ': 'ɲi', 'ኛ': 'ɲa', 'ኜ': 'ɲe', 'ኝ': 'ɲɨ', 'ኞ': 'ɲo',
    
    # አ-series (ʔ) - Glottal stop
    'አ': 'ʔə', 'ኡ': 'ʔu', 'ኢ': 'ʔi', 'ኣ': 'ʔa', 'ኤ': 'ʔe', 'እ': 'ʔɨ', 'ኦ': 'ʔo',
    
    # ከ-series (k) - Voiceless velar stop
    'ከ': 'kə', 'ኩ': 'ku', 'ኪ': 'ki', 'ካ': 'ka', 'ኬ': 'ke', 'ክ': 'kɨ', 'ኮ': 'ko',
    
    # ኸ-series (x) - Voiceless velar fricative
    'ኸ': 'xə', 'ኹ': 'xu', 'ኺ': 'xi', 'ኻ': 'xa', 'ኼ': 'xe', 'ኽ': 'xɨ', 'ኾ': 'xo',
    
    # ወ-series (w) - Labial-velar approximant
    'ወ': 'wə', 'ዉ': 'wu', 'ዊ': 'wi', 'ዋ': 'wa', 'ዌ': 'we', 'ው': 'wɨ', 'ዎ': 'wo',
    
    # ዐ-series (ʿ) - Pharyngeal fricative (often merges with ʔ)
    'ዐ': 'ʕə', 'ዑ': 'ʕu', 'ዒ': 'ʕi', 'ዓ': 'ʕa', 'ዔ': 'ʕe', 'ዕ': 'ʕɨ', 'ዖ': 'ʕo',
    
    # ዘ-series (z) - Voiced alveolar fricative
    'ዘ': 'zə', 'ዙ': 'zu', 'ዚ': 'zi', 'ዛ': 'za', 'ዜ': 'ze', 'ዝ': 'zɨ', 'ዞ': 'zo',
    
    # ዠ-series (ž) - Voiced postalveolar fricative
    'ዠ': 'ʒə', 'ዡ': 'ʒu', 'ዢ': 'ʒi', 'ዣ': 'ʒa', 'ዤ': 'ʒe', 'ዥ': 'ʒɨ', 'ዦ': 'ʒo',
    
    # የ-series (y) - Palatal approximant
    'የ': 'jə', 'ዩ': 'ju', 'ዪ': 'ji', 'ያ': 'ja', 'ዬ': 'je', 'ይ': 'jɨ', 'ዮ': 'jo',
    
    # ደ-series (d) - Voiced alveolar stop
    'ደ': 'də', 'ዱ': 'du', 'ዲ': 'di', 'ዳ': 'da', 'ዴ': 'de', 'ድ': 'dɨ', 'ዶ': 'do',
    
    # ጀ-series (ǧ) - Voiced postalveolar affricate
    'ጀ': 'dʒə', 'ጁ': 'dʒu', 'ጂ': 'dʒi', 'ጃ': 'dʒa', 'ጄ': 'dʒe', 'ጅ': 'dʒɨ', 'ጆ': 'dʒo',
    
    # ገ-series (g) - Voiced velar stop
    'ገ': 'gə', 'ጉ': 'gu', 'ጊ': 'gi', 'ጋ': 'ga', 'ጌ': 'ge', 'ግ': 'gɨ', 'ጎ': 'go',
    
    # ጠ-series (ṭ) - Alveolar ejective
    'ጠ': 'tʼə', 'ጡ': 'tʼu', 'ጢ': 'tʼi', 'ጣ': 'tʼa', 'ጤ': 'tʼe', 'ጥ': 'tʼɨ', 'ጦ': 'tʼo',
    
    # ጨ-series (č̣) - Postalveolar affricate ejective
    'ጨ': 'tʃʼə', 'ጩ': 'tʃʼu', 'ጪ': 'tʃʼi', 'ጫ': 'tʃʼa', 'ጬ': 'tʃʼe', 'ጭ': 'tʃʼɨ', 'ጮ': 'tʃʼo',
    
    # ጰ-series (ṗ) - Bilabial ejective
    'ጰ': 'pʼə', 'ጱ': 'pʼu', 'ጲ': 'pʼi', 'ጳ': 'pʼa', 'ጴ': 'pʼe', 'ጵ': 'pʼɨ', 'ጶ': 'pʼo',
    
    # ጸ-series (ṣ) - Alveolar ejective fricative
    'ጸ': 'sʼə', 'ጹ': 'sʼu', 'ጺ': 'sʼi', 'ጻ': 'sʼa', 'ጼ': 'sʼe', 'ጽ': 'sʼɨ', 'ጾ': 'sʼo',
    
    # ፀ-series (ṣ́) - Alternative alveolar ejective fricative (merges with ጸ)
    'ፀ': 'sʼə', 'ፁ': 'sʼu', 'ፂ': 'sʼi', 'ፃ': 'sʼa', 'ፄ': 'sʼe', 'ፅ': 'sʼɨ', 'ፆ': 'sʼo',
    
    # ፈ-series (f) - Voiceless labiodental fricative
    'ፈ': 'fə', 'ፉ': 'fu', 'ፊ': 'fi', 'ፋ': 'fa', 'ፌ': 'fe', 'ፍ': 'fɨ', 'ፎ': 'fo',
    
    # ፐ-series (p) - Voiceless bilabial stop
    'ፐ': 'pə', 'ፑ': 'pu', 'ፒ': 'pi', 'ፓ': 'pa', 'ፔ': 'pe', 'ፕ': 'pɨ', 'ፖ': 'po',
}

# Labiovelar variants (kʷ, gʷ, qʷ, xʷ)
LABIOVELAR_TABLE = {
    # ቈ-series (qʷ)
    'ቈ': 'qʷə', 'ቊ': 'qʷu', 'ቋ': 'qʷa', 'ቌ': 'qʷe', 'ቍ': 'qʷi',
    
    # ኰ-series (kʷ)
    'ኰ': 'kʷə', 'ኲ': 'kʷu', 'ኳ': 'kʷa', 'ኴ': 'kʷe', 'ኵ': 'kʷi',
    
    # ጐ-series (gʷ)
    'ጐ': 'gʷə', 'ጒ': 'gʷu', 'ጓ': 'gʷa', 'ጔ': 'gʷe', 'ጕ': 'gʷi',
    
    # ኸ-series labiovelars (xʷ)
    'ዀ': 'xʷə', 'ዂ': 'xʷu', 'ዃ': 'xʷa', 'ዄ': 'xʷe', 'ዅ': 'xʷi',
}

# Ethiopic punctuation
ETHIOPIC_PUNCTUATION = {
    '።': '.',   # Full stop (serez)
    '፣': ',',   # Comma (nesib)
    '፤': ';',   # Semicolon (mekfel)
    '፥': ':',   # Colon (nebteb)
    '፦': '::',  # Preface colon
    '፧': '?',   # Question mark
    '፨': '¶',   # Paragraph separator
    '፡': ' ',   # Word separator (convert to space)
}

# Ethiopic numerals (optional)
ETHIOPIC_NUMERALS = {
    '፩': '1', '፪': '2', '፫': '3', '፬': '4', '፭': '5',
    '፮': '6', '፯': '7', '፰': '8', '፱': '9', '፲': '10',
    '፳': '20', '፴': '30', '፵': '40', '፶': '50',
    '፷': '60', '፸': '70', '፹': '80', '፺': '90', '፻': '100',
}

# Combined complete table
COMPLETE_G2P_TABLE = {
    **ETHIOPIC_G2P_TABLE,
    **LABIOVELAR_TABLE,
    **ETHIOPIC_PUNCTUATION,
}

# Statistics
def get_table_stats():
    """Get statistics about the G2P table"""
    return {
        'total_mappings': len(COMPLETE_G2P_TABLE),
        'consonant_series': len(ETHIOPIC_G2P_TABLE) // 7,
        'labiovelar_forms': len(LABIOVELAR_TABLE),
        'punctuation_marks': len(ETHIOPIC_PUNCTUATION),
        'coverage': 'All 33 Ethiopic consonants × 7 orders'
    }


if __name__ == "__main__":
    # Display statistics
    stats = get_table_stats()
    print("Ethiopic G2P Table Statistics:")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key:20}: {value}")
    
    print("\n" + "=" * 50)
    print(f"Total entries in table: {len(COMPLETE_G2P_TABLE)}")
    
    # Test some conversions
    print("\nSample Conversions:")
    print("=" * 50)
    test_chars = ['ሰ', 'ላ', 'ም', 'አ', 'ማ', 'ር', 'ኛ']
    for char in test_chars:
        if char in COMPLETE_G2P_TABLE:
            print(f"{char} → {COMPLETE_G2P_TABLE[char]}")
