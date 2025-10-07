#!/usr/bin/env python3
"""
Interactive Demo for Enhanced Amharic G2P System

Showcases the comprehensive Ethiopic grapheme-to-phoneme conversion
with full phonological rule application.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from amharic_tts.g2p.amharic_g2p_enhanced import EnhancedAmharicG2P
from amharic_tts.g2p.ethiopic_g2p_table import get_table_stats


def print_header():
    """Print demo header"""
    print("=" * 80)
    print("🇪🇹  AMHARIC G2P SYSTEM - INTERACTIVE DEMO  🇪🇹")
    print("=" * 80)
    print()


def print_table_stats():
    """Display G2P table statistics"""
    stats = get_table_stats()
    
    print("📊 G2P Table Coverage:")
    print("-" * 80)
    print(f"  Total Mappings:       {stats['total_mappings']}")
    print(f"  Consonant Series:     {stats['consonant_series']}")
    print(f"  Labiovelar Forms:     {stats['labiovelar_forms']}")
    print(f"  Punctuation Marks:    {stats['punctuation_marks']}")
    print(f"  Coverage:             {stats['coverage']}")
    print()


def demo_basic_words(g2p):
    """Demonstrate basic word conversions"""
    print("📝 Basic Word Conversions:")
    print("-" * 80)
    
    examples = [
        ("ሰላም", "Peace/Hello"),
        ("አማርኛ", "Amharic language"),
        ("ኢትዮጵያ", "Ethiopia"),
        ("አዲስ አበባ", "Addis Ababa"),
        ("ስፖርት", "Sport"),
    ]
    
    for amharic, meaning in examples:
        phonemes = g2p.convert(amharic)
        print(f"  {amharic:15} → {phonemes:25} ({meaning})")
    print()


def demo_vowel_orders(g2p):
    """Demonstrate all 7 vowel orders"""
    print("🔤 Seven Vowel Orders (using ለ-series):")
    print("-" * 80)
    
    orders = [
        ("ለ", "1st order (ə)", "lə"),
        ("ሉ", "2nd order (u)", "lu"),
        ("ሊ", "3rd order (i)", "li"),
        ("ላ", "4th order (a)", "la"),
        ("ሌ", "5th order (e)", "le"),
        ("ል", "6th order (ɨ)", "lɨ"),
        ("ሎ", "7th order (o)", "lo"),
    ]
    
    for char, description, expected in orders:
        result = g2p.convert(char)
        check = "✅" if expected in result else "⚠️"
        print(f"  {char}  {description:20} → {result:10} {check}")
    print()


def demo_ejectives(g2p):
    """Demonstrate ejective consonants"""
    print("💥 Ejective Consonants (unique to Amharic):")
    print("-" * 80)
    
    examples = [
        ("ጠና", "tʼ - ejective t", "Health/wellness"),
        ("ጨርቅ", "tʃʼ - ejective ch", "Cloth"),
        ("ጸሐይ", "sʼ - ejective s", "Sun"),
        ("ጰራሪ", "pʼ - ejective p", "Louse"),
    ]
    
    for word, phoneme_desc, meaning in examples:
        result = g2p.convert(word)
        has_ejective = "ʼ" in result
        check = "✅" if has_ejective else "⚠️"
        print(f"  {word:10} → {result:20} {check} ({phoneme_desc} - {meaning})")
    print()


def demo_labiovelars(g2p):
    """Demonstrate labiovelar consonants"""
    print("🌐 Labiovelar Consonants:")
    print("-" * 80)
    
    examples = [
        ("ቋንቋ", "qʷ - labialized q", "Language"),
        ("ኳስ", "kʷ - labialized k", "Ball"),
        ("ጓደኛ", "gʷ - labialized g", "Friend"),
    ]
    
    for word, phoneme_desc, meaning in examples:
        result = g2p.convert(word)
        has_labiovelar = "ʷ" in result
        check = "✅" if has_labiovelar else "⚠️"
        print(f"  {word:10} → {result:20} {check} ({phoneme_desc} - {meaning})")
    print()


def demo_epenthesis(g2p):
    """Demonstrate epenthetic vowel insertion"""
    print("🔄 Epenthesis (ɨ insertion):")
    print("-" * 80)
    
    examples = [
        ("ክብር", "Honor/respect (k-b-r cluster)"),
        ("ትግራይ", "Tigray region (t-g-r cluster)"),
        ("ብርሃን", "Light/brightness (multiple clusters)"),
        ("ስፖርት", "Sport (loan word with clusters)"),
    ]
    
    for word, description in examples:
        result = g2p.convert(word)
        epenthesis_count = result.count("ɨ")
        print(f"  {word:10} → {result:25} ({epenthesis_count} ɨ insertions - {description})")
    print()


def demo_palatal_nasal(g2p):
    """Demonstrate palatal nasal (unique phoneme)"""
    print("👄 Palatal Nasal (ɲ - unique to Amharic):")
    print("-" * 80)
    
    examples = [
        ("አማርኛ", "Amharic language"),
        ("ኛ", "Our (possessive suffix)"),
        ("ምኞት", "Wish/desire"),
    ]
    
    for word, meaning in examples:
        result = g2p.convert(word)
        has_palatal = "ɲ" in result
        check = "✅" if has_palatal else "⚠️"
        print(f"  {word:10} → {result:25} {check} ({meaning})")
    print()


def demo_phrases(g2p):
    """Demonstrate phrase conversions"""
    print("💬 Phrase Conversions:")
    print("-" * 80)
    
    phrases = [
        ("ሰላም ነው", "It is peace / Greetings"),
        ("መልካም ቀን", "Good day"),
        ("እንዴት ነህ", "How are you? (masculine)"),
        ("እንዴት ነሽ", "How are you? (feminine)"),
        ("አመሰግናለሁ", "Thank you"),
    ]
    
    for phrase, meaning in phrases:
        result = g2p.convert(phrase)
        print(f"  {phrase:15} → {result}")
        print(f"  {' ' * 17} ({meaning})")
        print()


def demo_punctuation(g2p):
    """Demonstrate Ethiopic punctuation"""
    print("📍 Ethiopic Punctuation:")
    print("-" * 80)
    
    examples = [
        ("ሰላም።", "Full stop (።)"),
        ("ሰላም፣ እንዴት ነህ፧", "Comma (፣) and question mark (፧)"),
        ("አንድ፣ ሁለት፣ ሶስት።", "One, two, three."),
    ]
    
    for text, description in examples:
        result = g2p.convert(text)
        print(f"  {text:25} → {result}")
        print(f"  {' ' * 27} ({description})")
        print()


def interactive_mode(g2p):
    """Interactive conversion mode"""
    print("✏️  Interactive Mode:")
    print("-" * 80)
    print("Enter Amharic text to convert (or 'quit' to exit)")
    print()
    
    while True:
        try:
            text = input("Amharic> ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not text:
                continue
            
            result = g2p.convert(text)
            print(f"IPA>     {result}")
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()


def main():
    """Main demo function"""
    print_header()
    
    # Initialize G2P converter
    print("🔧 Initializing Enhanced Amharic G2P System...")
    g2p = EnhancedAmharicG2P()
    print("✅ Initialization complete!\n")
    
    # Show table statistics
    print_table_stats()
    
    # Run demonstrations
    demo_basic_words(g2p)
    demo_vowel_orders(g2p)
    demo_ejectives(g2p)
    demo_labiovelars(g2p)
    demo_epenthesis(g2p)
    demo_palatal_nasal(g2p)
    demo_phrases(g2p)
    demo_punctuation(g2p)
    
    # Interactive mode
    print()
    try:
        interactive_mode(g2p)
    except Exception as e:
        print(f"❌ Error in interactive mode: {e}")
    
    print()
    print("=" * 80)
    print("✅ Demo complete! Check docs/AMHARIC_G2P_PHASE3.md for full documentation.")
    print("=" * 80)


if __name__ == "__main__":
    main()
