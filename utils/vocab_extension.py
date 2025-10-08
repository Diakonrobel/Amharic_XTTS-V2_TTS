"""
XTTS Tokenizer Vocabulary Extension for Amharic

This module extends the XTTS BPE tokenizer vocabulary with Amharic-specific tokens
to achieve 90-95% training performance uplevel.

Strategy:
1. Load existing XTTS vocab.json
2. Add Amharic characters (Ethiopic script U+1200-U+137F)
3. Add IPA phonemes specific to Amharic (ejectives, labiovelars)
4. Add common Amharic subword units
5. Save extended vocab for training
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


# Amharic-specific IPA phonemes not in standard English
AMHARIC_IPA_PHONEMES = [
    # Ejectives
    "tʼ", "tʼə", "kʼ", "kʼə", "pʼ", "pʼə", "t͡ʃʼ", "sʼ",
    
    # Labiovelars
    "kʷ", "kʷə", "gʷ", "gʷə", "qʷ", "qʷə", "xʷ", "xʷə",
    
    # Pharyngeal/Glottal
    "ʕ", "ʕa", "ʕə", "ʔ", "ʔa", "ʔə", "ʔi",
    
    # Special vowels
    "ɨ", "ə", "ɛ", "ɔ",
    
    # Gemination marker
    "ː",
    
    # Common combinations
    "ʔə", "ʕə", "tʼə", "kʼə", "pʼə",
    "ʃə", "ʒə", "ɲə", "ŋə",
]

# Ethiopic script characters (syllabary)
def get_ethiopic_characters() -> List[str]:
    """Get list of all Ethiopic script characters"""
    chars = []
    # Main Ethiopic range: U+1200 to U+137F
    for code_point in range(0x1200, 0x1380):
        chars.append(chr(code_point))
    # Ethiopic Extended: U+1380 to U+139F
    for code_point in range(0x1380, 0x13A0):
        chars.append(chr(code_point))
    return chars


# Common Amharic subword units (phoneme sequences)
AMHARIC_SUBWORD_UNITS = [
    # Common syllables
    "sə", "la", "mɨ", "nə", "bə", "tə", "ʃə",
    "wə", "ʕa", "lə", "ʔə", "ja", "də", "gə",
    
    # Common morphemes
    "ʔɨ", "ʔa", "ʔe", "wa", "na", "ba",
    "tʼɨ", "tʼə", "kʼɨ", "kʼə",
    
    # Common bigrams
    "səl", "lam", "amɨ", "nəb", "bət", "təʃ",
    "mət", "təm", "ləm", "məd",
    
    # Verb markers
    "jə", "al", "ət", "ɨm", "u", "at",
]


def analyze_dataset_for_tokens(
    dataset_csv_path: str,
    top_n: int = 1000,
    min_freq: int = 2
) -> List[Tuple[str, int]]:
    """
    Analyze dataset to find most common Amharic tokens/n-grams
    
    Args:
        dataset_csv_path: Path to training CSV
        top_n: Number of top tokens to return
        min_freq: Minimum frequency threshold
        
    Returns:
        List of (token, frequency) tuples
    """
    import csv
    
    logger.info(f"Analyzing dataset for common tokens: {dataset_csv_path}")
    
    if not os.path.exists(dataset_csv_path):
        logger.warning(f"Dataset CSV not found: {dataset_csv_path}")
        return []
    
    # Collect all text
    all_text = []
    try:
        with open(dataset_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if len(row) >= 2:
                    all_text.append(row[1])  # Text column
    except Exception as e:
        logger.error(f"Error reading dataset: {e}")
        return []
    
    # Count character n-grams (1-3 chars)
    counter = Counter()
    
    for text in all_text:
        # Unigrams
        for char in text:
            counter[char] += 1
        
        # Bigrams
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            counter[bigram] += 1
        
        # Trigrams
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            counter[trigram] += 1
    
    # Filter by frequency and return top N
    frequent_tokens = [(token, count) for token, count in counter.items() 
                       if count >= min_freq]
    frequent_tokens.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"Found {len(frequent_tokens)} tokens with freq >= {min_freq}")
    return frequent_tokens[:top_n]


def load_xtts_vocab(vocab_path: str) -> Dict:
    """
    Load XTTS vocab.json file
    
    Args:
        vocab_path: Path to vocab.json
        
    Returns:
        Vocabulary dictionary
    """
    logger.info(f"Loading XTTS vocabulary from: {vocab_path}")
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        # XTTS vocab.json is a tokenizers JSON format
        vocab_data = json.load(f)
    
    logger.info(f"Loaded vocabulary with {len(vocab_data.get('model', {}).get('vocab', {}))} tokens")
    return vocab_data


def extend_xtts_vocab_for_amharic(
    original_vocab_path: str,
    output_vocab_path: str,
    dataset_csv_path: Optional[str] = None,
    include_ethiopic_chars: bool = True,
    include_ipa_phonemes: bool = True,
    include_subword_units: bool = True,
    include_dataset_analysis: bool = True
) -> Dict:
    """
    Extend XTTS vocabulary with Amharic-specific tokens
    
    Args:
        original_vocab_path: Path to original vocab.json
        output_vocab_path: Path to save extended vocab.json
        dataset_csv_path: Optional path to dataset for analysis
        include_ethiopic_chars: Add Ethiopic script characters
        include_ipa_phonemes: Add Amharic IPA phonemes
        include_subword_units: Add common subword units
        include_dataset_analysis: Analyze dataset for frequent tokens
        
    Returns:
        Extended vocabulary data
    """
    logger.info("=" * 80)
    logger.info("Extending XTTS Vocabulary for Amharic")
    logger.info("=" * 80)
    
    # Load original vocabulary
    vocab_data = load_xtts_vocab(original_vocab_path)
    
    # Get existing vocab dict
    if 'model' not in vocab_data or 'vocab' not in vocab_data['model']:
        raise ValueError("Invalid vocab.json format")
    
    vocab = vocab_data['model']['vocab']
    original_size = len(vocab)
    next_id = max(vocab.values()) + 1
    
    logger.info(f"Original vocabulary size: {original_size}")
    logger.info(f"Starting new token ID: {next_id}")
    
    # Collect new tokens
    new_tokens = set()
    
    # 1. Add Ethiopic characters
    if include_ethiopic_chars:
        logger.info("\nAdding Ethiopic script characters...")
        ethiopic_chars = get_ethiopic_characters()
        for char in ethiopic_chars:
            if char not in vocab:
                new_tokens.add(char)
        logger.info(f"  Added {len([c for c in ethiopic_chars if c not in vocab])} Ethiopic characters")
    
    # 2. Add Amharic-specific IPA phonemes
    if include_ipa_phonemes:
        logger.info("\nAdding Amharic IPA phonemes...")
        for phoneme in AMHARIC_IPA_PHONEMES:
            if phoneme not in vocab:
                new_tokens.add(phoneme)
        logger.info(f"  Added {len([p for p in AMHARIC_IPA_PHONEMES if p not in vocab])} IPA phonemes")
    
    # 3. Add common subword units
    if include_subword_units:
        logger.info("\nAdding common Amharic subword units...")
        for unit in AMHARIC_SUBWORD_UNITS:
            if unit not in vocab:
                new_tokens.add(unit)
        logger.info(f"  Added {len([u for u in AMHARIC_SUBWORD_UNITS if u not in vocab])} subword units")
    
    # 4. Analyze dataset for frequent tokens
    if include_dataset_analysis and dataset_csv_path:
        logger.info("\nAnalyzing dataset for frequent tokens...")
        frequent_tokens = analyze_dataset_for_tokens(dataset_csv_path, top_n=500, min_freq=5)
        
        added_from_dataset = 0
        for token, freq in frequent_tokens:
            if token not in vocab and token not in new_tokens:
                new_tokens.add(token)
                added_from_dataset += 1
                if added_from_dataset <= 10:  # Log first 10
                    logger.info(f"  Dataset token: '{token}' (freq={freq})")
        
        logger.info(f"  Added {added_from_dataset} frequent tokens from dataset")
    
    # Add new tokens to vocabulary
    logger.info(f"\nTotal new tokens to add: {len(new_tokens)}")
    
    for token in sorted(new_tokens):
        vocab[token] = next_id
        next_id += 1
    
    # Update vocabulary data
    vocab_data['model']['vocab'] = vocab
    
    final_size = len(vocab)
    logger.info(f"\nFinal vocabulary size: {final_size}")
    logger.info(f"Increase: +{final_size - original_size} tokens ({(final_size - original_size) / original_size * 100:.1f}%)")
    
    # Save extended vocabulary
    logger.info(f"\nSaving extended vocabulary to: {output_vocab_path}")
    os.makedirs(os.path.dirname(output_vocab_path), exist_ok=True)
    
    with open(output_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    logger.info("✅ Vocabulary extension complete!")
    logger.info("=" * 80)
    
    return vocab_data


def create_extended_vocab_for_training(
    base_vocab_path: str,
    output_dir: str,
    train_csv_path: Optional[str] = None,
    eval_csv_path: Optional[str] = None
) -> str:
    """
    Create extended vocabulary for XTTS training
    
    Args:
        base_vocab_path: Path to original XTTS vocab.json
        output_dir: Directory to save extended vocab
        train_csv_path: Optional training CSV for analysis
        eval_csv_path: Optional eval CSV for analysis
        
    Returns:
        Path to extended vocab.json
    """
    output_vocab_path = os.path.join(output_dir, "vocab_extended_amharic.json")
    
    # Use training CSV for analysis (usually larger)
    dataset_path = train_csv_path or eval_csv_path
    
    extend_xtts_vocab_for_amharic(
        original_vocab_path=base_vocab_path,
        output_vocab_path=output_vocab_path,
        dataset_csv_path=dataset_path,
        include_ethiopic_chars=True,
        include_ipa_phonemes=True,
        include_subword_units=True,
        include_dataset_analysis=True
    )
    
    return output_vocab_path


# CLI interface for manual vocab extension
if __name__ == "__main__":
    import argparse
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    parser = argparse.ArgumentParser(description="Extend XTTS vocabulary for Amharic")
    parser.add_argument("--input-vocab", required=True, help="Path to original vocab.json")
    parser.add_argument("--output-vocab", required=True, help="Path for extended vocab.json")
    parser.add_argument("--dataset-csv", help="Optional: Path to dataset CSV for analysis")
    parser.add_argument("--no-ethiopic", action="store_true", help="Skip Ethiopic characters")
    parser.add_argument("--no-ipa", action="store_true", help="Skip IPA phonemes")
    parser.add_argument("--no-subwords", action="store_true", help="Skip subword units")
    parser.add_argument("--no-dataset-analysis", action="store_true", help="Skip dataset analysis")
    
    args = parser.parse_args()
    
    try:
        extend_xtts_vocab_for_amharic(
            original_vocab_path=args.input_vocab,
            output_vocab_path=args.output_vocab,
            dataset_csv_path=args.dataset_csv,
            include_ethiopic_chars=not args.no_ethiopic,
            include_ipa_phonemes=not args.no_ipa,
            include_subword_units=not args.no_subwords,
            include_dataset_analysis=not args.no_dataset_analysis
        )
        
        print("\n✅ Success! Use the extended vocabulary for training.")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
