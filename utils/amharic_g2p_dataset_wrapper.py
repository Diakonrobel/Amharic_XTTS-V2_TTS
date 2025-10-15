"""
Amharic G2P Dataset Wrapper for XTTS Training

This module provides on-the-fly G2P preprocessing for training datasets,
allowing seamless integration with the XTTS training pipeline.
"""

import os
import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def detect_amharic_text(text: str) -> bool:
    """
    Detect if text contains Amharic (Ethiopic) script characters
    
    Args:
        text: Input text string
        
    Returns:
        True if text contains Amharic characters, False otherwise
    """
    if not text:
        return False
    
    # Ethiopic Unicode range: U+1200 to U+137F and U+1380 to U+139F
    for char in text:
        code_point = ord(char)
        if (0x1200 <= code_point <= 0x137F) or (0x1380 <= code_point <= 0x139F):
            return True
    return False


def is_dataset_already_preprocessed(csv_path: str, sample_size: int = 10) -> bool:
    """
    Check if a dataset CSV file is already preprocessed (contains phonemes vs Amharic script)
    
    Args:
        csv_path: Path to CSV file
        sample_size: Number of rows to check
        
    Returns:
        True if dataset appears to be preprocessed, False if contains Amharic script
    """
    if not os.path.exists(csv_path):
        logger.warning(f"CSV file not found: {csv_path}")
        return False
    
    try:
        amharic_count = 0
        total_checked = 0
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for i, row in enumerate(reader):
                if i >= sample_size:
                    break
                if len(row) < 2:
                    continue
                
                text = row[1]
                total_checked += 1
                
                if detect_amharic_text(text):
                    amharic_count += 1
        
        if total_checked == 0:
            logger.warning(f"No valid text found in {csv_path}")
            return False
        
        # If more than 50% of samples contain Amharic script, dataset is NOT preprocessed
        amharic_ratio = amharic_count / total_checked
        is_preprocessed = amharic_ratio < 0.5
        
        logger.info(f"Dataset check: {csv_path}")
        logger.info(f"  Amharic ratio: {amharic_ratio:.2%} ({amharic_count}/{total_checked} samples)")
        logger.info(f"  Status: {'Already preprocessed' if is_preprocessed else 'Needs G2P conversion'}")
        
        return is_preprocessed
        
    except Exception as e:
        logger.error(f"Error checking dataset: {e}")
        return False


def preprocess_training_samples_with_g2p(
    samples: List[Dict],
    g2p_tokenizer,
    language: str = "am"
) -> List[Dict]:
    """
    Preprocess training samples by converting Amharic text to phonemes
    
    Args:
        samples: List of training sample dictionaries
        g2p_tokenizer: Tokenizer instance with G2P enabled
        language: Language code (am/amh for Amharic)
        
    Returns:
        List of preprocessed samples with text converted to phonemes
    """
    preprocessed_samples = []
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    logger.info(f"Starting G2P preprocessing for {len(samples)} samples...")
    
    for idx, sample in enumerate(samples):
        try:
            original_text = sample.get("text", "")
            
            # Check if text is Amharic
            if not detect_amharic_text(original_text):
                # Not Amharic, keep as-is but update language to 'en' for consistency
                new_sample = sample.copy()
                new_sample["language"] = "en"  # Switch to 'en' for consistency
                preprocessed_samples.append(new_sample)
                skip_count += 1
                continue
            
            # Convert to phonemes
            phoneme_text = g2p_tokenizer.preprocess_text(original_text, lang=language)
            
            # Create new sample with phoneme text AND update language to 'en'
            new_sample = sample.copy()
            new_sample["text"] = phoneme_text
            new_sample["language"] = "en"  # CRITICAL: Switch to 'en' since phonemes use Latin alphabet
            preprocessed_samples.append(new_sample)
            success_count += 1
            
            # Log first few conversions
            if success_count <= 5:
                logger.info(f"  Sample {idx + 1}: {original_text[:50]}...")
                logger.info(f"             → {phoneme_text[:50]}...")
        
        except Exception as e:
            logger.warning(f"Failed to preprocess sample {idx + 1}: {e}")
            # Keep original sample on failure
            preprocessed_samples.append(sample.copy())
            fail_count += 1
    
    logger.info(f"G2P preprocessing complete:")
    logger.info(f"  ✓ Converted: {success_count} samples")
    logger.info(f"  ○ Skipped (non-Amharic): {skip_count} samples")
    logger.info(f"  ✗ Failed: {fail_count} samples")
    
    return preprocessed_samples


def apply_g2p_to_training_data(
    train_samples: List[Dict],
    eval_samples: List[Dict],
    train_csv_path: str,
    eval_csv_path: str,
    language: str = "am",
    g2p_backend: str = "transphone"
) -> tuple[List[Dict], List[Dict], str]:
    """
    Apply G2P preprocessing to training and evaluation samples
    
    Args:
        train_samples: Training sample list
        eval_samples: Evaluation sample list
        train_csv_path: Path to training CSV (for detection)
        eval_csv_path: Path to evaluation CSV (for detection)
        language: Language code
        g2p_backend: G2P backend to use
        
    Returns:
        Tuple of (preprocessed_train_samples, preprocessed_eval_samples, effective_language)
    """
    logger.info("=" * 80)
    logger.info("Amharic G2P Integration Enabled")
    logger.info("=" * 80)
    
    # Check if dataset is already preprocessed
    train_preprocessed = is_dataset_already_preprocessed(train_csv_path)
    eval_preprocessed = is_dataset_already_preprocessed(eval_csv_path)
    
    if train_preprocessed and eval_preprocessed:
        logger.info("✓ Dataset already contains phonemes - no conversion needed")
        logger.info("  Using 'en' language code for tokenization")
        return train_samples, eval_samples, "en"
    
    if train_preprocessed != eval_preprocessed:
        logger.warning("⚠ Inconsistent preprocessing detected!")
        logger.warning(f"  Train CSV preprocessed: {train_preprocessed}")
        logger.warning(f"  Eval CSV preprocessed: {eval_preprocessed}")
        logger.warning("  Will attempt to preprocess both to ensure consistency")
    
    # Load G2P tokenizer with dynamic backend
    logger.info(f"Loading Amharic G2P tokenizer (backend: {g2p_backend})...")
    
    try:
        from amharic_tts.tokenizer.xtts_tokenizer_wrapper import create_xtts_tokenizer
        
        # Create tokenizer with specified backend (NOT hardcoded!)
        tokenizer = create_xtts_tokenizer(
            vocab_file=None,
            use_phonemes=True,
            g2p_backend=g2p_backend  # Pass backend to tokenizer
        )
        logger.info(f"✓ Amharic G2P tokenizer loaded successfully (backend: {g2p_backend})")
        
    except ImportError as e:
        logger.error(f"✗ Failed to load Amharic G2P tokenizer: {e}")
        logger.error("  Continuing without G2P - training may fail with UNK token errors!")
        return train_samples, eval_samples, language
    
    # Preprocess training samples
    logger.info("\n" + "-" * 80)
    logger.info("Processing Training Samples")
    logger.info("-" * 80)
    train_samples_processed = preprocess_training_samples_with_g2p(
        train_samples,
        tokenizer,
        language
    )
    
    # Preprocess evaluation samples
    logger.info("\n" + "-" * 80)
    logger.info("Processing Evaluation Samples")
    logger.info("-" * 80)
    eval_samples_processed = preprocess_training_samples_with_g2p(
        eval_samples,
        tokenizer,
        language
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("G2P Integration Complete")
    logger.info("=" * 80)
    logger.info("✓ Training will use phoneme representations")
    logger.info("✓ Language code switched to 'en' for XTTS tokenizer compatibility")
    logger.info("=" * 80 + "\n")
    
    # Return preprocessed samples and switch language to 'en' for tokenizer
    return train_samples_processed, eval_samples_processed, "en"


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Amharic G2P Dataset Wrapper - Test")
    print("=" * 80)
    
    # Test detection
    test_texts = [
        "ሰላም ዓለም",  # Amharic
        "Hello world",  # English
        "salam ʕaləm",  # IPA phonemes
    ]
    
    print("\nText Detection Tests:")
    for text in test_texts:
        is_amharic = detect_amharic_text(text)
        print(f"  '{text}' -> {'Amharic' if is_amharic else 'Not Amharic'}")
    
    print("\n" + "=" * 80)
    print("Test complete!")
