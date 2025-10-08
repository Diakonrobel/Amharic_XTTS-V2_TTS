"""
Preprocess Amharic Dataset for XTTS Training

This script converts Amharic text in your training CSV files to IPA phonemes
so that the standard XTTS tokenizer can process them without unknown tokens.
"""

import os
import sys
import csv
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from amharic_tts.tokenizer.xtts_tokenizer_wrapper import XTTSAmharicTokenizer
    from amharic_tts.tokenizer.hybrid_tokenizer import HybridAmharicTokenizer
except ImportError as e:
    print(f"Error importing Amharic tokenizer: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def preprocess_csv(input_csv_path, output_csv_path, tokenizer):
    """
    Preprocess a CSV file by converting Amharic text to phonemes
    
    Args:
        input_csv_path: Path to input CSV
        output_csv_path: Path to output CSV
        tokenizer: Tokenizer instance with G2P enabled
    """
    print(f"\n📝 Processing: {input_csv_path}")
    print(f"   Output to: {output_csv_path}")
    
    processed_count = 0
    failed_count = 0
    rows_processed = []
    
    # Read input CSV
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        
        for row in reader:
            if len(row) < 2:
                rows_processed.append(row)
                continue
            
            audio_path = row[0]
            text = row[1]
            speaker_name = row[2] if len(row) > 2 else ""
            
            # Check if text is Amharic
            is_amharic = tokenizer.is_amharic(text)
            
            if is_amharic:
                try:
                    # Convert to phonemes/IPA
                    phoneme_text = tokenizer.preprocess_text(text, lang="am")
                    
                    # Create new row with phoneme text
                    new_row = [audio_path, phoneme_text, speaker_name] if speaker_name else [audio_path, phoneme_text]
                    rows_processed.append(new_row)
                    processed_count += 1
                    
                    if processed_count <= 5:  # Show first 5 examples
                        print(f"   ✓ {text[:50]}...")
                        print(f"     → {phoneme_text[:50]}...")
                
                except Exception as e:
                    print(f"   ✗ Failed to process: {text[:50]}...")
                    print(f"     Error: {e}")
                    # Keep original text on failure
                    rows_processed.append(row)
                    failed_count += 1
            else:
                # Keep non-Amharic text as-is
                rows_processed.append(row)
    
    # Write output CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(rows_processed)
    
    print(f"   ✅ Processed {processed_count} Amharic entries")
    if failed_count > 0:
        print(f"   ⚠️  Failed {failed_count} entries")
    
    return processed_count, failed_count


def main():
    parser = argparse.ArgumentParser(description="Preprocess Amharic dataset for XTTS training")
    parser.add_argument("--input-train", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--input-eval", type=str, required=True, help="Path to evaluation CSV")
    parser.add_argument("--output-train", type=str, help="Path for preprocessed training CSV (default: input_train_preprocessed.csv)")
    parser.add_argument("--output-eval", type=str, help="Path for preprocessed evaluation CSV (default: input_eval_preprocessed.csv)")
    parser.add_argument("--vocab", type=str, help="Path to vocab.json (optional)")
    parser.add_argument("--g2p-backend", type=str, default="auto", 
                       choices=["auto", "transphone", "epitran", "rule-based"],
                       help="G2P backend to use (default: auto)")
    
    args = parser.parse_args()
    
    # Set default output paths if not specified
    if not args.output_train:
        input_path = Path(args.input_train)
        args.output_train = str(input_path.parent / f"{input_path.stem}_preprocessed{input_path.suffix}")
    
    if not args.output_eval:
        input_path = Path(args.input_eval)
        args.output_eval = str(input_path.parent / f"{input_path.stem}_preprocessed{input_path.suffix}")
    
    print("=" * 80)
    print("🔧 Amharic Dataset Preprocessor for XTTS")
    print("=" * 80)
    print(f"\nG2P Backend: {args.g2p_backend}")
    print(f"Vocab file:  {args.vocab or 'Using default'}")
    
    # Create tokenizer with G2P enabled
    print("\n📦 Loading Amharic tokenizer with G2P...")
    try:
        tokenizer = XTTSAmharicTokenizer(
            vocab_file=args.vocab,
            use_phonemes=True  # Enable G2P preprocessing
        )
        print("   ✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load tokenizer: {e}")
        return 1
    
    # Process training CSV
    print("\n" + "=" * 80)
    print("Processing Training Data")
    print("=" * 80)
    train_processed, train_failed = preprocess_csv(
        args.input_train,
        args.output_train,
        tokenizer
    )
    
    # Process evaluation CSV
    print("\n" + "=" * 80)
    print("Processing Evaluation Data")
    print("=" * 80)
    eval_processed, eval_failed = preprocess_csv(
        args.input_eval,
        args.output_eval,
        tokenizer
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Summary")
    print("=" * 80)
    print(f"Training:   {train_processed} processed, {train_failed} failed")
    print(f"Evaluation: {eval_processed} processed, {eval_failed} failed")
    print(f"\n✅ Output files:")
    print(f"   - {args.output_train}")
    print(f"   - {args.output_eval}")
    print("\n💡 Next steps:")
    print("   1. Update your training configuration to use the preprocessed CSV files")
    print("   2. Set language to 'en' in the WebUI (phonemes are now Latin characters)")
    print("   3. Start training!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
