"""
Quick Amharic Dataset Preprocessor

This script automatically detects your filtered CSV files and preprocesses them.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the main preprocessing function
from preprocess_amharic_dataset import preprocess_csv, XTTSAmharicTokenizer


def find_csv_files(base_dir):
    """Find the filtered CSV files in the dataset directory"""
    dataset_dir = Path(base_dir) / "dataset"
    
    # Look for filtered CSV files
    train_candidates = list(dataset_dir.glob("*train*filtered*.csv"))
    eval_candidates = list(dataset_dir.glob("*eval*filtered*.csv"))
    
    # If not found, look for any train/eval CSV
    if not train_candidates:
        train_candidates = list(dataset_dir.glob("*train*.csv"))
    if not eval_candidates:
        eval_candidates = list(dataset_dir.glob("*eval*.csv"))
    
    return train_candidates, eval_candidates


def main():
    print("=" * 80)
    print("🚀 Quick Amharic Preprocessor")
    print("=" * 80)
    
    # Try to auto-detect your project directory
    possible_dirs = [
        "/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS",
        "/teamspace/studios/Amharic_XTTS-V2_TTS",
        os.path.expanduser("~/Amharic_XTTS-V2_TTS"),
        "/root/Amharic_XTTS-V2_TTS"
    ]
    
    base_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            base_dir = dir_path
            print(f"\n✅ Found project directory: {base_dir}")
            break
    
    if not base_dir:
        print("\n❌ Could not auto-detect project directory.")
        print("Please run the full script with explicit paths:")
        print("\npython preprocess_amharic_dataset.py \\")
        print("  --input-train /path/to/train_filtered.csv \\")
        print("  --input-eval /path/to/eval_filtered.csv")
        return 1
    
    # Find CSV files
    print("\n🔍 Searching for CSV files...")
    train_files, eval_files = find_csv_files(base_dir)
    
    if not train_files or not eval_files:
        print("\n❌ Could not find training or evaluation CSV files.")
        print(f"Searched in: {base_dir}/dataset/")
        print("\nMake sure you have files like:")
        print("  - metadata_train_filtered.csv")
        print("  - metadata_eval_filtered.csv")
        return 1
    
    # Use the first match
    input_train = str(train_files[0])
    input_eval = str(eval_files[0])
    
    print(f"\n📁 Found files:")
    print(f"   Training:   {input_train}")
    print(f"   Evaluation: {input_eval}")
    
    # Set output paths
    output_train = str(Path(input_train).parent / f"{Path(input_train).stem}_preprocessed.csv")
    output_eval = str(Path(input_eval).parent / f"{Path(input_eval).stem}_preprocessed.csv")
    
    # Create tokenizer
    print("\n📦 Loading Amharic tokenizer with G2P...")
    try:
        tokenizer = XTTSAmharicTokenizer(
            vocab_file=None,  # Will use default
            use_phonemes=True
        )
        print("   ✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load tokenizer: {e}")
        return 1
    
    # Process files
    print("\n" + "=" * 80)
    print("Processing Training Data")
    print("=" * 80)
    train_processed, train_failed = preprocess_csv(input_train, output_train, tokenizer)
    
    print("\n" + "=" * 80)
    print("Processing Evaluation Data")
    print("=" * 80)
    eval_processed, eval_failed = preprocess_csv(input_eval, output_eval, tokenizer)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Summary")
    print("=" * 80)
    print(f"Training:   {train_processed} processed, {train_failed} failed")
    print(f"Evaluation: {eval_processed} processed, {eval_failed} failed")
    print(f"\n✅ Preprocessed files saved:")
    print(f"   📄 {output_train}")
    print(f"   📄 {output_eval}")
    print("\n💡 Next steps:")
    print("   1. In the WebUI, update the training CSV paths to:")
    print(f"      Train: {output_train}")
    print(f"      Eval:  {output_eval}")
    print("   2. Set language to 'en' (phonemes are now Latin characters)")
    print("   3. Start training!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
