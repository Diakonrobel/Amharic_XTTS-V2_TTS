"""
Amharic Dataset Preprocessing Utility

This script preprocesses Amharic dataset CSV files by applying G2P conversion
to the text column, creating phoneme-based versions for improved TTS training.

Usage:
    python -m amharic_tts.utils.preprocess_dataset \\
        --input train.csv \\
        --output train_phonemes.csv \\
        --use-g2p

    # Or batch process
    python -m amharic_tts.utils.preprocess_dataset \\
        --input-dir data/ \\
        --output-dir data_phonemes/ \\
        --use-g2p
"""

import os
import sys
import csv
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from amharic_tts.g2p.amharic_g2p_enhanced import EnhancedAmharicG2P

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_csv(
    input_file: str,
    output_file: str,
    use_g2p: bool = True,
    text_column: str = "text",
    create_backup: bool = True
) -> int:
    """
    Preprocess a CSV file by applying G2P to text column
    
    Args:
        input_file: Input CSV file path
        output_file: Output CSV file path
        use_g2p: Whether to apply G2P conversion
        text_column: Name of column containing text
        create_backup: Whether to backup original file
        
    Returns:
        Number of rows processed
    """
    logger.info(f"Processing {input_file}...")
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return 0
    
    # Initialize G2P if needed
    g2p = None
    if use_g2p:
        logger.info("Initializing Amharic G2P converter...")
        g2p = EnhancedAmharicG2P()
        logger.info("G2P converter ready")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create backup if requested
    if create_backup and input_file != output_file:
        backup_file = input_file + ".backup"
        if not os.path.exists(backup_file):
            import shutil
            shutil.copy2(input_file, backup_file)
            logger.info(f"Created backup: {backup_file}")
    
    # Process CSV
    rows_processed = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \\
             open(output_file, 'w', encoding='utf-8', newline='') as f_out:
            
            reader = csv.DictReader(f_in)
            
            if not reader.fieldnames:
                logger.error("CSV file has no headers")
                return 0
            
            if text_column not in reader.fieldnames:
                logger.error(f"Text column '{text_column}' not found. Available: {reader.fieldnames}")
                return 0
            
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()
            
            for row in reader:
                original_text = row[text_column]
                
                if use_g2p and original_text.strip():
                    # Apply G2P conversion
                    try:
                        phonemes = g2p.convert(original_text)
                        row[text_column] = phonemes
                    except Exception as e:
                        logger.warning(f"G2P conversion failed for: {original_text[:50]}...")
                        logger.warning(f"Error: {e}")
                        # Keep original text on error
                
                writer.writerow(row)
                rows_processed += 1
                
                if rows_processed % 100 == 0:
                    logger.info(f"Processed {rows_processed} rows...")
        
        logger.info(f"✅ Successfully processed {rows_processed} rows")
        logger.info(f"Output saved to: {output_file}")
        return rows_processed
        
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        return 0


def preprocess_directory(
    input_dir: str,
    output_dir: str,
    use_g2p: bool = True,
    pattern: str = "*.csv"
) -> int:
    """
    Preprocess all CSV files in a directory
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        use_g2p: Whether to apply G2P
        pattern: File pattern to match
        
    Returns:
        Total number of files processed
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 0
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = list(input_path.glob(pattern))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {input_dir}")
        return 0
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    files_processed = 0
    
    for csv_file in csv_files:
        output_file = output_path / csv_file.name
        rows = preprocess_csv(
            str(csv_file),
            str(output_file),
            use_g2p=use_g2p
        )
        
        if rows > 0:
            files_processed += 1
    
    logger.info(f"\\n✅ Processed {files_processed}/{len(csv_files)} files")
    return files_processed


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Amharic dataset with G2P conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python -m amharic_tts.utils.preprocess_dataset \\
      --input train.csv --output train_phonemes.csv --use-g2p

  # Process directory
  python -m amharic_tts.utils.preprocess_dataset \\
      --input-dir data/ --output-dir data_phonemes/ --use-g2p

  # Without G2P (just copy/normalize)
  python -m amharic_tts.utils.preprocess_dataset \\
      --input train.csv --output train_clean.csv
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input CSV file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output CSV file'
    )
    
    parser.add_argument(
        '--input-dir', '-d',
        type=str,
        help='Input directory (for batch processing)'
    )
    
    parser.add_argument(
        '--output-dir', '-D',
        type=str,
        help='Output directory (for batch processing)'
    )
    
    parser.add_argument(
        '--use-g2p', '-g',
        action='store_true',
        default=True,
        help='Apply G2P conversion (default: True)'
    )
    
    parser.add_argument(
        '--no-g2p',
        action='store_true',
        help='Disable G2P conversion'
    )
    
    parser.add_argument(
        '--text-column', '-t',
        type=str,
        default='text',
        help='Name of text column (default: text)'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup of original files'
    )
    
    args = parser.parse_args()
    
    # Handle no-g2p flag
    use_g2p = args.use_g2p and not args.no_g2p
    
    # Validate arguments
    if args.input and args.input_dir:
        print("Error: Cannot specify both --input and --input-dir")
        return 1
    
    if not args.input and not args.input_dir:
        print("Error: Must specify either --input or --input-dir")
        parser.print_help()
        return 1
    
    print("=" * 70)
    print("🇪🇹  Amharic Dataset Preprocessing")
    print("=" * 70)
    print(f"G2P Mode: {'Enabled' if use_g2p else 'Disabled'}")
    print()
    
    # Process single file or directory
    if args.input:
        if not args.output:
            # Generate output filename
            base = os.path.splitext(args.input)[0]
            suffix = '_phonemes' if use_g2p else '_clean'
            args.output = f"{base}{suffix}.csv"
        
        rows = preprocess_csv(
            args.input,
            args.output,
            use_g2p=use_g2p,
            text_column=args.text_column,
            create_backup=not args.no_backup
        )
        
        return 0 if rows > 0 else 1
    
    elif args.input_dir:
        if not args.output_dir:
            suffix = '_phonemes' if use_g2p else '_clean'
            args.output_dir = args.input_dir + suffix
        
        files = preprocess_directory(
            args.input_dir,
            args.output_dir,
            use_g2p=use_g2p
        )
        
        return 0 if files > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
