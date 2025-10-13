"""
Incremental Dataset Merger Module
Adds newly processed datasets to existing datasets without overwriting.
Includes duplicate detection, validation, and conflict resolution.
"""

import os
import hashlib
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
import pandas as pd
from datetime import datetime


class IncrementalDatasetMerger:
    """Merge new datasets into existing ones with validation and deduplication."""
    
    def __init__(self, base_dataset_path: str):
        """
        Initialize the incremental merger.
        
        Args:
            base_dataset_path: Path to the base dataset directory
        """
        self.base_path = Path(base_dataset_path)
        self.wavs_dir = self.base_path / "wavs"
        self.train_csv = self.base_path / "metadata_train.csv"
        self.eval_csv = self.base_path / "metadata_eval.csv"
        self.lang_file = self.base_path / "lang.txt"
        
        # Statistics
        self.stats = {
            'added_train': 0,
            'added_eval': 0,
            'duplicates_skipped': 0,
            'errors': 0
        }
    
    def dataset_exists(self) -> bool:
        """Check if base dataset exists."""
        return (self.train_csv.exists() and 
                self.eval_csv.exists() and 
                self.wavs_dir.exists())
    
    def _compute_audio_hash(self, audio_path: Path) -> str:
        """
        Compute hash of audio file for duplicate detection.
        Only hashes first and last 8KB for speed.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            SHA256 hash (first 16 chars)
        """
        try:
            hasher = hashlib.sha256()
            file_size = audio_path.stat().st_size
            chunk_size = 8192
            
            with open(audio_path, 'rb') as f:
                # Hash first chunk
                chunk = f.read(chunk_size)
                hasher.update(chunk)
                
                # Hash last chunk if file is large enough
                if file_size > chunk_size * 2:
                    f.seek(-chunk_size, 2)
                    chunk = f.read(chunk_size)
                    hasher.update(chunk)
            
            return hasher.hexdigest()[:16]
        except Exception as e:
            print(f"  Warning: Could not hash audio file {audio_path.name}: {e}")
            return ""
    
    def _get_existing_audio_hashes(self) -> Set[str]:
        """
        Get hashes of all existing audio files in base dataset.
        
        Returns:
            Set of audio file hashes
        """
        existing_hashes = set()
        
        if not self.wavs_dir.exists():
            return existing_hashes
        
        print("  Computing hashes of existing audio files...")
        for audio_file in self.wavs_dir.glob("*.wav"):
            audio_hash = self._compute_audio_hash(audio_file)
            if audio_hash:
                existing_hashes.add(audio_hash)
        
        print(f"  Found {len(existing_hashes)} existing audio files")
        return existing_hashes
    
    def _get_next_audio_counter(self) -> int:
        """
        Get the next available audio file counter.
        
        Returns:
            Next available counter number
        """
        if not self.wavs_dir.exists():
            return 0
        
        # Find highest number in existing filenames
        max_num = -1
        for audio_file in self.wavs_dir.glob("*.wav"):
            try:
                # Extract number from filename (assuming format: prefix_NNNNNNNN.wav)
                stem = audio_file.stem
                # Try different patterns
                if '_' in stem:
                    num_str = stem.split('_')[-1]
                else:
                    # Try to extract trailing digits
                    import re
                    match = re.search(r'(\d+)$', stem)
                    if match:
                        num_str = match.group(1)
                    else:
                        continue
                
                num = int(num_str)
                if num > max_num:
                    max_num = num
            except (ValueError, IndexError):
                continue
        
        return max_num + 1
    
    def _validate_language_compatibility(self, new_dataset_path: Path) -> Tuple[bool, str]:
        """
        Validate that new dataset has compatible language.
        
        Args:
            new_dataset_path: Path to new dataset
            
        Returns:
            Tuple of (is_valid, message)
        """
        new_lang_file = new_dataset_path / "lang.txt"
        
        # If base dataset doesn't have lang.txt yet, any language is OK
        if not self.lang_file.exists():
            return True, "Base dataset has no language set yet"
        
        # If new dataset has no lang.txt, warn but allow
        if not new_lang_file.exists():
            return True, "Warning: New dataset has no language file"
        
        # Compare languages
        try:
            with open(self.lang_file, 'r', encoding='utf-8') as f:
                base_lang = f.read().strip()
            
            with open(new_lang_file, 'r', encoding='utf-8') as f:
                new_lang = f.read().strip()
            
            if base_lang != new_lang:
                return False, f"Language mismatch: base='{base_lang}', new='{new_lang}'"
            
            return True, f"Language compatible: '{base_lang}'"
        
        except Exception as e:
            return False, f"Error reading language files: {e}"
    
    def merge_new_dataset(
        self,
        new_dataset_path: str,
        check_duplicates: bool = True,
        keep_source: bool = False
    ) -> Dict[str, any]:
        """
        Merge a new dataset into the base dataset incrementally.
        
        Args:
            new_dataset_path: Path to new dataset directory
            check_duplicates: Whether to skip duplicate audio files
            keep_source: Whether to keep source dataset after merging
            
        Returns:
            Dictionary with merge statistics and results
        """
        new_path = Path(new_dataset_path)
        
        if not new_path.exists():
            return {
                'success': False,
                'error': f"New dataset path does not exist: {new_dataset_path}"
            }
        
        # Validate structure
        new_train_csv = new_path / "metadata_train.csv"
        new_eval_csv = new_path / "metadata_eval.csv"
        new_wavs_dir = new_path / "wavs"
        
        if not all([new_train_csv.exists(), new_eval_csv.exists(), new_wavs_dir.exists()]):
            return {
                'success': False,
                'error': "New dataset is missing required files (metadata_train.csv, metadata_eval.csv, or wavs directory)"
            }
        
        print(f"\n📦 Merging new dataset: {new_dataset_path}")
        
        # Validate language compatibility
        lang_valid, lang_msg = self._validate_language_compatibility(new_path)
        print(f"  {lang_msg}")
        if not lang_valid:
            return {
                'success': False,
                'error': lang_msg
            }
        
        # Create base dataset if it doesn't exist
        if not self.dataset_exists():
            print("  Base dataset doesn't exist, creating new one...")
            self.base_path.mkdir(parents=True, exist_ok=True)
            self.wavs_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy language file if available
            new_lang_file = new_path / "lang.txt"
            if new_lang_file.exists() and not self.lang_file.exists():
                shutil.copy2(new_lang_file, self.lang_file)
        
        # Load existing metadata (or create empty DataFrames)
        if self.train_csv.exists():
            base_train_df = pd.read_csv(self.train_csv, sep='|')
            base_eval_df = pd.read_csv(self.eval_csv, sep='|')
            print(f"  Base dataset: {len(base_train_df)} train, {len(base_eval_df)} eval segments")
        else:
            base_train_df = pd.DataFrame(columns=['audio_file', 'text', 'speaker_name'])
            base_eval_df = pd.DataFrame(columns=['audio_file', 'text', 'speaker_name'])
            print("  Base dataset is empty")
        
        # Load new metadata
        new_train_df = pd.read_csv(new_train_csv, sep='|')
        new_eval_df = pd.read_csv(new_eval_csv, sep='|')
        print(f"  New dataset: {len(new_train_df)} train, {len(new_eval_df)} eval segments")
        
        # Get existing audio hashes for duplicate detection
        existing_hashes = self._get_existing_audio_hashes() if check_duplicates else set()
        
        # Get next audio counter
        audio_counter = self._get_next_audio_counter()
        print(f"  Starting audio counter at: {audio_counter}")
        
        # Reset stats
        self.stats = {
            'added_train': 0,
            'added_eval': 0,
            'duplicates_skipped': 0,
            'errors': 0
        }
        
        # Process train data
        new_train_rows = []
        for _, row in new_train_df.iterrows():
            success, new_row = self._process_row(
                row=row,
                new_dataset_path=new_path,
                audio_counter=audio_counter,
                existing_hashes=existing_hashes,
                check_duplicates=check_duplicates
            )
            
            if success:
                new_train_rows.append(new_row)
                audio_counter += 1
                self.stats['added_train'] += 1
        
        # Process eval data
        new_eval_rows = []
        for _, row in new_eval_df.iterrows():
            success, new_row = self._process_row(
                row=row,
                new_dataset_path=new_path,
                audio_counter=audio_counter,
                existing_hashes=existing_hashes,
                check_duplicates=check_duplicates
            )
            
            if success:
                new_eval_rows.append(new_row)
                audio_counter += 1
                self.stats['added_eval'] += 1
        
        # Merge DataFrames
        if new_train_rows:
            merged_train_df = pd.concat([base_train_df, pd.DataFrame(new_train_rows)], ignore_index=True)
        else:
            merged_train_df = base_train_df
        
        if new_eval_rows:
            merged_eval_df = pd.concat([base_eval_df, pd.DataFrame(new_eval_rows)], ignore_index=True)
        else:
            merged_eval_df = base_eval_df
        
        # Save merged metadata
        merged_train_df.to_csv(self.train_csv, sep='|', index=False)
        merged_eval_df.to_csv(self.eval_csv, sep='|', index=False)
        
        # Update language file if needed
        new_lang_file = new_path / "lang.txt"
        if new_lang_file.exists() and not self.lang_file.exists():
            shutil.copy2(new_lang_file, self.lang_file)
        
        # Remove source dataset if requested
        if not keep_source:
            try:
                shutil.rmtree(new_path, ignore_errors=True)
                print(f"  Removed source dataset: {new_path}")
            except Exception as e:
                print(f"  Warning: Could not remove source dataset: {e}")
        
        # Print summary
        print(f"\n✅ Merge Complete!")
        print(f"  Added {self.stats['added_train']} train segments")
        print(f"  Added {self.stats['added_eval']} eval segments")
        if check_duplicates:
            print(f"  Skipped {self.stats['duplicates_skipped']} duplicates")
        if self.stats['errors'] > 0:
            print(f"  Errors: {self.stats['errors']}")
        print(f"  Total segments: {len(merged_train_df) + len(merged_eval_df)}")
        
        return {
            'success': True,
            'stats': self.stats,
            'total_train': len(merged_train_df),
            'total_eval': len(merged_eval_df),
            'total_segments': len(merged_train_df) + len(merged_eval_df)
        }
    
    def _process_row(
        self,
        row: pd.Series,
        new_dataset_path: Path,
        audio_counter: int,
        existing_hashes: Set[str],
        check_duplicates: bool
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Process a single row: copy audio, check for duplicates, update paths.
        
        Args:
            row: DataFrame row to process
            new_dataset_path: Path to new dataset
            audio_counter: Current audio counter
            existing_hashes: Set of existing audio hashes
            check_duplicates: Whether to check for duplicates
            
        Returns:
            Tuple of (success, new_row_dict or None)
        """
        try:
            # Get old audio path
            old_audio_path = new_dataset_path / row['audio_file']
            
            if not old_audio_path.exists():
                print(f"  Warning: Audio file not found: {old_audio_path}")
                self.stats['errors'] += 1
                return False, None
            
            # Check for duplicate
            if check_duplicates:
                audio_hash = self._compute_audio_hash(old_audio_path)
                if audio_hash in existing_hashes:
                    self.stats['duplicates_skipped'] += 1
                    return False, None
                # Add to existing hashes to prevent duplicates within same merge
                existing_hashes.add(audio_hash)
            
            # Generate new filename
            new_filename = f"merged_{str(audio_counter).zfill(8)}.wav"
            new_audio_path = self.wavs_dir / new_filename
            
            # Copy audio file
            shutil.copy2(old_audio_path, new_audio_path)
            
            # Create new row with updated path
            new_row = row.to_dict()
            new_row['audio_file'] = f"wavs/{new_filename}"
            
            return True, new_row
        
        except Exception as e:
            print(f"  Error processing row: {e}")
            self.stats['errors'] += 1
            return False, None


def merge_datasets_incremental(
    new_dataset_paths: List[str],
    base_dataset_path: str,
    check_duplicates: bool = True,
    keep_sources: bool = False
) -> Tuple[str, str, int, Dict]:
    """
    Merge multiple new datasets into a base dataset incrementally.
    
    Args:
        new_dataset_paths: List of new dataset directories to merge
        base_dataset_path: Path to base dataset
        check_duplicates: Whether to skip duplicate audio files
        keep_sources: Whether to keep source datasets after merging
        
    Returns:
        Tuple of (train_csv_path, eval_csv_path, total_segments, stats_dict)
    """
    merger = IncrementalDatasetMerger(base_dataset_path)
    
    print(f"\n🔄 Incremental Merge: Adding {len(new_dataset_paths)} dataset(s) to base")
    print(f"   Base: {base_dataset_path}")
    
    total_added_train = 0
    total_added_eval = 0
    total_duplicates = 0
    total_errors = 0
    
    for idx, new_dataset_path in enumerate(new_dataset_paths, 1):
        print(f"\n[{idx}/{len(new_dataset_paths)}] Processing: {new_dataset_path}")
        
        result = merger.merge_new_dataset(
            new_dataset_path=new_dataset_path,
            check_duplicates=check_duplicates,
            keep_source=keep_sources
        )
        
        if not result['success']:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
            total_errors += 1
            continue
        
        stats = result['stats']
        total_added_train += stats['added_train']
        total_added_eval += stats['added_eval']
        total_duplicates += stats['duplicates_skipped']
        total_errors += stats['errors']
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"✅ Incremental Merge Complete!")
    print(f"   Datasets processed: {len(new_dataset_paths)}")
    print(f"   Total train segments added: {total_added_train}")
    print(f"   Total eval segments added: {total_added_eval}")
    if check_duplicates:
        print(f"   Duplicates skipped: {total_duplicates}")
    if total_errors > 0:
        print(f"   Errors encountered: {total_errors}")
    
    # Get final counts
    base_path = Path(base_dataset_path)
    train_csv = base_path / "metadata_train.csv"
    eval_csv = base_path / "metadata_eval.csv"
    
    if train_csv.exists() and eval_csv.exists():
        train_df = pd.read_csv(train_csv, sep='|')
        eval_df = pd.read_csv(eval_csv, sep='|')
        total_segments = len(train_df) + len(eval_df)
        print(f"   Final total segments: {total_segments}")
    else:
        total_segments = 0
    
    summary_stats = {
        'added_train': total_added_train,
        'added_eval': total_added_eval,
        'duplicates_skipped': total_duplicates,
        'errors': total_errors,
        'total_segments': total_segments
    }
    
    return str(train_csv), str(eval_csv), total_segments, summary_stats
