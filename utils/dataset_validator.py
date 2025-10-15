#!/usr/bin/env python3
"""
Automatic Dataset Validator and Fixer

Runs before training to:
- Validate CSV format (3 columns, pipe-delimited)
- Fix common path issues (wavs/* -> finetune_models/dataset/wavs/*)
- Fix language codes ('speaker' -> 'am')
- Remove ALL rows with missing audio files (unlimited - handles 1, 10, 100, 1000+ missing files)
- Check text content quality
- Validate dataset size (stops if too small after removals)
- Create automatic backups before modifications
- Give clear feedback on issues found and fixed

IMPORTANT: This validator processes EVERY row and removes ALL rows with missing
audio files automatically. There is NO LIMIT on how many rows can be removed.
Only logging output is limited to first few samples for readability.
"""

import csv
import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetValidationError(Exception):
    """Raised when dataset validation fails with unfixable issues"""
    pass


class DatasetValidator:
    """
    Automatic dataset validator and fixer for XTTS training
    """
    
    def __init__(self, auto_fix: bool = True, verbose: bool = True):
        """
        Initialize validator
        
        Args:
            auto_fix: Automatically fix common issues
            verbose: Print detailed progress
        """
        self.auto_fix = auto_fix
        self.verbose = verbose
        self.issues_found = []
        self.issues_fixed = []
        self.warnings = []
    
    def validate_and_fix(
        self,
        train_csv: str,
        eval_csv: str,
        expected_language: str = "am"
    ) -> Tuple[bool, Dict]:
        """
        Validate and optionally fix dataset CSV files
        
        Args:
            train_csv: Path to training CSV
            eval_csv: Path to validation CSV
            expected_language: Expected language code
            
        Returns:
            Tuple of (is_valid, report_dict)
            
        Raises:
            DatasetValidationError: If critical unfixable issues found
        """
        self.issues_found = []
        self.issues_fixed = []
        self.warnings = []
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("🔍 AUTOMATIC DATASET VALIDATION")
            print("=" * 70)
            print()
        
        # Validate both CSV files
        all_valid = True
        
        for csv_path, csv_type in [(train_csv, "TRAIN"), (eval_csv, "VALIDATION")]:
            if self.verbose:
                print(f"Checking {csv_type} dataset: {csv_path}")
                print("-" * 70)
            
            if not os.path.exists(csv_path):
                raise DatasetValidationError(f"CSV file not found: {csv_path}")
            
            # Run validation checks
            valid = self._validate_csv(csv_path, csv_type, expected_language)
            all_valid = all_valid and valid
            
            if self.verbose:
                print()
        
        # Print summary
        if self.verbose:
            self._print_summary()
        
        # Create report
        report = {
            "valid": all_valid,
            "issues_found": len(self.issues_found),
            "issues_fixed": len(self.issues_fixed),
            "warnings": len(self.warnings),
            "details": {
                "issues": self.issues_found,
                "fixed": self.issues_fixed,
                "warnings": self.warnings
            }
        }
        
        return all_valid, report
    
    def _validate_csv(
        self,
        csv_path: str,
        csv_type: str,
        expected_language: str
    ) -> bool:
        """
        Validate and fix a single CSV file
        
        Returns:
            True if valid (or fixed), False if unfixable issues
        """
        # Check 1: CSV format
        format_ok = self._check_csv_format(csv_path, csv_type)
        
        # Check 2: Audio paths
        paths_ok = self._check_audio_paths(csv_path, csv_type)
        
        # Check 3: Language codes
        lang_ok = self._check_language_codes(csv_path, csv_type, expected_language)
        
        # Check 4: Text content
        text_ok = self._check_text_content(csv_path, csv_type)
        
        # Check 5: Dataset quality
        quality_ok = self._check_dataset_quality(csv_path, csv_type)
        
        return format_ok and paths_ok and lang_ok and text_ok and quality_ok
    
    def _check_csv_format(self, csv_path: str, csv_type: str) -> bool:
        """Check CSV has correct format (3 columns, pipe-delimited)"""
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                
                for row_num, row in enumerate(reader, 1):
                    if len(row) != 3:
                        issue = f"{csv_type} row {row_num}: Wrong column count ({len(row)}, expected 3)"
                        self.issues_found.append(issue)
                        
                        if not self.auto_fix:
                            logger.error(f"❌ {issue}")
                            return False
                        
                        # Can't auto-fix column count issues
                        logger.error(f"❌ {issue} - Cannot auto-fix")
                        return False
            
            return True
            
        except Exception as e:
            issue = f"{csv_type}: Failed to read CSV - {e}"
            self.issues_found.append(issue)
            logger.error(f"❌ {issue}")
            return False
    
    def _check_audio_paths(self, csv_path: str, csv_type: str) -> bool:
        """
        Check and fix audio file paths
        
        IMPORTANT: This method processes ALL rows and will:
        - Fix any fixable path issues (unlimited)
        - Remove ALL rows with missing audio files (unlimited)
        - Ensure paths are RELATIVE to CSV directory (TTS formatter requirement)
        - No limits on how many rows can be removed
        
        Only the logging is limited to first 3 samples for readability.
        """
        try:
            # Get CSV directory for making paths relative
            csv_dir = os.path.dirname(os.path.abspath(csv_path))
            
            # Read CSV
            rows = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                rows = list(reader)
            
            fixed_rows = []
            missing_files = 0
            fixed_paths = 0
            removed_rows = 0
            
            # Process EVERY row - no limits on how many can be fixed/removed
            for row_num, row in enumerate(rows, 1):
                if len(row) != 3:
                    fixed_rows.append(row)
                    continue
                
                audio_path, text, lang = row
                original_path = audio_path
                
                # Convert to absolute path for checking
                if not os.path.isabs(audio_path):
                    abs_audio_path = os.path.join(csv_dir, audio_path)
                else:
                    abs_audio_path = audio_path
                
                # Check if file exists
                if not os.path.exists(abs_audio_path):
                    missing_files += 1
                    
                    # Try to fix path
                    fixed_abs_path = self._try_fix_audio_path(abs_audio_path)
                    
                    if fixed_abs_path and os.path.exists(fixed_abs_path):
                        # Convert fixed absolute path to relative path (relative to CSV dir)
                        try:
                            relative_path = os.path.relpath(fixed_abs_path, csv_dir)
                            fixed_paths += 1
                            
                            if self.verbose and fixed_paths <= 3:
                                logger.info(f"  ✅ Fixed path: {original_path} → {relative_path}")
                            
                            # Add the fixed row with RELATIVE path
                            fixed_rows.append([relative_path, text, lang])
                        except ValueError:
                            # Can't make relative path (different drives on Windows?)
                            # Use absolute path as fallback
                            fixed_rows.append([fixed_abs_path, text, lang])
                    else:
                        # Path couldn't be fixed - REMOVE this row if auto_fix enabled
                        if self.auto_fix:
                            removed_rows += 1
                            if removed_rows <= 3:  # Only log first few
                                logger.warning(f"  🗑️  Removing row with missing file: {audio_path}")
                            # Don't append this row - effectively removing it
                        else:
                            # Not auto-fixing, keep the row but warn
                            if missing_files <= 3:
                                logger.warning(f"  ⚠️  Audio file not found: {audio_path}")
                            fixed_rows.append([audio_path, text, lang])
                else:
                    # File exists - convert to relative path and keep the row
                    try:
                        relative_path = os.path.relpath(abs_audio_path, csv_dir)
                        fixed_rows.append([relative_path, text, lang])
                    except ValueError:
                        # Can't make relative path, keep as-is
                        fixed_rows.append([audio_path, text, lang])
            
            # Write fixes if needed
            if fixed_paths > 0 or removed_rows > 0:
                if self.auto_fix:
                    self._backup_and_write_csv(csv_path, fixed_rows)
                    
                    if fixed_paths > 0:
                        fix_msg = f"{csv_type}: Fixed {fixed_paths} audio paths"
                        self.issues_fixed.append(fix_msg)
                        logger.info(f"✅ {fix_msg}")
                    
                    if removed_rows > 0:
                        fix_msg = f"{csv_type}: Removed {removed_rows} rows with missing audio files (ALL missing files cleaned)"
                        self.issues_fixed.append(fix_msg)
                        logger.info(f"✅ {fix_msg}")
                        
                        # Update dataset size info
                        original_count = len(rows)
                        new_count = len(fixed_rows)
                        logger.info(f"  ℹ️  Dataset size: {original_count} → {new_count} samples")
                        
                        # Show percentage removed if significant
                        removal_pct = (removed_rows / original_count * 100) if original_count > 0 else 0
                        if removal_pct > 10:
                            logger.warning(f"  ⚠️  Removed {removal_pct:.1f}% of dataset - consider investigating missing files")
                else:
                    if fixed_paths > 0:
                        issue = f"{csv_type}: {fixed_paths} audio paths need fixing"
                        self.issues_found.append(issue)
                        logger.warning(f"⚠️  {issue}")
                    
                    if removed_rows > 0:
                        issue = f"{csv_type}: {removed_rows} rows with missing files"
                        self.issues_found.append(issue)
                        logger.warning(f"⚠️  {issue}")
            
            # Report remaining missing files (if auto_fix disabled)
            if not self.auto_fix and missing_files > 0:
                warning = f"{csv_type}: {missing_files} audio files not found"
                self.warnings.append(warning)
                logger.warning(f"⚠️  {warning}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking audio paths: {e}")
            return False
    
    def _try_fix_audio_path(self, audio_path: str) -> Optional[str]:
        """
        Try to fix common audio path issues
        
        Returns:
            Fixed path if found, None otherwise
        """
        # Common fix 1: Add finetune_models/dataset/ prefix
        if audio_path.startswith('wavs/'):
            test_path = audio_path.replace('wavs/', 'finetune_models/dataset/wavs/', 1)
            if os.path.exists(test_path):
                return test_path
        
        # Common fix 2: Try dataset/ prefix
        if audio_path.startswith('wavs/'):
            test_path = audio_path.replace('wavs/', 'dataset/wavs/', 1)
            if os.path.exists(test_path):
                return test_path
        
        # Common fix 3: Try relative to common base paths
        basename = os.path.basename(audio_path)
        for prefix in ['finetune_models/dataset/wavs', 'dataset/wavs', 'wavs']:
            test_path = os.path.join(prefix, basename)
            if os.path.exists(test_path):
                return test_path
        
        return None
    
    def _check_language_codes(
        self,
        csv_path: str,
        csv_type: str,
        expected_language: str
    ) -> bool:
        """Check and fix language codes"""
        try:
            # Read CSV
            rows = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                rows = list(reader)
            
            fixed_rows = []
            wrong_lang_codes = 0
            
            for row_num, row in enumerate(rows, 1):
                if len(row) != 3:
                    fixed_rows.append(row)
                    continue
                
                audio_path, text, lang = row
                
                # Fix common wrong language codes
                if lang.strip().lower() in ['speaker', 'unknown', '']:
                    wrong_lang_codes += 1
                    lang = expected_language
                
                fixed_rows.append([audio_path, text, lang])
            
            # Write fixes if needed
            if wrong_lang_codes > 0:
                if self.auto_fix:
                    self._backup_and_write_csv(csv_path, fixed_rows)
                    fix_msg = f"{csv_type}: Fixed {wrong_lang_codes} language codes to '{expected_language}'"
                    self.issues_fixed.append(fix_msg)
                    logger.info(f"✅ {fix_msg}")
                else:
                    issue = f"{csv_type}: {wrong_lang_codes} incorrect language codes"
                    self.issues_found.append(issue)
                    logger.warning(f"⚠️  {issue}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking language codes: {e}")
            return False
    
    def _check_text_content(self, csv_path: str, csv_type: str) -> bool:
        """Check text content quality"""
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                
                empty_text = 0
                very_short = 0
                
                for row_num, row in enumerate(reader, 1):
                    if len(row) != 3:
                        continue
                    
                    text = row[1].strip()
                    
                    if not text:
                        empty_text += 1
                    elif len(text) < 3:
                        very_short += 1
                
                if empty_text > 0:
                    warning = f"{csv_type}: {empty_text} samples with empty text"
                    self.warnings.append(warning)
                    logger.warning(f"⚠️  {warning}")
                
                if very_short > 0:
                    warning = f"{csv_type}: {very_short} samples with very short text (<3 chars)"
                    self.warnings.append(warning)
                    logger.warning(f"⚠️  {warning}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking text content: {e}")
            return False
    
    def _check_dataset_quality(self, csv_path: str, csv_type: str) -> bool:
        """Check overall dataset quality metrics"""
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                rows = [row for row in reader if len(row) == 3]
            
            if len(rows) == 0:
                issue = f"{csv_type}: No valid rows found after validation"
                self.issues_found.append(issue)
                logger.error(f"❌ {issue}")
                return False
            
            # Check minimum dataset size
            if csv_type == "TRAIN":
                if len(rows) < 20:
                    issue = f"{csv_type}: Dataset too small ({len(rows)} samples). Need at least 20 samples for training."
                    self.issues_found.append(issue)
                    logger.error(f"❌ {issue}")
                    return False
                elif len(rows) < 50:
                    warning = f"{csv_type}: Small dataset ({len(rows)} samples). Recommend at least 50 samples for good results."
                    self.warnings.append(warning)
                    logger.warning(f"⚠️  {warning}")
            
            if csv_type == "VALIDATION":
                if len(rows) < 3:
                    issue = f"{csv_type}: Validation set too small ({len(rows)} samples). Need at least 3 samples."
                    self.issues_found.append(issue)
                    logger.error(f"❌ {issue}")
                    return False
                elif len(rows) < 5:
                    warning = f"{csv_type}: Small validation set ({len(rows)} samples). Recommend at least 5 samples."
                    self.warnings.append(warning)
                    logger.warning(f"⚠️  {warning}")
            
            if self.verbose:
                logger.info(f"  ℹ️  {csv_type}: {len(rows)} valid samples")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking dataset quality: {e}")
            return False
    
    def _backup_and_write_csv(self, csv_path: str, rows: List[List[str]]):
        """Backup original CSV and write fixed version"""
        # Create backup
        backup_path = f"{csv_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(csv_path, backup_path)
        
        # Write fixed CSV
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerows(rows)
    
    def _print_summary(self):
        """Print validation summary"""
        print("=" * 70)
        print("📊 VALIDATION SUMMARY")
        print("=" * 70)
        
        if not self.issues_found and not self.warnings:
            print("✅ No issues found - dataset is valid!")
        else:
            if self.issues_found:
                print(f"\n⚠️  Issues found: {len(self.issues_found)}")
                for issue in self.issues_found[:5]:  # Show first 5
                    print(f"   - {issue}")
                if len(self.issues_found) > 5:
                    print(f"   ... and {len(self.issues_found) - 5} more")
            
            if self.issues_fixed:
                print(f"\n✅ Issues auto-fixed: {len(self.issues_fixed)}")
                for fix in self.issues_fixed:
                    print(f"   - {fix}")
            
            if self.warnings:
                print(f"\n⚠️  Warnings: {len(self.warnings)}")
                for warning in self.warnings[:5]:  # Show first 5
                    print(f"   - {warning}")
                if len(self.warnings) > 5:
                    print(f"   ... and {len(self.warnings) - 5} more")
        
        print("=" * 70)
        print()


def validate_dataset_before_training(
    train_csv: str,
    eval_csv: str,
    expected_language: str = "am",
    auto_fix: bool = True
) -> bool:
    """
    Convenience function to validate dataset before training
    
    Args:
        train_csv: Path to training CSV
        eval_csv: Path to validation CSV
        expected_language: Expected language code
        auto_fix: Automatically fix common issues
        
    Returns:
        True if validation passed (or issues fixed)
        
    Raises:
        DatasetValidationError: If critical unfixable issues found
    """
    validator = DatasetValidator(auto_fix=auto_fix, verbose=True)
    is_valid, report = validator.validate_and_fix(train_csv, eval_csv, expected_language)
    
    if not is_valid:
        raise DatasetValidationError(
            f"Dataset validation failed with {report['issues_found']} issues. "
            "Please fix these issues before training."
        )
    
    return True


# Testing
if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Dataset Validator - Standalone Test")
    print("=" * 70)
    print()
    
    # Test with actual CSV files
    train_csv = "finetune_models/dataset/metadata_train.csv"
    eval_csv = "finetune_models/dataset/metadata_eval.csv"
    
    try:
        validate_dataset_before_training(train_csv, eval_csv, expected_language="am")
        print("✅ Validation passed!")
    except DatasetValidationError as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)
