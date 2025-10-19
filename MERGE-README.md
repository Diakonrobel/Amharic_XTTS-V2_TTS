# Dataset Merger for XTTS Fine-tuning

Merges multiple `temp_dataset_*` folders into the main `dataset` folder on Lightning AI.

## Files

- **`merge_datasets.py`** - Python script that performs the merge (runs on remote)
- **`run-merge.ps1`** - PowerShell wrapper to upload and execute (runs locally)

## Quick Start

```powershell
.\run-merge.ps1
```

This will:
1. Upload `merge_datasets.py` to Lightning AI
2. Execute the merge interactively
3. Let you download the report when done

## What It Does

### 1. Discovery
- Finds all `temp_dataset_*` folders in `/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/finetune_models`
- Counts audio files in each
- Validates structure (checks for required files)

### 2. Backup
- Creates timestamped backups of:
  - `dataset/metadata_train.csv`
  - `dataset/metadata_eval.csv`

### 3. Merge
- **CSV Files**: Merges all metadata CSVs, removing duplicates
- **Audio Files**: Copies all wav files from temp datasets to main dataset
  - Skips identical files (same size)
  - Renames conflicting files

### 4. Report
- Generates JSON report with:
  - Timestamp
  - List of merged datasets
  - Statistics (rows added, files copied, etc.)

### 5. Cleanup Options
After merge, choose:
1. **Keep** temp datasets (do nothing)
2. **Delete** temp datasets (permanent)
3. **Move** temp datasets to `merged_temps` folder (recommended)

## Structure Expected

Each dataset should have:
```
temp_dataset_X/
├── lang.txt
├── metadata_train.csv
├── metadata_eval.csv
└── wavs/
    ├── audio_001.wav
    ├── audio_002.wav
    └── ...
```

## Safety Features

✅ **Backups** created before any changes  
✅ **Duplicate detection** in CSV metadata  
✅ **Conflict resolution** for audio files  
✅ **Validation** of dataset structure  
✅ **Confirmation prompts** before destructive operations  
✅ **Detailed report** saved to disk  

## Manual Execution

If you prefer to run directly on Lightning AI:

```bash
# SSH into Lightning AI
ssh s_01k7x54qcrv1atww40z8bxf9a3@ssh.lightning.ai

# Navigate to directory
cd /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/finetune_models

# Upload script (from local machine)
scp merge_datasets.py s_01k7x54qcrv1atww40z8bxf9a3@ssh.lightning.ai:/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/finetune_models/

# Run script (on remote)
python3 merge_datasets.py
```

## Current State

Based on remote directory listing:

**Found temp datasets:**
- temp_dataset_1
- temp_dataset_2
- temp_dataset_4
- temp_dataset_5
- temp_dataset_6
- temp_dataset_7
- temp_dataset_8
- temp_dataset_9
- temp_dataset_10
- temp_dataset_12
- temp_dataset_13
- temp_dataset_14
- temp_dataset_15

**Total: 13 temp datasets to merge**

## Example Output

```
============================================================
  Dataset Merger for XTTS Fine-tuning
============================================================

ℹ Found 13 temp datasets:
  • temp_dataset_1: 45 wav files
  • temp_dataset_2: 52 wav files
  ...

⚠ About to merge 13 temp datasets into main dataset
Continue? (yes/no): yes

============================================================
  Creating Backups
============================================================

✓ Backup created: metadata_train.csv.backup_20251019_120530
✓ Backup created: metadata_eval.csv.backup_20251019_120530

============================================================
  Merging Metadata Files
============================================================

ℹ Merging 14 CSV files...
ℹ Main CSV has 2500 rows
ℹ   temp_dataset_1: 45 rows
ℹ   temp_dataset_2: 52 rows
...
✓ Merged CSV saved: 3147 total rows

============================================================
  Merge Summary
============================================================

Train metadata: 3147 rows (added 647 from temp)
Eval metadata:  785 rows (added 162 from temp)
Wav files:      3147 total
  - Copied:     809
  - Skipped:    0
  - Before:     2500

✓ Report saved: merge_report_20251019_120534.json

Cleanup Options:
1. Keep temp datasets
2. Delete temp datasets (PERMANENT)
3. Move temp datasets to 'merged_temps' folder

Select option (1-3): 3

✓ Merge completed successfully!
```

## Troubleshooting

**Script fails to upload:**
- Check SSH connection: `ssh s_01k7x54qcrv1atww40z8bxf9a3@ssh.lightning.ai`
- Verify Lightning AI session is active

**"No temp datasets found":**
- Check directory path in script matches your setup
- List remote directory: `.\lightning-ssh-manager.ps1 "ls -la finetune_models"`

**CSV merge errors:**
- Ensure pandas is installed on remote: `pip install pandas`
- Check CSV format (pipe-delimited, has 'audio_file' column)

**Permission denied:**
- Check file permissions on remote
- Ensure you have write access to `dataset/` folder

## Tips

💡 **Test first**: Run on a copy of your data to verify behavior  
💡 **Keep backups**: Option 3 (move) is safer than deletion  
💡 **Check reports**: Review JSON reports to verify merge results  
💡 **Incremental merge**: You can merge batches of temp datasets separately  
