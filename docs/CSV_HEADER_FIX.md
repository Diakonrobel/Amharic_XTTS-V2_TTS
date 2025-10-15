# CSV Header Fix for Coqui TTS Format

## Problem

When training XTTS models, the Coqui TTS library expects metadata CSV files to have a header row with specific column names. If the header is missing, training will fail with an AssertionError:

```
AssertionError: assert all(x in metadata[0] for x in ["audio_file", "text"])
```

This error occurs in `/TTS/tts/datasets/formatters.py` line 74 when the Coqui formatter tries to validate the CSV structure.

## Required Format

The CSV files must have a header row with pipe-separated column names:

```
audio_file|text|speaker_name
```

Followed by data rows:

```
audio_file|text|speaker_name
wavs/sample001.wav|Hello world|speaker
wavs/sample002.wav|This is a test|speaker
```

## Solution

### Automatic Fix

Use the provided utility script to automatically add headers to your CSV files:

```bash
# From project root
python utils/fix_csv_headers.py

# Or specify a custom dataset path
python utils/fix_csv_headers.py /path/to/your/dataset/
```

The script will:
1. Check if headers are already present
2. Create backup files (`.backup` extension) before making changes
3. Add the correct header row if missing
4. Verify the fix was successful

### Manual Fix

If you prefer to fix the files manually:

**On Linux/Mac:**
```bash
cd finetune_models/dataset
sed -i '1i audio_file|text|speaker_name' metadata_train.csv
sed -i '1i audio_file|text|speaker_name' metadata_eval.csv
```

**On Windows (PowerShell):**
```powershell
$header = "audio_file|text|speaker_name"
$trainContent = Get-Content metadata_train.csv
$evalContent = Get-Content metadata_eval.csv
Set-Content metadata_train.csv -Value $header, $trainContent
Set-Content metadata_eval.csv -Value $header, $evalContent
```

**Using Python:**
```python
header = 'audio_file|text|speaker_name\n'

for file in ['metadata_train.csv', 'metadata_eval.csv']:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(file, 'w', encoding='utf-8') as f:
        f.write(header + content)
```

## Verification

After applying the fix, verify the headers are correct:

```bash
# Check first line of each file
head -1 finetune_models/dataset/metadata_train.csv
head -1 finetune_models/dataset/metadata_eval.csv
```

Both should output:
```
audio_file|text|speaker_name
```

## Prevention

To prevent this issue in the future:

1. Always include headers when creating new metadata CSV files
2. Use the format: `audio_file|text|speaker_name`
3. Run the `fix_csv_headers.py` script as part of your dataset preparation workflow
4. Add dataset validation to your preprocessing pipeline

## Related Files

- **Fix Script**: `utils/fix_csv_headers.py`
- **Dataset Validator**: `utils/dataset_validator.py`
- **Format Documentation**: `DATASET_FORMAT.md`

## Technical Details

The Coqui TTS library uses the `coqui` formatter function which expects:
- CSV files with pipe (`|`) delimiters
- First row must be a dictionary-parseable header
- Required fields: `audio_file` and `text`
- Optional field: `speaker_name`

The formatter parses the first line into a dictionary structure and validates that required keys exist before processing the dataset.

## Troubleshooting

### Issue: Script reports "Dataset directory not found"
**Solution**: Create the directory or specify the correct path as an argument

### Issue: Permission denied when writing files
**Solution**: Ensure you have write permissions to the dataset directory

### Issue: Backup files already exist
**Solution**: The script won't overwrite existing backups. Delete `.backup` files if you want fresh backups

### Issue: Unicode errors when reading CSV
**Solution**: Ensure your CSV files are encoded in UTF-8

## See Also

- [Dataset Format Guide](DATASET_FORMAT.md)
- [Training Troubleshooting](TROUBLESHOOTING.md)
- [Coqui TTS Documentation](https://tts.readthedocs.io/)
