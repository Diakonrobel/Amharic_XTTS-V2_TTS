# Incremental Dataset Addition

## Overview

The **Incremental Dataset Addition** feature allows you to add newly downloaded and processed datasets to your existing training dataset **without overwriting** what's already there. This is perfect for iteratively growing your dataset by adding more YouTube videos or audio files over time.

## Key Features

### ✅ **Add Without Overwriting**
- Newly processed data is **merged into** your existing dataset
- Original data remains intact
- Perfect for building large datasets incrementally

### 🔍 **Duplicate Detection**
- Automatically detects and skips duplicate audio files
- Uses intelligent hashing (first + last chunks) for fast comparison
- Prevents wasting storage and training time on duplicates

### 🛡️ **Language Validation**
- Validates that new datasets have the same language as the base dataset
- Prevents accidental mixing of different languages
- Ensures training consistency

### 📊 **Smart File Naming**
- Automatically assigns unique sequential filenames (e.g., `merged_00000123.wav`)
- Continues numbering from where the last dataset left off
- No filename conflicts

### 📈 **Statistics & Reporting**
- Detailed merge statistics (added segments, duplicates skipped, errors)
- Progress tracking for each batch
- Final summary with total counts

---

## How It Works

### Standard Mode vs Incremental Mode

#### **Standard Mode (Default)**
```
New Videos → Process → Create Fresh Dataset
```
- Creates a **new dataset** from scratch
- Previous dataset is replaced
- Good for: Starting fresh, testing, single-batch processing

#### **Incremental Mode (New!)**
```
New Videos → Process → Add to Existing Dataset
```
- **Adds** to existing dataset
- Previous data is preserved
- Good for: Growing datasets over time, continuous improvement

---

## Usage

### 1. Using Batch YouTube Processing (Incremental Mode)

When processing YouTube videos in batch mode, you can enable incremental mode:

```python
from utils import batch_processor, youtube_downloader, srt_processor

urls = [
    "https://www.youtube.com/watch?v=VIDEO1",
    "https://www.youtube.com/watch?v=VIDEO2",
    "https://www.youtube.com/watch?v=VIDEO3"
]

# Process and ADD to existing dataset
train_csv, eval_csv, video_infos = batch_processor.process_youtube_batch(
    urls=urls,
    transcript_lang="amh",
    out_path="./finetune_models",
    youtube_downloader=youtube_downloader,
    srt_processor=srt_processor,
    incremental=True,           # ← Enable incremental mode
    check_duplicates=True        # ← Skip duplicate audio
)
```

### 2. Using Direct Incremental Merger

You can also use the incremental merger directly:

```python
from utils.incremental_dataset_merger import IncrementalDatasetMerger

# Initialize merger with base dataset path
merger = IncrementalDatasetMerger(
    base_dataset_path="./finetune_models/dataset"
)

# Merge a new dataset
result = merger.merge_new_dataset(
    new_dataset_path="./new_processed_dataset",
    check_duplicates=True,
    keep_source=False  # Remove source after merge
)

print(f"Added {result['stats']['added_train']} train segments")
print(f"Added {result['stats']['added_eval']} eval segments")
print(f"Skipped {result['stats']['duplicates_skipped']} duplicates")
```

### 3. Merging Multiple Datasets

```python
from utils.incremental_dataset_merger import merge_datasets_incremental

# Merge multiple new datasets into base
train_csv, eval_csv, total, stats = merge_datasets_incremental(
    new_dataset_paths=[
        "./temp_dataset_1",
        "./temp_dataset_2",
        "./temp_dataset_3"
    ],
    base_dataset_path="./finetune_models/dataset",
    check_duplicates=True,
    keep_sources=False
)

print(f"Final dataset has {total} total segments")
```

---

## Best Practices

### 📝 **When to Use Incremental Mode**

✅ **Good Use Cases:**
- Building a large dataset over multiple sessions
- Adding new speaker data to existing dataset
- Continuous dataset improvement
- Working with limited processing time/resources

❌ **When NOT to Use:**
- Starting a fresh dataset
- Testing different preprocessing parameters
- Need to completely replace dataset
- Experimenting with different audio sources

### 🎯 **Workflow Recommendations**

#### **Initial Dataset Creation**
```bash
# Step 1: Create base dataset (Standard Mode)
python process_youtube.py --urls video1,video2,video3 --mode standard

# Result: finetune_models/dataset/ created
```

#### **Growing Your Dataset**
```bash
# Step 2: Add more videos (Incremental Mode)
python process_youtube.py --urls video4,video5,video6 --mode incremental

# Result: video4-6 ADDED to existing dataset

# Step 3: Add even more (Incremental Mode)
python process_youtube.py --urls video7,video8,video9 --mode incremental

# Result: video7-9 ADDED to existing dataset
```

### 🔍 **Duplicate Detection Tips**

- **Enable for production**: Always use `check_duplicates=True` when building real datasets
- **Disable for testing**: Set `check_duplicates=False` for faster iteration during testing
- **Hashing strategy**: Uses first + last 8KB chunks for speed (good balance)

### ⚠️ **Common Pitfalls to Avoid**

1. **Language Mismatch**
   ```
   ❌ Don't: Mix English and Amharic in same dataset
   ✅ Do: Keep one language per dataset
   ```

2. **Overwriting by Accident**
   ```
   ❌ Don't: Use standard mode when you meant incremental
   ✅ Do: Double-check mode before processing large batches
   ```

3. **Not Checking Results**
   ```
   ❌ Don't: Blindly add without reviewing stats
   ✅ Do: Check added/skipped counts after each merge
   ```

---

## Validation & Safety

### Automatic Validations

The system performs these validations automatically:

1. **Dataset Structure Validation**
   - Checks for required files: `metadata_train.csv`, `metadata_eval.csv`, `wavs/`
   - Fails fast if structure is invalid

2. **Language Compatibility**
   - Compares `lang.txt` between base and new datasets
   - Prevents mixing different languages
   - Creates `lang.txt` if base dataset doesn't have one

3. **Audio File Existence**
   - Verifies each audio file exists before copying
   - Skips missing files with warning
   - Continues processing other files

4. **Duplicate Detection**
   - Hashes audio files for comparison
   - Skips exact duplicates
   - Tracks statistics

### Error Handling

The merger is robust against errors:

```python
result = merger.merge_new_dataset(...)

if not result['success']:
    print(f"Merge failed: {result['error']}")
else:
    stats = result['stats']
    if stats['errors'] > 0:
        print(f"⚠️ {stats['errors']} errors occurred during merge")
```

---

## Performance Considerations

### Speed

- **Duplicate checking**: ~0.1ms per audio file (fast hashing)
- **Copying audio**: Limited by disk I/O
- **Metadata merging**: Very fast (pandas operations)

### Memory Usage

- Loads CSVs into memory (typically small < 10MB)
- Doesn't load audio into memory
- Memory footprint: Minimal (< 100MB for large datasets)

### Disk Space

- Keeps original files until merge completes
- Removes source datasets after successful merge (optional)
- Temporary space needed: ~2x size of new dataset

---

## Example Output

### Successful Incremental Merge

```
🔄 Using INCREMENTAL mode: Adding to existing dataset...

🔄 Incremental Merge: Adding 3 dataset(s) to base
   Base: ./finetune_models/dataset

[1/3] Processing: ./temp_dataset_1

📦 Merging new dataset: ./temp_dataset_1
  Language compatible: 'amh'
  Base dataset: 450 train, 50 eval segments
  New dataset: 120 train, 13 eval segments
  Computing hashes of existing audio files...
  Found 500 existing audio files
  Starting audio counter at: 500

✅ Merge Complete!
  Added 118 train segments
  Added 13 eval segments
  Skipped 2 duplicates
  Total segments: 581

[2/3] Processing: ./temp_dataset_2
...

============================================================
✅ Incremental Merge Complete!
   Datasets processed: 3
   Total train segments added: 340
   Total eval segments added: 38
   Duplicates skipped: 5
   Final total segments: 883
```

---

## Troubleshooting

### Problem: "Language mismatch" error

**Cause**: New dataset has different language than base dataset

**Solution**:
```python
# Check current dataset language
with open("finetune_models/dataset/lang.txt") as f:
    print(f"Base language: {f.read().strip()}")

# Make sure new videos are transcribed in same language
```

### Problem: No segments added (all duplicates)

**Cause**: All audio files in new dataset already exist

**Solution**:
- This is normal if reprocessing same videos
- Verify you're adding NEW content
- Check video URLs are different

### Problem: Merge is slow

**Cause**: Large number of existing files (duplicate checking)

**Solution**:
```python
# Disable duplicate checking for faster merge
merge_new_dataset(
    ...,
    check_duplicates=False  # Faster but may create duplicates
)
```

---

## API Reference

### `IncrementalDatasetMerger`

#### Constructor
```python
merger = IncrementalDatasetMerger(base_dataset_path: str)
```

#### Methods

**`merge_new_dataset()`**
```python
result = merger.merge_new_dataset(
    new_dataset_path: str,
    check_duplicates: bool = True,
    keep_source: bool = False
) -> Dict
```

Returns:
```python
{
    'success': True,
    'stats': {
        'added_train': 120,
        'added_eval': 13,
        'duplicates_skipped': 2,
        'errors': 0
    },
    'total_train': 570,
    'total_eval': 63,
    'total_segments': 633
}
```

### `merge_datasets_incremental()`

```python
from utils.incremental_dataset_merger import merge_datasets_incremental

train_csv, eval_csv, total_segments, stats = merge_datasets_incremental(
    new_dataset_paths: List[str],
    base_dataset_path: str,
    check_duplicates: bool = True,
    keep_sources: bool = False
)
```

---

## Future Enhancements

Potential improvements for future versions:

- [ ] Text-based deduplication (not just audio)
- [ ] Configurable hashing strategy (full file vs chunks)
- [ ] Undo/rollback capability
- [ ] Dataset versioning
- [ ] Automatic backup before merge
- [ ] Speaker identification and tracking
- [ ] Quality filtering during merge

---

## Summary

The Incremental Dataset Addition feature makes it easy to grow your TTS datasets over time without losing existing work. Key points:

✅ **Safe**: Validates compatibility, detects duplicates  
✅ **Fast**: Efficient hashing and file operations  
✅ **Flexible**: Works with YouTube batch processing and manual merging  
✅ **Reliable**: Robust error handling and detailed statistics  

**Start using it today to build your perfect TTS dataset incrementally!**
