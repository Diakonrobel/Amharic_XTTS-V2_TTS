# Dataset Merger UI Integration Guide

## Overview
This guide explains how to integrate the new **Dataset Merger** tab into the existing Gradio WebUI.

## Files Created
1. `webui/dataset_merger_ui.py` - The UI component
2. `merge_datasets_simple.py` - Backend merge script (already exists)
3. `standardize_filenames.py` - Backend standardization script (already exists)

## Integration Steps

### Step 1: Import the Module

Add this import at the top of `xtts_demo.py` (around line 27 after other utils imports):

```python
from webui import dataset_merger_ui
```

### Step 2: Add the New Tab

Find the section where tabs are defined (search for `with gr.Tab` statements).
Add the new tab after the "Data Processing" tab (around line 1040):

```python
        with gr.Tab("🔀 Dataset Merger"):
            dataset_merger_ui.create_dataset_merger_tab()
```

## Complete Integration Example

Here's the exact code to add to `xtts_demo.py`:

### At the imports section (~line 27):
```python
from webui import dataset_merger_ui
```

### In the Gradio Blocks (~line 1040, after Data Processing tab):
```python
        with gr.Tab("🔀 Dataset Merger"):
            dataset_merger_ui.create_dataset_merger_tab()
```

## Features Included

### 🎯 Core Functionality
- **Scan**: Auto-detect all temp_dataset_* folders
- **Preview**: See what will happen before merging
- **Selective Merge**: Choose specific datasets or merge all
- **Backup**: Automatic backup of metadata files
- **Standardization**: Rename all files to merged_XXXXXXXX.wav format
- **Cleanup**: Choose to keep/move/delete temp datasets

### 🛡️ Safety Features
- ✅ Backup creation before any changes
- ✅ Duplicate detection and removal
- ✅ Transaction-safe rename operations
- ✅ Detailed logging and JSON reports
- ✅ Timeout protection (10-minute max)
- ✅ Error handling and rollback capability

### 📊 UI Elements
- Interactive dataset scanner with statistics
- Real-time preview of merge operations
- Progress indication
- Downloadable JSON reports
- Detailed logs for debugging

## Workflow Compliance

This implementation follows your specified workflow:

1. ✅ **Local development** → Make changes locally
2. ✅ **Commit/push to GitHub** → Use git to version control
3. ✅ **Lightning AI pulls** → Pull changes on Lightning AI
4. ✅ **Safe operations** → All operations include backups and confirmations

The merge operations run via subprocess calls to existing Python scripts, ensuring isolation and safety.

## Testing

After integration, test the new tab:

1. Start the Gradio app: `python xtts_demo.py`
2. Navigate to the "🔀 Dataset Merger" tab
3. Click "Scan for Temp Datasets"
4. Use "Preview" to see what will happen
5. Execute merge with your preferred settings

## Example Usage

### Typical Workflow:
1. Create multiple datasets using "Data Processing" tab
2. Switch to "Dataset Merger" tab
3. Scan for temp datasets
4. Preview the merge
5. Execute with "Standardize Filenames" enabled
6. Choose "Move to merged_temps/" for cleanup
7. Download the JSON report for records

## Troubleshooting

### "Base directory not found"
- Ensure the base path includes `finetune_models/` directory
- Default is current working directory

### "Merge scripts not found"
- Ensure `merge_datasets_simple.py` and `standardize_filenames.py` are in `finetune_models/` or project root
- Check file permissions

### "Subprocess timeout"
- Large datasets may take time
- Timeout is set to 10 minutes
- Can be adjusted in `webui/dataset_merger_ui.py` line 144

## Benefits

✅ **User-Friendly**: No command-line required  
✅ **Safe**: Multiple safety checks and backups  
✅ **Visible**: Real-time feedback and detailed logs  
✅ **Flexible**: Choose what to merge and how to clean up  
✅ **Professional**: SOTA UI with modern Gradio components  
✅ **Documented**: Comprehensive help text in the UI  

## Next Steps

1. Commit changes to Git
2. Push to GitHub
3. Pull on Lightning AI
4. Test the integration
5. Enjoy automated dataset management!

---

**Note**: All operations are logged and can be audited via the JSON reports generated in `finetune_models/`.
