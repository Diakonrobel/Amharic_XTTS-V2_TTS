# 💾 Google Drive Workflow for Amharic XTTS Training

**The recommended way to save and sync your training data across Colab sessions**

---

## 🎯 Why Google Drive?

✅ **15 GB Free Storage** - Enough for 2-3 complete training runs  
✅ **Native Colab Integration** - Mount with 2 lines of code  
✅ **Automatic Sync** - No manual upload/download needed  
✅ **Resume Training** - Continue from any checkpoint  
✅ **Access Anywhere** - View models on any device  
✅ **No Bandwidth Limits** - Unlimited transfers  
✅ **Version History** - 30 days of file versions  

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Complete Colab Workflow](#complete-colab-workflow)
3. [Checkpoint Management](#checkpoint-management)
4. [Best Practices](#best-practices)
5. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Step 1: Mount Google Drive in Colab

```python
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Create workspace
workspace = '/content/drive/MyDrive/XTTS_Training'
os.makedirs(workspace, exist_ok=True)
os.chdir(workspace)

print(f"✅ Google Drive mounted: {workspace}")
```

### Step 2: Clone Repository

```python
from pathlib import Path

# Clone in Google Drive (persists across sessions)
if not Path("Amharic_XTTS-V2_TTS").exists():
    !git clone https://github.com/Diakonrobel/Amharic_XTTS-V2_TTS.git
else:
    print("Repository already exists, pulling latest changes...")
    !cd Amharic_XTTS-V2_TTS && git pull

%cd Amharic_XTTS-V2_TTS
```

### Step 3: Train (Everything Saves Automatically!)

Your training data is now automatically saved to Google Drive!  
No manual steps needed - it just works! ✨

---

## 📚 Complete Colab Workflow

### Full Setup Cell

```python
#!/usr/bin/env python3
"""
Complete Google Drive setup for XTTS Training
Run this cell first in your Colab notebook
"""

from google.colab import drive
import os
import shutil
from pathlib import Path
from datetime import datetime

# ===== Mount Google Drive =====
print("📂 Mounting Google Drive...")
drive.mount('/content/drive')

# ===== Create Workspace =====
workspace = '/content/drive/MyDrive/XTTS_Training'
os.makedirs(workspace, exist_ok=True)
print(f"✅ Workspace: {workspace}")

# ===== Clone/Update Repository =====
os.chdir(workspace)

if not Path("Amharic_XTTS-V2_TTS").exists():
    print("🔽 Cloning repository...")
    !git clone https://github.com/Diakonrobel/Amharic_XTTS-V2_TTS.git
else:
    print("📂 Repository exists, updating...")
    !cd Amharic_XTTS-V2_TTS && git pull

os.chdir(f"{workspace}/Amharic_XTTS-V2_TTS")

# ===== Helper Functions =====
def save_checkpoint(description=""):
    """Save current training to Google Drive"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_name = f"checkpoint_{timestamp}"
    
    if description:
        checkpoint_name += f"_{description}"
    
    backup_dir = f"{workspace}/checkpoints/{checkpoint_name}"
    
    if os.path.exists('finetune_models'):
        os.makedirs(os.path.dirname(backup_dir), exist_ok=True)
        shutil.copytree('finetune_models', backup_dir, dirs_exist_ok=True)
        
        size = sum(f.stat().st_size for f in Path(backup_dir).rglob('*') if f.is_file())
        size_mb = size / (1024 * 1024)
        
        print(f"\n✅ Checkpoint saved!")
        print(f"📍 {backup_dir}")
        print(f"📊 {size_mb:.2f} MB")
        return backup_dir
    else:
        print("⚠️  No training data found!")
        return None

def list_checkpoints():
    """List all saved checkpoints"""
    checkpoint_dir = f"{workspace}/checkpoints"
    
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(os.listdir(checkpoint_dir))
        print(f"\n📋 Available checkpoints ({len(checkpoints)}):")
        print("=" * 80)
        
        for i, cp in enumerate(checkpoints, 1):
            cp_path = os.path.join(checkpoint_dir, cp)
            size = sum(f.stat().st_size for f in Path(cp_path).rglob('*') if f.is_file())
            size_mb = size / (1024 * 1024)
            print(f"  {i}. {cp} ({size_mb:.2f} MB)")
        
        print("=" * 80)
        return checkpoints
    else:
        print("📁 No checkpoints yet")
        return []

def load_checkpoint(checkpoint_name):
    """Load a specific checkpoint"""
    checkpoint_path = f"{workspace}/checkpoints/{checkpoint_name}"
    
    if os.path.exists(checkpoint_path):
        if os.path.exists('finetune_models'):
            shutil.rmtree('finetune_models')
        
        shutil.copytree(checkpoint_path, 'finetune_models')
        print(f"✅ Loaded: {checkpoint_name}")
        return True
    else:
        print(f"❌ Not found: {checkpoint_name}")
        return False

print("\n✅ Setup complete!")
print("\n📄 Available functions:")
print("  • save_checkpoint('description')  - Save training data")
print("  • list_checkpoints()              - List all checkpoints")
print("  • load_checkpoint('name')         - Load checkpoint")
print("\n💾 Everything in finetune_models/ is automatically saved to Google Drive!")
```

---

## 💾 Checkpoint Management

### Save Current Training

```python
# Save after completing an epoch
save_checkpoint("epoch_5_amharic")

# Save before taking a break
save_checkpoint("before_break")

# Save your best model
save_checkpoint("best_model_wer_0.15")
```

### List All Checkpoints

```python
# See all your saved checkpoints
list_checkpoints()
```

**Example output:**
```
📋 Available checkpoints (3):
================================================================================
  1. checkpoint_20250107_143022_epoch_5_amharic (1,872.45 MB)
  2. checkpoint_20250107_153045_before_break (1,856.32 MB)
  3. checkpoint_20250107_163011_best_model_wer_0.15 (1,891.67 MB)
================================================================================
```

### Load a Checkpoint

```python
# Resume from a specific checkpoint
load_checkpoint("checkpoint_20250107_143022_epoch_5_amharic")
```

### Delete Old Checkpoints

```python
import shutil

# Delete a specific checkpoint to save space
checkpoint_to_delete = "checkpoint_20250107_143022_epoch_5_amharic"
checkpoint_path = f"{workspace}/checkpoints/{checkpoint_to_delete}"

if os.path.exists(checkpoint_path):
    shutil.rmtree(checkpoint_path)
    print(f"✅ Deleted: {checkpoint_to_delete}")
```

---

## 📖 Best Practices

### 1. Save Regularly

```python
# Save after each training epoch
for epoch in range(num_epochs):
    train_one_epoch()
    save_checkpoint(f"epoch_{epoch+1}")
```

### 2. Use Descriptive Names

```python
# Good: Descriptive names
save_checkpoint("amharic_baseline_10epochs")
save_checkpoint("with_augmentation_lr_0.001")
save_checkpoint("best_model_loss_0.234")

# Bad: Generic names
save_checkpoint("model1")
save_checkpoint("test")
```

### 3. Clean Up Old Checkpoints

Keep your Drive organized by deleting old checkpoints you don't need:

```python
def cleanup_old_checkpoints(keep_last_n=3):
    """Keep only the N most recent checkpoints"""
    checkpoint_dir = f"{workspace}/checkpoints"
    
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(os.listdir(checkpoint_dir))
        
        # Delete old checkpoints
        for cp in checkpoints[:-keep_last_n]:
            cp_path = os.path.join(checkpoint_dir, cp)
            shutil.rmtree(cp_path)
            print(f"🗑️  Deleted: {cp}")
        
        print(f"✅ Kept {keep_last_n} most recent checkpoints")

# Run cleanup
cleanup_old_checkpoints(keep_last_n=5)
```

### 4. Monitor Storage Usage

```python
def check_storage_usage():
    """Check how much Drive space you're using"""
    checkpoint_dir = f"{workspace}/checkpoints"
    
    if os.path.exists(checkpoint_dir):
        total_size = 0
        for cp in os.listdir(checkpoint_dir):
            cp_path = os.path.join(checkpoint_dir, cp)
            size = sum(f.stat().st_size for f in Path(cp_path).rglob('*') if f.is_file())
            total_size += size
        
        total_gb = total_size / (1024 * 1024 * 1024)
        print(f"📊 Total storage used: {total_gb:.2f} GB")
        print(f"📊 Free tier remaining: {15 - total_gb:.2f} GB / 15 GB")
    else:
        print("📁 No checkpoints yet")

# Check usage
check_storage_usage()
```

---

## 🔄 Complete Training Workflow

### 1. Start New Training Session

```python
# Mount Drive and setup (run once per session)
from google.colab import drive
drive.mount('/content/drive')

workspace = '/content/drive/MyDrive/XTTS_Training'
%cd {workspace}/Amharic_XTTS-V2_TTS

# Load helper functions (see Complete Setup Cell above)
# ... paste save_checkpoint, list_checkpoints, load_checkpoint functions ...

# Start training!
!python xtts_demo.py
```

### 2. During Training

```python
# Train for a few epochs
# In the WebUI or CLI...

# Save checkpoint
save_checkpoint("epoch_5_completed")
```

### 3. If Colab Disconnects

```python
# Don't worry! Your data is safe in Google Drive

# Re-mount Drive
from google.colab import drive
drive.mount('/content/drive')

workspace = '/content/drive/MyDrive/XTTS_Training'
%cd {workspace}/Amharic_XTTS-V2_TTS

# List available checkpoints
list_checkpoints()

# Load your latest checkpoint
load_checkpoint("checkpoint_20250107_143022_epoch_5_completed")

# Continue training!
!python xtts_demo.py
```

### 4. Download Final Model (Optional)

```python
# Your model is already in Google Drive!
# But if you want to download to local computer:

from google.colab import files
import zipfile

# Create zip
!zip -r amharic_model.zip finetune_models/

# Download
files.download('amharic_model.zip')
```

---

## 🛠️ Troubleshooting

### Issue 1: "Permission denied" when mounting Drive

**Solution:**
1. Clear browser cache
2. Try incognito mode
3. Re-authorize Google Drive access
4. Restart Colab runtime

### Issue 2: "No space left on device"

**Solution:**
```python
# Check your usage
check_storage_usage()

# Clean up old checkpoints
cleanup_old_checkpoints(keep_last_n=2)

# Or upgrade to Google One (100 GB for $1.99/month)
```

### Issue 3: Slow sync to Drive

**Solution:**
```python
# Drive sync is automatic and happens in background
# For large files, give it a few minutes

# Check if sync is complete
!ls -lh finetune_models/

# Force sync (not usually needed)
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive')
```

### Issue 4: Can't find my checkpoints

**Solution:**
```python
# Check workspace path
print(f"Workspace: {workspace}")

# List directory contents
!ls -la {workspace}/checkpoints/

# Manually browse in Drive
print("Browse to: https://drive.google.com/drive/my-drive")
print("Then navigate to: XTTS_Training/checkpoints/")
```

---

## 📊 Storage Planning

### Free Tier (15 GB)

Perfect for:
- ✅ 2-3 complete training runs
- ✅ Learning and experimentation
- ✅ Single model development

Example:
- Base model download: ~2 GB
- Training checkpoint 1: ~2 GB
- Training checkpoint 2: ~2 GB  
- Final model: ~2 GB
- **Total: ~8 GB** (plenty of space!)

### Paid Tier ($1.99/month = 100 GB)

Perfect for:
- ✅ 10-15 training runs
- ✅ Multiple experiments
- ✅ Model version history
- ✅ Audio dataset storage

### Paid Tier ($2.99/month = 200 GB)

Perfect for:
- ✅ Heavy experimentation
- ✅ Multiple projects
- ✅ Large audio datasets
- ✅ Extended version history

---

## 🎯 Quick Reference

### Essential Commands

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Save checkpoint
save_checkpoint("description")

# List checkpoints
list_checkpoints()

# Load checkpoint
load_checkpoint("checkpoint_name")

# Check storage
check_storage_usage()

# Clean up
cleanup_old_checkpoints(keep_last_n=5)
```

### Directory Structure

```
Google Drive/
└── XTTS_Training/
    ├── Amharic_XTTS-V2_TTS/          # Your repository
    │   ├── finetune_models/           # Current training (auto-saved)
    │   ├── xtts_demo.py
    │   └── ...
    └── checkpoints/                   # Manual backups
        ├── checkpoint_20250107_143022_epoch_5/
        ├── checkpoint_20250107_153045_best_model/
        └── ...
```

---

## 🚀 Advanced Tips

### 1. Automatic Periodic Saves

```python
import time
import threading

def auto_save_loop(interval_minutes=30):
    """Auto-save every N minutes"""
    while True:
        time.sleep(interval_minutes * 60)
        save_checkpoint(f"auto_save_{datetime.now().strftime('%H%M')}")
        print(f"🔄 Auto-saved at {datetime.now().strftime('%H:%M')}")

# Start auto-save in background
auto_save_thread = threading.Thread(target=auto_save_loop, args=(30,), daemon=True)
auto_save_thread.start()
print("✅ Auto-save enabled (every 30 minutes)")
```

### 2. Checkpoint Comparison

```python
def compare_checkpoints(cp1, cp2):
    """Compare size of two checkpoints"""
    cp1_path = f"{workspace}/checkpoints/{cp1}"
    cp2_path = f"{workspace}/checkpoints/{cp2}"
    
    if os.path.exists(cp1_path) and os.path.exists(cp2_path):
        size1 = sum(f.stat().st_size for f in Path(cp1_path).rglob('*') if f.is_file())
        size2 = sum(f.stat().st_size for f in Path(cp2_path).rglob('*') if f.is_file())
        
        print(f"{cp1}: {size1/(1024*1024):.2f} MB")
        print(f"{cp2}: {size2/(1024*1024):.2f} MB")
        print(f"Difference: {abs(size1-size2)/(1024*1024):.2f} MB")
```

### 3. Export to Hugging Face Hub

```python
from huggingface_hub import HfApi, login

# Login to Hugging Face
login()

# Upload your best model
api = HfApi()
api.create_repo("your-username/amharic-xtts-v2", private=False)

api.upload_folder(
    folder_path="finetune_models/ready",
    repo_id="your-username/amharic-xtts-v2",
    repo_type="model"
)

print("✅ Model uploaded to Hugging Face Hub!")
```

---

## ✅ Summary

**Google Drive is perfect for Colab training:**

✅ **15 GB Free** - More than enough for most users  
✅ **2-Line Setup** - Mount and go  
✅ **Auto-Save** - Everything persists automatically  
✅ **No LFS Complexity** - No Git LFS configuration needed  
✅ **Better Value** - $1.99/month for 100 GB vs $60-300/year for GitHub LFS  

**Your complete workflow:**
1. Mount Google Drive (once per session)
2. Clone repository to Drive
3. Train (auto-saves to Drive)
4. Optional: Manual checkpoints with `save_checkpoint()`
5. Resume anytime with `load_checkpoint()`

---

**🎉 You're all set! Start training and never worry about losing your data!**
