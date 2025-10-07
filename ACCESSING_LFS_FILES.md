# 📂 How to Access Your GitHub LFS Files - Complete Guide

A comprehensive step-by-step guide for viewing and downloading all your files stored in GitHub LFS.

---

## 📋 Table of Contents

1. [Quick Overview](#quick-overview)
2. [Method 1: Using GitHub Web Interface](#method-1-using-github-web-interface)
3. [Method 2: Using Git Command Line (Windows)](#method-2-using-git-command-line-windows)
4. [Method 3: Using GitHub Desktop](#method-3-using-github-desktop)
5. [Method 4: Viewing LFS Storage Usage](#method-4-viewing-lfs-storage-usage)
6. [Troubleshooting](#troubleshooting)

---

## 🎯 Quick Overview

### What is GitHub LFS?

GitHub LFS (Large File Storage) stores large files separately from your main Git repository. When you push large files, GitHub stores:
- **Pointer files** in your Git repository (small text files)
- **Actual files** in GitHub LFS storage

### What You Can Access:

- ✅ **Model files** (`.pth`, `.ckpt`, `.safetensors`)
- ✅ **Audio files** (`.wav`, `.mp3`)
- ✅ **Archives** (`.zip`, `.tar.gz`)
- ✅ **Training data** in `finetune_models/` directory
- ✅ **Any file tracked by LFS**

---

## Method 1: Using GitHub Web Interface

### Step 1: View Files in Browser

1. **Go to your repository:**
   ```
   https://github.com/Diakonrobel/Amharic_XTTS-V2_TTS
   ```

2. **Navigate to the folder** containing LFS files:
   - Click on `finetune_models/` folder
   - Or any folder containing large files

3. **Identify LFS files:**
   - LFS-tracked files show a **"Stored with Git LFS"** badge
   - Files appear with their full size
   - You can preview some file types

### Step 2: Download Individual Files

**Option A: Direct Download**
1. Click on the LFS file name
2. Click the **"Download"** button (top right)
3. File downloads to your Downloads folder

**Option B: View File Details**
1. Click on the file
2. See file size, history, and SHA
3. Click **"Download"** to get the file

### Step 3: Download Entire Folder

**Using GitHub Download:**
1. Go to the main repository page
2. Click green **"Code"** button
3. Select **"Download ZIP"**
4. Extract the ZIP file
5. **Important**: Run `git lfs pull` afterward to get actual LFS files (see Method 2)

⚠️ **Note**: Direct ZIP download only includes LFS pointer files, not actual large files!

---

## Method 2: Using Git Command Line (Windows)

This is the **recommended method** for accessing all LFS files.

### Prerequisites

✅ Git installed (you already have it!)  
✅ Git LFS installed (you already have it!)  
✅ GitHub credentials configured

### Step 1: Check Your Current LFS Status

Open **PowerShell** or **Command Prompt** in your project folder:

```powershell
# Navigate to your project
cd D:\FINETUNE-XTTS-WEBUI-LIGHTNING\xtts-finetune-webui-fresh

# Check Git LFS status
git lfs env
```

**Expected output:**
```
git-lfs/3.3.0 (GitHub; windows amd64; go 1.19.3)
Endpoint=https://github.com/Diakonrobel/Amharic_XTTS-V2_TTS.git/info/lfs
```

### Step 2: List All LFS Files

```powershell
# See all LFS-tracked files in repository
git lfs ls-files

# See LFS files with more details
git lfs ls-files -s
```

**Example output:**
```
abc123def * finetune_models/model.pth
456789abc * finetune_models/checkpoint.ckpt
```

### Step 3: Pull All LFS Files

**Option A: Pull All LFS Files**
```powershell
# Download all LFS files from GitHub
git lfs pull

# Or use fetch + checkout
git lfs fetch --all
git lfs checkout
```

**Option B: Pull Latest Changes + LFS Files**
```powershell
# Get latest code and LFS files
git pull origin main
git lfs pull
```

**Option C: Pull Specific LFS Files**
```powershell
# Download only specific files
git lfs pull --include="finetune_models/model.pth"

# Download files from specific folder
git lfs pull --include="finetune_models/*"
```

### Step 4: Verify Downloaded Files

```powershell
# Check if LFS files are downloaded (not pointers)
git lfs ls-files

# See file sizes to confirm they're real files
Get-ChildItem -Recurse -File | Where-Object { $_.Extension -in ".pth",".ckpt",".wav" } | Select-Object FullName, Length
```

### Step 5: Access Your Files

Your LFS files are now in your local directory:

```
D:\FINETUNE-XTTS-WEBUI-LIGHTNING\xtts-finetune-webui-fresh\
├── finetune_models\
│   ├── model.pth          ← Your trained model
│   ├── checkpoint.ckpt    ← Training checkpoint
│   └── ...
```

---

## Method 3: Using GitHub Desktop

### Step 1: Install GitHub Desktop (Optional)

1. Download from: https://desktop.github.com/
2. Install and sign in with your GitHub account

### Step 2: Clone Repository

1. Open GitHub Desktop
2. Click **"File" → "Clone repository"**
3. Select **"URL"** tab
4. Enter: `https://github.com/Diakonrobel/Amharic_XTTS-V2_TTS.git`
5. Choose local path (e.g., `D:\GitHub\Amharic_XTTS-V2_TTS`)
6. Click **"Clone"**

### Step 3: Pull LFS Files

**Automatic (usually happens during clone):**
- GitHub Desktop automatically fetches LFS files

**Manual (if needed):**
1. Open Command Prompt in the cloned directory
2. Run: `git lfs pull`

### Step 4: Access Files

Browse to your cloned directory and find your LFS files:
```
D:\GitHub\Amharic_XTTS-V2_TTS\finetune_models\
```

---

## Method 4: Viewing LFS Storage Usage

### Check Your LFS Storage Quota

1. **Via GitHub Web:**
   - Go to https://github.com/settings/billing
   - Click **"Plans and usage"**
   - See **"Git LFS Data"** section
   - Shows: Used / Total (e.g., "0.5 GB / 1 GB")

2. **Via Command Line:**
   ```powershell
   # List all LFS objects with sizes
   git lfs ls-files -s
   
   # Calculate total size
   git lfs ls-files -s | Measure-Object -Property Length -Sum
   ```

### View LFS Files in Your Repository

**See which files are tracked:**
```powershell
# List tracked file patterns
cat .gitattributes

# See actual LFS files
git lfs ls-files
```

**Check file sizes:**
```powershell
# Windows PowerShell
Get-ChildItem finetune_models -Recurse | Select-Object Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length / 1MB, 2)}}
```

---

## 🔍 Complete Windows Workflow Example

Here's a complete example for accessing all your LFS files:

```powershell
# 1. Open PowerShell
# Press Win+X, select "Windows PowerShell"

# 2. Navigate to your project
cd D:\FINETUNE-XTTS-WEBUI-LIGHTNING\xtts-finetune-webui-fresh

# 3. Ensure Git LFS is initialized
git lfs install

# 4. Check current LFS status
git lfs ls-files

# 5. Pull latest changes
git pull origin main

# 6. Download all LFS files
git lfs pull

# 7. Verify files are downloaded
git lfs ls-files

# 8. Check file sizes (should be large, not ~130 bytes)
Get-ChildItem finetune_models -Recurse -File | Select-Object Name, Length

# 9. Your files are ready!
explorer finetune_models
```

---

## 🛠️ Troubleshooting

### Issue 1: "This repository is over its data quota"

**Problem:** You've exceeded free LFS storage (1GB)

**Solutions:**

1. **Upgrade GitHub plan:**
   - Go to https://github.com/settings/billing
   - Purchase additional LFS storage

2. **Clean up old LFS files:**
   ```powershell
   # Remove old LFS files from history
   git lfs prune
   
   # See what will be removed
   git lfs prune --dry-run
   ```

3. **Use Git LFS more selectively:**
   - Update `.gitattributes` to track fewer file types
   - Store very large files elsewhere (Google Drive, AWS S3)

---

### Issue 2: "LFS files are still pointers (small ~130 bytes)"

**Problem:** Files downloaded but are pointer files, not actual files

**Solution:**
```powershell
# Force download actual files
git lfs fetch --all
git lfs checkout

# Or pull again
git lfs pull --include="*"
```

---

### Issue 3: "Authentication failed"

**Problem:** GitHub credentials not configured

**Solution:**

**Option A: Use Personal Access Token (Recommended)**
```powershell
# When prompted for password, use your Personal Access Token
# Create token at: https://github.com/settings/tokens
# Scopes needed: repo, read:org
```

**Option B: Use GitHub CLI**
```powershell
# Install GitHub CLI
winget install GitHub.cli

# Authenticate
gh auth login
```

**Option C: Configure Git credentials**
```powershell
# Store credentials
git config --global credential.helper wincred

# Set username
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

### Issue 4: "Cannot download specific file"

**Problem:** File exists in LFS but won't download

**Solution:**
```powershell
# 1. Check if file is actually in LFS
git lfs ls-files | grep "filename"

# 2. Fetch that specific file
git lfs fetch --include="path/to/file.pth"

# 3. Checkout the file
git lfs checkout "path/to/file.pth"

# 4. Verify file size
Get-Item "path\to\file.pth" | Select-Object Length
```

---

### Issue 5: "Slow download speed"

**Problem:** LFS downloads are slow

**Solutions:**

1. **Use multiple connections:**
   ```powershell
   # Configure concurrent transfers (default is 8)
   git config lfs.concurrenttransfers 16
   ```

2. **Download specific files only:**
   ```powershell
   # Don't download everything
   git lfs pull --include="finetune_models/*.pth"
   ```

3. **Use SSH instead of HTTPS:**
   ```powershell
   # Change remote URL
   git remote set-url origin git@github.com:Diakonrobel/Amharic_XTTS-V2_TTS.git
   ```

---

### Issue 6: "File not found in LFS"

**Problem:** Expected file is missing

**Solution:**

1. **Check if file was actually pushed:**
   ```powershell
   # View commit history
   git log --oneline --all -- path/to/file
   
   # Check if file is tracked by LFS
   git check-attr filter path/to/file
   ```

2. **Check GitHub web interface:**
   - Browse to the file on GitHub
   - See if it shows "Stored with Git LFS" badge

3. **Re-push if needed:**
   ```powershell
   # Force add to LFS
   git lfs track "path/to/file"
   git add path/to/file
   git commit -m "Add file to LFS"
   git push origin main
   ```

---

## 📊 Useful Commands Reference

### Viewing LFS Files

```powershell
# List all LFS files
git lfs ls-files

# List with sizes
git lfs ls-files -s

# List with full paths
git lfs ls-files -n

# Show LFS environment
git lfs env

# Show tracked patterns
cat .gitattributes
```

### Downloading LFS Files

```powershell
# Download all LFS files
git lfs pull

# Download specific file
git lfs pull --include="file.pth"

# Download specific folder
git lfs pull --include="finetune_models/*"

# Fetch but don't checkout
git lfs fetch

# Checkout after fetch
git lfs checkout
```

### Managing LFS Files

```powershell
# Track new file type
git lfs track "*.model"

# Untrack file type
git lfs untrack "*.txt"

# See what's tracked
git lfs track

# Clean up old LFS files
git lfs prune

# Show what would be pruned
git lfs prune --dry-run
```

### Checking File Status

```powershell
# Check if file is a pointer or real file
git lfs pointer --file="path/to/file.pth"

# View file size
Get-Item "path\to\file.pth" | Select-Object Name, Length

# View all large files
Get-ChildItem -Recurse -File | Where-Object { $_.Length -gt 10MB } | Select-Object FullName, @{Name="Size(MB)";Expression={[math]::Round($_.Length / 1MB, 2)}}
```

---

## 🎯 Quick Action Checklist

Use this checklist to quickly access your LFS files:

- [ ] **Step 1:** Open PowerShell in project directory
- [ ] **Step 2:** Run `git lfs env` to verify LFS is working
- [ ] **Step 3:** Run `git lfs ls-files` to see tracked files
- [ ] **Step 4:** Run `git pull origin main` to get latest changes
- [ ] **Step 5:** Run `git lfs pull` to download LFS files
- [ ] **Step 6:** Run `Get-ChildItem finetune_models` to verify files exist
- [ ] **Step 7:** Open `finetune_models/` folder in File Explorer
- [ ] **Step 8:** Your LFS files are now accessible! ✅

---

## 💡 Pro Tips

1. **Use PowerShell Profile** for quick access:
   ```powershell
   # Add to PowerShell profile
   function Pull-LFS {
       git pull origin main
       git lfs pull
   }
   ```

2. **Create desktop shortcut:**
   - Right-click desktop → New → Shortcut
   - Location: `powershell.exe -NoExit -Command "cd D:\FINETUNE-XTTS-WEBUI-LIGHTNING\xtts-finetune-webui-fresh; git lfs pull"`
   - Name: "Pull XTTS LFS Files"

3. **Monitor LFS bandwidth:**
   - GitHub gives 1GB bandwidth/month free
   - Check usage: https://github.com/settings/billing
   - Each download counts toward bandwidth

4. **Save bandwidth with shallow clones:**
   ```powershell
   # Clone without full history
   git clone --depth 1 https://github.com/Diakonrobel/Amharic_XTTS-V2_TTS.git
   cd Amharic_XTTS-V2_TTS
   git lfs pull
   ```

---

## 📞 Getting Help

If you encounter issues not covered here:

1. **Check Git LFS documentation:**
   - https://git-lfs.github.com/

2. **View GitHub LFS help:**
   ```powershell
   git lfs help
   git lfs help pull
   ```

3. **GitHub Support:**
   - https://support.github.com/

4. **Repository Issues:**
   - https://github.com/Diakonrobel/Amharic_XTTS-V2_TTS/issues

---

## ✅ Success Indicators

You've successfully accessed your LFS files when:

- ✅ `git lfs ls-files` shows your files
- ✅ File sizes are large (MB/GB), not ~130 bytes
- ✅ You can open model files (`.pth`) in PyTorch
- ✅ Audio files (`.wav`) play correctly
- ✅ No error messages when running `git lfs pull`

---

**You're all set!** 🎉

Your GitHub LFS files are now accessible on your Windows machine. You can find them in:
```
D:\FINETUNE-XTTS-WEBUI-LIGHTNING\xtts-finetune-webui-fresh\finetune_models\
```
