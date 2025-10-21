# 🚀 Git Deployment Guide for Training Patches

## Quick Start

Follow these steps to deploy the automatic NaN fix to Lightning AI:

---

## Step 1: Test Locally (Windows)

```powershell
# Navigate to project directory
cd D:\FINETUNE-XTTS-WEBUI-LIGHTNING\xtts-finetune-webui-fresh

# Run the test script
python test_training_patches.py
```

**Expected output:**
```
🧪 TESTING TRAINING PATCHES
======================================================================
🔧 APPLYING AUTOMATIC TRAINING PATCHES FOR NaN LOSS FIX
✅ Patch 1: Fixed deprecated torch.cuda.amp.autocast
✅ Patch 2: Wrapped GradScaler with SafeGradScaler
✅ Patch 3: Enhanced gradient clipping with NaN safety
...
✅ ALL TESTS PASSED!
```

---

## Step 2: Commit to Git

```powershell
# Check what files were changed/added
git status

# You should see:
# - utils/training_patches.py (new)
# - headlessXttsTrain.py (modified)
# - TRAINING_PATCHES_README.md (new)
# - test_training_patches.py (new)
# - GIT_DEPLOYMENT.md (new)

# Add all the new files
git add utils/training_patches.py
git add headlessXttsTrain.py
git add TRAINING_PATCHES_README.md
git add test_training_patches.py
git add GIT_DEPLOYMENT.md

# Commit with descriptive message
git commit -m "🛡️ Add automatic training patches to fix NaN loss issue

- Added utils/training_patches.py: Auto-fixes FP16 NaN issues
- Modified headlessXttsTrain.py: Added automatic patch import
- Fixes: torch.cuda.amp.autocast deprecation
- Fixes: GradScaler init_scale too high (65536 → 1024)
- Adds: Automatic NaN detection and skip logic
- Adds: Safe gradient clipping with scaler
- Tested: All patches verified with test_training_patches.py

This resolves the NaN loss at step 50 issue in mixed precision training."
```

---

## Step 3: Push to GitHub

```powershell
# Push to your repository
git push origin main
# Or if your branch is named differently:
# git push origin master
# git push origin develop
```

---

## Step 4: Deploy to Lightning AI

### Option A: Using SSH (Recommended)
```bash
# SSH into your Lightning AI instance
ssh your-lightning-instance

# Navigate to project directory
cd /teamspace/studios/this_studio/your-project-name

# Pull latest changes
git pull origin main

# Verify patches are present
ls -la utils/training_patches.py
# Should show the file exists

# Test patches work
python test_training_patches.py
# Should show all tests pass
```

### Option B: Using Lightning AI Web Terminal
1. Open Lightning AI web interface
2. Open terminal
3. Run:
```bash
cd /teamspace/studios/this_studio/your-project-name
git pull origin main
python test_training_patches.py
```

---

## Step 5: Run Training

```bash
# On Lightning AI, run your training as usual
python headlessXttsTrain.py [your-training-arguments]

# OR if using the gradio webui:
python app.py
```

**You should see patch messages at startup:**
```
======================================================================
🔧 APPLYING AUTOMATIC TRAINING PATCHES FOR NaN LOSS FIX
======================================================================
✅ Patch 1: Fixed deprecated torch.cuda.amp.autocast
✅ Patch 2: Wrapped GradScaler with SafeGradScaler (conservative settings)
...
✅ ALL TRAINING PATCHES APPLIED SUCCESSFULLY
======================================================================
```

---

## Verification Checklist

Before pushing, verify:

- [ ] `test_training_patches.py` passes all tests locally
- [ ] `utils/training_patches.py` exists
- [ ] `headlessXttsTrain.py` line 22 has: `from utils import training_patches`
- [ ] All files committed to git

After pulling on Lightning AI, verify:

- [ ] `git pull` completed successfully
- [ ] `test_training_patches.py` passes on Lightning AI
- [ ] Training shows patch messages at startup
- [ ] No NaN loss after step 50

---

## Troubleshooting

### Problem: Git merge conflict
**Solution:**
```bash
git stash  # Save your local changes
git pull origin main
git stash pop  # Reapply your changes
# Resolve any conflicts manually
```

### Problem: Can't push to GitHub
**Solution:**
```powershell
# Check remote URL
git remote -v

# If using HTTPS and password doesn't work, use Personal Access Token
# Go to GitHub Settings → Developer Settings → Personal Access Tokens
# Generate token with 'repo' scope
# Use token as password when prompted
```

### Problem: Patches don't appear on Lightning AI
**Solution:**
```bash
# Force clean pull
git fetch --all
git reset --hard origin/main
# Warning: This discards local changes!

# Or check if files exist
find . -name "training_patches.py"
```

### Problem: Test fails on Lightning AI
**Solution:**
```bash
# Check Python version (should be 3.8+)
python --version

# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# If issues persist, check the full error:
python test_training_patches.py 2>&1 | tee test_output.txt
```

---

## One-Command Deployment (Advanced)

Create this script for faster deployment:

**deploy.ps1** (Windows):
```powershell
# Test locally
python test_training_patches.py
if ($LASTEXITCODE -ne 0) { Write-Error "Tests failed!"; exit 1 }

# Commit and push
git add -A
git commit -m "Update training patches"
git push origin main

Write-Host "✅ Deployed to GitHub! Now pull on Lightning AI."
```

**deploy.sh** (Lightning AI):
```bash
#!/bin/bash
cd /teamspace/studios/this_studio/your-project-name
git pull origin main
python test_training_patches.py
if [ $? -eq 0 ]; then
    echo "✅ Ready to train!"
else
    echo "❌ Tests failed!"
    exit 1
fi
```

---

## Summary

1. **Test locally**: `python test_training_patches.py`
2. **Commit**: `git add` + `git commit`
3. **Push**: `git push origin main`
4. **Pull on Lightning AI**: `git pull origin main`
5. **Verify**: `python test_training_patches.py`
6. **Train**: Run your training normally

**The patches apply automatically - no manual steps needed during training!**

---

## Need Help?

- Check `TRAINING_PATCHES_README.md` for detailed documentation
- Check logs for patch confirmation messages
- Verify `utils/training_patches.py` is present
- Ensure line 22 of `headlessXttsTrain.py` imports patches

Good luck! 🚀
