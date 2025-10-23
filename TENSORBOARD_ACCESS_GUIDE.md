# TensorBoard Access Guide

## 🎯 TensorBoard is Already Running!

TensorBoard is **automatically configured** when you start training. You just need to launch it in a separate terminal.

---

## 📊 Quick Access (3 Steps)

### Step 1: Find Your Training Directory

When training starts, you'll see this in the console:

```
 > Start Tensorboard: tensorboard --logdir=/path/to/your/output/run/training
```

**From your logs:**
```
 > Start Tensorboard: tensorboard --logdir=/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/finetune_models/run/training
```

### Step 2: Open a New Terminal

**Windows (PowerShell/CMD):**
```powershell
# Open new PowerShell window
# Navigate to your project
cd D:\FINETUNE-XTTS-WEBUI-LIGHTNING\xtts-finetune-webui-fresh
```

**Linux/Mac/Lightning AI:**
```bash
# Open new terminal tab/window
cd /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS
```

### Step 3: Launch TensorBoard

**Copy the command from your console** (look for "Start Tensorboard" line):

```bash
tensorboard --logdir=/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/finetune_models/run/training
```

**Or use the generic format:**
```bash
tensorboard --logdir=<your_output_path>/run/training
```

### Step 4: Open in Browser

TensorBoard will start and show:
```
TensorBoard 2.x.x at http://localhost:6006/ (Press CTRL+C to quit)
```

**Open your browser** and go to: **http://localhost:6006**

---

## 🚀 Platform-Specific Instructions

### Windows (Local)

**Option 1: PowerShell**
```powershell
# Open new PowerShell window
cd D:\FINETUNE-XTTS-WEBUI-LIGHTNING\xtts-finetune-webui-fresh
tensorboard --logdir=finetune_models\run\training
```

**Option 2: Command Prompt**
```cmd
cd D:\FINETUNE-XTTS-WEBUI-LIGHTNING\xtts-finetune-webui-fresh
tensorboard --logdir=finetune_models\run\training
```

**Option 3: Windows Terminal** (recommended)
1. Open Windows Terminal
2. New tab (Ctrl+Shift+T)
3. Run the tensorboard command above

### Linux/Mac (Local)

```bash
cd /path/to/xtts-finetune-webui-fresh
tensorboard --logdir=finetune_models/run/training
```

### Lightning AI / Remote Server

**Option 1: SSH Tunnel (Recommended)**

1. **On your local machine**, create SSH tunnel:
```bash
ssh -L 6006:localhost:6006 your-user@your-server
```

2. **On the remote server**, start TensorBoard:
```bash
tensorboard --logdir=/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/finetune_models/run/training
```

3. **On your local machine**, open: **http://localhost:6006**

**Option 2: Lightning AI Studio Interface**

If using Lightning AI Studio:
1. Look for the "Ports" or "Endpoints" tab
2. TensorBoard might be auto-exposed on port 6006
3. Click the provided URL

**Option 3: Public URL (Less Secure)**
```bash
tensorboard --logdir=/path/to/training --host=0.0.0.0 --port=6006
```
Then access via: `http://<your-server-ip>:6006`

---

## 📈 What You'll See in TensorBoard

### Main Dashboards

#### 1. **SCALARS** (Most Important)

**Training Metrics:**
- `train/loss_total` - Overall training loss (should decrease)
- `train/loss_text_ce` - Text prediction loss
- `train/loss_mel_ce` - Mel spectrogram loss
- `train/grad_norm` - Gradient norms (should stay 1-10)
- `train/lr` - Current learning rate

**Validation Metrics:**
- `eval/loss_total` - Validation loss
- `eval/loss_text_ce` - Validation text loss
- `eval/loss_mel_ce` - Validation mel loss

**What to Look For:**
- ✅ **train/loss_total** steadily decreasing
- ✅ **eval/loss_total** tracking train loss (maybe slightly higher)
- ✅ **train/grad_norm** staying in 1-10 range (not exploding)
- ✅ **train/lr** showing drops at 50k, 150k, 300k steps

#### 2. **AUDIO**

Listen to generated test sentences during training:
- Audio samples generated at regular intervals
- Compare quality across different checkpoints
- Hear pronunciation improvements over time

#### 3. **IMAGES** (if enabled)

- Mel spectrogram visualizations
- Attention plots
- Alignment matrices

#### 4. **GRAPHS**

- Model architecture visualization
- Computational graph

---

## 🎛️ Advanced TensorBoard Options

### Custom Port

If port 6006 is busy:
```bash
tensorboard --logdir=<path> --port=6007
```

### Multiple Training Runs Comparison

Compare different training runs:
```bash
tensorboard --logdir_spec=run1:<path1>,run2:<path2>,run3:<path3>
```

Example:
```bash
tensorboard --logdir_spec=baseline:finetune_models/run1/training,optimized:finetune_models/run2/training
```

### Reload Interval

Auto-refresh data every N seconds:
```bash
tensorboard --logdir=<path> --reload_interval=30
```

### Bind to Specific Interface

```bash
tensorboard --logdir=<path> --host=localhost  # Local only
tensorboard --logdir=<path> --host=0.0.0.0    # All interfaces
```

---

## 🔍 Monitoring Best Practices

### Key Metrics to Watch

**Every 100 Steps:**
1. Check `train/loss_total` - Should decrease smoothly
2. Check `train/grad_norm` - Should stay < 10
3. Check `train/lr` - Should match expected schedule

**Every 1000 Steps:**
1. Listen to audio samples - Quality should improve
2. Check `eval/loss_total` - Should decrease with train loss
3. Verify no NaN losses

**At LR Drops (50k, 150k, 300k):**
1. You should see small bumps in loss (normal)
2. Loss should then continue decreasing
3. LR value should halve

### Warning Signs

🚨 **Immediate Action Required:**
- `train/loss` = NaN → Stop training, check dataset
- `train/grad_norm` > 100 → Gradient explosion, reduce LR
- `eval/loss` increasing while `train/loss` decreasing → Overfitting

⚠️ **Monitor Closely:**
- Loss plateaus before 50k steps → May need more data
- Oscillating loss → Try reducing learning rate
- Very slow decrease → Increase LR slightly

---

## 🐛 Troubleshooting

### "TensorBoard not found"

Install TensorBoard:
```bash
pip install tensorboard
```

### "No dashboards are active"

**Problem**: TensorBoard can't find training logs

**Solutions:**
1. Make sure training has started and created log files
2. Check the path is correct: `ls <logdir>` should show event files
3. Wait a few minutes after training starts

### Can't Access from Browser

**Check if TensorBoard is running:**
```bash
# Should show TensorBoard listening on port 6006
netstat -an | grep 6006
```

**Try different URL:**
- http://localhost:6006
- http://127.0.0.1:6006
- http://0.0.0.0:6006

### Port Already in Use

```bash
# Find process using port 6006
netstat -ano | findstr :6006  # Windows
lsof -i :6006                 # Linux/Mac

# Kill process or use different port
tensorboard --logdir=<path> --port=6007
```

### Stale Data / Not Updating

```bash
# Force reload
tensorboard --logdir=<path> --reload_interval=10

# Or clear cache
rm -rf /tmp/.tensorboard-info/
tensorboard --logdir=<path>
```

---

## 📊 Example TensorBoard Session

### Starting TensorBoard

```bash
$ tensorboard --logdir=finetune_models/run/training

Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --host=0.0.0.0
TensorBoard 2.15.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

### What You'll See

**Browser: http://localhost:6006**

**SCALARS Tab:**
```
train/loss_total
  Step 0:    7.08
  Step 50:   6.92
  Step 100:  6.78
  ...
  Step 17000: 3.45  ← Your current progress

train/lr
  Step 0-50000:    5e-06
  Step 50000-150000: 2.5e-06  ← Should drop here
  Step 150000+:    1.25e-06
```

**AUDIO Tab:**
```
Generated Samples:
  - sample_0_step_1000.wav
  - sample_0_step_5000.wav
  - sample_0_step_10000.wav
  - sample_0_step_17000.wav  ← Latest
```

---

## 🎯 Quick Reference Card

| Task | Command |
|------|---------|
| **Start TensorBoard** | `tensorboard --logdir=<path>/run/training` |
| **Access in Browser** | http://localhost:6006 |
| **Different Port** | `tensorboard --logdir=<path> --port=6007` |
| **Auto-refresh** | `tensorboard --logdir=<path> --reload_interval=30` |
| **Compare Runs** | `tensorboard --logdir_spec=name1:<path1>,name2:<path2>` |
| **Stop TensorBoard** | Press `Ctrl+C` in terminal |

---

## 🚀 Pro Tips

### 1. Keep TensorBoard Running

Start TensorBoard in a **separate terminal** before training:
- It will automatically detect new log files
- No need to restart when resuming training
- Can monitor multiple training sessions

### 2. Bookmark Important Views

TensorBoard URLs are bookmarkable:
```
http://localhost:6006/#scalars&run=GPT_XTTS_FT-October-23-2025_07+57AM-cd6fafc
```

### 3. Download Data

You can download metrics as CSV:
1. Click on a metric chart
2. Click the download icon
3. Use for custom analysis

### 4. Mobile Monitoring

If you set up SSH tunnel or public access, you can monitor training on your phone!

### 5. Compare Checkpoints

To compare different checkpoints from the same run:
```bash
tensorboard --logdir=finetune_models/run/training --reload_multifile=true
```

---

## 📱 Access from Your Logs

Based on your training logs, **your exact command is:**

```bash
tensorboard --logdir=/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/finetune_models/run/training
```

**This directory contains:**
- Event files for TensorBoard
- Your training run: `GPT_XTTS_FT-October-23-2025_07+57AM-cd6fafc`
- All metrics, audio samples, and graphs

**Just run that command in a new terminal and open http://localhost:6006!** 🎉

---

## 📚 Related Documentation

- **Training Guide**: `README.md`
- **Checkpoint Guide**: `CHECKPOINT_GUIDE.md`
- **Optimizations**: `XTTS_V2_OPTIMIZATIONS_QUICKSTART.md`
- **Troubleshooting**: `TROUBLESHOOTING.md`

---

**TensorBoard Version**: Automatically installed with PyTorch/TensorFlow  
**Default Port**: 6006  
**Log Location**: `<output_path>/run/training/`  
**Update Frequency**: Real-time (checks every ~30 seconds)
