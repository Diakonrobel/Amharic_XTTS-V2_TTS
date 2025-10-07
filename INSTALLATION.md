# 🚀 Installation Guide

Complete installation guide for Amharic XTTS Fine-Tuning WebUI across multiple platforms.

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Advanced Installation](#advanced-installation)
- [Troubleshooting](#troubleshooting)
- [Manual Installation](#manual-installation)

---

## 🎯 Quick Start

### Prerequisites

- **Python 3.8-3.11** (Python 3.12+ may have compatibility issues)
- **Git** (for cloning the repository)
- **CUDA Toolkit** (optional, for NVIDIA GPU support)

### One-Command Installation

```bash
# Clone the repository
git clone https://github.com/Diakonrobel/Amharic_XTTS-V2_TTS.git
cd Amharic_XTTS-V2_TTS

# Run smart installer
python smart_install.py
```

**That's it!** The installer will:
- ✅ Detect your operating system
- ✅ Detect your GPU (CUDA/ROCm/Metal/CPU)
- ✅ Install appropriate PyTorch version
- ✅ Install all dependencies
- ✅ Create launch scripts
- ✅ Install to your current Python environment (no venv needed)

---

## 💻 Platform-Specific Instructions

### Windows

#### Method 1: Smart Installer (Recommended)
```cmd
# Using Command Prompt or PowerShell
python smart_install.py
```

#### Method 2: Batch Script
```cmd
# Double-click install.bat or run:
install.bat
```

#### After Installation:
```cmd
# Launch the application
launch.bat

# Or manually:
python xtts_demo.py
```

---

### Linux

#### Method 1: Smart Installer (Recommended)
```bash
python3 smart_install.py
```

#### Method 2: Shell Script
```bash
chmod +x install.sh
./install.sh
```

#### After Installation:
```bash
# Launch the application
./launch.sh

# Or manually:
python3 xtts_demo.py
```

---

### macOS

#### Intel Mac
```bash
python3 smart_install.py
```

#### Apple Silicon (M1/M2/M3)
```bash
# The installer will automatically detect Apple Silicon
# and install appropriate Metal-optimized PyTorch
python3 smart_install.py
```

#### After Installation:
```bash
# Launch the application
./launch.sh

# Or manually:
python3 xtts_demo.py
```

---

## 🔧 Advanced Installation

### Installation Options

The smart installer supports various options:

```bash
# Show all options
python smart_install.py --help

# Install with optional G2P backends (Transphone, Epitran)
python smart_install.py --with-backends

# Force CPU-only installation (no GPU)
python smart_install.py --cpu-only

# Specify CUDA version (if not auto-detected correctly)
python smart_install.py --cuda-version 11.8

# Skip PyTorch installation (if already installed)
python smart_install.py --skip-torch

# Run tests after installation
python smart_install.py --test

# Combine options
python smart_install.py --with-backends --test
```

---

### GPU-Specific Installation

#### NVIDIA GPU (CUDA)

**CUDA 11.8 (Recommended for RTX 20/30/40 series):**
```bash
python smart_install.py --cuda-version 11.8
```

**CUDA 12.1 (For newer GPUs):**
```bash
python smart_install.py --cuda-version 12.1
```

**Auto-detect (Default):**
```bash
python smart_install.py
# Installer will detect your CUDA version automatically
```

---

#### AMD GPU (ROCm) - Linux Only

```bash
# ROCm support is automatically detected on Linux
python smart_install.py

# Or force CPU if ROCm detection fails
python smart_install.py --cpu-only
```

---

#### Apple Silicon (M1/M2/M3)

```bash
# Metal support is automatically detected
python smart_install.py

# This will install PyTorch with MPS (Metal Performance Shaders) support
```

---

#### CPU Only

```bash
# For systems without GPU or for testing
python smart_install.py --cpu-only
```

---

## 🧪 Verify Installation

### Run Tests

```bash
# Install with tests
python smart_install.py --test

# Or test after installation
python -c "from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P; g2p = AmharicG2P(); print('✅ Installation verified!')"
```

### Check GPU Availability

```bash
# Check PyTorch GPU support
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Test G2P Backends

```bash
python -c "from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P; g2p = AmharicG2P(); print(f'Available backends: {g2p.get_available_backends()}')"
```

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: "Python version not supported"

**Solution:**
```bash
# Check Python version
python --version

# Use Python 3.8-3.11
# Install from: https://www.python.org/downloads/
```

---

#### Issue: "CUDA not detected"

**Solutions:**

1. **Install NVIDIA Drivers:**
   - Windows: [NVIDIA Driver Download](https://www.nvidia.com/Download/index.aspx)
   - Linux: `sudo apt install nvidia-driver-535` (or latest)

2. **Install CUDA Toolkit:**
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

3. **Verify Installation:**
   ```bash
   nvidia-smi
   ```

4. **Force CPU Installation:**
   ```bash
   python smart_install.py --cpu-only
   ```

---

#### Issue: "pip install failed"

**Solutions:**

1. **Upgrade pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Use system pip:**
   ```bash
   python -m pip install --user [package]
   ```

3. **Check internet connection**

4. **Use alternative mirrors:**
   ```bash
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple [package]
   ```

---

#### Issue: "Import errors after installation"

**Solutions:**

1. **Reinstall dependencies:**
   ```bash
   pip install --force-reinstall --no-cache-dir [package]
   ```

2. **Check Python path:**
   ```bash
   which python  # Linux/macOS
   where python  # Windows
   ```

3. **Run in the correct directory:**
   ```bash
   cd Amharic_XTTS-V2_TTS
   python xtts_demo.py
   ```

---

#### Issue: "Out of memory (CUDA)"

**Solutions:**

1. **Reduce batch size** in training settings

2. **Use gradient checkpointing**

3. **Close other GPU applications**

4. **Use CPU fallback:**
   ```bash
   python smart_install.py --cpu-only
   ```

---

#### Issue: "ModuleNotFoundError: No module named 'amharic_tts'"

**Solution:**
```bash
# Make sure you're in the project directory
cd Amharic_XTTS-V2_TTS

# Install in development mode
pip install -e .
```

---

### Platform-Specific Issues

#### Windows: "Microsoft Visual C++ required"

**Solution:**
- Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

---

#### Linux: "libsndfile not found"

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install libsndfile1

# Fedora/RHEL
sudo dnf install libsndfile

# Arch
sudo pacman -S libsndfile
```

---

#### macOS: "Command not found: brew"

**Solution:**
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Then install dependencies
brew install portaudio
```

---

## 📦 Manual Installation

If the smart installer doesn't work for your system:

### Step 1: Install PyTorch

**NVIDIA GPU (CUDA 11.8):**
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

**CPU Only:**
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
```

**Apple Silicon:**
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
```

---

### Step 2: Install Base Requirements

```bash
pip install faster_whisper==1.0.3
pip install gradio==4.44.1
pip install spacy==3.7.5
pip install coqui-tts[languages]==0.24.2
pip install cutlet==0.5.0
pip install "fugashi[unidic-lite]==1.4.0"
pip install fastapi==0.103.1
pip install pydantic==2.3.0
```

---

### Step 3: Install Optional Backends (Optional)

```bash
# Transphone
pip install --no-deps transphone panphon phonepiece
pip install --no-deps unicodecsv PyYAML regex editdistance munkres

# Epitran
pip install --no-deps epitran marisa-trie requests jamo ipapy iso-639
pip install charset-normalizer idna urllib3 certifi

# Compatibility
pip install importlib-resources zipp
```

---

### Step 4: Verify Installation

```bash
python -c "from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P; print('✅ Success!')"
```

---

## 🌐 Cloud/Colab Installation

For Google Colab or cloud environments, use the provided notebook:

```bash
# Use the comprehensive Colab notebook
colab_amharic_xtts_complete.ipynb
```

This notebook includes:
- ✅ Automatic environment setup
- ✅ GPU detection
- ✅ G2P backend installation
- ✅ WebUI launcher
- ✅ Model download helpers

---

## 📚 Additional Resources

- **Main Documentation:** [README.md](README.md)
- **G2P Backends Guide:** [OPTIONAL_BACKENDS_INSTALL.md](OPTIONAL_BACKENDS_INSTALL.md)
- **Git LFS Workflow:** [LFS_WORKFLOW_GUIDE.md](LFS_WORKFLOW_GUIDE.md)
- **Detailed G2P Docs:** [docs/](docs/)

---

## 💡 Tips

1. **Use the smart installer** - It handles most edge cases automatically
2. **Install optional backends** for better G2P quality
3. **Test with `--test` flag** to verify everything works
4. **Use GPU** for faster training (but CPU works fine too)
5. **Read error messages carefully** - they usually indicate the solution

---

## 🆘 Getting Help

If you encounter issues:

1. **Check this guide** for common solutions
2. **Run with `--test` flag** to identify problems
3. **Check GitHub Issues:** [Report a bug](https://github.com/Diakonrobel/Amharic_XTTS-V2_TTS/issues)
4. **Provide details:**
   - Operating system
   - Python version (`python --version`)
   - GPU type (if applicable)
   - Full error message
   - Installation command used

---

## ⭐ Success!

Once installed, you're ready to:

1. **Launch the WebUI:**
   ```bash
   python xtts_demo.py
   ```

2. **Start training** Amharic TTS models

3. **Use multiple G2P backends** for best quality

4. **Fine-tune** with your own voice data

---

**Happy fine-tuning! 🎤🇪🇹**
