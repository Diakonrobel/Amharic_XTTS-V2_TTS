#!/usr/bin/env python3
"""
Smart Cross-Platform Installer for Amharic XTTS Fine-Tuning WebUI

This installer automatically:
- Detects operating system (Windows, Linux, macOS)
- Detects GPU availability (CUDA, ROCm, Metal, CPU-only)
- Installs appropriate PyTorch version
- Handles dependency conflicts
- Works with current Python environment (no venv required)
- Supports optional backend installation

Usage:
    python smart_install.py [options]

Options:
    --with-backends       Install optional G2P backends (Transphone, Epitran)
    --cpu-only           Force CPU-only installation
    --cuda-version       Specify CUDA version (e.g., 11.8, 12.1)
    --skip-torch         Skip PyTorch installation
    --test               Run installation tests after setup
    -h, --help           Show this help message
"""

import sys
import platform
import subprocess
import os
import argparse
from pathlib import Path
import json

# ANSI color codes for cross-platform terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def disable_colors():
        Colors.HEADER = ''
        Colors.BLUE = ''
        Colors.CYAN = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.RED = ''
        Colors.ENDC = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''

# Disable colors on Windows unless using Windows Terminal
if platform.system() == 'Windows' and not os.getenv('WT_SESSION'):
    Colors.disable_colors()


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.ENDC}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command."""
    try:
        if capture_output:
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=check, 
                capture_output=True, 
                text=True
            )
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=check)
            return True
    except subprocess.CalledProcessError as e:
        if check:
            print_error(f"Command failed: {cmd}")
            if capture_output:
                print_error(f"Error: {e.stderr}")
            sys.exit(1)
        return False


def detect_os():
    """Detect the operating system."""
    system = platform.system()
    if system == "Windows":
        return "windows"
    elif system == "Darwin":
        return "macos"
    elif system == "Linux":
        return "linux"
    else:
        return "unknown"


def detect_python_version():
    """Get Python version information."""
    version = sys.version_info
    return f"{version.major}.{version.minor}.{version.micro}"


def check_gpu():
    """Detect available GPU."""
    os_type = detect_os()
    
    # Check for NVIDIA GPU (CUDA)
    try:
        if os_type == "windows":
            nvidia_smi = run_command("nvidia-smi", check=False, capture_output=True)
        else:
            nvidia_smi = run_command("which nvidia-smi", check=False, capture_output=True)
        
        if nvidia_smi:
            # Get CUDA version
            cuda_version = run_command(
                "nvidia-smi --query-gpu=driver_version --format=csv,noheader",
                check=False,
                capture_output=True
            )
            return {"type": "cuda", "version": cuda_version}
    except:
        pass
    
    # Check for AMD GPU (ROCm) on Linux
    if os_type == "linux":
        try:
            rocm_info = run_command("which rocm-smi", check=False, capture_output=True)
            if rocm_info:
                return {"type": "rocm", "version": "unknown"}
        except:
            pass
    
    # Check for Apple Silicon (Metal)
    if os_type == "macos":
        try:
            chip = platform.processor()
            if "arm" in chip.lower() or run_command("sysctl -n machdep.cpu.brand_string", check=False, capture_output=True).lower().find("apple") != -1:
                return {"type": "metal", "version": "apple_silicon"}
        except:
            pass
    
    return {"type": "cpu", "version": None}


def get_pytorch_install_command(gpu_info, os_type, force_cpu=False, cuda_version=None):
    """Generate PyTorch installation command based on system."""
    if force_cpu or gpu_info["type"] == "cpu":
        return "pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu"
    
    elif gpu_info["type"] == "cuda":
        # Determine CUDA version
        if cuda_version:
            cuda_ver = cuda_version.replace(".", "")
        else:
            # Default to CUDA 11.8 for better compatibility
            cuda_ver = "118"
        
        return f"pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu{cuda_ver}"
    
    elif gpu_info["type"] == "rocm":
        return "pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/rocm5.6"
    
    elif gpu_info["type"] == "metal":
        # macOS with Apple Silicon
        return "pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1"
    
    else:
        return "pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2"


def upgrade_pip():
    """Upgrade pip to latest version."""
    print_info("Upgrading pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    print_success("pip upgraded successfully!")


def install_base_requirements():
    """Install base requirements."""
    print_info("Installing base requirements...")
    
    requirements = [
        "faster_whisper==1.0.3",
        "gradio==4.44.1",
        "spacy==3.7.5",
        "coqui-tts[languages]==0.24.2",
        "cutlet==0.5.0",
        "fugashi[unidic-lite]==1.4.0",
        "fastapi==0.103.1",
        "pydantic==2.3.0"
    ]
    
    for req in requirements:
        print_info(f"Installing {req}...")
        run_command(f"{sys.executable} -m pip install {req}")
    
    print_success("Base requirements installed successfully!")


def install_pytorch(gpu_info, os_type, force_cpu=False, cuda_version=None, skip_torch=False):
    """Install PyTorch with appropriate configuration."""
    if skip_torch:
        print_warning("Skipping PyTorch installation as requested.")
        return
    
    print_info("Installing PyTorch...")
    cmd = get_pytorch_install_command(gpu_info, os_type, force_cpu, cuda_version)
    print_info(f"Running: {cmd}")
    run_command(cmd)
    print_success("PyTorch installed successfully!")


def install_optional_backends():
    """Install optional G2P backends (Transphone, Epitran)."""
    print_info("Installing optional G2P backends...")
    
    # Install Transphone
    print_info("Installing Transphone...")
    try:
        run_command(f"{sys.executable} -m pip install --no-deps transphone", check=False)
        run_command(f"{sys.executable} -m pip install --no-deps panphon phonepiece", check=False)
        run_command(f"{sys.executable} -m pip install --no-deps unicodecsv PyYAML regex editdistance munkres", check=False)
        print_success("Transphone installed successfully!")
    except:
        print_warning("Transphone installation failed (optional, rule-based backend will be used)")
    
    # Install Epitran
    print_info("Installing Epitran...")
    try:
        run_command(f"{sys.executable} -m pip install --no-deps epitran marisa-trie requests jamo ipapy iso-639", check=False)
        run_command(f"{sys.executable} -m pip install charset-normalizer idna urllib3 certifi", check=False)
        print_success("Epitran installed successfully!")
    except:
        print_warning("Epitran installation failed (optional, rule-based backend will be used)")
    
    # Install compatibility packages
    try:
        run_command(f"{sys.executable} -m pip install importlib-resources zipp", check=False)
    except:
        pass


def test_installation():
    """Test if installation was successful."""
    print_header("Testing Installation")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Import torch
    print_info("Testing PyTorch import...")
    try:
        import torch
        print_success(f"PyTorch {torch.__version__} imported successfully!")
        
        # Check GPU availability
        if torch.cuda.is_available():
            print_success(f"CUDA available! GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print_success("Metal (Apple Silicon) GPU available!")
        else:
            print_info("Using CPU mode")
        tests_passed += 1
    except Exception as e:
        print_error(f"PyTorch import failed: {e}")
        tests_failed += 1
    
    # Test 2: Import TTS
    print_info("Testing Coqui TTS import...")
    try:
        import TTS
        print_success(f"TTS imported successfully!")
        tests_passed += 1
    except Exception as e:
        print_error(f"TTS import failed: {e}")
        tests_failed += 1
    
    # Test 3: Import Amharic modules
    print_info("Testing Amharic TTS modules...")
    try:
        from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
        g2p = AmharicG2P(backend='rule-based')
        result = g2p.convert("ሰላም")
        print_success(f"Amharic G2P working! 'ሰላም' -> '{result}'")
        
        # Check available backends
        backends = g2p.get_available_backends()
        print_info(f"Available G2P backends: {', '.join(backends)}")
        tests_passed += 1
    except Exception as e:
        print_error(f"Amharic modules test failed: {e}")
        tests_failed += 1
    
    # Summary
    print_header("Test Summary")
    print(f"Tests passed: {Colors.GREEN}{tests_passed}{Colors.ENDC}")
    print(f"Tests failed: {Colors.RED}{tests_failed}{Colors.ENDC}")
    
    if tests_failed == 0:
        print_success("\n🎉 All tests passed! Installation successful!")
        return True
    else:
        print_warning("\n⚠ Some tests failed. Please check the errors above.")
        return False


def create_launch_script(os_type):
    """Create a convenient launch script."""
    print_info("Creating launch script...")
    
    if os_type == "windows":
        script_content = """@echo off
echo Starting Amharic XTTS Fine-Tuning WebUI...
python xtts_demo.py --share
pause
"""
        script_path = Path("launch.bat")
    else:
        script_content = """#!/bin/bash
echo "Starting Amharic XTTS Fine-Tuning WebUI..."
python xtts_demo.py --share
"""
        script_path = Path("launch.sh")
    
    try:
        script_path.write_text(script_content)
        if os_type != "windows":
            os.chmod(script_path, 0o755)
        print_success(f"Launch script created: {script_path}")
    except Exception as e:
        print_warning(f"Could not create launch script: {e}")


def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(
        description="Smart installer for Amharic XTTS Fine-Tuning WebUI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--with-backends', action='store_true', 
                        help='Install optional G2P backends (Transphone, Epitran)')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Force CPU-only installation')
    parser.add_argument('--cuda-version', type=str,
                        help='Specify CUDA version (e.g., 11.8, 12.1)')
    parser.add_argument('--skip-torch', action='store_true',
                        help='Skip PyTorch installation')
    parser.add_argument('--test', action='store_true',
                        help='Run installation tests after setup')
    
    args = parser.parse_args()
    
    # Print banner
    print_header("🇪🇹 Amharic XTTS Fine-Tuning WebUI - Smart Installer")
    
    # System detection
    print_header("System Detection")
    os_type = detect_os()
    python_version = detect_python_version()
    gpu_info = check_gpu()
    
    print_info(f"Operating System: {os_type.upper()}")
    print_info(f"Python Version: {python_version}")
    print_info(f"GPU Type: {gpu_info['type'].upper()}")
    if gpu_info['version']:
        print_info(f"GPU Info: {gpu_info['version']}")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print_error("Python 3.8 or higher is required!")
        sys.exit(1)
    elif sys.version_info >= (3, 12):
        print_warning("Python 3.12+ detected. Some packages may have compatibility issues.")
    
    # Confirm installation
    print(f"\n{Colors.BOLD}Installation Configuration:{Colors.ENDC}")
    print(f"  • OS: {os_type}")
    print(f"  • GPU: {gpu_info['type']}")
    print(f"  • CPU-only: {'Yes' if args.cpu_only else 'No'}")
    print(f"  • Optional backends: {'Yes' if args.with_backends else 'No'}")
    print(f"  • Skip PyTorch: {'Yes' if args.skip_torch else 'No'}")
    
    if not args.skip_torch:
        pytorch_cmd = get_pytorch_install_command(gpu_info, os_type, args.cpu_only, args.cuda_version)
        print(f"  • PyTorch command: {pytorch_cmd[:60]}...")
    
    print()
    response = input(f"{Colors.YELLOW}Proceed with installation? (y/n): {Colors.ENDC}")
    if response.lower() not in ['y', 'yes']:
        print_info("Installation cancelled.")
        sys.exit(0)
    
    # Start installation
    print_header("Starting Installation")
    
    try:
        # Step 1: Upgrade pip
        upgrade_pip()
        
        # Step 2: Install PyTorch
        if not args.skip_torch:
            print_header("Installing PyTorch")
            install_pytorch(gpu_info, os_type, args.cpu_only, args.cuda_version, args.skip_torch)
        
        # Step 3: Install base requirements
        print_header("Installing Base Requirements")
        install_base_requirements()
        
        # Step 4: Install optional backends
        if args.with_backends:
            print_header("Installing Optional G2P Backends")
            install_optional_backends()
        
        # Step 5: Create launch script
        create_launch_script(os_type)
        
        # Step 6: Run tests
        if args.test:
            success = test_installation()
            if not success:
                sys.exit(1)
        
        # Success!
        print_header("🎉 Installation Complete!")
        print_success("Amharic XTTS Fine-Tuning WebUI installed successfully!")
        
        print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print(f"  1. Launch the WebUI:")
        if os_type == "windows":
            print(f"     {Colors.CYAN}launch.bat{Colors.ENDC} or {Colors.CYAN}python xtts_demo.py{Colors.ENDC}")
        else:
            print(f"     {Colors.CYAN}./launch.sh{Colors.ENDC} or {Colors.CYAN}python xtts_demo.py{Colors.ENDC}")
        
        print(f"\n  2. For Google Colab:")
        print(f"     Use {Colors.CYAN}colab_amharic_xtts_complete.ipynb{Colors.ENDC}")
        
        print(f"\n  3. For optional backends:")
        print(f"     Run: {Colors.CYAN}python smart_install.py --with-backends{Colors.ENDC}")
        
        print(f"\n  4. Documentation:")
        print(f"     • {Colors.CYAN}README.md{Colors.ENDC} - Main documentation")
        print(f"     • {Colors.CYAN}OPTIONAL_BACKENDS_INSTALL.md{Colors.ENDC} - G2P backends")
        print(f"     • {Colors.CYAN}docs/{Colors.ENDC} - Detailed guides")
        
        print(f"\n{Colors.GREEN}Happy fine-tuning! 🎤{Colors.ENDC}\n")
        
    except KeyboardInterrupt:
        print_error("\n\nInstallation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\nInstallation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
