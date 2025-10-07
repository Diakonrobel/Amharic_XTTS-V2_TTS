"""
Automatic Dependency Installer for Amharic TTS

This module handles automatic installation of optional dependencies
for G2P backends (Transphone, Epitran) with user confirmation.
"""

import subprocess
import sys
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def check_package_installed(package_name: str) -> bool:
    """
    Check if a Python package is installed
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        True if installed, False otherwise
    """
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def install_package(
    package_name: str,
    display_name: Optional[str] = None,
    auto_install: bool = False
) -> Tuple[bool, str]:
    """
    Install a Python package with user confirmation
    
    Args:
        package_name: Name of the package to install (for pip)
        display_name: Display name for the package (optional)
        auto_install: If True, install without asking
        
    Returns:
        Tuple of (success, message)
    """
    if display_name is None:
        display_name = package_name
    
    # Check if already installed
    if check_package_installed(package_name):
        return True, f"{display_name} is already installed"
    
    # Ask user for confirmation unless auto_install is True
    if not auto_install:
        print("\n" + "=" * 70)
        print(f"📦 Optional Dependency: {display_name}")
        print("=" * 70)
        print(f"\n{display_name} is not installed but recommended for best quality.")
        print(f"\nWould you like to install it now? (y/n): ", end="")
        
        try:
            response = input().strip().lower()
            if response not in ['y', 'yes']:
                return False, f"User declined to install {display_name}"
        except (EOFError, KeyboardInterrupt):
            print("\nSkipping installation")
            return False, f"User declined to install {display_name}"
    
    # Attempt installation
    print(f"\n🔧 Installing {display_name}...")
    print(f"   Running: pip install {package_name}")
    
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"✅ {display_name} installed successfully!")
        return True, f"{display_name} installed successfully"
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to install {display_name}: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error installing {display_name}: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return False, error_msg


def ensure_transphone_installed(auto_install: bool = False) -> bool:
    """
    Ensure Transphone is installed, offering to install if not
    
    Args:
        auto_install: If True, install without asking
        
    Returns:
        True if Transphone is available, False otherwise
    """
    success, message = install_package(
        package_name="transphone",
        display_name="Transphone (State-of-the-art G2P)",
        auto_install=auto_install
    )
    
    if success:
        logger.info(message)
    else:
        logger.warning(message)
    
    return success


def ensure_epitran_installed(auto_install: bool = False) -> bool:
    """
    Ensure Epitran is installed, offering to install if not
    
    Args:
        auto_install: If True, install without asking
        
    Returns:
        True if Epitran is available, False otherwise
    """
    success, message = install_package(
        package_name="epitran",
        display_name="Epitran (Multilingual G2P)",
        auto_install=auto_install
    )
    
    if success:
        logger.info(message)
    else:
        logger.warning(message)
    
    return success


def ensure_g2p_backends(
    install_transphone: bool = True,
    install_epitran: bool = False,
    auto_install: bool = False
) -> dict:
    """
    Ensure G2P backends are available
    
    Args:
        install_transphone: Whether to check/install Transphone
        install_epitran: Whether to check/install Epitran
        auto_install: If True, install without asking
        
    Returns:
        Dictionary with backend availability status
    """
    results = {
        'transphone': False,
        'epitran': False,
        'rule_based': True  # Always available
    }
    
    if install_transphone:
        results['transphone'] = ensure_transphone_installed(auto_install)
    
    if install_epitran:
        results['epitran'] = ensure_epitran_installed(auto_install)
    
    return results


def print_installation_summary(results: dict):
    """
    Print a summary of backend availability
    
    Args:
        results: Dictionary with backend status
    """
    print("\n" + "=" * 70)
    print("📊 G2P Backend Status")
    print("=" * 70)
    
    backends = [
        ('Transphone (Primary)', 'transphone', '⭐ Best accuracy'),
        ('Epitran (Backup)', 'epitran', 'Good accuracy'),
        ('Rule-Based (Fallback)', 'rule_based', 'Always available')
    ]
    
    for name, key, description in backends:
        status = "✅ Available" if results.get(key, False) else "❌ Not available"
        print(f"{name:30} {status:20} {description}")
    
    print("=" * 70)
    
    # Recommendation
    if not results.get('transphone', False):
        print("\n💡 Recommendation:")
        print("   Install Transphone for best quality: pip install transphone")
    
    print()


def setup_wizard(auto_install: bool = False):
    """
    Interactive setup wizard for G2P backends
    
    Args:
        auto_install: If True, install all without asking
    """
    print("\n" + "=" * 70)
    print("🚀 Amharic TTS - G2P Backend Setup Wizard")
    print("=" * 70)
    print("\nThis wizard will help you set up the best G2P backends for Amharic.")
    print("\nRecommended backends:")
    print("  1. Transphone - State-of-the-art (⭐ Highly Recommended)")
    print("  2. Epitran - Good backup (Optional)")
    print("  3. Rule-Based - Always available (Built-in)")
    print()
    
    # Check current status
    transphone_installed = check_package_installed('transphone')
    epitran_installed = check_package_installed('epitran')
    
    print("Current status:")
    print(f"  Transphone: {'✅ Installed' if transphone_installed else '❌ Not installed'}")
    print(f"  Epitran:    {'✅ Installed' if epitran_installed else '❌ Not installed'}")
    print(f"  Rule-Based: ✅ Always available")
    print()
    
    if auto_install:
        print("🔧 Auto-install mode enabled. Installing missing dependencies...\n")
        install_transphone = not transphone_installed
        install_epitran = not epitran_installed
    else:
        # Ask user what to install
        if not transphone_installed:
            print("Install Transphone (Highly Recommended)? (y/n): ", end="")
            try:
                response = input().strip().lower()
                install_transphone = response in ['y', 'yes']
            except (EOFError, KeyboardInterrupt):
                print("\nSetup cancelled")
                return
        else:
            install_transphone = False
        
        if not epitran_installed:
            print("Install Epitran (Optional backup)? (y/n): ", end="")
            try:
                response = input().strip().lower()
                install_epitran = response in ['y', 'yes']
            except (EOFError, KeyboardInterrupt):
                print("\nSetup cancelled")
                return
        else:
            install_epitran = False
    
    # Install selected backends
    results = ensure_g2p_backends(
        install_transphone=install_transphone,
        install_epitran=install_epitran,
        auto_install=auto_install
    )
    
    # Print summary
    print_installation_summary(results)
    
    print("✅ Setup complete! You can now use the Amharic G2P system.\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Install Amharic TTS G2P dependencies")
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Automatically install all dependencies without prompting'
    )
    parser.add_argument(
        '--transphone-only',
        action='store_true',
        help='Only install Transphone (recommended)'
    )
    
    args = parser.parse_args()
    
    if args.transphone_only:
        print("Installing Transphone only...\n")
        ensure_transphone_installed(auto_install=args.auto)
    else:
        setup_wizard(auto_install=args.auto)
