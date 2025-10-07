#!/usr/bin/env python3
"""
Quick Setup Script for Amharic TTS G2P Backends

This script makes it easy to install the recommended G2P backends
for the best Amharic TTS quality.

Usage:
    # Interactive setup (recommended)
    python setup_amharic_g2p.py

    # Auto-install everything
    python setup_amharic_g2p.py --auto

    # Install only Transphone (recommended)
    python setup_amharic_g2p.py --transphone-only

    # Skip installation, just show status
    python setup_amharic_g2p.py --check-only
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amharic_tts.utils.dependency_installer import (
    setup_wizard,
    ensure_transphone_installed,
    check_package_installed,
    print_installation_summary
)


def check_backend_status():
    """Check and display current backend status"""
    results = {
        'transphone': check_package_installed('transphone'),
        'epitran': check_package_installed('epitran'),
        'rule_based': True
    }
    
    print_installation_summary(results)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Setup Amharic TTS G2P backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Interactive setup (recommended)
  %(prog)s --auto             # Auto-install all dependencies
  %(prog)s --transphone-only  # Install only Transphone
  %(prog)s --check-only       # Just check status
        """
    )
    
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Automatically install dependencies without prompting'
    )
    
    parser.add_argument(
        '--transphone-only',
        action='store_true',
        help='Only install Transphone (recommended for best quality)'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check backend status, do not install'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 70)
    print("🇪🇹  Amharic TTS - G2P Backend Setup")
    print("=" * 70)
    
    # Handle check-only mode
    if args.check_only:
        print("\n📊 Checking backend status...\n")
        check_backend_status()
        return 0
    
    # Handle transphone-only mode
    if args.transphone_only:
        print("\n📦 Installing Transphone (State-of-the-art G2P)...\n")
        success = ensure_transphone_installed(auto_install=args.auto)
        
        if success:
            print("\n✅ Transphone installed successfully!")
            print("\nYou can now use Amharic G2P with the best quality backend.")
        else:
            print("\n⚠️  Transphone installation was skipped.")
            print("The rule-based G2P backend will be used (still high quality).")
        
        print("\n📊 Final backend status:\n")
        check_backend_status()
        return 0 if success else 1
    
    # Run interactive setup wizard
    try:
        setup_wizard(auto_install=args.auto)
        return 0
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error during setup: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
