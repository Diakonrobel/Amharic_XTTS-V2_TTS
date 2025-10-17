#!/usr/bin/env python3
"""Quick fix for Transphone installation issues"""

import subprocess
import sys
import os

def run_cmd(cmd):
    """Run command and show output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠️  Warning: {result.stderr}")
    return result.returncode == 0

def check_conflicting_installations():
    """Check for conflicting local installations"""
    print("\n🔍 Checking for conflicting installations...")
    
    # Check common paths
    conflict_paths = [
        "D:\\CHATTERBOX-FINETUNE\\chatterbox\\epitran",
        os.path.expanduser("~/chatterbox/epitran"),
    ]
    
    found_conflicts = []
    for path in conflict_paths:
        if os.path.exists(path):
            found_conflicts.append(path)
            print(f"  ⚠️  Found conflicting installation: {path}")
    
    if found_conflicts:
        print("\n❌ Conflicting installations detected!")
        print("Please remove them manually:")
        for path in found_conflicts:
            if sys.platform == "win32":
                print(f'  Remove-Item -Recurse -Force "{path}"')
            else:
                print(f'  rm -rf "{path}"')
        return False
    else:
        print("  ✅ No conflicts found")
        return True

def main():
    print("=" * 70)
    print("🔧 Transphone Installation Fix")
    print("=" * 70)
    
    # Check for conflicts first
    if not check_conflicting_installations():
        print("\n⚠️  Please remove conflicting installations first, then re-run this script.")
        return
    
    # Uninstall potentially broken packages
    print("\n1️⃣  Removing old installations...")
    run_cmd(f"{sys.executable} -m pip uninstall -y transphone epitran panphon")
    
    # Reinstall with clean cache
    print("\n2️⃣  Reinstalling packages...")
    success = True
    success &= run_cmd(f"{sys.executable} -m pip install --no-cache-dir epitran")
    success &= run_cmd(f"{sys.executable} -m pip install --no-cache-dir panphon")
    success &= run_cmd(f"{sys.executable} -m pip install --no-cache-dir transphone")
    
    if not success:
        print("\n❌ Installation failed. Try creating a new virtual environment.")
        return
    
    # Verify
    print("\n3️⃣  Verifying installation...")
    try:
        from transphone import read_g2p
        print("  ✅ Transphone module imported")
        
        g2p = read_g2p('amh')
        print("  ✅ Amharic G2P initialized")
        
        result = g2p('ሰላም')
        print(f"  ✅ Test conversion: ሰላም → {result}")
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS! Transphone is now working correctly")
        print("=" * 70)
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check if you have conflicting installations in PYTHONPATH")
        print("2. Try creating a new virtual environment")
        print("3. Verify Python version (3.8+ required)")
        
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        print("\nThe package installed but failed to load Amharic.")
        print("This is usually a data file issue.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
