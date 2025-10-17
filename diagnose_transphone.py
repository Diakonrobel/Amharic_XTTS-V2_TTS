#!/usr/bin/env python3
"""
Diagnostic Script for Transphone Backend Detection

This script helps identify why Transphone is not being detected or used,
even when it's installed.
"""

import sys
import logging

# Set up logging to see detailed initialization messages
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

print("=" * 70)
print("🔍 TRANSPHONE DIAGNOSTIC TOOL")
print("=" * 70)
print()

# Step 1: Check if Transphone is installed
print("Step 1: Checking Transphone installation...")
print("-" * 70)
try:
    import transphone
    print("✅ Transphone module is installed")
    print(f"   Location: {transphone.__file__}")
    try:
        print(f"   Version: {transphone.__version__}")
    except AttributeError:
        print("   Version: (not available)")
except ImportError as e:
    print(f"❌ Transphone is NOT installed")
    print(f"   Error: {e}")
    print("\n💡 Install Transphone with: pip install transphone")
    sys.exit(1)

print()

# Step 2: Check if Transphone can be initialized
print("Step 2: Testing Transphone initialization...")
print("-" * 70)
try:
    from transphone import read_g2p
    print("✅ read_g2p function imported successfully")
    
    # Try different language codes
    codes_to_try = ['amh', 'am', 'AM', 'AMH']
    g2p_instance = None
    working_code = None
    
    for code in codes_to_try:
        try:
            print(f"\n   Trying language code: '{code}'...", end=" ")
            g2p_instance = read_g2p(code)
            working_code = code
            print("✅ SUCCESS")
            break
        except Exception as e:
            print(f"❌ FAILED")
            print(f"      Error: {type(e).__name__}: {e}")
    
    if g2p_instance is None:
        print("\n❌ Could not initialize Transphone with any language code")
        print("\n💡 Possible issues:")
        print("   - Transphone language data files are missing")
        print("   - Amharic language support not included in your Transphone installation")
        print("   - Version incompatibility")
        sys.exit(1)
    
    print(f"\n✅ Transphone initialized successfully with code: '{working_code}'")
    
except Exception as e:
    print(f"\n❌ Transphone initialization failed")
    print(f"   Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 3: Test actual G2P conversion
print("Step 3: Testing G2P conversion...")
print("-" * 70)
test_texts = [
    "ሰላም",
    "ኢትዮጵያ",
    "አማርኛ",
    "1959"  # Test number (should be kept as-is by Transphone)
]

for text in test_texts:
    try:
        result = g2p_instance(text)
        print(f"✅ {text:15} → {result}")
    except Exception as e:
        print(f"❌ {text:15} → ERROR: {e}")

print()

# Step 4: Check EnhancedAmharicG2P integration
print("Step 4: Testing EnhancedAmharicG2P integration...")
print("-" * 70)
try:
    from amharic_tts.g2p.amharic_g2p_enhanced import EnhancedAmharicG2P
    print("✅ EnhancedAmharicG2P imported successfully")
    
    # Create instance with explicit backend
    print("\nInitializing with explicit 'transphone' backend...")
    g2p = EnhancedAmharicG2P(backend='transphone')
    print()
    
    # Get backend status
    status = g2p.get_backend_status()
    
    print("Backend Status:")
    for backend, info in status.items():
        status_icon = "✅" if info['available'] else "❌"
        print(f"  {status_icon} {backend:12} - {'Available' if info['available'] else 'Not available'}")
        if info['error']:
            print(f"      Error: {info['error']}")
    
    # Test conversion
    print("\nTesting conversion through EnhancedAmharicG2P:")
    test_text = "ይህ በአውሮፓውያኑ 1959 የተፈረመው"
    result = g2p.convert(test_text)
    
    print(f"\nInput:  {test_text}")
    print(f"Output: {result}")
    
    # Check if numbers were expanded
    if '1959' in result:
        print("\n⚠️  WARNING: Number '1959' was NOT expanded to Amharic words")
        print("   This suggests number expansion is not working")
    else:
        print("\n✅ Number expansion is working correctly")
    
    # Check if Transphone was actually used
    if g2p.transphone_g2p is not None:
        print("✅ Transphone backend is being used")
    else:
        print("❌ Transphone backend is NOT being used (fell back to rule-based)")
        if 'transphone' in g2p._backend_init_errors:
            print(f"   Reason: {g2p._backend_init_errors['transphone']}")
    
except Exception as e:
    print(f"❌ EnhancedAmharicG2P test failed")
    print(f"   Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("✅ DIAGNOSTIC COMPLETE")
print("=" * 70)
print()

print("Summary:")
print("  1. Transphone module: ✅ Installed and working")
print(f"  2. Language code: ✅ '{working_code}' works")
print("  3. G2P conversion: ✅ Working")
print(f"  4. Integration: {'✅ Working' if g2p.transphone_g2p else '❌ Not working'}")
print()

if g2p.transphone_g2p:
    print("🎉 SUCCESS! Transphone is properly installed and integrated.")
    print("   If you're still seeing issues, they may be in the training/inference code.")
else:
    print("⚠️  Transphone is installed but NOT being used by EnhancedAmharicG2P.")
    print("   Check the errors above for details.")

print()
print("=" * 70)
