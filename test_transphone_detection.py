#!/usr/bin/env python3
"""
Test script to diagnose why Transphone is not being detected
"""

import sys
import traceback

print("=" * 80)
print("🔍 Transphone Detection Diagnostic")
print("=" * 80)

# Test 1: Can we import transphone?
print("\n1️⃣ Test Import")
try:
    import transphone
    print("   ✅ transphone module found")
    print(f"   📍 Location: {transphone.__file__}")
    print(f"   📦 Version: {getattr(transphone, '__version__', 'unknown')}")
except ImportError as e:
    print(f"   ❌ Cannot import transphone: {e}")
    sys.exit(1)

# Test 2: Can we import read_g2p?
print("\n2️⃣ Test read_g2p Function")
try:
    from transphone import read_g2p
    print("   ✅ read_g2p function imported")
except ImportError as e:
    print(f"   ❌ Cannot import read_g2p: {e}")
    sys.exit(1)

# Test 3: What languages are available?
print("\n3️⃣ Test Available Languages")
try:
    # Try different language codes
    test_codes = ['amh', 'am', 'amharic', 'AM', 'AMH']
    
    for code in test_codes:
        print(f"\n   Testing code: '{code}'")
        try:
            g2p = read_g2p(code)
            print(f"      ✅ '{code}' works!")
            
            # Test conversion
            test_text = "ሰላም"
            result = g2p(test_text)
            print(f"      📝 Test: '{test_text}' → '{result}'")
            
            # Success! Use this code
            print(f"\n   🎯 Working language code: '{code}'")
            break
            
        except Exception as e:
            print(f"      ❌ '{code}' failed: {type(e).__name__}: {e}")
    
except Exception as e:
    print(f"   ❌ Error testing languages: {e}")
    traceback.print_exc()

# Test 4: Try loading with error details
print("\n4️⃣ Detailed Load Test")
try:
    print("   Attempting: read_g2p('amh')")
    g2p = read_g2p('amh')
    print("   ✅ Successfully loaded Transphone for Amharic!")
    
    # Test with actual Amharic text
    print("\n   Testing with Amharic text:")
    test_texts = [
        "ሰላም ዓለም",
        "ኢትዮጵያ አማርኛ",
        "ሃሎ"
    ]
    
    for text in test_texts:
        try:
            result = g2p(text)
            print(f"      ✓ '{text}' → '{result}'")
        except Exception as e:
            print(f"      ✗ '{text}' failed: {e}")
            
except Exception as e:
    print(f"   ❌ Failed to load: {type(e).__name__}")
    print(f"   📝 Error message: {e}")
    traceback.print_exc()

# Test 5: Check if there's a model download issue
print("\n5️⃣ Check Model Files")
try:
    import os
    home = os.path.expanduser("~")
    transphone_cache = os.path.join(home, ".cache", "transphone")
    
    print(f"   📂 Cache location: {transphone_cache}")
    
    if os.path.exists(transphone_cache):
        print("   ✅ Cache directory exists")
        files = os.listdir(transphone_cache)
        print(f"   📁 Files: {len(files)} items")
        for f in files[:10]:  # Show first 10
            print(f"      - {f}")
    else:
        print("   ⚠️  Cache directory not found")
        print("   💡 Transphone may need to download models on first use")
        
except Exception as e:
    print(f"   ⚠️  Could not check cache: {e}")

print("\n" + "=" * 80)
print("🏁 Diagnostic Complete")
print("=" * 80)
