#!/usr/bin/env python3
"""Test correct usage of transphone.g2p.read_g2p()"""

import sys

print("=" * 70)
print("🧪 TESTING transphone.g2p.read_g2p()")
print("=" * 70)
print()

try:
    from transphone.g2p import read_g2p
    print("✅ Imported read_g2p from transphone.g2p")
    print()
    
    # Test 1: Call with model_name (as per signature)
    print("Test 1: Calling read_g2p(model_name='latest')...")
    print("-" * 70)
    try:
        g2p_obj = read_g2p(model_name='latest')
        print(f"✅ Success! Returned: {type(g2p_obj)}")
        print(f"   Object attributes: {[a for a in dir(g2p_obj) if not a.startswith('_')][:10]}")
        
        # Try to use it with Amharic
        print("\nTest 2: Trying to convert Amharic text...")
        print("-" * 70)
        test_text = "ሰላም"
        
        # Try calling directly
        if callable(g2p_obj):
            try:
                result = g2p_obj(test_text, src_lang='amh')
                print(f"✅ Direct call with src_lang: g2p_obj('{test_text}', src_lang='amh') → {result}")
            except Exception as e:
                print(f"❌ Direct call with src_lang failed: {type(e).__name__}: {e}")
                
                # Try without src_lang
                try:
                    result = g2p_obj(test_text)
                    print(f"✅ Direct call: g2p_obj('{test_text}') → {result}")
                except Exception as e2:
                    print(f"❌ Direct call failed: {type(e2).__name__}: {e2}")
        
        # Try common method names
        print("\nTest 3: Trying common methods...")
        print("-" * 70)
        for method_name in ['convert', 'inference', 'transform', 'transliterate', 'phonemize']:
            if hasattr(g2p_obj, method_name):
                method = getattr(g2p_obj, method_name)
                print(f"\n  Method found: {method_name}")
                
                # Try with language parameter
                for lang_param in ['lang', 'src_lang', 'language', 'lang_id']:
                    try:
                        result = method(test_text, **{lang_param: 'amh'})
                        print(f"    ✅ {method_name}('{test_text}', {lang_param}='amh') → {result}")
                        break
                    except Exception:
                        pass
                else:
                    # Try without language parameter
                    try:
                        result = method(test_text)
                        print(f"    ✅ {method_name}('{test_text}') → {result}")
                    except Exception as e:
                        print(f"    ❌ {method_name} failed: {type(e).__name__}")
        
        # Check if it's a G2P class instance
        print("\nTest 4: Checking object type and usage...")
        print("-" * 70)
        print(f"  Type: {type(g2p_obj)}")
        print(f"  Module: {type(g2p_obj).__module__}")
        print(f"  Class: {type(g2p_obj).__name__}")
        
        # List all methods
        methods = [m for m in dir(g2p_obj) if not m.startswith('_') and callable(getattr(g2p_obj, m))]
        print(f"  Available methods: {', '.join(methods)}")
        
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
