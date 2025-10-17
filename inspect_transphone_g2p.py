#!/usr/bin/env python3
"""Inspect transphone.g2p module to find correct API"""

import sys

print("=" * 70)
print("🔍 INSPECTING transphone.g2p MODULE")
print("=" * 70)
print()

try:
    from transphone import g2p
    print("✅ transphone.g2p imported successfully")
    print(f"   Location: {g2p.__file__}")
    print()
    
    # List all attributes in g2p module
    print("📋 Available attributes in transphone.g2p:")
    print("-" * 70)
    attrs = dir(g2p)
    for attr in sorted(attrs):
        if not attr.startswith('_'):
            obj = getattr(g2p, attr)
            obj_type = type(obj).__name__
            print(f"  {attr:30} ({obj_type})")
            
            # If it's a class, show its methods
            if isinstance(obj, type):
                methods = [m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m, None))]
                if methods:
                    print(f"    → Methods: {', '.join(methods[:8])}")
            
            # If it's a function, try to show signature
            elif callable(obj):
                try:
                    import inspect
                    sig = inspect.signature(obj)
                    print(f"    → Signature: {attr}{sig}")
                except:
                    pass
    
    print("\n" + "=" * 70)
    print("🧪 Testing common patterns...")
    print("=" * 70)
    
    # Test pattern 1: Look for classes
    classes = [attr for attr in attrs if not attr.startswith('_') and isinstance(getattr(g2p, attr), type)]
    if classes:
        print(f"\n✅ Found {len(classes)} classes:")
        for cls_name in classes:
            print(f"  • {cls_name}")
            cls = getattr(g2p, cls_name)
            
            # Try to instantiate with 'amh'
            try:
                instance = cls('amh')
                print(f"    ✅ Can instantiate: {cls_name}('amh')")
                
                # Try to convert text
                if hasattr(instance, '__call__'):
                    try:
                        result = instance('ሰላም')
                        print(f"    ✅ Can call: instance('ሰላም') → {result}")
                    except Exception as e:
                        print(f"    ❌ Call failed: {type(e).__name__}: {e}")
                        
                # Try common method names
                for method_name in ['convert', 'transliterate', 'transform', 'phonemize']:
                    if hasattr(instance, method_name):
                        method = getattr(instance, method_name)
                        try:
                            result = method('ሰላም')
                            print(f"    ✅ Method '{method_name}': instance.{method_name}('ሰላም') → {result}")
                        except Exception as e:
                            print(f"    ❌ Method '{method_name}' failed: {e}")
                            
            except Exception as e:
                print(f"    ❌ Instantiation failed: {type(e).__name__}: {e}")
    
    # Test pattern 2: Look for factory functions
    print("\n" + "-" * 70)
    functions = [attr for attr in attrs if not attr.startswith('_') and callable(getattr(g2p, attr)) and not isinstance(getattr(g2p, attr), type)]
    if functions:
        print(f"\n✅ Found {len(functions)} functions:")
        for func_name in functions:
            print(f"  • {func_name}")
            func = getattr(g2p, func_name)
            
            # Try calling with 'amh'
            try:
                result = func('amh')
                print(f"    ✅ Can call: {func_name}('amh') → {type(result)}")
                
                # If result is callable, try using it
                if callable(result):
                    try:
                        text_result = result('ሰላም')
                        print(f"    ✅ Result is callable: result('ሰላም') → {text_result}")
                    except Exception as e:
                        print(f"    ❌ Result call failed: {e}")
                        
            except Exception as e:
                print(f"    ❌ Call failed: {type(e).__name__}")
    
    print("\n" + "=" * 70)
    print("✅ Inspection complete!")
    print("=" * 70)
    
except ImportError as e:
    print(f"❌ Failed to import transphone.g2p: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
