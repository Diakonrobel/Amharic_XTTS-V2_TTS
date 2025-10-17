#!/usr/bin/env python3
"""Inspect Transphone API to find correct usage"""

import sys

print("=" * 70)
print("🔍 INSPECTING TRANSPHONE API")
print("=" * 70)
print()

try:
    import transphone
    print("✅ Transphone imported successfully")
    print(f"   Location: {transphone.__file__}")
    
    # List all attributes
    print("\n📋 Available attributes in transphone:")
    print("-" * 70)
    attrs = dir(transphone)
    for attr in sorted(attrs):
        if not attr.startswith('_'):
            obj = getattr(transphone, attr)
            obj_type = type(obj).__name__
            print(f"  {attr:30} ({obj_type})")
    
    print("\n" + "=" * 70)
    print("🔍 Looking for G2P-related functions...")
    print("=" * 70)
    
    # Try to find G2P functions
    g2p_candidates = [attr for attr in attrs if 'g2p' in attr.lower() or 'phoneme' in attr.lower()]
    if g2p_candidates:
        print(f"\nFound {len(g2p_candidates)} G2P-related attributes:")
        for attr in g2p_candidates:
            obj = getattr(transphone, attr)
            print(f"  • {attr}: {type(obj)}")
    else:
        print("\n❌ No obvious G2P functions found")
    
    # Check for classes
    print("\n" + "=" * 70)
    print("🔍 Looking for classes...")
    print("=" * 70)
    
    classes = [attr for attr in attrs if not attr.startswith('_') and isinstance(getattr(transphone, attr), type)]
    if classes:
        print(f"\nFound {len(classes)} classes:")
        for cls_name in classes:
            cls = getattr(transphone, cls_name)
            print(f"\n  Class: {cls_name}")
            methods = [m for m in dir(cls) if not m.startswith('_') and callable(getattr(cls, m, None))]
            if methods:
                print(f"    Methods: {', '.join(methods[:10])}")
    
    print("\n" + "=" * 70)
    print("✅ Inspection complete!")
    
except ImportError as e:
    print(f"❌ Failed to import transphone: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
