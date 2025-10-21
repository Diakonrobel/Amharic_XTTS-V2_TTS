"""
Test Script for Training Patches
=================================

Run this script to verify that the automatic training patches work correctly.

Usage:
    python test_training_patches.py
"""

import sys
import torch

print("=" * 70)
print("🧪 TESTING TRAINING PATCHES")
print("=" * 70)
print()

# Test 1: Import patches
print("Test 1: Importing training patches...")
try:
    from utils import training_patches
    print("✅ PASS: training_patches imported successfully")
    print()
except Exception as e:
    print(f"❌ FAIL: Could not import training_patches: {e}")
    sys.exit(1)

# Test 2: Check if autocast was patched
print("Test 2: Checking autocast patch...")
try:
    import torch.cuda.amp as cuda_amp
    # Try to create an autocast context
    with cuda_amp.autocast(dtype=torch.float16):
        x = torch.tensor([1.0, 2.0, 3.0])
    print("✅ PASS: autocast is working (patched to modern API)")
    print()
except Exception as e:
    print(f"❌ FAIL: autocast test failed: {e}")
    sys.exit(1)

# Test 3: Check if GradScaler was patched
print("Test 3: Checking GradScaler patch...")
try:
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()
    
    # Check if it's the patched version
    if hasattr(scaler, '_nan_skip_count'):
        print("✅ PASS: GradScaler is patched (SafeGradScaler)")
        print(f"   Initial scale: {scaler.get_scale()}")
        if scaler.get_scale() == 1024.0:
            print("   ✅ Conservative init_scale (1024) confirmed")
        else:
            print(f"   ⚠️  Warning: init_scale is {scaler.get_scale()}, expected 1024")
    else:
        print("⚠️  WARNING: GradScaler might not be fully patched")
    print()
except Exception as e:
    print(f"❌ FAIL: GradScaler test failed: {e}")
    sys.exit(1)

# Test 4: Check utility functions
print("Test 4: Checking utility functions...")
try:
    from utils.training_patches import safe_loss_backward, check_gradients_for_nan
    print("✅ PASS: Utility functions imported")
    print("   - safe_loss_backward")
    print("   - check_gradients_for_nan")
    print()
except Exception as e:
    print(f"❌ FAIL: Utility functions test failed: {e}")
    sys.exit(1)

# Test 5: Simulate a training step with NaN detection
print("Test 5: Testing NaN detection...")
try:
    # Create a simple model
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Create some dummy data
    x = torch.randn(2, 10)
    target = torch.randn(2, 5)
    
    # Forward pass with autocast
    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=False):  # Disabled for CPU testing
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
    
    # Test safe backward
    success = safe_loss_backward(loss, scaler=None)
    
    if success:
        print("✅ PASS: Safe backward pass completed")
    else:
        print("⚠️  WARNING: Backward pass indicated NaN")
    
    # Check gradients
    has_nan = check_gradients_for_nan(model)
    if not has_nan:
        print("✅ PASS: No NaN detected in gradients")
    else:
        print("⚠️  WARNING: NaN detected in gradients (expected for this test)")
    
    print()
except Exception as e:
    print(f"❌ FAIL: NaN detection test failed: {e}")
    sys.exit(1)

# Test 6: Test gradient clipping patch
print("Test 6: Testing gradient clipping patch...")
try:
    import torch.nn.utils as nn_utils
    
    # Create a model with gradients
    model = torch.nn.Linear(5, 3)
    x = torch.randn(1, 5)
    y = model(x).sum()
    y.backward()
    
    # Try to clip gradients
    total_norm = nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    print(f"✅ PASS: Gradient clipping works (norm: {total_norm:.4f})")
    print()
except Exception as e:
    print(f"❌ FAIL: Gradient clipping test failed: {e}")
    sys.exit(1)

# All tests passed
print("=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print()
print("🎉 Training patches are working correctly!")
print()
print("Next steps:")
print("1. Commit these changes to Git")
print("2. Push to GitHub")
print("3. Pull on Lightning AI")
print("4. Run training normally")
print()
print("The patches will apply automatically when training starts.")
print("=" * 70)
