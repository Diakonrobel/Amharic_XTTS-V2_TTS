#!/usr/bin/env python3
"""Test number expansion in Amharic G2P"""

from amharic_tts.g2p.amharic_g2p_enhanced import EnhancedAmharicG2P

def test_number_expansion():
    """Test that numbers are properly expanded to Amharic words"""
    
    print("=" * 70)
    print("🧪 Testing Number Expansion in Amharic G2P")
    print("=" * 70)
    
    # Initialize G2P (will use rule-based if Transphone unavailable)
    g2p = EnhancedAmharicG2P()
    
    # Test cases from your example
    test_cases = [
        # Your original text with year
        "ይህ በአውሮፓውያኑ 1959 በቀኝ ገዢዋ ብሪታኒያ የተፈረመው",
        
        # Numbers with decimals
        "ግብፅ 55.5 ቢሊዮን ኪዩቢክ ውሃ እንዲሁም ሱዳን 18.5 ቢሊዮን",
        
        # Simple year
        "በ 2025 ዓ.ም",
        
        # Large number
        "85 በመቶ ውሃን",
        
        # Multiple numbers
        "1000 እና 2000 ብር",
    ]
    
    print("\n📝 Test Results:\n")
    
    all_passed = True
    for i, text in enumerate(test_cases, 1):
        result = g2p.convert(text)
        
        # Check if any digits remain in output (they shouldn't)
        has_digits = any(c.isdigit() for c in result)
        status = "❌ FAIL" if has_digits else "✅ PASS"
        
        if has_digits:
            all_passed = False
        
        print(f"Test {i}: {status}")
        print(f"  Input:  {text}")
        print(f"  Output: {result}")
        print(f"  Digits remaining: {has_digits}")
        print()
    
    print("=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Numbers are being expanded correctly!")
    else:
        print("❌ SOME TESTS FAILED - Numbers still present in output")
    print("=" * 70)
    
    return all_passed

def test_direct_number_expander():
    """Test the number expander directly"""
    
    print("\n" + "=" * 70)
    print("🧪 Testing Number Expander Directly")
    print("=" * 70 + "\n")
    
    from amharic_tts.preprocessing.number_expander import AmharicNumberExpander
    
    expander = AmharicNumberExpander()
    
    test_numbers = [
        1959,
        55,
        18,
        85,
        1000,
        2025,
    ]
    
    print("Number → Amharic Word:\n")
    for num in test_numbers:
        amharic = expander.expand(num)
        print(f"  {num:6d} → {amharic}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    # Test number expander directly
    test_direct_number_expander()
    
    # Test full G2P pipeline with number expansion
    print("\n\n")
    success = test_number_expansion()
    
    import sys
    sys.exit(0 if success else 1)
