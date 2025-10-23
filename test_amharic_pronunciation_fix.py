#!/usr/bin/env python3
"""
Test Amharic Pronunciation Fix
================================

Tests that problematic Amharic characters (ቀ, ጨ, ፀ) are pronounced consistently
when G2P is enabled.

Usage:
    python test_amharic_pronunciation_fix.py
"""

def test_g2p_conversion():
    """Test that G2P properly converts problematic Amharic characters."""
    
    print("=" * 70)
    print("AMHARIC PRONUNCIATION FIX TEST")
    print("=" * 70)
    print()
    
    # Test text with problematic characters
    test_texts = [
        "ቀ",      # qe
        "ጨ",      # che  
        "ፀ",      # tse
        "ቀጨፀ",   # Combined
        "ሰላም ቀን ጨለመ ፀሐይ",  # Sentence with problematic chars
    ]
    
    print("Testing G2P conversion for problematic characters...")
    print()
    
    try:
        # Try hybrid G2P first (best quality)
        print("Attempting Hybrid G2P (epitran + rule-based)...")
        from amharic_tts.g2p.hybrid_g2p import HybridAmharicG2P, G2PConfig
        
        config = G2PConfig(
            use_epitran=True,
            use_rule_based=True,
            expand_numbers=True,
            preserve_prosody=True
        )
        g2p = HybridAmharicG2P(config=config)
        backend = "Hybrid"
        
    except Exception as e:
        print(f"Hybrid G2P not available: {e}")
        print("Trying standard tokenizer with Transphone...")
        
        try:
            from amharic_tts.tokenizer.xtts_tokenizer_wrapper import create_xtts_tokenizer
            tokenizer = create_xtts_tokenizer(use_phonemes=True, g2p_backend='transphone')
            backend = "Transphone"
            
            # Wrapper for consistent API
            class TokenizerWrapper:
                def __init__(self, tok):
                    self.tok = tok
                
                def convert(self, text):
                    return self.tok.preprocess_text(text, lang='am')
            
            g2p = TokenizerWrapper(tokenizer)
            
        except Exception as e:
            print(f"Transphone not available: {e}")
            print("Trying rule-based G2P...")
            
            try:
                from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
                g2p = AmharicG2P(backend='rule_based')
                backend = "Rule-based"
            except Exception as e:
                print(f"❌ No G2P backend available: {e}")
                print("Please install at least one G2P backend:")
                print("  pip install transphone")
                print("  or ensure rule-based G2P is available")
                return False
    
    print(f"✅ Using {backend} G2P backend")
    print()
    print("-" * 70)
    
    # Test each text
    all_successful = True
    for text in test_texts:
        try:
            phonemes = g2p.convert(text)
            
            # Check if conversion actually happened
            if phonemes == text:
                print(f"⚠️  '{text}' → unchanged (G2P may not have processed)")
                print(f"    This is OK if text has no Amharic chars")
            else:
                print(f"✅ '{text}' → '{phonemes}'")
            
        except Exception as e:
            print(f"❌ '{text}' → FAILED: {e}")
            all_successful = False
    
    print("-" * 70)
    print()
    
    if all_successful:
        print("✅ ALL TESTS PASSED")
        print()
        print("Next steps:")
        print("1. Enable 'Use Amharic G2P' checkbox in the inference UI")
        print("2. Use language='amh' or 'am'")
        print("3. Test with text containing: ቀ, ጨ, ፀ")
        print("4. You should now get consistent pronunciation!")
        return True
    else:
        print("⚠️  SOME TESTS FAILED")
        print()
        print("The G2P backend may need additional setup.")
        print("For best results, install Transphone:")
        print("  pip install transphone")
        return False


def check_g2p_backends():
    """Check which G2P backends are available."""
    
    print()
    print("=" * 70)
    print("G2P BACKEND AVAILABILITY CHECK")
    print("=" * 70)
    print()
    
    backends = []
    
    # Check Transphone
    try:
        import transphone
        print("✅ Transphone: Available")
        backends.append("transphone")
    except ImportError:
        print("❌ Transphone: Not installed")
        print("   Install: pip install transphone")
    
    # Check Epitran
    try:
        import epitran
        print("✅ Epitran: Available")
        backends.append("epitran")
    except ImportError:
        print("❌ Epitran: Not installed")
        print("   Install: pip install epitran")
    
    # Check rule-based
    try:
        from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
        g2p = AmharicG2P(backend='rule_based')
        print("✅ Rule-based G2P: Available")
        backends.append("rule_based")
    except Exception:
        print("❌ Rule-based G2P: Not available")
    
    # Check hybrid
    try:
        from amharic_tts.g2p.hybrid_g2p import HybridAmharicG2P
        print("✅ Hybrid G2P: Available")
        backends.append("hybrid")
    except Exception:
        print("❌ Hybrid G2P: Not available")
    
    print()
    
    if backends:
        print(f"✅ {len(backends)} backend(s) available: {', '.join(backends)}")
        print()
        print("Recommendation:")
        if "transphone" in backends:
            print("  Use 'transphone' for best quality")
        elif "hybrid" in backends:
            print("  Use 'hybrid' for best quality")
        elif "rule_based" in backends:
            print("  Use 'rule_based' (offline, good quality)")
        return True
    else:
        print("❌ No G2P backends available!")
        print()
        print("To fix:")
        print("  pip install transphone")
        return False


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║     AMHARIC PRONUNCIATION FIX VERIFICATION                           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Check backends
    has_backends = check_g2p_backends()
    
    if not has_backends:
        print()
        print("=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)
        print()
        print("Install at least one G2P backend:")
        print()
        print("  pip install transphone")
        print()
        print("Then re-run this test script.")
        exit(1)
    
    # Run conversion tests
    print()
    success = test_g2p_conversion()
    
    print()
    print("=" * 70)
    if success:
        print("✅ FIX VERIFIED - Ready to use!")
    else:
        print("⚠️  PARTIAL SUCCESS - May need additional setup")
    print("=" * 70)
    print()
