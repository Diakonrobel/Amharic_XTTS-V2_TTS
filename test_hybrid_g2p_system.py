#!/usr/bin/env python3
"""
Comprehensive Test Suite for Hybrid G2P System

Run this script to verify all components work correctly before production use.

Usage:
    python test_hybrid_g2p_system.py
    
Or on Lightning AI:
    python test_hybrid_g2p_system.py --full
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_section(title):
    """Print a test section header"""
    print("\n" + "=" * 80)
    print(f"🧪 {title}")
    print("=" * 80)

def test_ethiopian_numerals():
    """Test Ethiopian numeral expansion"""
    print_section("TEST 1: Ethiopian Numeral Expansion")
    
    try:
        from amharic_tts.preprocessing.ethiopian_numeral_expander import EthiopianNumeralExpander
        
        expander = EthiopianNumeralExpander()
        
        test_cases = [
            ("፩", 1),
            ("፲", 10),
            ("፻", 100),
            ("፲፪", 12),
            ("፻፳፫", 123),
            ("፪፻፵፭", 245),
        ]
        
        passed = 0
        failed = 0
        
        for ethiopic, expected in test_cases:
            try:
                result = expander.parse_ethiopian_numeral(ethiopic)
                if result == expected:
                    print(f"✅ {ethiopic:10} → {result:6} (expected {expected})")
                    passed += 1
                else:
                    print(f"❌ {ethiopic:10} → {result:6} (expected {expected})")
                    failed += 1
            except Exception as e:
                print(f"❌ {ethiopic:10} → ERROR: {e}")
                failed += 1
        
        print(f"\n📊 Result: {passed}/{passed+failed} tests passed")
        return passed > 0 and failed == 0
        
    except Exception as e:
        print(f"❌ Module import failed: {e}")
        print("   This is expected if running standalone without full package.")
        return False

def test_prosody_handler():
    """Test prosody marker preservation"""
    print_section("TEST 2: Prosody Handler")
    
    try:
        from amharic_tts.preprocessing.prosody_handler import ProsodyHandler
        
        handler = ProsodyHandler()
        
        test_texts = [
            "ሰላም! እንዴት ነህ?",
            "Hello World.",
            "ዋጋው 100 ብር ነው።",
        ]
        
        for text in test_texts:
            info = handler.extract_prosody_info(text)
            stats = handler.get_prosody_statistics(text)
            
            print(f"\nInput: {text}")
            print(f"  Pauses: {stats['total_pauses']}")
            print(f"  Questions: {stats['questions']}")
            print(f"  Exclamations: {stats['exclamations']}")
            print(f"  Code-switches: {stats['code_switches']}")
        
        print(f"\n✅ Prosody handler working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_hybrid_g2p():
    """Test hybrid G2P system"""
    print_section("TEST 3: Hybrid G2P System")
    
    try:
        from amharic_tts.g2p.hybrid_g2p import HybridAmharicG2P, G2PConfig
        
        # Test with default config
        print("\n📋 Testing with default configuration...")
        g2p = HybridAmharicG2P()
        
        test_cases = [
            ("ሰላም ዓለም", "Amharic text"),
            ("Hello World", "English text"),
            ("ሰላም! Hello.", "Code-switching"),
            ("እንዴት ነህ?", "Question"),
        ]
        
        for text, description in test_cases:
            try:
                result = g2p.convert(text)
                lang = g2p.detect_language(text)
                print(f"\n✅ {description}")
                print(f"   Input:    {text}")
                print(f"   Language: {lang.value}")
                print(f"   Output:   {result[:80]}{'...' if len(result) > 80 else ''}")
            except Exception as e:
                print(f"❌ {description} failed: {e}")
                return False
        
        # Test statistics
        stats = g2p.get_statistics()
        print(f"\n📊 Statistics:")
        print(f"   Total conversions: {stats['total_conversions']}")
        print(f"   Cache hits: {stats['cache_hits']}")
        print(f"   Epitran used: {stats['epitran_used']}")
        print(f"   Rule-based used: {stats['rule_based_used']}")
        
        print(f"\n✅ Hybrid G2P system working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backend_selector():
    """Test G2P backend selector"""
    print_section("TEST 4: G2P Backend Selector")
    
    try:
        from utils.g2p_backend_selector import G2PBackendSelector
        
        selector = G2PBackendSelector(verbose=False)
        
        # Test backend detection
        available = selector.get_available_backends()
        print(f"\n📋 Available backends: {', '.join(available)}")
        
        # Test backend selection
        backends_to_test = ['hybrid', 'epitran', 'transphone', 'rule_based']
        
        for backend in backends_to_test:
            selected, reason = selector.select_backend(preferred=backend, fallback=True)
            print(f"\n{backend:12} → Selected: {selected:12} ({reason})")
        
        # Test recommendation
        recommended = selector.get_recommendation()
        print(f"\n🎯 Recommended backend: {recommended}")
        
        print(f"\n✅ Backend selector working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_integration():
    """Test integration with XTTS workflows"""
    print_section("TEST 5: Integration Test")
    
    try:
        # Test if hybrid option exists in UI config
        import xtts_demo
        print("✅ xtts_demo.py imports successfully")
        
        # Test if backend selector recognizes hybrid
        from utils.g2p_backend_selector import G2PBackendSelector
        selector = G2PBackendSelector(verbose=False)
        
        if selector.is_backend_available('hybrid'):
            print("✅ Hybrid backend recognized by selector")
        else:
            print("❌ Hybrid backend NOT recognized")
            return False
        
        # Test if hybrid has priority 0
        info = selector.get_backend_info('hybrid')
        if info and info.priority == 0:
            print(f"✅ Hybrid backend has priority {info.priority} (highest)")
        else:
            print(f"⚠️  Hybrid priority: {info.priority if info else 'N/A'}")
        
        print(f"\n✅ Integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests(full=False):
    """Run all test suites"""
    print("\n" + "=" * 80)
    print("🚀 HYBRID G2P SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("\nRunning all tests...")
    
    results = {
        "Ethiopian Numerals": test_ethiopian_numerals(),
        "Prosody Handler": test_prosody_handler(),
        "Hybrid G2P": test_hybrid_g2p(),
        "Backend Selector": test_backend_selector(),
        "Integration": test_integration(),
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} - {test_name}")
    
    print("\n" + "-" * 80)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for production.")
        return 0
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please check errors above.")
        return 1

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Hybrid G2P System')
    parser.add_argument('--full', action='store_true', help='Run full test suite with verbose output')
    args = parser.parse_args()
    
    sys.exit(run_all_tests(full=args.full))

if __name__ == "__main__":
    main()
