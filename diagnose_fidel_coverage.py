#!/usr/bin/env python3
"""
Amharic Fidel Coverage Diagnostic
==================================

Checks if problematic Amharic fidels are present in vocabulary and dataset.
Specifically targets: ጨ ጠ ፀ ጰ ቀ and their variant forms.
"""

import json
import os
import csv
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

# Problematic fidels and their complete series
PROBLEMATIC_FIDELS = {
    'cha': ['ጨ', 'ጩ', 'ጪ', 'ጫ', 'ጬ', 'ጭ', 'ጮ'],  # cha series
    'tta': ['ጠ', 'ጡ', 'ጢ', 'ጣ', 'ጤ', 'ጥ', 'ጦ'],  # tta (emphatic t)
    'tsa': ['ፀ', 'ፁ', 'ፂ', 'ፃ', 'ፄ', 'ፅ', 'ፆ'],  # tsa series
    'ppa': ['ጰ', 'ጱ', 'ጲ', 'ጳ', 'ጴ', 'ጵ', 'ጶ'],  # ppa (emphatic p)
    'qha': ['ቀ', 'ቁ', 'ቂ', 'ቃ', 'ቄ', 'ቅ', 'ቆ'],  # qha series
}

def load_vocab(vocab_path: str) -> Dict:
    """Load vocabulary file"""
    print(f"📂 Loading vocabulary: {vocab_path}")
    
    if not os.path.exists(vocab_path):
        print(f"   ❌ File not found!")
        return None
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    vocab = vocab_data.get('model', {}).get('vocab', {})
    
    # Handle both dict and list formats
    if isinstance(vocab, list):
        vocab_set = set(vocab)
    elif isinstance(vocab, dict):
        vocab_set = set(vocab.keys())
    else:
        print(f"   ❌ Unknown vocab format!")
        return None
    
    print(f"   ✅ Loaded {len(vocab_set)} tokens")
    return vocab_set

def analyze_fidel_coverage(vocab_set: set) -> Dict:
    """Check coverage of problematic fidels in vocabulary"""
    print("\n" + "=" * 70)
    print("FIDEL COVERAGE ANALYSIS")
    print("=" * 70)
    
    results = {}
    all_covered = True
    
    for series_name, fidels in PROBLEMATIC_FIDELS.items():
        print(f"\n📊 {series_name.upper()} Series: {' '.join(fidels)}")
        
        covered = []
        missing = []
        
        for fidel in fidels:
            if fidel in vocab_set:
                covered.append(fidel)
            else:
                missing.append(fidel)
        
        coverage_pct = (len(covered) / len(fidels)) * 100
        
        print(f"   Covered: {len(covered)}/{len(fidels)} ({coverage_pct:.1f}%)")
        
        if covered:
            print(f"   ✅ Present: {' '.join(covered)}")
        if missing:
            print(f"   ❌ Missing: {' '.join(missing)}")
            all_covered = False
        
        results[series_name] = {
            'covered': covered,
            'missing': missing,
            'coverage_pct': coverage_pct
        }
    
    print("\n" + "-" * 70)
    if all_covered:
        print("✅ All problematic fidels are in vocabulary!")
    else:
        print("⚠️  Some fidels are missing from vocabulary!")
    
    return results

def analyze_dataset_coverage(dataset_csv: str, fidel_series: Dict) -> Dict:
    """Check how often problematic fidels appear in dataset"""
    print("\n" + "=" * 70)
    print("DATASET FREQUENCY ANALYSIS")
    print("=" * 70)
    
    if not os.path.exists(dataset_csv):
        print(f"   ❌ Dataset not found: {dataset_csv}")
        return {}
    
    print(f"📂 Analyzing: {dataset_csv}")
    
    # Count fidel occurrences
    fidel_counts = Counter()
    total_chars = 0
    total_lines = 0
    
    try:
        with open(dataset_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if len(row) >= 2:
                    text = row[1]
                    total_lines += 1
                    
                    for char in text:
                        total_chars += 1
                        # Check all problematic fidels
                        for series_name, fidels in fidel_series.items():
                            if char in fidels:
                                fidel_counts[char] += 1
    except Exception as e:
        print(f"   ❌ Error reading dataset: {e}")
        return {}
    
    print(f"   Total lines: {total_lines}")
    print(f"   Total characters: {total_chars}")
    print(f"\n   Problematic fidel frequencies:\n")
    
    results = {}
    all_fidels = []
    for fidels in fidel_series.values():
        all_fidels.extend(fidels)
    
    for series_name, fidels in fidel_series.items():
        print(f"\n   {series_name.upper()}:")
        series_total = 0
        
        for fidel in fidels:
            count = fidel_counts.get(fidel, 0)
            series_total += count
            freq_per_10k = (count / total_chars * 10000) if total_chars > 0 else 0
            
            status = "❌" if count == 0 else "⚠️" if count < 10 else "✅"
            print(f"      {status} {fidel}: {count} times ({freq_per_10k:.2f} per 10k chars)")
        
        results[series_name] = series_total
        print(f"      Series total: {series_total}")
    
    # Overall statistics
    total_problematic = sum(fidel_counts.values())
    print(f"\n   📈 Total problematic fidel occurrences: {total_problematic}")
    print(f"   📈 Percentage of dataset: {(total_problematic/total_chars*100):.3f}%")
    
    return results

def check_g2p_phoneme_mapping():
    """Check if G2P has correct mappings for these fidels"""
    print("\n" + "=" * 70)
    print("G2P PHONEME MAPPING CHECK")
    print("=" * 70)
    
    try:
        from amharic_tts.g2p.ethiopic_g2p_table import ETHIOPIC_TO_IPA
        
        print("✅ G2P table found!")
        print("\nPhoneme mappings for problematic fidels:\n")
        
        for series_name, fidels in PROBLEMATIC_FIDELS.items():
            print(f"   {series_name.upper()}:")
            for fidel in fidels:
                phoneme = ETHIOPIC_TO_IPA.get(fidel, "NOT FOUND")
                status = "✅" if phoneme != "NOT FOUND" else "❌"
                print(f"      {status} {fidel} → {phoneme}")
            print()
        
        return True
    except ImportError:
        print("⚠️  G2P module not available (not used in your training)")
        return False

def generate_recommendations(vocab_results: Dict, dataset_results: Dict):
    """Generate actionable recommendations"""
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    # Check for missing vocab
    missing_from_vocab = []
    for series_name, data in vocab_results.items():
        if data['missing']:
            missing_from_vocab.extend(data['missing'])
    
    # Check for underrepresented fidels
    underrepresented = []
    for series_name, count in dataset_results.items():
        if count < 100:  # Less than 100 occurrences
            underrepresented.append(series_name)
    
    print("\n📋 Action Items:\n")
    
    if missing_from_vocab:
        print("❌ CRITICAL: Missing fidels in vocabulary!")
        print(f"   Missing: {' '.join(missing_from_vocab)}")
        print("\n   Solution: Rebuild vocabulary with all Ethiopic characters")
        print("   Command: python utils/vocab_extension.py \\")
        print("            --input-vocab base_models/v2.0.2/vocab.json \\")
        print("            --output-vocab output/ready/vocab_fixed.json \\")
        print("            --dataset-csv output/metadata_train.csv")
        print()
    
    if underrepresented:
        print("⚠️  WARNING: Underrepresented fidel series!")
        print(f"   Series: {', '.join(underrepresented)}")
        print("\n   Solution: Augment dataset with more examples")
        print("   Run: python augment_fidel_dataset.py")
        print()
    
    if not missing_from_vocab and not underrepresented:
        print("✅ Vocabulary and dataset look good!")
        print("\n   Possible causes of pronunciation issues:")
        print("   1. Not enough training epochs (try 5-10 more epochs)")
        print("   2. Learning rate too high/low")
        print("   3. Model hasn't learned character-phoneme mapping yet")
        print("\n   Solution: Continue training from checkpoint")
        print("   - Load your existing checkpoint")
        print("   - Train for 5-10 more epochs")
        print("   - Monitor loss specifically on problematic fidels")
    else:
        print("\n🔧 QUICK FIX (No full retrain):")
        print("   1. Fix vocabulary if needed (add missing fidels)")
        print("   2. Augment dataset with fidel examples")
        print("   3. Continue training from your epoch 10 checkpoint")
        print("   4. Train for 5-10 more epochs")
        print("\n   This preserves your existing 15k steps of training!")

def main():
    print("\n" + "=" * 70)
    print("AMHARIC FIDEL PRONUNCIATION DIAGNOSTIC")
    print("=" * 70)
    print("\nAnalyzing vocabulary and dataset for problematic fidels:")
    print("ጨ ጠ ፀ ጰ ቀ (and their variant forms)\n")
    
    # Get paths
    vocab_path = input("Enter path to vocab.json: ").strip().strip('"')
    dataset_csv = input("Enter path to metadata_train.csv (optional, press Enter to skip): ").strip().strip('"')
    
    if not os.path.exists(vocab_path):
        print(f"\n❌ Vocab file not found: {vocab_path}")
        return
    
    # Load and analyze vocabulary
    vocab_set = load_vocab(vocab_path)
    if vocab_set is None:
        return
    
    vocab_results = analyze_fidel_coverage(vocab_set)
    
    # Analyze dataset if provided
    dataset_results = {}
    if dataset_csv and os.path.exists(dataset_csv):
        dataset_results = analyze_dataset_coverage(dataset_csv, PROBLEMATIC_FIDELS)
    
    # Check G2P mappings
    check_g2p_phoneme_mapping()
    
    # Generate recommendations
    generate_recommendations(vocab_results, dataset_results)
    
    print("\n" + "=" * 70)
    print("✅ Diagnostic complete!")
    print("=" * 70)
    print()

if __name__ == "__main__":
    main()
