#!/usr/bin/env python3
"""
Fidel Dataset Augmentation
===========================

Creates augmented training examples focusing on problematic Amharic fidels.
Helps model learn pronunciation of: ጨ ጠ ፀ ጰ ቀ
"""

import os
import csv
import shutil
from pathlib import Path
from typing import List, Dict

# Problematic fidel series with example words
FIDEL_EXAMPLES = {
    'cha': {
        'fidels': ['ጨ', 'ጩ', 'ጪ', 'ጫ', 'ጬ', 'ጭ', 'ጮ'],
        'words': [
            'ጨዋ',      # chewa (gentleman)
            'ጨዋታ',    # chewata (game)
            'ጨለማ',    # chelema (darkness)
            'ጨረስ',    # cheres (finish)
            'ጨው',     # chew (salt)
        ]
    },
    'tta': {
        'fidels': ['ጠ', 'ጡ', 'ጢ', 'ጣ', 'ጤ', 'ጥ', 'ጦ'],
        'words': [
            'ጠዋት',    # tewat (morning)
            'ጠየቀ',    # teyeke (asked)
            'ጠንካራ',   # tenkara (strong)
            'ጠላ',     # tela (local beer)
            'ጤና',     # tena (health)
        ]
    },
    'tsa': {
        'fidels': ['ፀ', 'ፁ', 'ፂ', 'ፃ', 'ፄ', 'ፅ', 'ፆ'],
        'words': [
            'ፀሐይ',    # tsehay (sun)
            'ፀጉር',    # tsegur (hair)
            'ፀሎት',    # tselot (prayer)
            'ፀረ',     # tsere (anti)
            'ፀና',     # tsena (be calm)
        ]
    },
    'ppa': {
        'fidels': ['ጰ', 'ጱ', 'ጲ', 'ጳ', 'ጴ', 'ጵ', 'ጶ'],
        'words': [
            'ጰጦስ',    # ppetos (Peter - biblical name)
            'ጲላጦስ',   # ppilatos (Pilate)
            'ጳውሎስ',   # ppawlos (Paul)
        ]
    },
    'qha': {
        'fidels': ['ቀ', 'ቁ', 'ቂ', 'ቃ', 'ቄ', 'ቅ', 'ቆ'],
        'words': [
            'ቀን',     # qen (day)
            'ቀላል',    # qelal (easy/light)
            'ቀይ',     # qey (red)
            'ቅዳሜ',    # qedame (Saturday)
            'ቆዳ',     # qoda (skin/leather)
            'ቀበሌ',    # qebele (kebele/district)
        ]
    }
}

def create_fidel_sentences() -> List[Dict[str, str]]:
    """Generate training sentences focusing on problematic fidels"""
    sentences = []
    
    print("📝 Generating fidel-focused training sentences...\n")
    
    for series_name, data in FIDEL_EXAMPLES.items():
        print(f"   {series_name.upper()} series:")
        fidels = data['fidels']
        words = data['words']
        
        # Create sentences with multiple occurrences
        for word in words:
            # Simple repetition for emphasis
            sentences.append({
                'text': word,
                'type': f'{series_name}_word'
            })
            
            # Word in context (basic pattern)
            sentences.append({
                'text': f'{word} ነው',  # "it is [word]"
                'type': f'{series_name}_context'
            })
            
            # Multiple words from same series
            if len(words) >= 2:
                sentences.append({
                    'text': f'{word} እና {words[0]}',  # "[word] and [word]"
                    'type': f'{series_name}_multiple'
                })
        
        # Character progression (all forms)
        fidel_sequence = ' '.join(fidels)
        sentences.append({
            'text': fidel_sequence,
            'type': f'{series_name}_progression'
        })
        
        print(f"      Generated {len([s for s in sentences if series_name in s['type']])} examples")
    
    print(f"\n   Total generated: {len(sentences)} sentences")
    return sentences

def extract_fidel_examples_from_dataset(dataset_csv: str, min_count: int = 2) -> List[Dict]:
    """Extract existing examples containing problematic fidels"""
    print(f"\n📂 Extracting fidel examples from: {dataset_csv}")
    
    if not os.path.exists(dataset_csv):
        print("   ⚠️  Dataset not found, skipping extraction")
        return []
    
    extracted = []
    all_fidels = []
    for data in FIDEL_EXAMPLES.values():
        all_fidels.extend(data['fidels'])
    
    try:
        with open(dataset_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if len(row) >= 2:
                    audio_path, text = row[0], row[1]
                    
                    # Count fidel occurrences
                    fidel_count = sum(1 for char in text if char in all_fidels)
                    
                    if fidel_count >= min_count:
                        extracted.append({
                            'audio_path': audio_path,
                            'text': text,
                            'fidel_count': fidel_count
                        })
    except Exception as e:
        print(f"   ❌ Error reading dataset: {e}")
        return []
    
    print(f"   ✅ Extracted {len(extracted)} examples with problematic fidels")
    return extracted

def create_augmented_csv(
    output_csv: str,
    generated_sentences: List[Dict],
    extracted_examples: List[Dict],
    repeat_factor: int = 3
):
    """Create augmented training CSV with fidel-focused examples"""
    print(f"\n💾 Creating augmented dataset: {output_csv}")
    
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        
        # Write extracted real examples (repeated for emphasis)
        for _ in range(repeat_factor):
            for example in extracted_examples:
                writer.writerow([example['audio_path'], example['text']])
        
        # Note: Generated sentences need corresponding audio
        # For now, we'll create a reference file
    
    print(f"   ✅ Written {len(extracted_examples) * repeat_factor} augmented examples")
    print(f"\n   📋 Note: This augmented file contains:")
    print(f"      - {len(extracted_examples)} unique fidel-rich examples")
    print(f"      - Repeated {repeat_factor}x for emphasis during training")

def create_fidel_wordlist(output_file: str):
    """Create wordlist for recording/synthesis"""
    print(f"\n📝 Creating fidel wordlist: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Amharic Fidel Pronunciation Wordlist\n")
        f.write("# Record these words to improve model pronunciation\n\n")
        
        for series_name, data in FIDEL_EXAMPLES.items():
            f.write(f"\n## {series_name.upper()} Series\n")
            f.write(f"# Fidels: {' '.join(data['fidels'])}\n\n")
            
            for word in data['words']:
                f.write(f"{word}\n")
    
    print(f"   ✅ Wordlist created with {sum(len(d['words']) for d in FIDEL_EXAMPLES.values())} words")
    print(f"   💡 Tip: Record these words with your target voice for best results!")

def main():
    print("\n" + "=" * 70)
    print("AMHARIC FIDEL DATASET AUGMENTATION")
    print("=" * 70)
    print("\nThis script helps improve pronunciation of problematic fidels:")
    print("ጨ ጠ ፀ ጰ ቀ\n")
    
    # Get paths
    dataset_csv = input("Enter path to your training CSV (metadata_train.csv): ").strip().strip('"')
    output_dir = input("Enter output directory [./augmented_data]: ").strip().strip('"') or "./augmented_data"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate fidel-focused sentences
    generated = create_fidel_sentences()
    
    # Extract existing fidel-rich examples
    extracted = extract_fidel_examples_from_dataset(dataset_csv)
    
    # Create augmented CSV
    augmented_csv = os.path.join(output_dir, "metadata_fidel_augmented.csv")
    create_augmented_csv(augmented_csv, generated, extracted, repeat_factor=3)
    
    # Create wordlist for recording
    wordlist_file = os.path.join(output_dir, "fidel_wordlist.txt")
    create_fidel_wordlist(wordlist_file)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"\n1. ✅ Augmented dataset created: {augmented_csv}")
    print(f"2. ✅ Wordlist created: {wordlist_file}")
    print("\n3. 📋 To use the augmented data:\n")
    print("   Option A - Merge with existing dataset:")
    print(f"   cat {dataset_csv} {augmented_csv} > metadata_train_augmented.csv")
    print("\n   Option B - Train on fidel data only (for targeted improvement):")
    print(f"   Use {augmented_csv} for a few epochs, then switch back")
    print("\n4. 🎤 Optional: Record words from fidel_wordlist.txt")
    print("   - Record each word 3-5 times")
    print("   - Use same voice as main dataset")
    print("   - Add to training data for maximum impact")
    print("\n5. 🚀 Continue training from your checkpoint:")
    print("   - See CONTINUE_TRAINING_GUIDE.md for instructions")
    print("\n" + "=" * 70)
    print()

if __name__ == "__main__":
    main()
