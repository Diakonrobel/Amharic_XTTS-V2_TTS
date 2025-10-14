#!/usr/bin/env python3
"""
Validation Data Diagnostic Script

Analyzes train/val split to identify potential data quality issues
causing high validation loss.

Usage:
    python diagnose_validation_data.py
"""

import os
import csv
import librosa
import numpy as np
from pathlib import Path
from collections import defaultdict
import json


def load_csv_data(csv_path):
    """Load data from CSV file."""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if len(row) >= 2:
                audio_path = row[0]
                text = row[1]
                data.append({
                    'audio': audio_path,
                    'text': text,
                    'exists': os.path.exists(audio_path)
                })
    return data


def analyze_audio_file(audio_path):
    """Analyze a single audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # RMS energy (loudness proxy)
        rms = librosa.feature.rms(y=y)[0]
        avg_rms = np.mean(rms)
        
        # Spectral centroid (brightness/quality proxy)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_centroid = np.mean(spectral_centroid)
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'rms': avg_rms,
            'spectral_centroid': avg_centroid,
            'success': True
        }
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }


def analyze_dataset(data, name):
    """Analyze a dataset (train or val)."""
    print(f"\n{'='*70}")
    print(f"📊 Analyzing {name.upper()} Dataset")
    print(f"{'='*70}\n")
    
    total = len(data)
    exists = sum(1 for d in data if d['exists'])
    missing = total - exists
    
    print(f"Total samples: {total}")
    print(f"Existing files: {exists}")
    print(f"Missing files: {missing}")
    
    if missing > 0:
        print(f"⚠️  WARNING: {missing} audio files are missing!")
    
    # Analyze text
    text_lengths = [len(d['text']) for d in data]
    avg_text_len = np.mean(text_lengths)
    std_text_len = np.std(text_lengths)
    
    print(f"\nText Analysis:")
    print(f"  Average length: {avg_text_len:.1f} chars")
    print(f"  Std deviation: {std_text_len:.1f} chars")
    print(f"  Min length: {min(text_lengths)} chars")
    print(f"  Max length: {max(text_lengths)} chars")
    
    # Analyze audio (sample-based to save time)
    print(f"\nAudio Analysis (sampling {min(50, exists)} files):")
    
    audio_data = []
    sample_size = min(50, exists)
    existing_files = [d for d in data if d['exists']]
    
    for i, d in enumerate(existing_files[:sample_size]):
        if i % 10 == 0:
            print(f"  Processing {i}/{sample_size}...", end='\r')
        
        result = analyze_audio_file(d['audio'])
        if result['success']:
            audio_data.append(result)
    
    print(f"  Processing {sample_size}/{sample_size}... Done!")
    
    if audio_data:
        durations = [a['duration'] for a in audio_data]
        sample_rates = [a['sample_rate'] for a in audio_data]
        rms_values = [a['rms'] for a in audio_data]
        centroids = [a['spectral_centroid'] for a in audio_data]
        
        print(f"\n  Duration:")
        print(f"    Average: {np.mean(durations):.2f}s")
        print(f"    Std dev: {np.std(durations):.2f}s")
        print(f"    Min: {min(durations):.2f}s")
        print(f"    Max: {max(durations):.2f}s")
        
        print(f"\n  Sample Rate:")
        unique_sr = set(sample_rates)
        for sr in unique_sr:
            count = sample_rates.count(sr)
            print(f"    {sr} Hz: {count}/{len(sample_rates)} files")
        
        print(f"\n  Audio Quality (RMS Energy):")
        print(f"    Average: {np.mean(rms_values):.4f}")
        print(f"    Std dev: {np.std(rms_values):.4f}")
        
        print(f"\n  Spectral Brightness:")
        print(f"    Average: {np.mean(centroids):.1f} Hz")
        print(f"    Std dev: {np.std(centroids):.1f} Hz")
    
    return {
        'total': total,
        'exists': exists,
        'text_lengths': text_lengths,
        'audio_data': audio_data
    }


def compare_datasets(train_stats, val_stats):
    """Compare train and validation datasets."""
    print(f"\n{'='*70}")
    print(f"🔍 TRAIN vs VALIDATION COMPARISON")
    print(f"{'='*70}\n")
    
    # Size comparison
    print(f"Dataset Sizes:")
    print(f"  Train: {train_stats['total']} samples")
    print(f"  Val:   {val_stats['total']} samples")
    print(f"  Ratio: {val_stats['total']/train_stats['total']*100:.1f}% val")
    
    # Text length comparison
    train_avg_text = np.mean(train_stats['text_lengths'])
    val_avg_text = np.mean(val_stats['text_lengths'])
    text_diff = abs(train_avg_text - val_avg_text) / train_avg_text * 100
    
    print(f"\nText Length:")
    print(f"  Train avg: {train_avg_text:.1f} chars")
    print(f"  Val avg:   {val_avg_text:.1f} chars")
    print(f"  Difference: {text_diff:.1f}%", end="")
    
    if text_diff > 20:
        print(f"  ⚠️  SIGNIFICANT MISMATCH!")
    else:
        print(f"  ✅")
    
    # Audio comparison
    if train_stats['audio_data'] and val_stats['audio_data']:
        train_durations = [a['duration'] for a in train_stats['audio_data']]
        val_durations = [a['duration'] for a in val_stats['audio_data']]
        
        train_avg_dur = np.mean(train_durations)
        val_avg_dur = np.mean(val_durations)
        dur_diff = abs(train_avg_dur - val_avg_dur) / train_avg_dur * 100
        
        print(f"\nAudio Duration:")
        print(f"  Train avg: {train_avg_dur:.2f}s")
        print(f"  Val avg:   {val_avg_dur:.2f}s")
        print(f"  Difference: {dur_diff:.1f}%", end="")
        
        if dur_diff > 30:
            print(f"  ⚠️  SIGNIFICANT MISMATCH!")
        else:
            print(f"  ✅")
        
        # Quality comparison
        train_rms = [a['rms'] for a in train_stats['audio_data']]
        val_rms = [a['rms'] for a in val_stats['audio_data']]
        
        train_avg_rms = np.mean(train_rms)
        val_avg_rms = np.mean(val_rms)
        rms_diff = abs(train_avg_rms - val_avg_rms) / train_avg_rms * 100
        
        print(f"\nAudio Quality (RMS):")
        print(f"  Train avg: {train_avg_rms:.4f}")
        print(f"  Val avg:   {val_avg_rms:.4f}")
        print(f"  Difference: {rms_diff:.1f}%", end="")
        
        if rms_diff > 30:
            print(f"  ⚠️  SIGNIFICANT MISMATCH!")
        else:
            print(f"  ✅")


def diagnose(train_csv, val_csv):
    """Main diagnostic function."""
    print(f"\n{'='*70}")
    print(f"🔍 VALIDATION DATA DIAGNOSTIC")
    print(f"{'='*70}\n")
    
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV:   {val_csv}")
    
    # Load data
    print(f"\n📂 Loading datasets...")
    train_data = load_csv_data(train_csv)
    val_data = load_csv_data(val_csv)
    
    # Analyze each dataset
    train_stats = analyze_dataset(train_data, "train")
    val_stats = analyze_dataset(val_data, "validation")
    
    # Compare
    compare_datasets(train_stats, val_stats)
    
    # Recommendations
    print(f"\n{'='*70}")
    print(f"💡 RECOMMENDATIONS")
    print(f"{'='*70}\n")
    
    issues = []
    
    # Check text length mismatch
    train_avg_text = np.mean(train_stats['text_lengths'])
    val_avg_text = np.mean(val_stats['text_lengths'])
    text_diff = abs(train_avg_text - val_avg_text) / train_avg_text * 100
    
    if text_diff > 20:
        issues.append(f"⚠️  Text length mismatch ({text_diff:.1f}% difference)")
        print(f"1. RESPLIT DATA: Validation text is significantly different length")
        print(f"   → Use random split to ensure similar distributions")
    
    # Check audio duration mismatch
    if train_stats['audio_data'] and val_stats['audio_data']:
        train_durations = [a['duration'] for a in train_stats['audio_data']]
        val_durations = [a['duration'] for a in val_stats['audio_data']]
        train_avg_dur = np.mean(train_durations)
        val_avg_dur = np.mean(val_durations)
        dur_diff = abs(train_avg_dur - val_avg_dur) / train_avg_dur * 100
        
        if dur_diff > 30:
            issues.append(f"⚠️  Audio duration mismatch ({dur_diff:.1f}% difference)")
            print(f"2. RESPLIT DATA: Validation audio durations significantly different")
            print(f"   → Ensure train/val have similar duration distributions")
    
    # Check quality mismatch
    if train_stats['audio_data'] and val_stats['audio_data']:
        train_rms = [a['rms'] for a in train_stats['audio_data']]
        val_rms = [a['rms'] for a in val_stats['audio_data']]
        train_avg_rms = np.mean(train_rms)
        val_avg_rms = np.mean(val_rms)
        rms_diff = abs(train_avg_rms - val_avg_rms) / train_avg_rms * 100
        
        if rms_diff > 30:
            issues.append(f"⚠️  Audio quality mismatch ({rms_diff:.1f}% difference)")
            print(f"3. CHECK AUDIO QUALITY: Validation audio quality significantly different")
            print(f"   → Verify recording conditions are similar")
    
    # Check validation size
    val_ratio = val_stats['total'] / train_stats['total'] * 100
    if val_ratio < 5:
        issues.append(f"⚠️  Validation set too small ({val_ratio:.1f}%)")
        print(f"4. INCREASE VALIDATION SIZE: Currently only {val_ratio:.1f}% of data")
        print(f"   → Aim for 10-15% validation split")
    elif val_ratio > 25:
        issues.append(f"⚠️  Validation set too large ({val_ratio:.1f}%)")
        print(f"5. DECREASE VALIDATION SIZE: Currently {val_ratio:.1f}% of data")
        print(f"   → Aim for 10-15% validation split")
    
    if not issues:
        print(f"✅ No significant data quality issues detected")
        print(f"\nIf eval loss is still high, consider:")
        print(f"  - Checking if validation samples are from same speaker(s)")
        print(f"  - Verifying G2P preprocessing consistency")
        print(f"  - Trying different random split")
    else:
        print(f"\n🚨 SUMMARY: {len(issues)} issues detected")
        for issue in issues:
            print(f"  {issue}")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Default paths (adjust if needed)
    train_csv = "finetune_models/metadata_train.csv"
    val_csv = "finetune_models/metadata_eval.csv"
    
    # Check if files exist
    if not os.path.exists(train_csv):
        print(f"❌ Train CSV not found: {train_csv}")
        print(f"Please specify the correct path")
        exit(1)
    
    if not os.path.exists(val_csv):
        print(f"❌ Validation CSV not found: {val_csv}")
        print(f"Please specify the correct path")
        exit(1)
    
    diagnose(train_csv, val_csv)
