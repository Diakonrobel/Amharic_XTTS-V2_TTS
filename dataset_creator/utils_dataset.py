"""
Dataset utilities - Helper functions for dataset management

Features:
- Calculate dataset statistics
- Validate audio quality
- Export dataset in multiple formats
- Duration formatting
"""

import os
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List
import pandas as pd
import librosa
import numpy as np


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def calculate_dataset_statistics(project_path: Path) -> Dict:
    """Calculate comprehensive dataset statistics"""
    
    project_path = Path(project_path)
    
    # Read metadata
    train_csv = project_path / "metadata_train.csv"
    
    if not train_csv.exists():
        return {
            "total_segments": 0,
            "total_duration": 0,
            "avg_duration": 0,
            "min_duration": 0,
            "max_duration": 0,
            "total_characters": 0,
            "total_words": 0,
            "avg_words": 0,
            "num_speakers": 0,
            "speakers": [],
            "total_size_mb": 0
        }
    
    df = pd.read_csv(train_csv, sep="|")
    
    # Audio statistics
    wavs_dir = project_path / "wavs"
    audio_files = list(wavs_dir.glob("*.wav"))
    
    durations = []
    total_size = 0
    
    for audio_file in audio_files:
        try:
            # Get duration
            duration = librosa.get_duration(path=str(audio_file))
            durations.append(duration)
            
            # Get file size
            total_size += audio_file.stat().st_size
        except:
            continue
    
    # Text statistics
    all_text = " ".join(df["text"].astype(str))
    total_characters = len(all_text)
    total_words = len(all_text.split())
    
    # Speaker statistics
    speakers = df["speaker_name"].unique().tolist()
    
    return {
        "total_segments": len(df),
        "total_duration": sum(durations),
        "avg_duration": np.mean(durations) if durations else 0,
        "min_duration": min(durations) if durations else 0,
        "max_duration": max(durations) if durations else 0,
        "total_characters": total_characters,
        "total_words": total_words,
        "avg_words": total_words / len(df) if len(df) > 0 else 0,
        "num_speakers": len(speakers),
        "speakers": speakers,
        "total_size_mb": total_size / (1024 * 1024)
    }


def validate_audio_quality(audio_path: str, min_snr: float = 10.0) -> Dict:
    """Validate audio quality metrics"""
    
    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        avg_rms = np.mean(rms)
        
        # Check for clipping
        clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
        
        # Estimate SNR (simplified)
        signal_power = np.mean(audio ** 2)
        noise_estimate = np.mean(rms[rms < np.percentile(rms, 20)]) ** 2
        snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
        
        # Calculate spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        
        return {
            "is_valid": snr >= min_snr and clipping_ratio < 0.01,
            "snr": float(snr),
            "avg_rms": float(avg_rms),
            "clipping_ratio": float(clipping_ratio),
            "spectral_centroid_mean": float(np.mean(spectral_centroids)),
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
            "duration": len(audio) / sr
        }
    except Exception as e:
        return {
            "is_valid": False,
            "error": str(e)
        }


def export_dataset(
    project_path: Path,
    format_type: str = "csv",
    include_audio: bool = True
) -> str:
    """Export dataset in specified format"""
    
    project_path = Path(project_path)
    export_dir = project_path / "exports"
    export_dir.mkdir(exist_ok=True)
    
    train_csv = project_path / "metadata_train.csv"
    
    if not train_csv.exists():
        raise ValueError("No metadata found")
    
    df = pd.read_csv(train_csv, sep="|")
    
    if format_type == "csv":
        # Export as CSV
        output_path = export_dir / "dataset.csv"
        df.to_csv(output_path, index=False)
        return str(output_path)
    
    elif format_type == "json":
        # Export as JSON
        output_path = export_dir / "dataset.json"
        data = df.to_dict(orient="records")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return str(output_path)
    
    elif format_type == "metadata.txt":
        # Export as LJSpeech-style metadata
        output_path = export_dir / "metadata.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                # Format: filename|text
                filename = Path(row["audio_file"]).stem
                f.write(f"{filename}|{row['text']}\\n")
        return str(output_path)
    
    elif format_type == "ljspeech":
        # Export complete LJSpeech-compatible dataset
        ljspeech_dir = export_dir / "ljspeech"
        ljspeech_dir.mkdir(exist_ok=True)
        
        wavs_output = ljspeech_dir / "wavs"
        wavs_output.mkdir(exist_ok=True)
        
        # Copy audio files
        if include_audio:
            wavs_dir = project_path / "wavs"
            for audio_file in wavs_dir.glob("*.wav"):
                shutil.copy(audio_file, wavs_output / audio_file.name)
        
        # Create metadata.csv
        metadata_path = ljspeech_dir / "metadata.csv"
        with open(metadata_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                filename = Path(row["audio_file"]).stem
                f.write(f"{filename}|{row['text']}|{row['text']}\\n")
        
        # Create README
        readme_path = ljspeech_dir / "README.txt"
        with open(readme_path, "w") as f:
            f.write("LJSpeech-compatible dataset\\n")
            f.write(f"Total segments: {len(df)}\\n")
            f.write(f"\\nFormat: filename|text|normalized_text\\n")
        
        # Create zip archive
        zip_path = export_dir / "ljspeech_dataset.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in ljspeech_dir.rglob("*"):
                if file.is_file():
                    zipf.write(file, file.relative_to(ljspeech_dir.parent))
        
        return str(zip_path)
    
    else:
        raise ValueError(f"Unknown format: {format_type}")


def split_train_eval(project_path: Path, eval_ratio: float = 0.15):
    """Split dataset into train and eval sets"""
    
    project_path = Path(project_path)
    train_csv = project_path / "metadata_train.csv"
    
    if not train_csv.exists():
        return
    
    df = pd.read_csv(train_csv, sep="|")
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split
    eval_size = int(len(df) * eval_ratio)
    eval_df = df[:eval_size]
    train_df = df[eval_size:]
    
    # Save
    train_df.to_csv(train_csv, sep="|", index=False)
    eval_df.to_csv(project_path / "metadata_eval.csv", sep="|", index=False)
    
    print(f"✅ Split complete: {len(train_df)} train, {len(eval_df)} eval")


def validate_dataset(project_path: Path) -> Dict:
    """Validate entire dataset and return report"""
    
    project_path = Path(project_path)
    train_csv = project_path / "metadata_train.csv"
    
    if not train_csv.exists():
        return {"valid": False, "error": "No metadata found"}
    
    df = pd.read_csv(train_csv, sep="|")
    wavs_dir = project_path / "wavs"
    
    issues = []
    valid_segments = 0
    
    for _, row in df.iterrows():
        audio_path = project_path / row["audio_file"]
        
        # Check if file exists
        if not audio_path.exists():
            issues.append(f"Missing audio: {row['audio_file']}")
            continue
        
        # Check text
        if not row["text"] or len(str(row["text"]).strip()) < 3:
            issues.append(f"Invalid text: {row['audio_file']}")
            continue
        
        # Validate audio
        quality = validate_audio_quality(str(audio_path))
        if not quality.get("is_valid", False):
            issues.append(f"Low quality audio: {row['audio_file']}")
            continue
        
        valid_segments += 1
    
    return {
        "valid": len(issues) == 0,
        "total_segments": len(df),
        "valid_segments": valid_segments,
        "issues": issues[:10],  # Limit to first 10 issues
        "total_issues": len(issues)
    }
