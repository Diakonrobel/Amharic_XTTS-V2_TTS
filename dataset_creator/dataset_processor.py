"""
Dataset Processor - Core processing logic for audio dataset creation

Features:
- YouTube video processing with yt-dlp
- Audio file segmentation with VAD
- Automatic transcription with Faster Whisper
- Quality validation and filtering
- Multi-format export
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Callable
import librosa
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
import torch
import torchaudio

# Add parent utils to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils import youtube_downloader
    from utils import audio_slicer
except ImportError:
    print("Warning: Could not import parent utils")


class DatasetProcessor:
    """Main dataset processing class"""
    
    def __init__(self, output_dir="./datasets", whisper_model="large-v3"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.whisper_model_name = whisper_model
        self.whisper_model = None
        self.current_project = None
        
    def set_project(self, project_path: str):
        """Set the current project directory"""
        self.current_project = Path(project_path)
        
    def _load_whisper_model(self):
        """Lazy load Whisper model"""
        if self.whisper_model is None:
            print(f"📥 Loading Whisper model: {self.whisper_model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            self.whisper_model = WhisperModel(
                self.whisper_model_name,
                device=device,
                compute_type=compute_type
            )
            print(f"✅ Whisper model loaded on {device}")
        
        return self.whisper_model
    
    def process_youtube_url(
        self,
        url: str,
        language: str,
        speaker_name: str,
        min_duration: float = 1.0,
        max_duration: float = 15.0,
        quality_threshold: float = 0.7,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """Process YouTube video to create dataset"""
        
        if progress_callback:
            progress_callback(0.1, "Downloading audio from YouTube...")
        
        # Download audio
        temp_dir = tempfile.mkdtemp()
        try:
            audio_path = self._download_youtube_audio(url, temp_dir)
            
            if progress_callback:
                progress_callback(0.3, "Processing audio file...")
            
            # Process the audio file
            result = self.process_audio_file(
                audio_path=audio_path,
                language=language,
                speaker_name=speaker_name,
                min_duration=min_duration,
                max_duration=max_duration,
                quality_threshold=quality_threshold,
                progress_callback=progress_callback
            )
            
            return result
            
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _download_youtube_audio(self, url: str, output_dir: str) -> str:
        """Download audio from YouTube"""
        try:
            # Try using the parent utils if available
            if 'youtube_downloader' in sys.modules:
                result = youtube_downloader.download_youtube_audio(
                    url=url,
                    output_dir=output_dir,
                    format="wav"
                )
                return result['audio_path']
        except Exception as e:
            print(f"Parent downloader failed: {e}, using fallback...")
        
        # Fallback to yt-dlp directly
        import yt_dlp
        
        output_path = Path(output_dir) / "audio.wav"
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(output_path.with_suffix('')),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        return str(output_path)
    
    def process_audio_file(
        self,
        audio_path: str,
        language: str,
        speaker_name: str,
        min_duration: float = 1.0,
        max_duration: float = 15.0,
        quality_threshold: float = 0.7,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """Process a single audio file"""
        
        if not self.current_project:
            raise ValueError("No project set. Call set_project() first.")
        
        if progress_callback:
            progress_callback(0.4, "Loading audio file...")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        if progress_callback:
            progress_callback(0.5, "Segmenting audio...")
        
        # Segment audio using VAD
        segments = self._segment_audio(audio, sr, min_duration, max_duration)
        
        if progress_callback:
            progress_callback(0.6, "Transcribing segments...")
        
        # Transcribe segments
        model = self._load_whisper_model()
        
        segments_created = 0
        total_duration = 0
        wavs_dir = self.current_project / "wavs"
        wavs_dir.mkdir(exist_ok=True)
        
        # Get next segment number
        existing_wavs = list(wavs_dir.glob("*.wav"))
        next_idx = len(existing_wavs)
        
        metadata_entries = []
        
        for i, (start, end) in enumerate(segments):
            if progress_callback:
                progress_callback(
                    0.6 + 0.3 * (i / len(segments)),
                    f"Transcribing segment {i+1}/{len(segments)}..."
                )
            
            # Extract segment
            segment_audio = audio[int(start * sr):int(end * sr)]
            duration = len(segment_audio) / sr
            
            # Filter by duration
            if duration < min_duration or duration > max_duration:
                continue
            
            # Check quality
            if not self._check_audio_quality(segment_audio, sr, quality_threshold):
                continue
            
            # Save temporary file for transcription
            temp_segment_path = tempfile.mktemp(suffix=".wav")
            sf.write(temp_segment_path, segment_audio, sr)
            
            try:
                # Transcribe
                segments_result, info = model.transcribe(
                    temp_segment_path,
                    language=language,
                    vad_filter=False,  # Already segmented
                    word_timestamps=False
                )
                
                text = " ".join([seg.text.strip() for seg in segments_result])
                
                if not text or len(text) < 3:
                    continue
                
                # Save segment
                segment_filename = f"segment_{next_idx:06d}.wav"
                segment_path = wavs_dir / segment_filename
                sf.write(str(segment_path), segment_audio, sr)
                
                # Add to metadata
                metadata_entries.append({
                    "audio_file": f"wavs/{segment_filename}",
                    "text": text,
                    "speaker_name": speaker_name,
                    "duration": duration
                })
                
                segments_created += 1
                total_duration += duration
                next_idx += 1
                
            finally:
                if os.path.exists(temp_segment_path):
                    os.remove(temp_segment_path)
        
        # Save metadata
        if metadata_entries:
            self._save_metadata(metadata_entries)
        
        if progress_callback:
            progress_callback(1.0, "Processing complete!")
        
        return {
            "segments_created": segments_created,
            "total_duration": total_duration,
            "summary": f"Created {segments_created} segments, total duration: {total_duration:.2f}s"
        }
    
    def _segment_audio(
        self,
        audio: np.ndarray,
        sr: int,
        min_duration: float,
        max_duration: float
    ) -> List[tuple]:
        """Segment audio using VAD and silence detection"""
        
        try:
            # Try using parent audio slicer if available
            if 'audio_slicer' in sys.modules:
                slicer = audio_slicer.Slicer(
                    sr=sr,
                    threshold=-40.0,
                    min_length=int(min_duration * 1000),
                    min_interval=300,
                    hop_size=10,
                    max_sil_kept=500
                )
                chunks = slicer.slice(audio)
                
                # Convert chunks to time segments
                segments = []
                current_pos = 0
                for chunk in chunks:
                    duration = len(chunk) / sr
                    if min_duration <= duration <= max_duration:
                        segments.append((current_pos, current_pos + duration))
                    current_pos += duration
                
                return segments
        except Exception as e:
            print(f"Slicer failed: {e}, using fallback...")
        
        # Fallback: simple energy-based segmentation
        return self._simple_segmentation(audio, sr, min_duration, max_duration)
    
    def _simple_segmentation(
        self,
        audio: np.ndarray,
        sr: int,
        min_duration: float,
        max_duration: float
    ) -> List[tuple]:
        """Simple energy-based segmentation"""
        
        # Calculate energy
        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = int(sr * 0.010)    # 10ms hop
        
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Threshold
        threshold = np.median(energy) * 0.3
        
        # Find speech regions
        is_speech = energy > threshold
        
        # Convert to time segments
        segments = []
        in_speech = False
        start = 0
        
        for i, speech in enumerate(is_speech):
            time = i * hop_length / sr
            
            if speech and not in_speech:
                start = time
                in_speech = True
            elif not speech and in_speech:
                duration = time - start
                if min_duration <= duration <= max_duration:
                    segments.append((start, time))
                in_speech = False
        
        # Handle last segment
        if in_speech:
            duration = len(audio) / sr - start
            if min_duration <= duration <= max_duration:
                segments.append((start, len(audio) / sr))
        
        return segments
    
    def _check_audio_quality(
        self,
        audio: np.ndarray,
        sr: int,
        threshold: float
    ) -> bool:
        """Check if audio segment meets quality standards"""
        
        # Check for silence
        rms = librosa.feature.rms(y=audio)[0]
        if np.mean(rms) < 0.01:
            return False
        
        # Check for clipping
        if np.max(np.abs(audio)) > 0.99:
            return False
        
        # Check SNR (simplified)
        signal_power = np.mean(audio ** 2)
        if signal_power < threshold * 0.01:
            return False
        
        return True
    
    def _save_metadata(self, entries: List[Dict]):
        """Save metadata entries to CSV"""
        import pandas as pd
        
        df = pd.DataFrame(entries)
        
        # Save to both train and eval
        train_path = self.current_project / "metadata_train.csv"
        eval_path = self.current_project / "metadata_eval.csv"
        
        # Append or create
        if train_path.exists():
            existing = pd.read_csv(train_path, sep="|")
            df = pd.concat([existing, df], ignore_index=True)
        
        df.to_csv(train_path, sep="|", index=False)
        
        # Copy to eval (will be split later)
        df.to_csv(eval_path, sep="|", index=False)
    
    def add_manual_segment(
        self,
        audio_path: str,
        text: str,
        speaker_name: str
    ) -> Dict:
        """Add a manually transcribed segment"""
        
        if not self.current_project:
            raise ValueError("No project set")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = len(audio) / sr
        
        # Get next segment number
        wavs_dir = self.current_project / "wavs"
        wavs_dir.mkdir(exist_ok=True)
        existing_wavs = list(wavs_dir.glob("*.wav"))
        next_idx = len(existing_wavs)
        
        # Save segment
        segment_filename = f"segment_{next_idx:06d}.wav"
        segment_path = wavs_dir / segment_filename
        sf.write(str(segment_path), audio, sr)
        
        # Save metadata
        self._save_metadata([{
            "audio_file": f"wavs/{segment_filename}",
            "text": text,
            "speaker_name": speaker_name,
            "duration": duration
        }])
        
        return {
            "segments_created": 1,
            "total_duration": duration
        }
