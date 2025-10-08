"""
VAD-Enhanced Audio Slicer
Combines Voice Activity Detection with word boundary detection for precise audio segmentation.
Integrates with SRT/VTT timestamps when available.
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch
import torchaudio
from dataclasses import dataclass


@dataclass
class AudioSegment:
    """Represents an audio segment with metadata"""
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    audio: np.ndarray  # Audio samples
    text: Optional[str] = None  # Transcription/subtitle text
    confidence: float = 1.0  # Confidence score for the segment


class VADSlicer:
    """
    Voice Activity Detection-based audio slicer with word boundary detection.
    
    Features:
    - Uses Silero VAD for accurate voice detection
    - Aligns cuts to word boundaries when transcription is available
    - Respects SRT/VTT timestamp hints
    - Prevents cutting in the middle of words
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 30,
        vad_threshold: float = 0.5,
        max_segment_duration: float = 15.0,
        min_segment_duration: float = 1.0,
        use_onnx: bool = False
    ):
        """
        Initialize VAD-based slicer.
        
        Args:
            sample_rate: Audio sample rate
            min_speech_duration_ms: Minimum speech chunk duration (ms)
            min_silence_duration_ms: Minimum silence duration to split (ms)
            speech_pad_ms: Padding around speech segments (ms)
            vad_threshold: VAD confidence threshold (0-1)
            max_segment_duration: Maximum segment duration (seconds)
            min_segment_duration: Minimum segment duration (seconds)
            use_onnx: Use ONNX runtime for faster inference
        """
        self.sample_rate = sample_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.vad_threshold = vad_threshold
        self.max_segment_duration = max_segment_duration
        self.min_segment_duration = min_segment_duration
        self.use_onnx = use_onnx
        
        # Load VAD model
        self.vad_model = None
        self._load_vad_model()
    
    def _load_vad_model(self):
        """Load Silero VAD model"""
        try:
            import torch
            
            if self.use_onnx:
                # Try ONNX runtime first (faster)
                try:
                    model, utils = torch.hub.load(
                        repo_or_dir='snakers4/silero-vad',
                        model='silero_vad',
                        force_reload=False,
                        onnx=True
                    )
                    print("✓ Loaded Silero VAD (ONNX)")
                except Exception as onnx_error:
                    print(f"ONNX load failed, falling back to PyTorch: {onnx_error}")
                    model, utils = torch.hub.load(
                        repo_or_dir='snakers4/silero-vad',
                        model='silero_vad',
                        force_reload=False,
                        onnx=False
                    )
                    print("✓ Loaded Silero VAD (PyTorch)")
            else:
                # Use PyTorch model
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                print("✓ Loaded Silero VAD (PyTorch)")
            
            self.vad_model = model
            self.vad_utils = utils
            
        except Exception as e:
            print(f"⚠ Failed to load Silero VAD: {e}")
            print("  VAD slicing will fall back to energy-based detection")
            self.vad_model = None
    
    def detect_speech_segments(self, audio: np.ndarray) -> List[Dict]:
        """
        Detect speech segments using VAD.
        
        Args:
            audio: Audio waveform (numpy array)
            
        Returns:
            List of dictionaries with 'start', 'end' (in seconds), and 'confidence'
        """
        if self.vad_model is None:
            # Fallback to energy-based detection
            return self._energy_based_detection(audio)
        
        try:
            # Ensure audio is torch tensor
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio
            
            # Get timestamps from VAD
            (get_speech_timestamps, _, read_audio, _, _) = self.vad_utils
            
            # Resample if needed (VAD expects 16kHz)
            vad_sample_rate = 16000
            if self.sample_rate != vad_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=self.sample_rate,
                    new_freq=vad_sample_rate
                )
                audio_for_vad = resampler(audio_tensor)
            else:
                audio_for_vad = audio_tensor
            
            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(
                audio_for_vad,
                self.vad_model,
                threshold=self.vad_threshold,
                sampling_rate=vad_sample_rate,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms
            )
            
            # Convert to seconds in original sample rate
            segments = []
            for ts in speech_timestamps:
                start_sample = int(ts['start'] * self.sample_rate / vad_sample_rate)
                end_sample = int(ts['end'] * self.sample_rate / vad_sample_rate)
                
                segments.append({
                    'start': start_sample / self.sample_rate,
                    'end': end_sample / self.sample_rate,
                    'confidence': 1.0  # Silero doesn't provide confidence per segment
                })
            
            return segments
            
        except Exception as e:
            print(f"⚠ VAD detection failed: {e}, falling back to energy-based")
            return self._energy_based_detection(audio)
    
    def _energy_based_detection(self, audio: np.ndarray) -> List[Dict]:
        """
        Fallback energy-based speech detection.
        
        Args:
            audio: Audio waveform
            
        Returns:
            List of speech segments
        """
        from utils.audio_slicer import get_rms
        
        # Calculate RMS energy
        hop_length = int(self.sample_rate * 0.02)  # 20ms hop
        frame_length = int(self.sample_rate * 0.05)  # 50ms frame
        
        rms = get_rms(audio, frame_length=frame_length, hop_length=hop_length).squeeze()
        
        # Dynamic threshold (mean + 0.5 * std)
        threshold = np.mean(rms) + 0.5 * np.std(rms)
        
        # Find speech regions
        is_speech = rms > threshold
        
        # Convert to segments
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                # Start of speech
                start_frame = i
                in_speech = True
            elif not speech and in_speech:
                # End of speech
                end_frame = i
                
                start_time = start_frame * hop_length / self.sample_rate
                end_time = end_frame * hop_length / self.sample_rate
                
                # Check minimum duration
                if (end_time - start_time) * 1000 >= self.min_speech_duration_ms:
                    segments.append({
                        'start': start_time,
                        'end': end_time,
                        'confidence': 0.8  # Lower confidence for energy-based
                    })
                
                in_speech = False
        
        # Handle last segment
        if in_speech:
            end_time = len(audio) / self.sample_rate
            start_time = start_frame * hop_length / self.sample_rate
            if (end_time - start_time) * 1000 >= self.min_speech_duration_ms:
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'confidence': 0.8
                })
        
        return segments
    
    def align_to_word_boundaries(
        self,
        segments: List[Dict],
        word_timestamps: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Align segment boundaries to word boundaries when available.
        
        Args:
            segments: VAD-detected segments
            word_timestamps: List of {'start': float, 'end': float, 'word': str}
            
        Returns:
            Adjusted segments aligned to word boundaries
        """
        if not word_timestamps:
            return segments
        
        aligned_segments = []
        
        for seg in segments:
            seg_start = seg['start']
            seg_end = seg['end']
            
            # Find closest word boundary for start
            closest_word_start = None
            min_start_diff = float('inf')
            
            for word in word_timestamps:
                diff = abs(word['start'] - seg_start)
                if diff < min_start_diff:
                    min_start_diff = diff
                    closest_word_start = word['start']
            
            # Find closest word boundary for end
            closest_word_end = None
            min_end_diff = float('inf')
            
            for word in word_timestamps:
                diff = abs(word['end'] - seg_end)
                if diff < min_end_diff:
                    min_end_diff = diff
                    closest_word_end = word['end']
            
            # Use aligned boundaries if they're close (within 200ms)
            final_start = closest_word_start if min_start_diff < 0.2 else seg_start
            final_end = closest_word_end if min_end_diff < 0.2 else seg_end
            
            aligned_segments.append({
                'start': final_start,
                'end': final_end,
                'confidence': seg['confidence']
            })
        
        return aligned_segments
    
    def merge_short_segments(
        self,
        segments: List[Dict],
        max_gap: float = 0.5
    ) -> List[Dict]:
        """
        Merge segments that are too short or too close together.
        
        Args:
            segments: List of segments
            max_gap: Maximum gap to merge (seconds)
            
        Returns:
            Merged segments
        """
        if not segments:
            return []
        
        merged = []
        current = segments[0].copy()
        
        for next_seg in segments[1:]:
            gap = next_seg['start'] - current['end']
            current_duration = current['end'] - current['start']
            
            # Merge if gap is small or current segment is too short
            if gap <= max_gap or current_duration < self.min_segment_duration:
                # Extend current segment
                current['end'] = next_seg['end']
                current['confidence'] = (current['confidence'] + next_seg['confidence']) / 2
            else:
                # Save current and start new
                if current_duration >= self.min_segment_duration:
                    merged.append(current)
                current = next_seg.copy()
        
        # Add last segment
        if (current['end'] - current['start']) >= self.min_segment_duration:
            merged.append(current)
        
        return merged
    
    def split_long_segments(
        self,
        segments: List[Dict],
        audio: np.ndarray,
        word_timestamps: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Split segments that are too long at natural pauses or word boundaries.
        
        Args:
            segments: List of segments
            audio: Full audio waveform
            word_timestamps: Optional word-level timestamps
            
        Returns:
            Segments with long ones split
        """
        result = []
        
        for seg in segments:
            duration = seg['end'] - seg['start']
            
            if duration <= self.max_segment_duration:
                result.append(seg)
                continue
            
            # Segment is too long, need to split
            # Try to find natural split points
            
            if word_timestamps:
                # Split at word boundaries
                split_segments = self._split_at_word_boundaries(
                    seg, word_timestamps
                )
                result.extend(split_segments)
            else:
                # Split at silence points
                split_segments = self._split_at_silence(
                    seg, audio
                )
                result.extend(split_segments)
        
        return result
    
    def _split_at_word_boundaries(
        self,
        segment: Dict,
        word_timestamps: List[Dict]
    ) -> List[Dict]:
        """Split long segment at word boundaries"""
        target_duration = self.max_segment_duration * 0.8  # 80% of max
        
        # Find words within this segment
        words_in_segment = [
            w for w in word_timestamps
            if w['start'] >= segment['start'] and w['end'] <= segment['end']
        ]
        
        if not words_in_segment:
            # No word info, split at midpoint
            mid = (segment['start'] + segment['end']) / 2
            return [
                {'start': segment['start'], 'end': mid, 'confidence': segment['confidence']},
                {'start': mid, 'end': segment['end'], 'confidence': segment['confidence']}
            ]
        
        # Find best split point (closest to target duration)
        best_split_idx = 0
        best_diff = float('inf')
        
        accumulated_duration = 0
        for i, word in enumerate(words_in_segment):
            accumulated_duration = word['end'] - segment['start']
            diff = abs(accumulated_duration - target_duration)
            
            if diff < best_diff:
                best_diff = diff
                best_split_idx = i
        
        # Split at chosen word boundary
        if best_split_idx > 0 and best_split_idx < len(words_in_segment) - 1:
            split_time = words_in_segment[best_split_idx]['end']
            
            return [
                {'start': segment['start'], 'end': split_time, 'confidence': segment['confidence']},
                {'start': split_time, 'end': segment['end'], 'confidence': segment['confidence']}
            ]
        else:
            return [segment]
    
    def _split_at_silence(
        self,
        segment: Dict,
        audio: np.ndarray
    ) -> List[Dict]:
        """Split long segment at silence points"""
        target_duration = self.max_segment_duration * 0.8
        
        start_sample = int(segment['start'] * self.sample_rate)
        end_sample = int(segment['end'] * self.sample_rate)
        segment_audio = audio[start_sample:end_sample]
        
        # Find local minima in energy as potential split points
        from utils.audio_slicer import get_rms
        
        hop_length = int(self.sample_rate * 0.02)
        rms = get_rms(segment_audio, hop_length=hop_length).squeeze()
        
        # Find target split frame
        target_frame = int(target_duration * self.sample_rate / hop_length)
        target_frame = min(target_frame, len(rms) - 1)
        
        # Find minimum energy within ±0.5s of target
        search_radius = int(0.5 * self.sample_rate / hop_length)
        search_start = max(0, target_frame - search_radius)
        search_end = min(len(rms), target_frame + search_radius)
        
        if search_end > search_start:
            local_rms = rms[search_start:search_end]
            min_idx = np.argmin(local_rms)
            split_frame = search_start + min_idx
            
            split_time = segment['start'] + (split_frame * hop_length / self.sample_rate)
            
            return [
                {'start': segment['start'], 'end': split_time, 'confidence': segment['confidence']},
                {'start': split_time, 'end': segment['end'], 'confidence': segment['confidence']}
            ]
        
        # Fallback to midpoint
        mid = (segment['start'] + segment['end']) / 2
        return [
            {'start': segment['start'], 'end': mid, 'confidence': segment['confidence']},
            {'start': mid, 'end': segment['end'], 'confidence': segment['confidence']}
        ]
    
    def slice_audio(
        self,
        audio: np.ndarray,
        word_timestamps: Optional[List[Dict]] = None,
        srt_segments: Optional[List[Tuple[float, float, str]]] = None
    ) -> List[AudioSegment]:
        """
        Main slicing method combining VAD, word boundaries, and SRT hints.
        
        Args:
            audio: Audio waveform (numpy array)
            word_timestamps: Optional word-level timestamps
            srt_segments: Optional SRT segments (start, end, text)
            
        Returns:
            List of AudioSegment objects
        """
        print("\n🎤 VAD-Enhanced Audio Slicing...")
        
        # Step 1: Detect speech segments with VAD
        print("  Step 1: Detecting speech with VAD...")
        vad_segments = self.detect_speech_segments(audio)
        print(f"    Found {len(vad_segments)} speech segments")
        
        # Step 2: Align to word boundaries if available
        if word_timestamps:
            print("  Step 2: Aligning to word boundaries...")
            vad_segments = self.align_to_word_boundaries(vad_segments, word_timestamps)
        
        # Step 3: Merge short/close segments
        print("  Step 3: Merging short segments...")
        vad_segments = self.merge_short_segments(vad_segments)
        print(f"    After merging: {len(vad_segments)} segments")
        
        # Step 4: Split long segments
        print("  Step 4: Splitting long segments...")
        vad_segments = self.split_long_segments(vad_segments, audio, word_timestamps)
        print(f"    After splitting: {len(vad_segments)} segments")
        
        # Step 5: Extract audio and create AudioSegment objects
        print("  Step 5: Extracting audio segments...")
        result = []
        
        for seg in vad_segments:
            start_sample = int(seg['start'] * self.sample_rate)
            end_sample = int(seg['end'] * self.sample_rate)
            
            # Ensure valid range
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            segment_audio = audio[start_sample:end_sample]
            
            # Find matching SRT text if available
            text = None
            if srt_segments:
                for srt_start, srt_end, srt_text in srt_segments:
                    # Check if there's significant overlap
                    overlap_start = max(seg['start'], srt_start)
                    overlap_end = min(seg['end'], srt_end)
                    overlap = overlap_end - overlap_start
                    
                    if overlap > 0:
                        segment_duration = seg['end'] - seg['start']
                        if overlap / segment_duration > 0.5:  # >50% overlap
                            text = srt_text
                            break
            
            audio_segment = AudioSegment(
                start_time=seg['start'],
                end_time=seg['end'],
                audio=segment_audio,
                text=text,
                confidence=seg['confidence']
            )
            
            result.append(audio_segment)
        
        print(f"\n✓ Slicing complete: {len(result)} segments")
        return result


def slice_audio_with_vad(
    audio_path: str,
    output_dir: str,
    sample_rate: int = 22050,
    min_segment_duration: float = 1.0,
    max_segment_duration: float = 15.0,
    vad_threshold: float = 0.5,
    word_timestamps: Optional[List[Dict]] = None,
    srt_segments: Optional[List[Tuple[float, float, str]]] = None
) -> List[str]:
    """
    Convenience function to slice audio file with VAD.
    
    Args:
        audio_path: Path to audio file
        output_dir: Output directory for segments
        sample_rate: Target sample rate
        min_segment_duration: Minimum segment duration (seconds)
        max_segment_duration: Maximum segment duration (seconds)
        vad_threshold: VAD confidence threshold
        word_timestamps: Optional word-level timestamps
        srt_segments: Optional SRT segments
        
    Returns:
        List of output file paths
    """
    import librosa
    import soundfile as sf
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # Initialize VAD slicer
    slicer = VADSlicer(
        sample_rate=sample_rate,
        min_segment_duration=min_segment_duration,
        max_segment_duration=max_segment_duration,
        vad_threshold=vad_threshold
    )
    
    # Slice audio
    segments = slicer.slice_audio(y, word_timestamps, srt_segments)
    
    # Save segments
    output_paths = []
    base_name = Path(audio_path).stem
    
    for i, segment in enumerate(segments):
        filename = f"{base_name}_vad_{str(i).zfill(4)}.wav"
        filepath = output_path / filename
        
        sf.write(str(filepath), segment.audio, sample_rate)
        output_paths.append(str(filepath))
        
        duration = segment.end_time - segment.start_time
        print(f"  Saved: {filename} ({duration:.2f}s, conf={segment.confidence:.2f})")
    
    return output_paths
