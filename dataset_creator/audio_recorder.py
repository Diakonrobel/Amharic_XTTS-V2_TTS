"""
Audio Recorder - Handle microphone input and recording

Features:
- Record audio from microphone
- Simple interface for Gradio integration
- Automatic file management
"""

import tempfile
import soundfile as sf
import numpy as np


class AudioRecorder:
    """Simple audio recorder for dataset creation"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.recording_count = 0
    
    def record_segment(self, audio_data=None) -> str:
        """
        Record or save an audio segment
        
        Args:
            audio_data: Audio data from Gradio (tuple of sr, audio array)
        
        Returns:
            Path to saved audio file
        """
        if audio_data is None:
            raise ValueError("No audio data provided")
        
        # Gradio returns (sample_rate, audio_array)
        if isinstance(audio_data, tuple):
            sr, audio = audio_data
        else:
            # If it's already a file path
            return audio_data
        
        # Save to temp file
        self.recording_count += 1
        output_path = f"{self.temp_dir}/recording_{self.recording_count:04d}.wav"
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Save
        sf.write(output_path, audio, sr)
        
        return output_path
