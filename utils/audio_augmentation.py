#!/usr/bin/env python3
"""
Audio Augmentation for XTTS v2 Training
========================================

Lightweight audio augmentation to increase effective dataset size.
Designed to work with small datasets without external dependencies.
"""

import torch
import torchaudio
import random
import numpy as np
from typing import Tuple, Optional


class AudioAugmenter:
    """
    Simple audio augmentation for TTS training.
    
    Implements:
    - Pitch shifting
    - Time stretching
    - Background noise injection
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        pitch_shift_prob: float = 0.3,
        pitch_shift_range: Tuple[int, int] = (-2, 2),
        time_stretch_prob: float = 0.3,
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        noise_prob: float = 0.2,
        noise_level_range: Tuple[float, float] = (0.001, 0.01),
    ):
        self.sample_rate = sample_rate
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range
        self.noise_prob = noise_prob
        self.noise_level_range = noise_level_range
    
    def augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to waveform.
        
        Args:
            waveform: Audio tensor (channels, samples)
            
        Returns:
            Augmented waveform
        """
        # Pitch shift
        if random.random() < self.pitch_shift_prob:
            waveform = self._pitch_shift(waveform)
        
        # Time stretch
        if random.random() < self.time_stretch_prob:
            waveform = self._time_stretch(waveform)
        
        # Add noise
        if random.random() < self.noise_prob:
            waveform = self._add_noise(waveform)
        
        return waveform
    
    def _pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Shift pitch by random semitones.
        
        Uses simple resampling method for speed.
        """
        try:
            n_steps = random.randint(*self.pitch_shift_range)
            if n_steps == 0:
                return waveform
            
            # Pitch shift using resampling
            # Factor = 2^(n_steps/12)
            factor = 2 ** (n_steps / 12.0)
            new_sample_rate = int(self.sample_rate * factor)
            
            # Resample to shift pitch
            resampler = torchaudio.transforms.Resample(
                orig_freq=self.sample_rate,
                new_freq=new_sample_rate
            )
            shifted = resampler(waveform)
            
            # Resample back to original rate
            resampler_back = torchaudio.transforms.Resample(
                orig_freq=new_sample_rate,
                new_freq=self.sample_rate
            )
            return resampler_back(shifted)
        
        except Exception as e:
            # Return original on error
            return waveform
    
    def _time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Stretch or compress time by random factor.
        """
        try:
            factor = random.uniform(*self.time_stretch_range)
            if abs(factor - 1.0) < 0.01:  # Skip if factor is ~1.0
                return waveform
            
            # Time stretch using interpolation
            original_length = waveform.shape[-1]
            new_length = int(original_length / factor)
            
            # Use linear interpolation
            stretched = torch.nn.functional.interpolate(
                waveform.unsqueeze(0),
                size=new_length,
                mode='linear',
                align_corners=False
            ).squeeze(0)
            
            # Pad or trim to original length
            if stretched.shape[-1] < original_length:
                # Pad
                pad_length = original_length - stretched.shape[-1]
                stretched = torch.nn.functional.pad(stretched, (0, pad_length))
            elif stretched.shape[-1] > original_length:
                # Trim
                stretched = stretched[..., :original_length]
            
            return stretched
        
        except Exception as e:
            return waveform
    
    def _add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Add subtle background noise.
        """
        try:
            noise_level = random.uniform(*self.noise_level_range)
            noise = torch.randn_like(waveform) * noise_level
            return waveform + noise
        
        except Exception as e:
            return waveform


# Simpler version that just adds noise (most effective, least risky)
class SimpleAudioAugmenter:
    """
    Simplified augmenter that only adds noise.
    Most stable and effective for TTS training.
    """
    
    def __init__(
        self,
        noise_prob: float = 0.3,
        noise_level_range: Tuple[float, float] = (0.001, 0.01),
    ):
        self.noise_prob = noise_prob
        self.noise_level_range = noise_level_range
    
    def augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add subtle noise to waveform."""
        if random.random() < self.noise_prob:
            noise_level = random.uniform(*self.noise_level_range)
            noise = torch.randn_like(waveform) * noise_level
            return waveform + noise
        return waveform
