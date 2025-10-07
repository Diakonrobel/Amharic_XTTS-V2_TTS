"""
Amharic TTS Enhancement Module

This module provides comprehensive Amharic language support for XTTS v2,
including G2P (Grapheme-to-Phoneme) conversion, Ethiopic script tokenization,
and text preprocessing.
"""

__version__ = "0.1.0"
__author__ = "XTTS Amharic Enhancement Team"

from .g2p.amharic_g2p import AmharicG2P
from .preprocessing.text_normalizer import AmharicTextNormalizer
from .preprocessing.number_expander import expand_amharic_numbers

__all__ = [
    "AmharicG2P",
    "AmharicTextNormalizer", 
    "expand_amharic_numbers",
]
