"""Amharic Configuration Module.

This package exposes configuration helpers and presets used across the
Amharic XTTS stack, including tokenizer, G2P backend ordering, and other
runtime hints.
"""

from .amharic_config import (  # re-export key classes/constants
    AmharicTTSConfig,
    DEFAULT_CONFIG,
    G2PBackend,
    G2PConfiguration,
    G2PQualityThresholds,
    PRESET_CONFIGS,
    TokenizerConfiguration,
    TokenizerMode,
    get_config,
)

__all__ = [
    "AmharicTTSConfig",
    "DEFAULT_CONFIG",
    "G2PBackend",
    "G2PConfiguration",
    "G2PQualityThresholds",
    "PRESET_CONFIGS",
    "TokenizerConfiguration",
    "TokenizerMode",
    "get_config",
]
