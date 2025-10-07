"""Text preprocessing module for Amharic."""

from .text_normalizer import AmharicTextNormalizer, normalize_amharic_text
from .number_expander import expand_amharic_numbers, AmharicNumberExpander

__all__ = [
    "AmharicTextNormalizer",
    "normalize_amharic_text",
    "expand_amharic_numbers",
    "AmharicNumberExpander",
]
