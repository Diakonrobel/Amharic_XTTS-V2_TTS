"""
Prosody Marker Handler for Amharic TTS

Preserves and enhances punctuation markers for natural pauses, emotions, and
prosody in TTS synthesis. Handles both Ethiopic and Latin punctuation.

Enterprise TTS Features:
- Pause duration markers
- Emotion/emphasis preservation
- Question intonation
- Exclamation energy
- Multi-sentence pacing
- Code-switching prosody consistency

Based on SOTA TTS practices:
- FastSpeech2: Explicit duration modeling
- Tacotron2: Attention-based prosody
- XTTS: Multilingual prosody transfer
"""

import re
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProsodyMarker(Enum):
    """Prosody marker types"""
    PAUSE_SHORT = "pause_short"       # , ፣
    PAUSE_MEDIUM = "pause_medium"     # ; ፤
    PAUSE_LONG = "pause_long"         # . ።
    PAUSE_PARAGRAPH = "pause_para"    # ¶ ፨
    
    QUESTION = "question"             # ? ፧
    EXCLAMATION = "exclamation"       # !
    EMPHASIS = "emphasis"             # Bold/caps context
    
    COLON = "colon"                   # : ፥
    QUOTE_START = "quote_start"       # " '
    QUOTE_END = "quote_end"           # " '


class ProsodyHandler:
    """
    Handle prosody markers for natural TTS synthesis
    
    Features:
    - Preserves punctuation for pause modeling
    - Adds prosody markers for emphasis
    - Code-switching prosody consistency
    - Emotion/intonation preservation
    
    Usage:
        handler = ProsodyHandler()
        
        # Preserve prosody
        text_with_markers = handler.add_markers("ሰላም! እንዴት ነህ?")
        
        # Clean for phoneme conversion (keeps markers)
        processed = handler.process_for_g2p(text)
        
        # Extract prosody info for TTS model
        prosody_info = handler.extract_prosody_info(text)
    """
    
    def __init__(self, preserve_ethiopic: bool = True):
        """
        Initialize prosody handler
        
        Args:
            preserve_ethiopic: Keep Ethiopic punctuation (recommended for Amharic)
        """
        self.preserve_ethiopic = preserve_ethiopic
        self._load_punctuation_mappings()
        self._load_prosody_rules()
    
    def _load_punctuation_mappings(self):
        """Load Ethiopic and Latin punctuation mappings"""
        
        # Ethiopic to Latin punctuation (for normalization)
        self.ethiopic_to_latin = {
            '።': '.',    # Full stop (serez)
            '፣': ',',    # Comma (nesib)
            '፤': ';',    # Semicolon (mekfel)
            '፥': ':',    # Colon (nebteb)
            '፦': '::',   # Preface colon
            '፧': '?',    # Question mark
            '፨': '¶',    # Paragraph separator
            '፡': ' ',    # Word separator
        }
        
        # Pause duration mapping (in relative units)
        self.pause_durations = {
            ',': 0.2,   # Short pause
            '፣': 0.2,
            ';': 0.3,   # Medium pause
            '፤': 0.3,
            ':': 0.3,
            '፥': 0.3,
            '.': 0.5,   # Long pause
            '።': 0.5,
            '!': 0.5,   # Exclamation pause
            '?': 0.5,   # Question pause
            '፧': 0.5,
            '¶': 0.8,   # Paragraph pause
            '፨': 0.8,
            '::': 0.4,  # Preface colon
            '፦': 0.4,
        }
        
        # Intonation markers
        self.intonation_markers = {
            '?': 'rising',     # Question intonation
            '፧': 'rising',
            '!': 'emphatic',   # Exclamation energy
            '.': 'falling',    # Statement falling
            '።': 'falling',
        }
    
    def _load_prosody_rules(self):
        """Load prosody enhancement rules"""
        
        # Patterns for emphasis detection
        self.emphasis_patterns = [
            (r'[A-Z]{2,}', 'caps_emphasis'),      # ALL CAPS
            (r'\*\*(.+?)\*\*', 'bold_emphasis'),  # **bold**
            (r'_(.+?)_', 'italic_emphasis'),      # _italic_
            (r'!+', 'multiple_exclaim'),          # Multiple !!!
            (r'\?+', 'multiple_question'),        # Multiple ???
        ]
        
        # Multi-sentence pacing rules
        self.sentence_spacing = {
            'single': 0.5,      # Single sentence end
            'multiple': 0.7,    # Multiple sentences
            'paragraph': 1.0,   # Paragraph break
        }
    
    def normalize_punctuation(self, text: str, to_latin: bool = True) -> str:
        """
        Normalize punctuation (Ethiopic ↔ Latin)
        
        Args:
            text: Input text
            to_latin: Convert Ethiopic → Latin (True) or preserve (False)
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        if to_latin:
            # Convert Ethiopic punctuation to Latin
            for ethiopic, latin in self.ethiopic_to_latin.items():
                text = text.replace(ethiopic, latin)
        
        return text
    
    def add_prosody_markers(self, text: str) -> str:
        """
        Add explicit prosody markers to text
        
        Markers format: <pause:0.5> <intonation:rising> <emphasis>
        
        Args:
            text: Input text with punctuation
            
        Returns:
            Text with prosody markers
            
        Example:
            "ሰላም! እንዴት ነህ?" → "ሰላም<pause:0.5><intonation:emphatic>! እንዴት ነህ<pause:0.5><intonation:rising>?"
        """
        if not text:
            return ""
        
        result = text
        
        # Add pause markers based on punctuation
        for punct, duration in self.pause_durations.items():
            if punct in result:
                # Add pause marker before punctuation
                marker = f'<pause:{duration}>'
                result = result.replace(punct, f'{marker}{punct}')
        
        # Add intonation markers
        for punct, intonation in self.intonation_markers.items():
            if punct in result:
                marker = f'<intonation:{intonation}>'
                # Insert before the punctuation
                pattern = re.escape(punct)
                result = re.sub(f'({pattern})', f'{marker}\\1', result)
        
        # Detect and mark emphasis
        for pattern, emphasis_type in self.emphasis_patterns:
            if re.search(pattern, result):
                result = re.sub(pattern, f'<emphasis:{emphasis_type}>\\g<0></emphasis>', result)
        
        return result
    
    def extract_prosody_info(self, text: str) -> Dict[str, List[Tuple[int, str, float]]]:
        """
        Extract prosody information from text
        
        Args:
            text: Input text with punctuation
            
        Returns:
            Dictionary with prosody events:
            {
                'pauses': [(position, type, duration), ...],
                'intonations': [(position, type), ...],
                'emphasis': [(start, end, type), ...]
            }
        """
        prosody_info = {
            'pauses': [],
            'intonations': [],
            'emphasis': [],
            'code_switches': []  # For multilingual handling
        }
        
        # Extract pause positions
        for punct, duration in self.pause_durations.items():
            for match in re.finditer(re.escape(punct), text):
                position = match.start()
                prosody_info['pauses'].append((position, punct, duration))
        
        # Extract intonation markers
        for punct, intonation in self.intonation_markers.items():
            for match in re.finditer(re.escape(punct), text):
                position = match.start()
                prosody_info['intonations'].append((position, intonation))
        
        # Extract emphasis spans
        for pattern, emphasis_type in self.emphasis_patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                prosody_info['emphasis'].append((start, end, emphasis_type))
        
        # Detect code-switching boundaries (Ethiopic ↔ Latin)
        prosody_info['code_switches'] = self._detect_code_switches(text)
        
        return prosody_info
    
    def _detect_code_switches(self, text: str) -> List[Tuple[int, str]]:
        """
        Detect language switching boundaries
        
        Args:
            text: Input text
            
        Returns:
            List of (position, language) tuples
        """
        switches = []
        current_script = None
        
        for i, char in enumerate(text):
            # Detect script type
            if 0x1200 <= ord(char) <= 0x137F:  # Ethiopic
                script = 'ethiopic'
            elif char.isalpha() and ord(char) < 0x1200:  # Latin
                script = 'latin'
            elif 0x0600 <= ord(char) <= 0x06FF:  # Arabic
                script = 'arabic'
            else:
                continue
            
            # Record switch
            if script != current_script and current_script is not None:
                switches.append((i, script))
            
            current_script = script
        
        return switches
    
    def process_for_g2p(self, text: str, keep_markers: bool = True) -> str:
        """
        Process text for G2P conversion while preserving prosody info
        
        Args:
            text: Input text
            keep_markers: Keep prosody markers in output
            
        Returns:
            Processed text suitable for G2P
        """
        if not text:
            return ""
        
        # Step 1: Add prosody markers if requested
        if keep_markers:
            text = self.add_prosody_markers(text)
        
        # Step 2: Normalize whitespace around punctuation
        # Add space after punctuation for proper pause handling
        text = re.sub(r'([።፣፤፥፧፨.!?,;:])([^\s<])', r'\1 \2', text)
        
        # Step 3: Clean multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def clean_markers_for_tts(self, text: str) -> str:
        """
        Remove prosody markers for final TTS input (if model doesn't support them)
        
        Args:
            text: Text with prosody markers
            
        Returns:
            Clean text
        """
        # Remove all prosody markers
        text = re.sub(r'<pause:\d+\.?\d*>', '', text)
        text = re.sub(r'<intonation:\w+>', '', text)
        text = re.sub(r'<emphasis:\w+>', '', text)
        text = re.sub(r'</emphasis>', '', text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def enhance_for_emotion(
        self,
        text: str,
        emotion: str = 'neutral',
        intensity: float = 1.0
    ) -> str:
        """
        Enhance text with emotion markers
        
        Args:
            text: Input text
            emotion: Emotion type (happy, sad, angry, excited, neutral)
            intensity: Emotion intensity (0.0 - 2.0)
            
        Returns:
            Text with emotion markers
        """
        if emotion == 'neutral' or intensity < 0.1:
            return text
        
        # Add emotion marker at start
        emotion_marker = f'<emotion:{emotion}:{intensity:.1f}>'
        return f'{emotion_marker}{text}</emotion>'
    
    def get_prosody_statistics(self, text: str) -> Dict[str, int]:
        """
        Get prosody statistics for text analysis
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with counts
        """
        info = self.extract_prosody_info(text)
        
        return {
            'total_pauses': len(info['pauses']),
            'short_pauses': sum(1 for _, _, d in info['pauses'] if d <= 0.2),
            'medium_pauses': sum(1 for _, _, d in info['pauses'] if 0.2 < d <= 0.4),
            'long_pauses': sum(1 for _, _, d in info['pauses'] if d > 0.4),
            'questions': sum(1 for _, intonation in info['intonations'] if intonation == 'rising'),
            'exclamations': sum(1 for _, intonation in info['intonations'] if intonation == 'emphatic'),
            'emphasis_spans': len(info['emphasis']),
            'code_switches': len(info['code_switches']),
        }


# Convenience function
def add_prosody_markers(text: str) -> str:
    """
    Quick function to add prosody markers to text
    
    Args:
        text: Input text
        
    Returns:
        Text with prosody markers
    """
    handler = ProsodyHandler()
    return handler.add_prosody_markers(text)


# Example usage and tests
if __name__ == "__main__":
    print("=" * 80)
    print("PROSODY HANDLER - DEMONSTRATION")
    print("=" * 80)
    print()
    
    handler = ProsodyHandler()
    
    # Test texts
    test_texts = [
        "ሰላም! እንዴት ነህ?",
        "ከዕለታት አንድ ቀን አንዲት መነኩሲት ጎኅ ሲቀድ ወደ ቤተ ክርስቲያን ሊሄዱ ይነሳሉ።",
        "When they reached a tree, they prayed!",
        "ዋጋው ፻ ብር ነው። ይህ ውድ ነው?",
    ]
    
    print("📝 Prosody Marker Examples:")
    print("-" * 80)
    
    for text in test_texts:
        print(f"\nOriginal:  {text}")
        
        # Add markers
        with_markers = handler.add_prosody_markers(text)
        print(f"Markers:   {with_markers[:100]}...")
        
        # Extract info
        info = handler.extract_prosody_info(text)
        print(f"Pauses:    {len(info['pauses'])} ({', '.join([f'{p}:{d}' for _, p, d in info['pauses'][:3]])}...)")
        print(f"Intonation: {', '.join([t for _, t in info['intonations']])}")
        print(f"Switches:  {len(info['code_switches'])} code-switching boundaries")
        
        # Statistics
        stats = handler.get_prosody_statistics(text)
        print(f"Stats:     {stats}")
    
    print()
    print("=" * 80)
    print("✅ Demonstration complete!")
    print("=" * 80)
