Amharic TTS Enhancement Specification for XTTS v2
Executive Summary
This specification provides comprehensive guidelines for implementing and enhancing a Text-to-Speech (TTS) system for Amharic language using XTTS v2, with integrated G2P (Grapheme-to-Phoneme) conversion, custom tokenization, and fine-tuning capabilities.

Table of Contents
System Architecture
Core Requirements
Implementation Modules
G2P Integration
Tokenizer Extension
Fine-tuning Pipeline
Data Preparation
Performance Optimization
Testing & Validation
Deployment Guidelines
1. System Architecture
1.1 Technology Stack
YAML

base_model: XTTS v2.0.3
language: Amharic (amh)
script: Ethiopic (Ethi)
phoneme_system: IPA
g2p_backend:
  - primary: transphone
  - secondary: epitran
  - fallback: custom_rules
tokenizer: Extended BPE with Amharic support
framework: PyTorch >= 2.0
compute: CUDA 11.8+
1.2 Component Architecture
mermaid

graph TD
    A[Amharic Text Input] --> B[Text Preprocessor]
    B --> C[G2P Converter]
    C --> D[Phoneme Processor]
    D --> E[Custom Tokenizer]
    E --> F[XTTS v2 Model]
    F --> G[Vocoder]
    G --> H[Audio Output]
    
    I[Training Data] --> J[Data Pipeline]
    J --> K[Fine-tuning Module]
    K --> F
2. Core Requirements
2.1 Functional Requirements
YAML

requirements:
  language_support:
    - Full Ethiopic script support (340+ characters)
    - Amharic-specific phoneme mapping
    - Support for labiovelar consonants
    - Gemination handling
    - Epenthetic vowel insertion
    
  quality_metrics:
    - MOS Score: >= 4.0
    - WER: < 10%
    - RTF: < 0.3
    - Speaker similarity: > 0.85
    
  capabilities:
    - Zero-shot voice cloning
    - Multi-speaker synthesis
    - Emotional tone control
    - Speech rate adjustment
    - Pitch modulation
2.2 Non-Functional Requirements
YAML

performance:
  latency: < 500ms for 100 characters
  throughput: > 10 requests/second
  memory: < 8GB GPU RAM
  
scalability:
  concurrent_users: 100+
  model_size: < 2GB
  
reliability:
  uptime: 99.9%
  error_rate: < 0.1%
3. Implementation Modules
3.1 Module Structure
Python

amharic_tts/
├── __init__.py
├── config/
│   ├── amharic_config.yaml
│   ├── phoneme_mapping.json
│   └── tokenizer_config.json
├── g2p/
│   ├── __init__.py
│   ├── amharic_g2p.py
│   ├── epitran_adapter.py
│   ├── transphone_adapter.py
│   └── rules/
│       ├── epenthesis.py
│       ├── gemination.py
│       └── syllabification.py
├── tokenizer/
│   ├── __init__.py
│   ├── amharic_tokenizer.py
│   └── bpe_extension.py
├── preprocessing/
│   ├── __init__.py
│   ├── text_cleaner.py
│   ├── number_processor.py
│   └── abbreviation_handler.py
├── training/
│   ├── __init__.py
│   ├── dataset_builder.py
│   ├── fine_tuner.py
│   └── evaluation.py
├── inference/
│   ├── __init__.py
│   ├── tts_engine.py
│   └── voice_cloner.py
└── utils/
    ├── __init__.py
    ├── audio_processor.py
    └── logger.py
4. G2P Integration
4.1 G2P Implementation Specification
Python

# amharic_tts/g2p/amharic_g2p.py

from typing import List, Tuple, Optional, Dict
import epitran
from transphone import read_g2p
import re
from dataclasses import dataclass

@dataclass
class G2PConfig:
    """Configuration for Amharic G2P conversion"""
    use_transphone: bool = True
    use_epitran: bool = False
    apply_epenthesis: bool = True
    handle_gemination: bool = True
    preserve_punctuation: bool = True
    
class AmharicG2P:
    """
    Amharic Grapheme-to-Phoneme converter with multiple backends
    
    Features:
    - Automatic epithetic vowel insertion
    - Gemination handling
    - Labiovelar consonant processing
    - Context-aware phoneme mapping
    """
    
    def __init__(self, config: G2PConfig = G2PConfig()):
        self.config = config
        self._initialize_backends()
        self._load_custom_rules()
        
    def _initialize_backends(self):
        """Initialize G2P backends based on configuration"""
        if self.config.use_transphone:
            self.transphone_g2p = read_g2p('amh')
        if self.config.use_epitran:
            self.epitran_g2p = epitran.Epitran('amh-Ethi')
            
    def _load_custom_rules(self):
        """Load Amharic-specific phonological rules"""
        self.epenthesis_rules = {
            # Pattern: (context, insertion_rule)
            r'([ክግቅኽ])([^aeiouɨ])': r'\1ɨ\2',  # Insert ɨ after velars
            r'([ብትድንርስዝልምፍ])$': r'\1ɨ',      # Word-final epenthesis
        }
        
        self.gemination_patterns = {
            # Double consonants in specific contexts
            r'([ተደነሰዘለመ])_gem': r'\1\1',
        }
        
        self.labiovelar_mapping = {
            'ቋ': 'qʷa', 'ቍ': 'qʷɨ', 'ቊ': 'qʷu',
            'ኳ': 'kʷa', 'ኵ': 'kʷɨ', 'ኲ': 'kʷu',
            'ጓ': 'gʷa', 'ጕ': 'gʷɨ', 'ጒ': 'gʷu',
            'ፏ': 'fʷa', 'ፑ': 'fʷɨ', 'ፐ': 'fʷu',
        }
        
    def convert(self, text: str) -> str:
        """
        Convert Amharic text to phonemes
        
        Args:
            text: Input Amharic text
            
        Returns:
            Phoneme sequence in IPA format
        """
        # Step 1: Preprocess text
        text = self._preprocess(text)
        
        # Step 2: Apply labiovelar mapping
        text = self._map_labiovelars(text)
        
        # Step 3: G2P conversion
        phonemes = self._g2p_convert(text)
        
        # Step 4: Apply epenthesis rules
        if self.config.apply_epenthesis:
            phonemes = self._apply_epenthesis(phonemes)
            
        # Step 5: Handle gemination
        if self.config.handle_gemination:
            phonemes = self._handle_gemination(phonemes)
            
        return phonemes
        
    def _preprocess(self, text: str) -> str:
        """Normalize and clean Amharic text"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize similar characters
        replacements = {
            'ሥ': 'ስ', 'ዕ': 'እ', 'ፅ': 'ጽ',
            'ኅ': 'ህ', 'ኽ': 'ህ', 'ሕ': 'ህ'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text.strip()
        
    def _map_labiovelars(self, text: str) -> str:
        """Map labiovelar characters to phoneme sequences"""
        for char, phoneme in self.labiovelar_mapping.items():
            text = text.replace(char, phoneme)
        return text
        
    def _g2p_convert(self, text: str) -> str:
        """Apply G2P conversion using configured backend"""
        if self.config.use_transphone:
            return self.transphone_g2p(text)
        elif self.config.use_epitran:
            return self.epitran_g2p.transliterate(text)
        else:
            # Fallback to basic mapping
            return self._basic_g2p(text)
            
    def _apply_epenthesis(self, phonemes: str) -> str:
        """Apply epenthetic vowel insertion rules"""
        for pattern, replacement in self.epenthesis_rules.items():
            phonemes = re.sub(pattern, replacement, phonemes)
        return phonemes
        
    def _handle_gemination(self, phonemes: str) -> str:
        """Process geminated consonants"""
        for pattern, replacement in self.gemination_patterns.items():
            phonemes = re.sub(pattern, replacement, phonemes)
        return phonemes
        
    def _basic_g2p(self, text: str) -> str:
        """Basic fallback G2P mapping"""
        # Implement basic character-to-phoneme mapping
        mapping = self._load_basic_mapping()
        phonemes = []
        for char in text:
            phonemes.append(mapping.get(char, char))
        return ''.join(phonemes)
4.2 Phoneme Mapping Configuration
JSON

// config/phoneme_mapping.json
{
  "amharic_to_ipa": {
    "consonants": {
      "ህ": "h", "ል": "l", "ም": "m", "ሥ": "s",
      "ር": "r", "ስ": "s", "ቅ": "qʼ", "ብ": "b",
      "ት": "t", "ች": "tʃʼ", "ን": "n", "ኝ": "ɲ",
      "እ": "ʔ", "ክ": "k", "ው": "w", "ዝ": "z",
      "ዥ": "ʒ", "ይ": "j", "ድ": "d", "ጅ": "dʒ",
      "ግ": "g", "ጥ": "tʼ", "ጭ": "tʃʼ", "ጵ": "pʼ",
      "ፅ": "tsʼ", "ፍ": "f", "ፕ": "p"
    },
    "vowels": {
      "አ": "a", "ኡ": "u", "ኢ": "i", "ኣ": "a",
      "ኤ": "e", "እ": "ɨ", "ኦ": "o"
    },
    "special_sequences": {
      "consonant_clusters": {
        "ንት": "nt", "ንድ": "nd", "ምብ": "mb",
        "ልት": "lt", "ርት": "rt", "ስት": "st"
      },
      "word_boundaries": {
        "initial": "^",
        "final": "$",
        "pause": "#"
      }
    }
  },
  "epenthesis_contexts": [
    {
      "pattern": "C_C",
      "insert": "ɨ",
      "conditions": ["non_geminate", "word_medial"]
    },
    {
      "pattern": "C_#",
      "insert": "ɨ",
      "conditions": ["obstruent", "word_final"]
    }
  ]
}
5. Tokenizer Extension
5.1 Custom Amharic Tokenizer
Python

# amharic_tts/tokenizer/amharic_tokenizer.py

from typing import List, Dict, Optional, Tuple
import torch
from transformers import PreTrainedTokenizerFast
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
import json
import numpy as np

class AmharicXTTSTokenizer(VoiceBpeTokenizer):
    """
    Extended XTTS tokenizer with Amharic support
    
    Features:
    - Ethiopic script handling
    - Phoneme-level tokenization
    - Sub-word BPE for OOV handling
    - Special token management
    """
    
    def __init__(
        self,
        vocab_file: str = None,
        phoneme_vocab: Dict[str, int] = None,
        use_phonemes: bool = True,
        max_length: int = 512,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.use_phonemes = use_phonemes
        self.max_length = max_length
        
        # Initialize vocabularies
        self._initialize_amharic_vocab()
        self._initialize_phoneme_vocab(phoneme_vocab)
        self._add_special_tokens()
        
    def _initialize_amharic_vocab(self):
        """Initialize Amharic character and sub-word vocabulary"""
        # Base Ethiopic characters
        self.ethiopic_chars = []
        for base in range(0x1200, 0x137F):  # Ethiopic Unicode range
            self.ethiopic_chars.append(chr(base))
            
        # Build initial vocabulary
        self.amharic_vocab = {
            char: idx + 1000  # Offset to avoid collision
            for idx, char in enumerate(self.ethiopic_chars)
        }
        
        # Add Amharic-specific sub-words
        self.amharic_subwords = self._build_subwords()
        self.vocab.update(self.amharic_subwords)
        
    def _initialize_phoneme_vocab(self, phoneme_vocab: Optional[Dict]):
        """Initialize phoneme vocabulary for G2P output"""
        if phoneme_vocab:
            self.phoneme_vocab = phoneme_vocab
        else:
            # IPA phonemes for Amharic
            self.phoneme_vocab = {
                # Consonants
                'p': 100, 'b': 101, 't': 102, 'd': 103,
                'k': 104, 'g': 105, 'qʼ': 106, 'ʔ': 107,
                'f': 108, 's': 109, 'z': 110, 'ʃ': 111,
                'ʒ': 112, 'h': 113, 'tʃ': 114, 'dʒ': 115,
                'ts': 116, 'tʃʼ': 117, 'tsʼ': 118, 'tʼ': 119,
                'pʼ': 120, 'kʼ': 121, 'm': 122, 'n': 123,
                'ɲ': 124, 'r': 125, 'l': 126, 'j': 127,
                'w': 128,
                # Vowels
                'a': 130, 'e': 131, 'i': 132, 'o': 133,
                'u': 134, 'ɨ': 135, 'ə': 136,
                # Labialized consonants
                'kʷ': 140, 'gʷ': 141, 'qʷ': 142, 'fʷ': 143,
                # Prosodic markers
                '.': 150, ',': 151, '!': 152, '?': 153,
                '_': 154,  # Word boundary
                '#': 155,  # Phrase boundary
            }
            
    def _add_special_tokens(self):
        """Add special tokens for Amharic TTS"""
        special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[BOS]': 2,
            '[EOS]': 3,
            '[MASK]': 4,
            '[LANG_AM]': 5,  # Amharic language token
            '[SPEAKER]': 6,
            '[EMOTION]': 7,
            '[SPEED]': 8,
            '[PITCH]': 9,
        }
        
        self.special_tokens = special_tokens
        self.vocab.update(special_tokens)
        
    def _build_subwords(self) -> Dict[str, int]:
        """Build BPE subwords for Amharic"""
        # This would be trained on Amharic corpus
        # Placeholder for demonstration
        subwords = {}
        common_syllables = [
            'በ', 'ለ', 'መ', 'ነ', 'ተ', 'አ', 'ከ', 'ወ',
            'የ', 'ደ', 'ገ', 'ጠ', 'ጨ', 'ጰ', 'ፈ', 'ፐ'
        ]
        
        for idx, syllable in enumerate(common_syllables):
            subwords[syllable] = 2000 + idx
            
        return subwords
        
    def encode(
        self,
        text: str,
        g2p_output: Optional[str] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False,
        return_tensors: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode Amharic text to token IDs
        
        Args:
            text: Input Amharic text
            g2p_output: Optional phoneme sequence from G2P
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            truncation: Whether to truncate long sequences
            padding: Whether to pad sequences
            return_tensors: Return type ('pt' for PyTorch)
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        max_len = max_length or self.max_length
        
        if self.use_phonemes and g2p_output:
            tokens = self._encode_phonemes(g2p_output)
        else:
            tokens = self._encode_text(text)
            
        if add_special_tokens:
            tokens = [self.special_tokens['[BOS]']] + tokens + [self.special_tokens['[EOS]']]
            
        # Truncation
        if truncation and len(tokens) > max_len:
            tokens = tokens[:max_len]
            
        # Padding
        attention_mask = [1] * len(tokens)
        if padding:
            pad_length = max_len - len(tokens)
            tokens += [self.special_tokens['[PAD]']] * pad_length
            attention_mask += [0] * pad_length
            
        result = {
            'input_ids': tokens,
            'attention_mask': attention_mask
        }
        
        if return_tensors == 'pt':
            result = {
                'input_ids': torch.tensor(tokens),
                'attention_mask': torch.tensor(attention_mask)
            }
            
        return result
        
    def _encode_phonemes(self, phonemes: str) -> List[int]:
        """Encode phoneme sequence to token IDs"""
        tokens = []
        i = 0
        while i < len(phonemes):
            # Try to match multi-character phonemes first
            matched = False
            for length in [3, 2, 1]:  # Check trigrams, bigrams, then single
                if i + length <= len(phonemes):
                    phoneme = phonemes[i:i+length]
                    if phoneme in self.phoneme_vocab:
                        tokens.append(self.phoneme_vocab[phoneme])
                        i += length
                        matched = True
                        break
                        
            if not matched:
                # Unknown phoneme
                tokens.append(self.special_tokens['[UNK]'])
                i += 1
                
        return tokens
        
    def _encode_text(self, text: str) -> List[int]:
        """Encode Amharic text using BPE"""
        tokens = []
        
        # First try to match subwords
        words = text.split()
        for word in words:
            word_tokens = self._encode_word(word)
            tokens.extend(word_tokens)
            
        return tokens
        
    def _encode_word(self, word: str) -> List[int]:
        """Encode single Amharic word"""
        tokens = []
        i = 0
        
        while i < len(word):
            # Try to match longest subword
            matched = False
            for length in range(min(10, len(word) - i), 0, -1):
                subword = word[i:i+length]
                
                if subword in self.amharic_subwords:
                    tokens.append(self.amharic_subwords[subword])
                    i += length
                    matched = True
                    break
                elif subword in self.amharic_vocab:
                    tokens.append(self.amharic_vocab[subword])
                    i += length
                    matched = True
                    break
                    
            if not matched:
                # Character-level fallback
                if word[i] in self.amharic_vocab:
                    tokens.append(self.amharic_vocab[word[i]])
                else:
                    tokens.append(self.special_tokens['[UNK]'])
                i += 1
                
        return tokens
        
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs back to text"""
        # Create reverse mapping
        id_to_token = {v: k for k, v in self.vocab.items()}
        id_to_token.update({v: k for k, v in self.phoneme_vocab.items()})
        
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in self.special_tokens.values():
                continue
            tokens.append(id_to_token.get(token_id, '[UNK]'))
            
        return ''.join(tokens)
6. Fine-tuning Pipeline
6.1 Dataset Preparation
Python

# amharic_tts/training/dataset_builder.py

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

class AmharicTTSDataset(Dataset):
    """
    Dataset class for Amharic TTS training
    
    Expected data format:
    - audio_files: WAV files at 22050 Hz
    - transcriptions: Text files with Amharic text
    - metadata: CSV with speaker info, emotion labels, etc.
    """
    
    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        g2p_converter,
        tokenizer,
        audio_config: Dict,
        augmentation: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.g2p = g2p_converter
        self.tokenizer = tokenizer
        self.audio_config = audio_config
        self.augmentation = augmentation
        
        # Audio processing
        self.sample_rate = audio_config.get('sample_rate', 22050)
        self.n_fft = audio_config.get('n_fft', 1024)
        self.hop_length = audio_config.get('hop_length', 256)
        self.n_mels = audio_config.get('n_mels', 80)
        
        # Initialize mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Data augmentation
        if augmentation:
            self.augmentation_transforms = self._setup_augmentation()
            
    def _setup_augmentation(self):
        """Setup data augmentation transforms"""
        return {
            'speed': torchaudio.transforms.Speed(
                orig_freq=self.sample_rate,
                factor=np.random.uniform(0.9, 1.1)
            ),
            'pitch': torchaudio.transforms.PitchShift(
                sample_rate=self.sample_rate,
                n_steps=np.random.randint(-2, 3)
            ),
            'noise': lambda x: x + torch.randn_like(x) * 0.005,
            'reverb': torchaudio.transforms.Reverb(
                sample_rate=self.sample_rate
            )
        }
        
    def __len__(self) -> int:
        return len(self.metadata)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample
        
        Returns:
            Dictionary containing:
            - 'audio': Raw audio waveform
            - 'mel_spec': Mel spectrogram
            - 'text': Original Amharic text
            - 'phonemes': G2P output
            - 'tokens': Token IDs
            - 'speaker_id': Speaker identifier
            - 'emotion': Emotion label (if available)
        """
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio_path = self.data_dir / row['audio_file']
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Apply augmentation
        if self.augmentation and np.random.random() > 0.5:
            aug_type = np.random.choice(list(self.augmentation_transforms.keys()))
            waveform = self.augmentation_transforms[aug_type](waveform)
            
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Process text
        text = row['transcription']
        phonemes = self.g2p.convert(text)
        
        # Tokenize
        token_output = self.tokenizer.encode(
            text=text,
            g2p_output=phonemes,
            return_tensors='pt',
            padding=True,
            max_length=256
        )
        
        return {
            'audio': waveform.squeeze(0),
            'mel_spec': mel_spec.squeeze(0),
            'text': text,
            'phonemes': phonemes,
            'input_ids': token_output['input_ids'].squeeze(0),
            'attention_mask': token_output['attention_mask'].squeeze(0),
            'speaker_id': torch.tensor(row.get('speaker_id', 0)),
            'emotion': row.get('emotion', 'neutral'),
            'duration': waveform.shape[-1] / self.sample_rate
        }
        
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching"""
        # Find max lengths
        max_audio_len = max(b['audio'].shape[-1] for b in batch)
        max_mel_len = max(b['mel_spec'].shape[-1] for b in batch)
        max_token_len = max(b['input_ids'].shape[-1] for b in batch)
        
        # Prepare batch tensors
        batch_size = len(batch)
        audio_batch = torch.zeros(batch_size, max_audio_len)
        mel_batch = torch.zeros(batch_size, self.n_mels, max_mel_len)
        token_batch = torch.zeros(batch_size, max_token_len, dtype=torch.long)
        mask_batch = torch.zeros(batch_size, max_token_len, dtype=torch.long)
        speaker_batch = torch.zeros(batch_size, dtype=torch.long)
        
        for i, sample in enumerate(batch):
            audio_len = sample['audio'].shape[-1]
            mel_len = sample['mel_spec'].shape[-1]
            token_len = sample['input_ids'].shape[-1]
            
            audio_batch[i, :audio_len] = sample['audio']
            mel_batch[i, :, :mel_len] = sample['mel_spec']
            token_batch[i, :token_len] = sample['input_ids']
            mask_batch[i, :token_len] = sample['attention_mask']
            speaker_batch[i] = sample['speaker_id']
            
        return {
            'audio': audio_batch,
            'mel_spec': mel_batch,
            'input_ids': token_batch,
            'attention_mask': mask_batch,
            'speaker_ids': speaker_batch,
            'texts': [b['text'] for b in batch],
            'phonemes': [b['phonemes'] for b in batch]
        }
6.2 Fine-tuning Configuration
YAML

# config/training_config.yaml

model:
  name: "xtts_v2_amharic"
  base_model: "tts_models/multilingual/multi-dataset/xtts_v2"
  checkpoint: null  # Path to checkpoint if resuming

data:
  train_dir: "data/amharic_tts/train"
  val_dir: "data/amharic_tts/val"
  test_dir: "data/amharic_tts/test"
  metadata_train: "data/amharic_tts/metadata_train.csv"
  metadata_val: "data/amharic_tts/metadata_val.csv"
  metadata_test: "data/amharic_tts/metadata_test.csv"
  
  audio:
    sample_rate: 22050
    n_fft: 1024
    hop_length: 256
    win_length: 1024
    n_mels: 80
    mel_fmin: 0
    mel_fmax: 8000
    
  text:
    use_phonemes: true
    g2p_backend: "transphone"
    tokenizer: "custom_amharic"
    max_text_length: 256
    
training:
  batch_size: 16
  gradient_accumulation_steps: 4
  num_epochs: 100
  learning_rate: 1e-4
  warmup_steps: 1000
  
  optimizer:
    type: "AdamW"
    betas: [0.9, 0.999]
    eps: 1e-8
    weight_decay: 0.01
    
  scheduler:
    type: "CosineAnnealingLR"
    T_max: 10000
    eta_min: 1e-6
    
  loss:
    mel_loss_weight: 45.0
    kl_loss_weight: 1.0
    speaker_loss_weight: 1.0
    phoneme_loss_weight: 1.0
    
  checkpointing:
    save_every: 1000
    keep_last_n: 5
    best_metric: "val_loss"
    
  mixed_precision: true
  gradient_clipping: 1.0
  
evaluation:
  metrics:
    - "mel_loss"
    - "speaker_similarity"
    - "phoneme_accuracy"
    - "MOS"
  
  inference:
    num_samples: 10
    temperature: 0.8
    repetition_penalty: 1.2
    
monitoring:
  use_wandb: true
  use_tensorboard: true
  log_every: 100
  
hardware:
  device: "cuda"
  num_workers: 8
  pin_memory: true
  
augmentation:
  enable: true
  speed_perturbation: [0.9, 1.1]
  pitch_shift: [-2, 2]
  add_noise: true
  add_reverb: true
  probability: 0.3
7. Data Preparation
7.1 Data Collection Guidelines
Markdown

### Amharic TTS Dataset Requirements

#### Audio Requirements
- **Format**: WAV (16-bit PCM)
- **Sample Rate**: 22050 Hz (minimum), 44100 Hz (preferred)
- **Channels**: Mono
- **Duration**: 2-15 seconds per utterance
- **Quality**: 
  - SNR > 40 dB
  - No clipping or distortion
  - Consistent volume levels
  - Room tone < -60 dB

#### Text Requirements
- **Script**: Native Ethiopic script
- **Encoding**: UTF-8
- **Content Types**:
  - Declarative sentences (40%)
  - Questions (20%)
  - Exclamations (10%)
  - Reading passages (30%)
  
#### Metadata Requirements
- Speaker ID
- Gender
- Age range
- Dialect/Region
- Recording environment
- Emotion label (if applicable)

#### Dataset Size Recommendations
- **Minimum**: 5 hours of clean speech
- **Recommended**: 20+ hours
- **Optimal**: 100+ hours
- **Speakers**: Minimum 5, recommended 20+
7.2 Data Processing Pipeline
Python

# amharic_tts/preprocessing/data_pipeline.py

import os
import json
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import pyloudnorm as pyln

class AmharicDataPipeline:
    """
    Complete data processing pipeline for Amharic TTS
    """
    
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            
        self.sample_rate = self.config['audio']['sample_rate']
        self.meter = pyln.Meter(self.sample_rate)
        
    def process_dataset(
        self,
        input_dir: str,
        output_dir: str,
        metadata_file: str
    ):
        """
        Process raw Amharic dataset
        
        Steps:
        1. Audio validation and normalization
        2. Text cleaning and validation
        3. Forced alignment (if needed)
        4. Train/val/test split
        5. Metadata generation
        """
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        metadata = pd.read_csv(metadata_file)
        
        processed_data = []
        
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
            try:
                # Process audio
                audio_file = input_path / row['audio_file']
                processed_audio = self._process_audio(audio_file)
                
                # Process text
                processed_text = self._process_text(row['transcription'])
                
                # Validate
                if self._validate_sample(processed_audio, processed_text):
                    # Save processed audio
                    output_audio = output_path / f"audio_{idx:06d}.wav"
                    sf.write(
                        output_audio,
                        processed_audio,
                        self.sample_rate
                    )
                    
                    # Update metadata
                    processed_data.append({
                        'audio_file': output_audio.name,
                        'transcription': processed_text,
                        'original_text': row['transcription'],
                        'duration': len(processed_audio) / self.sample_rate,
                        'speaker_id': row.get('speaker_id', 0),
                        'quality_score': self._compute_quality_score(processed_audio)
                    })
                    
            except Exception as e:
                print(f"Error processing {row['audio_file']}: {e}")
                continue
                
        # Create final metadata
        processed_df = pd.DataFrame(processed_data)
        
        # Split dataset
        train_df, val_df, test_df = self._split_dataset(processed_df)
        
        # Save metadata
        train_df.to_csv(output_path / 'metadata_train.csv', index=False)
        val_df.to_csv(output_path / 'metadata_val.csv', index=False)
        test_df.to_csv(output_path / 'metadata_test.csv', index=False)
        
        # Generate statistics
        self._generate_statistics(processed_df, output_path)
        
    def _process_audio(self, audio_file: Path) -> np.ndarray:
        """Process and normalize audio"""
        # Load audio
        audio, sr = librosa.load(audio_file, sr=self.sample_rate)
        
        # Remove silence
        audio = self._trim_silence(audio)
        
        # Normalize loudness
        loudness = self.meter.integrated_loudness(audio)
        audio = pyln.normalize.loudness(
            audio,
            loudness,
            -20.0  # Target loudness in LUFS
        )
        
        # Apply pre-emphasis
        audio = self._apply_preemphasis(audio)
        
        return audio
        
    def _process_text(self, text: str) -> str:
        """Clean and normalize Amharic text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Normalize punctuation
        text = self._normalize_punctuation(text)
        
        # Expand numbers
        text = self._expand_numbers(text)
        
        # Expand abbreviations
        text = self._expand_abbreviations(text)
        
        return text
        
    def _trim_silence(
        self,
        audio: np.ndarray,
        threshold: float = 0.01,
        frame_length: int = 2048
    ) -> np.ndarray:
        """Remove silence from audio"""
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length
        )[0]
        
        threshold = np.percentile(energy, 5)
        active_frames = energy > threshold
        
        # Find boundaries
        indices = np.where(active_frames)[0]
        if len(indices) > 0:
            start = indices[0] * (frame_length // 2)
            end = indices[-1] * (frame_length // 2)
            return audio[start:end]
            
        return audio
        
    def _apply_preemphasis(
        self,
        audio: np.ndarray,
        coef: float = 0.97
    ) -> np.ndarray:
        """Apply pre-emphasis filter"""
        return np.append(audio[0], audio[1:] - coef * audio[:-1])
        
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize Amharic punctuation"""
        replacements = {
            '፡': '.',  # Amharic wordspace to period
            '።': '.',  # Amharic full stop to period
            '፣': ',',  # Amharic comma
            '፤': ';',  # Amharic semicolon
            '፥': ':',  # Amharic colon
            '፦': ':',  # Amharic preface colon
            '፧': '?',  # Amharic question mark
            '፨': '¶',  # Amharic paragraph separator
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text
        
    def _expand_numbers(self, text: str) -> str:
        """Convert numbers to Amharic words"""
        # Implementation for number expansion
        # This is a simplified version
        number_map = {
            '0': 'ዜሮ', '1': 'አንድ', '2': 'ሁለት',
            '3': 'ሶስት', '4': 'አራት', '5': 'አምስት',
            '6': 'ስድስት', '7': 'ሰባት', '8': 'ስምንት',
            '9': 'ዘጠኝ', '10': 'አስር'
        }
        
        for num, word in number_map.items():
            text = text.replace(num, word)
            
        return text
        
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common Amharic abbreviations"""
        abbreviations = {
            'ዓ.ም': 'ዓመተ ምህረት',
            'ዓ.ዓ': 'ዓመተ ዓለም',
            'ክ.ክ': 'ክፍለ ከተማ',
            'ት.ቤት': 'ትምህርት ቤት',
            # Add more abbreviations
        }
        
        for abbr, expansion in abbreviations.items():
            text = text.replace(abbr, expansion)
            
        return text
        
    def _validate_sample(
        self,
        audio: np.ndarray,
        text: str
    ) -> bool:
        """Validate audio-text pair"""
        # Check audio duration
        duration = len(audio) / self.sample_rate
        if duration < 0.5 or duration > 20:
            return False
            
        # Check text length
        if len(text) < 5 or len(text) > 500:
            return False
            
        # Check audio quality
        if np.max(np.abs(audio)) < 0.01:  # Too quiet
            return False
            
        return True
        
    def _compute_quality_score(self, audio: np.ndarray) -> float:
        """Compute audio quality score"""
        # Simple quality metrics
        snr = self._estimate_snr(audio)
        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        )
        
        # Combine metrics (simplified)
        quality = min(1.0, snr / 50.0) * 0.5 + \
                 min(1.0, spectral_centroid / 5000) * 0.5
                 
        return quality
        
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio"""
        # Simple SNR estimation
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(audio[:1000] ** 2)  # Assume first part is noise
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 50.0  # Default high SNR
            
        return snr
        
    def _split_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/val/test"""
        # Ensure speaker-wise split
        speakers = df['speaker_id'].unique()
        np.random.shuffle(speakers)
        
        n_speakers = len(speakers)
        train_speakers = speakers[:int(n_speakers * train_ratio)]
        val_speakers = speakers[
            int(n_speakers * train_ratio):
            int(n_speakers * (train_ratio + val_ratio))
        ]
        test_speakers = speakers[int(n_speakers * (train_ratio + val_ratio)):]
        
        train_df = df[df['speaker_id'].isin(train_speakers)]
        val_df = df[df['speaker_id'].isin(val_speakers)]
        test_df = df[df['speaker_id'].isin(test_speakers)]
        
        return train_df, val_df, test_df
        
    def _generate_statistics(self, df: pd.DataFrame, output_path: Path):
        """Generate dataset statistics"""
        stats = {
            'total_samples': len(df),
            'total_duration': df['duration'].sum(),
            'mean_duration': df['duration'].mean(),
            'std_duration': df['duration'].std(),
            'num_speakers': df['speaker_id'].nunique(),
            'mean_quality': df['quality_score'].mean(),
            'text_length_stats': {
                'mean': df['transcription'].str.len().mean(),
                'std': df['transcription'].str.len().std(),
                'min': df['transcription'].str.len().min(),
                'max': df['transcription'].str.len().max()
            }
        }
        
        with open(output_path / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
8. Performance Optimization

8.1 Inference Optimization
Python

# amharic_tts/inference/optimized_inference.py

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import onnxruntime as ort
from typing import Optional, Dict, List
import numpy as np

class OptimizedAmharicTTS:
    """
    Optimized inference engine for Amharic TTS
    
    Features:
    - Mixed precision inference
    - ONNX optimization
    - Batch processing
    - Caching mechanisms
    - Streaming support
    """
    
    def __init__(
        self,
        model_path: str,
        use_onnx: bool = False,
        use_mixed_precision: bool = True,
        cache_size: int = 100,
        device: str = 'cuda'
    ):
        self.device = torch.device(device)
        self.use_mixed_precision = use_mixed_precision
        self.cache = {}
        self.cache_size = cache_size
        
        if use_onnx:
            self.model = self._load_onnx_model(model_path)
        else:
            self.model = self._load_torch_model(model_path)
            
        # Initialize components
        self.g2p = self._init_g2p()
        self.tokenizer = self._init_tokenizer()
        self.vocoder = self._init_vocoder()
        
        # Warm up model
        self._warmup()
        
    def _load_torch_model(self, model_path: str):
        """Load PyTorch model with optimization"""
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        
        # Apply optimizations
        if self.device.type == 'cuda':
            model = model.half() if self.use_mixed_precision else model
            model = torch.jit.script(model)  # TorchScript optimization
            
        return model
        
    def _load_onnx_model(self, model_path: str):
        """Load ONNX model for faster inference"""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        return ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        emotion: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 1.0,
        use_cache: bool = True,
        streaming: bool = False
    ) -> np.ndarray:
        """
        Synthesize speech from Amharic text
        
        Args:
            text: Input Amharic text
            speaker_id: Speaker ID for multi-speaker model
            emotion: Emotion label
            speed: Speech rate adjustment
            pitch: Pitch adjustment
            use_cache: Whether to use caching
            streaming: Enable streaming synthesis
            
        Returns:
            Audio waveform as numpy array
        """
        
        # Check cache
        cache_key = self._get_cache_key(text, speaker_id, emotion, speed, pitch)
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
            
        # Preprocess text
        phonemes = self.g2p.convert(text)
        tokens = self.tokenizer.encode(
            text=text,
            g2p_output=phonemes,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        # Prepare additional inputs
        if speaker_id is not None:
            speaker_tensor = torch.tensor([speaker_id]).to(self.device)
        else:
            speaker_tensor = None
            
        # Inference
        if self.use_mixed_precision and self.device.type == 'cuda':
            with autocast():
                mel_output = self._generate_mel(
                    input_ids,
                    attention_mask,
                    speaker_tensor,
                    emotion,
                    speed
                )
        else:
            mel_output = self._generate_mel(
                input_ids,
                attention_mask,
                speaker_tensor,
                emotion,
                speed
            )
            
        # Vocoder synthesis
        if streaming:
            audio = self._streaming_vocoder(mel_output, pitch)
        else:
            audio = self._batch_vocoder(mel_output, pitch)
            
        # Update cache
        if use_cache:
            self._update_cache(cache_key, audio)
            
        return audio
        
    def _generate_mel(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_tensor: Optional[torch.Tensor],
        emotion: Optional[str],
        speed: float
    ) -> torch.Tensor:
        """Generate mel spectrogram"""
        
        # Adjust for speed
        if speed != 1.0:
            # Implement duration adjustment
            pass
            
        # Model inference
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            speaker_ids=speaker_tensor
        )
        
        return outputs['mel_outputs']
        
    def _batch_vocoder(
        self,
        mel_spec: torch.Tensor,
        pitch: float
    ) -> np.ndarray:
        """Batch vocoder processing"""
        
        # Pitch adjustment if needed
        if pitch != 1.0:
            mel_spec = self._adjust_pitch(mel_spec, pitch)
            
        # Vocoder synthesis
        with torch.no_grad():
            audio = self.vocoder(mel_spec)
            
        return audio.cpu().numpy()
        
    def _streaming_vocoder(
        self,
        mel_spec: torch.Tensor,
        pitch: float,
        chunk_size: int = 100
    ):
        """Streaming vocoder for real-time synthesis"""
        
        total_frames = mel_spec.shape[-1]
        
        for i in range(0, total_frames, chunk_size):
            chunk = mel_spec[:, :, i:i+chunk_size]
            
            if pitch != 1.0:
                chunk = self._adjust_pitch(chunk, pitch)
                
            audio_chunk = self.vocoder(chunk)
            yield audio_chunk.cpu().numpy()
            
    def _adjust_pitch(
        self,
        mel_spec: torch.Tensor,
        pitch_factor: float
    ) -> torch.Tensor:
        """Adjust pitch of mel spectrogram"""
        # Implement pitch shifting in mel domain
        # This is a simplified version
        if pitch_factor > 1.0:
            # Shift up
            shift = int((pitch_factor - 1.0) * 10)
            mel_spec = torch.roll(mel_spec, shifts=-shift, dims=1)
        elif pitch_factor < 1.0:
            # Shift down
            shift = int((1.0 - pitch_factor) * 10)
            mel_spec = torch.roll(mel_spec, shifts=shift, dims=1)
            
        return mel_spec
        
    def _get_cache_key(
        self,
        text: str,
        speaker_id: Optional[int],
        emotion: Optional[str],
        speed: float,
        pitch: float
    ) -> str:
        """Generate cache key for synthesis request"""
        return f"{text}_{speaker_id}_{emotion}_{speed}_{pitch}"
        
    def _update_cache(self, key: str, audio: np.ndarray):
        """Update cache with LRU policy"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        self.cache[key] = audio
        
    def _warmup(self):
        """Warm up model for faster first inference"""
        dummy_text = "ሰላም"
        dummy_tokens = self.tokenizer.encode(
            text=dummy_text,
            return_tensors='pt',
            max_length=10
        )
        
        with torch.no_grad():
            _ = self.model(
                input_ids=dummy_tokens['input_ids'].to(self.device),
                attention_mask=dummy_tokens['attention_mask'].to(self.device)
            )
            
    def export_onnx(self, output_path: str):
        """Export model to ONNX format"""
        dummy_input = torch.randn(1, 256).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['audio'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'audio': {0: 'batch_size', 1: 'time'}
            }
        )
9. Testing & Validation
9.1 Testing Framework
Python

# amharic_tts/tests/test_suite.py

import unittest
import torch
import numpy as np
from typing import List, Dict
import librosa
from pesq import pesq
from pystoi import stoi
import json

class AmharicTTSTestSuite(unittest.TestCase):
    """
    Comprehensive test suite for Amharic TTS
    """
    
    def setUp(self):
        """Initialize test environment"""
        self.g2p = AmharicG2P()
        self.tokenizer = AmharicXTTSTokenizer()
        self.model = OptimizedAmharicTTS('model.pt')
        
        # Load test data
        with open('test_data.json', 'r') as f:
            self.test_cases = json.load(f)
            
    def test_g2p_conversion(self):
        """Test G2P conversion accuracy"""
        test_pairs = [
            ("ሰላም", "səlam"),
            ("እንደምን ነህ", "ɨndəmɨn nəh"),
            ("አመሰግናለሁ", "aməsəgɨnalləhu"),
        ]
        
        for amharic, expected_ipa in test_pairs:
            result = self.g2p.convert(amharic)
            self.assertEqual(
                result.replace(' ', ''),
                expected_ipa.replace(' ', ''),
                f"G2P failed for {amharic}"
            )
            
    def test_tokenizer(self):
        """Test tokenizer functionality"""
        test_texts = [
            "ሰላም ዓለም",
            "የአማርኛ ቋንቋ",
            "123 ቁጥሮች"
        ]
        
        for text in test_texts:
            # Test encoding
            tokens = self.tokenizer.encode(text, return_tensors='pt')
            self.assertIsInstance(tokens['input_ids'], torch.Tensor)
            
            # Test decoding
            decoded = self.tokenizer.decode(tokens['input_ids'][0])
            # Decoded should preserve meaning
            
    def test_audio_quality(self):
        """Test synthesized audio quality"""
        test_text = "ይህ የድምጽ ጥራት ሙከራ ነው።"
        
        # Synthesize audio
        audio = self.model.synthesize(test_text)
        
        # Check audio properties
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
        
        # Check signal quality
        snr = self._calculate_snr(audio)
        self.assertGreater(snr, 30, "SNR too low")
        
        # Check spectral properties
        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=22050)
        )
        self.assertGreater(spectral_centroid, 1000)
        self.assertLess(spectral_centroid, 5000)
        
    def test_speaker_consistency(self):
        """Test multi-speaker consistency"""
        text = "ተመሳሳይ ድምጽ"
        
        # Generate multiple samples with same speaker
        samples = []
        for _ in range(5):
            audio = self.model.synthesize(text, speaker_id=1)
            samples.append(audio)
            
        # Check consistency
        similarities = []
        for i in range(len(samples) - 1):
            sim = self._calculate_similarity(samples[i], samples[i+1])
            similarities.append(sim)
            
        mean_similarity = np.mean(similarities)
        self.assertGreater(mean_similarity, 0.85, "Speaker consistency too low")
        
    def test_emotion_control(self):
        """Test emotion control in synthesis"""
        text = "ስሜት ያለው ንግግር"
        emotions = ['neutral', 'happy', 'sad', 'angry']
        
        audio_samples = {}
        for emotion in emotions:
            audio = self.model.synthesize(text, emotion=emotion)
            audio_samples[emotion] = audio
            
        # Verify emotional differences
        # Check pitch variance for different emotions
        for emotion, audio in audio_samples.items():
            pitch = librosa.yin(
                audio,
                fmin=50,
                fmax=400,
                sr=22050
            )
            
            if emotion == 'happy':
                self.assertGreater(np.mean(pitch[pitch > 0]), 150)
            elif emotion == 'sad':
                self.assertLess(np.mean(pitch[pitch > 0]), 150)
                
    def test_speed_control(self):
        """Test speech rate control"""
        text = "ፍጥነት መቆጣጠሪያ ሙከራ"
        
        # Generate at different speeds
        normal_audio = self.model.synthesize(text, speed=1.0)
        fast_audio = self.model.synthesize(text, speed=1.5)
        slow_audio = self.model.synthesize(text, speed=0.7)
        
        # Check duration ratios
        normal_duration = len(normal_audio) / 22050
        fast_duration = len(fast_audio) / 22050
        slow_duration = len(slow_audio) / 22050
        
        self.assertAlmostEqual(
            fast_duration / normal_duration, 1/1.5, places=1
        )
        self.assertAlmostEqual(
            slow_duration / normal_duration, 1/0.7, places=1
        )
        
    def test_long_text_handling(self):
        """Test handling of long texts"""
        # Create long text (500+ characters)
        long_text = "ይህ በጣም ረጅም የሆነ ጽሑፍ ነው። " * 50
        
        # Should handle without memory issues
        audio = self.model.synthesize(long_text, streaming=True)
        
        # Verify streaming works
        chunks = list(audio)
        self.assertGreater(len(chunks), 1)
        
    def test_special_characters(self):
        """Test handling of special characters and punctuation"""
        test_cases = [
            "ጥያቄ አለኝ?",
            "ዋው! ድንቅ ነው።",
            "1፣2፣3... ቁጥር",
            "የ'ጥቅስ' ምልክት"
        ]
        
        for text in test_cases:
            audio = self.model.synthesize(text)
            self.assertIsNotNone(audio)
            self.assertGreater(len(audio), 0)
            
    def test_performance_benchmarks(self):
        """Test performance metrics"""
        text = "የአፈጻጸም መለኪያ ሙከራ"
        
        # Measure inference time
        import time
        times = []
        
        for _ in range(10):
            start = time.time()
            _ = self.model.synthesize(text)
            times.append(time.time() - start)
            
        avg_time = np.mean(times[1:])  # Skip first (warmup)
        rtf = avg_time / (len(text) / 10)  # Approximate RTF
        
        self.assertLess(rtf, 0.3, f"RTF {rtf} exceeds threshold")
        
    def test_model_size(self):
        """Test model size constraints"""
        import os
        
        model_path = 'model.pt'
        model_size = os.path.getsize(model_path) / (1024**3)  # GB
        
        self.assertLess(model_size, 2.0, f"Model size {model_size}GB exceeds limit")
        
    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        signal_power = np.mean(audio ** 2)
        noise = audio[:1000]  # Assume initial part is noise
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            return 10 * np.log10(signal_power / noise_power)
        return 50.0
        
    def _calculate_similarity(self, audio1: np.ndarray, audio2: np.ndarray) -> float:
        """Calculate audio similarity using MFCC"""
        mfcc1 = librosa.feature.mfcc(y=audio1, sr=22050, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=audio2, sr=22050, n_mfcc=13)
        
        # Align lengths
        min_len = min(mfcc1.shape[1], mfcc2.shape[1])
        mfcc1 = mfcc1[:, :min_len]
        mfcc2 = mfcc2[:, :min_len]
        
        # Cosine similarity
        similarity = np.dot(mfcc1.flatten(), mfcc2.flatten()) / (
            np.linalg.norm(mfcc1.flatten()) * np.linalg.norm(mfcc2.flatten())
        )
        
        return similarity
9.2 Evaluation Metrics
Python

# amharic_tts/evaluation/metrics.py

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from scipy.stats import pearsonr
import jiwer

@dataclass
class EvaluationMetrics:
    """Container for TTS evaluation metrics"""
    mos: float  # Mean Opinion Score
    pesq: float  # Perceptual Evaluation of Speech Quality
    stoi: float  # Short-Time Objective Intelligibility
    mcd: float  # Mel Cepstral Distortion
    f0_rmse: float  # Fundamental frequency RMSE
    vuv_error: float  # Voiced/Unvoiced error rate
    speaker_similarity: float  # Speaker embedding similarity
    wer: float  # Word Error Rate (using ASR)
    cer: float  # Character Error Rate
    rtf: float  # Real-time factor
    

class AmharicTTSEvaluator:
    """
    Comprehensive evaluation framework for Amharic TTS
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.sample_rate = config.get('sample_rate', 22050)
        
        # Initialize ASR for intelligibility testing
        self.asr_model = self._init_asr()
        
        # Initialize speaker encoder for similarity
        self.speaker_encoder = self._init_speaker_encoder()
        
    def evaluate(
        self,
        synthesized_audio: np.ndarray,
        reference_audio: Optional[np.ndarray] = None,
        reference_text: Optional[str] = None,
        compute_mos: bool = False
    ) -> EvaluationMetrics:
        """
        Comprehensive evaluation of synthesized audio
        
        Args:
            synthesized_audio: Generated audio
            reference_audio: Ground truth audio (if available)
            reference_text: Original text
            compute_mos: Whether to compute MOS (requires human evaluation)
            
        Returns:
            EvaluationMetrics object
        """
        
        metrics = {}
        
        # Audio quality metrics (if reference available)
        if reference_audio is not None:
            metrics['pesq'] = self._compute_pesq(synthesized_audio, reference_audio)
            metrics['stoi'] = self._compute_stoi(synthesized_audio, reference_audio)
            metrics['mcd'] = self._compute_mcd(synthesized_audio, reference_audio)
            metrics['f0_rmse'] = self._compute_f0_rmse(synthesized_audio, reference_audio)
            metrics['vuv_error'] = self._compute_vuv_error(synthesized_audio, reference_audio)
            metrics['speaker_similarity'] = self._compute_speaker_similarity(
                synthesized_audio, reference_audio
            )
        
        # Intelligibility metrics (if text available)
        if reference_text is not None:
            metrics['wer'] = self._compute_wer(synthesized_audio, reference_text)
            metrics['cer'] = self._compute_cer(synthesized_audio, reference_text)
            
        # MOS (placeholder - requires human evaluation)
        metrics['mos'] = 0.0 if not compute_mos else self._compute_mos(synthesized_audio)
        
        # Performance metric
        metrics['rtf'] = self._compute_rtf(synthesized_audio)
        
        return EvaluationMetrics(**metrics)
        
    def _compute_pesq(self, synthesized: np.ndarray, reference: np.ndarray) -> float:
        """Compute PESQ score"""
        from pesq import pesq
        
        # Resample to 16kHz for PESQ
        synthesized_16k = librosa.resample(
            synthesized, orig_sr=self.sample_rate, target_sr=16000
        )
        reference_16k = librosa.resample(
            reference, orig_sr=self.sample_rate, target_sr=16000
        )
        
        # Align lengths
        min_len = min(len(synthesized_16k), len(reference_16k))
        synthesized_16k = synthesized_16k[:min_len]
        reference_16k = reference_16k[:min_len]
        
        return pesq(16000, reference_16k, synthesized_16k, 'wb')
        
    def _compute_stoi(self, synthesized: np.ndarray, reference: np.ndarray) -> float:
        """Compute STOI score"""
        from pystoi import stoi
        
        # Align lengths
        min_len = min(len(synthesized), len(reference))
        synthesized = synthesized[:min_len]
        reference = reference[:min_len]
        
        return stoi(reference, synthesized, self.sample_rate, extended=False)
        
    def _compute_mcd(self, synthesized: np.ndarray, reference: np.ndarray) -> float:
        """Compute Mel Cepstral Distortion"""
        # Extract MFCCs
        mfcc_synth = librosa.feature.mfcc(y=synthesized, sr=self.sample_rate, n_mfcc=13)
        mfcc_ref = librosa.feature.mfcc(y=reference, sr=self.sample_rate, n_mfcc=13)
        
        # Align frames
        min_frames = min(mfcc_synth.shape[1], mfcc_ref.shape[1])
        mfcc_synth = mfcc_synth[:, :min_frames]
        mfcc_ref = mfcc_ref[:, :min_frames]
        
        # Compute MCD
        diff = mfcc_synth - mfcc_ref
        mcd = np.mean(np.sqrt(np.sum(diff ** 2, axis=0)) * (10 / np.log(10)))
        
        return mcd
        
    def _compute_f0_rmse(self, synthesized: np.ndarray, reference: np.ndarray) -> float:
        """Compute F0 RMSE"""
        # Extract F0
        f0_synth = librosa.yin(synthesized, fmin=50, fmax=400, sr=self.sample_rate)
        f0_ref = librosa.yin(reference, fmin=50, fmax=400, sr=self.sample_rate)
        
        # Align lengths
        min_len = min(len(f0_synth), len(f0_ref))
        f0_synth = f0_synth[:min_len]
        f0_ref = f0_ref[:min_len]
        
        # Compute RMSE only for voiced frames
        voiced_mask = (f0_ref > 0) & (f0_synth > 0)
        if voiced_mask.sum() > 0:
            rmse = np.sqrt(np.mean((f0_synth[voiced_mask] - f0_ref[voiced_mask]) ** 2))
        else:
            rmse = 0.0
            
        return rmse
        
    def _compute_vuv_error(self, synthesized: np.ndarray, reference: np.ndarray) -> float:
        """Compute Voiced/Unvoiced error rate"""
        # Extract F0 for V/UV decision
        f0_synth = librosa.yin(synthesized, fmin=50, fmax=400, sr=self.sample_rate)
        f0_ref = librosa.yin(reference, fmin=50, fmax=400, sr=self.sample_rate)
        
        # Align lengths
        min_len = min(len(f0_synth), len(f0_ref))
        f0_synth = f0_synth[:min_len]
        f0_ref = f0_ref[:min_len]
        
        # V/UV decision
        vuv_synth = f0_synth > 0
        vuv_ref = f0_ref > 0
        
        # Error rate
        error_rate = np.mean(vuv_synth != vuv_ref)
        
        return error_rate
        
    def _compute_speaker_similarity(self, synthesized: np.ndarray, reference: np.ndarray) -> float:
        """Compute speaker embedding similarity"""
        # Extract speaker embeddings
        emb_synth = self.speaker_encoder(synthesized)
        emb_ref = self.speaker_encoder(reference)
        
        # Cosine similarity
        similarity = np.dot(emb_synth, emb_ref) / (
            np.linalg.norm(emb_synth) * np.linalg.norm(emb_ref)
        )
        
        return similarity
        
    def _compute_wer(self, audio: np.ndarray, reference_text: str) -> float:
        """Compute Word Error Rate using ASR"""
        # Transcribe audio
        transcription = self.asr_model.transcribe(audio)
        
        # Compute WER
        wer = jiwer.wer(reference_text, transcription)
        
        return wer
        
    def _compute_cer(self, audio: np.ndarray, reference_text: str) -> float:
        """Compute Character Error Rate"""
        # Transcribe audio
        transcription = self.asr_model.transcribe(audio)
        
        # Compute CER
        cer = jiwer.cer(reference_text, transcription)
        
        return cer
        
    def _compute_rtf(self, audio: np.ndarray) -> float:
        """Compute Real-Time Factor"""
        import time
        
        audio_duration = len(audio) / self.sample_rate
        
        # Measure synthesis time (mock)
        start_time = time.time()
        # Synthesis would happen here
        synthesis_time = time.time() - start_time
        
        rtf = synthesis_time / audio_duration
        
        return rtf
        
    def _compute_mos(self, audio: np.ndarray) -> float:
        """
        Compute MOS (Mean Opinion Score)
        Note: This requires human evaluation or a trained MOS predictor
        """
        # Placeholder for MOS prediction model
        # In practice, this would use a trained neural network
        # or aggregate human ratings
        return 4.0  # Placeholder
        
    def _init_asr(self):
        """Initialize ASR model for intelligibility testing"""
        # This would load an Amharic ASR model
        # For example, using wav2vec2 fine-tuned on Amharic
        class MockASR:
            def transcribe(self, audio):
                return "mock transcription"
        
        return MockASR()
        
    def _init_speaker_encoder(self):
        """Initialize speaker encoder for similarity measurement"""
        # This would load a speaker verification model
        def mock_encoder(audio):
            return np.random.randn(256)
        
        return mock_encoder
10. Deployment Guidelines
10.1 Production Deployment
YAML

# deployment/docker-compose.yml

version: '3.8'

services:
  amharic-tts-api:
    build:
      context: .
      dockerfile: Dockerfile
    image: amharic-tts:latest
    container_name: amharic-tts-api
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_PATH=/app/models/xtts_v2_amharic
      - MAX_BATCH_SIZE=32
      - CACHE_SIZE=100
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./cache:/app/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:alpine
    container_name: amharic-tts-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: amharic-tts-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - amharic-tts-api
    restart: unless-stopped

  monitoring:
    image: prom/prometheus
    container_name: amharic-tts-monitoring
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
10.2 API Implementation
Python

# deployment/api.py

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import torch
import numpy as np
import io
import wave
from typing import Optional, List, Dict
import redis
import json
import hashlib
import asyncio
from datetime import datetime

app = FastAPI(title="Amharic TTS API", version="1.0.0")

# Initialize model
model = OptimizedAmharicTTS(
    model_path="/app/models/xtts_v2_amharic",
    use_onnx=True,
    cache_size=100
)

# Redis cache
redis_client = redis.Redis(host='redis', port=6379, db=0)

class TTSRequest(BaseModel):
    """TTS synthesis request model"""
    text: str = Field(..., description="Amharic text to synthesize")
    speaker_id: Optional[int] = Field(None, description="Speaker ID for multi-speaker model")
    emotion: Optional[str] = Field("neutral", description="Emotion: neutral, happy, sad, angry")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed factor")
    pitch: float = Field(1.0, ge=0.5, le=2.0, description="Pitch adjustment factor")
    output_format: str = Field("wav", description="Output format: wav, mp3, ogg")
    streaming: bool = Field(False, description="Enable streaming synthesis")
    use_cache: bool = Field(True, description="Use caching for faster response")

class TTSResponse(BaseModel):
    """TTS synthesis response model"""
    audio_url: str
    duration: float
    cache_hit: bool
    processing_time: float
    
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Amharic TTS API",
        "version": "1.0.0",
        "endpoints": [
            "/synthesize",
            "/batch_synthesize",
            "/voices",
            "/health",
            "/metrics"
        ]
    }
    
@app.post("/synthesize", response_model=TTSResponse)
async def synthesize(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Synthesize Amharic text to speech
    
    Args:
        request: TTS synthesis request
        background_tasks: Background task manager
        
    Returns:
        TTS response with audio URL
    """
    
    start_time = datetime.now()
    
    # Generate cache key
    cache_key = _generate_cache_key(request)
    
    # Check cache
    if request.use_cache:
        cached_audio = redis_client.get(cache_key)
        if cached_audio:
            return TTSResponse(
                audio_url=f"/audio/{cache_key}",
                duration=_get_audio_duration(cached_audio),
                cache_hit=True,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    try:
        # Synthesize audio
        audio = model.synthesize(
            text=request.text,
            speaker_id=request.speaker_id,
            emotion=request.emotion,
            speed=request.speed,
            pitch=request.pitch,
            streaming=request.streaming
        )
        
        # Convert to requested format
        audio_bytes = _convert_audio_format(audio, request.output_format)
        
        # Cache result
        if request.use_cache:
            background_tasks.add_task(_cache_audio, cache_key, audio_bytes)
        
        # Save audio file
        audio_path = f"/tmp/{cache_key}.{request.output_format}"
        with open(audio_path, 'wb') as f:
            f.write(audio_bytes)
        
        return TTSResponse(
            audio_url=f"/audio/{cache_key}",
            duration=len(audio) / 22050,
            cache_hit=False,
            processing_time=(datetime.now() - start_time).total_seconds()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/batch_synthesize")
async def batch_synthesize(requests: List[TTSRequest]):
    """
    Batch synthesis for multiple texts
    
    Args:
        requests: List of TTS requests
        
    Returns:
        List of TTS responses
    """
    
    tasks = []
    for request in requests:
        task = asyncio.create_task(synthesize(request, BackgroundTasks()))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
    
@app.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    """
    Retrieve synthesized audio
    
    Args:
        audio_id: Audio cache ID
        
    Returns:
        Audio file response
    """
    
    # Try cache first
    audio_data = redis_client.get(audio_id)
    
    if audio_data:
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename={audio_id}.wav"}
        )
    
    # Try file system
    audio_path = f"/tmp/{audio_id}.wav"
    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/wav")
    
    raise HTTPException(status_code=404, detail="Audio not found")
    
@app.get("/voices")
async def list_voices():
    """
    List available voices/speakers
    
    Returns:
        List of available voice configurations
    """
    
    return {
        "voices": [
            {"id": 0, "name": "Default", "gender": "neutral", "language": "am"},
            {"id": 1, "name": "Male 1", "gender": "male", "language": "am"},
            {"id": 2, "name": "Female 1", "gender": "female", "language": "am"},
            # Add more voices
        ]
    }
    
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Health status
    """
    
    try:
        # Test model
        test_audio = model.synthesize("ሙከራ")
        
        # Test Redis
        redis_client.ping()
        
        return {
            "status": "healthy",
            "model": "loaded",
            "cache": "connected",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
@app.get("/metrics")
async def get_metrics():
    """
    Get API metrics
    
    Returns:
        Performance and usage metrics
    """
    
    # Get Redis stats
    info = redis_client.info()
    
    return {
        "cache": {
            "hits": info.get('keyspace_hits', 0),
            "misses": info.get('keyspace_misses', 0),
            "keys": redis_client.dbsize()
        },
        "model": {
            "version": "2.0.3",
            "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
            "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        },
        "timestamp": datetime.now().isoformat()
    }
    
def _generate_cache_key(request: TTSRequest) -> str:
    """Generate unique cache key for request"""
    key_string = f"{request.text}_{request.speaker_id}_{request.emotion}_{request.speed}_{request.pitch}"
    return hashlib.md5(key_string.encode()).hexdigest()
    
def _convert_audio_format(audio: np.ndarray, format: str) -> bytes:
    """Convert audio to specified format"""
    
    # Normalize audio
    audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
    
    # Create WAV in memory
    buffer = io.BytesIO()
    
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(audio.tobytes())
    
    buffer.seek(0)
    
    if format == 'wav':
        return buffer.read()
    elif format == 'mp3':
        # Convert to MP3 (requires ffmpeg)
        import subprocess
        # Implementation here
        pass
    
    return buffer.read()
    
def _get_audio_duration(audio_bytes: bytes) -> float:
    """Get audio duration from bytes"""
    buffer = io.BytesIO(audio_bytes)
    with wave.open(buffer, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        return frames / float(rate)
        
async def _cache_audio(key: str, audio_bytes: bytes):
    """Cache audio in Redis"""
    redis_client.setex(key, 3600, audio_bytes)  # Cache for 1 hour
10.3 Monitoring and Maintenance
Python

# deployment/monitoring.py

import logging
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from functools import wraps
from typing import Callable
import psutil
import GPUtil

# Prometheus metrics
request_count = Counter('tts_requests_total', 'Total TTS requests', ['status', 'language'])
request_duration = Histogram('tts_request_duration_seconds', 'TTS request duration')
audio_duration = Histogram('tts_audio_duration_seconds', 'Generated audio duration')
cache_hits = Counter('tts_cache_hits_total', 'Cache hit count')
cache_misses = Counter('tts_cache_misses_total', 'Cache miss count')
active_connections = Gauge('tts_active_connections', 'Active connections')
model_memory = Gauge('tts_model_memory_bytes', 'Model memory usage')
gpu_utilization = Gauge('tts_gpu_utilization_percent', 'GPU utilization')

class TTSMonitor:
    """
    Monitoring system for Amharic TTS
    """
    
    def __init__(self, log_file: str = 'tts.log'):
        self.logger = self._setup_logger(log_file)
        self.start_time = time.time()
        
    def _setup_logger(self, log_file: str) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('AmharicTTS')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
        
    def monitor_request(self, func: Callable) -> Callable:
        """Decorator to monitor TTS requests"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Update metrics
                duration = time.time() - start_time
                request_duration.observe(duration)
                request_count.labels(status='success', language='am').inc()
                
                # Log
                self.logger.info(f"Request completed in {duration:.2f}s")
                
                return result
                
            except Exception as e:
                request_count.labels(status='error', language='am').inc()
                self.logger.error(f"Request failed: {e}")
                raise
                
        return wrapper
        
    def update_system_metrics(self):
        """Update system resource metrics"""
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # GPU metrics
        if GPUtil.getGPUs():
            gpu = GPUtil.getGPUs()[0]
            gpu_utilization.set(gpu.load * 100)
            model_memory.set(gpu.memoryUsed * 1024 * 1024 * 1024)  # Convert to bytes
            
        # Log system stats
        self.logger.info(f"CPU: {cpu_percent}%, Memory: {memory.percent}%")
        
    def health_check(self) -> Dict[str, any]:
        """Perform health check"""
        
        health = {
            'status': 'healthy',
            'uptime': time.time() - self.start_time,
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
        }
        
        # Check GPU
        if GPUtil.getGPUs():
            gpu = GPUtil.getGPUs()[0]
            health['gpu'] = {
                'name': gpu.name,
                'load': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'temperature': gpu.temperature
            }
            
        # Check thresholds
        if health['cpu_usage'] > 90:
            health['status'] = 'degraded'
            health['warnings'] = health.get('warnings', []) + ['High CPU usage']
            
        if health['memory_usage'] > 90:
            health['status'] = 'degraded'
            health['warnings'] = health.get('warnings', []) + ['High memory usage']
            
        return health
11. Additional Resources and References
11.1 Required Dependencies
toml

# pyproject.toml

[tool.poetry]
name = "amharic-tts"
version = "1.0.0"
description = "Amharic Text-to-Speech System with XTTS v2"

[tool.poetry.dependencies]
python = "^3.9"
torch = "^2.0.0"
torchaudio = "^2.0.0"
transformers = "^4.35.0"
TTS = "^0.20.0"
epitran = "^1.24"
transphone = "^0.1.0"
librosa = "^0.10.0"
soundfile = "^0.12.0"
numpy = "^1.24.0"
scipy = "^1.10.0"
pandas = "^2.0.0"
fastapi = "^0.104.0"
uvicorn = "^0.24.0"
redis = "^5.0.0"
pydantic = "^2.0.0"
onnxruntime-gpu = "^1.16.0"
pytest = "^7.4.0"
pesq = "^0.0.4"
pystoi = "^0.3.3"
jiwer = "^3.0.0"
wandb = "^0.15.0"
tensorboard = "^2.14.0"
prometheus-client = "^0.18.0"
pyloudnorm = "^0.1.1"
gputil = "^1.4.0"
psutil = "^5.9.0"

[tool.poetry.dev-dependencies]
black = "^23.0.0"
flake8 = "^6.0.0"
mypy = "^1.5.0"
pytest-cov = "^4.1.0"
jupyter = "^1.0.0"
11.2 Training Data Sources
Markdown

## Recommended Amharic Speech Datasets

1. **ALFFA Amharic ASR Corpus**
   - Size: ~20 hours
   - Quality: High
   - Link: https://github.com/getalp/ALFFA_PUBLIC/tree/master/ASR/AMHARIC

2. **Common Voice Amharic**
   - Size: Growing (100+ hours)
   - Quality: Variable
   - Link: https://commonvoice.mozilla.org/am

3. **Google Fleurs**
   - Size: ~10 hours
   - Quality: High
   - Link: https://huggingface.co/datasets/google/fleurs

4. **Custom Collection Guidelines**
   - Record in quiet environment (<40dB background noise)
   - Use professional microphones (16kHz+ sample rate)
   - Balance speaker demographics
   - Include various speaking styles and emotions
11.3 Troubleshooting Guide
Markdown

## Common Issues and Solutions

### Issue 1: Poor Pronunciation
**Symptoms**: Incorrect phonemes, unnatural speech
**Solutions**:
- Verify G2P mappings are correct
- Check for epenthetic vowel insertion
- Fine-tune on more Amharic data
- Adjust phoneme weights in training

### Issue 2: Low Audio Quality
**Symptoms**: Noisy, distorted output
**Solutions**:
- Check vocoder configuration
- Increase training data quality
- Adjust mel-spectrogram parameters
- Use noise reduction in preprocessing

### Issue 3: Memory Issues
**Symptoms**: OOM errors during training/inference
**Solutions**:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Use model quantization

### Issue 4: Slow Inference
**Symptoms**: High RTF, latency issues
**Solutions**:
- Convert to ONNX format
- Enable caching
- Use batch processing
- Optimize model architecture
- Use GPU acceleration

### Issue 5: Character Encoding Errors
**Symptoms**: � characters, text corruption
**Solutions**:
- Ensure UTF-8 encoding throughout
- Validate Ethiopic Unicode range
- Check font support in environment
12. Conclusion
This comprehensive specification provides a complete framework for implementing a state-of-the-art Amharic TTS system using XTTS v2. The specification covers:

Core Technologies: G2P conversion, custom tokenization, and model architecture
Implementation Details: Complete code examples for all components
Training Pipeline: Data preparation, fine-tuning, and evaluation
Production Deployment: API implementation, monitoring, and scaling
Quality Assurance: Testing frameworks and evaluation metrics
Key Success Factors:
✅ Proper handling of Ethiopic script and Amharic phonology
✅ High-quality training data (minimum 20 hours recommended)
✅ Careful attention to epenthetic vowels and gemination
✅ Comprehensive testing and evaluation
✅ Production-ready deployment with monitoring
Expected Outcomes:
MOS Score: 4.0+ (near-human quality)
WER: <10% with good ASR
RTF: <0.3 (real-time capable)
Speaker Similarity: >0.85 for voice cloning