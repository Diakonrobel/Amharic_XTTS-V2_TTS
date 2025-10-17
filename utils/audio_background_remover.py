"""
Audio Background Music Remover Module
Uses Demucs for source separation to extract clean vocals from audio with background music.

Demucs is a state-of-the-art music source separation model developed by Meta.
It separates audio into vocals, drums, bass, and other instruments.

Installation:
    pip install demucs

Models available (in order of quality vs speed):
    - htdemucs: Best quality, slowest (default)
    - htdemucs_ft: Fine-tuned version, slightly better
    - mdx_extra: Good balance of speed/quality
    - mdx: Faster, good quality
    
For TTS dataset creation, we only need the vocals track.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Literal
import torch
import torchaudio

# Model types
ModelType = Literal["htdemucs", "htdemucs_ft", "mdx_extra", "mdx", "htdemucs_6s"]


class AudioBackgroundRemover:
    """
    Remove background music from audio files using Demucs source separation.
    
    Features:
    - Multiple model options (quality vs speed trade-off)
    - GPU acceleration support
    - Automatic cleanup of temporary files
    - Preserves original audio format
    """
    
    def __init__(
        self,
        model: ModelType = "htdemucs",
        device: Optional[str] = None,
        shifts: int = 1,
        overlap: float = 0.25,
        verbose: bool = True
    ):
        """
        Initialize background remover.
        
        Args:
            model: Demucs model to use
                - "htdemucs": Best quality (default)
                - "htdemucs_ft": Fine-tuned, slightly better
                - "mdx_extra": Balanced speed/quality
                - "mdx": Faster inference
                - "htdemucs_6s": 6-stem model (vocals, drums, bass, other, guitar, piano)
            device: Device to use ("cuda", "cpu", or None for auto-detect)
            shifts: Number of random shifts for better quality (1=fast, 10=best quality)
            overlap: Overlap between splits (0.25 = good default)
            verbose: Print progress information
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.shifts = shifts
        self.overlap = overlap
        self.verbose = verbose
        
        # Check if demucs is installed
        self._check_installation()
        
        if self.verbose:
            print(f"✓ Audio Background Remover initialized")
            print(f"  Model: {self.model}")
            print(f"  Device: {self.device}")
            print(f"  Quality: {'High' if shifts > 1 else 'Fast'} (shifts={shifts})")
    
    def _check_installation(self):
        """Check if demucs is installed."""
        try:
            import demucs
            if self.verbose:
                print(f"  Demucs version: {demucs.__version__}")
        except ImportError:
            raise ImportError(
                "Demucs is not installed. Install with: pip install demucs"
            )
    
    def remove_background(
        self,
        input_audio: str,
        output_audio: str,
        extract_component: str = "vocals",
        keep_temp: bool = False
    ) -> str:
        """
        Remove background music and extract vocals from audio file.
        
        Args:
            input_audio: Path to input audio file
            output_audio: Path to save output audio file
            extract_component: Component to extract ("vocals", "drums", "bass", "other")
            keep_temp: Keep temporary separation files
            
        Returns:
            Path to output audio file with background removed
        """
        input_path = Path(input_audio)
        output_path = Path(output_audio)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input audio not found: {input_audio}")
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temporary directory for Demucs output
        temp_dir = Path(tempfile.mkdtemp(prefix="demucs_"))
        
        try:
            if self.verbose:
                print(f"\n🎵 Removing background music from: {input_path.name}")
                print(f"   Using model: {self.model}")
            
            # Build demucs command
            cmd = [
                "python", "-m", "demucs.separate",
                "-n", self.model,
                "-o", str(temp_dir),
                "--device", self.device,
                "--shifts", str(self.shifts),
                "--overlap", str(self.overlap),
            ]
            
            # Add two stems mode if we only want vocals
            if extract_component == "vocals":
                cmd.extend(["--two-stems", "vocals"])
            
            cmd.append(str(input_path))
            
            # Run demucs
            if self.verbose:
                print("   Processing... (this may take a few minutes)")
                result = subprocess.run(cmd, capture_output=False, text=True)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Demucs separation failed: {result.stderr}")
            
            # Find the separated vocals file
            # Demucs output structure: temp_dir / model_name / input_stem / component.wav
            model_output_dir = temp_dir / self.model / input_path.stem
            component_file = model_output_dir / f"{extract_component}.wav"
            
            if not component_file.exists():
                raise FileNotFoundError(
                    f"Expected output not found: {component_file}\n"
                    f"Available files: {list(model_output_dir.glob('*.wav'))}"
                )
            
            # Copy to output location
            import shutil
            shutil.copy2(component_file, output_path)
            
            if self.verbose:
                # Get file sizes for comparison
                input_size = input_path.stat().st_size / (1024 * 1024)
                output_size = output_path.stat().st_size / (1024 * 1024)
                print(f"   ✓ Background removed successfully!")
                print(f"   Input:  {input_size:.2f} MB")
                print(f"   Output: {output_size:.2f} MB")
                print(f"   Saved to: {output_path}")
            
            return str(output_path)
        
        finally:
            # Cleanup temporary files
            if not keep_temp:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def process_in_place(self, audio_path: str) -> str:
        """
        Remove background music and replace original file.
        
        Args:
            audio_path: Path to audio file to process
            
        Returns:
            Path to processed audio file (same as input)
        """
        temp_output = audio_path + ".temp_nobg.wav"
        
        try:
            self.remove_background(audio_path, temp_output)
            
            # Replace original
            import shutil
            shutil.move(temp_output, audio_path)
            
            return audio_path
        
        except Exception as e:
            # Cleanup temp file on error
            if os.path.exists(temp_output):
                os.remove(temp_output)
            raise e


def remove_background_music(
    input_audio: str,
    output_audio: Optional[str] = None,
    model: ModelType = "htdemucs",
    device: Optional[str] = None,
    quality: Literal["fast", "balanced", "best"] = "balanced",
    verbose: bool = True
) -> str:
    """
    Simple function to remove background music from audio file.
    
    Args:
        input_audio: Path to input audio file
        output_audio: Path to save output (None = replace input file)
        model: Demucs model to use
        device: Device to use ("cuda", "cpu", or None for auto)
        quality: Quality preset ("fast", "balanced", "best")
        verbose: Print progress
        
    Returns:
        Path to output audio file
        
    Example:
        >>> # Remove background and save to new file
        >>> output = remove_background_music(
        ...     "podcast_with_music.wav",
        ...     "podcast_clean.wav",
        ...     quality="best"
        ... )
        
        >>> # Remove background in-place (replace original)
        >>> remove_background_music("audio.wav", quality="fast")
    """
    # Map quality to shifts parameter
    quality_map = {
        "fast": 1,       # Fastest, good quality
        "balanced": 3,   # Good balance
        "best": 10       # Best quality, slowest
    }
    shifts = quality_map.get(quality, 3)
    
    # Initialize remover
    remover = AudioBackgroundRemover(
        model=model,
        device=device,
        shifts=shifts,
        verbose=verbose
    )
    
    # Process
    if output_audio is None:
        # In-place processing
        return remover.process_in_place(input_audio)
    else:
        # Save to new file
        return remover.remove_background(input_audio, output_audio)


# Convenience function for checking if background removal is available
def is_available() -> bool:
    """
    Check if background removal is available (demucs installed).
    
    Returns:
        True if demucs is installed and can be used
    """
    try:
        import demucs
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    # Test/demo usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python audio_background_remover.py <input_audio> [output_audio] [quality]")
        print("  quality: fast, balanced, or best (default: balanced)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    quality = sys.argv[3] if len(sys.argv) > 3 else "balanced"
    
    try:
        result = remove_background_music(
            input_file,
            output_file,
            quality=quality
        )
        print(f"\n✅ Success! Clean audio saved to: {result}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
