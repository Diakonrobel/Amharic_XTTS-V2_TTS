"""
Test Script for Advanced Dataset Processing Features
Tests SRT processor, YouTube downloader, and Audio slicer modules
"""

import os
import sys
from pathlib import Path
import traceback

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def print_header(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def print_success(message):
    """Print success message"""
    print(f"✓ {message}")

def print_error(message):
    """Print error message"""
    print(f"✗ {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ {message}")


def test_imports():
    """Test if all modules can be imported"""
    print_header("TEST 1: Module Imports")
    
    success = True
    
    # Test SRT processor
    try:
        from utils.srt_processor import parse_srt_file, process_srt_with_media
        print_success("SRT Processor imported successfully")
    except Exception as e:
        print_error(f"Failed to import SRT Processor: {e}")
        success = False
    
    # Test YouTube downloader
    try:
        from utils.youtube_downloader import download_youtube_video, get_video_info
        print_success("YouTube Downloader imported successfully")
    except Exception as e:
        print_error(f"Failed to import YouTube Downloader: {e}")
        success = False
    
    # Test Audio slicer
    try:
        from utils.audio_slicer import Slicer, slice_audio_file
        print_success("Audio Slicer imported successfully")
    except Exception as e:
        print_error(f"Failed to import Audio Slicer: {e}")
        success = False
    
    # Test dependencies
    print("\n--- Checking Dependencies ---")
    
    deps = {
        'pysrt': 'SRT parsing',
        'yt_dlp': 'YouTube download',
        'soundfile': 'Audio I/O',
        'librosa': 'Audio processing',
        'torchaudio': 'PyTorch audio',
        'pandas': 'Dataset metadata',
        'tqdm': 'Progress bars'
    }
    
    for dep, description in deps.items():
        try:
            __import__(dep)
            print_success(f"{dep:15s} - {description}")
        except ImportError:
            print_error(f"{dep:15s} - {description} (MISSING!)")
            success = False
    
    return success


def test_srt_parser():
    """Test SRT parsing functionality"""
    print_header("TEST 2: SRT Parser")
    
    try:
        from utils.srt_processor import parse_srt_file
        import tempfile
        
        # Create a test SRT file
        test_srt_content = """1
00:00:01,000 --> 00:00:03,000
Hello, this is a test.

2
00:00:04,000 --> 00:00:07,000
Testing SRT file parsing functionality.

3
00:00:08,500 --> 00:00:11,000
This should work correctly.
"""
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
            f.write(test_srt_content)
            temp_srt = f.name
        
        print_info(f"Created test SRT file: {temp_srt}")
        
        # Parse the SRT file
        segments = parse_srt_file(temp_srt)
        
        # Cleanup
        os.unlink(temp_srt)
        
        if segments and len(segments) == 3:
            print_success(f"Parsed {len(segments)} segments successfully")
            for i, (start, end, text) in enumerate(segments[:2], 1):
                print(f"  Segment {i}: {start:.2f}s - {end:.2f}s | {text[:30]}...")
            return True
        else:
            print_error(f"Expected 3 segments, got {len(segments) if segments else 0}")
            return False
            
    except Exception as e:
        print_error(f"SRT Parser test failed: {e}")
        traceback.print_exc()
        return False


def test_audio_slicer():
    """Test audio slicer with a synthetic audio file"""
    print_header("TEST 3: Audio Slicer")
    
    try:
        from utils.audio_slicer import Slicer
        import numpy as np
        import tempfile
        
        # Create synthetic audio: 5 seconds of audio with silence
        sr = 22050
        duration = 5
        
        # Create audio with speech (noise) and silence pattern
        audio = np.zeros(sr * duration, dtype=np.float32)
        
        # Add "speech" (noise) in segments
        audio[0:sr] = np.random.randn(sr) * 0.3  # 1 sec speech
        audio[sr:int(sr*1.5)] = 0  # 0.5 sec silence
        audio[int(sr*1.5):int(sr*3)] = np.random.randn(int(sr*1.5)) * 0.3  # 1.5 sec speech
        audio[int(sr*3):int(sr*3.5)] = 0  # 0.5 sec silence
        audio[int(sr*3.5):int(sr*5)] = np.random.randn(int(sr*1.5)) * 0.3  # 1.5 sec speech
        
        print_info(f"Created synthetic audio: {duration}s at {sr}Hz")
        
        # Test slicer
        slicer = Slicer(
            sr=sr,
            threshold=-40.0,
            min_length=5000,
            min_interval=300,
            hop_size=20,
            max_sil_kept=500
        )
        
        chunks = slicer.slice(audio)
        
        if chunks:
            print_success(f"Audio slicer produced {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                duration_sec = len(chunk) / sr
                print(f"  Chunk {i+1}: {duration_sec:.2f}s")
            return True
        else:
            print_error("Audio slicer produced no chunks")
            return False
            
    except Exception as e:
        print_error(f"Audio Slicer test failed: {e}")
        traceback.print_exc()
        return False


def test_youtube_info():
    """Test YouTube video info fetching (no download)"""
    print_header("TEST 4: YouTube Info Fetching")
    
    try:
        from utils.youtube_downloader import get_video_info
        
        # Use a known stable YouTube video (YouTube's own "YouTube Rewind 2010")
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        
        print_info(f"Fetching info from: {test_url}")
        print_info("(This doesn't download anything, just gets metadata)")
        
        info = get_video_info(test_url)
        
        if info and 'title' in info:
            print_success("Successfully fetched video info")
            print(f"  Title: {info['title']}")
            print(f"  Duration: {info.get('duration', 'N/A')}s")
            print(f"  Has subtitles: {info.get('has_subtitles', False)}")
            print(f"  Has auto-captions: {info.get('has_automatic_captions', False)}")
            return True
        else:
            print_error("Failed to fetch video info")
            return False
            
    except Exception as e:
        print_error(f"YouTube info test failed: {e}")
        print_info("Note: This might fail if you're behind a firewall or yt-dlp needs updating")
        traceback.print_exc()
        return False


def test_ffmpeg():
    """Test if FFmpeg is available"""
    print_header("TEST 5: FFmpeg Availability")
    
    try:
        import subprocess
        
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print_success(f"FFmpeg is available: {version_line}")
            return True
        else:
            print_error("FFmpeg command failed")
            return False
            
    except FileNotFoundError:
        print_error("FFmpeg not found in PATH")
        print_info("Install FFmpeg: winget install FFmpeg")
        print_info("Or download from: https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print_error(f"FFmpeg test failed: {e}")
        return False


def test_integration_srt():
    """Test full SRT + media processing with synthetic data"""
    print_header("TEST 6: SRT + Media Integration (Synthetic)")
    
    try:
        from utils.srt_processor import extract_segments_from_audio
        import numpy as np
        import soundfile as sf
        import tempfile
        from pathlib import Path
        
        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create synthetic audio (10 seconds)
        sr = 22050
        duration = 10
        audio = np.random.randn(sr * duration).astype(np.float32) * 0.1
        
        # Save audio
        audio_file = temp_dir / "test_audio.wav"
        sf.write(str(audio_file), audio, sr)
        
        print_info(f"Created test audio: {audio_file}")
        
        # Create SRT segments
        srt_segments = [
            (0.0, 2.0, "First segment"),
            (2.5, 5.0, "Second segment"),
            (6.0, 9.0, "Third segment")
        ]
        
        print_info(f"Testing with {len(srt_segments)} segments")
        
        # Process segments
        output_dir = temp_dir / "output"
        train_csv, eval_csv = extract_segments_from_audio(
            audio_path=str(audio_file),
            srt_segments=srt_segments,
            output_dir=str(output_dir),
            speaker_name="test_speaker",
            min_duration=0.5,
            max_duration=15.0
        )
        
        # Check results
        if Path(train_csv).exists() and Path(eval_csv).exists():
            import pandas as pd
            train_df = pd.read_csv(train_csv, sep='|')
            eval_df = pd.read_csv(eval_csv, sep='|')
            
            total_segments = len(train_df) + len(eval_df)
            print_success(f"Created dataset with {total_segments} segments")
            print(f"  Training: {len(train_df)} samples")
            print(f"  Evaluation: {len(eval_df)} samples")
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)
            
            return True
        else:
            print_error("Failed to create CSV files")
            return False
            
    except Exception as e:
        print_error(f"SRT integration test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and provide summary"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "ADVANCED FEATURES TEST SUITE" + " "*35 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        ("Module Imports", test_imports),
        ("SRT Parser", test_srt_parser),
        ("Audio Slicer", test_audio_slicer),
        ("YouTube Info", test_youtube_info),
        ("FFmpeg Check", test_ffmpeg),
        ("SRT Integration", test_integration_srt),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} | {test_name}")
    
    print("\n" + "-"*80)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("All tests passed! Features are ready to use.")
        return 0
    else:
        print_error(f"{total - passed} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    print("\nAdvanced Dataset Processing Features - Test Script")
    print("Testing: SRT processor, YouTube downloader, Audio slicer\n")
    
    try:
        exit_code = run_all_tests()
        
        print("\n" + "="*80)
        print("\nNext steps:")
        print("  1. If all tests passed, proceed with UI integration")
        print("  2. If tests failed, check the error messages above")
        print("  3. See IMPLEMENTATION_PLAN.md for detailed integration guide")
        print("\n" + "="*80 + "\n")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        sys.exit(1)
