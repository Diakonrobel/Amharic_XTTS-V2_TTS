"""
Unit tests for batch_processor.pair_srt_with_media function.

Tests cover:
- Exact stem matches
- Simple language suffix stripping (e.g., ".en")
- Year+language suffix stripping (e.g., ".2025am")
- No match scenarios
- Multiple potential language-like suffixes
"""

import unittest
from pathlib import Path
from utils.batch_processor import pair_srt_with_media


class TestPairSrtWithMedia(unittest.TestCase):
    """Test suite for pair_srt_with_media function."""

    def test_exact_stem_matches(self):
        """Test that files with exact stem matches are correctly paired."""
        srt_files = [
            "C:/videos/movie1.srt",
            "C:/videos/documentary.srt",
            "C:/videos/episode_01.srt"
        ]
        media_files = [
            "C:/videos/movie1.mp4",
            "C:/videos/documentary.mkv",
            "C:/videos/episode_01.wav"
        ]
        
        pairs = pair_srt_with_media(srt_files, media_files)
        
        self.assertEqual(len(pairs), 3)
        self.assertIn(("C:/videos/movie1.srt", "C:/videos/movie1.mp4"), pairs)
        self.assertIn(("C:/videos/documentary.srt", "C:/videos/documentary.mkv"), pairs)
        self.assertIn(("C:/videos/episode_01.srt", "C:/videos/episode_01.wav"), pairs)

    def test_simple_language_suffix_stripping(self):
        """Test that files are paired after stripping simple language suffixes."""
        srt_files = [
            "C:/videos/movie1.en.srt",
            "C:/videos/tutorial_am.srt",
            "C:/videos/conference-fr.srt",
            "C:/videos/lecture.eng.srt"
        ]
        media_files = [
            "C:/videos/movie1.mp4",
            "C:/videos/tutorial.mp4",
            "C:/videos/conference.mkv",
            "C:/videos/lecture.wav"
        ]
        
        pairs = pair_srt_with_media(srt_files, media_files)
        
        self.assertEqual(len(pairs), 4)
        self.assertIn(("C:/videos/movie1.en.srt", "C:/videos/movie1.mp4"), pairs)
        self.assertIn(("C:/videos/tutorial_am.srt", "C:/videos/tutorial.mp4"), pairs)
        self.assertIn(("C:/videos/conference-fr.srt", "C:/videos/conference.mkv"), pairs)
        self.assertIn(("C:/videos/lecture.eng.srt", "C:/videos/lecture.wav"), pairs)

    def test_year_language_suffix_stripping(self):
        """Test that files are paired after stripping year+language suffixes."""
        srt_files = [
            "C:/videos/news.2025am.srt",
            "C:/videos/report_2024en.srt",
            "C:/videos/movie-2023es.srt",
            "C:/videos/documentary.2022amh.srt"
        ]
        media_files = [
            "C:/videos/news.mp4",
            "C:/videos/report.mp4",
            "C:/videos/movie.mkv",
            "C:/videos/documentary.wav"
        ]
        
        pairs = pair_srt_with_media(srt_files, media_files)
        
        self.assertEqual(len(pairs), 4)
        self.assertIn(("C:/videos/news.2025am.srt", "C:/videos/news.mp4"), pairs)
        self.assertIn(("C:/videos/report_2024en.srt", "C:/videos/report.mp4"), pairs)
        self.assertIn(("C:/videos/movie-2023es.srt", "C:/videos/movie.mkv"), pairs)
        self.assertIn(("C:/videos/documentary.2022amh.srt", "C:/videos/documentary.wav"), pairs)

    def test_no_match_found(self):
        """Test that no pairing occurs when no match is found."""
        srt_files = [
            "C:/videos/movie1.srt",
            "C:/videos/movie2.en.srt",
            "C:/videos/movie3.2025am.srt"
        ]
        media_files = [
            "C:/videos/other_video.mp4",
            "C:/videos/different_file.mkv"
        ]
        
        pairs = pair_srt_with_media(srt_files, media_files)
        
        # No pairs should be made
        self.assertEqual(len(pairs), 0)

    def test_multiple_language_suffixes_validation(self):
        """Test handling of files with multiple potential language-like suffixes."""
        srt_files = [
            # Valid language codes that should be stripped
            "C:/videos/movie.am.srt",
            "C:/videos/movie.amh.srt",
            "C:/videos/movie.en.srt",
            "C:/videos/movie.eng.srt",
            "C:/videos/movie.es.srt",
            "C:/videos/movie.fr.srt",
            "C:/videos/movie.2025am.srt",
            
            # Invalid or non-language suffixes that should NOT be stripped
            "C:/videos/movie.final.srt",
            "C:/videos/movie.v2.srt",
            "C:/videos/movie.edit.srt"
        ]
        media_files = [
            "C:/videos/movie.mp4",
            "C:/videos/movie.final.mp4",
            "C:/videos/movie.v2.mp4",
            "C:/videos/movie.edit.mp4"
        ]
        
        pairs = pair_srt_with_media(srt_files, media_files)
        
        # First 7 SRT files should match "movie.mp4" (language codes stripped)
        # Last 3 SRT files should match their exact counterparts
        valid_lang_pairs = [
            ("C:/videos/movie.am.srt", "C:/videos/movie.mp4"),
            ("C:/videos/movie.amh.srt", "C:/videos/movie.mp4"),
            ("C:/videos/movie.en.srt", "C:/videos/movie.mp4"),
            ("C:/videos/movie.eng.srt", "C:/videos/movie.mp4"),
            ("C:/videos/movie.es.srt", "C:/videos/movie.mp4"),
            ("C:/videos/movie.fr.srt", "C:/videos/movie.mp4"),
            ("C:/videos/movie.2025am.srt", "C:/videos/movie.mp4"),
        ]
        
        exact_match_pairs = [
            ("C:/videos/movie.final.srt", "C:/videos/movie.final.mp4"),
            ("C:/videos/movie.v2.srt", "C:/videos/movie.v2.mp4"),
            ("C:/videos/movie.edit.srt", "C:/videos/movie.edit.mp4"),
        ]
        
        # Check that all valid language suffix pairs are present
        for pair in valid_lang_pairs:
            self.assertIn(pair, pairs, f"Expected pair {pair} not found")
        
        # Check that exact match pairs are present
        for pair in exact_match_pairs:
            self.assertIn(pair, pairs, f"Expected exact match pair {pair} not found")
        
        self.assertEqual(len(pairs), 10)

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        srt_files = [
            "C:/videos/MOVIE.srt",
            "C:/videos/Tutorial.EN.srt"
        ]
        media_files = [
            "C:/videos/movie.mp4",
            "C:/videos/tutorial.mp4"
        ]
        
        pairs = pair_srt_with_media(srt_files, media_files)
        
        self.assertEqual(len(pairs), 2)
        self.assertIn(("C:/videos/MOVIE.srt", "C:/videos/movie.mp4"), pairs)
        self.assertIn(("C:/videos/Tutorial.EN.srt", "C:/videos/tutorial.mp4"), pairs)

    def test_mixed_separators(self):
        """Test pairing with different separators (., -, _) in language suffixes."""
        srt_files = [
            "C:/videos/movie.am.srt",
            "C:/videos/tutorial-en.srt",
            "C:/videos/lecture_fr.srt"
        ]
        media_files = [
            "C:/videos/movie.mp4",
            "C:/videos/tutorial.mp4",
            "C:/videos/lecture.mp4"
        ]
        
        pairs = pair_srt_with_media(srt_files, media_files)
        
        self.assertEqual(len(pairs), 3)
        self.assertIn(("C:/videos/movie.am.srt", "C:/videos/movie.mp4"), pairs)
        self.assertIn(("C:/videos/tutorial-en.srt", "C:/videos/tutorial.mp4"), pairs)
        self.assertIn(("C:/videos/lecture_fr.srt", "C:/videos/lecture.mp4"), pairs)

    def test_empty_inputs(self):
        """Test behavior with empty input lists."""
        # Empty SRT list
        pairs = pair_srt_with_media([], ["C:/videos/movie.mp4"])
        self.assertEqual(len(pairs), 0)
        
        # Empty media list
        pairs = pair_srt_with_media(["C:/videos/movie.srt"], [])
        self.assertEqual(len(pairs), 0)
        
        # Both empty
        pairs = pair_srt_with_media([], [])
        self.assertEqual(len(pairs), 0)

    def test_preference_for_exact_match(self):
        """Test that exact matches take precedence over suffix-stripped matches."""
        srt_files = [
            "C:/videos/movie.en.srt"
        ]
        media_files = [
            "C:/videos/movie.mp4",
            "C:/videos/movie.en.mp4"
        ]
        
        pairs = pair_srt_with_media(srt_files, media_files)
        
        # Should prefer exact match "movie.en" over suffix-stripped "movie"
        self.assertEqual(len(pairs), 1)
        self.assertIn(("C:/videos/movie.en.srt", "C:/videos/movie.en.mp4"), pairs)

    def test_various_language_codes(self):
        """Test various language codes from the regex pattern."""
        srt_files = [
            "C:/videos/video.ar.srt",    # Arabic
            "C:/videos/video.zh.srt",    # Chinese
            "C:/videos/video.ja.srt",    # Japanese
            "C:/videos/video.ko.srt",    # Korean
            "C:/videos/video.hi.srt",    # Hindi
            "C:/videos/video.sw.srt",    # Swahili
            "C:/videos/video.ha.srt",    # Hausa
            "C:/videos/video.yo.srt",    # Yoruba
        ]
        media_files = [
            "C:/videos/video.mp4"
        ]
        
        pairs = pair_srt_with_media(srt_files, media_files)
        
        # All should match the single video file
        self.assertEqual(len(pairs), 8)
        for srt_file in srt_files:
            self.assertIn((srt_file, "C:/videos/video.mp4"), pairs)


if __name__ == '__main__':
    unittest.main()
