"""
Dataset Tracker Module
Tracks processed datasets to prevent duplicate processing and provide history.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs


class DatasetTracker:
    """Track processed datasets to prevent duplicates."""
    
    def __init__(self, tracking_file: str = "dataset_history.json"):
        """
        Initialize dataset tracker.
        
        Args:
            tracking_file: Path to JSON tracking file
        """
        self.tracking_file = Path(tracking_file)
        self.history = self._load_history()
    
    def _load_history(self) -> Dict:
        """Load tracking history from JSON file."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load dataset history: {e}")
                return {"datasets": []}
        return {"datasets": []}
    
    def _save_history(self):
        """Save tracking history to JSON file."""
        try:
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save dataset history: {e}")
    
    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """
        Extract YouTube video ID from URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID or None
        """
        try:
            parsed = urlparse(url)
            
            # Handle different YouTube URL formats
            if 'youtube.com' in parsed.netloc:
                if parsed.path == '/watch':
                    query = parse_qs(parsed.query)
                    return query.get('v', [None])[0]
                elif parsed.path.startswith('/embed/'):
                    return parsed.path.split('/')[2]
                elif parsed.path.startswith('/v/'):
                    return parsed.path.split('/')[2]
            elif 'youtu.be' in parsed.netloc:
                return parsed.path[1:]  # Remove leading slash
            
            return None
        except:
            return None
    
    def _compute_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """
        Compute SHA256 hash of file for tracking.
        Only hashes first and last chunks for speed on large files.
        
        Args:
            file_path: Path to file
            chunk_size: Size of chunks to read
            
        Returns:
            SHA256 hash string
        """
        try:
            hasher = hashlib.sha256()
            file_size = os.path.getsize(file_path)
            
            with open(file_path, 'rb') as f:
                # Hash first chunk
                chunk = f.read(chunk_size)
                hasher.update(chunk)
                
                # Hash last chunk if file is large
                if file_size > chunk_size * 2:
                    f.seek(-chunk_size, 2)  # Seek to last chunk
                    chunk = f.read(chunk_size)
                    hasher.update(chunk)
                elif file_size > chunk_size:
                    # Hash remaining data
                    chunk = f.read()
                    hasher.update(chunk)
            
            return hasher.hexdigest()[:16]  # Use first 16 chars
        except Exception as e:
            print(f"Warning: Could not compute file hash: {e}")
            return ""
    
    def is_youtube_processed(
        self, 
        url: str, 
        language: str
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if YouTube video has been processed.
        
        Args:
            url: YouTube URL
            language: Language code
            
        Returns:
            Tuple of (is_processed, dataset_info)
        """
        video_id = self._extract_youtube_id(url)
        if not video_id:
            return False, None
        
        for dataset in self.history.get("datasets", []):
            if (dataset.get("type") == "youtube" and 
                dataset.get("video_id") == video_id and
                dataset.get("language") == language):
                return True, dataset
        
        return False, None
    
    def is_file_processed(
        self, 
        file_path: str,
        file_type: str = "srt"
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if file has been processed.
        
        Args:
            file_path: Path to file
            file_type: Type of file (srt, audio, etc.)
            
        Returns:
            Tuple of (is_processed, dataset_info)
        """
        file_name = Path(file_path).name
        file_hash = self._compute_file_hash(file_path)
        
        for dataset in self.history.get("datasets", []):
            if (dataset.get("type") == file_type and
                (dataset.get("file_name") == file_name or
                 dataset.get("file_hash") == file_hash)):
                return True, dataset
        
        return False, None
    
    def add_youtube_dataset(
        self,
        url: str,
        video_id: str,
        title: str,
        language: str,
        duration: float,
        num_segments: int,
        output_path: str
    ):
        """
        Add YouTube dataset to history.
        
        Args:
            url: YouTube URL
            video_id: YouTube video ID
            title: Video title
            language: Language code
            duration: Video duration in seconds
            num_segments: Number of segments created
            output_path: Path to dataset
        """
        dataset_entry = {
            "type": "youtube",
            "video_id": video_id,
            "url": url,
            "title": title,
            "language": language,
            "duration": duration,
            "num_segments": num_segments,
            "output_path": output_path,
            "processed_at": datetime.now().isoformat(),
            "timestamp": datetime.now().timestamp()
        }
        
        self.history.setdefault("datasets", []).append(dataset_entry)
        self._save_history()
        print(f"✓ Added YouTube dataset to history: {title}")
    
    def add_file_dataset(
        self,
        file_path: str,
        file_type: str,
        language: str,
        num_segments: int,
        output_path: str,
        media_file: Optional[str] = None
    ):
        """
        Add file-based dataset to history.
        
        Args:
            file_path: Path to source file
            file_type: Type of file (srt, audio, etc.)
            language: Language code
            num_segments: Number of segments created
            output_path: Path to dataset
            media_file: Optional path to media file (for SRT processing)
        """
        file_name = Path(file_path).name
        file_hash = self._compute_file_hash(file_path)
        
        dataset_entry = {
            "type": file_type,
            "file_name": file_name,
            "file_path": str(file_path),
            "file_hash": file_hash,
            "language": language,
            "num_segments": num_segments,
            "output_path": output_path,
            "processed_at": datetime.now().isoformat(),
            "timestamp": datetime.now().timestamp()
        }
        
        if media_file:
            dataset_entry["media_file"] = str(media_file)
            dataset_entry["media_hash"] = self._compute_file_hash(media_file)
        
        self.history.setdefault("datasets", []).append(dataset_entry)
        self._save_history()
        print(f"✓ Added {file_type} dataset to history: {file_name}")
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get processing history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of dataset entries, sorted by most recent
        """
        datasets = self.history.get("datasets", [])
        # Sort by timestamp (most recent first)
        datasets_sorted = sorted(
            datasets, 
            key=lambda x: x.get("timestamp", 0), 
            reverse=True
        )
        
        if limit:
            return datasets_sorted[:limit]
        return datasets_sorted
    
    def remove_dataset(self, index: int) -> bool:
        """
        Remove dataset from history by index.
        
        Args:
            index: Index of dataset to remove
            
        Returns:
            True if successful
        """
        try:
            datasets = self.history.get("datasets", [])
            if 0 <= index < len(datasets):
                removed = datasets.pop(index)
                self._save_history()
                print(f"✓ Removed dataset from history: {removed.get('title', removed.get('file_name', 'Unknown'))}")
                return True
            return False
        except Exception as e:
            print(f"Error removing dataset: {e}")
            return False
    
    def clear_history(self):
        """Clear all history."""
        self.history = {"datasets": []}
        self._save_history()
        print("✓ Cleared all dataset history")
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about processed datasets.
        
        Returns:
            Dictionary with statistics
        """
        datasets = self.history.get("datasets", [])
        
        stats = {
            "total_datasets": len(datasets),
            "youtube_datasets": len([d for d in datasets if d.get("type") == "youtube"]),
            "srt_datasets": len([d for d in datasets if d.get("type") == "srt"]),
            "audio_datasets": len([d for d in datasets if d.get("type") == "audio"]),
            "total_segments": sum(d.get("num_segments", 0) for d in datasets),
            "languages": list(set(d.get("language", "unknown") for d in datasets)),
            "total_duration": sum(d.get("duration", 0) for d in datasets if d.get("duration"))
        }
        
        return stats
    
    def format_history_display(self, limit: int = 10) -> str:
        """
        Format history for display in UI.
        
        Args:
            limit: Maximum number of entries to show
            
        Returns:
            Formatted string
        """
        datasets = self.get_history(limit)
        
        if not datasets:
            return "No datasets processed yet."
        
        lines = ["📊 Dataset Processing History", "=" * 50]
        
        for i, dataset in enumerate(datasets, 1):
            dataset_type = dataset.get("type", "unknown")
            date = dataset.get("processed_at", "unknown")[:19].replace("T", " ")
            
            if dataset_type == "youtube":
                title = dataset.get("title", "Unknown")
                lang = dataset.get("language", "?")
                segments = dataset.get("num_segments", 0)
                duration = dataset.get("duration", 0)
                lines.append(f"\n{i}. 📹 YouTube: {title}")
                lines.append(f"   Language: {lang} | Segments: {segments} | Duration: {duration:.0f}s")
                lines.append(f"   Processed: {date}")
            else:
                file_name = dataset.get("file_name", "Unknown")
                lang = dataset.get("language", "?")
                segments = dataset.get("num_segments", 0)
                lines.append(f"\n{i}. 📄 {dataset_type.upper()}: {file_name}")
                lines.append(f"   Language: {lang} | Segments: {segments}")
                lines.append(f"   Processed: {date}")
        
        # Add statistics
        stats = self.get_statistics()
        lines.append("\n" + "=" * 50)
        lines.append(f"Total Datasets: {stats['total_datasets']} | Total Segments: {stats['total_segments']}")
        lines.append(f"Languages: {', '.join(stats['languages'])}")
        
        return "\n".join(lines)


# Global tracker instance
_tracker = None

def get_tracker(tracking_file: str = "dataset_history.json") -> DatasetTracker:
    """
    Get global dataset tracker instance.
    
    Args:
        tracking_file: Path to tracking file
        
    Returns:
        DatasetTracker instance
    """
    global _tracker
    if _tracker is None:
        _tracker = DatasetTracker(tracking_file)
    return _tracker
