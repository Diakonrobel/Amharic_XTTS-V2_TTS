"""
Dataset Creator with Gradio WebUI - Comprehensive Audio Dataset Processing

Features:
- Multiple input sources: YouTube, local audio files, microphone recording
- Automatic transcription with Faster Whisper
- Advanced audio segmentation with VAD
- Quality filtering and validation
- Real-time progress tracking
- Export to multiple formats (CSV, JSON, metadata.txt)
- Optimized for Colab/Kaggle/Local environments
"""

import os
import sys
import gradio as gr
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from dataset_processor import DatasetProcessor
from audio_recorder import AudioRecorder
from utils_dataset import (
    format_duration,
    calculate_dataset_statistics,
    validate_audio_quality,
    export_dataset
)

class DatasetCreatorApp:
    """Main application class for dataset creation with Gradio UI"""
    
    def __init__(self, output_dir="./datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processor = DatasetProcessor(output_dir=str(self.output_dir))
        self.recorder = AudioRecorder()
        self.current_project = None
        
    def create_new_project(self, project_name, language, speaker_name):
        """Create a new dataset project"""
        if not project_name:
            return "❌ Please provide a project name", None
        
        project_path = self.output_dir / project_name
        if project_path.exists():
            return f"⚠️  Project '{project_name}' already exists", None
        
        project_path.mkdir(parents=True, exist_ok=True)
        (project_path / "wavs").mkdir(exist_ok=True)
        
        # Save project metadata
        metadata = {
            "project_name": project_name,
            "language": language,
            "speaker_name": speaker_name,
            "created_at": datetime.now().isoformat(),
            "total_segments": 0,
            "total_duration": 0
        }
        
        with open(project_path / "project_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.current_project = project_path
        self.processor.set_project(str(project_path))
        
        return f"✅ Project '{project_name}' created successfully!", str(project_path)
    
    def process_youtube_video(self, url, language, speaker_name, 
                             min_duration, max_duration, quality_threshold,
                             progress=gr.Progress()):
        """Process YouTube video to create dataset"""
        if not url:
            return "❌ Please provide a YouTube URL", None, None
        
        if not self.current_project:
            return "❌ Please create a project first", None, None
        
        try:
            progress(0, desc="Downloading video...")
            result = self.processor.process_youtube_url(
                url=url,
                language=language,
                speaker_name=speaker_name,
                min_duration=min_duration,
                max_duration=max_duration,
                quality_threshold=quality_threshold,
                progress_callback=lambda p, d: progress(p, desc=d)
            )
            
            stats = calculate_dataset_statistics(self.current_project)
            stats_text = self._format_statistics(stats)
            
            return (
                f"✅ Processed successfully!\n\n{result['summary']}",
                stats_text,
                str(self.current_project / "metadata_train.csv")
            )
            
        except Exception as e:
            return f"❌ Error: {str(e)}", None, None
    
    def process_audio_files(self, files, language, speaker_name,
                           min_duration, max_duration, quality_threshold,
                           progress=gr.Progress()):
        """Process uploaded audio files"""
        if not files:
            return "❌ Please upload audio files", None, None
        
        if not self.current_project:
            return "❌ Please create a project first", None, None
        
        try:
            progress(0, desc="Processing audio files...")
            
            results = []
            for i, file in enumerate(files):
                progress((i + 1) / len(files), desc=f"Processing file {i+1}/{len(files)}...")
                
                result = self.processor.process_audio_file(
                    audio_path=file.name if hasattr(file, 'name') else file,
                    language=language,
                    speaker_name=speaker_name,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    quality_threshold=quality_threshold
                )
                results.append(result)
            
            total_segments = sum(r['segments_created'] for r in results)
            stats = calculate_dataset_statistics(self.current_project)
            stats_text = self._format_statistics(stats)
            
            summary = f"Processed {len(files)} files\n"
            summary += f"Created {total_segments} segments\n"
            summary += f"Total duration: {format_duration(sum(r['total_duration'] for r in results))}"
            
            return (
                f"✅ Processing complete!\n\n{summary}",
                stats_text,
                str(self.current_project / "metadata_train.csv")
            )
            
        except Exception as e:
            return f"❌ Error: {str(e)}", None, None
    
    def record_audio_segment(self, language, speaker_name, text=None):
        """Record audio segment from microphone"""
        if not self.current_project:
            return "❌ Please create a project first", None
        
        try:
            audio_path = self.recorder.record_segment()
            
            if text:
                # Manual transcription provided
                result = self.processor.add_manual_segment(
                    audio_path=audio_path,
                    text=text,
                    speaker_name=speaker_name
                )
            else:
                # Automatic transcription
                result = self.processor.process_audio_file(
                    audio_path=audio_path,
                    language=language,
                    speaker_name=speaker_name,
                    min_duration=0.5,
                    max_duration=15.0
                )
            
            stats = calculate_dataset_statistics(self.current_project)
            stats_text = self._format_statistics(stats)
            
            return f"✅ Recording added successfully!", stats_text
            
        except Exception as e:
            return f"❌ Error: {str(e)}", None
    
    def export_dataset_files(self, format_type, include_audio=True):
        """Export dataset in specified format"""
        if not self.current_project:
            return "❌ Please create a project first", None
        
        try:
            export_path = export_dataset(
                project_path=self.current_project,
                format_type=format_type,
                include_audio=include_audio
            )
            
            return f"✅ Dataset exported to: {export_path}", str(export_path)
            
        except Exception as e:
            return f"❌ Error: {str(e)}", None
    
    def _format_statistics(self, stats):
        """Format statistics for display"""
        text = f"""
📊 Dataset Statistics:

🎵 Audio:
  • Total segments: {stats['total_segments']}
  • Total duration: {format_duration(stats['total_duration'])}
  • Average duration: {format_duration(stats['avg_duration'])}
  • Min duration: {format_duration(stats['min_duration'])}
  • Max duration: {format_duration(stats['max_duration'])}

📝 Text:
  • Total characters: {stats['total_characters']:,}
  • Total words: {stats['total_words']:,}
  • Average words per segment: {stats['avg_words']:.1f}

🎤 Speakers:
  • Number of speakers: {stats['num_speakers']}
  • Speakers: {', '.join(stats['speakers'])}

💾 Storage:
  • Total size: {stats['total_size_mb']:.2f} MB
        """
        return text.strip()
    
    def load_existing_project(self, project_name):
        """Load an existing project"""
        project_path = self.output_dir / project_name
        
        if not project_path.exists():
            return f"❌ Project '{project_name}' not found", None
        
        self.current_project = project_path
        self.processor.set_project(str(project_path))
        
        stats = calculate_dataset_statistics(project_path)
        stats_text = self._format_statistics(stats)
        
        return f"✅ Project '{project_name}' loaded successfully!", stats_text
    
    def list_projects(self):
        """List all available projects"""
        projects = [d.name for d in self.output_dir.iterdir() if d.is_dir()]
        return projects if projects else ["No projects found"]


def create_interface():
    """Create Gradio interface"""
    app = DatasetCreatorApp()
    
    with gr.Blocks(
        title="XTTS Dataset Creator",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {max-width: 1400px !important;}
        .stat-box {background: #f0f0f0; padding: 15px; border-radius: 8px; margin: 10px 0;}
        h1 {text-align: center; margin-bottom: 1em;}
        .compact-row {gap: 0.5em !important;}
        """
    ) as demo:
        
        gr.Markdown("# 🎙️ XTTS Dataset Creator", elem_classes=["text-center"])
        gr.Markdown("### Professional Audio Dataset Creation for Voice Cloning", elem_classes=["text-center"])
        
        with gr.Tab("📁 Project Setup"):
            gr.Markdown("## Create or Load a Dataset Project")
            
            with gr.Row():
                with gr.Column(scale=2):
                    project_name_input = gr.Textbox(
                        label="Project Name",
                        placeholder="my-voice-dataset",
                        info="Unique identifier for your dataset"
                    )
                    
                    with gr.Row():
                        language_input = gr.Dropdown(
                            label="Language",
                            choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "hu", "ko", "ja", "am", "amh"],
                            value="en",
                            info="Target language for transcription"
                        )
                        
                        speaker_name_input = gr.Textbox(
                            label="Speaker Name",
                            value="speaker",
                            info="Name/ID for the voice"
                        )
                    
                    with gr.Row():
                        create_btn = gr.Button("🆕 Create New Project", variant="primary")
                        load_btn = gr.Button("📂 Load Existing Project")
                
                with gr.Column(scale=1):
                    existing_projects = gr.Dropdown(
                        label="Existing Projects",
                        choices=app.list_projects(),
                        interactive=True
                    )
                    refresh_projects_btn = gr.Button("🔄 Refresh")
            
            project_status = gr.Textbox(label="Status", interactive=False)
            current_project_path = gr.Textbox(label="Project Path", interactive=False, visible=False)
        
        with gr.Tab("🎬 YouTube Input"):
            gr.Markdown("## Process YouTube Videos")
            
            with gr.Row():
                youtube_url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    scale=3
                )
            
            with gr.Accordion("⚙️ Processing Options", open=False):
                with gr.Row():
                    yt_min_duration = gr.Slider(
                        minimum=0.5, maximum=5.0, value=1.0, step=0.1,
                        label="Minimum Segment Duration (seconds)"
                    )
                    yt_max_duration = gr.Slider(
                        minimum=5.0, maximum=30.0, value=15.0, step=0.5,
                        label="Maximum Segment Duration (seconds)"
                    )
                
                yt_quality_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                    label="Quality Threshold (0-1)",
                    info="Higher = stricter quality filtering"
                )
            
            process_yt_btn = gr.Button("🎬 Process YouTube Video", variant="primary", size="lg")
            yt_output = gr.Textbox(label="Processing Result", lines=5)
        
        with gr.Tab("📁 File Upload"):
            gr.Markdown("## Upload Audio Files")
            
            audio_files = gr.Files(
                label="Upload Audio Files",
                file_types=["audio"],
                file_count="multiple"
            )
            
            with gr.Accordion("⚙️ Processing Options", open=False):
                with gr.Row():
                    file_min_duration = gr.Slider(
                        minimum=0.5, maximum=5.0, value=1.0, step=0.1,
                        label="Minimum Segment Duration (seconds)"
                    )
                    file_max_duration = gr.Slider(
                        minimum=5.0, maximum=30.0, value=15.0, step=0.5,
                        label="Maximum Segment Duration (seconds)"
                    )
                
                file_quality_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                    label="Quality Threshold (0-1)"
                )
            
            process_files_btn = gr.Button("📁 Process Audio Files", variant="primary", size="lg")
            files_output = gr.Textbox(label="Processing Result", lines=5)
        
        with gr.Tab("🎤 Record Audio"):
            gr.Markdown("## Record Audio Segments")
            
            with gr.Row():
                audio_input = gr.Audio(
                    label="Record Audio",
                    sources=["microphone"],
                    type="filepath"
                )
            
            manual_text = gr.Textbox(
                label="Manual Transcription (optional)",
                placeholder="Enter transcription or leave empty for automatic",
                lines=2
            )
            
            record_btn = gr.Button("🎤 Add Recording to Dataset", variant="primary")
            record_output = gr.Textbox(label="Result", lines=3)
        
        with gr.Tab("📊 Dataset Overview"):
            gr.Markdown("## Current Dataset Statistics")
            
            stats_display = gr.Textbox(
                label="Statistics",
                lines=20,
                elem_classes=["stat-box"]
            )
            
            refresh_stats_btn = gr.Button("🔄 Refresh Statistics")
            
            gr.Markdown("### 📥 Export Dataset")
            
            with gr.Row():
                export_format = gr.Radio(
                    choices=["csv", "json", "metadata.txt", "ljspeech"],
                    value="csv",
                    label="Export Format"
                )
                include_audio_export = gr.Checkbox(
                    label="Include Audio Files",
                    value=True
                )
            
            export_btn = gr.Button("📥 Export Dataset", variant="primary")
            export_output = gr.Textbox(label="Export Result")
            export_file = gr.File(label="Download", visible=False)
        
        with gr.Tab("ℹ️ Help & Guide"):
            gr.Markdown("""
## 📚 Quick Start Guide

### 1️⃣ Create a Project
1. Go to **Project Setup** tab
2. Enter a project name
3. Select language and speaker name
4. Click **Create New Project**

### 2️⃣ Add Data
Choose one of three methods:

**🎬 YouTube:**
- Paste YouTube URL
- Adjust segmentation settings
- Click Process

**📁 File Upload:**
- Upload audio files (WAV, MP3, FLAC)
- Configure processing options
- Click Process

**🎤 Recording:**
- Record directly from microphone
- Optionally provide manual transcription
- Add to dataset

### 3️⃣ Review & Export
- Check statistics in **Dataset Overview**
- Export in your preferred format
- Download the complete dataset

---

## ⚙️ Processing Options

### Segment Duration
- **Minimum:** Shorter segments may lack context
- **Maximum:** Longer segments are harder to learn
- **Recommended:** 1-15 seconds

### Quality Threshold
- **Low (0.3-0.5):** Accept more segments, lower quality
- **Medium (0.6-0.7):** Balanced quality/quantity
- **High (0.8-1.0):** Strict quality, fewer segments

---

## 📊 Export Formats

- **CSV:** Pandas-compatible format
- **JSON:** Structured data format  
- **metadata.txt:** LJSpeech format
- **LJSpeech:** Complete LJSpeech-compatible dataset

---

## 💡 Best Practices

1. **Audio Quality:**
   - Use clear, noise-free audio
   - Consistent volume levels
   - Sample rate: 22050 Hz or higher

2. **Dataset Size:**
   - Minimum: 5-10 minutes (testing)
   - Recommended: 30-60 minutes (good quality)
   - Optimal: 2-4 hours (excellent quality)

3. **Speaker Consistency:**
   - Same voice throughout
   - Consistent speaking style
   - Avoid background noise

4. **Language:**
   - Use correct language code
   - Consistent language per project
   - For Amharic: use 'am' or 'amh'

---

## 🔧 Troubleshooting

**YouTube Download Fails:**
- Check URL is valid
- Video may be region-restricted
- Try updating yt-dlp

**Transcription Errors:**
- Verify correct language selected
- Check audio quality
- Ensure sufficient audio length

**Low Quality Segments:**
- Lower quality threshold
- Check source audio quality
- Adjust min/max duration

---

## 📖 Additional Resources

- [XTTS Documentation](https://github.com/coqui-ai/TTS)
- [Faster Whisper](https://github.com/systran/faster-whisper)
- [Dataset Best Practices](https://docs.coqui.ai/en/latest/tutorial_for_nervous_beginners.html)
            """)
        
        # Event handlers
        create_btn.click(
            fn=app.create_new_project,
            inputs=[project_name_input, language_input, speaker_name_input],
            outputs=[project_status, current_project_path]
        )
        
        load_btn.click(
            fn=app.load_existing_project,
            inputs=[existing_projects],
            outputs=[project_status, stats_display]
        )
        
        refresh_projects_btn.click(
            fn=app.list_projects,
            outputs=[existing_projects]
        )
        
        process_yt_btn.click(
            fn=app.process_youtube_video,
            inputs=[
                youtube_url, language_input, speaker_name_input,
                yt_min_duration, yt_max_duration, yt_quality_threshold
            ],
            outputs=[yt_output, stats_display, gr.File()]
        )
        
        process_files_btn.click(
            fn=app.process_audio_files,
            inputs=[
                audio_files, language_input, speaker_name_input,
                file_min_duration, file_max_duration, file_quality_threshold
            ],
            outputs=[files_output, stats_display, gr.File()]
        )
        
        record_btn.click(
            fn=app.record_audio_segment,
            inputs=[language_input, speaker_name_input, manual_text],
            outputs=[record_output, stats_display]
        )
        
        refresh_stats_btn.click(
            fn=lambda: app._format_statistics(
                calculate_dataset_statistics(app.current_project)
            ) if app.current_project else "No project loaded",
            outputs=[stats_display]
        )
        
        export_btn.click(
            fn=app.export_dataset_files,
            inputs=[export_format, include_audio_export],
            outputs=[export_output, export_file]
        )
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="XTTS Dataset Creator")
    parser.add_argument("--port", type=int, default=7861, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--output-dir", default="./datasets", help="Output directory")
    
    args = parser.parse_args()
    
    demo = create_interface()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True
    )
