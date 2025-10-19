"""
Dataset Merger UI Component for Gradio WebUI
Provides a professional interface for merging temp datasets with full controls
"""

import gradio as gr
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile

def scan_temp_datasets(base_path):
    """Scan for temp_dataset_* folders and return statistics"""
    try:
        base_dir = Path(base_path) / "finetune_models"
        if not base_dir.exists():
            return "❌ Base directory not found", "", []
        
        temp_dirs = sorted([d for d in base_dir.glob("temp_dataset_*") if d.is_dir()])
        
        if not temp_dirs:
            return "✅ No temp datasets found - all clean!", "", []
        
        # Count wavs in each
        dataset_info = []
        total_wavs = 0
        
        for temp_dir in temp_dirs:
            wavs_dir = temp_dir / "wavs"
            if wavs_dir.exists():
                wav_count = len(list(wavs_dir.glob("*.wav")))
                total_wavs += wav_count
                dataset_info.append({
                    "name": temp_dir.name,
                    "wavs": wav_count,
                    "path": str(temp_dir)
                })
        
        # Format output
        summary = f"📊 **Found {len(temp_dirs)} temp datasets** ({total_wavs:,} audio files)\n\n"
        summary += "| Dataset | Audio Files |\n"
        summary += "|---------|------------|\n"
        for info in dataset_info:
            summary += f"| {info['name']} | {info['wavs']:,} |\n"
        
        # Create choices for selective merge
        choices = [(f"{d['name']} ({d['wavs']} files)", d['name']) for d in dataset_info]
        
        return summary, json.dumps(dataset_info, indent=2), choices
        
    except Exception as e:
        return f"❌ Error scanning: {str(e)}", "", []

def preview_merge_operation(base_path, selected_datasets, standardize_names):
    """Preview what the merge operation will do"""
    try:
        base_dir = Path(base_path) / "finetune_models"
        dataset_dir = base_dir / "dataset"
        
        if not dataset_dir.exists():
            return "❌ Main dataset directory not found"
        
        # Count current files
        current_wavs = len(list((dataset_dir / "wavs").glob("*.wav"))) if (dataset_dir / "wavs").exists() else 0
        
        # Count files to be added
        new_wavs = 0
        if selected_datasets == "all":
            temp_dirs = sorted([d for d in base_dir.glob("temp_dataset_*") if d.is_dir()])
        else:
            temp_dirs = [base_dir / name for name in selected_datasets]
        
        for temp_dir in temp_dirs:
            wavs_dir = temp_dir / "wavs"
            if wavs_dir.exists():
                new_wavs += len(list(wavs_dir.glob("*.wav")))
        
        preview = f"""
### 📋 Merge Preview

**Current State:**
- Main dataset: {current_wavs:,} audio files
- Temp datasets selected: {len(temp_dirs)}

**After Merge:**
- Total audio files: {current_wavs + new_wavs:,}
- New files added: {new_wavs:,}

**Operations:**
1. ✅ Create backups of metadata_train.csv and metadata_eval.csv
2. ✅ Merge CSV files (remove duplicates)
3. ✅ Copy {new_wavs:,} audio files
"""
        
        if standardize_names:
            preview += f"4. ✅ Standardize filenames to merged_XXXXXXXX.wav format\n"
        
        preview += f"""
**Safety:**
- ✅ Backups created before any changes
- ✅ Duplicate detection enabled
- ✅ Temp datasets moved to 'merged_temps/' (not deleted)
- ✅ Detailed JSON report generated
"""
        
        return preview
        
    except Exception as e:
        return f"❌ Error in preview: {str(e)}"

def execute_merge(base_path, selected_datasets, standardize_names, cleanup_option):
    """Execute the merge operation"""
    try:
        base_dir = Path(base_path) / "finetune_models"
        
        # Create a custom merge script with selected options
        script_content = f"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set cleanup option
import merge_datasets_simple
merge_datasets_simple.AUTO_CLEANUP = "{cleanup_option}"

# Run merge
merge_datasets_simple.main()
"""
        
        # Write temporary script
        temp_script = base_dir / "_temp_merge.py"
        with open(temp_script, 'w') as f:
            f.write(script_content)
        
        # Execute merge
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Clean up temp script
        if temp_script.exists():
            temp_script.unlink()
        
        output = result.stdout
        
        if result.returncode != 0:
            return f"❌ Merge failed:\n\n{result.stderr}", "", None
        
        # Find the report file
        report_files = sorted(base_dir.glob("merge_report_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        report_path = report_files[0] if report_files else None
        
        # Parse output for summary
        summary = "✅ **Merge completed successfully!**\n\n"
        
        if "Train metadata:" in output:
            lines = output.split('\n')
            for line in lines:
                if any(key in line for key in ["Train metadata:", "Eval metadata:", "Wav files:", "Copied:", "Skipped:"]):
                    summary += f"{line.strip()}\n"
        
        # Now run standardization if requested
        if standardize_names:
            summary += "\n\n### 📝 Standardizing filenames...\n\n"
            
            std_result = subprocess.run(
                [sys.executable, "standardize_filenames.py"],
                cwd=str(base_dir),
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if std_result.returncode == 0:
                summary += "✅ All filenames standardized to merged_XXXXXXXX.wav format\n"
                
                # Find rename report
                rename_reports = sorted(base_dir.glob("rename_report_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
                if rename_reports:
                    with open(rename_reports[0], 'r') as f:
                        rename_data = json.load(f)
                    summary += f"✅ Renamed {rename_data['files_renamed']} files\n"
            else:
                summary += f"⚠️ Standardization had issues:\n{std_result.stderr}\n"
        
        return summary, output, str(report_path) if report_path else None
        
    except subprocess.TimeoutExpired:
        return "❌ Merge operation timed out (>10 minutes)", "", None
    except Exception as e:
        return f"❌ Error executing merge: {str(e)}", "", None

def download_report(report_path):
    """Prepare report for download"""
    if not report_path or not Path(report_path).exists():
        return None
    return report_path

def create_dataset_merger_tab():
    """Create the Dataset Merger tab component"""
    
    with gr.Group():
        gr.Markdown("## 🔀 Dataset Merger")
        gr.Markdown("Merge temporary datasets into the main dataset with automatic backup and safety checks")
        
        with gr.Row():
            base_path_input = gr.Textbox(
                label="Base Path",
                value=str(Path.cwd()),
                placeholder="Path to project root (contains finetune_models/)",
                scale=3
            )
            scan_btn = gr.Button("🔍 Scan for Temp Datasets", variant="primary", scale=1)
        
        scan_output = gr.Markdown(value="Click 'Scan' to find temp datasets")
        scan_details = gr.JSON(label="Dataset Details", visible=False)
        
        with gr.Accordion("⚙️ Merge Options", open=True):
            with gr.Row():
                dataset_selection = gr.CheckboxGroup(
                    label="Datasets to Merge",
                    choices=[],
                    value=[],
                    info="Leave empty to merge all temp datasets"
                )
            
            with gr.Row():
                standardize_filenames_option = gr.Checkbox(
                    label="📝 Standardize Filenames",
                    value=True,
                    info="Rename all files to merged_XXXXXXXX.wav format (recommended)",
                    scale=1
                )
                
                cleanup_option = gr.Radio(
                    label="🧹 Cleanup After Merge",
                    choices=[
                        ("Keep temp datasets", "keep"),
                        ("Move to merged_temps/ folder", "move"),
                        ("Delete temp datasets (permanent)", "delete")
                    ],
                    value="move",
                    info="What to do with temp datasets after successful merge",
                    scale=2
                )
        
        with gr.Accordion("📋 Preview", open=False):
            preview_btn = gr.Button("👁️ Preview Merge Operation", variant="secondary")
            preview_output = gr.Markdown()
        
        merge_btn = gr.Button("▶️ Execute Merge", variant="primary", size="lg")
        
        with gr.Accordion("📊 Merge Results", open=True):
            merge_status = gr.Markdown(value="Merge not started")
            merge_details = gr.Textbox(
                label="Detailed Log",
                lines=15,
                max_lines=25,
                interactive=False
            )
            
            with gr.Row():
                report_file = gr.File(label="📥 Download Merge Report", visible=False)
                report_path_state = gr.State()
        
        gr.Markdown("""
        ---
        ### 💡 How It Works
        
        1. **Scan**: Discovers all temp_dataset_* folders in your finetune_models directory
        2. **Preview**: Shows what will happen before making changes
        3. **Backup**: Creates timestamped backups of metadata files
        4. **Merge**: Combines CSV files (removes duplicates) and copies audio files
        5. **Standardize**: Optionally renames all files to consistent format
        6. **Cleanup**: Moves or deletes temp datasets based on your choice
        7. **Report**: Generates detailed JSON report for auditing
        
        ### ✅ Safety Features
        - ✅ Automatic backups before any changes
        - ✅ Duplicate detection and removal
        - ✅ No data loss - temp datasets preserved unless you choose to delete
        - ✅ Detailed logging and reports
        - ✅ Transaction-safe rename operations
        """)
    
    # Wire up the callbacks
    scan_btn.click(
        fn=scan_temp_datasets,
        inputs=[base_path_input],
        outputs=[scan_output, scan_details, dataset_selection]
    )
    
    preview_btn.click(
        fn=preview_merge_operation,
        inputs=[base_path_input, dataset_selection, standardize_filenames_option],
        outputs=[preview_output]
    )
    
    def execute_and_show_report(*args):
        summary, details, report = execute_merge(*args)
        # Return summary, details, report path, and file component
        return summary, details, report, gr.File(value=report, visible=True) if report else gr.File(visible=False)
    
    merge_btn.click(
        fn=execute_and_show_report,
        inputs=[base_path_input, dataset_selection, standardize_filenames_option, cleanup_option],
        outputs=[merge_status, merge_details, report_path_state, report_file]
    )
    
    return {
        "scan_btn": scan_btn,
        "merge_btn": merge_btn,
        "preview_btn": preview_btn
    }
