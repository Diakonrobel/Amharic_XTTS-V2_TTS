import argparse
import os
import sys
import tempfile
from pathlib import Path

import shutil
import glob

import gradio as gr
import librosa.display
import numpy as np

import torch
import torchaudio
import traceback
from utils.formatter import format_audio_list,find_latest_best_model, list_audios
from utils.gpt_train import train_gpt
from utils import srt_processor
from utils import youtube_downloader, srt_processor, audio_slicer, dataset_tracker, batch_processor
from utils import audio_slicer

from faster_whisper import WhisperModel

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import requests

def download_file(url, destination):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file to {destination}")
        return destination
    except Exception as e:
        print(f"Failed to download the file: {e}")
        return None

# Clear logs
def remove_log_file(file_path):
     log_file = Path(file_path)

     if log_file.exists() and log_file.is_file():
         log_file.unlink()

# remove_log_file(str(Path.cwd() / "log.out"))

def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

XTTS_MODEL = None

def create_zip(folder_path, zip_name):
    zip_path = os.path.join(tempfile.gettempdir(), f"{zip_name}.zip")
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)
    return zip_path

def get_model_zip(out_path):
    ready_folder = os.path.join(out_path, "ready")
    if os.path.exists(ready_folder):
        return create_zip(ready_folder, "optimized_model")
    return None

def get_dataset_zip(out_path):
    dataset_folder = os.path.join(out_path, "dataset")
    if os.path.exists(dataset_folder):
        return create_zip(dataset_folder, "dataset")
    return None

def load_model(xtts_checkpoint, xtts_config, xtts_vocab,xtts_speaker):
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to run the previous steps or manually set the `XTTS checkpoint path`, `XTTS config path`, and `XTTS vocab path` fields !!"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model! ")
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab,speaker_file_path=xtts_speaker, use_deepspeed=False)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("Model Loaded!")
    return "Model Loaded!"

def run_tts(lang, tts_text, speaker_audio_file, temperature, length_penalty,repetition_penalty,top_k,top_p,sentence_split,use_config):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model !!", None, None

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(audio_path=speaker_audio_file, gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, max_ref_length=XTTS_MODEL.config.max_ref_len, sound_norm_refs=XTTS_MODEL.config.sound_norm_refs)
    
    if use_config:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=XTTS_MODEL.config.temperature, # Add custom parameters here
            length_penalty=XTTS_MODEL.config.length_penalty,
            repetition_penalty=XTTS_MODEL.config.repetition_penalty,
            top_k=XTTS_MODEL.config.top_k,
            top_p=XTTS_MODEL.config.top_p,
            enable_text_splitting = True
        )
    else:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature, # Add custom parameters here
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting = sentence_split
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return "Speech generated !", out_path, speaker_audio_file


def load_params_tts(out_path,version):
    
    out_path = Path(out_path)

    # base_model_path = Path.cwd() / "models" / version 

    # if not base_model_path.exists():
    #     return "Base model not found !","","",""

    ready_model_path = out_path / "ready" 

    vocab_path =  ready_model_path / "vocab.json"
    config_path = ready_model_path / "config.json"
    speaker_path =  ready_model_path / "speakers_xtts.pth"
    reference_path  = ready_model_path / "reference.wav"

    model_path = ready_model_path / "model.pth"

    if not model_path.exists():
        model_path = ready_model_path / "unoptimize_model.pth"
        if not model_path.exists():
          return "Params for TTS not found", "", "", ""         

    return "Params for TTS loaded", model_path, config_path, vocab_path,speaker_path, reference_path
     

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""XTTS fine-tuning demo\n\n"""
        """
        Example runs:
        python3 TTS/demos/xtts_ft_demo/xtts_demo.py --port 
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        help="Name of the whisper model selected by default (Optional) Choices are: ['large-v3','large-v2', 'large', 'medium', 'small']   Default Value: 'large-v3'",
        default="large-v3",
    )
    parser.add_argument(
        "--audio_folder_path",
        type=str,
        help="Path to the folder with audio files (optional)",
        default="",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Enable sharing of the Gradio interface via public link.",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to run the gradio demo. Default: 5003",
        default=5003,
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="Output path (where data and checkpoints will be saved) Default: output/",
        default=str(Path.cwd() / "finetune_models"),
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train. Default: 6",
        default=6,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size. Default: 2",
        default=2,
    )
    parser.add_argument(
        "--grad_acumm",
        type=int,
        help="Grad accumulation steps. Default: 1",
        default=1,
    )
    parser.add_argument(
        "--max_audio_length",
        type=int,
        help="Max permitted audio size in seconds. Default: 11",
        default=11,
    )

    args = parser.parse_args()

    with gr.Blocks(title=os.environ.get("APP_NAME", "Gradio")) as demo:
        with gr.Tab("1 - Data processing"):
            out_path = gr.Textbox(
                label="Output path (where data and checkpoints will be saved):",
                value=args.out_path,
            )
            # upload_file = gr.Audio(
            #     sources="upload",
            #     label="Select here the audio files that you want to use for XTTS trainining !",
            #     type="filepath",
            # )
            upload_file = gr.File(
                file_count="multiple",
                label="Select here the audio files that you want to use for XTTS trainining (Supported formats: wav, mp3, and flac)",
            )
            
            audio_folder_path = gr.Textbox(
                label="Path to the folder with audio files (optional):",
                value=args.audio_folder_path,
            )
            
            # Advanced Dataset Processing Options
            gr.Markdown("---")
            gr.Markdown("### 🎬 Advanced Dataset Processing Options")
            gr.Markdown("Process SRT subtitles, YouTube videos, or use intelligent audio slicing")
            
            with gr.Accordion("📝 SRT + Media File Processing", open=False) as srt_accordion:
                gr.Markdown(
                    "Upload subtitle files (SRT/VTT) with corresponding audio/video files for precise timestamp-based dataset creation."
                )
                srt_file = gr.File(
                    file_count="single",
                    label="SRT/VTT Subtitle File",
                    file_types=[".srt", ".vtt"],
                )
                media_file = gr.File(
                    file_count="single",
                    label="Media File (Audio or Video)",
                    file_types=[".mp4", ".mkv", ".avi", ".wav", ".mp3", ".flac"],
                )
                process_srt_btn = gr.Button(value="Process SRT + Media", variant="secondary")
                srt_status = gr.Textbox(label="SRT Processing Status", interactive=False)
            
            with gr.Accordion("📹 YouTube Video Download", open=False) as youtube_accordion:
                gr.Markdown(
                    "Download YouTube videos and extract available transcripts/subtitles automatically."
                )
                youtube_url = gr.Textbox(
                    label="YouTube URL(s)",
                    placeholder="Single URL or multiple URLs (comma/newline separated)\nExample: https://youtube.com/watch?v=VIDEO1, https://youtube.com/watch?v=VIDEO2",
                    lines=3,  # Multi-line support
                    max_lines=10
                )
                youtube_transcript_lang = gr.Dropdown(
                    label="Preferred Transcript Language",
                    value="en",
                    choices=[
                        # Major languages
                        ("English", "en"),
                        ("Spanish", "es"),
                        ("French", "fr"),
                        ("German", "de"),
                        ("Italian", "it"),
                        ("Portuguese", "pt"),
                        ("Russian", "ru"),
                        ("Chinese", "zh"),
                        ("Japanese", "ja"),
                        ("Korean", "ko"),
                        ("Arabic", "ar"),
                        # Ethiopian languages
                        ("Amharic (አማርኛ)", "am"),
                        ("Oromo (Oromoo)", "om"),
                        ("Tigrinya (ትግርኛ)", "ti"),
                        ("Somali (Soomaali)", "so"),
                        ("Afar", "aa"),
                        # Other African languages
                        ("Swahili", "sw"),
                        ("Hausa", "ha"),
                        ("Yoruba", "yo"),
                        ("Zulu", "zu"),
                        ("Xhosa", "xh"),
                        # Asian languages
                        ("Hindi", "hi"),
                        ("Bengali", "bn"),
                        ("Vietnamese", "vi"),
                        ("Thai", "th"),
                        ("Indonesian", "id"),
                        ("Filipino", "fil"),
                        ("Urdu", "ur"),
                        ("Persian", "fa"),
                        # European languages
                        ("Polish", "pl"),
                        ("Turkish", "tr"),
                        ("Dutch", "nl"),
                        ("Czech", "cs"),
                        ("Hungarian", "hu"),
                        ("Ukrainian", "uk"),
                        ("Greek", "el"),
                        ("Hebrew", "he"),
                        ("Romanian", "ro"),
                        ("Swedish", "sv"),
                        ("Norwegian", "no"),
                        ("Danish", "da"),
                        ("Finnish", "fi"),
                    ],
                    info="Language for transcript/subtitle extraction (auto-fallback to English if unavailable)",
                    allow_custom_value=True
                )
                
                youtube_batch_mode = gr.Checkbox(
                    label="🎬 Batch Mode (Process multiple URLs as single dataset)",
                    value=False,
                    info="Enable to process multiple URLs. Videos will be merged into one unified dataset."
                )
                
                download_youtube_btn = gr.Button(value="Download & Process YouTube", variant="secondary")
                youtube_status = gr.Textbox(label="YouTube Processing Status", interactive=False, lines=10)
            
            with gr.Accordion("✂️ RMS-Based Audio Slicing", open=False) as slicer_accordion:
                gr.Markdown(
                    "Intelligently segment long audio files based on silence detection without manual editing."
                )
                slicer_audio_file = gr.File(
                    file_count="single",
                    label="Audio File to Slice",
                    file_types=[".wav", ".mp3", ".flac"],
                )
                with gr.Row():
                    slicer_threshold_db = gr.Slider(
                        label="Silence Threshold (dB)",
                        minimum=-60,
                        maximum=-10,
                        step=1,
                        value=-40,
                        info="Volume threshold for silence detection"
                    )
                    slicer_min_length = gr.Slider(
                        label="Min Segment Length (seconds)",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.5,
                        value=5.0,
                        info="Minimum duration of each segment"
                    )
                with gr.Row():
                    slicer_min_interval = gr.Slider(
                        label="Min Silence Interval (seconds)",
                        minimum=0.1,
                        maximum=2.0,
                        step=0.1,
                        value=0.3,
                        info="Minimum silence duration to split"
                    )
                    slicer_max_sil_kept = gr.Slider(
                        label="Silence Padding (seconds)",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=0.5,
                        info="Silence kept at segment boundaries"
                    )
                slicer_auto_transcribe = gr.Checkbox(
                    label="Auto-transcribe sliced segments with Whisper",
                    value=True,
                    info="Automatically generate transcriptions for sliced audio"
                )
                slice_audio_btn = gr.Button(value="Slice Audio", variant="secondary")
                slicer_status = gr.Textbox(label="Audio Slicing Status", interactive=False)
            
            gr.Markdown("---")

            whisper_model = gr.Dropdown(
                label="Whisper Model",
                value=args.whisper_model,
                choices=[
                    "large-v3",
                    "large-v2",
                    "large",
                    "medium",
                    "small"
                ],
            )

            lang = gr.Dropdown(
                label="Dataset Language",
                value="en",
                choices=[
                    "en",
                    "es",
                    "fr",
                    "de",
                    "it",
                    "pt",
                    "pl",
                    "tr",
                    "ru",
                    "nl",
                    "cs",
                    "ar",
                    "zh",
                    "hu",
                    "ko",
                    "ja",
                    "amh"  # Amharic
                ],
            )
            
            # Amharic G2P Options for Dataset Preprocessing
            with gr.Accordion("Amharic G2P Options (for 'amh' language)", open=False) as amh_g2p_accordion:
                use_amharic_g2p_preprocessing = gr.Checkbox(
                    label="Enable Amharic G2P preprocessing for dataset",
                    value=False,
                    info="Convert Amharic text to phonemes during dataset preparation"
                )
                g2p_backend_selection = gr.Dropdown(
                    label="G2P Backend",
                    value="transphone",
                    choices=["transphone", "epitran", "rule_based"],
                    info="Primary G2P backend (will auto-fallback if unavailable)"
                )
            progress_data = gr.Label(
                label="Progress:"
            )
            # demo.load(read_logs, None, logs, every=1)

            prompt_compute_btn = gr.Button(value="Step 1 - Create dataset")
        
            # Advanced processing functions
            def process_srt_media(srt_file_path, media_file_path, language, out_path, progress=gr.Progress(track_tqdm=True)):
                """Process SRT subtitle file with corresponding media file"""
                try:
                    if not srt_file_path or not media_file_path:
                        return "Please upload both SRT file and media file!"
                    
                    # Check if already processed
                    tracker = dataset_tracker.get_tracker(os.path.join(out_path, "dataset_history.json"))
                    is_processed, existing_dataset = tracker.is_file_processed(srt_file_path, "srt")
                    
                    if is_processed:
                        date = existing_dataset.get("processed_at", "unknown")[:19].replace("T", " ")
                        return f"⚠ SRT File Already Processed!\n\nThis file was already processed:\n" \
                               f"File: {existing_dataset.get('file_name', 'Unknown')}\n" \
                               f"Language: {existing_dataset.get('language', '?')}\n" \
                               f"Segments: {existing_dataset.get('num_segments', 0)}\n" \
                               f"Processed: {date}\n\n" \
                               f"ℹ If you want to reprocess, use a different output directory."
                    
                    progress(0, desc="Initializing SRT processor...")
                    output_path = os.path.join(out_path, "dataset")
                    os.makedirs(output_path, exist_ok=True)
                    
                    progress(0.3, desc="Processing SRT and extracting audio segments...")
                    train_csv, eval_csv, duration = srt_processor.process_srt_with_media(
                        srt_path=srt_file_path,
                        media_path=media_file_path,
                        output_dir=output_path,
                        language=language,
                        gradio_progress=progress
                    )
                    
                    # Count segments from train CSV
                    import pandas as pd
                    train_df = pd.read_csv(train_csv, sep='|')
                    eval_df = pd.read_csv(eval_csv, sep='|')
                    num_segments = len(train_df) + len(eval_df)
                    
                    # Track this dataset
                    tracker.add_file_dataset(
                        file_path=srt_file_path,
                        file_type="srt",
                        language=language,
                        num_segments=num_segments,
                        output_path=output_path,
                        media_file=media_file_path
                    )
                    
                    progress(1.0, desc="SRT processing complete!")
                    return f"✓ SRT Processing Complete!\nProcessed {num_segments} segments\nTotal audio: {duration:.1f}s\nDataset created at: {output_path}\n\nℹ This dataset has been saved to history and won't be reprocessed."
                    
                except Exception as e:
                    traceback.print_exc()
                    return f"❌ Error processing SRT: {str(e)}"
            
            def process_youtube_batch_urls(urls, transcript_lang, out_path, progress):
                """Process multiple YouTube URLs in batch mode"""
                try:
                    progress(0, desc=f"Initializing batch processing for {len(urls)} videos...")
                    
                    # Process all videos
                    train_csv, eval_csv, video_infos = batch_processor.process_youtube_batch(
                        urls=urls,
                        transcript_lang=transcript_lang,
                        out_path=out_path,
                        youtube_downloader=youtube_downloader,
                        srt_processor=srt_processor,
                        progress_callback=lambda p, desc: progress(p, desc=desc)
                    )
                    
                    # Count total segments
                    import pandas as pd
                    train_df = pd.read_csv(train_csv, sep='|')
                    eval_df = pd.read_csv(eval_csv, sep='|')
                    total_segments = len(train_df) + len(eval_df)
                    
                    # Track batch as single entry
                    tracker = dataset_tracker.get_tracker(os.path.join(out_path, "dataset_history.json"))
                    
                    # Add batch entry (using first video info as representative)
                    if video_infos:
                        first_video = video_infos[0]
                        video_id = tracker._extract_youtube_id(first_video['url'])
                        if video_id:
                            tracker.add_youtube_dataset(
                                url=f"BATCH: {len(video_infos)} videos",
                                video_id=f"batch_{video_id}",
                                title=f"Batch: {', '.join([v['title'][:30] for v in video_infos[:3]])}{'...' if len(video_infos) > 3 else ''}",
                                language=transcript_lang,
                                duration=sum(v['duration'] for v in video_infos),
                                num_segments=total_segments,
                                output_path=os.path.join(out_path, "dataset")
                            )
                    
                    # Format summary
                    summary = batch_processor.format_batch_summary(video_infos, total_segments)
                    summary += "\n\nℹ This batch dataset has been saved to history."
                    
                    progress(1.0, desc="Batch processing complete!")
                    return summary
                    
                except Exception as e:
                    traceback.print_exc()
                    return f"❌ Error in batch processing: {str(e)}"
            
            def download_youtube_video(url, transcript_lang, language, out_path, batch_mode, progress=gr.Progress(track_tqdm=True)):
                """Download YouTube video(s) and extract transcripts"""
                try:
                    if not url:
                        return "Please enter a YouTube URL!"
                    
                    # Parse URLs
                    urls = batch_processor.parse_youtube_urls(url)
                    
                    if not urls:
                        return "❌ No valid YouTube URLs found. Please check your input."
                    
                    # Check if batch mode and multiple URLs
                    if batch_mode and len(urls) > 1:
                        return process_youtube_batch_urls(urls, transcript_lang, out_path, progress)
                    
                    # Single URL processing (existing logic)
                    url = urls[0]  # Use first URL
                    
                    # Check if already processed
                    tracker = dataset_tracker.get_tracker(os.path.join(out_path, "dataset_history.json"))
                    is_processed, existing_dataset = tracker.is_youtube_processed(url, transcript_lang)
                    
                    if is_processed:
                        date = existing_dataset.get("processed_at", "unknown")[:19].replace("T", " ")
                        return f"⚠ Video Already Processed!\n\nThis video was already downloaded and processed:\n" \
                               f"Title: {existing_dataset.get('title', 'Unknown')}\n" \
                               f"Language: {existing_dataset.get('language', '?')}\n" \
                               f"Segments: {existing_dataset.get('num_segments', 0)}\n" \
                               f"Processed: {date}\n" \
                               f"Output: {existing_dataset.get('output_path', 'Unknown')}\n\n" \
                               f"ℹ If you want to reprocess, please delete the dataset first or use a different output directory."
                    
                    progress(0, desc="Initializing YouTube downloader...")
                    temp_dir = tempfile.mkdtemp()
                    
                    progress(0.2, desc="Downloading video and subtitles...")
                    audio_path, srt_path, info = youtube_downloader.download_youtube_video(
                        url=url,
                        output_dir=temp_dir,
                        language=transcript_lang,
                        audio_only=True,
                        download_subtitles=True,
                        auto_update=True
                    )
                    
                    if not audio_path:
                        return "❌ Failed to download YouTube video. Check URL and try again."
                    
                    if not srt_path:
                        return "❌ No transcripts/subtitles available for this video. Try a different video or language."
                    
                    progress(0.6, desc="Processing transcript and audio...")
                    output_path = os.path.join(out_path, "dataset")
                    os.makedirs(output_path, exist_ok=True)
                    
                    # Use transcript language as dataset language (transcript_lang is the actual content language)
                    dataset_language = transcript_lang
                    print(f"Setting dataset language to '{dataset_language}' (from YouTube transcript language)")
                    
                    train_csv, eval_csv, duration = srt_processor.process_srt_with_media(
                        srt_path=srt_path,
                        media_path=audio_path,
                        output_dir=output_path,
                        language=dataset_language,
                        gradio_progress=progress
                    )
                    
                    # Count segments
                    import pandas as pd
                    train_df = pd.read_csv(train_csv, sep='|')
                    eval_df = pd.read_csv(eval_csv, sep='|')
                    num_segments = len(train_df) + len(eval_df)
                    
                    # Track this dataset
                    video_id = tracker._extract_youtube_id(url)
                    if video_id:
                        tracker.add_youtube_dataset(
                            url=url,
                            video_id=video_id,
                            title=info.get('title', 'Unknown'),
                            language=dataset_language,
                            duration=info.get('duration', 0),
                            num_segments=num_segments,
                            output_path=output_path
                        )
                    
                    # Cleanup temp directory
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except:
                        pass
                    
                    progress(1.0, desc="YouTube processing complete!")
                    return f"✓ YouTube Processing Complete!\nTitle: {info.get('title', 'Unknown')}\nDuration: {info.get('duration', 0):.0f}s\nProcessed {num_segments} segments\nDataset created at: {output_path}\n\nℹ This dataset has been saved to history and won't be reprocessed."
                    
                except Exception as e:
                    traceback.print_exc()
                    return f"❌ Error downloading YouTube video: {str(e)}"
            
            def slice_audio_file(audio_file_path, threshold_db, min_length, min_interval, max_sil_kept, auto_transcribe, whisper_model, language, out_path, progress=gr.Progress(track_tqdm=True)):
                """Slice audio file using RMS-based silence detection"""
                try:
                    if not audio_file_path:
                        return "Please upload an audio file to slice!"
                    
                    progress(0, desc="Initializing audio slicer...")
                    
                    output_path = os.path.join(out_path, "dataset", "wavs")
                    os.makedirs(output_path, exist_ok=True)
                    
                    # Load audio
                    import librosa
                    import soundfile as sf
                    progress(0.1, desc="Loading audio...")
                    audio, sr = librosa.load(audio_file_path, sr=22050, mono=True)
                    
                    # Initialize slicer
                    progress(0.2, desc="Slicing audio...")
                    slicer = audio_slicer.Slicer(
                        sr=sr,
                        threshold=threshold_db,
                        min_length=int(min_length * 1000),  # Convert to milliseconds
                        min_interval=int(min_interval * 1000),
                        hop_size=10,
                        max_sil_kept=int(max_sil_kept * 1000)
                    )
                    
                    # Slice audio
                    chunks = slicer.slice(audio)
                    
                    if not chunks:
                        return "❌ No segments created. Try adjusting the slicing parameters."
                    
                    progress(0.4, desc=f"Saving {len(chunks)} segments...")
                    
                    # Save segments
                    segment_files = []
                    for i, chunk in enumerate(chunks):
                        segment_filename = f"{Path(audio_file_path).stem}_segment_{str(i).zfill(4)}.wav"
                        segment_path = os.path.join(output_path, segment_filename)
                        sf.write(segment_path, chunk, sr)
                        segment_files.append(segment_path)
                    
                    # Auto-transcribe if requested
                    transcription_status = "Disabled"
                    if auto_transcribe:
                        progress(0.6, desc="Transcribing segments with Whisper...")
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        compute_type = "float16" if torch.cuda.is_available() else "float32"
                        asr_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
                        
                        # Create metadata by transcribing segments
                        # This processes all segments and creates train/eval CSVs
                        try:
                            train_meta, eval_meta, total_size = format_audio_list(
                                segment_files,
                                asr_model=asr_model,
                                target_language=language,
                                out_path=os.path.join(out_path, "dataset"),
                                gradio_progress=progress
                            )
                            transcription_status = "Enabled - Metadata created"
                        except Exception as e:
                            print(f"Warning: Could not create metadata: {e}")
                            transcription_status = "Enabled - Segments created, metadata pending"
                    
                    progress(1.0, desc="Audio slicing complete!")
                    return f"✓ Audio Slicing Complete!\nCreated {len(chunks)} segments\nSegments saved to: {output_path}\nAuto-transcription: {transcription_status}"
                    
                except Exception as e:
                    traceback.print_exc()
                    return f"❌ Error slicing audio: {str(e)}"
        
            def preprocess_dataset(audio_path, audio_folder_path, language, whisper_model, out_path, train_csv, eval_csv, use_g2p_preprocessing=False, g2p_backend="transphone", progress=gr.Progress(track_tqdm=True)):
                clear_gpu_cache()
            
                train_csv = ""
                eval_csv = ""
            
                out_path = os.path.join(out_path, "dataset")
                os.makedirs(out_path, exist_ok=True)
                
                # Configure G2P if Amharic preprocessing is enabled
                g2p_converter = None
                if use_g2p_preprocessing and language == "amh":
                    try:
                        from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
                        print(f"Initializing Amharic G2P with backend: {g2p_backend}")
                        g2p_converter = AmharicG2P(backend=g2p_backend)
                        print("G2P converter initialized successfully")
                    except ImportError as e:
                        print(f"Warning: Could not load Amharic G2P: {e}")
                        print("Dataset will be created without G2P preprocessing")
                    except Exception as e:
                        print(f"Error initializing G2P: {e}")
                        print("Dataset will be created without G2P preprocessing")
            
                if audio_folder_path:
                    audio_files = list(list_audios(audio_folder_path))
                else:
                    audio_files = audio_path
            
                if not audio_files:
                    return "No audio files found! Please provide files via Gradio or specify a folder path.", "", ""
                else:
                    try:
                        # Loading Whisper
                        device = "cuda" if torch.cuda.is_available() else "cpu" 
                        
                        # Detect compute type 
                        if torch.cuda.is_available():
                            compute_type = "float16"
                        else:
                            compute_type = "float32"
                        
                        asr_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
                        
                        # Apply G2P preprocessing if converter is available
                        # Note: This would require modifications to format_audio_list to accept g2p_converter
                        # For now, we'll note this and users can preprocess separately
                        if g2p_converter:
                            print("Note: G2P preprocessing will be applied during training")
                            print("For dataset-level G2P, use the preprocessing utility")
                        
                        train_meta, eval_meta, audio_total_size = format_audio_list(audio_files, asr_model=asr_model, target_language=language, out_path=out_path, gradio_progress=progress)
                    except:
                        traceback.print_exc()
                        error = traceback.format_exc()
                        return f"The data processing was interrupted due an error !! Please check the console to verify the full error message! \n Error summary: {error}", "", ""
            
                # clear_gpu_cache()
            
                # if audio total len is less than 2 minutes raise an error
                if audio_total_size < 120:
                    message = "The sum of the duration of the audios that you provided should be at least 2 minutes!"
                    print(message)
                    return message, "", ""
            
                print("Dataset Processed!")
                return "Dataset Processed!", train_meta, eval_meta


        with gr.Tab("2 - Fine-tuning XTTS Encoder"):
            load_params_btn = gr.Button(value="Load Params from output folder")
            version = gr.Dropdown(
                label="XTTS base version",
                value="v2.0.2",
                choices=[
                    "v2.0.3",
                    "v2.0.2",
                    "v2.0.1",
                    "v2.0.0",
                    "main"
                ],
            )
            train_csv = gr.Textbox(
                label="Train CSV:",
            )
            eval_csv = gr.Textbox(
                label="Eval CSV:",
            )
            custom_model = gr.Textbox(
                label="(Optional) Custom model.pth file , leave blank if you want to use the base file.",
                value="",
            )
            num_epochs =  gr.Slider(
                label="Number of epochs:",
                minimum=1,
                maximum=100,
                step=1,
                value=args.num_epochs,
            )
            batch_size = gr.Slider(
                label="Batch size:",
                minimum=2,
                maximum=512,
                step=1,
                value=args.batch_size,
            )
            grad_acumm = gr.Slider(
                label="Grad accumulation steps:",
                minimum=2,
                maximum=128,
                step=1,
                value=args.grad_acumm,
            )
            max_audio_length = gr.Slider(
                label="Max permitted audio size in seconds:",
                minimum=2,
                maximum=20,
                step=1,
                value=args.max_audio_length,
            )
            clear_train_data = gr.Dropdown(
                label="Clear train data, you will delete selected folder, after optimizing",
                value="none",
                choices=[
                    "none",
                    "run",
                    "dataset",
                    "all"
                ])
            
            # Amharic G2P Training Options
            with gr.Accordion("Amharic G2P Training Options (for 'amh' language)", open=False) as amh_training_accordion:
                enable_amharic_g2p = gr.Checkbox(
                    label="Enable Amharic G2P for training",
                    value=False,
                    info="Use phoneme tokenization for Amharic training"
                )
                g2p_backend_train = gr.Dropdown(
                    label="G2P Backend for Training",
                    value="transphone",
                    choices=["transphone", "epitran", "rule_based"],
                    info="Backend used for G2P conversion during training"
                )
            
            progress_train = gr.Label(
                label="Progress:"
            )

            # demo.load(read_logs, None, logs_tts_train, every=1)
            train_btn = gr.Button(value="Step 2 - Run the training")
            optimize_model_btn = gr.Button(value="Step 2.5 - Optimize the model")
            
            import os
            import shutil
            from pathlib import Path
            import traceback
            
            def train_model(custom_model, version, language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length, enable_amharic_g2p=False, g2p_backend_train="transphone"):
                clear_gpu_cache()
          
                # Check if `custom_model` is a URL and download it if true.
                if custom_model.startswith("http"):
                    print("Downloading custom model from URL...")
                    custom_model = download_file(custom_model, "custom_model.pth")
                    if not custom_model:
                        return "Failed to download the custom model.", "", "", "", ""
            
                run_dir = Path(output_path) / "run"
            
                # Remove train dir
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                
                # Check if the dataset language matches the language you specified 
                lang_file_path = Path(output_path) / "dataset" / "lang.txt"
            
                # Check if lang.txt already exists and contains a different language
                current_language = None
                if lang_file_path.exists():
                    with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
                        current_language = existing_lang_file.read().strip()
                        if current_language != language:
                            print("The language that was prepared for the dataset does not match the specified language. Change the language to the one specified in the dataset")
                            language = current_language
                        
                if not train_csv or not eval_csv:
                    return "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !", "", "", "", ""
                # Configure Amharic G2P for training
                use_amharic_g2p = enable_amharic_g2p and language == "amh"
                if use_amharic_g2p:
                    print(f"Amharic G2P enabled with backend: {g2p_backend_train}")
                    # Set backend order based on user selection
                    try:
                        from amharic_tts.config.amharic_config import G2PBackend
                        # This will be used in the training pipeline
                    except ImportError:
                        print("Warning: Amharic G2P module not available")
                        use_amharic_g2p = False
                
                try:
                    # convert seconds to waveform frames
                    max_audio_length = int(max_audio_length * 22050)
                    speaker_xtts_path, config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(custom_model, version, language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path=output_path, max_audio_length=max_audio_length, use_amharic_g2p=use_amharic_g2p)
                except:
                    traceback.print_exc()
                    error = traceback.format_exc()
                    return f"The training was interrupted due to an error !! Please check the console to check the full error message! \n Error summary: {error}", "", "", "", ""
            
                ready_dir = Path(output_path) / "ready"
            
                ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
            
                shutil.copy(ft_xtts_checkpoint, ready_dir / "unoptimize_model.pth")
            
                ft_xtts_checkpoint = os.path.join(ready_dir, "unoptimize_model.pth")
            
                # Move reference audio to output folder and rename it
                speaker_reference_path = Path(speaker_wav)
                speaker_reference_new_path = ready_dir / "reference.wav"
                shutil.copy(speaker_reference_path, speaker_reference_new_path)
            
                print("Model training done!")
                return "Model training done!", config_path, vocab_file, ft_xtts_checkpoint, speaker_xtts_path, speaker_reference_new_path

            def optimize_model(out_path, clear_train_data):
                # print(out_path)
                out_path = Path(out_path)  # Ensure that out_path is a Path object.
            
                ready_dir = out_path / "ready"
                run_dir = out_path / "run"
                dataset_dir = out_path / "dataset"
            
                # Clear specified training data directories.
                if clear_train_data in {"run", "all"} and run_dir.exists():
                    try:
                        shutil.rmtree(run_dir)
                    except PermissionError as e:
                        print(f"An error occurred while deleting {run_dir}: {e}")
            
                if clear_train_data in {"dataset", "all"} and dataset_dir.exists():
                    try:
                        shutil.rmtree(dataset_dir)
                    except PermissionError as e:
                        print(f"An error occurred while deleting {dataset_dir}: {e}")
            
                # Get full path to model
                model_path = ready_dir / "unoptimize_model.pth"

                if not model_path.is_file():
                    return "Unoptimized model not found in ready folder", ""
            
                # Load the checkpoint and remove unnecessary parts.
                checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
                del checkpoint["optimizer"]

                for key in list(checkpoint["model"].keys()):
                    if "dvae" in key:
                        del checkpoint["model"][key]

                # Make sure out_path is a Path object or convert it to Path
                os.remove(model_path)

                  # Save the optimized model.
                optimized_model_file_name="model.pth"
                optimized_model=ready_dir/optimized_model_file_name
            
                torch.save(checkpoint, optimized_model)
                ft_xtts_checkpoint=str(optimized_model)

                clear_gpu_cache()
        
                return f"Model optimized and saved at {ft_xtts_checkpoint}!", ft_xtts_checkpoint

            def load_params(out_path):
                path_output = Path(out_path)
                
                dataset_path = path_output / "dataset"

                if not dataset_path.exists():
                    return "The output folder does not exist!", "", ""

                eval_train = dataset_path / "metadata_train.csv"
                eval_csv = dataset_path / "metadata_eval.csv"

                # Write the target language to lang.txt in the output directory
                lang_file_path =  dataset_path / "lang.txt"

                # Check if lang.txt already exists and contains a different language
                current_language = None
                if os.path.exists(lang_file_path):
                    with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
                        current_language = existing_lang_file.read().strip()

                clear_gpu_cache()

                print(current_language)
                return "The data has been updated", eval_train, eval_csv, current_language

        with gr.Tab("3 - Inference"):
            with gr.Row():
                with gr.Column() as col1:
                    load_params_tts_btn = gr.Button(value="Load params for TTS from output folder")
                    xtts_checkpoint = gr.Textbox(
                        label="XTTS checkpoint path:",
                        value="",
                    )
                    xtts_config = gr.Textbox(
                        label="XTTS config path:",
                        value="",
                    )

                    xtts_vocab = gr.Textbox(
                        label="XTTS vocab path:",
                        value="",
                    )
                    xtts_speaker = gr.Textbox(
                        label="XTTS speaker path:",
                        value="",
                    )
                    progress_load = gr.Label(
                        label="Progress:"
                    )
                    load_btn = gr.Button(value="Step 3 - Load Fine-tuned XTTS model")

                with gr.Column() as col2:
                    speaker_reference_audio = gr.Textbox(
                        label="Speaker reference audio:",
                        value="",
                    )
                    tts_language = gr.Dropdown(
                        label="Language",
                        value="en",
                        choices=[
                            "en",
                            "es",
                            "fr",
                            "de",
                            "it",
                            "pt",
                            "pl",
                            "tr",
                            "ru",
                            "nl",
                            "cs",
                            "ar",
                            "zh",
                            "hu",
                            "ko",
                            "ja",
                            "amh",  # Amharic
                        ]
                    )
                    tts_text = gr.Textbox(
                        label="Input Text.",
                        value="This model sounds really good and above all, it's reasonably fast.",
                    )
                    with gr.Accordion("Advanced settings", open=False) as acr:
                        temperature = gr.Slider(
                            label="temperature",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.75,
                        )
                        length_penalty  = gr.Slider(
                            label="length_penalty",
                            minimum=-10.0,
                            maximum=10.0,
                            step=0.5,
                            value=1,
                        )
                        repetition_penalty = gr.Slider(
                            label="repetition penalty",
                            minimum=1,
                            maximum=10,
                            step=0.5,
                            value=5,
                        )
                        top_k = gr.Slider(
                            label="top_k",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=50,
                        )
                        top_p = gr.Slider(
                            label="top_p",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.85,
                        )
                        sentence_split = gr.Checkbox(
                            label="Enable text splitting",
                            value=True,
                        )
                        use_config = gr.Checkbox(
                            label="Use Inference settings from config, if disabled use the settings above",
                            value=False,
                        )
                    tts_btn = gr.Button(value="Step 4 - Inference")
                    
                    model_download_btn = gr.Button("Step 5 - Download Optimized Model ZIP")
                    dataset_download_btn = gr.Button("Step 5 - Download Dataset ZIP")
                
                    model_zip_file = gr.File(label="Download Optimized Model", interactive=False)
                    dataset_zip_file = gr.File(label="Download Dataset", interactive=False)



                with gr.Column() as col3:
                    progress_gen = gr.Label(
                        label="Progress:"
                    )
                    tts_output_audio = gr.Audio(label="Generated Audio.")
                    reference_audio = gr.Audio(label="Reference audio used.")

            prompt_compute_btn.click(
                fn=preprocess_dataset,
                inputs=[
                    upload_file,
                    audio_folder_path,
                    lang,
                    whisper_model,
                    out_path,
                    train_csv,
                    eval_csv,
                    use_amharic_g2p_preprocessing,
                    g2p_backend_selection,
                ],
                outputs=[
                    progress_data,
                    train_csv,
                    eval_csv,
                ],
            )
            
            # Advanced features button handlers
            process_srt_btn.click(
                fn=process_srt_media,
                inputs=[
                    srt_file,
                    media_file,
                    lang,
                    out_path,
                ],
                outputs=[srt_status],
            )
            
            download_youtube_btn.click(
                fn=download_youtube_video,
                inputs=[
                    youtube_url,
                    youtube_transcript_lang,
                    lang,
                    out_path,
                    youtube_batch_mode,  # Add batch mode parameter
                ],
                outputs=[youtube_status],
            )
            
            slice_audio_btn.click(
                fn=slice_audio_file,
                inputs=[
                    slicer_audio_file,
                    slicer_threshold_db,
                    slicer_min_length,
                    slicer_min_interval,
                    slicer_max_sil_kept,
                    slicer_auto_transcribe,
                    whisper_model,
                    lang,
                    out_path,
                ],
                outputs=[slicer_status],
            )


            load_params_btn.click(
                fn=load_params,
                inputs=[out_path],
                outputs=[
                    progress_train,
                    train_csv,
                    eval_csv,
                    lang
                ]
            )


            train_btn.click(
                fn=train_model,
                inputs=[
                    custom_model,
                    version,
                    lang,
                    train_csv,
                    eval_csv,
                    num_epochs,
                    batch_size,
                    grad_acumm,
                    out_path,
                    max_audio_length,
                    enable_amharic_g2p,
                    g2p_backend_train,
                ],
                outputs=[progress_train, xtts_config, xtts_vocab, xtts_checkpoint,xtts_speaker, speaker_reference_audio],
            )

            optimize_model_btn.click(
                fn=optimize_model,
                inputs=[
                    out_path,
                    clear_train_data
                ],
                outputs=[progress_train,xtts_checkpoint],
            )
            
            load_btn.click(
                fn=load_model,
                inputs=[
                    xtts_checkpoint,
                    xtts_config,
                    xtts_vocab,
                    xtts_speaker
                ],
                outputs=[progress_load],
            )

            tts_btn.click(
                fn=run_tts,
                inputs=[
                    tts_language,
                    tts_text,
                    speaker_reference_audio,
                    temperature,
                    length_penalty,
                    repetition_penalty,
                    top_k,
                    top_p,
                    sentence_split,
                    use_config
                ],
                outputs=[progress_gen, tts_output_audio,reference_audio],
            )

            load_params_tts_btn.click(
                fn=load_params_tts,
                inputs=[
                    out_path,
                    version
                    ],
                outputs=[progress_load,xtts_checkpoint,xtts_config,xtts_vocab,xtts_speaker,speaker_reference_audio],
            )
             
            model_download_btn.click(
                fn=get_model_zip,
                inputs=[out_path],
                outputs=[model_zip_file]
            )
            
            dataset_download_btn.click(
                fn=get_dataset_zip,
                inputs=[out_path],
                outputs=[dataset_zip_file]
            )

    demo.launch(
        share=args.share,
        debug=False,
        server_port=args.port,
        # inweb=True,
        # server_name="localhost"
    )
