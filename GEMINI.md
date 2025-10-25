# GEMINI.md - Project Context

## Project Overview

This project is a comprehensive suite for fine-tuning Coqui XTTS (Text-to-Speech) models. It provides both a user-friendly web interface and a powerful headless script for automated workflows. The primary goal is to simplify the process of dataset creation, model training, and inference for creating custom voice clones.

**Main Technologies:**
*   **Python:** The core language of the project.
*   **Gradio:** Used to create the interactive web UI.
*   **PyTorch:** The underlying deep learning framework for XTTS.
*   **faster-whisper:** Used for accurate and fast audio transcription during the data processing phase.
*   **Coqui TTS:** The core XTTS model and utilities.

**Key Features:**
*   **Multi-source Data Processing:** Ingests data from various sources including local audio files, SRT/VTT subtitle files paired with media, and direct YouTube video URLs.
*   **Advanced Audio Processing:** Includes features like RMS-based audio slicing for long recordings and AI-powered background music removal (via Demucs) to create clean vocal datasets.
*   **Comprehensive Fine-tuning:** Offers detailed control over the training process, including setting epochs, batch size, learning rate, and advanced options like layer freezing and early stopping to prevent overfitting.
*   **Headless & UI Modes:** Can be run through an interactive Gradio web UI (`xtts_demo.py`) for experimentation or via a command-line script (`headlessXttsTrain.py`) for automation and integration into larger pipelines.
*   **Specialized Amharic Support:** Includes advanced support for the Amharic language, featuring multiple Grapheme-to-Phoneme (G2P) backends (Transphone, Epitran, rule-based) to improve pronunciation accuracy.
*   **Checkpoint Management:** Provides tools to select, test, and manage different model checkpoints to find the best-performing one.

## Building and Running

### Installation

The project uses Python and relies on dependencies listed in `requirements.txt`.

1.  **Prerequisites:**
    *   Python 3.10+
    *   PyTorch with CUDA support (for GPU acceleration).
    *   FFmpeg (must be installed and available in the system's PATH).

2.  **Installation Commands:**
    *   **On Linux/macOS:**
        ```bash
        bash install.sh
        ```
        This script utilizes `smart_install.py` to handle the setup.
    *   **On Windows:**
        ```bat
        install.bat
        ```

### Running the Application

The application can be run in two modes:

**1. Web UI (Gradio Interface)**

This is the recommended way for interactive use.

*   **On Linux/macOS:**
    ```bash
    bash start.sh
    ```
    This script activates a virtual environment and launches the Gradio server.
*   **On Windows:**
    ```bat
    start.bat
    ```
*   **Directly:**
    ```bash
    python xtts_demo.py
    ```

The interface will be available at `http://127.0.0.1:5003` by default.

**2. Headless Mode (Command-Line)**

For automated training and processing, use the `headlessXttsTrain.py` script. It provides a wide range of arguments to control the entire fine-tuning pipeline.

**Key Arguments for `headlessXttsTrain.py`:**
*   `--input_audio`: Path to a single audio file.
*   `--srt_file` & `--media_file`: Paths to a subtitle file and its corresponding media.
*   `--youtube_url`: URL of a YouTube video to process.
*   `--lang`: Language of the dataset (e.g., `en`, `es`, `am` for Amharic).
*   `--epochs`: Number of training epochs.
*   `--output_dir_base`: Base directory to save the output model and dataset.
*   `--model_name`: A specific name for the trained model.

**Example Headless Commands:**
*   **From an audio file:**
    ```bash
    python headlessXttsTrain.py --input_audio speaker.wav --lang en --epochs 10 --model_name my_voice
    ```
*   **From a YouTube video:**
    ```bash
    python headlessXttsTrain.py --youtube_url "https://youtube.com/watch?v=VIDEO_ID" --lang en --epochs 10
    ```

## Development Conventions

*   **Modular Structure:** The project is organized into several directories, with `utils` containing core functionalities like audio processing, SRT parsing, and training logic. The `webui` and `amharic_tts` directories contain specialized UI and language components.
*   **Configuration Files:** The project uses `.json` files for model configuration (`config.json`) and vocabulary (`vocab.json`).
*   **Shell Scripts:** Wrapper scripts (`.sh`, `.bat`) are provided for easy installation and execution across different operating systems.
*   **Headless First:** The core logic is implemented in a way that it can be used independently of the UI, as demonstrated by the `headlessXttsTrain.py` script. This promotes reusability and automation.
*   **Extensive Documentation:** The project contains numerous Markdown files (`.md`) that explain features, provide guides, and document implementation details, indicating a strong emphasis on clear documentation.
