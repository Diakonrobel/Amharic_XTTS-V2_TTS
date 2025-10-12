#!/bin/bash
################################################################################
# Complete Amharic XTTS Retraining Script
################################################################################
# This script handles EVERYTHING from dataset to training with proper vocab
################################################################################

set -e  # Exit on error

echo "========================================================================"
echo "🇪🇹 Amharic XTTS Retraining - Clean Setup"
echo "========================================================================"
echo ""

# Configuration
WORKSPACE="$(pwd)"
LANG_CODE="amh"
USE_G2P="true"
G2P_BACKEND="rule_based"
EPOCHS=30
BATCH_SIZE=4
GRAD_ACCUM=4

echo "📂 Working directory: $WORKSPACE"
echo "🌍 Language: $LANG_CODE"
echo "🔤 G2P Enabled: $USE_G2P"
echo "📊 Training: $EPOCHS epochs, batch=$BATCH_SIZE, grad_accum=$GRAD_ACCUM"
echo ""

# Step 1: Prepare clean vocabulary
echo "========================================================================"
echo "Step 1: Prepare Reference Vocabulary"
echo "========================================================================"

if [ -f "finetune_models/ready/vocab_extended_amharic_7537_backup.json" ]; then
    echo "✅ Using backup vocab (7537 tokens)"
    cp finetune_models/ready/vocab_extended_amharic_7537_backup.json vocab_reference_clean.json
elif [ -f "finetune_models/ready/vocab_extended_amharic.json" ]; then
    echo "⚠️  Using current vocab - checking size..."
    cp finetune_models/ready/vocab_extended_amharic.json vocab_reference_clean.json
else
    echo "❌ No vocab file found!"
    echo "Please ensure you have a vocab file in finetune_models/ready/"
    exit 1
fi

VOCAB_SIZE=$(python -c "import json; print(len(json.load(open('vocab_reference_clean.json'))['model']['vocab']))")
echo "📚 Vocab size: $VOCAB_SIZE tokens"

if [ "$VOCAB_SIZE" -eq 7537 ]; then
    echo "✅ Perfect! Using 7537-token vocab"
elif [ "$VOCAB_SIZE" -eq 7536 ]; then
    echo "⚠️  Vocab is 7536 tokens - this will work, but differs from original"
else
    echo "⚠️  Unexpected vocab size: $VOCAB_SIZE"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 2: Verify dataset exists
echo ""
echo "========================================================================"
echo "Step 2: Verify Dataset"
echo "========================================================================"

if [ ! -d "finetune_models/dataset/wavs" ]; then
    echo "❌ No dataset found at finetune_models/dataset/wavs/"
    echo "Please prepare your dataset first!"
    echo ""
    echo "Quick guide:"
    echo "1. Put audio files (.wav) in finetune_models/dataset/wavs/"
    echo "2. Create metadata.csv with format: filename.wav|Amharic text"
    echo "3. Run this script again"
    exit 1
fi

AUDIO_COUNT=$(ls finetune_models/dataset/wavs/*.wav 2>/dev/null | wc -l)
echo "🎵 Found $AUDIO_COUNT audio files"

if [ $AUDIO_COUNT -lt 10 ]; then
    echo "⚠️  Warning: Only $AUDIO_COUNT files found. Need at least 50 for decent quality."
    read -p "Continue anyway for testing? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for metadata
if [ ! -f "finetune_models/dataset/metadata.csv" ]; then
    echo "❌ No metadata.csv found!"
    echo "Please create finetune_models/dataset/metadata.csv"
    echo "Format: filename.wav|Amharic transcription text"
    exit 1
fi

METADATA_LINES=$(wc -l < finetune_models/dataset/metadata.csv)
echo "📝 Metadata entries: $METADATA_LINES"

# Step 3: Choose speaker reference audio
echo ""
echo "========================================================================"
echo "Step 3: Speaker Reference Audio"
echo "========================================================================"

# Try to find a good reference automatically
FIRST_WAV=$(ls finetune_models/dataset/wavs/*.wav 2>/dev/null | head -1)
if [ -z "$FIRST_WAV" ]; then
    echo "❌ No audio files found!"
    exit 1
fi

echo "🎤 Using speaker reference: $FIRST_WAV"
echo ""
echo "This audio will be used to capture the speaker's voice characteristics."
echo "It should be:"
echo "  - Clear, no background noise"
echo "  - 5-30 seconds long"
echo "  - Representative of the speaker's voice"
echo ""

SPEAKER_REF="$FIRST_WAV"

# Ask user if they want to change it
read -p "Use this audio as speaker reference? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Enter path to your preferred speaker reference audio:"
    read SPEAKER_REF
    if [ ! -f "$SPEAKER_REF" ]; then
        echo "❌ File not found: $SPEAKER_REF"
        exit 1
    fi
fi

# Step 4: Create training-ready dataset
echo ""
echo "========================================================================"
echo "Step 4: Create Training Dataset (This may take 10-30 minutes)"
echo "========================================================================"

# Create output directory
READY_DIR="finetune_models/ready_clean_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$READY_DIR"

echo "📦 Output directory: $READY_DIR"
echo ""
echo "Processing dataset with:"
echo "  - G2P: $USE_G2P"
echo "  - Vocab: vocab_reference_clean.json"
echo "  - Language: $LANG_CODE"
echo ""

# Run dataset creation using headlessXttsTrain.py (it handles preprocessing)
python headlessXttsTrain.py \
    --input_audio "$SPEAKER_REF" \
    --lang "$LANG_CODE" \
    --dataset_path "finetune_models/dataset" \
    --out_path "$READY_DIR" \
    --prepare_only \
    --use_g2p \
    --g2p_backend "$G2P_BACKEND" \
    --custom_vocab "vocab_reference_clean.json"

if [ $? -ne 0 ]; then
    echo "❌ Dataset preparation failed!"
    echo "Check the error messages above."
    exit 1
fi

echo ""
echo "✅ Dataset preparation complete!"
echo ""

# Verify dataset was created
if [ ! -f "$READY_DIR/metadata_train.csv" ]; then
    echo "❌ Training metadata not found!"
    exit 1
fi

TRAIN_SAMPLES=$(wc -l < "$READY_DIR/metadata_train.csv")
echo "📊 Training samples: $TRAIN_SAMPLES"

if [ -f "$READY_DIR/metadata_eval.csv" ]; then
    EVAL_SAMPLES=$(wc -l < "$READY_DIR/metadata_eval.csv")
    echo "📊 Evaluation samples: $EVAL_SAMPLES"
fi

# Step 5: Start Training
echo ""
echo "========================================================================"
echo "Step 5: Start Training"
echo "========================================================================"
echo ""
echo "Training configuration:"
echo "  - Epochs: $EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Gradient accumulation: $GRAD_ACCUM"
echo "  - Effective batch: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  - G2P enabled: $USE_G2P"
echo "  - Language: $LANG_CODE"
echo ""
echo "Expected duration:"
echo "  - V100 GPU: ~4-6 hours"
echo "  - T4 GPU: ~8-12 hours"
echo "  - A100 GPU: ~2-3 hours"
echo ""

read -p "Start training now? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled. You can start it manually later:"
    echo ""
    echo "python headlessXttsTrain.py \\"
    echo "    --input_audio \"$SPEAKER_REF\" \\"
    echo "    --lang $LANG_CODE \\"
    echo "    --train_csv \"$READY_DIR/metadata_train.csv\" \\"
    echo "    --eval_csv \"$READY_DIR/metadata_eval.csv\" \\"
    echo "    --out_path \"$READY_DIR\" \\"
    echo "    --epochs $EPOCHS \\"
    echo "    --batch_size $BATCH_SIZE \\"
    echo "    --grad_acumm $GRAD_ACCUM \\"
    echo "    --use_g2p \\"
    echo "    --g2p_backend $G2P_BACKEND"
    exit 0
fi

echo ""
echo "🚀 Starting training..."
echo "💡 Tip: Training runs in foreground. Use 'screen' or 'tmux' for long sessions."
echo ""

# Run training
python headlessXttsTrain.py \
    --input_audio "$SPEAKER_REF" \
    --lang "$LANG_CODE" \
    --train_csv "$READY_DIR/metadata_train.csv" \
    --eval_csv "$READY_DIR/metadata_eval.csv" \
    --out_path "$READY_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --grad_acumm "$GRAD_ACCUM" \
    --use_g2p \
    --g2p_backend "$G2P_BACKEND" \
    --custom_vocab "$READY_DIR/vocab.json"

echo ""
echo "========================================================================"
echo "✅ Training Complete!"
echo "========================================================================"
echo ""
echo "📁 Model files are in: $READY_DIR"
echo ""
echo "Next steps:"
echo "1. Test the model:"
echo "   python test_amharic_modes.py"
echo ""
echo "2. Use the new checkpoint in xtts_demo.py"
echo "   Checkpoint: $READY_DIR/best_model.pth"
echo "   Vocab: $READY_DIR/vocab.json"
echo ""
echo "3. Inference settings:"
echo "   - Language: amh"
echo "   - Enable G2P: YES"
echo "   - Use vocab from: $READY_DIR/vocab.json"
echo ""
echo "🎉 Your Amharic TTS model is ready!"
