#!/bin/bash
# Fix corrupted yt-dlp installation
# Run this on Lightning AI if you get "_ALL_CLASSES" import errors

echo "🔧 Fixing yt-dlp installation..."

# Uninstall completely
echo "Removing corrupted yt-dlp..."
pip uninstall yt-dlp -y

# Clear pip cache
echo "Clearing pip cache..."
pip cache purge || true

# Reinstall stable version
echo "Installing stable yt-dlp version..."
pip install 'yt-dlp>=2024.11.18,<2025.0.0' --no-cache-dir

# Verify installation
echo ""
echo "✅ Verifying installation..."
python -c "import yt_dlp; print(f'yt-dlp version: {yt_dlp.version.__version__}')"
python -c "from yt_dlp.extractor.extractors import _ALL_CLASSES; print('Import test: OK')" 2>/dev/null || echo "⚠️  Import test failed - try restarting Python kernel"

echo ""
echo "Done! If you still have errors, restart the Python kernel/runtime."
