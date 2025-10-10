#!/bin/bash
# Script to verify the Amharic TTS pronunciation fix on Lightning AI machine

echo "====================================================="
echo "Amharic TTS Pronunciation Fix Verification"
echo "====================================================="

# Check if the fix is applied in xtts_demo.py
echo "Checking if the fix is applied in xtts_demo.py..."
if grep -q "FIXED: Don't override Amharic to English for G2P" xtts_demo.py; then
  echo "✅ Fix is applied in xtts_demo.py"
else
  echo "❌ Fix is NOT applied in xtts_demo.py"
  echo "Please pull the latest changes from GitHub"
  exit 1
fi

# Run the standalone verification script
echo -e "\nRunning standalone verification script..."
python amharic_fix_verification_standalone.py

# Check if G2P backends are installed
echo -e "\nChecking G2P backend installation..."
python -c "import importlib; print('Transphone:', importlib.util.find_spec('transphone') is not None); print('Epitran:', importlib.util.find_spec('epitran') is not None)"

echo -e "\n====================================================="
echo "Next steps:"
echo "1. Test with actual inference using your fine-tuned model"
echo "2. If G2P backends are missing, install them:"
echo "   pip install transphone epitran"
echo "====================================================="