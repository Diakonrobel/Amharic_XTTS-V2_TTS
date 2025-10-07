#!/bin/bash
echo "========================================"
echo "Amharic XTTS Fine-Tuning WebUI Installer"
echo "========================================"
echo ""
echo "This will install to your current Python environment."
echo "No virtual environment will be created."
echo ""
read -p "Press Enter to continue..."

# Run smart installer
python3 smart_install.py

echo ""
echo "Installation complete!"
echo "Run './launch.sh' or 'python3 xtts_demo.py' to start."

