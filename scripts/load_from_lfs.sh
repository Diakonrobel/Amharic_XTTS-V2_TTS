#!/bin/bash
#
# Load Training Data from GitHub LFS
# Usage: ./scripts/load_from_lfs.sh
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔽 Loading training data from GitHub LFS...${NC}\n"

# Pull latest changes
echo -e "${YELLOW}📥 Pulling latest changes...${NC}"
git pull origin main

# Pull LFS files
echo -e "${YELLOW}📦 Pulling LFS files...${NC}"
git lfs pull

echo ""
echo -e "${GREEN}✅ Training data loaded successfully!${NC}"
echo ""

# Show what was loaded
if [ -d "finetune_models" ]; then
    echo -e "${YELLOW}📊 Available training data:${NC}"
    du -sh finetune_models/* 2>/dev/null || echo "No training data found"
else
    echo -e "${YELLOW}ℹ️  No training data available yet.${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Ready to continue training!${NC}"
