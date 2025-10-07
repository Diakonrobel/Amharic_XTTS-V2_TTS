#!/bin/bash
#
# Save Training Data to GitHub LFS
# Usage: ./scripts/save_to_lfs.sh ["Optional commit message"]
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}💾 Saving training data to GitHub LFS...${NC}\n"

# Check if finetune_models directory exists
if [ ! -d "finetune_models" ]; then
    echo -e "${RED}❌ No finetune_models directory found!${NC}"
    echo "   Train a model first before saving."
    exit 1
fi

# Get commit message
COMMIT_MSG="${1:-Training checkpoint at $(date '+%Y-%m-%d %H:%M:%S')}"

# Show what will be saved
echo -e "${YELLOW}📊 Current training data:${NC}"
du -sh finetune_models/* 2>/dev/null || echo "No files yet"
echo ""

# Add all changes
echo -e "${YELLOW}📝 Adding changes...${NC}"
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo -e "${YELLOW}ℹ️  No changes to commit.${NC}"
    exit 0
fi

# Commit
echo -e "${YELLOW}💾 Committing: ${COMMIT_MSG}${NC}"
git commit -m "$COMMIT_MSG"

# Push to GitHub (including LFS)
echo -e "${YELLOW}⬆️  Pushing to GitHub LFS...${NC}"
git push origin main

echo ""
echo -e "${GREEN}✅ Training data saved successfully!${NC}"
echo -e "${GREEN}🎉 Your progress is now safely stored on GitHub!${NC}"
echo ""
echo "View at: https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/')"
