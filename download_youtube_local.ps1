# YouTube Batch Downloader for Windows
# Downloads and processes YouTube videos locally where YouTube blocking is less severe
# 
# Usage: .\download_youtube_local.ps1

param(
    [string]$Language = "am",  # Amharic
    [string]$OutputPath = ".\finetune_models"
)

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "YouTube Batch Downloader (Local)" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Your YouTube URLs
$urls = @(
    "https://www.youtube.com/watch?v=qXFRHxF3rAM",
    "https://www.youtube.com/watch?v=-MCmw5Pugqw",
    "https://www.youtube.com/watch?v=-EOH7Ub3hss",
    "https://www.youtube.com/watch?v=qC_UgojfoQA",
    "https://www.youtube.com/watch?v=6MLcr6UqafQ",
    "https://www.youtube.com/watch?v=LKnW_uD6TQA",
    "https://www.youtube.com/watch?v=4oGa4gd0PX4"
)

Write-Host "Will process $($urls.Count) YouTube videos" -ForegroundColor Yellow
Write-Host "Language: $Language" -ForegroundColor Yellow
Write-Host "Output: $OutputPath" -ForegroundColor Yellow
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python first." -ForegroundColor Red
    exit 1
}

# Check if yt-dlp is installed
try {
    $ytdlpVersion = yt-dlp --version 2>&1
    Write-Host "✓ yt-dlp found: $ytdlpVersion" -ForegroundColor Green
} catch {
    Write-Host "⚠ yt-dlp not found. Installing..." -ForegroundColor Yellow
    pip install -U yt-dlp
}

# Start the WebUI in background
Write-Host ""
Write-Host "Starting WebUI on http://localhost:7860 ..." -ForegroundColor Cyan
Write-Host "Please use the UI to process the videos:" -ForegroundColor Cyan
Write-Host "  1. Go to 'YouTube Processing' tab" -ForegroundColor White
Write-Host "  2. Paste these URLs (comma-separated):" -ForegroundColor White
Write-Host ""

foreach ($url in $urls) {
    Write-Host "     $url" -ForegroundColor Gray
}

Write-Host ""
Write-Host "  3. Set language to 'Amharic (አማርኛ)'" -ForegroundColor White
Write-Host "  4. Enable 'Batch Mode'" -ForegroundColor White
Write-Host "  5. Click 'Download & Process'" -ForegroundColor White
Write-Host ""
Write-Host "OR use this comma-separated list:" -ForegroundColor Yellow
Write-Host ($urls -join ",") -ForegroundColor Gray
Write-Host ""

# Launch UI
python xtts_demo.py --port 7860

Write-Host ""
Write-Host "WebUI closed." -ForegroundColor Cyan
Write-Host ""
Write-Host "If processing was successful, your dataset is in:" -ForegroundColor Green
Write-Host "  $OutputPath\dataset\" -ForegroundColor White
Write-Host ""
Write-Host "To upload to Lightning:" -ForegroundColor Yellow
Write-Host "  1. Compress: tar -czf dataset.tar.gz -C $OutputPath dataset" -ForegroundColor White
Write-Host "  2. Upload dataset.tar.gz to Lightning" -ForegroundColor White
Write-Host "  3. Extract on Lightning: tar -xzf dataset.tar.gz -C finetune_models/" -ForegroundColor White
