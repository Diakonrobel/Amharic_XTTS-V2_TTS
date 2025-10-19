# Upload and Run Dataset Merger on Lightning AI

$SSH_HOST = "s_01k7x54qcrv1atww40z8bxf9a3@ssh.lightning.ai"
$REMOTE_DIR = "/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/finetune_models"
$LOCAL_SCRIPT = "merge_datasets.py"

Write-Host "`n╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Dataset Merger - Remote Executor     ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Check if script exists
if (-not (Test-Path $LOCAL_SCRIPT)) {
    Write-Host "✗ Error: $LOCAL_SCRIPT not found in current directory" -ForegroundColor Red
    exit 1
}

# Upload the script
Write-Host "ℹ Uploading merge script to Lightning AI..." -ForegroundColor Cyan
scp $LOCAL_SCRIPT "${SSH_HOST}:${REMOTE_DIR}/"

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to upload script" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Script uploaded successfully`n" -ForegroundColor Green

# Run the script interactively
Write-Host "ℹ Starting merge process on remote server..." -ForegroundColor Cyan
Write-Host "⚠ This will run interactively - follow the prompts`n" -ForegroundColor Yellow

ssh -t $SSH_HOST "cd $REMOTE_DIR && python3 merge_datasets.py"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Merge completed successfully!" -ForegroundColor Green
    
    # Ask if user wants to download the report
    $download = Read-Host "`nDownload merge report? (yes/no)"
    if ($download -eq "yes") {
        Write-Host "ℹ Fetching latest report..." -ForegroundColor Cyan
        scp "${SSH_HOST}:${REMOTE_DIR}/merge_report_*.json" .
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Report downloaded" -ForegroundColor Green
        }
    }
} else {
    Write-Host "`n✗ Merge failed or was cancelled" -ForegroundColor Red
}

Write-Host ""
