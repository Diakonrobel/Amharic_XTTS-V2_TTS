# Lightning AI SSH Manager
# Manages remote operations on Lightning AI workspace

# Configuration
$SSH_HOST = "s_01k7x54qcrv1atww40z8bxf9a3@ssh.lightning.ai"
$REMOTE_DIR = "/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS"

# Color output functions
function Write-Success { param($msg) Write-Host "✓ $msg" -ForegroundColor Green }
function Write-Error { param($msg) Write-Host "✗ $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "ℹ $msg" -ForegroundColor Cyan }
function Write-Warning { param($msg) Write-Host "⚠ $msg" -ForegroundColor Yellow }

# Execute remote command
function Invoke-RemoteCommand {
    param(
        [string]$Command,
        [switch]$Interactive
    )
    
    $fullCommand = "cd $REMOTE_DIR && $Command"
    Write-Info "Executing: $Command"
    
    if ($Interactive) {
        ssh -t $SSH_HOST $fullCommand
    } else {
        ssh $SSH_HOST $fullCommand
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Command completed successfully"
        return $true
    } else {
        Write-Error "Command failed with exit code: $LASTEXITCODE"
        return $false
    }
}

# Git Operations
function Invoke-GitPull {
    Write-Info "Pulling latest changes from GitHub..."
    Invoke-RemoteCommand "git pull origin main"
}

function Invoke-GitStatus {
    Write-Info "Checking git status..."
    Invoke-RemoteCommand "git status"
}

function Invoke-GitLog {
    param([int]$Count = 5)
    Write-Info "Showing last $Count commits..."
    Invoke-RemoteCommand "git log --oneline -n $Count"
}

function Invoke-GitDiff {
    Write-Info "Showing uncommitted changes..."
    Invoke-RemoteCommand "git diff"
}

function Invoke-GitPush {
    param([string]$Branch = "main")
    Write-Warning "Pushing changes to GitHub..."
    $confirm = Read-Host "Are you sure you want to push to $Branch? (yes/no)"
    if ($confirm -eq "yes") {
        Invoke-RemoteCommand "git push origin $Branch"
    } else {
        Write-Info "Push cancelled"
    }
}

# File Operations
function Get-RemoteFileContent {
    param([string]$FilePath)
    Write-Info "Reading file: $FilePath"
    Invoke-RemoteCommand "cat $FilePath"
}

function Get-RemoteDirectoryListing {
    param([string]$Path = ".")
    Write-Info "Listing directory: $Path"
    Invoke-RemoteCommand "ls -lah $Path"
}

function Find-RemoteFiles {
    param(
        [string]$Pattern,
        [string]$Path = "."
    )
    Write-Info "Searching for: $Pattern"
    Invoke-RemoteCommand "find $Path -name '$Pattern'"
}

function Search-RemoteContent {
    param(
        [string]$Pattern,
        [string]$FilePattern = "*"
    )
    Write-Info "Searching content for: $Pattern"
    Invoke-RemoteCommand "grep -r '$Pattern' --include='$FilePattern' ."
}

# Analysis Operations
function Get-RepoAnalysis {
    Write-Info "Analyzing repository structure..."
    Invoke-RemoteCommand @"
echo '=== Repository Structure ===' && \
tree -L 2 -I '__pycache__|*.pyc|node_modules' || find . -maxdepth 2 -type d && \
echo '' && \
echo '=== File Count by Type ===' && \
find . -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -10 && \
echo '' && \
echo '=== Repository Size ===' && \
du -sh . && \
echo '' && \
echo '=== Git Branch Info ===' && \
git branch -a && \
echo '' && \
echo '=== Recent Activity ===' && \
git log --oneline -n 5
"@
}

function Get-PythonEnvironment {
    Write-Info "Checking Python environment..."
    Invoke-RemoteCommand @"
python --version && \
echo '' && \
echo '=== Installed Packages ===' && \
pip list | head -20 && \
echo '... (showing first 20)'
"@
}

# Issue Detection
function Test-CodeIssues {
    Write-Info "Running code quality checks..."
    Invoke-RemoteCommand @"
echo '=== Syntax Errors ===' && \
find . -name '*.py' -exec python -m py_compile {} \; 2>&1 | grep -i error || echo 'No syntax errors found' && \
echo '' && \
echo '=== TODO/FIXME Comments ===' && \
grep -rn 'TODO\|FIXME' --include='*.py' . | head -10 || echo 'None found'
"@
}

# Safe File Operations
function Copy-LocalToRemote {
    param(
        [string]$LocalPath,
        [string]$RemotePath
    )
    Write-Warning "Copying $LocalPath to remote: $RemotePath"
    $confirm = Read-Host "Confirm copy operation? (yes/no)"
    if ($confirm -eq "yes") {
        scp $LocalPath "${SSH_HOST}:${REMOTE_DIR}/${RemotePath}"
        if ($LASTEXITCODE -eq 0) {
            Write-Success "File copied successfully"
        } else {
            Write-Error "Copy failed"
        }
    }
}

function Copy-RemoteToLocal {
    param(
        [string]$RemotePath,
        [string]$LocalPath
    )
    Write-Info "Copying remote file: $RemotePath to $LocalPath"
    scp "${SSH_HOST}:${REMOTE_DIR}/${RemotePath}" $LocalPath
    if ($LASTEXITCODE -eq 0) {
        Write-Success "File downloaded successfully"
    } else {
        Write-Error "Download failed"
    }
}

# Interactive Shell
function Start-RemoteShell {
    Write-Info "Starting interactive SSH session..."
    Write-Warning "You will be in: $REMOTE_DIR"
    ssh -t $SSH_HOST "cd $REMOTE_DIR && exec bash -l"
}

# Testing Operations
function Invoke-RemoteTests {
    param([string]$TestPath = ".")
    Write-Info "Running tests..."
    Invoke-RemoteCommand "python -m pytest $TestPath -v"
}

# Backup before operations
function Backup-RemoteRepo {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    Write-Info "Creating backup: backup_$timestamp"
    Invoke-RemoteCommand "cd .. && tar -czf backup_${timestamp}.tar.gz $(basename $REMOTE_DIR) && echo 'Backup created successfully'"
}

# Main Menu
function Show-Menu {
    Write-Host "`n╔════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║  Lightning AI SSH Manager             ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host "`nGit Operations:" -ForegroundColor Yellow
    Write-Host "  1. Pull from GitHub"
    Write-Host "  2. Check Git Status"
    Write-Host "  3. View Git Log"
    Write-Host "  4. View Git Diff"
    Write-Host "  5. Push to GitHub"
    Write-Host "`nAnalysis:" -ForegroundColor Yellow
    Write-Host "  6. Analyze Repository"
    Write-Host "  7. Check Python Environment"
    Write-Host "  8. Detect Code Issues"
    Write-Host "`nFile Operations:" -ForegroundColor Yellow
    Write-Host "  9. List Directory"
    Write-Host " 10. Read File"
    Write-Host " 11. Search Files"
    Write-Host " 12. Search Content"
    Write-Host "`nAdvanced:" -ForegroundColor Yellow
    Write-Host " 13. Download File"
    Write-Host " 14. Upload File"
    Write-Host " 15. Create Backup"
    Write-Host " 16. Run Tests"
    Write-Host " 17. Interactive Shell"
    Write-Host " 18. Custom Command"
    Write-Host "`n  0. Exit" -ForegroundColor Red
    Write-Host ""
}

# Main execution
if ($args.Count -eq 0) {
    # Interactive mode
    while ($true) {
        Show-Menu
        $choice = Read-Host "Select option"
        
        switch ($choice) {
            "1" { Invoke-GitPull }
            "2" { Invoke-GitStatus }
            "3" { Invoke-GitLog }
            "4" { Invoke-GitDiff }
            "5" { Invoke-GitPush }
            "6" { Get-RepoAnalysis }
            "7" { Get-PythonEnvironment }
            "8" { Test-CodeIssues }
            "9" { 
                $path = Read-Host "Enter path (. for current)"
                Get-RemoteDirectoryListing $path 
            }
            "10" { 
                $file = Read-Host "Enter file path"
                Get-RemoteFileContent $file 
            }
            "11" { 
                $pattern = Read-Host "Enter filename pattern (e.g., *.py)"
                Find-RemoteFiles $pattern 
            }
            "12" { 
                $pattern = Read-Host "Enter search text"
                Search-RemoteContent $pattern 
            }
            "13" { 
                $remote = Read-Host "Remote file path"
                $local = Read-Host "Local destination"
                Copy-RemoteToLocal $remote $local 
            }
            "14" { 
                $local = Read-Host "Local file path"
                $remote = Read-Host "Remote destination"
                Copy-LocalToRemote $local $remote 
            }
            "15" { Backup-RemoteRepo }
            "16" { Invoke-RemoteTests }
            "17" { Start-RemoteShell; break }
            "18" { 
                $cmd = Read-Host "Enter command"
                Invoke-RemoteCommand $cmd 
            }
            "0" { Write-Info "Goodbye!"; break }
            default { Write-Error "Invalid option" }
        }
        
        if ($choice -ne "17" -and $choice -ne "0") {
            Read-Host "`nPress Enter to continue"
        } else {
            break
        }
    }
} else {
    # Command line mode
    $command = $args -join " "
    Invoke-RemoteCommand $command
}
