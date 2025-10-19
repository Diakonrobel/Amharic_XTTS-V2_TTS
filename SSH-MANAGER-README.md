# Lightning AI SSH Manager - Quick Reference

## Usage

### Interactive Menu Mode (Recommended)
```powershell
.\lightning-ssh-manager.ps1
```

### Direct Command Mode
```powershell
.\lightning-ssh-manager.ps1 "git status"
.\lightning-ssh-manager.ps1 "ls -la"
.\lightning-ssh-manager.ps1 "python train.py"
```

## Features

### 🔄 Git Operations
- **Pull**: Safely pull latest changes from GitHub
- **Status**: Check what files are modified
- **Log**: View recent commits
- **Diff**: See uncommitted changes
- **Push**: Push with confirmation prompt (safe)

### 📊 Analysis Tools
- **Analyze Repository**: Get complete overview (structure, file types, size, branches)
- **Check Python Environment**: View Python version and installed packages
- **Detect Issues**: Find syntax errors and TODO comments

### 📁 File Operations
- **List Directory**: Browse remote folders
- **Read File**: View file contents
- **Search Files**: Find files by name pattern
- **Search Content**: Grep through code

### 🚀 Advanced Features
- **Download/Upload**: Transfer files safely with SCP
- **Backup**: Create timestamped backup before risky operations
- **Run Tests**: Execute pytest remotely
- **Interactive Shell**: Drop into full SSH session
- **Custom Commands**: Run any command you need

## Safety Features

✅ **Confirmation prompts** for dangerous operations (push, upload)  
✅ **No direct code modification** - work via git workflow  
✅ **Backup function** before major changes  
✅ **Clear status messages** with color coding  
✅ **Error handling** with exit code checks  

## Common Workflows

### 1. Check Remote Status
```powershell
.\lightning-ssh-manager.ps1
# Select: 2 (Git Status)
# Then: 6 (Analyze Repository)
```

### 2. Pull Changes & Test
```powershell
.\lightning-ssh-manager.ps1
# Select: 1 (Pull from GitHub)
# Then: 16 (Run Tests)
```

### 3. Fix Issue Workflow
**On your local machine:**
1. Make changes to code
2. Commit and push to GitHub

**Using this script:**
```powershell
.\lightning-ssh-manager.ps1
# Select: 1 (Pull from GitHub)
# Select: 6 (Analyze Repository) - verify changes
# Select: 16 (Run Tests) - check if it works
```

### 4. Debug Remote Issue
```powershell
.\lightning-ssh-manager.ps1
# Select: 8 (Detect Code Issues) - find problems
# Select: 10 (Read File) - examine specific file
# Select: 12 (Search Content) - find related code
```

### 5. Download Logs/Results
```powershell
.\lightning-ssh-manager.ps1
# Select: 13 (Download File)
# Enter remote path: logs/training.log
# Enter local path: ./local_logs/training.log
```

## Configuration

Edit the script to change:
```powershell
$SSH_HOST = "your_ssh_connection_string"
$REMOTE_DIR = "/your/remote/directory"
```

## Tips

💡 **Always pull before analyzing** to ensure you're looking at latest code  
💡 **Use backup before major operations** (option 15)  
💡 **Check git status** before pushing (option 2 then 5)  
💡 **Use interactive shell** (option 17) for complex debugging  
💡 **Test locally first**, push to GitHub, then pull on Lightning AI  

## Troubleshooting

**SSH Connection Issues:**
```powershell
# Test basic connection
ssh s_01k7x54qcrv1atww40z8bxf9a3@ssh.lightning.ai "pwd"
```

**Permission Denied:**
- Ensure SSH keys are properly configured
- Check Lightning AI session is active

**Command Hangs:**
- Press `Ctrl+C` to cancel
- Avoid commands that open editors (vim, nano)
- Use interactive shell (option 17) for complex operations
