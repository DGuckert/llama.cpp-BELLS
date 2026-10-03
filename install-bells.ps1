# BELLS one-line installer for Windows
# Usage: powershell -ExecutionPolicy Bypass -File install-bells.ps1
# Or:    iex (irm https://raw.githubusercontent.com/DGuckert/llama.cpp-BELLS/bells-next/install-bells.ps1)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "  ____  _____ _     _     ____" -ForegroundColor Cyan
Write-Host " | __ )| ____| |   | |   / ___|" -ForegroundColor Cyan
Write-Host " |  _ \|  _| | |   | |   \___ \" -ForegroundColor Cyan
Write-Host " | |_) | |___| |___| |___ ___) |" -ForegroundColor Cyan
Write-Host " |____/|_____|_____|_____|____/" -ForegroundColor Cyan
Write-Host ""
Write-Host " Per-layer VRAM expert cache for MoE models" -ForegroundColor Gray
Write-Host ""

# Check Python
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    $py = Get-Command python3 -ErrorAction SilentlyContinue
}
if (-not $py) {
    Write-Host "[BELLS] Python 3.10+ required. Install from https://python.org" -ForegroundColor Red
    exit 1
}

$pyExe = $py.Source
$ver = & $pyExe --version 2>&1
Write-Host "[BELLS] Found $ver" -ForegroundColor Green

# Check if we're inside the repo already
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoDir = $scriptDir

if (-not (Test-Path (Join-Path $repoDir "tools\bells-manager\install.py"))) {
    # Not in repo — clone it
    $repoDir = Join-Path $env:USERPROFILE "llama.cpp-BELLS"
    if (-not (Test-Path $repoDir)) {
        Write-Host "[BELLS] Cloning repository..." -ForegroundColor Yellow
        git clone https://github.com/DGuckert/llama.cpp-BELLS.git $repoDir
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[BELLS] Git clone failed. Install git: https://git-scm.com" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "[BELLS] Repository already exists at $repoDir" -ForegroundColor Green
    }
}

# Run the Python installer
$installer = Join-Path $repoDir "tools\bells-manager\install.py"
Write-Host "[BELLS] Running installer..." -ForegroundColor Yellow
& $pyExe $installer

Write-Host ""
Write-Host "[BELLS] Done! Run 'bells' to start." -ForegroundColor Green
