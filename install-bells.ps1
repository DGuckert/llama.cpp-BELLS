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

# Check Python — try py launcher first (always real on Windows), then python3, then python.
# The Windows Store stubs for python/python3 resolve via Get-Command but fail at runtime.
$pyExe = $null
foreach ($candidate in @("py", "python3", "python")) {
    $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($cmd) {
        try {
            $testVer = & $cmd.Source --version 2>&1
            if ($testVer -match "Python \d+\.\d+") {
                $pyExe = $cmd.Source
                break
            }
        } catch {}
    }
}
if (-not $pyExe) {
    Write-Host "[BELLS] Python 3.10+ required. Install from https://python.org" -ForegroundColor Red
    exit 1
}

$ver = & $pyExe --version 2>&1
Write-Host "[BELLS] Found $ver" -ForegroundColor Green

# Check if we're inside the repo already (MyCommand.Path is null when run via iex)
$scriptPath = $MyInvocation.MyCommand.Path
$repoDir = $null
if ($scriptPath) {
    $scriptDir = Split-Path -Parent $scriptPath
    if (Test-Path (Join-Path $scriptDir "tools\bells-manager\install.py")) {
        $repoDir = $scriptDir
    }
}

if (-not $repoDir) {
    $repoDir = Join-Path $env:USERPROFILE "llama.cpp-BELLS"
    if (-not (Test-Path $repoDir)) {
        Write-Host "[BELLS] Cloning repository..." -ForegroundColor Yellow
        git clone https://github.com/DGuckert/llama.cpp-BELLS.git $repoDir
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[BELLS] Git clone failed. Install git: https://git-scm.com" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "[BELLS] Updating repository at $repoDir..." -ForegroundColor Yellow
        try { git -C $repoDir fetch origin 2>&1 | Out-Null } catch {}
        try { git -C $repoDir checkout master 2>&1 | Out-Null } catch {}
        try { git -C $repoDir reset --hard origin/master 2>&1 | Out-Null } catch {}
    }
}

# Run the Python installer
$installer = Join-Path $repoDir "tools\bells-manager\install.py"
Write-Host "[BELLS] Running installer..." -ForegroundColor Yellow
& $pyExe $installer

Write-Host ""
Write-Host "[BELLS] Done! Run 'bells' to start." -ForegroundColor Green
