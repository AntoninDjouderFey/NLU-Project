<#
setup-venv.ps1
Creates a venv using a specific Python version or interpreter.
Interactive by default; can be used non-interactively with -PythonSpecifier.
Examples:
  .\setup-venv.ps1                 
  .\setup-venv.ps1 -PythonSpecifier "3.10"   # uses `py -3.10 -m venv`
  .\setup-venv.ps1 -PythonSpecifier "C:\Python311\python.exe"
#>

param(
    [string]$VenvDir = ".venv",
    [string]$ReqFile = "requirements.txt",
    [string]$PythonSpecifier = ""   # can be version (3.10), "py" style (-3.10), or full path to python.exe
)

$ErrorActionPreference = "Stop"

function Show-Header {
    Write-Host ""
    Write-Host "Setup virtual environment"
    Write-Host "Project: $(Get-Location)"
    Write-Host "Venv dir: $VenvDir"
    Write-Host ""
}

function Detect-PyLauncher {
    return [bool](Get-Command py -ErrorAction SilentlyContinue)
}

function Show-Installed-Pythons {
    if (Detect-PyLauncher) {
        Write-Host "Detected 'py' launcher. Installed interpreters:"
        py -0p 2>$null | ForEach-Object { Write-Host "  $_" }
    }
    else {
        Write-Host "'py' launcher not found. Using 'where python' to locate Python on PATH:"
        where.exe python 2>$null | ForEach-Object { Write-Host "  $_" } 
    }
}

function Resolve-PythonCommand {
    param([string]$specifier)
    if ([string]::IsNullOrWhiteSpace($specifier)) {
        if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
        if (Detect-PyLauncher) { return "py" }
        throw "No 'python' on PATH and no 'py' launcher found. Install Python or pass full path to python.exe."
    }

    # Absolute path to python.exe?
    if ($specifier -match "^[a-zA-Z]:\\.*python(\.exe)?$" -or $specifier -like "*\python.exe") {
        if (Test-Path $specifier) { return $specifier }
        throw "Specified python path not found: $specifier"
    }

    # If specifier like "3.10" -> use py -3.10
    if ($specifier -match '^\d+(\.\d+)?$') {
        if (Detect-PyLauncher) { return "py -$specifier" }
        throw "You requested Python $specifier but 'py' launcher is not available. Provide full path to python.exe instead."
    }

    # If user passed "py -3.10" or similar
    if ($specifier -match '^py\s*-\d+(\.\d+)?$') { return $specifier }

    # Otherwise try to resolve as command on PATH
    if (Get-Command $specifier -ErrorAction SilentlyContinue) { return $specifier }

    throw "Could not interpret Python specifier: $specifier"
}

function Create-Venv {
    param([string]$pythonCmd, [string]$venvDir)
    Write-Host ""
    Write-Host "Creating venv using: $pythonCmd"
    $parts = $pythonCmd -split '\s+'
    $exe = $parts[0]
    $args = @()
    if ($parts.Count -gt 1) { $args += $parts[1..($parts.Count-1)] }
    $args += "-m"; $args += "venv"; $args += $venvDir

    & $exe @args
}

function Get-Venv-PythonPath {
    param([string]$venvDir)
    $p1 = Join-Path $venvDir "Scripts\python.exe"
    $p2 = Join-Path $venvDir "bin\python"
    if (Test-Path $p1) { return (Resolve-Path $p1).Path }
    if (Test-Path $p2) { return (Resolve-Path $p2).Path }
    throw "Cannot find python executable inside venv ($venvDir)."
}

## -------- Script start --------
Show-Header

if (-not $PythonSpecifier) {
    Show-Installed-Pythons
    Write-Host ""
    $inputSpec = Read-Host "Enter Python version (e.g. 3.10), full path to python.exe, or leave empty to use 'python' on PATH"
    if ($inputSpec) { $PythonSpecifier = $inputSpec }
}

try {
    $pythonCmd = Resolve-PythonCommand -specifier $PythonSpecifier
} catch {
    Write-Error $_.Exception.Message
    exit 1
}

if (-not (Test-Path $VenvDir)) {
    try {
        Create-Venv -pythonCmd $pythonCmd -venvDir $VenvDir
        Write-Host "venv created at $VenvDir"
    } catch {
        Write-Error "Failed to create venv: $($_.Exception.Message)"
        exit 2
    }
} else {
    Write-Host "venv already exists at $VenvDir (skipping creation)"
}

try {
    $VenvPython = Get-Venv-PythonPath -venvDir $VenvDir
} catch {
    Write-Error $_.Exception.Message
    exit 3
}

Write-Host ""
Write-Host "Upgrading pip, setuptools and wheel in venv..."
& $VenvPython -m pip install --upgrade pip setuptools wheel

if (Test-Path $ReqFile) {
    Write-Host ""
    Write-Host "Installing dependencies from $ReqFile ..."
    & $VenvPython -m pip install -r $ReqFile
} else {
    Write-Host ""
    Write-Host "$ReqFile not found — creating a default requirements.txt and installing..."
    ##Those are my common requiements
    @"
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
transformers>=4.20
datasets>=2.0
tqdm>=4.60
torch>=1.13
"@ | Out-File -Encoding utf8 $ReqFile

    & $VenvPython -m pip install -r $ReqFile
}

Write-Host ""
Write-Host "Environment ready!"
Write-Host "Activate it in PowerShell with:"
Write-Host "  .\$VenvDir\Scripts\Activate.ps1"
Write-Host "or in cmd.exe with:"
Write-Host "  $VenvDir\Scripts\activate.bat"
Write-Host "You can check the venv python version with:"
Write-Host "  $VenvPython --version"
Write-Host ""
