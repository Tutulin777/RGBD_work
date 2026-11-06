param(
    [ValidateSet('cpu', 'cu128')]
    [string]$Torch = 'cpu'
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Creating venv failed. Install Python 3.13 first.' }
}
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $pythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Updating pip failed.' }
& $pythonPath -m pip install "torch==2.10.0+$Torch" "torchvision==0.25.0+$Torch" --index-url "https://download.pytorch.org/whl/$Torch"
if ($LASTEXITCODE -ne 0) { throw 'Installing PyTorch failed.' }
& $pythonPath -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Installing dependencies failed.' }
& $pythonPath -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
& $pythonPath main.py doctor
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
Write-Host 'Ready. See README.md for the small-sample demo and full experiments.'
