# Download the official CPU ONNX Runtime release for Windows.
# Usage: .\scripts\download-onnxruntime.ps1

$ErrorActionPreference = "Stop"

$Version = "1.26.0"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$DestDir = Join-Path $ProjectRoot "libs\onnxruntime"

# Detect architecture
$Arch = [System.Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITECTURE")
if ($Arch -eq "AMD64") {
    $Platform = "win-x64"
} elseif ($Arch -eq "ARM64") {
    $Platform = "win-arm64"
} else {
    Write-Host "Unsupported architecture: $Arch" -ForegroundColor Red
    exit 1
}

$Filename = "onnxruntime-$Platform-$Version.zip"

$Url = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/$Filename"

Write-Host "Downloading ONNX Runtime $Version (CPU) for $Platform..." -ForegroundColor Cyan
Write-Host "URL: $Url" -ForegroundColor Gray

# Clean up existing installation
if (Test-Path $DestDir) {
    Write-Host "Removing existing ONNX Runtime installation..."
    Remove-Item -Path $DestDir -Recurse -Force
}

# Create destination directory
New-Item -ItemType Directory -Path $DestDir -Force | Out-Null

# Download
$ZipPath = Join-Path $DestDir $Filename
Write-Host "Downloading to $ZipPath..."
try {
    Invoke-WebRequest -Uri $Url -OutFile $ZipPath
} catch {
    Write-Host "Download failed: $_" -ForegroundColor Red
    exit 1
}

# Extract
Write-Host "Extracting..."
$TempExtractDir = Join-Path $DestDir "_temp_extract"
Expand-Archive -Path $ZipPath -DestinationPath $TempExtractDir -Force

# Move contents up one level (strip the top-level folder)
$ExtractedFolder = Get-ChildItem -Path $TempExtractDir -Directory | Select-Object -First 1
Get-ChildItem -Path $ExtractedFolder.FullName | Move-Item -Destination $DestDir -Force

# Cleanup
Remove-Item -Path $TempExtractDir -Recurse -Force
Remove-Item -Path $ZipPath -Force

$MarkerPath = Join-Path $DestDir ".gpu_build"
if (Test-Path $MarkerPath) {
    Remove-Item -Path $MarkerPath -Force
}
Write-Host ""
Write-Host "[OK] ONNX Runtime $Version (CPU) installed to: $DestDir" -ForegroundColor Green

Write-Host ""
Write-Host "Contents:" -ForegroundColor Gray
Get-ChildItem -Path $DestDir | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "Now rebuild your project:" -ForegroundColor Cyan
Write-Host "  Remove-Item -Recurse -Force build-release" -ForegroundColor Gray
Write-Host "  cmake --preset release" -ForegroundColor Gray
Write-Host "  cmake --build build-release --config Release" -ForegroundColor Gray
