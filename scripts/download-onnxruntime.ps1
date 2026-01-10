# Download the official ONNX Runtime release for Windows
# Usage: .\scripts\download-onnxruntime.ps1
#        .\scripts\download-onnxruntime.ps1 -GPU   # For CUDA/GPU support

param(
    [switch]$GPU = $false
)

$ErrorActionPreference = "Stop"

$Version = "1.22.0"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$DestDir = Join-Path $ProjectRoot "libs\onnxruntime"

# Detect architecture
$Arch = [System.Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITECTURE")
if ($Arch -eq "AMD64") {
    $Platform = "win-x64"
} elseif ($Arch -eq "ARM64") {
    $Platform = "win-arm64"
    if ($GPU) {
        Write-Host "Warning: GPU/CUDA is not available for ARM64. Downloading CPU version." -ForegroundColor Yellow
        $GPU = $false
    }
} else {
    Write-Host "Unsupported architecture: $Arch" -ForegroundColor Red
    exit 1
}

# Build filename based on GPU flag
if ($GPU) {
    $Filename = "onnxruntime-$Platform-gpu-$Version.zip"
    $VariantName = "GPU (CUDA)"
} else {
    $Filename = "onnxruntime-$Platform-$Version.zip"
    $VariantName = "CPU"
}

$Url = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/$Filename"

Write-Host "Downloading ONNX Runtime $Version ($VariantName) for $Platform..." -ForegroundColor Cyan
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
    if ($GPU) {
        Write-Host ""
        Write-Host "GPU version may not be available. Try without -GPU flag for CPU version:" -ForegroundColor Yellow
        Write-Host "  .\scripts\download-onnxruntime.ps1" -ForegroundColor Gray
    }
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

# Create a marker file to indicate GPU version
if ($GPU) {
    $MarkerPath = Join-Path $DestDir ".gpu_build"
    Set-Content -Path $MarkerPath -Value "CUDA"
    Write-Host ""
    Write-Host "[OK] ONNX Runtime $Version (GPU/CUDA) installed to: $DestDir" -ForegroundColor Green
    Write-Host ""
    Write-Host "Note: GPU inference requires:" -ForegroundColor Yellow
    Write-Host "  - NVIDIA GPU with CUDA support" -ForegroundColor Gray
    Write-Host "  - CUDA Toolkit 12.x installed" -ForegroundColor Gray
    Write-Host "  - cuDNN 9.x installed" -ForegroundColor Gray
    Write-Host "  The plugin will automatically fall back to CPU if CUDA is unavailable." -ForegroundColor Gray
} else {
    # Remove GPU marker if it exists (switching from GPU to CPU)
    $MarkerPath = Join-Path $DestDir ".gpu_build"
    if (Test-Path $MarkerPath) {
        Remove-Item -Path $MarkerPath -Force
    }
    Write-Host ""
    Write-Host "[OK] ONNX Runtime $Version (CPU) installed to: $DestDir" -ForegroundColor Green
    Write-Host ""
    Write-Host "Tip: For NVIDIA GPU acceleration, download the GPU version:" -ForegroundColor Cyan
    Write-Host "  .\scripts\download-onnxruntime.ps1 -GPU" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Contents:" -ForegroundColor Gray
Get-ChildItem -Path $DestDir | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "Now rebuild your project:" -ForegroundColor Cyan
Write-Host "  Remove-Item -Recurse -Force build-release" -ForegroundColor Gray
Write-Host "  cmake --preset release" -ForegroundColor Gray
Write-Host "  cmake --build build-release --config Release" -ForegroundColor Gray
