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

$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("stemgenrt-ort-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$ZipPath = Join-Path $TempRoot $Filename
Write-Host "Downloading to $ZipPath..."
try {
    Invoke-WebRequest -Uri $Url -OutFile $ZipPath
} catch {
    Write-Host "Download failed: $_" -ForegroundColor Red
    Remove-Item -Path $TempRoot -Recurse -Force
    exit 1
}

# Extract
Write-Host "Extracting..."
$TempExtractDir = Join-Path $TempRoot "extracted"
Expand-Archive -Path $ZipPath -DestinationPath $TempExtractDir -Force

# Validate the complete SDK before replacing an existing installation.
$ExtractedFolder = Join-Path $TempExtractDir "onnxruntime-$Platform-$Version"
if (-not (Test-Path $ExtractedFolder -PathType Container)) {
    Remove-Item -Path $TempRoot -Recurse -Force
    throw "Downloaded archive did not contain the expected SDK directory: $ExtractedFolder"
}

$VersionPath = Join-Path $ExtractedFolder "VERSION_NUMBER"
$InstalledVersion = (Get-Content -Path $VersionPath -Raw).Trim()
if ($InstalledVersion -ne $Version) {
    Remove-Item -Path $TempRoot -Recurse -Force
    throw "Downloaded SDK reports ONNX Runtime $InstalledVersion; expected $Version"
}

$RequiredFiles = @(
    (Join-Path $ExtractedFolder "include\onnxruntime_c_api.h"),
    (Join-Path $ExtractedFolder "lib\onnxruntime.lib"),
    (Join-Path $ExtractedFolder "lib\onnxruntime.dll")
)
foreach ($RequiredFile in $RequiredFiles) {
    if (-not (Test-Path $RequiredFile -PathType Leaf)) {
        Remove-Item -Path $TempRoot -Recurse -Force
        throw "Downloaded SDK is incomplete; missing $RequiredFile"
    }
}

if (Test-Path $DestDir) {
    Write-Host "Removing existing ONNX Runtime installation..."
    Remove-Item -Path $DestDir -Recurse -Force
}
New-Item -ItemType Directory -Path (Split-Path -Parent $DestDir) -Force | Out-Null
Move-Item -Path $ExtractedFolder -Destination $DestDir

# Cleanup
Remove-Item -Path $TempRoot -Recurse -Force

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
Write-Host "  cmake --preset release" -ForegroundColor Gray
Write-Host "  cmake --build --preset release --config Release" -ForegroundColor Gray
