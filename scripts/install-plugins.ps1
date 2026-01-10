# Install StemgenRT plugins to system plugin directories (Windows)
# Usage: .\install-plugins.ps1 [-Release] [-Config <Release|Debug>]
#
# For local development: .\install-plugins.ps1
# For release builds:    .\install-plugins.ps1 -Release
# Specify config:        .\install-plugins.ps1 -Config Debug

param(
    [switch]$Release,
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

# Determine build directory
if ($Release) {
    $BuildDir = "build-release"
} else {
    $BuildDir = "build"
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$ArtefactsDir = Join-Path $ProjectRoot "$BuildDir\plugin\AudioPlugin_artefacts"

# Plugin paths (MSBuild puts artifacts in a config subdirectory)
$VST3Plugin = Join-Path $ArtefactsDir "$Config\VST3\StemgenRT.vst3"

# System plugin directories (Windows standard locations)
$VST3DestSystem = "C:\Program Files\Common Files\VST3"
$VST3DestUser = Join-Path $env:LOCALAPPDATA "Programs\Common\VST3"

Write-Host "Installing StemgenRT plugins from $BuildDir ($Config)..." -ForegroundColor Cyan
Write-Host ""

# Check if running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

# Choose destination based on privileges
if ($isAdmin) {
    $VST3Dest = $VST3DestSystem
    Write-Host "Running as Administrator - installing to system directory" -ForegroundColor Yellow
} else {
    $VST3Dest = $VST3DestUser
    Write-Host "Running as user - installing to user directory" -ForegroundColor Yellow
    Write-Host "(Run as Administrator to install to $VST3DestSystem)" -ForegroundColor Gray
}

# Install VST3 plugin
if (Test-Path $VST3Plugin) {
    Write-Host "Installing VST3 plugin..."
    
    # Create destination directory if it doesn't exist
    if (-not (Test-Path $VST3Dest)) {
        New-Item -ItemType Directory -Path $VST3Dest -Force | Out-Null
    }
    
    $DestPath = Join-Path $VST3Dest "StemgenRT.vst3"
    
    # Remove existing plugin
    if (Test-Path $DestPath) {
        Remove-Item -Path $DestPath -Recurse -Force
    }
    
    # Copy the plugin (VST3 is a folder on Windows too)
    Copy-Item -Path $VST3Plugin -Destination $DestPath -Recurse
    
    Write-Host "  [OK] Installed to $DestPath" -ForegroundColor Green
} else {
    Write-Host "  [X] VST3 plugin not found at $VST3Plugin" -ForegroundColor Red
    Write-Host "      Make sure you've built the project first:" -ForegroundColor Gray
    Write-Host "        cmake --preset release" -ForegroundColor Gray
    Write-Host "        cmake --build build-release --config $Config" -ForegroundColor Gray
    Write-Host "      Or try a different config: .\install-plugins.ps1 -Config Debug" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Done! You may need to restart your DAW to detect the new plugins." -ForegroundColor Cyan
Write-Host ""
Write-Host "VST3 plugin locations:" -ForegroundColor Gray
Write-Host "  System: $VST3DestSystem" -ForegroundColor Gray
Write-Host "  User:   $VST3DestUser" -ForegroundColor Gray

