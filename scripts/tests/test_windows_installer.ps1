param([string]$TestRoot = (Join-Path ([IO.Path]::GetTempPath()) ('stemgenrt-install-test-' + [guid]::NewGuid().ToString('N'))))
$ErrorActionPreference = 'Stop'
. (Join-Path (Split-Path -Parent $PSScriptRoot) 'windows-install-utils.ps1')

# Inject failures in the same filesystem operations the installer uses. Tests
# operate only on tiny synthetic bundles under a new temporary directory.
$script:FaultMode = ''
function Copy-Item {
    [CmdletBinding()]
    param([string]$LiteralPath, [string]$Destination, [switch]$Recurse, [switch]$Force)
    if ($script:FaultMode -eq 'copy') { throw 'injected copy failure' }
    Microsoft.PowerShell.Management\Copy-Item @PSBoundParameters
    if ($script:FaultMode -eq 'staged_binary') {
        Set-Content -LiteralPath (Join-Path $Destination 'Contents/x86_64-win/StemgenRT.vst3') -Value 'corrupt copy'
    }
}
function Move-Item {
    [CmdletBinding()]
    param([string]$LiteralPath, [string]$Destination)
    if ($script:FaultMode -in @('promotion', 'rollback') -and $LiteralPath.Contains('.stemgenrt-install-')) {
        throw 'injected promotion failure'
    }
    if ($script:FaultMode -eq 'rollback' -and $LiteralPath.Contains('.stemgenrt-backup-')) {
        throw 'injected rollback failure'
    }
    Microsoft.PowerShell.Management\Move-Item @PSBoundParameters
}
function Assert-True($condition, $message) {
    if (-not $condition) { throw $message }
}
if (Test-Path -LiteralPath $TestRoot) { throw 'Use a new test allocation' }
New-Item -ItemType Directory -Path $TestRoot | Out-Null
try {
    foreach ($case in @('fresh', 'upgrade', 'missing_bundle', 'missing_runtime', 'missing_notice', 'corrupt_model', 'copy', 'staged_binary', 'promotion', 'rollback')) {
        $script:FaultMode = ''
        $root = Join-Path $TestRoot $case
        $source = Join-Path $root 'source/StemgenRT.vst3'
        $dest = Join-Path $root 'installed'
        $installed = Join-Path $dest 'StemgenRT.vst3'
        foreach ($relative in @('source/StemgenRT.vst3/Contents/x86_64-win', 'source/StemgenRT.vst3/Contents/Resources/parallel', 'model/parallel')) {
            New-Item -ItemType Directory -Path (Join-Path $root $relative) -Force | Out-Null
        }
        Set-Content -LiteralPath (Join-Path $source 'Contents/x86_64-win/StemgenRT.vst3') -Value 'new plugin'
        Set-Content -LiteralPath (Join-Path $source 'Contents/x86_64-win/onnxruntime.dll') -Value 'runtime'
        Set-Content -LiteralPath (Join-Path $source 'Contents/Resources/ONNXRuntime-LICENSE.txt') -Value 'runtime license'
        Set-Content -LiteralPath (Join-Path $source 'Contents/Resources/ONNXRuntime-ThirdPartyNotices.txt') -Value 'runtime notices'
        $records = @()
        foreach ($stem in @('model', 'drums', 'bass', 'vocals')) {
            $relative = if ($stem -eq 'model') { 'model.onnx' } else { 'parallel/' + $stem + '.onnx' }
            $path = Join-Path $source ('Contents/Resources/' + $relative)
            Set-Content -LiteralPath $path -Value $stem
            $records += @{ path = $stem + '.onnx'; bytes = (Get-Item -LiteralPath $path).Length; sha256 = (Get-FileHash -LiteralPath $path).Hash }
        }
        @{ onnx_sha256 = $records[0].sha256; bytes = $records[0].bytes } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'model/independent-model.json')
        @{ parent_sha256 = $records[0].sha256; branches = $records[1..3] } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $root 'model/parallel/manifest.json')
        if ($case -ne 'fresh') {
            New-Item -ItemType Directory -Path $installed -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $installed 'old.txt') -Value 'old plugin'
        }
        if ($case -eq 'missing_bundle') { Remove-Item -LiteralPath $source -Recurse -Force }
        if ($case -eq 'missing_runtime') { Remove-Item -LiteralPath (Join-Path $source 'Contents/x86_64-win/onnxruntime.dll') }
        if ($case -eq 'missing_notice') { Remove-Item -LiteralPath (Join-Path $source 'Contents/Resources/ONNXRuntime-ThirdPartyNotices.txt') }
        if ($case -eq 'corrupt_model') { Set-Content -LiteralPath (Join-Path $source 'Contents/Resources/parallel/drums.onnx') -Value 'bad weights' }
        $script:FaultMode = $case
        $failed = $false
        try { Install-StemgenRTBundle -Source $source -DestinationDirectory $dest -ProjectRoot $root }
        catch { $failed = $true }
        if ($case -in @('fresh', 'upgrade')) {
            Assert-True (-not $failed) "$case unexpectedly failed"
            Assert-True (-not (Test-Path -LiteralPath (Join-Path $installed 'old.txt'))) 'Old files leaked into the new bundle'
            $null = Get-StemgenRTBundleInventory -Bundle $installed -ProjectRoot $root
        } elseif ($case -eq 'rollback') {
            Assert-True $failed 'Rollback failure was not reported'
            $saved = @(Get-ChildItem -LiteralPath $dest -Directory -Force -Filter '.stemgenrt-backup-*')
            Assert-True ($saved.Count -eq 1) 'Previous bundle backup was lost'
            Assert-True (Test-Path -LiteralPath (Join-Path $saved[0].FullName 'StemgenRT.vst3/old.txt')) 'Previous bundle is not recoverable'
        } else {
            Assert-True $failed "$case must fail"
            Assert-True ((Get-Content -LiteralPath (Join-Path $installed 'old.txt') -Raw).Trim() -eq 'old plugin') 'Previous installation changed'
        }
        Assert-True (@(Get-ChildItem -LiteralPath $dest -Force -Filter '.stemgenrt-install-*').Count -eq 0) 'Staging directory leaked'
        if ($case -ne 'rollback') {
            Assert-True (@(Get-ChildItem -LiteralPath $dest -Force -Filter '.stemgenrt-backup-*').Count -eq 0) 'Backup directory leaked'
        }
        Write-Host "PASS: $case"
    }
} finally {
    $script:FaultMode = ''
    Remove-Item -LiteralPath $TestRoot -Recurse -Force
}
