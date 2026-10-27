# Shared by the installer and its filesystem failure-path tests.
function Get-StemgenRTBundleInventory {
    param([string]$Bundle, [string]$ProjectRoot)
    if (-not (Test-Path -LiteralPath $Bundle -PathType Container)) {
        throw "VST3 bundle not found: $Bundle"
    }
    $model = Get-Content -LiteralPath (Join-Path $ProjectRoot 'model/independent-model.json') -Raw | ConvertFrom-Json
    $partitions = Get-Content -LiteralPath (Join-Path $ProjectRoot 'model/parallel/manifest.json') -Raw | ConvertFrom-Json
    if ($partitions.parent_sha256 -ne $model.onnx_sha256) {
        throw 'Parent and partition manifests disagree'
    }
    foreach ($relative in @('Contents/x86_64-win/StemgenRT.vst3', 'Contents/x86_64-win/onnxruntime.dll',
                           'Contents/Resources/ONNXRuntime-LICENSE.txt', 'Contents/Resources/ONNXRuntime-ThirdPartyNotices.txt',
                           'Contents/Resources/licenses/StemgenRT-MIT.txt', 'Contents/Resources/licenses/AGPL-3.0.txt',
                           'Contents/Resources/licenses/JUCE-NOTICES.txt', 'Contents/Resources/licenses/WEIGHTS-LICENSE.md',
                           'Contents/Resources/licenses/CC-BY-NC-SA-4.0.txt', 'Contents/Resources/licenses/README.txt')) {
        $path = Join-Path $Bundle $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -eq 0) {
            throw "Required plugin/runtime file or notice missing or empty: $path"
        }
    }
    $models = @(@{ path = 'model.onnx'; sha256 = $model.onnx_sha256; bytes = $model.bytes })
    foreach ($branch in $partitions.branches) {
        if ($branch.path -notin @('drums.onnx', 'bass.onnx', 'vocals.onnx')) {
            throw 'Unexpected partition path'
        }
        $models += @{ path = 'parallel/' + $branch.path; sha256 = $branch.sha256; bytes = $branch.bytes }
    }
    if ($models.Count -ne 4 -or (@($partitions.branches.path | Sort-Object -Unique)).Count -ne 3) {
        throw 'Require all three independent partitions'
    }
    foreach ($entry in $models) {
        $path = Join-Path $Bundle ('Contents/Resources/' + $entry.path)
        if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or
            (Get-Item -LiteralPath $path).Length -ne $entry.bytes -or
            (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $entry.sha256) {
            throw "Model identity differs: $path"
        }
    }
    $root = (Get-Item -LiteralPath $Bundle -Force).FullName.TrimEnd('\', '/')
    $items = @((Get-Item -LiteralPath $Bundle -Force)) + @(Get-ChildItem -LiteralPath $Bundle -Recurse -Force)
    if (@($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
        throw 'Install a complete bundle without reparse points'
    }
    # Check every copied file, including the plugin binary, DLL and manifest.
    @($items | Where-Object { -not $_.PSIsContainer } | ForEach-Object {
        $_.FullName.Substring($root.Length + 1) + ':' + (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    } | Sort-Object)
}

function Install-StemgenRTBundle {
    param([string]$Source, [string]$DestinationDirectory, [string]$ProjectRoot)
    $expected = @(Get-StemgenRTBundleInventory -Bundle $Source -ProjectRoot $ProjectRoot)
    $sourcePath = (Get-Item -LiteralPath $Source).FullName.TrimEnd('\', '/')
    $destination = Join-Path ([IO.Path]::GetFullPath($DestinationDirectory)) 'StemgenRT.vst3'
    if ($destination.Equals($sourcePath, [StringComparison]::OrdinalIgnoreCase) -or
        $destination.StartsWith($sourcePath + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'The installation destination must be outside the source bundle'
    }
    New-Item -ItemType Directory -Path $DestinationDirectory -Force -ErrorAction Stop | Out-Null
    $stagingRoot = Join-Path $DestinationDirectory ('.stemgenrt-install-' + [guid]::NewGuid().ToString('N'))
    $backupRoot = Join-Path $DestinationDirectory ('.stemgenrt-backup-' + [guid]::NewGuid().ToString('N'))
    $staged = Join-Path $stagingRoot 'StemgenRT.vst3'
    $backup = Join-Path $backupRoot 'StemgenRT.vst3'
    $previousMoved = $false
    try {
        New-Item -ItemType Directory -Path $stagingRoot -ErrorAction Stop | Out-Null
        Copy-Item -LiteralPath $Source -Destination $staged -Recurse -Force -ErrorAction Stop
        $actual = @(Get-StemgenRTBundleInventory -Bundle $staged -ProjectRoot $ProjectRoot)
        if (Compare-Object -ReferenceObject $expected -DifferenceObject $actual) {
            throw 'Staged bundle differs from the verified source'
        }
        if (Test-Path -LiteralPath $destination) {
            New-Item -ItemType Directory -Path $backupRoot -ErrorAction Stop | Out-Null
            Move-Item -LiteralPath $destination -Destination $backup -ErrorAction Stop
            $previousMoved = $true
        }
        try {
            Move-Item -LiteralPath $staged -Destination $destination -ErrorAction Stop
        } catch {
            $promotionError = $_
            if ($previousMoved) {
                try {
                    Move-Item -LiteralPath $backup -Destination $destination -ErrorAction Stop
                    $previousMoved = $false
                } catch {
                    throw "Installation and rollback failed. The previous bundle is preserved at $backup. $promotionError"
                }
            }
            throw $promotionError
        }
        if ($previousMoved) {
            Remove-Item -LiteralPath $backup -Recurse -Force -ErrorAction Stop
            $previousMoved = $false
        }
    } finally {
        if (Test-Path -LiteralPath $stagingRoot) {
            Remove-Item -LiteralPath $stagingRoot -Recurse -Force
        }
        if (-not $previousMoved -and (Test-Path -LiteralPath $backupRoot)) {
            Remove-Item -LiteralPath $backupRoot -Force
        }
    }
    Write-Host "Installed verified bundle: $destination"
}
