#requires -Version 7.0
<#
Validate the actual distributable ZIP, without installing it into a DAW.
Use a fresh output directory for each attempt; evidence survives failures.
#>
param(
    [Parameter(Mandatory = $true)] [string]$PluginArchive,
    [Parameter(Mandatory = $true)] [string]$PluginvalPath,
    [Parameter(Mandatory = $true)] [string]$OutputDirectory,
    [string]$ExpectedOrtDll
)

$ErrorActionPreference = "Stop"
$archive = (Resolve-Path -LiteralPath $PluginArchive).Path
$validator = (Resolve-Path -LiteralPath $PluginvalPath).Path
$cmake = (Get-Command cmake -CommandType Application | Select-Object -First 1).Path
$projectRoot = Split-Path -Parent $PSScriptRoot
$evidence = [System.IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $evidence) {
    throw "Use a fresh output directory: $evidence"
}
New-Item -ItemType Directory -Path $evidence | Out-Null
$result = [ordered]@{
    started_utc = [DateTime]::UtcNow.ToString("o")
    os = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
    architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    cpu = $env:PROCESSOR_IDENTIFIER
    archive_sha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
    validator_sha256 = (Get-FileHash -LiteralPath $validator -Algorithm SHA256).Hash
    package_exit_code = $null
    validator_exit_code = $null
    status = "started"
}
$result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evidence "result.json")

# Let .NET quote individual arguments for the target platform. Record the launch
# inputs before starting so setup failures retain the executable and directory.
function Invoke-EvidenceProcess {
    param([string]$Executable, [string[]]$Arguments, [string]$Name, [string]$WorkingDirectory)
    [ordered]@{
        executable = $Executable
        executable_exists = Test-Path -LiteralPath $Executable -PathType Leaf
        arguments = $Arguments
        working_directory = $WorkingDirectory
        working_directory_exists = Test-Path -LiteralPath $WorkingDirectory -PathType Container
    } | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $evidence "$Name-invocation.json")
    $start = [System.Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $Executable
    $start.WorkingDirectory = $WorkingDirectory
    $start.UseShellExecute = $false
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    foreach ($argument in $Arguments) {
        $start.ArgumentList.Add($argument)
    }
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $start
    $stdout = [System.IO.FileStream]::new((Join-Path $evidence "$Name-stdout.log"),
        [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::Read, 1, [System.IO.FileOptions]::Asynchronous)
    $stderr = [System.IO.FileStream]::new((Join-Path $evidence "$Name-stderr.log"),
        [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::Read, 1, [System.IO.FileOptions]::Asynchronous)
    try {
        if (-not $process.Start()) {
            throw "Could not start $Name at $Executable"
        }
        # Drain both pipes concurrently to files, including during noisy tests.
        # With buffering disabled, logs also survive a CI timeout/process kill.
        $stdoutCopy = $process.StandardOutput.BaseStream.CopyToAsync($stdout)
        $stderrCopy = $process.StandardError.BaseStream.CopyToAsync($stderr)
        $process.WaitForExit()
        [void]$stdoutCopy.GetAwaiter().GetResult()
        [void]$stderrCopy.GetAwaiter().GetResult()
        return $process.ExitCode
    } finally {
        $stdout.Dispose()
        $stderr.Dispose()
        $process.Dispose()
    }
}

try {
    $unpacked = Join-Path $evidence "unpacked"
    Expand-Archive -LiteralPath $archive -DestinationPath $unpacked
    $bundle = Join-Path $unpacked "StemgenRT.vst3"
    . (Join-Path $PSScriptRoot 'windows-install-utils.ps1')
    $null = Get-StemgenRTBundleInventory -Bundle $bundle -ProjectRoot $projectRoot
    $packageArguments = @(
        "-DSTEMGENRT_BUNDLE_PATH=$bundle",
        "-P", (Join-Path $projectRoot "cmake/VerifyWindowsBundle.cmake")
    )
    if ($ExpectedOrtDll) {
        $sdkDll = (Resolve-Path -LiteralPath $ExpectedOrtDll).Path
        $packageArguments = @("-DSTEMGENRT_EXPECTED_ORT_DLL=$sdkDll") + $packageArguments
    }
    $result.package_exit_code = Invoke-EvidenceProcess -Executable $cmake `
        -Arguments $packageArguments -Name "package" -WorkingDirectory $evidence
    if ($result.package_exit_code -ne 0) {
        throw "Package inspection failed; see package-stderr.log in $evidence"
    }

    @(Get-ChildItem -LiteralPath $bundle -Recurse -File | ForEach-Object {
        [ordered]@{
            path = [System.IO.Path]::GetRelativePath($bundle, $_.FullName)
            bytes = $_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
        }
    }) | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evidence "bundle-files.json")

    $logs = Join-Path $evidence "pluginval"
    New-Item -ItemType Directory -Path $logs | Out-Null
    $validatorArguments = @(
        "--strictness-level", "5",
        "--sample-rates", "44100,48000,96000",
        "--block-sizes", "64,128,256,512,1024,4096",
        "--timeout-ms", "120000",
        "--output-dir", $logs,
        "--validate", $bundle
    )
    # Keep GUI tests enabled: opening the editor is part of the reported crash
    # scenario. Its separate process isolates a crash from this evidence script.
    $result.validator_exit_code = Invoke-EvidenceProcess -Executable $validator `
        -Arguments $validatorArguments -Name "pluginval" -WorkingDirectory $evidence
    if ($result.validator_exit_code -ne 0) {
        throw "Plugin validation failed; see pluginval logs in $evidence"
    }
    $result.status = "passed"
    Write-Host "Windows package validation passed. Evidence: $evidence"
} catch {
    $result.status = "failed"
    $result.error = $_.Exception.Message
    throw
} finally {
    $result.finished_utc = [DateTime]::UtcNow.ToString("o")
    $result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evidence "result.json")
}
