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
$cmake = (Get-Command cmake -CommandType Application).Source
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

# Start-Process joins ArgumentList into a command line on Windows. Quote every
# argument so paths with spaces survive. Windows filenames cannot contain ".
function Invoke-EvidenceProcess {
    param([string]$Executable, [string[]]$Arguments, [string]$Name, [string]$WorkingDirectory)
    $quotedArguments = $Arguments | ForEach-Object { '"' + $_ + '"' }
    $process = Start-Process -FilePath $Executable -ArgumentList $quotedArguments `
        -WorkingDirectory $WorkingDirectory -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $evidence "$Name-stdout.log") `
        -RedirectStandardError (Join-Path $evidence "$Name-stderr.log")
    return $process.ExitCode
}

try {
    $unpacked = Join-Path $evidence "unpacked"
    Expand-Archive -LiteralPath $archive -DestinationPath $unpacked
    $bundle = Join-Path $unpacked "StemgenRT.vst3"
    $packageArguments = @(
        "-DSTEMGENRT_BUNDLE_PATH=$bundle",
        "-P", (Join-Path $projectRoot "cmake/VerifyWindowsBundle.cmake")
    )
    if ($ExpectedOrtDll) {
        $sdkDll = (Resolve-Path -LiteralPath $ExpectedOrtDll).Path
        $packageArguments = @("-DSTEMGENRT_EXPECTED_ORT_DLL=$sdkDll") + $packageArguments
    }
    $result.package_exit_code = Invoke-EvidenceProcess $cmake $packageArguments "package" $evidence
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
    # scenario. The validator's default child process isolates a crashing plugin.
    $result.validator_exit_code = Invoke-EvidenceProcess $validator $validatorArguments "pluginval" $evidence
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
