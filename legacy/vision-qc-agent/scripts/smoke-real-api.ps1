[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$environmentPath = Join-Path $repositoryRoot ".env"
$apiDirectory = Join-Path $repositoryRoot "apps\api"
$virtualEnvironmentPython = Join-Path $apiDirectory ".venv\Scripts\python.exe"

function Get-ConfiguredAiMode {
    if (-not [string]::IsNullOrWhiteSpace($env:AI_MODE)) {
        return $env:AI_MODE.Trim()
    }

    if (-not (Test-Path -LiteralPath $environmentPath -PathType Leaf)) {
        return ""
    }

    foreach ($line in [IO.File]::ReadLines($environmentPath)) {
        if ($line -match "^\s*AI_MODE\s*=\s*(.*?)\s*$") {
            return $Matches[1].Trim("'`"")
        }
    }

    return ""
}

$aiMode = Get-ConfiguredAiMode
if ($aiMode.ToLowerInvariant() -ne "real") {
    [Console]::Error.WriteLine(
        "Refusing paid API smoke. Set AI_MODE=real in the ignored .env file."
    )
    exit 2
}

if (Test-Path -LiteralPath $virtualEnvironmentPython -PathType Leaf) {
    $python = $virtualEnvironmentPython
}
else {
    $pythonCommand = Get-Command python -ErrorAction Stop
    $python = $pythonCommand.Source
}

Push-Location $apiDirectory
try {
    & $python -m app.real_api_smoke
    $smokeExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $smokeExitCode
