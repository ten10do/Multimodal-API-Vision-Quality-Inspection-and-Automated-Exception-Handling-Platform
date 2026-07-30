[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$examplePath = Join-Path $repositoryRoot ".env.example"
$environmentPath = Join-Path $repositoryRoot ".env"
$backupPath = Join-Path $repositoryRoot ".env.backup"
$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)

function ConvertFrom-SecureValue {
    param(
        [Parameter(Mandatory)]
        [Security.SecureString]$Value
    )

    $pointer = [IntPtr]::Zero
    try {
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Value)
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        if ($pointer -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
        }
    }
}

function Set-EnvironmentValue {
    param(
        [Parameter(Mandatory)]
        [string[]]$Lines,

        [Parameter(Mandatory)]
        [string]$Name,

        [Parameter(Mandatory)]
        [string]$Value
    )

    $matches = 0
    $updated = foreach ($line in $Lines) {
        if ($line.StartsWith("$Name=", [StringComparison]::Ordinal)) {
            $matches += 1
            "$Name=$Value"
        }
        else {
            $line
        }
    }

    if ($matches -ne 1) {
        throw ".env.example must contain exactly one $Name setting."
    }

    return $updated
}

function Assert-ValidSecret {
    param(
        [AllowEmptyString()]
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value) -or $Value.Contains("`r") -or $Value.Contains("`n")) {
        throw "A non-empty single-line API key is required."
    }
}

if (-not (Test-Path -LiteralPath $examplePath -PathType Leaf)) {
    throw ".env.example was not found at the repository root."
}

$bailianSecure = $null
$deepseekSecure = $null
$bailianPlain = $null
$deepseekPlain = $null
$lines = $null

try {
    $bailianSecure = Read-Host "Enter BAILIAN_API_KEY" -AsSecureString
    $deepseekSecure = Read-Host "Enter DEEPSEEK_API_KEY" -AsSecureString

    $lines = [IO.File]::ReadAllLines($examplePath, $utf8WithoutBom)
    $lines = Set-EnvironmentValue -Lines $lines -Name "AI_MODE" -Value "real"

    $bailianPlain = ConvertFrom-SecureValue -Value $bailianSecure
    Assert-ValidSecret -Value $bailianPlain
    $lines = Set-EnvironmentValue -Lines $lines -Name "BAILIAN_API_KEY" -Value $bailianPlain
    $bailianPlain = $null

    $deepseekPlain = ConvertFrom-SecureValue -Value $deepseekSecure
    Assert-ValidSecret -Value $deepseekPlain
    $lines = Set-EnvironmentValue -Lines $lines -Name "DEEPSEEK_API_KEY" -Value $deepseekPlain
    $deepseekPlain = $null

    if (Test-Path -LiteralPath $environmentPath -PathType Leaf) {
        Copy-Item -LiteralPath $environmentPath -Destination $backupPath -Force
        Write-Host "Existing .env backed up to ignored .env.backup."
    }

    [IO.File]::WriteAllLines($environmentPath, [string[]]$lines, $utf8WithoutBom)
    Write-Host "Local .env configured for AI_MODE=real. API keys were not displayed."
}
finally {
    $bailianPlain = $null
    $deepseekPlain = $null
    $lines = $null
    if ($null -ne $bailianSecure) {
        $bailianSecure.Dispose()
    }
    if ($null -ne $deepseekSecure) {
        $deepseekSecure.Dispose()
    }
}
