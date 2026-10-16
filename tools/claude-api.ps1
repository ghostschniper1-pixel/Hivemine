<#
.SYNOPSIS
    Start Claude Code in this repository on your own Anthropic API key, with Claude Opus 5.5.

.DESCRIPTION
    The first run asks for your API key (typing is hidden) and saves it encrypted for your Windows
    account only (Windows DPAPI), in %APPDATA%\hivemind\anthropic-api-key. Later runs read it, put it
    in ANTHROPIC_API_KEY for the Claude Code process only (never in your user environment), and start
    Claude Code in the repository folder with the model claude-opus-5-5.

    Usage is billed to the Anthropic Console account that owns the key, not to your claude.ai plan.
    Claude Code asks once whether to use the key: answer yes. Inside Claude Code, /status shows which
    login is active. Remote Control and cloud sessions cannot use an API key; they always use the
    claude.ai plan.

.PARAMETER Reset
    Ask for a new key and replace the saved one.

.PARAMETER Forget
    Delete the saved key and exit.

.PARAMETER Model
    The model to start with. Default: claude-opus-5-5 (needs Claude Code 2.1.280 or newer; run
    "claude update" if it is not accepted).

.EXAMPLE
    .\tools\claude-api.ps1
.EXAMPLE
    .\tools\claude-api.ps1 -Reset
.EXAMPLE
    .\tools\claude-api.ps1 -Forget
#>
[CmdletBinding()]
param(
    [switch]$Reset,
    [switch]$Forget,
    [string]$Model = 'claude-opus-5-5'
)
$ErrorActionPreference = 'Stop'

$keyDir = Join-Path $env:APPDATA 'hivemind'
$keyFile = Join-Path $keyDir 'anthropic-api-key'

if ($Forget) {
    if (Test-Path $keyFile) {
        Remove-Item $keyFile
        Write-Host "Deleted the saved API key ($keyFile)."
    } else {
        Write-Host 'No saved API key to delete.'
    }
    return
}

if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
    throw 'Claude Code (the "claude" command) is not installed or not on PATH. Install it first: https://code.claude.com/docs/en/setup'
}

if ($Reset -or -not (Test-Path $keyFile)) {
    $secure = Read-Host 'Paste your Anthropic API key (input is hidden)' -AsSecureString
    $plain = [System.Net.NetworkCredential]::new('', $secure).Password
    if (-not $plain.StartsWith('sk-ant-')) {
        throw 'That does not look like an Anthropic API key (they start with "sk-ant-"). Nothing was saved.'
    }
    New-Item -ItemType Directory -Force -Path $keyDir | Out-Null
    # Without -Key, ConvertFrom-SecureString encrypts with DPAPI: only this Windows account on this PC can read it.
    $secure | ConvertFrom-SecureString | Set-Content -Path $keyFile -Encoding ASCII
    Write-Host "Saved, encrypted for your Windows account: $keyFile"
} else {
    $secure = Get-Content -Path $keyFile | ConvertTo-SecureString
    $plain = [System.Net.NetworkCredential]::new('', $secure).Password
}

$repo = Split-Path -Parent $PSScriptRoot
$previous = $env:ANTHROPIC_API_KEY
try {
    $env:ANTHROPIC_API_KEY = $plain
    Push-Location $repo
    Write-Host "Starting Claude Code in $repo with $Model on your API key."
    Write-Host 'If it asks whether to use this API key, answer yes. /status shows the active login.'
    & claude --model $Model
} finally {
    Pop-Location
    $env:ANTHROPIC_API_KEY = $previous
    $plain = $null
}
