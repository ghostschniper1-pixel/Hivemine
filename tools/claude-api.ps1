<#
.SYNOPSIS
    Start Claude Code in this repository on your own Anthropic key, with Claude Opus 5.5.

.DESCRIPTION
    The first run asks for your key (typing is hidden) and saves it encrypted for your Windows account
    only (Windows DPAPI), in %APPDATA%\hivemind\. Later runs read it and start Claude Code in the
    repository folder with the model claude-opus-5-5. The key is set for the Claude Code process only,
    never in your user environment.

    A key is sent one of two ways: as an API key (ANTHROPIC_API_KEY, the x-api-key header; regular
    Console keys, "sk-ant-api...") or as a bearer token (ANTHROPIC_AUTH_TOKEN, the Authorization
    header). -Test tries the saved key both ways against the API's model list, which costs nothing,
    and says which works. -Bearer makes Claude Code use the bearer form.

    Usage is billed to the Anthropic account that owns the key, not to your claude.ai plan. Claude Code
    may ask once whether to use the key: answer yes. Inside Claude Code, /status shows which login is
    active. Remote Control and cloud sessions cannot use a key; they always use the claude.ai plan.

.PARAMETER Reset
    Ask for a new key and replace the saved one.

.PARAMETER Bearer
    Send the key as a bearer token (ANTHROPIC_AUTH_TOKEN) from now on. Use it when -Test says only
    the bearer form works. Give it together with -Reset, or on its own to switch a saved key.

.PARAMETER ApiKey
    Send the key as an API key (ANTHROPIC_API_KEY) from now on (the default).

.PARAMETER Test
    Check the saved key both ways against the model list (no tokens used), then exit.

.PARAMETER Forget
    Delete the saved key and exit.

.PARAMETER Model
    The model to start with. Default: claude-opus-5-5 (needs Claude Code 2.1.280 or newer; run
    "claude update" if it is not accepted).

.EXAMPLE
    .\tools\claude-api.ps1 -Reset -Test
.EXAMPLE
    .\tools\claude-api.ps1 -Bearer
.EXAMPLE
    .\tools\claude-api.ps1
#>
[CmdletBinding()]
param(
    [switch]$Reset,
    [switch]$Bearer,
    [switch]$ApiKey,
    [switch]$Test,
    [switch]$Forget,
    [string]$Model = 'claude-opus-5-5'
)
$ErrorActionPreference = 'Stop'

$keyDir = Join-Path $env:APPDATA 'hivemind'
$keyFile = Join-Path $keyDir 'anthropic-api-key'
$modeFile = Join-Path $keyDir 'anthropic-auth-mode'

if ($Forget) {
    foreach ($f in $keyFile, $modeFile) { if (Test-Path $f) { Remove-Item $f } }
    Write-Host 'Deleted the saved key (if there was one).'
    return
}

if ($Reset -or -not (Test-Path $keyFile)) {
    $secure = Read-Host 'Paste your Anthropic key (input is hidden)' -AsSecureString
    $plain = [System.Net.NetworkCredential]::new('', $secure).Password.Trim()
    if ($plain.StartsWith('sk-ant-admin')) {
        throw 'That is an Admin API key: it manages an organization but cannot run Claude. Nothing was saved.'
    }
    if (-not $plain.StartsWith('sk-ant-')) {
        throw 'That does not look like an Anthropic key (they start with "sk-ant-"). Nothing was saved.'
    }
    New-Item -ItemType Directory -Force -Path $keyDir | Out-Null
    # Without -Key, ConvertFrom-SecureString encrypts with DPAPI: only this Windows account on this PC can read it.
    ConvertTo-SecureString $plain -AsPlainText -Force | ConvertFrom-SecureString | Set-Content -Path $keyFile -Encoding ASCII
    Write-Host "Saved, encrypted for your Windows account: $keyFile"
} else {
    $plain = [System.Net.NetworkCredential]::new('', (Get-Content -Path $keyFile | ConvertTo-SecureString)).Password
}

if ($Bearer -and $ApiKey) { throw 'Choose -Bearer or -ApiKey, not both.' }
if ($Bearer) { Set-Content -Path $modeFile -Value 'bearer' -Encoding ASCII }
if ($ApiKey) { Set-Content -Path $modeFile -Value 'api-key' -Encoding ASCII }
$mode = if (Test-Path $modeFile) { (Get-Content -Path $modeFile -Raw).Trim() } else { 'api-key' }

if ($Test) {
    $url = 'https://api.anthropic.com/v1/models'
    $tries = @(
        @{ Name = 'API key (x-api-key; ANTHROPIC_API_KEY; the default)'; Headers = @{ 'x-api-key' = $plain; 'anthropic-version' = '2023-06-01' } },
        @{ Name = 'bearer token (Authorization; ANTHROPIC_AUTH_TOKEN; use -Bearer)'; Headers = @{ 'Authorization' = "Bearer $plain"; 'anthropic-version' = '2023-06-01'; 'anthropic-beta' = 'oauth-2025-04-20' } }
    )
    foreach ($t in $tries) {
        try {
            $models = (Invoke-RestMethod -Uri $url -Headers $t.Headers -Method Get).data.id
            $opus = if ($models -contains $Model) { "$Model is available" } else { "$Model is NOT in the list" }
            Write-Host "WORKS as $($t.Name): $($models.Count) models; $opus."
        } catch {
            # The API explains the failure in the response body (error.type and error.message).
            $body = if ($_.ErrorDetails -and $_.ErrorDetails.Message) { $_.ErrorDetails.Message } else { '(no response body)' }
            Write-Host "Fails as $($t.Name): $($_.Exception.Message)"
            Write-Host "    server says: $body"
        }
    }
    Write-Host "Saved mode: $mode."
    return
}

if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
    throw 'Claude Code (the "claude" command) is not installed or not on PATH. Install it first: https://code.claude.com/docs/en/setup'
}

$repo = Split-Path -Parent $PSScriptRoot
$previous = @{ ApiKey = $env:ANTHROPIC_API_KEY; AuthToken = $env:ANTHROPIC_AUTH_TOKEN }
try {
    # Only one of the two may be set: ANTHROPIC_AUTH_TOKEN outranks ANTHROPIC_API_KEY in Claude Code.
    if ($mode -eq 'bearer') {
        $env:ANTHROPIC_AUTH_TOKEN = $plain
        Remove-Item Env:\ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
    } else {
        $env:ANTHROPIC_API_KEY = $plain
        Remove-Item Env:\ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue
    }
    Push-Location $repo
    Write-Host "Starting Claude Code in $repo with $Model, key sent as: $mode."
    Write-Host 'If it asks whether to use this key, answer yes. /status shows the active login.'
    & claude --model $Model
} finally {
    Pop-Location
    $env:ANTHROPIC_API_KEY = $previous.ApiKey
    $env:ANTHROPIC_AUTH_TOKEN = $previous.AuthToken
    $plain = $null
}
