<#
.SYNOPSIS
    Run tools/agent.py (Claude through the Claude Agent SDK) on the key that tools/claude-api.ps1 saved.

.DESCRIPTION
    Reads the key that tools/claude-api.ps1 saved, encrypted for your Windows account (DPAPI), in
    %APPDATA%\hivemind\anthropic-api-key, and the way it is sent, from
    %APPDATA%\hivemind\anthropic-auth-mode. Then it runs .venv\Scripts\python tools\agent.py with
    the arguments you give, with ANTHROPIC_API_KEY set (or ANTHROPIC_AUTH_TOKEN in bearer mode).
    The variable is set only while the runner runs and is removed again afterwards; it is never
    written to your user environment. The key is never printed.

    Arguments go to tools/agent.py unchanged:
        (none)              an interactive session in the repository root
        --once "prompt"     one turn, then exit
        --check             show the settings; no API call, Claude Code is not started
        --help              every option of tools/agent.py
    --check and --help (or -h) also run without a saved key.

    Save a key first with: .\tools\claude-api.ps1 -Reset -Test
    See tools/README.md for the setup and the billing check.

.EXAMPLE
    .\tools\agent.ps1 --check
.EXAMPLE
    .\tools\agent.ps1 --once "Reply with the single word: ok"
.EXAMPLE
    .\tools\agent.ps1
#>
$ErrorActionPreference = 'Stop'

# No param() block: every argument, such as --once or --check, arrives in $args as it was typed.
$passThrough = @($args)

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$runner = Join-Path $PSScriptRoot 'agent.py'
$installHint = '.venv\Scripts\python -m pip install -e ".[agent]"'

if (-not (Test-Path $python)) {
    Write-Host "There is no virtual environment at $python."
    Write-Host 'Create it in the repository root, then install the SDK:'
    Write-Host '    py -3 -m venv .venv'
    Write-Host "    $installHint"
    exit 2
}

# find_spec prints nothing, so this check needs no stderr redirection (which Windows PowerShell 5.1
# turns into an error under $ErrorActionPreference = 'Stop').
& $python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('claude_agent_sdk') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host 'The Claude Agent SDK is not installed in .venv. In the repository root, run:'
    Write-Host "    $installHint"
    exit 2
}

$keyDir = Join-Path $env:APPDATA 'hivemind'
$keyFile = Join-Path $keyDir 'anthropic-api-key'
$modeFile = Join-Path $keyDir 'anthropic-auth-mode'
# These options make no API call, so they run without a saved key.
$noKeyNeeded = @($passThrough | Where-Object { $_ -in '--check', '--help', '-h' }).Count -gt 0

$plain = $null
if (Test-Path $keyFile) {
    $plain = [System.Net.NetworkCredential]::new('', (Get-Content -Path $keyFile | ConvertTo-SecureString)).Password
} elseif (-not $noKeyNeeded) {
    Write-Host "No saved key at $keyFile."
    Write-Host 'Save one first with: .\tools\claude-api.ps1 -Reset -Test'
    exit 2
}
$mode = if (Test-Path $modeFile) { (Get-Content -Path $modeFile -Raw).Trim() } else { 'api-key' }
if ($mode -ne 'bearer') { $mode = 'api-key' }

$previous = @{ ApiKey = $env:ANTHROPIC_API_KEY; AuthToken = $env:ANTHROPIC_AUTH_TOKEN }
$code = 1
try {
    # Only one of the two may be set: ANTHROPIC_AUTH_TOKEN outranks ANTHROPIC_API_KEY in Claude Code.
    Remove-Item Env:\ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:\ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue
    if ($plain) {
        if ($mode -eq 'bearer') { $env:ANTHROPIC_AUTH_TOKEN = $plain } else { $env:ANTHROPIC_API_KEY = $plain }
        Write-Host "Using the key saved by tools\claude-api.ps1, sent as: $mode."
    } else {
        Write-Host 'No saved key: only --check and --help can run.'
    }
    & $python $runner @passThrough
    $code = $LASTEXITCODE
} finally {
    $env:ANTHROPIC_API_KEY = $previous.ApiKey
    $env:ANTHROPIC_AUTH_TOKEN = $previous.AuthToken
    $plain = $null
}
exit $code
