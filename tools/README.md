# Tools for working on this repository with Claude

Two launchers run Claude on this repository with your own Anthropic API key, so the usage is
billed to that key's account, not to a claude.ai plan.

| Launcher | What it starts | Status on the owner's PC |
| --- | --- | --- |
| `tools/claude-api.ps1` | Claude Code itself (the `claude` command), with the key | Claude Code on this key reports "Credit balance too low", while a direct API call with the same key works (`-Ping`). |
| `tools/agent.ps1` | `tools/agent.py`: a small terminal session built on the Claude Agent SDK, with the key | New. Whether its traffic draws on the $200 credit has to be checked once (below). |

Both read the same saved key. `tools/claude-api.ps1` saves it, encrypted for your Windows account
(DPAPI), in `%APPDATA%\hivemind\anthropic-api-key`, and the way it is sent (API key or bearer
token) in `%APPDATA%\hivemind\anthropic-auth-mode`. Neither script prints the key or writes it to
your user environment.

Remote Control and cloud sessions cannot use an API key: they always run on the claude.ai plan.
Only local runs, such as these two launchers, bill the key.

## Why a second launcher

The credit lists "Agent SDK, API, Batch API, Playground, Managed Agents", not Claude Code. The Agent
SDK runs Claude Code too (the Python package `claude-agent-sdk` starts a Claude Code process and
talks to it over pipes), but marks its requests as SDK traffic: their user agent reads
`claude-cli/<version> (external, sdk-py, agent-sdk/<version>)`. Whether the billing counts that as
"Agent SDK" cannot be seen from this repository; the first short run below shows it.

## Setup on Windows

In PowerShell, in the repository folder, one command per line. Skip the venv line if `.venv`
already exists.

```powershell
git pull
py -3 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,agent]"
.\tools\claude-api.ps1 -Reset -Test
.\tools\claude-api.ps1 -Ping
.\tools\agent.ps1 --check
```

- The `agent` extra installs `claude-agent-sdk` at the exact version the runner was written against
  (see `pyproject.toml`). Its Windows x64 wheel carries its own Claude Code (`claude.exe`, about
  117 MB download), which the SDK uses in preference to the Claude Code you installed. To run your
  installed one instead: `.\tools\agent.ps1 --cli $env:USERPROFILE\.local\bin\claude.exe`.
- `-Reset -Test` is only needed when no key is saved yet. `-Ping` sends one tiny message straight to
  the API (a fraction of a cent) and shows that the key and its credit work outside Claude Code.
- `--check` prints the settings the session would use (model, working folder, CLAUDE.md, permission
  policy, which Claude Code, which credential) without starting Claude Code or calling the API.
- If PowerShell refuses to run scripts, start them as
  `powershell -ExecutionPolicy Bypass -File .\tools\agent.ps1 --check`.

## Using the runner

```powershell
.\tools\agent.ps1 --once "Reply with the single word: ok"
.\tools\agent.ps1
.\tools\agent.ps1 --help
```

- `--once "prompt"` runs one turn and exits. Without it you get an interactive session in the
  repository root: type a message at `you>`, `/exit` quits (or Ctrl+Z then Enter). Claude Code's own
  slash commands, such as `/compact`, are passed on.
- Ctrl+C interrupts the current turn and the session goes on; a second Ctrl+C quits.
- The model is `claude-opus-5-5` (`--model` changes it). The project's `CLAUDE.md` is loaded
  (`setting_sources=["project"]`); your user settings in `~\.claude\settings.json` (their permission
  rules, hooks and MCP servers) are not.
- Permissions: tools that only read files inside the repository (Read, Grep, Glob and similar) run
  without asking. Everything else (Bash, Edit, Write, WebFetch, ...) shows what it would do and asks
  `Allow? [y/n]`; anything but `y` says no. Without a terminal (input redirected) the answer is no.
  Claude Code itself already lets some harmless calls through without asking (some read-only shell
  commands, for example `echo`), so not every Bash call reaches the question.
- Before the first prompt is sent, the runner checks that Claude Code uses the key from the
  environment; if Claude Code reports another credential (such as your claude.ai login), it stops
  before anything is billed.
- `--max-budget-usd 0.50`: once Claude Code's cost estimate for the session passes 50 cents, the
  turn ends with "Reached maximum budget" and later turns make no model calls. The turn that
  crosses the line finishes first, so it can overshoot a little.
- In Windows PowerShell 5.1, double quotes inside a `--once` prompt do not reach Python intact; use
  single quotes inside the prompt, or the interactive session.

## The billing check (once, after setup)

1. In the Anthropic Console, on the key's organization, note the credit balance (Billing) and
   today's usage for the key (Usage).
2. Run one short turn: `.\tools\agent.ps1 --once "Reply with the single word: ok"`.
3. The runner prints the cost of every turn and the session total. These are Claude Code's own
   estimates at list prices. They include Claude Code's small side calls to a Haiku model (for
   example to name the session), and the first turn costs more than later ones of the same size,
   because it writes Claude Code's long prompt into the prompt cache.
4. A few minutes later, look at the Usage page again: the run should appear under the key, as
   `claude-opus-5-5` (and a little Haiku), for about the printed total. Then look at the credit:
   - The credit went down by about that amount: Agent SDK traffic draws on the credit. Use the runner.
   - The usage appears but is charged elsewhere (not to the credit): the credit does not cover this
     traffic either; decide before using it further.
   - The run failed with `billing_error` / "Credit balance too low": the API treats this traffic like
     Claude Code on this credit. The runner cannot change that; the remaining options are the direct
     API, Managed Agents or the claude.ai plan.

The printed numbers are an estimate, not an invoice: the Console is the bill.

## Tests

`tools/agent.py` has offline tests: `.venv\Scripts\python -m pytest tests\test_agent_runner.py`.
With the SDK installed, `.venv\Scripts\python -m pytest -m slow tests\test_agent_runner.py` also runs
the real SDK and its Claude Code end to end against a small fake API on your own machine, with a
made-up key: no model call reaches Anthropic and nothing is billed.
