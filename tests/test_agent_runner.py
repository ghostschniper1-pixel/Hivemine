"""Tests of tools/agent.py, the Claude Agent SDK runner, and of tools/agent.ps1.

Everything here runs offline. Most tests use a small stand-in for the ``claude_agent_sdk`` module
(its classes and fields copied from claude-agent-sdk 0.2.168), so they run without the SDK. Tests
that need the real SDK are skipped when it is not installed; with it installed, the slow test runs
the real SDK and its Claude Code against a fake Messages API on 127.0.0.1 with a made-up key.
"""

from __future__ import annotations

import _thread
import argparse
import asyncio
import dataclasses
import gzip
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import tomllib
import types
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
AGENT_PATH = REPO / "tools" / "agent.py"


def _load_agent() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("hivemind_tools_agent", AGENT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


agent = _load_agent()

try:
    import claude_agent_sdk as real_sdk  # noqa: F401
except ImportError:
    real_sdk = None
needs_sdk = pytest.mark.skipif(real_sdk is None, reason="claude-agent-sdk is not installed (pip install -e .[agent])")


# ------------------------------------------------------------------------------------------------
# A stand-in for claude_agent_sdk (names and fields as in 0.2.168)
# ------------------------------------------------------------------------------------------------

def make_fake_sdk(script: list[list[Any]] | None = None, account: Any = "default") -> types.ModuleType:
    sdk = types.ModuleType("fake_claude_agent_sdk")
    sdk.__version__ = "0.0.0-fake"
    sdk.__file__ = "/nowhere/fake_claude_agent_sdk/__init__.py"

    @dataclass
    class TextBlock:
        text: str

    @dataclass
    class ToolUseBlock:
        id: str
        name: str
        input: dict

    @dataclass
    class ToolResultBlock:
        tool_use_id: str
        content: Any = None
        is_error: bool | None = None

    @dataclass
    class ServerToolUseBlock:
        id: str
        name: str
        input: dict

    @dataclass
    class AssistantMessage:
        content: list
        model: str = "claude-opus-5-5"
        parent_tool_use_id: str | None = None
        error: str | None = None
        message_id: str | None = None

    @dataclass
    class UserMessage:
        content: Any
        parent_tool_use_id: str | None = None

    @dataclass
    class SystemMessage:
        subtype: str
        data: dict

    @dataclass
    class ResultMessage:
        subtype: str = "success"
        duration_ms: int = 0
        duration_api_ms: int = 0
        is_error: bool = False
        num_turns: int = 1
        session_id: str = "s"
        total_cost_usd: float | None = None
        errors: list | None = None
        api_error_status: int | None = None
        terminal_reason: str | None = "completed"

    @dataclass
    class StreamEvent:
        uuid: str
        session_id: str
        event: dict
        parent_tool_use_id: str | None = None

    @dataclass
    class RateLimitEvent:
        rate_limit_info: Any
        uuid: str = "u"
        session_id: str = "s"

    @dataclass
    class ConversationResetMessage:
        new_conversation_id: str = "c"
        uuid: str = "u"
        session_id: str = "s"

    @dataclass
    class PermissionResultAllow:
        behavior: str = "allow"
        updated_input: dict | None = None

    @dataclass
    class PermissionResultDeny:
        behavior: str = "deny"
        message: str = ""
        interrupt: bool = False

    @dataclass
    class ToolPermissionContext:
        tool_use_id: str | None = None
        title: str | None = None
        decision_reason: str | None = None
        description: str | None = None

    class ClaudeSDKError(Exception):
        pass

    @dataclass
    class ClaudeAgentOptions:
        model: str | None = None
        cwd: str | None = None
        system_prompt: Any = None
        setting_sources: list | None = None
        permission_mode: str | None = None
        can_use_tool: Any = None
        include_partial_messages: bool = False
        stderr: Any = None
        cli_path: str | None = None
        effort: str | None = None
        max_budget_usd: float | None = None

    class ClaudeSDKClient:
        instances: list = []

        def __init__(self, options=None):
            self.options = options
            self.script = [list(turn) for turn in (script or [])]
            self.prompts: list[str] = []
            self.permission_results: list[Any] = []
            self.interrupts = 0
            self.connected = False
            self.disconnected = False
            ClaudeSDKClient.instances.append(self)

        async def connect(self, prompt=None):
            self.connected = True
            self._interrupted = asyncio.Event()

        async def get_server_info(self):
            info = {"claude_code_version": "9.9.9"}
            if account == "default":
                info["account"] = {"apiKeySource": "ANTHROPIC_API_KEY", "apiProvider": "firstParty"}
            elif account is not None:
                info["account"] = account
            return info

        async def query(self, prompt, session_id="default"):
            self.prompts.append(prompt)

        async def receive_response(self):
            for item in self.script.pop(0):
                if item == "CTRL_C":
                    _thread.interrupt_main()  # as if Ctrl+C were pressed in the terminal
                    await asyncio.wait_for(self._interrupted.wait(), 10)
                elif isinstance(item, tuple) and item[0] == "permission":
                    _, name, tool_input, context = item
                    self.permission_results.append(await self.options.can_use_tool(name, tool_input, context))
                else:
                    yield item

        async def interrupt(self):
            self.interrupts += 1
            self._interrupted.set()

        async def disconnect(self):
            self.disconnected = True

    for value in list(locals().values()):
        if isinstance(value, type):
            setattr(sdk, value.__name__, value)
    return sdk


def args_for(*argv: str) -> argparse.Namespace:
    return agent.parse_args(list(argv))


def run_session(sdk, turns: list[str], *, answers=(), interactive=True):
    out = io.StringIO()
    replies = list(answers)
    asked: list[str] = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        reply = replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    session = agent.Session(sdk, args_for(), out=out, ask=ask, interactive=interactive,
                            env={"ANTHROPIC_API_KEY": "x"}, detach_cli=False)
    results = []
    try:
        session.open()
        for prompt in turns:
            results.append(session.turn(prompt))
    finally:
        session.close()
    return out.getvalue(), session, asked, results


def stream(message_id: str, *pieces: str) -> list[dict]:
    """Raw API stream events for one message whose text arrives in pieces."""
    events = [{"type": "message_start", "message": {"id": message_id}}]
    events += [{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": p}} for p in pieces]
    return events


# ------------------------------------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------------------------------------

def test_defaults():
    args = args_for()
    assert args.once is None and args.check is False
    assert args.model == "claude-opus-5-5"
    assert args.effort is None and args.max_budget_usd is None and args.cli is None


def test_once_and_options():
    args = args_for("--once", "say ok", "--effort", "high", "--max-budget-usd", "0.5", "--cli", "C:/x/claude.exe")
    assert args.once == "say ok" and args.effort == "high" and args.max_budget_usd == 0.5
    assert args.cli == "C:/x/claude.exe"


@pytest.mark.parametrize("argv", [
    ["--once", "x", "--check"],          # one mode at a time
    ["--once", "   "],                   # an empty prompt
    ["--effort", "extreme"],
    ["--max-budget-usd", "0"],
    ["--max-budget-usd", "lots"],
    ["--unknown"],
])
def test_bad_arguments(argv, capsys):
    with pytest.raises(SystemExit):
        agent.parse_args(argv)


def test_option_values():
    async def cb(*_):
        return None
    values = agent.option_values(args_for("--effort", "low", "--max-budget-usd", "2"), cb)
    assert values["model"] == "claude-opus-5-5"
    assert values["cwd"] == str(REPO)
    assert values["setting_sources"] == ["project"]
    assert values["system_prompt"] == {"type": "preset", "preset": "claude_code"}
    assert values["permission_mode"] == "default"
    assert values["can_use_tool"] is cb
    assert values["include_partial_messages"] is True
    assert values["effort"] == "low" and values["max_budget_usd"] == 2.0
    assert "cli_path" not in values and "stderr" not in values


# ------------------------------------------------------------------------------------------------
# Permission policy
# ------------------------------------------------------------------------------------------------

@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "secret.txt").write_text("no\n")
    return root


@pytest.mark.parametrize("tool, tool_input, allow", [
    ("Read", {"file_path": "src/a.py"}, True),
    ("Read", {"file_path": "{root}/src/a.py"}, True),
    ("Read", {"file_path": "{root}/../secret.txt"}, False),
    ("Read", {"file_path": "../secret.txt"}, False),
    ("Read", {"file_path": "{outside}/secret.txt"}, False),
    ("Read", {}, False),
    ("Read", {"file_path": 7}, False),
    ("NotebookRead", {"notebook_path": "src/n.ipynb"}, True),
    ("Grep", {"pattern": "x"}, True),
    ("Grep", {"pattern": "x", "path": "src"}, True),
    ("Grep", {"pattern": "x", "path": "{outside}"}, False),
    ("Glob", {"pattern": "**/*.py"}, True),
    ("Glob", {"pattern": "src/**/*.py", "path": "{root}"}, True),
    ("Glob", {"pattern": "../*"}, False),
    ("Glob", {"pattern": "{outside}/*.txt"}, False),
    ("Glob", {"pattern": "*", "path": "{outside}"}, False),
    ("Glob", {"path": "src"}, False),
    ("LS", {"path": "src"}, True),
    ("ToolSearch", {"query": "select:Read"}, True),
    ("TaskList", {}, True),
    ("Bash", {"command": "ls"}, False),
    ("PowerShell", {"command": "Get-ChildItem"}, False),
    ("Edit", {"file_path": "src/a.py", "old_string": "1", "new_string": "2"}, False),
    ("Write", {"file_path": "src/b.py", "content": ""}, False),
    ("NotebookEdit", {"notebook_path": "src/n.ipynb"}, False),
    ("WebFetch", {"url": "https://example.com"}, False),
    ("WebSearch", {"query": "x"}, False),
    ("Task", {"description": "look around"}, False),
    ("mcp__github__create_issue", {}, False),
])
def test_permission_verdict(repo, tool, tool_input, allow):
    outside = str(repo.parent)
    filled = {k: (v.format(root=repo, outside=outside) if isinstance(v, str) else v) for k, v in tool_input.items()}
    verdict = agent.permission_verdict(tool, filled, repo)
    assert verdict.allow is allow, verdict.reason


def test_read_through_a_link_that_leaves_the_repository(repo):
    link = repo / "src" / "link"
    try:
        link.symlink_to(repo.parent / "secret.txt")
    except (OSError, NotImplementedError):
        pytest.skip("cannot make symbolic links here")
    assert not agent.permission_verdict("Read", {"file_path": str(link)}, repo).allow


def test_verdict_names_the_outside_path(repo):
    verdict = agent.permission_verdict("Read", {"file_path": "../secret.txt"}, repo)
    assert "outside the repository" in verdict.reason and "secret.txt" in verdict.reason


# ------------------------------------------------------------------------------------------------
# Text: tool summaries, permission questions, costs
# ------------------------------------------------------------------------------------------------

def test_summaries(repo):
    assert agent.summarize_tool_call("Bash", {"command": "git status\n  && git log"}, repo) == "Bash: git status && git log"
    assert agent.summarize_tool_call("Read", {"file_path": str(repo / "src" / "a.py")}, repo) == \
        f"Read: {os.path.join('src', 'a.py')}"
    assert agent.summarize_tool_call("Grep", {"pattern": "def ", "path": str(repo / "src")}, repo) == "Grep: def in src"
    assert agent.summarize_tool_call("TodoWrite", {"todos": [1, 2, 3]}, repo) == "TodoWrite: 3 items"
    assert agent.summarize_tool_call("mcp__x__y", {"n": 1, "title": "hello"}, repo) == "mcp__x__y: hello"
    assert agent.summarize_tool_call("ExitPlanMode", {}, repo) == "ExitPlanMode"
    long = agent.summarize_tool_call("Bash", {"command": "echo " + "a" * 500}, repo, width=60)
    assert len(long) == 60 and long.endswith("...")


def test_permission_question_shows_the_whole_command(repo):
    lines = agent.describe_tool_request("Bash", {"command": "git add -A\ngit commit -m x", "description": "Commit"}, repo)
    assert lines[0] == "  ? Claude wants to use Bash"
    assert "      git add -A" in lines and "      git commit -m x" in lines and "    (Commit)" in lines


def test_permission_question_previews_an_edit(repo):
    lines = agent.describe_tool_request(
        "Edit", {"file_path": str(repo / "src" / "a.py"), "old_string": "x = 1", "new_string": "x = 2\ny = 3"}, repo)
    assert f"    file: {os.path.join('src', 'a.py')}" in lines
    assert "    - x = 1" in lines and "    + x = 2" in lines and "    + y = 3" in lines
    many = agent.describe_tool_request("Write", {"file_path": "f", "content": "\n".join(map(str, range(30)))}, repo)
    assert "    file: f (30 lines)" in many and any("more lines" in line for line in many)
    other = agent.describe_tool_request("mcp__x__y", {"b": 1, "a": "z"}, repo)
    assert other[1] == '    input: {"a": "z", "b": 1}'


def test_format_usd():
    assert agent.format_usd(0.006) == "$0.0060"
    assert agent.format_usd(12.34567) == "$12.3457"
    assert agent.format_usd(1234.5) == "$1,234.5000"
    assert agent.format_usd(-1e-12) == "$0.0000"
    assert agent.format_usd(None) == "unknown"
    assert agent.cost_line(0.0122, 0.0183) == "  [cost of this turn: $0.0122; session so far: $0.0183]"
    assert "unknown" in agent.cost_line(None, 0.0)


def test_cost_meter_turns_running_totals_into_turn_costs():
    meter = agent.CostMeter()
    # running totals as Claude Code 2.1.301 reported them for three turns against the fake API
    assert meter.record(0.006) == pytest.approx(0.006)
    assert meter.record(0.01815) == pytest.approx(0.01215)
    assert meter.record(0.0243) == pytest.approx(0.00615)
    assert meter.total == pytest.approx(0.0243)
    assert meter.record(None) is None and meter.total == pytest.approx(0.0243)
    meter.conversation_reset()                       # /clear: the next total starts from zero
    assert meter.record(0.002) == pytest.approx(0.002)
    assert meter.total == pytest.approx(0.0263)
    assert meter.record(0.001) == pytest.approx(0.001)  # a drop without notice also starts a new count
    assert meter.total == pytest.approx(0.0273)


# ------------------------------------------------------------------------------------------------
# Credentials
# ------------------------------------------------------------------------------------------------

def test_credential_mode():
    assert agent.credential_mode({}) is None
    assert agent.credential_mode({"ANTHROPIC_API_KEY": ""}) is None
    assert agent.credential_mode({"ANTHROPIC_API_KEY": "k"}) == "api-key"
    assert agent.credential_mode({"ANTHROPIC_AUTH_TOKEN": "t"}) == "bearer"
    assert agent.credential_mode({"ANTHROPIC_API_KEY": "k", "ANTHROPIC_AUTH_TOKEN": "t"}) == "bearer"


def test_credential_problem():
    # account entries as Claude Code 2.1.301 reports them
    assert agent.credential_problem({"apiKeySource": "ANTHROPIC_API_KEY", "tokenSource": "claude.ai"}, "api-key") is None
    assert agent.credential_problem({"tokenSource": "ANTHROPIC_AUTH_TOKEN"}, "bearer") is None
    assert "nowhere" in agent.credential_problem({"subscriptionType": "Claude API"}, "api-key")
    assert "claude.ai" in agent.credential_problem({"tokenSource": "claude.ai"}, "bearer")
    assert agent.credential_problem(None, "api-key") is None  # cannot tell: no check


# ------------------------------------------------------------------------------------------------
# Keeping Ctrl+C away from Claude Code
# ------------------------------------------------------------------------------------------------

def test_child_process_flags():
    assert agent.child_process_flags("nt") == {"creationflags": 0x08000000}
    assert agent.child_process_flags("posix") == {"start_new_session": True}


@pytest.mark.parametrize("os_name, expected", [
    ("posix", {"stdin": 1, "start_new_session": True}),
    ("nt", {"stdin": 1, "creationflags": 0x08000000 | 0x10}),
])
def test_open_process_is_wrapped_once(os_name, expected):
    calls = []

    async def open_process(command, **kwargs):
        calls.append((command, kwargs))
        return "process"

    fake_anyio = types.SimpleNamespace(open_process=open_process)
    assert agent.keep_ctrl_c_from_cli(fake_anyio, os_name)
    wrapped = fake_anyio.open_process
    assert agent.keep_ctrl_c_from_cli(fake_anyio, os_name)
    assert fake_anyio.open_process is wrapped  # not wrapped twice
    extra = {"creationflags": 0x10} if os_name == "nt" else {}
    assert asyncio.run(wrapped(["claude"], stdin=1, **extra)) == "process"
    assert calls == [(["claude"], expected)]


# ------------------------------------------------------------------------------------------------
# A whole session, with the stand-in SDK
# ------------------------------------------------------------------------------------------------

def test_session_streams_text_once_asks_and_costs_each_turn():
    def build(sdk):
        ctx = sdk.ToolPermissionContext(tool_use_id="t1", decision_reason="Bash needs approval")
        turn1 = [sdk.SystemMessage("init", {"model": "claude-opus-5-5", "claude_code_version": "9.9.9",
                                            "apiKeySource": "ANTHROPIC_API_KEY"})]
        turn1 += [sdk.StreamEvent("u", "s", e) for e in stream("m1", "Hel", "lo there")]
        turn1 += [
            sdk.AssistantMessage([sdk.TextBlock("Hello there")], message_id="m1"),
            sdk.AssistantMessage([sdk.ToolUseBlock("t1", "Bash", {"command": "pytest -q"})], message_id="m1"),
            ("permission", "Bash", {"command": "pytest -q"}, ctx),
            ("permission", "Read", {"file_path": "README.md"}, sdk.ToolPermissionContext()),
            sdk.UserMessage([sdk.ToolResultBlock("t1", "3 passed", False)]),
            sdk.ResultMessage(total_cost_usd=0.0100),
        ]
        turn2 = [
            sdk.AssistantMessage([sdk.TextBlock("Not streamed.")], message_id="m2"),
            sdk.ResultMessage(total_cost_usd=0.0250),
        ]
        return [turn1, turn2]

    sdk = scripted_sdk(build)
    sdk.ClaudeSDKClient.instances.clear()
    out, session, asked, results = run_session(sdk, ["run the tests", "and now?"], answers=["y"])
    client = sdk.ClaudeSDKClient.instances[-1]
    assert client.prompts == ["run the tests", "and now?"]
    assert results == [True, True]
    assert out.count("Hello there") == 1                       # streamed, not printed again
    assert "Not streamed." in out
    assert "  > Bash: pytest -q" in out
    assert "  ? Claude wants to use Bash" in out and "      pytest -q" in out
    assert "    (Claude Code: Bash needs approval)" in out
    assert asked == ["  Allow? [y/n] "]                        # Read inside the repository: no question
    assert [type(r).__name__ for r in client.permission_results] == ["PermissionResultAllow", "PermissionResultAllow"]
    assert "  [cost of this turn: $0.0100; session so far: $0.0100]" in out
    assert "  [cost of this turn: $0.0150; session so far: $0.0250]" in out
    assert "[Claude Code 9.9.9: model claude-opus-5-5, API key source: ANTHROPIC_API_KEY]" in out
    assert client.disconnected


def scripted_sdk(build) -> types.ModuleType:
    """A stand-in SDK whose client plays the turns ``build(sdk)`` returns."""
    sdk = make_fake_sdk()
    turns = build(sdk)
    base = sdk.ClaudeSDKClient

    class Client(base):
        def __init__(self, options=None):
            super().__init__(options)
            self.script = [list(t) for t in turns]

    sdk.ClaudeSDKClient = Client
    return sdk


def test_refusal_is_sent_and_not_reported_twice():
    def build(sdk):
        ctx = sdk.ToolPermissionContext(tool_use_id="t9")
        return [[
            sdk.AssistantMessage([sdk.ToolUseBlock("t9", "Write", {"file_path": "x.txt", "content": "a"})]),
            ("permission", "Write", {"file_path": "x.txt", "content": "a"}, ctx),
            sdk.UserMessage([sdk.ToolResultBlock("t9", "The user declined this tool call.", True)]),
            sdk.UserMessage([sdk.ToolResultBlock("t8", "boom: no such file", True)]),
            sdk.ResultMessage(total_cost_usd=0.001),
        ]]

    sdk = scripted_sdk(build)
    sdk.ClaudeSDKClient.instances.clear()
    out, _, asked, _ = run_session(sdk, ["write it"], answers=["n"])
    result = sdk.ClaudeSDKClient.instances[-1].permission_results[0]
    assert isinstance(result, sdk.PermissionResultDeny) and not result.interrupt
    assert "  [not allowed]" in out
    assert "declined" not in out                       # the refused call's result is not echoed
    assert "  ! a tool did not run or failed: boom: no such file" in out


def test_ctrl_c_at_the_question_refuses_and_interrupts():
    def build(sdk):
        return [[("permission", "Bash", {"command": "rm -rf build"}, sdk.ToolPermissionContext(tool_use_id="t1")),
                 sdk.ResultMessage(is_error=True, subtype="error_during_execution",
                                   terminal_reason="aborted_tools", total_cost_usd=0.002)]]

    sdk = scripted_sdk(build)
    sdk.ClaudeSDKClient.instances.clear()
    out, _, _, results = run_session(sdk, ["clean"], answers=[KeyboardInterrupt()])
    result = sdk.ClaudeSDKClient.instances[-1].permission_results[0]
    assert isinstance(result, sdk.PermissionResultDeny) and result.interrupt
    assert "[not allowed; the turn is interrupted]" in out and "[the turn was interrupted]" in out
    assert results == [False]


def test_without_a_terminal_every_question_is_no():
    def build(sdk):
        return [[("permission", "Edit", {"file_path": "a", "old_string": "1", "new_string": "2"},
                  sdk.ToolPermissionContext(tool_use_id="t1")),
                 sdk.ResultMessage(total_cost_usd=0.0)]]

    sdk = scripted_sdk(build)
    sdk.ClaudeSDKClient.instances.clear()
    out, _, asked, _ = run_session(sdk, ["edit"], answers=[], interactive=False)
    assert asked == []
    assert isinstance(sdk.ClaudeSDKClient.instances[-1].permission_results[0], sdk.PermissionResultDeny)
    assert "no terminal to ask" in out


def test_ctrl_c_during_a_turn_interrupts_it_and_the_session_goes_on():
    def build(sdk):
        return [
            [sdk.AssistantMessage([sdk.TextBlock("working on it")]), "CTRL_C",
             sdk.ResultMessage(is_error=True, subtype="error_during_execution",
                               terminal_reason="aborted_streaming", total_cost_usd=0.003)],
            [sdk.AssistantMessage([sdk.TextBlock("still here")]), sdk.ResultMessage(total_cost_usd=0.005)],
        ]

    sdk = scripted_sdk(build)
    sdk.ClaudeSDKClient.instances.clear()
    out, _, _, results = run_session(sdk, ["long task", "next"])
    assert sdk.ClaudeSDKClient.instances[-1].interrupts == 1
    assert "[Ctrl+C: interrupting the turn; press Ctrl+C again to quit]" in out
    assert "[the turn was interrupted]" in out and "still here" in out
    assert results == [False, True]
    assert "  [cost of this turn: $0.0020; session so far: $0.0050]" in out


def test_errors_are_explained():
    def build(sdk):
        return [[sdk.AssistantMessage([sdk.TextBlock("API Error: Credit balance is too low")], error="billing_error"),
                 sdk.ResultMessage(is_error=True, api_error_status=400, total_cost_usd=0.0)]]

    sdk = scripted_sdk(build)
    out, _, _, results = run_session(sdk, ["hi"])
    assert "Credit balance is too low" in out
    assert "the API refused for billing; see the billing check in tools/README.md" in out
    assert "  ! the turn ended with an error: HTTP 400" in out
    assert results == [False]


def test_a_cleared_conversation_keeps_the_session_total():
    def build(sdk):
        return [[sdk.ResultMessage(total_cost_usd=0.010)],
                [sdk.ConversationResetMessage(), sdk.ResultMessage(total_cost_usd=0.0)],
                [sdk.ResultMessage(total_cost_usd=0.004)]]

    sdk = scripted_sdk(build)
    out, session, _, _ = run_session(sdk, ["a", "/clear", "b"])
    assert session.meter.total == pytest.approx(0.014)
    assert "  [cost of this turn: $0.0040; session so far: $0.0140]" in out


def test_open_stops_when_claude_code_uses_another_credential():
    sdk = make_fake_sdk([], account={"tokenSource": "claude.ai", "apiKeySource": "none"})
    session = agent.Session(sdk, args_for(), out=io.StringIO(), env={"ANTHROPIC_API_KEY": "x"}, detach_cli=False)
    try:
        with pytest.raises(agent.CredentialError, match="not from ANTHROPIC_API_KEY"):
            session.open()
    finally:
        session.close()


# ------------------------------------------------------------------------------------------------
# main(): --check, a missing SDK, missing credentials, --once
# ------------------------------------------------------------------------------------------------

def test_check_with_the_stand_in_sdk(monkeypatch, capsys):
    monkeypatch.setattr(agent, "load_sdk", lambda: make_fake_sdk())
    assert agent.main(["--check", "--effort", "high"], env={"ANTHROPIC_API_KEY": "sk-ant-secret"}) == 0
    out = capsys.readouterr().out
    assert "Claude Agent SDK 0.0.0-fake (this runner was written against" in out
    assert f"Working directory: {REPO}" in out
    assert "Setting sources: ['project'] (CLAUDE.md found" in out
    assert "System prompt: Claude Code's own (preset claude_code)" in out
    assert "Model: claude-opus-5-5" in out and "Effort: high" in out
    assert "Credentials: ANTHROPIC_API_KEY is set" in out
    assert "No API call was made and Claude Code was not started." in out
    assert "sk-ant-secret" not in out
    assert sys.modules.get("fake_claude_agent_sdk") is None


def test_without_the_sdk(monkeypatch, capsys):
    monkeypatch.setattr(agent, "load_sdk", lambda: None)
    assert agent.main(["--check"], env={}) == 2
    assert 'pip install -e ".[agent]"' in capsys.readouterr().out


def test_without_credentials(monkeypatch, capsys):
    monkeypatch.setattr(agent, "load_sdk", lambda: make_fake_sdk())
    assert agent.main(["--once", "hi"], env={}) == 2
    assert "Neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN is set" in capsys.readouterr().out


def test_once(monkeypatch, capsys):
    sdk = scripted_sdk(lambda sdk: [[sdk.AssistantMessage([sdk.TextBlock("ok")]), sdk.ResultMessage(total_cost_usd=0.0061)]])
    sdk.ClaudeSDKClient.instances.clear()
    monkeypatch.setattr(agent, "load_sdk", lambda: sdk)
    monkeypatch.setattr(agent, "keep_ctrl_c_from_cli", lambda *a: True)
    assert agent.main(["--once", "Reply with ok"], env={"ANTHROPIC_API_KEY": "x"}) == 0
    out = capsys.readouterr().out
    assert sdk.ClaudeSDKClient.instances[-1].prompts == ["Reply with ok"]
    assert "Claude Code confirms it uses it (ANTHROPIC_API_KEY)" in out
    assert "  [cost of this turn: $0.0061; session so far: $0.0061]" in out
    assert "Session total: $0.0061." in out and "Usage page" in out


# ------------------------------------------------------------------------------------------------
# The real SDK (skipped without it)
# ------------------------------------------------------------------------------------------------

@needs_sdk
def test_real_sdk_has_what_the_runner_uses():
    sdk = real_sdk
    for name in ("ClaudeSDKClient", "ClaudeAgentOptions", "AssistantMessage", "UserMessage", "SystemMessage",
                 "ResultMessage", "StreamEvent", "RateLimitEvent", "ConversationResetMessage", "TextBlock",
                 "ToolUseBlock", "ToolResultBlock", "ServerToolUseBlock", "PermissionResultAllow",
                 "PermissionResultDeny", "ToolPermissionContext", "ClaudeSDKError"):
        assert hasattr(sdk, name), name
    fields = lambda cls: {f.name for f in dataclasses.fields(cls)}  # noqa: E731
    assert {"total_cost_usd", "is_error", "subtype", "errors", "api_error_status", "terminal_reason"} <= fields(sdk.ResultMessage)
    assert {"content", "error", "message_id", "parent_tool_use_id"} <= fields(sdk.AssistantMessage)
    assert {"event", "parent_tool_use_id"} <= fields(sdk.StreamEvent)
    assert {"message", "interrupt"} <= fields(sdk.PermissionResultDeny)
    assert {"tool_use_id", "title", "decision_reason"} <= fields(sdk.ToolPermissionContext)
    for cls in (sdk.ClaudeSDKClient,):
        for method in ("connect", "query", "receive_response", "interrupt", "disconnect", "get_server_info"):
            assert callable(getattr(cls, method, None)), method
    # every stand-in class carries only fields the real one has
    fake = make_fake_sdk()
    for name in ("TextBlock", "ToolUseBlock", "ToolResultBlock", "ServerToolUseBlock", "AssistantMessage",
                 "UserMessage", "SystemMessage", "ResultMessage", "StreamEvent", "RateLimitEvent",
                 "ConversationResetMessage", "PermissionResultAllow", "PermissionResultDeny",
                 "ToolPermissionContext", "ClaudeAgentOptions"):
        assert fields(getattr(fake, name)) <= fields(getattr(sdk, name)), name


@needs_sdk
def test_real_options_and_check(capsys):
    async def cb(*_):
        return None
    options = real_sdk.ClaudeAgentOptions(**agent.option_values(args_for("--effort", "high"), cb, print))
    assert options.model == "claude-opus-5-5" and options.setting_sources == ["project"]
    assert options.cwd == str(REPO) and options.can_use_tool is cb
    assert agent.main(["--check"], env={}) == 0
    out = capsys.readouterr().out
    assert f"Claude Agent SDK {real_sdk.__version__}" in out and "Claude Code: " in out
    assert "No API call was made" in out


@needs_sdk
def test_real_anyio_open_process_takes_the_flags():
    import inspect

    import anyio
    parameters = inspect.signature(anyio.open_process).parameters
    assert "start_new_session" in parameters and "creationflags" in parameters
    source = inspect.getsource(importlib.import_module("claude_agent_sdk._internal.transport.subprocess_cli"))
    assert "anyio.open_process(" in source  # the call the runner wraps


# ------------------------------------------------------------------------------------------------
# End to end: the real SDK and its Claude Code against a fake Messages API (slow)
# ------------------------------------------------------------------------------------------------

class FakeMessagesAPI(BaseHTTPRequestHandler):
    """Answers /v1/messages like the API: a greeting, or a Write outside the repository."""

    requests: list[dict] = []
    outside_file = ""

    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.headers.get("content-encoding") == "gzip":
            raw = gzip.decompress(raw)
        body = json.loads(raw or b"{}")
        users = [m for m in body.get("messages", []) if m.get("role") == "user"]
        last = json.dumps(users[-1]) if users else ""
        FakeMessagesAPI.requests.append({
            "path": self.path, "model": body.get("model"),
            "x_api_key": self.headers.get("x-api-key"), "authorization": self.headers.get("authorization"),
            "user_agent": self.headers.get("user-agent", ""), "first_user": json.dumps(users[0]) if users else ""})
        if not self.path.startswith("/v1/messages") or "count_tokens" in self.path:
            return self._json({"input_tokens": 10})
        main_model = "haiku" not in str(body.get("model"))
        if '"tool_result"' in last:
            blocks = [{"type": "text", "text": "Done."}]
        elif main_model and "WRITE_OUTSIDE" in last:
            blocks = [{"type": "tool_use", "id": "toolu_1", "name": "Write",
                       "input": {"file_path": FakeMessagesAPI.outside_file, "content": "x"}}]
        else:
            blocks = [{"type": "text", "text": "Hello from the fake API."}]
        stop = "tool_use" if blocks[0]["type"] == "tool_use" else "end_turn"
        usage = {"input_tokens": 1000, "output_tokens": 100, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        message = {"id": f"msg_{len(FakeMessagesAPI.requests)}", "type": "message", "role": "assistant",
                   "model": body.get("model"), "content": blocks, "stop_reason": stop, "stop_sequence": None, "usage": usage}
        if not body.get("stream"):
            return self._json(message)
        events = [{"type": "message_start", "message": {**message, "content": [], "stop_reason": None,
                                                        "usage": {**usage, "output_tokens": 1}}}]
        for i, block in enumerate(blocks):
            if block["type"] == "text":
                events.append({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
                for word in block["text"].split(" "):
                    events.append({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": word + " "}})
            else:
                events.append({"type": "content_block_start", "index": i, "content_block": {**block, "input": {}}})
                events.append({"type": "content_block_delta", "index": i,
                               "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}})
            events.append({"type": "content_block_stop", "index": i})
        events.append({"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 100}})
        events.append({"type": "message_stop"})
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write("".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode())

    def _json(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@needs_sdk
@pytest.mark.slow
def test_end_to_end_against_a_fake_api(tmp_path):
    if agent.bundled_cli(real_sdk) is None and shutil.which("claude") is None:
        pytest.skip("no Claude Code: neither bundled with the SDK nor on PATH")
    FakeMessagesAPI.requests = []
    FakeMessagesAPI.outside_file = str(tmp_path / "must_not_exist.txt")
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeMessagesAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ANTHROPIC_", "CLAUDE", "CCR_")) and k not in ("HOME", "USERPROFILE")}
    env.update({
        "HOME": str(home), "USERPROFILE": str(home), "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{server.server_address[1]}",
        "ANTHROPIC_API_KEY": "sk-ant-api03-made-up-for-a-test",
        "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    })
    try:
        runs = []
        for prompt in ("Say hello", "Please WRITE_OUTSIDE the repository"):
            runs.append(subprocess.run([sys.executable, str(AGENT_PATH), "--once", prompt], env=env,
                                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180))
    finally:
        server.shutdown()
    hello, write = runs
    assert hello.returncode == 0, hello.stdout + hello.stderr
    assert "Claude Code confirms it uses it (ANTHROPIC_API_KEY)" in hello.stdout
    assert hello.stdout.count("Hello from the fake API.") == 1
    assert re.search(r"\[cost of this turn: \$0\.0\d{3}; session so far: \$0\.0\d{3}\]", hello.stdout), hello.stdout
    assert write.returncode == 0, write.stdout + write.stderr
    assert "> Write: " in write.stdout and "no terminal to ask" in write.stdout and "Done." in write.stdout
    assert not Path(FakeMessagesAPI.outside_file).exists()
    model_calls = [r for r in FakeMessagesAPI.requests if r["path"].startswith("/v1/messages")]
    assert model_calls and all(r["x_api_key"] == env["ANTHROPIC_API_KEY"] for r in model_calls)
    assert not any(r["authorization"] for r in model_calls)
    assert any(r["model"] == "claude-opus-5-5" for r in model_calls)
    assert all("sdk-py" in r["user_agent"] for r in model_calls)
    first = next(r for r in model_calls if r["model"] == "claude-opus-5-5")
    assert "Minecraft Hivemind" in first["first_user"]  # the project's CLAUDE.md reached the model


# ------------------------------------------------------------------------------------------------
# tools/agent.ps1 and the other files (static checks: there is no PowerShell here)
# ------------------------------------------------------------------------------------------------

def ps_structure_problems(text: str) -> list[str]:
    """Unbalanced brackets or unterminated strings/comments, PowerShell lexing rules (simplified)."""
    problems: list[str] = []
    stack: list[tuple[str, int]] = []
    pairs = {")": "(", "}": "{", "]": "["}
    i, line, n = 0, 1, len(text)

    def scan_double(start: int) -> int:
        j = start + 1
        nonlocal line
        while j < n:
            c = text[j]
            if c == "\n":
                line += 1
            if c == "`":
                j += 2
                continue
            if c == '"':
                if j + 1 < n and text[j + 1] == '"':
                    j += 2
                    continue
                return j + 1
            if c == "$" and j + 1 < n and text[j + 1] == "(":
                depth, j = 1, j + 2
                while j < n and depth:
                    if text[j] == "(":
                        depth += 1
                    elif text[j] == ")":
                        depth -= 1
                    elif text[j] == "'":
                        j = text.find("'", j + 1)
                        if j < 0:
                            break
                    j += 1
                continue
            j += 1
        problems.append(f"line {line}: unterminated double-quoted string")
        return n

    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
        elif text.startswith("<#", i):
            end = text.find("#>", i + 2)
            if end < 0:
                problems.append(f"line {line}: unterminated block comment")
                break
            line += text.count("\n", i, end)
            i = end + 2
            continue
        elif c == "#":
            end = text.find("\n", i)
            i = n if end < 0 else end
            continue
        elif c == "`":
            i += 2
            continue
        elif c == "'":
            j = i + 1
            while True:
                j = text.find("'", j)
                if j < 0:
                    problems.append(f"line {line}: unterminated single-quoted string")
                    return problems
                if j + 1 < n and text[j + 1] == "'":
                    j += 2
                    continue
                break
            line += text.count("\n", i, j)
            i = j + 1
            continue
        elif c == '"':
            i = scan_double(i)
            continue
        elif c in "({[":
            stack.append((c, line))
        elif c in ")}]":
            if not stack or stack[-1][0] != pairs[c]:
                problems.append(f"line {line}: unexpected {c}")
            else:
                stack.pop()
        i += 1
    problems += [f"line {ln}: {c} is never closed" for c, ln in stack]
    return problems


def ps_code_only(text: str) -> str:
    """The script with comments and string contents removed (for token checks)."""
    text = re.sub(r"<#.*?#>", "", text, flags=re.S)
    text = re.sub(r"'(?:[^']|'')*'", "''", text)
    text = re.sub(r'"(?:[^"`]|`.|"")*"', '""', text)
    return re.sub(r"#[^\n]*", "", text)


PS51_UNSUPPORTED = [  # syntax that Windows PowerShell 5.1 does not have (PowerShell 7 only)
    (r"&&|\|\|", "pipeline chain operators && and ||"),
    (r"\?\?", "null-coalescing ??"),
    (r"\?\.|\?\[", "null-conditional ?. / ?["),
    (r"-Parallel\b", "ForEach-Object -Parallel"),
    (r"\$PSStyle\b", "$PSStyle"),
    (r"\bclean\s*\{", "clean block"),
    (r"ConvertFrom-SecureString\s+[^\n]*-AsPlainText", "ConvertFrom-SecureString -AsPlainText"),
]


@pytest.mark.parametrize("name", ["agent.ps1", "claude-api.ps1"])
def test_powershell_scripts_parse_and_suit_windows_powershell_5_1(name):
    data = (REPO / "tools" / name).read_bytes()
    assert data.isascii(), "Windows PowerShell 5.1 reads a file without BOM as ANSI: keep it ASCII"
    assert b"\r" not in data, "LF line endings (.gitattributes)"
    text = data.decode("ascii")
    assert ps_structure_problems(text) == []
    code = ps_code_only(text)
    for pattern, what in PS51_UNSUPPORTED:
        assert not re.search(pattern, code), f"{what} is not in Windows PowerShell 5.1"


def test_the_structure_check_catches_mistakes():
    assert ps_structure_problems("if ($a) { 'x' ") == ["line 1: { is never closed"]
    assert ps_structure_problems("$x = 'abc") == ["line 1: unterminated single-quoted string"]
    assert ps_structure_problems('Write-Host "a $($b.c) d"') == []
    assert ps_structure_problems("$a = @{ k = 'v' }\n)") == ["line 2: unexpected )"]


def test_agent_ps1_reads_the_saved_key_and_never_prints_it():
    text = (REPO / "tools" / "agent.ps1").read_text()
    launcher = (REPO / "tools" / "claude-api.ps1").read_text()
    for needle in ("Join-Path $env:APPDATA 'hivemind'", "'anthropic-api-key'", "'anthropic-auth-mode'"):
        assert needle in text and needle in launcher  # the same files as tools/claude-api.ps1
    assert "ConvertTo-SecureString" in text and "[System.Net.NetworkCredential]::new" in text
    assert "$env:ANTHROPIC_API_KEY = $plain" in text and "$env:ANTHROPIC_AUTH_TOKEN = $plain" in text
    assert "& $python $runner @passThrough" in text
    assert '.venv\\Scripts\\python -m pip install -e ".[agent]"' in text
    assert "param(" not in ps_code_only(text)  # arguments pass through $args unchanged
    for line in text.splitlines():
        if re.search(r"\b(Write-Host|Write-Output|Write-Verbose|echo)\b", line):
            assert "$plain" not in line, line
    assert re.search(r"finally\s*\{[^}]*\$env:ANTHROPIC_API_KEY = \$previous\.ApiKey", text, re.S)


def test_pyproject_pins_the_sdk_the_runner_was_written_against():
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert project["optional-dependencies"]["agent"] == [f"claude-agent-sdk=={agent.SDK_VERSION_READ}"]
    assert "claude-agent-sdk" not in " ".join(project["dependencies"])  # optional, not for hivemind/


def test_readme_covers_setup_billing_and_remote_sessions():
    text = (REPO / "tools" / "README.md").read_text()
    for needle in ('.venv\\Scripts\\python -m pip install -e ".[dev,agent]"', ".\\tools\\agent.ps1 --check",
                   "The billing check", "Usage page", "Remote Control and cloud sessions cannot use an API key"):
        assert needle in text, needle


def test_hivemind_does_not_import_the_runner():
    # runtime code under hivemind/ stays standard library and NumPy only
    for path in (REPO / "hivemind").rglob("*.py"):
        assert "claude_agent_sdk" not in path.read_text(encoding="utf-8"), path
