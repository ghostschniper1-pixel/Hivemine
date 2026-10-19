#!/usr/bin/env python3
"""Work on this repository with Claude through the Claude Agent SDK, billed to an Anthropic API key.

On Windows, start it through tools/agent.ps1, which sets the saved key for this process only.
Elsewhere, set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) yourself:

    python tools/agent.py                    an interactive session in the repository root
    python tools/agent.py --once "prompt"    one turn, then exit
    python tools/agent.py --check            show the settings; no API call, Claude Code is not started

How it works. Everything below was read from the source of claude-agent-sdk 0.2.168 (the version
pinned as the "agent" extra in pyproject.toml) and tried with the Claude Code 2.1.301 bundled in
that wheel, against a local fake API (no real calls):

- The SDK runs a Claude Code process and talks to it over pipes (``ClaudeSDKClient``). The wheels
  for Windows x64, macOS and Linux carry their own Claude Code (``claude_agent_sdk/_bundled/``),
  which the SDK prefers to an installed one; ``--cli PATH`` runs another executable instead.
- Settings: ``setting_sources=[]`` loads no settings file, neither yours (~/.claude/settings.json)
  nor the repository's (.claude/settings.json, .mcp.json). With the project's settings loaded,
  Claude Code runs their hooks and apiKeyHelper commands and starts their MCP servers at startup,
  and a hook there can allow a tool call, all without asking anyone. The repository's CLAUDE.md
  is given to Claude Code instead as an addition to its own system prompt
  (``system_prompt={"type": "preset", "preset": "claude_code", "append": ...}``).
- Permissions: Claude Code calls ``can_use_tool`` only when it would ask the user, and in default
  mode it runs many calls without asking (EnterWorktree, Task, Skill, read-only Bash commands such
  as ``git status``). So a PreToolUse hook (``pre_tool_use_decision``) sees every tool call first.
  Reading files inside the repository is left to Claude Code, which allows it; a few calls are
  refused (``refusal_reason``); every other call goes to ``can_use_tool``, and the runner asks
  y/n in the terminal. Without a terminal to ask, it says no.
- What is shown: every character a terminal would not show as written (escape sequences,
  carriage returns, bidirectional overrides and the like) is printed escaped, as \\x1b or \\u202e,
  in streamed text and in the questions alike, so the text of a tool call cannot rewrite the
  question about it. Claude Code 2.1.301 refuses control characters in Bash commands, but not in
  the content of a Write, for example.
- Switched off: background tasks and scheduled prompts (the environment variables in ``CLI_ENV``)
  and the tools in ``SWITCHED_OFF_TOOLS``. The runner can ask about a tool call only while a turn
  of the person's runs, so all work has to happen inside those turns.
- Turns: one reader task shows the messages as they arrive. A turn Claude Code starts by itself
  (a message from another Claude session, for example) is shown as such, with its cost; the
  result of the person's prompt is the one whose ``origin`` is None or human. A permission
  question that comes while no prompt of the person's runs is refused, with a note.
- Ctrl+C interrupts the current turn (``ClaudeSDKClient.interrupt()``) and the session goes on.
  Claude Code ends its whole process on SIGINT, so it is started where the terminal's Ctrl+C does
  not reach it: in its own process session (POSIX) or without the console (Windows).
- Before the first prompt, the runner checks that Claude Code really uses the key from the
  environment (the ``account`` entry of ``get_server_info()``); otherwise it stops, so nothing is
  billed to another login by mistake.
- Cost: ``ResultMessage.total_cost_usd`` is Claude Code's running total for the session (an
  estimate at list prices, including its small side calls to a Haiku model); the cost of a turn
  is the difference between two totals. The bill itself is on the Usage page of the Console.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import importlib
import json
import os
import queue
import re
import sys
import threading
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "claude-opus-5-5"
SDK_VERSION_READ = "0.2.168"  # the claude-agent-sdk version this runner was written against
CLI_VERSION_TRIED = "2.1.301"  # the Claude Code bundled with it, which the statements here were tried on
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")  # claude_agent_sdk.types.EffortLevel
CREATE_NO_WINDOW = 0x08000000  # Windows process creation flag, as subprocess.CREATE_NO_WINDOW
PREVIEW_LINES = 12  # lines of a file edit or a prompt shown when asking for permission ("a" shows all)
PREVIEW_WIDTH = 200  # characters of such a line shown when asking ("a" shows all)
GLOB_EXPANSION_LIMIT = 64  # brace alternatives of a glob pattern worked out before asking instead
# Characters a terminal does not show as written: control characters (C0, DEL, C1; they can move
# the cursor, erase lines or hide text), format characters (Cf: bidirectional overrides,
# zero-width and tag characters), surrogates, and line and paragraph separators. Everything the
# runner prints shows them escaped (\x1b, ‮), apart from newline and tab.
HIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})
_NOT_PLAIN = re.compile(r"[^\t\n\x20-\x7e]")  # what may be hidden: everything but tab, newline, printable ASCII

# Tools that only read files, with the input fields that name what they read. They are allowed
# without asking only when every such path lies inside the repository. Claude Code 2.1.301 has
# only Read of these (it searches through Bash); the others are names that other Claude Code
# builds use, which --cli may run.
READ_ONLY_FILE_TOOLS: dict[str, tuple[str, ...]] = {
    "Read": ("file_path",),
    "NotebookRead": ("notebook_path",),
    "Grep": ("path",),
    "Glob": ("path",),
    "LS": ("path",),
}
PATH_REQUIRED = frozenset({"Read", "NotebookRead"})
# Tools of other Claude Code builds that only read the session's own state (no files, nothing
# sent anywhere). Claude Code 2.1.301 has none of them.
READ_ONLY_SESSION_TOOLS = frozenset({"ToolSearch", "TaskList", "TaskGet", "TaskOutput", "BashOutput"})

# Tools removed from the session (ClaudeAgentOptions.disallowed_tools). All are in Claude Code
# 2.1.301, where each ran without asking. They would start work outside the person's turns
# (CronCreate and the other cron tools, ScheduleWakeup, Workflow: "workflows run in the
# background"), reach other Claude sessions on this machine or in the cloud (SendMessage,
# ListAgents), or leave the branch the session works on (EnterWorktree, ExitWorktree).
# AskUserQuestion is removed because the terminal cannot answer its choices: allowed, it reports
# "The user did not answer the questions"; without it Claude asks in its reply.
SWITCHED_OFF_TOOLS = ("AskUserQuestion", "CronCreate", "CronDelete", "CronList", "EnterWorktree",
                      "ExitWorktree", "ListAgents", "ScheduleWakeup", "SendMessage", "Workflow")
# Environment for Claude Code. With these (read by 2.1.301) Task runs its agent in the foreground,
# Bash loses run_in_background, and the cron tools are gone.
CLI_ENV = {"CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1", "CLAUDE_CODE_DISABLE_CRON": "1"}
HOOK_ASK_REASON = "tools/agent.py asks the user about every tool call except reading inside the repository"
TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "stopped", "killed"})  # claude_agent_sdk.types

# The input field a one-line summary of a tool call shows.
SUMMARY_FIELDS = {
    "Bash": "command", "PowerShell": "command",
    "Read": "file_path", "Write": "file_path", "Edit": "file_path", "MultiEdit": "file_path",
    "NotebookRead": "notebook_path", "NotebookEdit": "notebook_path",
    "Glob": "pattern", "Grep": "pattern", "LS": "path",
    "WebFetch": "url", "WebSearch": "query", "ToolSearch": "query",
    "Task": "description", "Agent": "description", "Skill": "skill",
}
PATH_FIELDS = frozenset({"file_path", "notebook_path", "path"})

# What the origin of a turn Claude Code started by itself means (claude_agent_sdk.types.MessageOrigin).
ORIGIN_WORDS = {
    "task-notification": "a background task reported back",
    "peer": "a message from another Claude session",
    "channel": "a message from an MCP server",
}
ORIGIN_SUBKIND_WORDS = {
    "scheduled-trigger": "a scheduled prompt fired",
    "peer-send-message": "a message from another of your Claude sessions",
}

INSTALL_HINT = (
    '.venv\\Scripts\\python -m pip install -e ".[agent]"' if os.name == "nt"
    else '.venv/bin/python -m pip install -e ".[agent]"'
)
NO_KEY_MESSAGE = (
    "Neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN is set, so Claude Code would fall back to\n"
    "another login (such as your claude.ai plan). This runner is for API-key billing: on Windows,\n"
    "start it with .\\tools\\agent.ps1 (it sets the key that tools\\claude-api.ps1 saved), or set\n"
    "ANTHROPIC_API_KEY for this process yourself."
)
BILLING_NOTE = (
    "Costs are Claude Code's estimates at list prices. The bill is on the Usage page of the\n"
    "Anthropic Console; compare the two after the first short run (tools/README.md, billing check)."
)
POLICY_LINE = ("Reading files inside the repository runs without asking. Every other tool call asks y/n, "
               "every Bash command included;")
SWITCHED_OFF_LINE = ("background work, scheduled prompts, git worktrees and messages to other Claude "
                     "sessions are switched off.")
# Why a question was refused without asking: (shown to the person, told to Claude).
NOT_IN_TURN = ("it came while no prompt of yours was running, so you were not asked",
               "Not allowed: the request came while the user had no prompt running, so nobody was asked.")
TURN_OVER = ("your turn ended before you could be asked",
             "Not allowed: the user's turn ended before they could be asked.")
NO_TERMINAL = ("no terminal to ask; run the session interactively to approve tools",
               "Not allowed: this session has no terminal where the user could approve the tool call.")
# What Claude is told about a call refused by rule (refusal_reason). Claude Code 2.1.301 shows a
# PreToolUse hook's deny reason to Claude as "PreToolUse:<tool> hook error: <reason>", so the reason
# itself has to say that this is a refusal and not a fault.
REFUSAL_MESSAGE = "Not allowed: tools/agent.py refuses this without asking the user ({reason}). Do not retry this call."


# ----------------------------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------------------------

def _prompt_text(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("the prompt is empty")
    return value


def _usd(value: str) -> float:
    try:
        amount = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from None
    if not amount > 0:
        raise argparse.ArgumentTypeError("the budget must be more than 0")
    return amount


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tools/agent.py",
        description="Claude on this repository through the Claude Agent SDK, billed to an API key. "
                    "In a session: Ctrl+C interrupts the current turn, /exit quits.",
    )
    what = parser.add_mutually_exclusive_group()
    what.add_argument("--once", metavar="PROMPT", type=_prompt_text,
                      help="run one turn with this prompt, then exit")
    what.add_argument("--check", action="store_true",
                      help="import the SDK, build the options and print them; no API call")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"model (default: {DEFAULT_MODEL})")
    parser.add_argument("--effort", choices=EFFORT_LEVELS,
                        help="effort level (default: Claude Code's own default)")
    parser.add_argument("--max-budget-usd", type=_usd, metavar="USD",
                        help="make no more model calls once Claude Code's cost estimate for the session "
                             "passes this (the turn that crosses it can overshoot a little)")
    parser.add_argument("--cli", metavar="PATH",
                        help="run this Claude Code executable instead of the one bundled with the SDK")
    return parser.parse_args(argv)


# ----------------------------------------------------------------------------------------------
# Permission policy (pure)
# ----------------------------------------------------------------------------------------------

class Verdict(NamedTuple):
    allow: bool  # True: allow without asking; False: ask the person at the terminal
    reason: str


def is_inside(path_text: str, root: Path) -> bool:
    """Whether a path, relative ones taken from ``root``, lies in ``root`` (symbolic links followed)."""
    if not path_text:
        return True
    expanded = os.path.expanduser(path_text)
    candidate = expanded if os.path.isabs(expanded) else os.path.join(root, expanded)
    real = os.path.normcase(os.path.realpath(candidate))
    base = os.path.normcase(os.path.realpath(root))
    try:
        return os.path.commonpath([real, base]) == base
    except ValueError:  # different drives on Windows
        return False


def _fixed_part(pattern: str) -> str:
    """The directory part of a glob pattern before its first wildcard ("src/a*/b" -> "src")."""
    head = re.split(r"[*?\[{]", pattern, maxsplit=1)[0]
    return head if head.endswith(("/", "\\")) or head == pattern else os.path.dirname(head)


def _brace_expansions(pattern: str) -> list[str] | str:
    """The patterns that the brace groups of a glob pattern stand for ("a{b,c}" -> ["ab", "ac"]),
    or why they were not worked out: a group holds ".." (a parent folder, or a range such as
    {a..c} that a shell fills with characters not written in the pattern), or there would be more
    than GLOB_EXPANSION_LIMIT of them."""
    done: list[str] = []
    todo = [pattern]
    while todo:
        text = todo.pop()
        group = re.search(r"\{([^{}]*)\}", text)  # an innermost group
        if group is None:
            done.append(text)
        elif ".." in group.group(1):
            return f"a brace group holds .. (a parent folder or a range): {pattern}"
        else:
            todo += [text[:group.start()] + choice + text[group.end():] for choice in group.group(1).split(",")]
        if len(done) + len(todo) > GLOB_EXPANSION_LIMIT:
            return f"more than {GLOB_EXPANSION_LIMIT} brace alternatives to check: {pattern}"
    return done


def _could_name_parent(segment: str) -> bool:
    """Whether one path segment of a glob pattern could match "..", the parent folder.

    A leading dot in a name is matched only by a written dot or a bracket expression (the rule
    for hidden names that shells and glob libraries share), so a segment that starts with * or ?
    cannot. Every bracket expression is taken to match a dot."""
    if not segment or segment[0] in "*?":
        return False
    regex, i = "", 0
    while i < len(segment):
        char = segment[i]
        if char == "[":
            j = i + 1
            if j < len(segment) and segment[j] in "!^":
                j += 1
            if j < len(segment) and segment[j] == "]":  # a ] right after [ or [! belongs to the set
                j += 1
            end = segment.find("]", j)
            regex += "."
            i = i + 1 if end < 0 else end + 1  # an unclosed [ is taken as a wildcard too
            continue
        regex += ".*" if char == "*" else "." if char == "?" else re.escape(char)
        i += 1
    return re.fullmatch(regex, "..", re.S) is not None


def glob_reach_problem(pattern: str, base: str, root: Path) -> str | None:
    """Why the names a glob pattern matches might lie outside ``root``, or None when they cannot.

    ``base`` is the folder the pattern is taken from ("" for the working directory). A backslash
    is read both ways: as a path separator (Windows), and as an escape of the character after it
    (glob libraries; only dots, separators and glob syntax are unescaped, which is what can change
    where a pattern reaches, so that a Windows path such as C:\\Users keeps its meaning)."""
    readings = {pattern.replace("\\", "/"), re.sub(r"\\([.*?\[\]{},/\\])", r"\1", pattern)}
    for reading in sorted(readings):
        expansions = _brace_expansions(reading)
        if isinstance(expansions, str):
            return expansions
        for expanded in expansions:
            if any(_could_name_parent(part) for part in expanded.split("/")):
                return f"the pattern can name a parent folder (..): {pattern}"
            fixed = _fixed_part(expanded)
            path = os.path.join(base, fixed) if base and fixed else (fixed or base)
            if not is_inside(path, root):
                return f"reads outside the repository: {path}"
    return None


def permission_verdict(tool_name: str, tool_input: Mapping[str, Any], root: Path = REPO_ROOT) -> Verdict:
    """Allow a tool call without asking, or ask: which calls the runner lets through."""
    if tool_name in READ_ONLY_SESSION_TOOLS:
        return Verdict(True, "reads only the session's own state")
    fields = READ_ONLY_FILE_TOOLS.get(tool_name)
    if fields is None:
        return Verdict(False, "not a read-only tool")
    paths: list[str] = []
    for name in fields:
        value = tool_input.get(name)
        if value is None or value == "":
            if tool_name in PATH_REQUIRED:
                return Verdict(False, f"no {name} given")
            continue
        if not isinstance(value, str):
            return Verdict(False, f"{name} is not a path")
        paths.append(value)
    for path in paths:
        if not is_inside(path, root):
            return Verdict(False, f"reads outside the repository: {path}")
    # Glob's pattern names what it lists; Grep's glob only filters, but is held to the same rule.
    pattern_field = {"Glob": "pattern", "Grep": "glob"}.get(tool_name)
    if pattern_field is not None:
        pattern = tool_input.get(pattern_field)
        if tool_name == "Glob" and (not isinstance(pattern, str) or not pattern):
            return Verdict(False, "no text pattern given")
        if pattern is not None and pattern != "":
            if not isinstance(pattern, str):
                return Verdict(False, f"{pattern_field} is not text")
            problem = glob_reach_problem(pattern, tool_input.get("path") or "", root)
            if problem:
                return Verdict(False, problem)
    return Verdict(True, "reads only, inside the repository")


def refusal_reason(tool_name: str, tool_input: Mapping[str, Any]) -> str | None:
    """Why a tool call is refused without asking, or None when it is not."""
    if tool_name in SWITCHED_OFF_TOOLS:
        return f"{tool_name} is switched off in this runner"
    isolation = tool_input.get("isolation")
    if isolation == "worktree":
        return "it would give the agent a git worktree on a new branch; work stays on the current branch"
    if isolation not in (None, ""):
        return f"agents with isolation {isolation!r} run elsewhere in the background, which is switched off"
    if tool_input.get("run_in_background") not in (None, False):
        return "background work is switched off (questions can be asked only during a turn)"
    return None


def refusal_message(reason: str) -> str:
    """What Claude is told when a call is refused by rule (``refusal_reason``)."""
    return REFUSAL_MESSAGE.format(reason=reason)


def pre_tool_use_decision(tool_name: str, tool_input: Mapping[str, Any],
                          root: Path = REPO_ROOT) -> tuple[str | None, str]:
    """What the runner's PreToolUse hook decides for a tool call: (decision, reason).

    "deny" refuses the call (the reason is the one ``refusal_reason`` gives); "ask" makes Claude
    Code put it to ``can_use_tool`` even when it would run it without asking; None leaves the call
    to Claude Code's own rules, which is done only for calls the runner would allow anyway
    (reading inside the repository)."""
    refusal = refusal_reason(tool_name, tool_input)
    if refusal:
        return "deny", refusal
    verdict = permission_verdict(tool_name, tool_input, root)
    if verdict.allow:
        return None, verdict.reason
    return "ask", HOOK_ASK_REASON


def hook_output(decision: str | None, reason: str) -> dict[str, Any]:
    """The PreToolUse hook's answer in the form Claude Code reads (PreToolUseHookSpecificOutput).

    For "deny", the reason Claude reads is ``refusal_message(reason)``."""
    if decision is None:
        return {}
    told = refusal_message(reason) if decision == "deny" else reason
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision,
                                   "permissionDecisionReason": told}}


def is_own_result(msg: Any) -> bool:
    """Whether a ResultMessage ends a turn that a prompt sent by this runner started.

    Claude Code marks the turns it starts by itself (background-task notifications, scheduled
    prompts, messages from other sessions) with an ``origin``; prompts sent through the SDK have
    none, or the human kind (claude_agent_sdk.types.MessageOrigin)."""
    origin = getattr(msg, "origin", None)
    if origin is None:
        return True
    return isinstance(origin, Mapping) and origin.get("kind") == "human"


def origin_text(origin: Any) -> str:
    if not isinstance(origin, Mapping):
        return "Claude Code did not say why"
    words = ORIGIN_SUBKIND_WORDS.get(origin.get("subkind")) or ORIGIN_WORDS.get(origin.get("kind"))
    kind = f"origin {origin.get('kind')}"
    return f"{words}; {kind}" if words else kind


# ----------------------------------------------------------------------------------------------
# Text shown in the terminal (pure)
# ----------------------------------------------------------------------------------------------

def _is_hidden(char: str) -> bool:
    return unicodedata.category(char) in HIDDEN_CATEGORIES


def _escaped(char: str) -> str:
    code = ord(char)
    return f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}" if code < 0x10000 else f"\\U{code:08x}"


def visible(text: Any) -> str:
    """Text as it may reach the terminal: every character that a terminal would not show as written
    (HIDDEN_CATEGORIES, newline and tab apart) replaced by an escape such as \\x1b or \\u202e."""
    return _NOT_PLAIN.sub(lambda m: _escaped(m.group()) if _is_hidden(m.group()) else m.group(), str(text))


def hidden_characters(value: Any, found: dict[str, None] | None = None) -> list[str]:
    """The distinct characters ``visible`` escapes, in the order they occur in a tool input's
    strings (keys included, nested lists and mappings searched)."""
    found = {} if found is None else found
    if isinstance(value, str):
        for char in _NOT_PLAIN.findall(value):
            if _is_hidden(char):
                found.setdefault(char)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            hidden_characters(key, found)
            hidden_characters(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            hidden_characters(item, found)
    return list(found)


def one_line(text: Any, width: int) -> str:
    flat = " ".join(visible(text).split())
    return flat if len(flat) <= width else flat[: max(width - 3, 1)] + "..."


def display_path(path_text: str, root: Path = REPO_ROOT) -> str:
    """A path relative to the repository when it lies inside, else as given."""
    if not isinstance(path_text, str) or not path_text:
        return str(path_text)
    if os.path.isabs(path_text) and is_inside(path_text, root):
        rel = os.path.relpath(os.path.realpath(path_text), os.path.realpath(root))
        return "." if rel == "." else rel
    return path_text


def summarize_tool_call(tool_name: str, tool_input: Mapping[str, Any], root: Path = REPO_ROOT,
                        width: int = 110) -> str:
    """One line naming a tool call and what it acts on, e.g. "Bash: git status"."""
    detail = ""
    field = SUMMARY_FIELDS.get(tool_name)
    value = tool_input.get(field) if field else None
    if isinstance(value, str) and value:
        detail = display_path(value, root) if field in PATH_FIELDS else value
        if tool_name == "Grep" and isinstance(tool_input.get("path"), str) and tool_input["path"]:
            detail += " in " + display_path(tool_input["path"], root)
    elif tool_name == "TodoWrite" and isinstance(tool_input.get("todos"), list):
        detail = f"{len(tool_input['todos'])} items"
    else:
        detail = next((v for v in tool_input.values() if isinstance(v, str) and v), "")
    name = visible(tool_name)
    detail = one_line(detail, max(width - len(name) - 2, 10))
    return f"{name}: {detail}" if detail else name


def _lines(text: Any) -> list[str]:
    """The lines of a text, split only at newlines (every other line break is shown escaped)."""
    return visible(text).splitlines()


def _preview(text: Any, prefix: str, full: bool, limit: int = PREVIEW_LINES) -> list[str]:
    """The lines of a text, each after ``prefix``: all of them when ``full``, else a cut preview."""
    lines = _lines(text) or [""]
    if full:
        return [prefix + line for line in lines]
    shown = [prefix + (line if len(line) <= PREVIEW_WIDTH else line[: PREVIEW_WIDTH - 3] + "...")
             for line in lines[:limit]]
    if len(lines) > limit:
        shown.append(f"{prefix}... ({len(lines) - limit} more lines)")
    return shown


def _text(value: Any, width: int, full: bool) -> str:
    return " ".join(visible(value).split()) if full else one_line(value, width)


def describe_tool_request(tool_name: str, tool_input: Mapping[str, Any], root: Path = REPO_ROOT,
                          full: bool = False) -> list[str]:
    """The lines shown before asking whether a tool call may run.

    Commands are always shown whole. File contents, edits and prompts are cut to a preview unless
    ``full``; the question then offers "a" to show all of it. Characters a terminal would not show
    as written are shown escaped, and a last line says so."""
    lines = [f"  ? Claude wants to use {tool_name}"]
    get = tool_input.get
    if tool_name in ("Bash", "PowerShell") and isinstance(get("command"), str):
        lines += _preview(get("command"), "      ", full=True)
        if isinstance(get("description"), str) and get("description"):
            lines.append(f"    ({_text(get('description'), 100, full)})")
        if get("timeout") is not None:
            lines.append(f"    (timeout: {get('timeout')} ms)")
    elif tool_name == "Write" and isinstance(get("file_path"), str):
        content = get("content") if isinstance(get("content"), str) else ""
        count = len(_lines(content))
        lines.append(f"    file: {display_path(get('file_path'), root)} ({count} line{'' if count == 1 else 's'})")
        lines += _preview(content, "    + ", full)
    elif tool_name == "Edit" and isinstance(get("file_path"), str):
        lines.append(f"    file: {display_path(get('file_path'), root)}"
                     + (" (every occurrence)" if get("replace_all") else ""))
        lines += _preview(get("old_string", ""), "    - ", full)
        lines += _preview(get("new_string", ""), "    + ", full)
    elif tool_name == "MultiEdit" and isinstance(get("file_path"), str):
        edits = get("edits") if isinstance(get("edits"), list) else []
        lines.append(f"    file: {display_path(get('file_path'), root)} ({len(edits)} edit{'' if len(edits) == 1 else 's'})")
        for number, edit in enumerate(edits, 1):
            edit = edit if isinstance(edit, Mapping) else {}
            lines.append(f"    edit {number}" + (" (every occurrence)" if edit.get("replace_all") else "") + ":")
            lines += _preview(edit.get("old_string", ""), "    - ", full)
            lines += _preview(edit.get("new_string", ""), "    + ", full)
    elif tool_name == "NotebookEdit" and isinstance(get("notebook_path"), str):
        mode = get("edit_mode") or "replace (the default)"
        cell = get("cell_id") or "none given"
        kind = f"; cell type: {get('cell_type')}" if get("cell_type") else ""
        lines.append(f"    file: {display_path(get('notebook_path'), root)}")
        lines.append(f"    edit mode: {mode}; cell: {cell}{kind}")
        if get("edit_mode") != "delete":
            lines += _preview(get("new_source", ""), "    + ", full)
    elif tool_name in ("Task", "Agent") and isinstance(get("prompt"), str):
        lines.append(f"    agent: {get('subagent_type') or 'general-purpose (the default)'}"
                     + (f"; model: {get('model')}" if get("model") else "")
                     + (f"; effort: {get('effort')}" if get("effort") else ""))
        if isinstance(get("description"), str) and get("description"):
            lines.append(f"    task: {_text(get('description'), 100, full)}")
        lines.append("    prompt:")
        lines += _preview(get("prompt"), "    | ", full)
    elif tool_name == "WebFetch" and isinstance(get("url"), str):
        lines.append(f"    url: {get('url')}")
        if isinstance(get("prompt"), str):
            lines.append(f"    asks: {_text(get('prompt'), 100, full)}")
    elif tool_name == "WebSearch" and isinstance(get("query"), str):
        lines.append(f"    query: {get('query')}")
    else:
        text = json.dumps(dict(tool_input), ensure_ascii=False, sort_keys=True, default=str)
        if full:
            pretty = json.dumps(dict(tool_input), ensure_ascii=False, sort_keys=True, default=str, indent=2)
            lines.append("    input:")
            lines += _preview(pretty, "      ", full=True)
        else:
            lines.append(f"    input: {one_line(text, 600)}")
    hidden = hidden_characters([tool_name, dict(tool_input)])
    if hidden:
        shown = ", ".join(_escaped(char) for char in hidden[:6]) + (", ..." if len(hidden) > 6 else "")
        lines.append(f"    (the input contains characters a terminal would not show as written: {shown}; "
                     "they are shown escaped)")
    # Every line once more through visible(): values put in directly above (paths, URLs, models)
    # must not reach the terminal raw either.
    return [visible(line) for line in lines]


def format_usd(amount: float | None) -> str:
    if amount is None:
        return "unknown"
    return f"${max(amount, 0.0):,.4f}"


def cost_line(turn: float | None, total: float, which: str = "this") -> str:
    turn_text = format_usd(turn) if turn is not None else "unknown (Claude Code reported no cost)"
    return f"  [cost of {which} turn: {turn_text}; session so far: {format_usd(total)}]"


class CostMeter:
    """Turns Claude Code's running cost totals into the cost of each turn.

    In a ClaudeSDKClient session every ResultMessage carries the running total so far (the CLI's
    own description of ``total_cost_usd``); a /clear starts a new conversation whose total starts
    again from zero (announced by a ConversationResetMessage)."""

    def __init__(self) -> None:
        self.earlier = 0.0  # totals of conversations that were reset
        self.current = 0.0  # the latest running total of the current conversation

    def record(self, running_total: float | None) -> float | None:
        """Take the total from a ResultMessage and return the cost of the turn it ends."""
        if running_total is None:
            return None
        if running_total < self.current - 1e-9:  # the count started again without notice
            self.conversation_reset()
        turn = max(running_total - self.current, 0.0)
        self.current = running_total
        return turn

    def conversation_reset(self) -> None:
        self.earlier += self.current
        self.current = 0.0

    @property
    def total(self) -> float:
        return self.earlier + self.current


# ----------------------------------------------------------------------------------------------
# Credentials (pure)
# ----------------------------------------------------------------------------------------------

def credential_mode(env: Mapping[str, str]) -> str | None:
    """"bearer" (ANTHROPIC_AUTH_TOKEN, which outranks the key in Claude Code), "api-key", or None."""
    if env.get("ANTHROPIC_AUTH_TOKEN"):
        return "bearer"
    if env.get("ANTHROPIC_API_KEY"):
        return "api-key"
    return None


def credential_problem(account: Any, mode: str) -> str | None:
    """Why Claude Code would not bill the key from the environment, or None when it would.

    ``account`` is the ``account`` entry of ClaudeSDKClient.get_server_info(): Claude Code 2.1.301
    reports {"apiKeySource": "ANTHROPIC_API_KEY", ...} when it sends that key, and
    {"tokenSource": "ANTHROPIC_AUTH_TOKEN", ...} for a bearer token. When the entry is missing
    (another Claude Code version), the check cannot be made and is skipped."""
    if not isinstance(account, Mapping):
        return None
    if mode == "api-key":
        source = account.get("apiKeySource")
        if source == "ANTHROPIC_API_KEY":
            return None
        return (f"Claude Code says its API key comes from {source or 'nowhere'}, not from "
                "ANTHROPIC_API_KEY, so this session would not be billed to your key. Nothing was sent.")
    source = account.get("tokenSource")
    if source == "ANTHROPIC_AUTH_TOKEN":
        return None
    return (f"Claude Code says its token comes from {source or 'nowhere'}, not from "
            "ANTHROPIC_AUTH_TOKEN, so this session would not be billed to your key. Nothing was sent.")


def credential_text(mode: str | None) -> str:
    if mode == "bearer":
        return "ANTHROPIC_AUTH_TOKEN is set (the key is sent as a bearer token)"
    if mode == "api-key":
        return "ANTHROPIC_API_KEY is set (the key is sent as an API key, x-api-key)"
    return "neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN is set: a session would refuse to start"


# ----------------------------------------------------------------------------------------------
# The SDK
# ----------------------------------------------------------------------------------------------

def load_sdk() -> Any | None:
    try:
        return importlib.import_module("claude_agent_sdk")
    except ImportError:
        return None


def project_instructions(root: Path = REPO_ROOT) -> str | None:
    """The repository's CLAUDE.md as the text appended to Claude Code's system prompt, or None."""
    path = root / "CLAUDE.md"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not text.strip():
        return None
    return f"Contents of {path} (the project's instructions, checked into the repository):\n\n{text.rstrip()}\n"


def option_values(args: argparse.Namespace, can_use_tool: Callable[..., Any],
                  stderr: Callable[[str], None] | None = None, *, hooks: Any = None,
                  instructions: str | None = None) -> dict[str, Any]:
    """The keyword arguments for claude_agent_sdk.ClaudeAgentOptions."""
    system_prompt: dict[str, str] = {"type": "preset", "preset": "claude_code"}
    if instructions:
        system_prompt["append"] = instructions
    values: dict[str, Any] = {
        "model": args.model,
        "cwd": str(REPO_ROOT),
        "system_prompt": system_prompt,
        "setting_sources": [],
        "permission_mode": "default",
        "can_use_tool": can_use_tool,
        "disallowed_tools": list(SWITCHED_OFF_TOOLS),
        "env": dict(CLI_ENV),
        "include_partial_messages": True,
    }
    if hooks is not None:
        values["hooks"] = hooks
    if stderr is not None:
        values["stderr"] = stderr
    if args.cli:
        values["cli_path"] = args.cli
    if args.effort:
        values["effort"] = args.effort
    if args.max_budget_usd is not None:
        values["max_budget_usd"] = args.max_budget_usd
    return values


def pre_tool_use_hooks(sdk: Any, callback: Callable[..., Any]) -> dict[str, Any]:
    """ClaudeAgentOptions.hooks: ``callback`` sees every tool call (no matcher)."""
    return {"PreToolUse": [sdk.HookMatcher(matcher=None, hooks=[callback])]}


def bundled_cli(sdk: Any) -> Path | None:
    """Where the SDK keeps its own Claude Code (the place SubprocessCLITransport looks first)."""
    try:
        path = Path(sdk.__file__).parent / "_bundled" / ("claude.exe" if os.name == "nt" else "claude")
    except (AttributeError, TypeError):
        return None
    return path if path.is_file() else None


def bundled_cli_version(sdk: Any) -> str | None:
    try:
        return importlib.import_module(sdk.__name__ + "._cli_version").__cli_version__
    except (ImportError, AttributeError):
        return None


def child_process_flags(os_name: str = os.name) -> dict[str, Any]:
    """Keyword arguments for anyio.open_process that keep the terminal's Ctrl+C from Claude Code."""
    if os_name == "nt":
        return {"creationflags": CREATE_NO_WINDOW}
    return {"start_new_session": True}


def keep_ctrl_c_from_cli(anyio_module: Any = None, os_name: str = os.name) -> bool:
    """Make the SDK start Claude Code where the terminal's Ctrl+C does not reach it.

    Claude Code aborts the turn and then exits on SIGINT (seen with 2.1.301), which would end the
    session, and the SDK (0.2.168) has no option for the flags that prevent it. Its transport calls
    ``anyio.open_process(...)`` through the module attribute, so wrapping that attribute adds them.
    Returns False when anyio cannot be imported."""
    if anyio_module is None:
        try:
            anyio_module = importlib.import_module("anyio")
        except ImportError:
            return False
    original = anyio_module.open_process
    if getattr(original, "keeps_ctrl_c_from_child", False):
        return True
    flags = child_process_flags(os_name)

    async def open_process(command: Any, **kwargs: Any) -> Any:
        for key, value in flags.items():
            if key == "creationflags":
                kwargs[key] = kwargs.get(key, 0) | value
            else:
                kwargs.setdefault(key, value)
        return await original(command, **kwargs)

    open_process.keeps_ctrl_c_from_child = True  # type: ignore[attr-defined]
    anyio_module.open_process = open_process
    return True


# ----------------------------------------------------------------------------------------------
# Showing the message stream
# ----------------------------------------------------------------------------------------------

class Display:
    """Prints the session: streamed text, one line per tool call, errors and costs.

    Everything goes through ``visible()`` on its way out, so nothing Claude, a tool or Claude Code
    writes can move the cursor, erase or hide lines, or change how later text (a question) looks."""

    def __init__(self, sdk: Any, out: Any, root: Path = REPO_ROOT, requested_model: str | None = None) -> None:
        self.sdk = sdk
        self.out = out
        self.root = root
        self.requested_model = requested_model
        self.lock = threading.RLock()
        self._mid_line = False
        self._stream_id: str | None = None
        self._stream_had_text = False
        self._streamed_ids: set[str] = set()
        self._tool_names: dict[str, str] = {}
        self.refused: set[str] = set()  # tool_use ids the runner said no to (already reported)
        self.init_data: dict[str, Any] | None = None

    def write(self, text: str) -> None:
        if not text:
            return
        text = visible(text)
        with self.lock:
            self.out.write(text)
            self.out.flush()
            self._mid_line = not text.endswith("\n")

    def line(self, text: str = "") -> None:
        text = visible(text)
        with self.lock:
            if self._mid_line:
                self.out.write("\n")
            self.out.write(text + "\n")
            self.out.flush()
            self._mid_line = False

    def handle(self, msg: Any) -> None:
        sdk = self.sdk
        if isinstance(msg, sdk.StreamEvent):
            self._on_stream(msg)
        elif isinstance(msg, sdk.AssistantMessage):
            self._on_assistant(msg)
        elif isinstance(msg, sdk.UserMessage):
            self._on_user(msg)
        elif isinstance(msg, sdk.SystemMessage):
            self._on_system(msg)
        elif isinstance(msg, sdk.RateLimitEvent):
            status = getattr(msg.rate_limit_info, "status", None)
            if status and status != "allowed":
                self.line(f"  ! rate limit: {status}")

    def _on_stream(self, msg: Any) -> None:
        if msg.parent_tool_use_id is not None:
            return
        event = msg.event if isinstance(msg.event, Mapping) else {}
        kind = event.get("type")
        if kind == "message_start":
            self._stream_id = (event.get("message") or {}).get("id")
            self._stream_had_text = False
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                if self._stream_id:
                    self._streamed_ids.add(self._stream_id)
                self._stream_had_text = True
                self.write(delta["text"])

    def _was_streamed(self, message_id: str | None) -> bool:
        if message_id:
            return message_id in self._streamed_ids
        return self._stream_had_text

    def _on_assistant(self, msg: Any) -> None:
        sdk = self.sdk
        nested = msg.parent_tool_use_id is not None  # a subagent's call
        indent = "    " if nested else "  "
        for block in msg.content:
            if isinstance(block, sdk.TextBlock):
                if not nested and not self._was_streamed(msg.message_id):
                    self.write(block.text)
            elif isinstance(block, sdk.ToolUseBlock):
                self._tool_names[block.id] = block.name
                self.line(f"{indent}> {summarize_tool_call(block.name, block.input, self.root)}")
            elif isinstance(block, sdk.ServerToolUseBlock):
                self.line(f"{indent}> {summarize_tool_call(block.name, block.input, self.root)} (run by the API)")
        if msg.error:
            hint = ""
            if msg.error == "billing_error":
                hint = " (the API refused for billing; see the billing check in tools/README.md)"
            elif msg.error == "authentication_failed":
                hint = " (the key was not accepted)"
            self.line(f"  ! the API answered with an error: {msg.error}{hint}")

    def _on_user(self, msg: Any) -> None:
        if not isinstance(msg.content, list):
            return
        for block in msg.content:
            if isinstance(block, self.sdk.ToolResultBlock) and block.is_error:
                if block.tool_use_id in self.refused:
                    continue
                name = self._tool_names.get(block.tool_use_id, "a tool")
                content = block.content
                if isinstance(content, list):
                    content = " ".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
                first = (str(content or "").strip().splitlines() or [""])[0]
                self.line(f"  ! {name} did not run or failed: {one_line(first, 140)}")

    def _on_system(self, msg: Any) -> None:
        if msg.subtype != "init" or self.init_data is not None:
            return
        self.init_data = dict(msg.data)
        model = self.init_data.get("model")
        self.line(f"  [Claude Code {self.init_data.get('claude_code_version', '?')}: model {model}, "
                  f"API key source: {self.init_data.get('apiKeySource', '?')}]")
        if self.requested_model and model and model != self.requested_model:
            self.line(f"  ! Claude Code runs {model}, not the requested {self.requested_model}")

    def turn_on_its_own(self) -> None:
        with self.lock:
            self.write("\n")  # the person may be at the you> prompt
            self.line("  [Claude Code starts a turn by itself; it does not answer a prompt of yours]")

    def result(self, msg: Any, turn_cost: float | None, total: float, *, own: bool = True) -> None:
        if not own:
            self.line(f"  [end of a turn that no prompt of yours started ({origin_text(getattr(msg, 'origin', None))})]")
        reason = getattr(msg, "terminal_reason", None) or ""
        if reason.startswith("aborted"):
            self.line("  [the turn was interrupted]")
        elif msg.is_error:
            parts = []
            if msg.subtype and msg.subtype != "success":
                parts.append(msg.subtype)
            if getattr(msg, "api_error_status", None):
                parts.append(f"HTTP {msg.api_error_status}")
            parts += [one_line(e, 200) for e in (getattr(msg, "errors", None) or [])]
            self.line(f"  ! the turn ended with an error: {'; '.join(parts) or 'see the message above'}")
        self.line(cost_line(turn_cost, total, "this" if own else "that"))


# ----------------------------------------------------------------------------------------------
# The session
# ----------------------------------------------------------------------------------------------

class CredentialError(Exception):
    pass


class SessionEnded(Exception):
    """Claude Code stopped sending messages; the session cannot go on."""


@dataclass
class Question:
    tool_name: str
    tool_input: dict[str, Any]
    notes: list[str]
    answer: concurrent.futures.Future
    tool_use_id: str | None = None


class EventLoopThread:
    """An asyncio loop in a background thread, where the SDK client lives.

    The main thread keeps everything that reads the terminal (the prompt and the y/n questions),
    so Ctrl+C always arrives there as KeyboardInterrupt, on Windows as well."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="agent-sdk", daemon=True)

    def _run(self) -> None:
        # Made here, not in the main thread: on Windows a loop made in the main thread also takes
        # over the main thread's signal wake-up handle.
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        self.loop.run_forever()

    def start(self) -> None:
        if not self.thread.is_alive():
            self.thread.start()
            self._ready.wait()

    def submit(self, coro: Any) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self) -> None:
        if not self.thread.is_alive():
            return

        async def cancel_rest() -> None:
            rest = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in rest:
                task.cancel()
            await asyncio.gather(*rest, return_exceptions=True)

        try:
            self.submit(cancel_rest()).result(timeout=5)
        except (concurrent.futures.TimeoutError, RuntimeError):
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


class Session:
    def __init__(self, sdk: Any, args: argparse.Namespace, *, out: Any = None,
                 ask: Callable[[str], str] = input, interactive: bool | None = None,
                 env: Mapping[str, str] | None = None, detach_cli: bool = True) -> None:
        self.sdk = sdk
        self.args = args
        self.out = out if out is not None else sys.stdout
        self.ask = ask
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self.mode = credential_mode(os.environ if env is None else env)
        self.detach_cli = detach_cli
        self.display = Display(sdk, self.out, requested_model=args.model)
        self.meter = CostMeter()
        self.questions: queue.Queue[Question | None] = queue.Queue()  # None: some work finished
        self.runner = EventLoopThread()
        self.client: Any = None
        self.server_info: Mapping[str, Any] = {}
        self.instructions = project_instructions(REPO_ROOT)
        # The prompt of the person's that waits for its result. Questions are put to the person
        # only while it is set; the lock keeps that check and the queue in step.
        self._lock = threading.Lock()
        self._pending: concurrent.futures.Future | None = None
        self._reader: concurrent.futures.Future | None = None
        self._reader_error: BaseException | None = None
        self._turn_open = False  # on the loop thread: between a turn's init message and its result
        self.tasks: dict[str, str] = {}  # task ids Claude Code reported started and not yet ended

    # -- lifecycle (main thread) ---------------------------------------------------------------

    def open(self) -> None:
        if self.detach_cli:
            keep_ctrl_c_from_cli()
        values = option_values(self.args, self._can_use_tool, self._stderr,
                               hooks=pre_tool_use_hooks(self.sdk, self._pre_tool_use),
                               instructions=self.instructions)
        client = self.sdk.ClaudeSDKClient(options=self.sdk.ClaudeAgentOptions(**values))
        self.runner.start()
        self.wait(self.runner.submit(client.connect()))
        self.client = client
        info = self.wait(self.runner.submit(client.get_server_info()))
        self.server_info = info if isinstance(info, Mapping) else {}
        problem = credential_problem(self.server_info.get("account"), self.mode or "api-key")
        if problem:
            raise CredentialError(problem)
        self._reader = self.runner.submit(self._read_messages(client))

    def banner(self) -> None:
        account = self.server_info.get("account")
        if isinstance(account, Mapping):
            source = account.get("tokenSource") if self.mode == "bearer" else account.get("apiKeySource")
            confirmed = f"Claude Code confirms it uses it ({source})"
        else:
            confirmed = "this Claude Code does not say which credential it uses"
        cli = self.args.cli or bundled_cli(self.sdk) or "found by the SDK"
        version = self.server_info.get("claude_code_version") or (
            None if self.args.cli else bundled_cli_version(self.sdk)) or "?"
        self.display.line(f"Claude Agent SDK {getattr(self.sdk, '__version__', '?')}, Claude Code {version} ({cli})")
        self.display.line(f"Model {self.args.model}, working in {REPO_ROOT}")
        self.display.line(f"{credential_text(self.mode)}; {confirmed}.")
        self.display.line("No settings files are loaded; " + (
            "the repository's CLAUDE.md is given to Claude Code with its system prompt." if self.instructions
            else "the repository has no CLAUDE.md to give to Claude Code."))
        self.display.line(f"{POLICY_LINE} {SWITCHED_OFF_LINE}")
        if self.args.once is None:
            self.display.line("Ctrl+C interrupts the current turn; /exit (or Ctrl+D, on Windows Ctrl+Z then Enter) quits.")
        self.display.line(BILLING_NOTE)

    def turn(self, prompt: str) -> bool:
        """Send one prompt and wait for the turn it starts to end; True when it ended without an error."""
        if self._reader is None or self._reader.done():
            raise SessionEnded("Claude Code is no longer running"
                               + (f" ({self._reader_error})" if self._reader_error else ""))
        done: concurrent.futures.Future = concurrent.futures.Future()
        with self._lock:
            self._pending = done
        try:
            self.wait(self.runner.submit(self.client.query(prompt)))
            result = self.wait(done)
        finally:
            self._stop_asking()
        if result is None:
            self.display.line("  ! the messages stopped before the turn ended (Claude Code may have stopped)")
            return False
        return not result.is_error

    def close(self) -> None:
        try:
            if self.client is not None:
                client, self.client = self.client, None
                if self.tasks:
                    self.display.line(f"  ! {len(self.tasks)} task(s) still running end with the session: "
                                      + "; ".join(one_line(d, 60) for d in self.tasks.values()))
                self.wait(self.runner.submit(client.disconnect()), interruptible=False)
        finally:
            self.runner.stop()

    def summary(self) -> None:
        self.display.line(f"Session total: {format_usd(self.meter.total)}.")
        self.display.line(BILLING_NOTE)

    # -- waiting, interrupting and asking (main thread) ----------------------------------------

    def wait(self, future: concurrent.futures.Future, *, interruptible: bool = True) -> Any:
        """Wait for work on the loop thread, answering its permission questions meanwhile.

        The first Ctrl+C interrupts the turn; a second one gives up waiting (KeyboardInterrupt).
        Questions that come after an interrupt are answered no."""
        interrupting = False
        future.add_done_callback(lambda _: self.questions.put(None))  # wake up at once when done
        while True:
            try:
                while not future.done():  # inside the try, so a Ctrl+C between two polls is caught as well
                    try:
                        # A short timeout: on Windows a Ctrl+C is only seen once the wait returns.
                        question = self.questions.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if question is None:
                        continue
                    if interrupting:
                        self._answer(question, self.sdk.PermissionResultDeny(
                            message="The user interrupted the turn.", interrupt=True))
                    else:
                        self._ask(question)
                return future.result()
            except KeyboardInterrupt:
                if interrupting or not interruptible or self.client is None:
                    raise
                interrupting = True
                self.display.line("  [Ctrl+C: interrupting the turn; press Ctrl+C again to quit]")
                self.runner.submit(self.client.interrupt())

    def _stop_asking(self) -> None:
        """The person's turn is over: no more questions, and the ones still waiting are refused."""
        with self._lock:
            self._pending = None
            left: list[Question] = []
            while True:
                try:
                    question = self.questions.get_nowait()
                except queue.Empty:
                    break
                if question is not None:
                    left.append(question)
        for question in left:
            self._refuse(question.tool_name, question.tool_input, question.tool_use_id, *TURN_OVER,
                         answer=question.answer)

    def _answer(self, question: Question, result: Any) -> None:
        if isinstance(result, self.sdk.PermissionResultDeny) and question.tool_use_id:
            self.display.refused.add(question.tool_use_id)
        if not question.answer.done():
            try:
                question.answer.set_result(result)
            except concurrent.futures.InvalidStateError:
                pass

    def _ask(self, question: Question) -> None:
        sdk = self.sdk
        if question.answer.done():  # the request was cancelled meanwhile
            return
        shown = describe_tool_request(question.tool_name, question.tool_input)
        whole = describe_tool_request(question.tool_name, question.tool_input, full=True)
        with self.display.lock:
            for text in shown + question.notes:
                self.display.line(text)
        prompt = "  Allow? [y/n] " if shown == whole else "  Allow? [y/n, or a to show all of it] "
        while True:
            try:
                reply = self.ask(prompt)
            except EOFError:
                reply = ""
            except KeyboardInterrupt:
                self._answer(question, sdk.PermissionResultDeny(
                    message="The user pressed Ctrl+C instead of answering and interrupted the turn.",
                    interrupt=True))
                self.display.line("  [not allowed; the turn is interrupted]")
                return
            if reply.strip().lower() in ("a", "all") and shown != whole:
                with self.display.lock:
                    for text in whole[1:]:
                        self.display.line(text)
                shown, prompt = whole, "  Allow? [y/n] "
                continue
            break
        if reply.strip().lower() in ("y", "yes"):
            self._answer(question, sdk.PermissionResultAllow())
        else:
            self._answer(question, sdk.PermissionResultDeny(message="The user declined this tool call."))
            self.display.line("  [not allowed]")

    # -- on the loop thread --------------------------------------------------------------------

    async def _read_messages(self, client: Any) -> None:
        """Show every message as it comes, and hand each result of a prompt to the turn waiting for it."""
        error: BaseException | None = None
        try:
            async for msg in client.receive_messages():
                self._route(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # Claude Code stopped or the stream broke: end the waiting turn
            error = exc
        finally:
            self._reader_error = error
            with self._lock:
                pending, self._pending = self._pending, None
            if pending is not None and not pending.done():
                if error is not None:
                    pending.set_exception(error)
                else:
                    pending.set_result(None)

    def _route(self, msg: Any) -> None:
        sdk = self.sdk
        if isinstance(msg, sdk.ResultMessage):
            self._on_result(msg)
            return
        if isinstance(msg, sdk.ConversationResetMessage):
            self.meter.conversation_reset()
            return
        if isinstance(msg, sdk.SystemMessage):
            self._track_task(msg)
            if msg.subtype == "init" and not self._turn_open:
                self._turn_open = True
                with self._lock:
                    prompted = self._pending is not None
                if not prompted:
                    self.display.turn_on_its_own()
        self.display.handle(msg)

    def _on_result(self, msg: Any) -> None:
        self._turn_open = False
        turn_cost = self.meter.record(msg.total_cost_usd)
        pending = None
        if is_own_result(msg):
            with self._lock:
                pending, self._pending = self._pending, None
        self.display.result(msg, turn_cost, self.meter.total, own=pending is not None)
        if pending is not None and not pending.done():
            pending.set_result(msg)

    def _track_task(self, msg: Any) -> None:
        data = msg.data if isinstance(getattr(msg, "data", None), Mapping) else {}
        task_id = data.get("task_id")
        if not isinstance(task_id, str):
            return
        if msg.subtype == "task_started":
            self.tasks[task_id] = str(data.get("description") or task_id)
        elif msg.subtype in ("task_notification", "task_updated"):
            patch = data.get("patch") if isinstance(data.get("patch"), Mapping) else {}
            if (data.get("status") or patch.get("status")) in TERMINAL_TASK_STATUSES:
                self.tasks.pop(task_id, None)

    async def _pre_tool_use(self, hook_input: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        hook_input = hook_input if isinstance(hook_input, Mapping) else {}
        tool_name = str(hook_input.get("tool_name") or "")
        tool_input = hook_input.get("tool_input") if isinstance(hook_input.get("tool_input"), Mapping) else {}
        decision, reason = pre_tool_use_decision(tool_name, tool_input)
        if decision == "deny":
            self._refuse(tool_name, tool_input, tool_use_id or hook_input.get("tool_use_id"), reason)
        return hook_output(decision, reason)

    async def _can_use_tool(self, tool_name: str, tool_input: dict[str, Any], context: Any) -> Any:
        tool_use_id = getattr(context, "tool_use_id", None)
        refusal = refusal_reason(tool_name, tool_input)
        if refusal:
            return self._refuse(tool_name, tool_input, tool_use_id, refusal)
        verdict = permission_verdict(tool_name, tool_input)
        if verdict.allow:
            return self.sdk.PermissionResultAllow()
        if not self.interactive:
            return self._refuse(tool_name, tool_input, tool_use_id, *NO_TERMINAL)
        notes = [f"    ({verdict.reason})"] if verdict.reason != "not a read-only tool" else []
        if getattr(context, "agent_id", None):
            notes.append("    (asked by a subagent)")
        said = [getattr(context, name, None) for name in ("title", "decision_reason")]
        notes += [f"    (Claude Code: {one_line(text, 160)})" for text in said
                  if isinstance(text, str) and text and text != HOOK_ASK_REASON]
        question = Question(tool_name, dict(tool_input), notes, concurrent.futures.Future(), tool_use_id)
        with self._lock:
            asking = self._pending is not None
            if asking:
                self.questions.put(question)
        if not asking:
            return self._refuse(tool_name, tool_input, tool_use_id, *NOT_IN_TURN)
        return await asyncio.wrap_future(question.answer)

    def _refuse(self, tool_name: str, tool_input: Mapping[str, Any], tool_use_id: Any, reason: str,
                message: str | None = None, *, answer: concurrent.futures.Future | None = None) -> Any:
        """Say no without asking, tell the person why, and return the answer for Claude Code."""
        if isinstance(tool_use_id, str) and tool_use_id:
            self.display.refused.add(tool_use_id)
        self.display.line(f"  ! not allowed: {summarize_tool_call(tool_name, tool_input)} ({reason})")
        result = self.sdk.PermissionResultDeny(message=message or refusal_message(reason))
        if answer is not None and not answer.done():
            try:
                answer.set_result(result)
            except concurrent.futures.InvalidStateError:
                pass
        return result

    def _stderr(self, text: str) -> None:
        self.display.line(f"  [Claude Code stderr] {text}")


# ----------------------------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------------------------

def check_report(sdk: Any, args: argparse.Namespace, env: Mapping[str, str]) -> list[str]:
    """The settings a session would run with, read back from the constructed ClaudeAgentOptions."""

    async def never_called(*_: Any) -> Any:
        raise RuntimeError("--check runs no tools")

    instructions = project_instructions(REPO_ROOT)
    options = sdk.ClaudeAgentOptions(**option_values(
        args, never_called, hooks=pre_tool_use_hooks(sdk, never_called), instructions=instructions))
    get = lambda name: getattr(options, name, None)  # noqa: E731
    version = getattr(sdk, "__version__", "?")
    pin = "" if version == SDK_VERSION_READ else f" (this runner was written against {SDK_VERSION_READ})"
    cwd = Path(get("cwd") or ".")
    if get("cli_path"):
        cli = f"{get('cli_path')} (given with --cli)"
    elif bundled_cli(sdk):
        cli = f"bundled with the SDK, version {bundled_cli_version(sdk) or '?'}: {bundled_cli(sdk)}"
    else:
        cli = "not bundled with this SDK; the SDK looks for claude on PATH (Windows: claude.exe)"
    prompt = get("system_prompt")
    preset = isinstance(prompt, Mapping) and prompt.get("preset") == "claude_code"
    prompt_text = ("Claude Code's own (preset claude_code)" if preset else repr(prompt))
    if preset and prompt.get("append"):
        prompt_text += f", followed by {cwd / 'CLAUDE.md'}"
    elif preset:
        prompt_text += f"; {cwd / 'CLAUDE.md'} NOT found, so no project instructions"
    sources = get("setting_sources")
    ignored = [name for name in (".claude/settings.json", ".claude/settings.local.json", ".mcp.json")
               if (cwd / name).is_file()]
    settings = (f"Setting sources: {sources} (no settings file is loaded: no hooks, permission rules, "
                "environment, MCP servers or skills from the repository's .claude/ or .mcp.json, nor from "
                "your user folder)")
    if ignored:
        settings += f"; ignored here: {', '.join(ignored)}"
    hooks = get("hooks") or {}
    hooked = "a PreToolUse hook sees every tool call" if hooks.get("PreToolUse") else "NO PreToolUse hook"
    budget = get("max_budget_usd")
    other_reads = sorted(set(READ_ONLY_FILE_TOOLS) - {"Read"})
    cli_env = get("env") or {}
    return [
        f"Claude Agent SDK {version}{pin}: {Path(getattr(sdk, '__file__', '?')).parent}",
        f"Claude Code: {cli}",
        f"Working directory: {cwd}",
        settings,
        f"System prompt: {prompt_text}",
        f"Model: {get('model')}",
        f"Effort: {get('effort') or 'Claude Code default'}",
        f"Budget cap: {format_usd(budget) if budget is not None else 'none'}",
        f"Permission mode: {get('permission_mode')}; {hooked}. Allowed without asking: Read inside the "
        f"repository (with other Claude Code builds also {', '.join(other_reads)} inside the repository and "
        f"{', '.join(sorted(READ_ONLY_SESSION_TOOLS))}). Every other tool call asks y/n, every Bash command "
        f"included (Claude Code {CLI_VERSION_TRIED} has no Grep or Glob tool: searching goes through Bash). "
        "Refused without asking: agents with isolation (worktree or remote) and background runs",
        f"Tools switched off: {', '.join(get('disallowed_tools') or []) or 'none'}",
        f"Environment for Claude Code: {', '.join(f'{k}={v}' for k, v in sorted(cli_env.items())) or 'nothing added'}"
        " (no background tasks, no scheduled prompts)",
        f"Streaming: include_partial_messages = {get('include_partial_messages')}",
        f"Credentials: {credential_text(credential_mode(env))}",
        "No API call was made and Claude Code was not started.",
    ]


def repl(session: Session) -> int:
    while True:
        try:
            text = session.ask("\nyou> ")
        except EOFError:
            return 0
        except KeyboardInterrupt:
            session.display.line("\n  [Ctrl+C at the prompt does nothing; type /exit to quit]")
            continue
        if text.strip() in ("/exit", "/quit"):
            return 0
        if text.strip():
            session.turn(text)


def _safe_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # text the console cannot show must not crash the run
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    args = parse_args(argv)
    env = os.environ if env is None else env
    _safe_output()
    sdk = load_sdk()
    if sdk is None:
        print(f"The Claude Agent SDK is not installed in this Python ({sys.executable}).\n"
              f"Install it from the repository root with:\n    {INSTALL_HINT}")
        return 2
    if args.check:
        print("\n".join(check_report(sdk, args, env)))
        return 0
    if credential_mode(env) is None:
        print(NO_KEY_MESSAGE)
        return 2
    session = Session(sdk, args, env=env)
    code = 0
    try:
        session.open()
        session.banner()
        if args.once is not None:
            code = 0 if session.turn(args.once) else 1
        else:
            code = repl(session)
    except CredentialError as error:
        session.display.line(f"Stopped: {error}")
        code = 2
    except SessionEnded as error:
        session.display.line(f"Stopped: {error}")
        code = 1
    except KeyboardInterrupt:
        code = 130
    except sdk.ClaudeSDKError as error:
        session.display.line(f"Claude Code failed: {error}")
        code = 1
    finally:
        try:
            session.close()
        except KeyboardInterrupt:
            code = 130
        if code != 2:
            session.summary()
    return code


if __name__ == "__main__":
    sys.exit(main())
