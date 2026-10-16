"""Set up and check Minecraft Hivemind server instances (DESIGN.md 11.2, 11.3; bridge/server/README.md).

Standard library only, so it runs on the owner's Windows PC with any Python 3.11+.

Subcommands:
  render            write an instance's server.properties from the frozen template
  check-properties  compare an instance's server.properties (as rewritten by the server) with the template
  rcon              send commands over RCON, from --file and/or the command line
  check-gamerules   read the frozen gamerules and the difficulty back over RCON and compare
  check-mods        hash server.jar and mods/*.jar and compare them with mods-1.21.1.json
  check-generator   compare vanilla data-generator output with the pinned sources and data/rules-1.21.1.json

Instance K (0-9) uses game port 25565 + K and RCON port 25575 + K, both bound to 127.0.0.1.
The RCON password comes from the environment variable HIVEMIND_RCON_PASSWORD and is never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import struct
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TEMPLATE = HERE / "server.properties"
GAMERULES = HERE / "gamerules-1.21.1.mcfunction"
MODS = HERE / "mods-1.21.1.json"
RULES = REPO / "data" / "rules-1.21.1.json"
SOURCES = REPO / "data" / "sources.json"
MCMETA_DATA = REPO / "data" / "sources" / "mcmeta-1.21.1-data"

GAME_PORT_BASE = 25565  # [mem] vanilla default server-port
RCON_PORT_BASE = 25575  # [24] DESIGN.md 11.2
MAX_INSTANCES = 10  # game ports 25565..25574 stay below the RCON base
RCON_HOST = "127.0.0.1"  # DESIGN.md 11.2: RCON localhost only
PASSWORD_ENV = "HIVEMIND_RCON_PASSWORD"
MIN_PASSWORD_LEN = 16  # [est] long enough for a random token; RCON sends it in plain text
SIDECAR = "hivemind-instance.json"
CONTROLS = ("none", "pvp-off", "keep-inventory")  # DESIGN.md 10.2


# ---------------------------------------------------------------------------------------------
# Java .properties (the subset Minecraft reads and writes)
# ---------------------------------------------------------------------------------------------

def _logical_lines(text: str) -> Iterable[str]:
    """Yield logical lines: comments and blank lines dropped, backslash continuations joined."""
    buf: str | None = None
    for raw in text.splitlines():
        line = raw.lstrip(" \t\f")
        if buf is None and (not line or line[0] in "#!"):
            continue
        trailing = len(line) - len(line.rstrip("\\"))
        if trailing % 2 == 1:
            buf = (buf or "") + line[:-1]
            continue
        yield (buf or "") + line
        buf = None
    if buf is not None:
        yield buf


def _split_key_value(line: str) -> tuple[str, str]:
    """Split at the first unescaped '=', ':' or whitespace, as java.util.Properties does."""
    i = 0
    while i < len(line):
        if line[i] == "\\":
            i += 2
            continue
        if line[i] in "=: \t\f":
            break
        i += 1
    j = i
    while j < len(line) and line[j] in " \t\f":
        j += 1
    if j < len(line) and line[j] in "=:":
        j += 1
        while j < len(line) and line[j] in " \t\f":
            j += 1
    return line[:i], line[j:]


_ESCAPES = {"t": "\t", "n": "\n", "r": "\r", "f": "\f"}


def _unescape(s: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c != "\\" or i + 1 == len(s):
            out.append(c)
            i += 1
            continue
        nxt = s[i + 1]
        if nxt == "u" and i + 6 <= len(s):
            out.append(chr(int(s[i + 2:i + 6], 16)))
            i += 6
        else:
            out.append(_ESCAPES.get(nxt, nxt))
            i += 2
    return "".join(out)


def escape_value(value: str) -> str:
    """Escape a value the way java.util.Properties.store does (ASCII values only)."""
    out = []
    for n, c in enumerate(value):
        if c in "\\=:#!":
            out.append("\\" + c)
        elif c == " " and n == 0:
            out.append("\\ ")
        elif c in "\t\n\r\f":
            out.append("\\" + {"\t": "t", "\n": "n", "\r": "r", "\f": "f"}[c])
        else:
            out.append(c)
    return "".join(out)


def parse_properties(text: str) -> dict[str, str]:
    """Parse .properties text into {key: value} (later keys win, as in Java)."""
    props: dict[str, str] = {}
    for line in _logical_lines(text):
        key, value = _split_key_value(line)
        props[_unescape(key)] = _unescape(value)
    return props


# ---------------------------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------------------------

def ports(instance: int) -> tuple[int, int]:
    """(game port, RCON port) of an instance."""
    if not 0 <= instance < MAX_INSTANCES:
        raise SystemExit(f"instance must be 0..{MAX_INSTANCES - 1}, got {instance}")
    return GAME_PORT_BASE + instance, RCON_PORT_BASE + instance


def rcon_password(required: bool = True) -> str | None:
    """The RCON password from the environment, validated; None if absent and not required."""
    pw = os.environ.get(PASSWORD_ENV, "")
    if not pw:
        if required:
            raise SystemExit(
                f"{PASSWORD_ENV} is not set. Create one, e.g.\n"
                f'  python -c "import secrets; print(secrets.token_urlsafe(24))"\n'
                f"and set it (PowerShell: $env:{PASSWORD_ENV} = '<token>'; setx {PASSWORD_ENV} <token>)."
            )
        return None
    if len(pw) < MIN_PASSWORD_LEN:
        raise SystemExit(f"{PASSWORD_ENV} must be at least {MIN_PASSWORD_LEN} characters")
    if not all(33 <= ord(c) < 127 for c in pw):
        raise SystemExit(f"{PASSWORD_ENV} must be printable ASCII without spaces")
    return pw


def overrides(instance: int, password: str | None, seed: str | None, control: str) -> dict[str, str]:
    """The only keys an instance may change relative to the frozen template."""
    game, rcon = ports(instance)
    o = {"server-port": str(game), "query.port": str(game), "rcon.port": str(rcon)}
    if password is not None:
        o["rcon.password"] = password
    if seed is not None:
        o["level-seed"] = seed
    if control == "pvp-off":
        o["pvp"] = "false"
    return o


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def render_text(template: str, values: dict[str, str]) -> str:
    """Replace the values of the given keys in the template text, keeping everything else."""
    seen: set[str] = set()
    out = []
    for raw in template.splitlines():
        stripped = raw.lstrip(" \t\f")
        if stripped and stripped[0] not in "#!":
            key = _unescape(_split_key_value(stripped)[0])
            if key in values:
                raw = f"{key}={escape_value(values[key])}"
                seen.add(key)
        out.append(raw)
    missing = set(values) - seen
    if missing:
        raise SystemExit(f"template has no line for {sorted(missing)}")
    return "\n".join(out) + "\n"


def cmd_render(args: argparse.Namespace) -> int:
    pw = rcon_password(required=True)
    target = Path(args.dir)
    target.mkdir(parents=True, exist_ok=True)
    values = overrides(args.instance, pw, args.seed, args.control)
    header = (
        f"# Rendered by bridge/server/servertool.py for instance {args.instance} "
        f"(template sha256 {sha256_file(TEMPLATE)[:16]}). Do not edit; re-render instead.\n"
    )
    (target / "server.properties").write_text(header + render_text(TEMPLATE.read_text("utf-8"), values),
                                             encoding="utf-8")
    game, rcon = ports(args.instance)
    sidecar = {
        "instance": args.instance,
        "game_port": game,
        "rcon_port": rcon,
        "seed": args.seed,
        "control": args.control,
        "template_sha256": sha256_file(TEMPLATE),
        "gamerules_sha256": sha256_file(GAMERULES),
        "rendered_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (target / SIDECAR).write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target / 'server.properties'} (game port {game}, RCON port {rcon}, password from {PASSWORD_ENV})")
    if not (target / "eula.txt").exists():
        print("note: eula.txt is not written by this tool. Read the EULA and set eula=true yourself.")
    return 0


def cmd_check_properties(args: argparse.Namespace) -> int:
    live_path = Path(args.dir) / "server.properties"
    live = parse_properties(live_path.read_text("utf-8"))
    expected = parse_properties(TEMPLATE.read_text("utf-8"))
    pw = rcon_password(required=False)
    expected.update(overrides(args.instance, pw, args.seed, args.control))
    problems = []
    for key, want in expected.items():
        if key == "level-seed" and args.seed is None:
            continue
        if key == "rcon.password":
            have = live.get(key, "")
            if not have:
                problems.append("rcon.password is empty: RCON will not start")
            elif pw is not None and have != pw:
                problems.append(f"rcon.password differs from {PASSWORD_ENV}")
            continue
        if key not in live:
            problems.append(f"{key}: missing (frozen value {want!r})")
        elif live[key] != want:
            problems.append(f"{key}: {live[key]!r} != frozen {want!r}")
    added = sorted(set(live) - set(expected))
    for key in added:
        problems.append(f"{key}={live[key]!r}: known to this server but not in the frozen template; "
                        f"add it to bridge/server/server.properties")
    for p in problems:
        print("DIFF", p)
    print(f"{'FAIL' if problems else 'PASS'} {live_path}: {len(expected)} frozen keys, {len(problems)} problems")
    return 1 if problems else 0


# ---------------------------------------------------------------------------------------------
# RCON (Source RCON protocol as used by Minecraft [mem]: little-endian int32 length, id, type)
# ---------------------------------------------------------------------------------------------

RCON_LOGIN, RCON_COMMAND, RCON_RESPONSE = 3, 2, 0  # [mem] packet types
RCON_CHUNK = 4096  # [mem] the server splits long replies into packets of this payload size
_ERROR_REPLY = re.compile(r"^(Unknown or incomplete command|Incorrect argument for command|Unknown game ?rule)",
                          re.IGNORECASE)  # [mem] vanilla error texts


class RconError(Exception):
    pass


@dataclass
class Rcon:
    """A minimal blocking RCON client. Use as a context manager."""

    port: int
    password: str
    host: str = RCON_HOST
    timeout: float = 10.0
    _sock: socket.socket | None = field(default=None, repr=False)
    _next_id: int = 1

    def __enter__(self) -> Self:
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        req = self._send(RCON_LOGIN, self.password)
        while True:
            rid, ptype, _ = self._recv()
            if ptype == RCON_COMMAND:  # the auth response uses type 2 [mem]
                if rid == -1:
                    raise RconError("RCON login refused: wrong password")
                if rid == req:
                    return self

    def __exit__(self, *exc: object) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def command(self, cmd: str) -> str:
        """Run one command and return the server's reply text."""
        req = self._send(RCON_COMMAND, cmd)
        parts: list[str] = []
        while True:
            rid, _, payload = self._recv()
            if rid == req:
                parts.append(payload)
                if len(payload.encode("utf-8")) < RCON_CHUNK:
                    return "".join(parts)
                break
        # A reply that filled a whole packet may continue in more packets; wait briefly for them.
        # Only long replies get here (not gamerule or /tick replies).
        self._sock.settimeout(0.5)
        try:
            while True:
                rid, _, payload = self._recv()
                if rid == req:
                    parts.append(payload)
                    if len(payload.encode("utf-8")) < RCON_CHUNK:
                        break
        except TimeoutError:
            pass
        finally:
            self._sock.settimeout(self.timeout)
        return "".join(parts)

    def _send(self, ptype: int, payload: str) -> int:
        rid = self._next_id
        self._next_id += 1
        body = struct.pack("<ii", rid, ptype) + payload.encode("utf-8") + b"\x00\x00"
        self._sock.sendall(struct.pack("<i", len(body)) + body)
        return rid

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise RconError("RCON connection closed")
            buf += chunk
        return bytes(buf)

    def _recv(self) -> tuple[int, int, str]:
        (length,) = struct.unpack("<i", self._recv_exact(4))
        if not 10 <= length <= 1 << 20:
            raise RconError(f"bad RCON packet length {length}")
        data = self._recv_exact(length)
        rid, ptype = struct.unpack("<ii", data[:8])
        return rid, ptype, data[8:-2].decode("utf-8", errors="replace")


def read_commands(path: Path) -> list[str]:
    """Non-comment, non-blank lines of an .mcfunction file, without a leading slash."""
    cmds = []
    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            cmds.append(line.lstrip("/"))
    return cmds


def _rcon_port(args: argparse.Namespace) -> int:
    return args.port if args.port is not None else ports(args.instance)[1]


def cmd_rcon(args: argparse.Namespace) -> int:
    cmds = (read_commands(Path(args.file)) if args.file else []) + list(args.commands)
    if not cmds:
        raise SystemExit("nothing to send: give --file and/or commands")
    failed = 0
    with Rcon(_rcon_port(args), rcon_password()) as rc:
        for cmd in cmds:
            reply = rc.command(cmd)
            bad = bool(_ERROR_REPLY.match(reply))
            failed += bad
            print(f"{'ERR' if bad else 'ok '} {cmd}\n    -> {reply}")
    return 1 if failed else 0


_GAMERULE_REPLY = re.compile(r"currently set to:\s*(\S+)")  # [mem] "Gamerule X is currently set to: Y"
_DIFFICULTY_REPLY = re.compile(r"difficulty is (\w+)", re.IGNORECASE)  # [mem] "The difficulty is Normal"


def frozen_gamerules(control: str) -> dict[str, str]:
    """{rule: value} from the active lines of the gamerules file, with a control applied."""
    rules = {}
    for cmd in read_commands(GAMERULES):
        parts = cmd.split()
        if len(parts) == 3 and parts[0] == "gamerule":
            rules[parts[1]] = parts[2]
    if control == "keep-inventory":
        rules["keepInventory"] = "true"
    return rules


def cmd_check_gamerules(args: argparse.Namespace) -> int:
    want = frozen_gamerules(args.control)
    want_difficulty = parse_properties(TEMPLATE.read_text("utf-8"))["difficulty"]
    problems = 0
    with Rcon(_rcon_port(args), rcon_password()) as rc:
        for rule, value in want.items():
            reply = rc.command(f"gamerule {rule}")
            m = _GAMERULE_REPLY.search(reply)
            ok = bool(m) and m.group(1).lower() == value.lower()
            problems += not ok
            print(f"{'ok  ' if ok else 'DIFF'} {rule} = {value}   (server: {reply!r})")
        reply = rc.command("difficulty")
        m = _DIFFICULTY_REPLY.search(reply)
        ok = bool(m) and m.group(1).lower() == want_difficulty
        problems += not ok
        print(f"{'ok  ' if ok else 'DIFF'} difficulty = {want_difficulty}   (server: {reply!r})")
    print(f"{'FAIL' if problems else 'PASS'} {len(want) + 1} checks, {problems} problems")
    return 1 if problems else 0


# ---------------------------------------------------------------------------------------------
# Mods and jars
# ---------------------------------------------------------------------------------------------

def _pinned(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def cmd_check_mods(args: argparse.Namespace) -> int:
    pins = json.loads(MODS.read_text("utf-8"))
    inst = Path(args.dir)
    jars = {p.name: sha256_file(p) for p in sorted((inst / "mods").glob("*.jar"))}
    problems = []
    server_jar = inst / pins["server_jar"]["file"]
    if server_jar.exists():
        digest = sha256_file(server_jar)
        sha1 = hashlib.sha1(server_jar.read_bytes()).hexdigest()
        print(f"server.jar sha256 {digest}  sha1 {sha1}")
        if _pinned(pins["server_jar"]["sha256"]) and digest != pins["server_jar"]["sha256"]:
            problems.append("server.jar differs from the pinned sha256")
    else:
        problems.append(f"{server_jar} not found")
    unpinned = [m["name"] for m in pins["mods"] if not _pinned(m["sha256"])]
    pinned = {m["sha256"]: m["name"] for m in pins["mods"] if _pinned(m["sha256"])}
    for digest, name in pinned.items():
        if digest not in jars.values():
            problems.append(f"pinned mod {name} not found in mods/")
    for jar, digest in jars.items():
        status = f"= {pinned[digest]}" if digest in pinned else "not pinned"
        print(f"mods/{jar} sha256 {digest}  {status}")
        if digest not in pinned and not unpinned:
            problems.append(f"mods/{jar} is not in the pinned set")
    if unpinned:
        problems.append(f"NOT FROZEN: still TODO-pin in {MODS.name}: {', '.join(unpinned)}")
    for p in problems:
        print("DIFF", p)
    print(f"{'FAIL' if problems else 'PASS'} {len(jars)} mod jars, {len(problems)} problems")
    return 1 if problems else 0


# ---------------------------------------------------------------------------------------------
# Data generator cross-check (DESIGN.md 4.1)
# ---------------------------------------------------------------------------------------------

@dataclass
class Check:
    """Outcome of one comparison: differences found, items compared, or a reason it was skipped."""

    name: str
    compared: int = 0
    diffs: list[str] = field(default_factory=list)
    skipped: str | None = None

    def report(self, max_examples: int) -> bool:
        if self.skipped:
            print(f"SKIP {self.name}: {self.skipped}")
            return True
        if not self.diffs:
            print(f"PASS {self.name} ({self.compared} compared)")
            return True
        print(f"FAIL {self.name}: {len(self.diffs)} differences ({self.compared} compared)")
        for d in self.diffs[:max_examples]:
            print(f"     {d}")
        return False


def _strip_ns(name: str) -> str:
    return name.split(":", 1)[1] if name.startswith("minecraft:") else name


def _json_files(root: Path) -> dict[str, Path]:
    return {p.relative_to(root).with_suffix("").as_posix(): p for p in root.rglob("*.json")}


def _load(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def check_against_mcmeta(gen: Path, mcmeta: Path, paths: list[str]) -> Check:
    """The generator's data files must equal the pinned mcmeta 1.21.1-data files (JSON-equal)."""
    c = Check("generator data == mcmeta 1.21.1-data (" + ", ".join(p.rsplit("/", 1)[-1] for p in paths) + ")")
    if not mcmeta.is_dir():
        c.skipped = f"{mcmeta} not found; run `python -m hivemind.rules.fetch_sources` first"
        return c
    for rel in paths:
        mine, theirs = _json_files(gen / rel), _json_files(mcmeta / rel)
        for key in sorted(set(mine) ^ set(theirs)):
            c.diffs.append(f"{rel}/{key}.json only in {'generator' if key in mine else 'mcmeta'}")
        for key in sorted(set(mine) & set(theirs)):
            c.compared += 1
            if _load(mine[key]) != _load(theirs[key]):
                c.diffs.append(f"{rel}/{key}.json differs")
    return c


def check_loot_tables(gen: Path, rules: dict) -> Check:
    c = Check("loot tables == rules (verbatim)")
    root = gen / "data" / "minecraft" / "loot_table"
    for key, table in rules["loot_tables"].items():
        path = root / f"{key}.json"
        c.compared += 1
        if not path.exists():
            c.diffs.append(f"{key}: no generator file")
        elif _load(path) != table:
            c.diffs.append(f"{key}: differs")
    return c


def _result_of(recipe: dict) -> tuple[str, int] | None:
    res = recipe.get("result")
    if isinstance(res, str):
        return _strip_ns(res), 1
    if isinstance(res, dict):
        item = res.get("id", res.get("item"))
        if item:
            return _strip_ns(item), int(res.get("count", 1))
    return None


def check_recipes(gen: Path, rules: dict) -> Check:
    c = Check("recipes: ids, results, cooking time and experience == rules")
    files = _json_files(gen / "data" / "minecraft" / "recipe")
    ours = {r["id"]: r for r in rules["recipes"]}
    for rid in sorted(set(files) ^ set(ours)):
        c.diffs.append(f"{rid}: only in {'generator' if rid in files else 'rules'}")
    for rid in sorted(set(files) & set(ours)):
        c.compared += 1
        g, r = _load(files[rid]), ours[rid]
        if r.get("result") is not None:
            want = (r["result"]["item"], int(r["result"].get("count", 1)))
            if _result_of(g) != want:
                c.diffs.append(f"{rid}: result {_result_of(g)} != rules {want}")
        for key in ("cookingtime", "experience"):
            if key in g and r.get(key) is not None and abs(float(g[key]) - float(r[key])) > 1e-9:
                c.diffs.append(f"{rid}: {key} {g[key]} != rules {r[key]}")
    return c


def _resolve_tag(root: Path, tag: str, seen: set[str] | None = None) -> set[str]:
    seen = set() if seen is None else seen
    if tag in seen:
        return set()
    seen.add(tag)
    path = root / f"{tag}.json"
    if not path.exists():
        raise KeyError(tag)
    out: set[str] = set()
    for value in _load(path).get("values", []):
        ref = value["id"] if isinstance(value, dict) else value
        if ref.startswith("#"):
            try:
                out |= _resolve_tag(root, _strip_ns(ref[1:]), seen)
            except KeyError:
                if not (isinstance(value, dict) and value.get("required") is False):
                    raise
        else:
            out.add(_strip_ns(ref))
    return out


def check_tags(gen: Path, rules: dict) -> Check:
    c = Check("block and item tags (resolved) == rules")
    for kind in ("block", "item"):
        base = gen / "data" / "minecraft" / "tags"
        root = base / kind if (base / kind).is_dir() else base / f"{kind}s"
        for tag, members in rules["tags"][kind].items():
            c.compared += 1
            try:
                got = _resolve_tag(root, tag)
            except KeyError as e:
                c.diffs.append(f"{kind} tag {tag}: generator has no tag {e}")
                continue
            if got != set(members):
                c.diffs.append(f"{kind} tag {tag}: +{sorted(got - set(members))[:5]} -{sorted(set(members) - got)[:5]}")
    return c


def _report(gen: Path, name: str, c: Check) -> dict | None:
    path = gen / "reports" / name
    if not path.exists():
        c.skipped = f"{path} not found (run the generator with --reports)"
        return None
    data = _load(path)
    if not isinstance(data, dict):
        c.skipped = f"{name}: unexpected format"
        return None
    return data


def check_block_states(gen: Path, rules: dict) -> Check:
    c = Check("reports/blocks.json state ids == rules (min, max, default)")
    report = _report(gen, "blocks.json", c)
    if report is None:
        return c
    names = {_strip_ns(k): v for k, v in report.items()}
    for name in sorted(set(names) ^ set(rules["blocks"])):
        c.diffs.append(f"{name}: only in {'report' if name in names else 'rules'}")
    for name in sorted(set(names) & set(rules["blocks"])):
        states = names[name].get("states")
        if not isinstance(states, list) or not states:
            c.skipped = f"{name}: no 'states' list; report format not recognised"
            return c
        ids = [s["id"] for s in states]
        default = [s["id"] for s in states if s.get("default")]
        got = (min(ids), max(ids), default[0] if default else None)
        b = rules["blocks"][name]
        want = (b["min_state"], b["max_state"], b["default_state"])
        c.compared += 1
        if got != want:
            c.diffs.append(f"{name}: (min, max, default) {got} != rules {want}")
    return c


def check_items(gen: Path, rules: dict) -> Check:
    c = Check("reports/items.json components == rules (stack size, durability, food)")
    report = _report(gen, "items.json", c)
    if report is None:
        return c
    names = {_strip_ns(k): v for k, v in report.items()}
    for name in sorted(set(names) ^ set(rules["items"])):
        c.diffs.append(f"{name}: only in {'report' if name in names else 'rules'}")
    for name in sorted(set(names) & set(rules["items"])):
        comps = names[name].get("components", names[name])
        it = rules["items"][name]
        food = comps.get("minecraft:food")
        got = (comps.get("minecraft:max_stack_size"), comps.get("minecraft:max_damage"),
               None if food is None else (food["nutrition"], round(float(food["saturation"]), 6)))
        want = (it["stack_size"], it["max_damage"],
                None if it["food"] is None else (it["food"]["nutrition"], round(float(it["food"]["saturation"]), 6)))
        c.compared += 1
        if got != want:
            c.diffs.append(f"{name}: (stack, durability, food) {got} != rules {want}")
    return c


def check_registry_ids(gen: Path, rules: dict) -> Check:
    c = Check("reports/registries.json protocol ids == rules ids (block, item, entity_type)")
    report = _report(gen, "registries.json", c)
    if report is None:
        return c
    for registry, section in (("minecraft:block", "blocks"), ("minecraft:item", "items"),
                              ("minecraft:entity_type", "entities")):
        entries = report.get(registry, {}).get("entries")
        if not isinstance(entries, dict):
            c.diffs.append(f"{registry}: no entries in report")
            continue
        got = {_strip_ns(k): v.get("protocol_id") for k, v in entries.items()}
        ours = rules[section]
        for name in sorted(set(got) ^ set(ours)):
            c.diffs.append(f"{registry} {name}: only in {'report' if name in got else 'rules'}")
        for name in sorted(set(got) & set(ours)):
            c.compared += 1
            if got[name] != ours[name]["id"]:
                c.diffs.append(f"{registry} {name}: protocol id {got[name]} != rules id {ours[name]['id']}")
    return c


def cmd_check_generator(args: argparse.Namespace) -> int:
    gen = Path(args.generated)
    if not (gen / "data").is_dir() and not (gen / "reports").is_dir():
        raise SystemExit(f"{gen} has neither data/ nor reports/; point --generated at the generator's output folder")
    rules = _load(Path(args.rules))
    if rules.get("version") != "1.21.1":
        raise SystemExit(f"{args.rules} is for version {rules.get('version')}, not 1.21.1")
    paths = _load(SOURCES)["sources"]["mcmeta-data"]["paths"]
    checks = [
        check_against_mcmeta(gen, Path(args.mcmeta), paths),
        check_loot_tables(gen, rules),
        check_recipes(gen, rules),
        check_tags(gen, rules),
        check_block_states(gen, rules),
        check_items(gen, rules),
        check_registry_ids(gen, rules),
    ]
    ok = [c.report(args.max_examples) for c in checks]
    print("PASS" if all(ok) else "FAIL", f"generator cross-check against {Path(args.rules).name}")
    return 0 if all(ok) else 1


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def instance_args(sp: argparse.ArgumentParser, with_control: bool = True) -> None:
        sp.add_argument("--instance", type=int, default=0, help="instance number K (ports 25565+K, 25575+K)")
        if with_control:
            sp.add_argument("--control", choices=CONTROLS, default="none", help="DESIGN.md 10.2 control run")

    sp = sub.add_parser("render", help="write <dir>/server.properties for an instance")
    instance_args(sp)
    sp.add_argument("--dir", required=True, help="instance directory")
    sp.add_argument("--seed", help="level-seed for a new world (from the world pool)")
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("check-properties", help="compare <dir>/server.properties with the frozen template")
    instance_args(sp)
    sp.add_argument("--dir", required=True)
    sp.add_argument("--seed", help="also check level-seed")
    sp.set_defaults(func=cmd_check_properties)

    sp = sub.add_parser("rcon", help="send commands over RCON")
    instance_args(sp, with_control=False)
    sp.add_argument("--port", type=int, help="RCON port (overrides --instance)")
    sp.add_argument("--file", help=".mcfunction file: one command per non-comment line")
    sp.add_argument("commands", nargs="*", help="commands, e.g. 'tick query'")
    sp.set_defaults(func=cmd_rcon)

    sp = sub.add_parser("check-gamerules", help="read the frozen gamerules back over RCON")
    instance_args(sp)
    sp.add_argument("--port", type=int, help="RCON port (overrides --instance)")
    sp.set_defaults(func=cmd_check_gamerules)

    sp = sub.add_parser("check-mods", help="hash server.jar and mods/*.jar against mods-1.21.1.json")
    sp.add_argument("--dir", required=True, help="instance directory")
    sp.set_defaults(func=cmd_check_mods)

    sp = sub.add_parser("check-generator", help="compare data-generator output with the pinned sources and rules")
    sp.add_argument("--generated", required=True, help="the generator's output folder (contains data/ and reports/)")
    sp.add_argument("--rules", default=str(RULES))
    sp.add_argument("--mcmeta", default=str(MCMETA_DATA), help="mcmeta 1.21.1-data checkout")
    sp.add_argument("--max-examples", type=int, default=10)
    sp.set_defaults(func=cmd_check_generator)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, RconError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
