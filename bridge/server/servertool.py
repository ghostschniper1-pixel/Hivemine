"""Set up and check Minecraft Hivemind server instances (DESIGN.md 11.2, 11.3; bridge/server/README.md).

Standard library only, so it runs on the owner's Windows PC with any Python 3.11+.

Subcommands:
  render            write an instance's server.properties from the frozen template
  check-properties  compare an instance's server.properties (as rewritten by the server) with the template
  rcon              send commands over RCON, from --file and/or the command line
  check-gamerules   read every gamerule and the difficulty back over RCON and compare
  check-mods        check Java, server.jar, the Fabric launcher and mods/*.jar against mods-1.21.1.json
  check-generator   compare vanilla data-generator output with the pinned sources and data/rules-1.21.1.json

Instance K (0-9) uses game port 25565 + K and RCON port 25575 + K, both bound to 127.0.0.1.
The RCON password comes from the environment variable HIVEMIND_RCON_PASSWORD and is never printed.

Exit codes: 0 every check passed, 1 a difference was found, 2 an I/O or RCON error,
3 (check-generator only) INCOMPLETE: nothing differed, but some checks could not run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import struct
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TEMPLATE = HERE / "server.properties"
GAMERULES = HERE / "gamerules-1.21.1.mcfunction"
GAMERULE_DEFAULTS = HERE / "gamerule-defaults-1.21.1.json"
MODS = HERE / "mods-1.21.1.json"
RULES = REPO / "data" / "rules-1.21.1.json"
SOURCES = REPO / "data" / "sources.json"
MCMETA_DATA = REPO / "data" / "sources" / "mcmeta-1.21.1-data"

GAME_VERSION = "1.21.1"  # contracts section 0
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
    """SHA-256 of a file's bytes (jars and other binary files)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(path: Path) -> str:
    """SHA-256 of a committed text file with CRLF and CR turned into LF.

    The hash then does not depend on how Git checked the file out (Git for Windows may use CRLF).
    """
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def read_sidecar(directory: Path) -> dict[str, Any] | None:
    """The hivemind-instance.json that `render` wrote into an instance folder, or None."""
    path = directory / SIDECAR
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))


@dataclass(frozen=True)
class InstanceSettings:
    """What an instance was rendered with: from its sidecar if present, else from the command line."""

    instance: int
    seed: str | None
    control: str
    seed_known: bool  # False: no sidecar and no --seed, so level-seed cannot be checked
    from_sidecar: bool


def instance_settings(args: argparse.Namespace) -> InstanceSettings:
    """Resolve instance, seed and control for a check; command-line values must agree with the sidecar."""
    cli = {"instance": args.instance, "seed": getattr(args, "seed", None), "control": args.control}
    directory = getattr(args, "dir", None)
    side = read_sidecar(Path(directory)) if directory else None
    if side is None:
        seed = cli["seed"]
        return InstanceSettings(instance=0 if cli["instance"] is None else cli["instance"], seed=seed,
                                control=cli["control"] or "none", seed_known=seed is not None,
                                from_sidecar=False)
    conflicts = [f"--{key} {value!r} but {SIDECAR} has {side.get(key)!r}"
                 for key, value in cli.items() if value is not None and value != side.get(key)]
    if conflicts:
        raise SystemExit(f"{Path(directory) / SIDECAR} disagrees with the command line: " + "; ".join(conflicts)
                         + ". Re-render the instance, or drop the argument.")
    return InstanceSettings(instance=int(side["instance"]), seed=side.get("seed"),
                            control=side.get("control") or "none", seed_known=True, from_sidecar=True)


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
        f"(template sha256 {sha256_text(TEMPLATE)[:16]}). Do not edit; re-render instead.\n"
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
        "template_sha256": sha256_text(TEMPLATE),
        "gamerules_sha256": sha256_text(GAMERULES),
        "gamerule_defaults_sha256": sha256_text(GAMERULE_DEFAULTS),
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
    s = instance_settings(args)
    origin = SIDECAR if s.from_sidecar else "the command line"
    print(f"instance {s.instance}, seed {s.seed!r}, control {s.control} (from {origin})")
    expected.update(overrides(s.instance, pw, s.seed, s.control))
    problems = []
    for key, want in expected.items():
        if key == "level-seed" and not s.seed_known:
            print(f"NOTE level-seed not checked: no {SIDECAR} in {args.dir} and no --seed")
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


def _rule_text(value: Any) -> str:
    """A gamerule value as the server prints it: true/false or a decimal integer."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def gamerule_defaults(path: Path = GAMERULE_DEFAULTS) -> dict[str, str]:
    """{rule: vanilla default} for every 1.21.1 gamerule (names checked; defaults see defaults_origin)."""
    return {rule: _rule_text(v) for rule, v in json.loads(path.read_text("utf-8"))["rules"].items()}


def defaults_origin(path: Path = GAMERULE_DEFAULTS) -> str:
    """How check-gamerules labels a rule left at its default: verified on a real server, or still [mem]."""
    verified = json.loads(path.read_text("utf-8")).get("defaults_verified")
    return f"vanilla default, verified {verified}" if verified else "vanilla default [mem]"


def frozen_gamerules(control: str, path: Path = GAMERULES) -> dict[str, str]:
    """{rule: value} from the active lines of the gamerules file, with a control applied."""
    rules = {}
    for cmd in read_commands(path):
        parts = cmd.split()
        if parts[0] != "gamerule":
            continue  # another command (e.g. 'weather clear'): applied by `rcon --file`, not read back here
        if len(parts) != 3:
            raise SystemExit(f"{path.name}: expected 'gamerule <name> <value>', got {cmd!r}")
        rules[parts[1]] = parts[2]
    if control == "keep-inventory":
        rules["keepInventory"] = "true"
    return rules


def expected_gamerules(control: str) -> dict[str, tuple[str, str]]:
    """{rule: (value, origin)} for every gamerule: the vanilla default unless frozen or set by a control."""
    defaults = gamerule_defaults()
    frozen = frozen_gamerules(control)
    unknown = sorted(set(frozen) - set(defaults))
    if unknown:
        raise SystemExit(f"{GAMERULES.name} sets {unknown}, which are not 1.21.1 gamerules ({GAMERULE_DEFAULTS.name})")
    origin = defaults_origin()
    want = {rule: (value, origin) for rule, value in defaults.items()}
    want.update({rule: (value, "frozen") for rule, value in frozen.items()})
    return want


def cmd_check_gamerules(args: argparse.Namespace) -> int:
    s = instance_settings(args)
    port = args.port if args.port is not None else ports(s.instance)[1]
    want = expected_gamerules(s.control)
    want_difficulty = parse_properties(TEMPLATE.read_text("utf-8"))["difficulty"]
    print(f"instance {s.instance} (RCON port {port}), control {s.control}")
    problems = 0
    with Rcon(port, rcon_password()) as rc:
        for rule, (value, origin) in want.items():
            reply = rc.command(f"gamerule {rule}")
            m = _GAMERULE_REPLY.search(reply)
            ok = bool(m) and m.group(1).lower() == value.lower()
            problems += not ok
            line = f"{'ok  ' if ok else 'DIFF'} {rule} = {value} ({origin})"
            print(line if ok else f"{line}   server: {reply!r}")
        reply = rc.command("difficulty")
        m = _DIFFICULTY_REPLY.search(reply)
        ok = bool(m) and m.group(1).lower() == want_difficulty
        problems += not ok
        print(f"{'ok  ' if ok else 'DIFF'} difficulty = {want_difficulty} (frozen)   server: {reply!r}")
    if problems:
        print(f"note: on a fresh world, a DIFF on a 'vanilla default' rule means {GAMERULE_DEFAULTS.name} is wrong; "
              f"otherwise the world was changed")
    print(f"{'FAIL' if problems else 'PASS'} {len(want) + 1} checks, {problems} problems")
    return 1 if problems else 0


# ---------------------------------------------------------------------------------------------
# Mods and jars
# ---------------------------------------------------------------------------------------------

_HASH_FORMATS = {"sha256": re.compile(r"[0-9a-f]{64}"), "sha1": re.compile(r"[0-9a-f]{40}")}
_JAVA_VERSION = re.compile(r'version "(\d+)')  # [mem] first line of `java -version`, e.g. openjdk version "21.0.4"


def _is_todo(value: Any) -> bool:
    """True for a pin that is not filled in yet: missing, empty, or text starting with TODO or 'planned'."""
    if not isinstance(value, str):
        return value is None
    return not value.strip() or value.strip().lower().startswith(("todo", "planned"))


@dataclass
class PinCheck:
    """Collects the pins of mods-1.21.1.json: unpinned entries and problems."""

    unpinned: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def pin(self, label: str, value: Any, kind: str = "text") -> str | None:
        """The pinned value, or None if it is still TODO (recorded) or malformed (a problem)."""
        if _is_todo(value):
            self.unpinned.append(label)
            return None
        text = str(value).strip()
        if kind in _HASH_FORMATS and not _HASH_FORMATS[kind].fullmatch(text):
            self.problems.append(f"{label}: {value!r} is not a lower-case hex {kind}")
            return None
        return text


def java_version(java: str) -> str | None:
    """The output of `<java> -version` (Java prints it on stderr), or None if it cannot run."""
    try:
        out = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stderr or out.stdout).strip() or None


def _jar_hashes(path: Path) -> tuple[str, str]:
    """(sha256, sha1) of a file."""
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), hashlib.sha1(data).hexdigest()


def cmd_check_mods(args: argparse.Namespace) -> int:
    """Every entry of the pins file must be filled in and match the instance (README section 4, step 4)."""
    pins_path = Path(args.pins)
    pins = json.loads(pins_path.read_text("utf-8"))
    inst = Path(args.dir)
    pc = PinCheck()
    if pins.get("game_version") != GAME_VERSION:
        pc.problems.append(f"{pins_path.name} is for {pins.get('game_version')!r}, not {GAME_VERSION}")

    # Java: the major version always; the exact `java -version` line once it is pinned.
    java = pins["java"]
    java_pin = pc.pin("java.version", java.get("version"))
    out = java_version(args.java)
    if out is None:
        pc.problems.append(f"cannot run `{args.java} -version`; pass --java <the java that starts the server>")
    else:
        # The version line; skip notices such as "Picked up JAVA_TOOL_OPTIONS: ..." printed before it.
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        first = next((line for line in lines if _JAVA_VERSION.search(line)), lines[0])
        print(f"java: {first}")
        m = _JAVA_VERSION.search(first)
        if not m or int(m.group(1)) != int(java["major"]):
            pc.problems.append(f"java major version is not {java['major']}: {first!r}")
        if java_pin is not None and first != java_pin:
            pc.problems.append(f"java: {first!r} != pinned {java_pin!r}")

    # The vanilla server jar: SHA-256 and Mojang's SHA-1.
    sj = pins["server_jar"]
    pin256 = pc.pin("server_jar.sha256", sj.get("sha256"), "sha256")
    pin1 = pc.pin("server_jar.sha1", sj.get("sha1"), "sha1")
    server_jar = inst / sj["file"]
    if server_jar.exists():
        d256, d1 = _jar_hashes(server_jar)
        print(f"{sj['file']} sha256 {d256}  sha1 {d1}")
        if pin256 is not None and d256 != pin256:
            pc.problems.append(f"{sj['file']} differs from the pinned sha256")
        if pin1 is not None and d1 != pin1:
            pc.problems.append(f"{sj['file']} differs from the pinned sha1")
    else:
        pc.problems.append(f"{server_jar} not found (the Fabric launcher downloads it on the first start)")

    # The Fabric loader, through the jar that starts the server (launcher or fabric-server-launch.jar).
    loader = pins["loader"]
    pc.pin("loader.version", loader.get("version"))
    pc.pin("loader.installer_version", loader.get("installer_version"))
    launcher = pc.pin("loader.launcher_file", loader.get("launcher_file"))
    launcher_pin = pc.pin("loader.launcher_sha256", loader.get("launcher_sha256"), "sha256")
    if launcher is not None:
        path = inst / launcher
        if path.exists():
            digest = sha256_file(path)
            print(f"{launcher} sha256 {digest}")
            if launcher_pin is not None and digest != launcher_pin:
                pc.problems.append(f"{launcher} differs from the pinned launcher_sha256")
        else:
            pc.problems.append(f"{path} not found")
    else:
        for path in sorted(inst.glob("*.jar")):
            if path.name != sj["file"]:
                print(f"{path.name} sha256 {sha256_file(path)}  (launcher candidate)")

    # Mods: every pinned jar present, and once all are pinned, nothing else in mods/.
    mod_pins: dict[str, str] = {}
    for m in pins["mods"]:
        pc.pin(f"mods.{m['name']}.version", m.get("version"))
        digest = pc.pin(f"mods.{m['name']}.sha256", m.get("sha256"), "sha256")
        if digest is not None:
            mod_pins[digest] = m["name"]
    all_mods_pinned = len(mod_pins) == len(pins["mods"])
    jars = {p.name: sha256_file(p) for p in sorted((inst / "mods").glob("*.jar"))}
    for digest, name in mod_pins.items():
        if digest not in jars.values():
            pc.problems.append(f"pinned mod {name} not found in mods/")
    for jar, digest in jars.items():
        print(f"mods/{jar} sha256 {digest}  {'= ' + mod_pins[digest] if digest in mod_pins else 'not pinned'}")
        if digest not in mod_pins and all_mods_pinned:
            pc.problems.append(f"mods/{jar} is not in the pinned set")

    problems = pc.problems + ([f"NOT FROZEN: still TODO-pin in {pins_path.name}: {', '.join(pc.unpinned)}"]
                              if pc.unpinned else [])
    for p in problems:
        print("DIFF", p)
    print(f"{'FAIL' if problems else 'PASS'} {len(jars)} mod jars, {len(problems)} problems")
    return 1 if problems else 0


# ---------------------------------------------------------------------------------------------
# Data generator cross-check (DESIGN.md 4.1)
# ---------------------------------------------------------------------------------------------

@dataclass
class Check:
    """Outcome of one comparison: differences found, items compared, or a reason it could not finish."""

    name: str
    compared: int = 0
    diffs: list[str] = field(default_factory=list)
    skipped: str | None = None
    tolerated: list[str] = field(default_factory=list)  # differences explicitly allowed, each with its reason

    @property
    def status(self) -> str:
        """FAIL if any difference was found (even if the check then stopped), else SKIP if it did not
        finish, else PASS."""
        if self.diffs:
            return "FAIL"
        return "SKIP" if self.skipped else "PASS"

    def report(self, max_examples: int) -> str:
        """Print the outcome and return the status."""
        status = self.status
        if status == "PASS":
            extra = f", {len(self.tolerated)} tolerated" if self.tolerated else ""
            print(f"PASS {self.name} ({self.compared} compared{extra})")
            for t in self.tolerated[:max_examples]:
                print(f"     tolerated: {t}")
        elif status == "SKIP":
            print(f"SKIP {self.name}: {self.skipped}")
        else:
            print(f"FAIL {self.name}: {len(self.diffs)} differences ({self.compared} compared)")
            if self.skipped:
                print(f"     and the check stopped early: {self.skipped}")
            for d in self.diffs[:max_examples]:
                print(f"     {d}")
        return status


def verdict(statuses: Iterable[str]) -> str:
    """PASS only if every check ran and passed; FAIL if any differed; otherwise INCOMPLETE."""
    statuses = list(statuses)
    if "FAIL" in statuses:
        return "FAIL"
    return "INCOMPLETE" if any(s != "PASS" for s in statuses) else "PASS"


EXIT_CODES = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}


def _strip_ns(name: str) -> str:
    return name.split(":", 1)[1] if name.startswith("minecraft:") else name


def _json_files(root: Path) -> dict[str, Path]:
    return {p.relative_to(root).with_suffix("").as_posix(): p for p in root.rglob("*.json")}


def _load(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _has_data(c: Check, gen: Path) -> bool:
    """False (and the check skipped) when the generator output has no data/ folder."""
    if (gen / "data").is_dir():
        return True
    c.skipped = f"{gen / 'data'} not found"
    return False


# JSON lists the game treats as sets, keyed by a path fragment of the file and the keys leading to the list.
# The vanilla generator writes them in hash-set order, which need not match mcmeta's: on the owner's PC
# (2026-10-19) the four flat-world presets differed only in the order of settings.structure_overrides, a set of
# structure-set ids. Such files are compared with those lists sorted, and reported as tolerated, not as equal.
UNORDERED_LISTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("/worldgen/flat_level_generator_preset/", ("settings", "structure_overrides")),
)


def _with_sets_sorted(rel_key: str, data: Any) -> Any:
    """A copy of a data file with the lists of UNORDERED_LISTS that apply to it sorted."""
    data = json.loads(json.dumps(data))
    for fragment, keys in UNORDERED_LISTS:
        if fragment not in f"/{rel_key}":
            continue
        node = data
        for k in keys[:-1]:
            node = node.get(k) if isinstance(node, dict) else None
        if isinstance(node, dict) and isinstance(node.get(keys[-1]), list):
            node[keys[-1]] = sorted(node[keys[-1]], key=lambda v: json.dumps(v, sort_keys=True))
    return data


def check_against_mcmeta(c: Check, gen: Path, mcmeta: Path, paths: list[str]) -> None:
    """The generator's data files must equal the pinned mcmeta 1.21.1-data files (JSON-equal), except for the
    order of the set-like lists in UNORDERED_LISTS, which is reported as tolerated."""
    if not _has_data(c, gen):
        return
    if not mcmeta.is_dir():
        c.skipped = f"{mcmeta} not found; run `python -m hivemind.rules.fetch_sources` first, or pass --mcmeta"
        return
    for rel in paths:
        mine, theirs = _json_files(gen / rel), _json_files(mcmeta / rel)
        for key in sorted(set(mine) ^ set(theirs)):
            c.diffs.append(f"{rel}/{key}.json only in {'generator' if key in mine else 'mcmeta'}")
        for key in sorted(set(mine) & set(theirs)):
            c.compared += 1
            a, b = _load(mine[key]), _load(theirs[key])
            if a == b:
                continue
            name = f"{rel}/{key}"
            if _with_sets_sorted(name, a) == _with_sets_sorted(name, b):
                c.tolerated.append(f"{name}.json: same up to the order of a set-like list (UNORDERED_LISTS)")
            else:
                c.diffs.append(f"{name}.json differs")


def check_loot_tables(c: Check, gen: Path, rules: dict) -> None:
    """Every loot table in the rules equals the generator's file."""
    if not _has_data(c, gen):
        return
    root = gen / "data" / "minecraft" / "loot_table"
    for key, table in rules["loot_tables"].items():
        path = root / f"{key}.json"
        c.compared += 1
        if not path.exists():
            c.diffs.append(f"{key}: no generator file")
        elif _load(path) != table:
            c.diffs.append(f"{key}: differs")


def _result_of(recipe: dict) -> tuple[str, int] | None:
    res = recipe.get("result")
    if isinstance(res, str):
        return _strip_ns(res), 1
    if isinstance(res, dict):
        item = res.get("id", res.get("item"))
        if item:
            return _strip_ns(item), int(res.get("count", 1))
    return None


def check_recipes(c: Check, gen: Path, rules: dict) -> None:
    """Recipe ids, results, cooking time and experience equal the rules."""
    if not _has_data(c, gen):
        return
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


def check_tags(c: Check, gen: Path, rules: dict) -> None:
    """Every block and item tag in the rules, fully resolved, equals the generator's."""
    if not _has_data(c, gen):
        return
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


def _report(gen: Path, name: str, c: Check) -> dict | None:
    """A reports/ file as a dict, or None with the check skipped."""
    path = gen / "reports" / name
    if not path.exists():
        c.skipped = f"{path} not found (run the generator with --reports)"
        return None
    data = _load(path)
    if not isinstance(data, dict):
        c.skipped = f"{name}: top level is not an object; report format not recognised"
        return None
    return data


def _state_ids(entry: Any) -> tuple[int, int, int | None] | None:
    """(min, max, default) state id of one blocks.json entry, or None if it has no usable 'states' list."""
    states = entry.get("states") if isinstance(entry, dict) else None
    if not isinstance(states, list) or not states or not all(isinstance(s, dict) and "id" in s for s in states):
        return None
    ids = [s["id"] for s in states]
    default = [s["id"] for s in states if s.get("default")]
    return min(ids), max(ids), default[0] if default else None


def check_block_states(c: Check, gen: Path, rules: dict) -> None:
    """Each block's min, max and default state id equals the rules (format [mem])."""
    report = _report(gen, "blocks.json", c)
    if report is None:
        return
    names = {_strip_ns(k): v for k, v in report.items()}
    got = {name: _state_ids(v) for name, v in names.items()}
    if not any(got.values()):
        c.skipped = "no entry has a 'states' list with ids; report format not recognised"
        return
    for name in sorted(set(names) ^ set(rules["blocks"])):
        c.diffs.append(f"{name}: only in {'report' if name in names else 'rules'}")
    for name in sorted(set(names) & set(rules["blocks"])):
        c.compared += 1
        b = rules["blocks"][name]
        want = (b["min_state"], b["max_state"], b["default_state"])
        if got[name] is None:
            c.diffs.append(f"{name}: no 'states' list with ids in the report (rules {want})")
        elif got[name] != want:
            c.diffs.append(f"{name}: (min, max, default) {got[name]} != rules {want}")


def _components(entry: Any) -> dict | None:
    """The components of one items.json entry as {type: value}: a map (1.21.1 [mem]) or a list of
    {type, value} (later versions [mem]); None if neither."""
    comps = entry.get("components", entry) if isinstance(entry, dict) else None
    if isinstance(comps, dict):
        return comps
    if isinstance(comps, list) and all(isinstance(x, dict) and "type" in x for x in comps):
        return {x["type"]: x.get("value") for x in comps}
    return None


def _food(value: Any) -> tuple[Any, float] | None | str:
    """(nutrition, saturation) of a food component, None if absent, 'malformed' otherwise."""
    if value is None:
        return None
    if isinstance(value, dict) and "nutrition" in value and "saturation" in value:
        return value["nutrition"], round(float(value["saturation"]), 6)
    return "malformed"


def check_items(c: Check, gen: Path, rules: dict) -> None:
    """Stack size, durability and food of every item equal the rules (format [mem])."""
    report = _report(gen, "items.json", c)
    if report is None:
        return
    names = {_strip_ns(k): v for k, v in report.items()}
    comps = {name: _components(v) for name, v in names.items()}
    if not any(comps.values()):
        c.skipped = "no entry has components as a map or a list of {type, value}; report format not recognised"
        return
    for name in sorted(set(names) ^ set(rules["items"])):
        c.diffs.append(f"{name}: only in {'report' if name in names else 'rules'}")
    for name in sorted(set(names) & set(rules["items"])):
        c.compared += 1
        it = rules["items"][name]
        food = it["food"]
        want = (it["stack_size"], it["max_damage"],
                None if food is None else (food["nutrition"], round(float(food["saturation"]), 6)))
        mine = comps[name]
        if mine is None:
            c.diffs.append(f"{name}: components not recognised in the report (rules {want})")
            continue
        got = (mine.get("minecraft:max_stack_size"), mine.get("minecraft:max_damage"), _food(mine.get("minecraft:food")))
        if got != want:
            c.diffs.append(f"{name}: (stack, durability, food) {got} != rules {want}")


def check_registry_ids(c: Check, gen: Path, rules: dict) -> None:
    """Block, item and entity-type protocol ids equal the rules ids (format [mem])."""
    report = _report(gen, "registries.json", c)
    if report is None:
        return
    for registry, section in (("minecraft:block", "blocks"), ("minecraft:item", "items"),
                              ("minecraft:entity_type", "entities")):
        entries = report.get(registry, {}).get("entries")
        if not isinstance(entries, dict):
            c.diffs.append(f"{registry}: no entries in report")
            continue
        got = {_strip_ns(k): v.get("protocol_id") if isinstance(v, dict) else None for k, v in entries.items()}
        ours = rules[section]
        for name in sorted(set(got) ^ set(ours)):
            c.diffs.append(f"{registry} {name}: only in {'report' if name in got else 'rules'}")
        for name in sorted(set(got) & set(ours)):
            c.compared += 1
            if got[name] != ours[name]["id"]:
                c.diffs.append(f"{registry} {name}: protocol id {got[name]} != rules id {ours[name]['id']}")


# Errors that mean "this input is not in the format the check expects" (the report formats are [mem]).
_FORMAT_ERRORS = (KeyError, IndexError, TypeError, ValueError, AttributeError)


def run_check(name: str, fn: Callable[[Check], None]) -> Check:
    """Run one check; a format error stops it (SKIP, or FAIL if it had already found differences)."""
    c = Check(name)
    try:
        fn(c)
    except _FORMAT_ERRORS as e:
        c.skipped = f"stopped on unexpected input ({type(e).__name__}: {e}); format not recognised"
    return c


def generator_checks(gen: Path, rules: dict, mcmeta: Path, paths: list[str]) -> list[tuple[str, Callable[[Check], None]]]:
    """(name, check) for every comparison of check-generator, in report order."""
    folders = ", ".join(p.rsplit("/", 1)[-1] for p in paths)
    return [
        (f"generator data == mcmeta 1.21.1-data ({folders})", lambda c: check_against_mcmeta(c, gen, mcmeta, paths)),
        ("loot tables == rules (verbatim)", lambda c: check_loot_tables(c, gen, rules)),
        ("recipes: ids, results, cooking time and experience == rules", lambda c: check_recipes(c, gen, rules)),
        ("block and item tags (resolved) == rules", lambda c: check_tags(c, gen, rules)),
        ("reports/blocks.json state ids == rules (min, max, default)", lambda c: check_block_states(c, gen, rules)),
        ("reports/items.json components == rules (stack size, durability, food)",
         lambda c: check_items(c, gen, rules)),
        ("reports/registries.json protocol ids == rules ids (block, item, entity_type)",
         lambda c: check_registry_ids(c, gen, rules)),
    ]


def cmd_check_generator(args: argparse.Namespace) -> int:
    gen = Path(args.generated)
    if not (gen / "data").is_dir() and not (gen / "reports").is_dir():
        raise SystemExit(f"{gen} has neither data/ nor reports/; point --generated at the generator's output folder")
    rules = _load(Path(args.rules))
    if rules.get("version") != GAME_VERSION:
        raise SystemExit(f"{args.rules} is for version {rules.get('version')}, not {GAME_VERSION}")
    paths = _load(SOURCES)["sources"]["mcmeta-data"]["paths"]
    statuses = []
    for name, fn in generator_checks(gen, rules, Path(args.mcmeta), paths):
        statuses.append(run_check(name, fn).report(args.max_examples))  # printed as soon as each finishes
    result = verdict(statuses)
    skipped = statuses.count("SKIP")
    note = f" ({skipped} of {len(statuses)} checks did not run)" if result == "INCOMPLETE" else ""
    print(f"{result} generator cross-check against {Path(args.rules).name}{note}")
    return EXIT_CODES[result]


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def instance_args(sp: argparse.ArgumentParser, with_control: bool = True, from_sidecar: bool = False) -> None:
        """--instance and --control; for checks they default to the instance folder's hivemind-instance.json."""
        if from_sidecar:
            sp.add_argument("--instance", type=int, help=f"instance number K (default: from {SIDECAR}, else 0)")
            if with_control:
                sp.add_argument("--control", choices=CONTROLS,
                                help=f"DESIGN.md 10.2 control run (default: from {SIDECAR}, else none)")
            return
        sp.add_argument("--instance", type=int, default=0, help="instance number K (ports 25565+K, 25575+K)")
        if with_control:
            sp.add_argument("--control", choices=CONTROLS, default="none", help="DESIGN.md 10.2 control run")

    sp = sub.add_parser("render", help="write <dir>/server.properties and <dir>/hivemind-instance.json")
    instance_args(sp)
    sp.add_argument("--dir", required=True, help="instance directory")
    sp.add_argument("--seed", help="level-seed for a new world (from the world pool)")
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("check-properties", help="compare <dir>/server.properties with the frozen template")
    instance_args(sp, from_sidecar=True)
    sp.add_argument("--dir", required=True, help="instance directory")
    sp.add_argument("--seed", help=f"expected level-seed (default: from {SIDECAR}; must agree with it)")
    sp.set_defaults(func=cmd_check_properties)

    sp = sub.add_parser("rcon", help="send commands over RCON")
    instance_args(sp, with_control=False)
    sp.add_argument("--port", type=int, help="RCON port (overrides --instance)")
    sp.add_argument("--file", help=".mcfunction file: one command per non-comment line")
    sp.add_argument("commands", nargs="*", help="commands, e.g. 'tick query'")
    sp.set_defaults(func=cmd_rcon)

    sp = sub.add_parser("check-gamerules", help="read every gamerule and the difficulty back over RCON")
    instance_args(sp, from_sidecar=True)
    sp.add_argument("--dir", help=f"instance directory: instance and control from its {SIDECAR}")
    sp.add_argument("--port", type=int, help="RCON port (overrides the instance)")
    sp.set_defaults(func=cmd_check_gamerules)

    sp = sub.add_parser("check-mods", help="check Java, server.jar, the launcher and mods/*.jar against the pins")
    sp.add_argument("--dir", required=True, help="instance directory")
    sp.add_argument("--java", default="java", help="the java executable that starts the server (default: java)")
    sp.add_argument("--pins", default=str(MODS), help="pins file (default: bridge/server/mods-1.21.1.json)")
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
