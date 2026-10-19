"""Tests for bridge/server: the frozen config files and servertool.py's verdicts.

Fast and offline: synthetic instance folders, a tiny synthetic data-generator output and an
in-process fake RCON server. Works from bridge/server/ or from tests/ (it finds servertool.py).
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import socket
import struct
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest


def _server_dir() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent / "bridge" / "server"):
        if (candidate / "servertool.py").exists():
            return candidate
    raise RuntimeError("bridge/server/servertool.py not found")


SERVER = _server_dir()
BRIDGE = SERVER.parent
REPO = BRIDGE.parent
_spec = importlib.util.spec_from_file_location("servertool", SERVER / "servertool.py")
st = importlib.util.module_from_spec(_spec)
sys.modules["servertool"] = st
_spec.loader.exec_module(st)

PASSWORD = "test-password-0123456789"
LANGUAGE = REPO / "data" / "sources" / "minecraft-data" / "data" / "pc" / "1.21.1" / "language.json"
MCMETA = REPO / "data" / "sources" / "mcmeta-1.21.1-data"


@pytest.fixture
def password(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(st.PASSWORD_ENV, PASSWORD)
    return PASSWORD


# ---------------------------------------------------------------------------------------------
# Frozen values (DESIGN.md 11.1, 11.2)
# ---------------------------------------------------------------------------------------------

def test_template_holds_the_frozen_values() -> None:
    props = st.parse_properties(st.TEMPLATE.read_text("utf-8"))
    frozen = {  # DESIGN.md 11.2, plus the exposure rules of 11.6 and the region reader of 4.4
        "view-distance": "10", "simulation-distance": "10", "difficulty": "normal", "online-mode": "false",
        "spawn-protection": "0", "enable-rcon": "true", "rcon.port": "25575", "enforce-secure-profile": "false",
        "server-ip": "127.0.0.1", "rcon.password": "", "level-seed": "", "pvp": "true",
        "region-file-compression": "deflate", "entity-broadcast-range-percentage": "100",
        "level-type": "minecraft:normal", "enable-command-block": "false",
    }
    assert {k: props.get(k) for k in frozen} == frozen


def test_frozen_gamerules_and_controls() -> None:
    rules = st.frozen_gamerules("none")
    assert rules["playersSleepingPercentage"] == "100"  # DESIGN.md 11.2
    assert int(rules["commandModificationBlockLimit"]) >= 4_194_304  # DESIGN.md 11.2
    assert rules["keepInventory"] == "false"
    assert st.frozen_gamerules("keep-inventory")["keepInventory"] == "true"  # DESIGN.md 10.2
    assert st.overrides(0, None, None, "pvp-off")["pvp"] == "false"


def test_gamerule_defaults_cover_every_rule() -> None:
    table = json.loads(st.GAMERULE_DEFAULTS.read_text("utf-8"))
    defaults = table["rules"]
    assert len(defaults) == 51  # children of `gamerule` in the pinned mcmeta command tree
    assert set(st.frozen_gamerules("none")) <= set(defaults)
    # Every proposal in the commented block names a real rule too.
    proposed = re.findall(r"^#\s*gamerule (\w+) \S+$", st.GAMERULES.read_text("utf-8"), re.MULTILINE)
    assert proposed and set(proposed) <= set(defaults)
    assert set(table["bridge_depends_on"]) <= set(defaults)
    want = st.expected_gamerules("none")
    # Verified on a fresh 1.21.1 world on the owner's PC (2026-10-19), so no longer labelled [mem].
    assert table["defaults_verified"] == "2026-10-19"
    assert want["showDeathMessages"] == ("true", "vanilla default, verified 2026-10-19")
    assert want["keepInventory"] == ("false", "frozen")


def test_unverified_defaults_are_labelled_mem(tmp_path: Path) -> None:
    table = json.loads(st.GAMERULE_DEFAULTS.read_text("utf-8"))
    table["defaults_verified"] = None
    path = tmp_path / "defaults.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    assert st.defaults_origin(path) == "vanilla default [mem]"


@pytest.mark.skipif(not LANGUAGE.exists(), reason="data/sources/ not fetched")
def test_gamerule_names_match_the_pinned_language_file() -> None:
    lang = json.loads(LANGUAGE.read_text("utf-8"))
    names = {k.split(".")[1] for k in lang if k.startswith("gamerule.") and k.count(".") == 1}
    defaults = st.gamerule_defaults()
    # The one language key that is not a command in 1.21.1 (see the defaults file's note).
    assert names - set(defaults) == {"entitiesWithPassengersCanUsePortals"}
    assert set(defaults) <= names


def test_node_packages_are_pinned_exactly() -> None:
    pkg = json.loads((BRIDGE / "package.json").read_text("utf-8"))
    assert pkg["dependencies"] == {  # DESIGN.md 11.1
        "@msgpack/msgpack": "3.1.3", "mineflayer": "4.39.0", "mineflayer-armor-manager": "2.0.1",
        "mineflayer-auto-eat": "5.0.3", "mineflayer-collectblock": "1.6.0", "mineflayer-pathfinder": "2.4.5",
        "mineflayer-tool": "1.2.0",
    }
    lock = json.loads((BRIDGE / "package-lock.json").read_text("utf-8"))["packages"]
    assert lock[""]["dependencies"] == pkg["dependencies"]
    for name, version in pkg["dependencies"].items():
        assert lock[f"node_modules/{name}"]["version"] == version
    assert lock["node_modules/minecraft-data"]["version"] == "3.117.0"  # bridge/README.md


def test_text_hashes_ignore_line_endings(tmp_path: Path) -> None:
    lf, crlf = tmp_path / "lf.txt", tmp_path / "crlf.txt"
    lf.write_bytes(b"a=1\nb=2\n")
    crlf.write_bytes(b"a=1\r\nb=2\r\n")
    assert st.sha256_text(lf) == st.sha256_text(crlf)
    assert st.sha256_file(lf) != st.sha256_file(crlf)
    assert "eol=lf" in (BRIDGE / ".gitattributes").read_text("utf-8")


# ---------------------------------------------------------------------------------------------
# .properties, render and check-properties
# ---------------------------------------------------------------------------------------------

def test_properties_parser() -> None:
    text = "# c\n! c\nlevel-type=minecraft\\:normal\nmotd = a\\\n   b\nkey\\ with\\ space:v\nempty=\nbare\n"
    assert st.parse_properties(text) == {
        "level-type": "minecraft:normal", "motd": "ab", "key with space": "v", "empty": "", "bare": "",
    }
    for value in ("minecraft:normal", " lead", "a=b#c!d", "tab\there", ""):
        assert st.parse_properties("k=" + st.escape_value(value))["k"] == value


def server_rewrite(path: Path) -> None:
    """Imitate the vanilla server's rewrite on start-up [mem]: comments gone, keys reordered."""
    props = st.parse_properties(path.read_text("utf-8"))
    lines = ["#Minecraft server properties", "#Thu Oct 16 12:00:00 UTC 2026"]
    lines += [f"{k}={st.escape_value(v)}" for k, v in sorted(props.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(tmp_path: Path, *extra: str) -> Path:
    inst = tmp_path / "inst"
    assert st.main(["render", "--dir", str(inst), *extra]) == 0
    return inst


def test_render_then_check_round_trip(tmp_path: Path, password: str, capsys: pytest.CaptureFixture) -> None:
    inst = render(tmp_path, "--instance", "2", "--seed", "-42")
    side = json.loads((inst / st.SIDECAR).read_text("utf-8"))
    assert (side["instance"], side["seed"], side["control"], side["rcon_port"]) == (2, "-42", "none", 25577)
    props = st.parse_properties((inst / "server.properties").read_text("utf-8"))
    assert (props["server-port"], props["rcon.port"], props["rcon.password"]) == ("25567", "25577", password)
    assert st.main(["check-properties", "--dir", str(inst)]) == 0
    server_rewrite(inst / "server.properties")
    assert st.main(["check-properties", "--dir", str(inst)]) == 0
    out = capsys.readouterr().out
    assert "from hivemind-instance.json" in out and "NOTE level-seed" not in out


@pytest.mark.parametrize("edit", [
    ("view-distance=10", "view-distance=9"),
    ("level-seed=-42", "level-seed=7"),
    ("difficulty=normal", "difficulty=hard"),
])
def test_check_properties_catches_a_changed_value(tmp_path: Path, password: str, edit: tuple[str, str]) -> None:
    inst = render(tmp_path, "--instance", "2", "--seed", "-42")
    path = inst / "server.properties"
    text = path.read_text("utf-8")
    assert edit[0] in text
    path.write_text(text.replace(edit[0], edit[1]), encoding="utf-8")
    assert st.main(["check-properties", "--dir", str(inst)]) == 1


def test_check_properties_reports_an_unknown_key(tmp_path: Path, password: str) -> None:
    inst = render(tmp_path)
    with (inst / "server.properties").open("a", encoding="utf-8") as f:
        f.write("some-new-key=1\n")
    assert st.main(["check-properties", "--dir", str(inst)]) == 1


def test_check_properties_uses_the_sidecar_control(tmp_path: Path, password: str) -> None:
    inst = render(tmp_path, "--control", "pvp-off", "--seed", "1")
    assert st.main(["check-properties", "--dir", str(inst)]) == 0  # no --control needed
    with pytest.raises(SystemExit, match="disagrees"):
        st.main(["check-properties", "--dir", str(inst), "--control", "none"])
    with pytest.raises(SystemExit, match="disagrees"):
        st.main(["check-properties", "--dir", str(inst), "--seed", "2"])
    assert st.main(["check-properties", "--dir", str(inst), "--seed", "1", "--instance", "0"]) == 0


def test_check_properties_without_sidecar_says_seed_unchecked(tmp_path: Path, password: str,
                                                             capsys: pytest.CaptureFixture) -> None:
    inst = render(tmp_path, "--seed", "5")
    (inst / st.SIDECAR).unlink()
    assert st.main(["check-properties", "--dir", str(inst)]) == 0
    assert "NOTE level-seed not checked" in capsys.readouterr().out
    assert st.main(["check-properties", "--dir", str(inst), "--seed", "6"]) == 1


# ---------------------------------------------------------------------------------------------
# check-mods
# ---------------------------------------------------------------------------------------------

JAVA_LINE = 'openjdk version "21.0.4" 2024-07-16 LTS'


@pytest.fixture
def instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    """A synthetic instance folder and a fully pinned pins dict for it."""
    monkeypatch.setattr(st, "java_version", lambda java: f"Picked up JAVA_TOOL_OPTIONS: -x\n{JAVA_LINE}\nOpenJDK VM")
    inst = tmp_path / "inst"
    (inst / "mods").mkdir(parents=True)
    (inst / "server.jar").write_bytes(b"vanilla server")
    (inst / "fabric-server-launch.jar").write_bytes(b"launcher")
    mods = []
    for name in ("lithium", "carpet"):
        (inst / "mods" / f"{name}.jar").write_bytes(name.encode())
        mods.append({"name": name, "version": "1.0", "sha256": st.sha256_file(inst / "mods" / f"{name}.jar")})
    sha256, sha1 = st._jar_hashes(inst / "server.jar")
    pins = {
        "game_version": "1.21.1",
        "java": {"major": 21, "version": JAVA_LINE},
        "server_jar": {"file": "server.jar", "sha1": sha1, "sha256": sha256},
        "loader": {"name": "fabric-loader", "version": "0.16.0", "installer_version": "1.0.1",
                   "launcher_file": "fabric-server-launch.jar",
                   "launcher_sha256": st.sha256_file(inst / "fabric-server-launch.jar")},
        "mods": mods,
    }
    return inst, pins


def check_mods(tmp_path: Path, inst: Path, pins: dict) -> int:
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(pins), encoding="utf-8")
    return st.main(["check-mods", "--dir", str(inst), "--pins", str(path)])


def test_check_mods_passes_when_everything_is_pinned(tmp_path: Path, instance: tuple[Path, dict]) -> None:
    assert check_mods(tmp_path, *instance) == 0


def test_shipped_pins_wait_only_for_the_harness_mod(tmp_path: Path, instance: tuple[Path, dict],
                                                    capsys: pytest.CaptureFixture) -> None:
    """Pinned on the owner's PC on 2026-10-19; only the harness mod (M2) is still TODO."""
    inst, _ = instance
    assert st.main(["check-mods", "--dir", str(inst)]) == 1
    not_frozen = [line for line in capsys.readouterr().out.splitlines() if "NOT FROZEN" in line]
    assert len(not_frozen) == 1
    todo = not_frozen[0].split(": ", 2)[-1].split(", ")
    assert todo == ["mods.hivemind-harness.version", "mods.hivemind-harness.sha256"]


def test_shipped_pins_are_well_formed() -> None:
    pins = json.loads((st.HERE / "mods-1.21.1.json").read_text("utf-8"))
    assert re.fullmatch(r'openjdk version "21\.[0-9.]+" \S+ LTS', pins["java"]["version"])
    assert re.fullmatch(r"[0-9a-f]{40}", pins["server_jar"]["sha1"])
    hashes = [pins["server_jar"]["sha256"], pins["loader"]["launcher_sha256"]]
    hashes += [m["sha256"] for m in pins["mods"] if m["name"] != "hivemind-harness"]
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes)
    assert len(set(hashes)) == len(hashes)
    assert pins["loader"]["launcher_file"].startswith(f"fabric-server-mc.1.21.1-loader.{pins['loader']['version']}-")


@pytest.mark.parametrize("field", [
    ("java", "version"), ("server_jar", "sha256"), ("server_jar", "sha1"),
    ("loader", "version"), ("loader", "launcher_file"), ("loader", "launcher_sha256"),
])
def test_check_mods_is_not_frozen_while_any_pin_is_todo(tmp_path: Path, instance: tuple[Path, dict],
                                                        field: tuple[str, str],
                                                        capsys: pytest.CaptureFixture) -> None:
    inst, pins = instance
    pins[field[0]][field[1]] = "TODO-pin"
    assert check_mods(tmp_path, inst, pins) == 1
    assert f"NOT FROZEN: still TODO-pin in pins.json: {field[0]}.{field[1]}" in capsys.readouterr().out


def test_check_mods_is_not_frozen_while_a_mod_is_todo(tmp_path: Path, instance: tuple[Path, dict]) -> None:
    inst, pins = instance
    pins["mods"][1]["sha256"] = "TODO-pin"
    assert check_mods(tmp_path, inst, pins) == 1


@pytest.mark.parametrize("change", ["server.jar", "fabric-server-launch.jar", "mods/carpet.jar", "mods/extra.jar",
                                    "remove mods/lithium.jar", "java", "sha1 only"])
def test_check_mods_catches_a_change(tmp_path: Path, instance: tuple[Path, dict], change: str,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    inst, pins = instance
    if change == "java":
        monkeypatch.setattr(st, "java_version", lambda java: 'openjdk version "21.0.5" 2024-10-15 LTS')
    elif change == "sha1 only":
        pins["server_jar"]["sha1"] = "0" * 40
    elif change.startswith("remove "):
        (inst / change.split()[1]).unlink()
    else:
        (inst / change).write_bytes(b"changed")
    assert check_mods(tmp_path, inst, pins) == 1


def test_check_mods_needs_java(tmp_path: Path, instance: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch) -> None:
    inst, pins = instance
    monkeypatch.setattr(st, "java_version", lambda java: None)
    assert check_mods(tmp_path, inst, pins) == 1
    monkeypatch.setattr(st, "java_version", lambda java: 'openjdk version "17.0.2" 2022-01-18')
    pins["java"]["version"] = 'openjdk version "17.0.2" 2022-01-18'
    assert check_mods(tmp_path, inst, pins) == 1  # major version 21 is required


def test_malformed_hash_pin_is_a_problem(tmp_path: Path, instance: tuple[Path, dict]) -> None:
    inst, pins = instance
    pins["server_jar"]["sha256"] = "ABC"
    assert check_mods(tmp_path, inst, pins) == 1


# ---------------------------------------------------------------------------------------------
# check-gamerules against a fake RCON server
# ---------------------------------------------------------------------------------------------

class FakeRcon:
    """Answers `gamerule <name>` and `difficulty` in the vanilla reply formats [mem]."""

    def __init__(self, rules: dict[str, str], difficulty: str = "Normal") -> None:
        self.rules, self.difficulty = rules, difficulty
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def reply(self, cmd: str) -> str:
        parts = cmd.split()
        if parts[0] == "gamerule" and len(parts) == 2 and parts[1] in self.rules:
            return f"Gamerule {parts[1]} is currently set to: {self.rules[parts[1]]}"
        if parts == ["difficulty"]:
            return f"The difficulty is {self.difficulty}"
        if parts == ["big"]:
            return "x" * 9000
        return f"Unknown or incomplete command, see below for error{cmd}<--[HERE]"

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    @staticmethod
    def _send(conn: socket.socket, rid: int, ptype: int, payload: bytes) -> None:
        conn.sendall(struct.pack("<iii", len(payload) + 10, rid, ptype) + payload + b"\0\0")

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            while header := conn.recv(4):
                (n,) = struct.unpack("<i", header)
                data = b""
                while len(data) < n:
                    data += conn.recv(n - len(data))
                rid, ptype = struct.unpack("<ii", data[:8])
                body = data[8:-2].decode()
                if ptype == 3:
                    self._send(conn, rid if body == PASSWORD else -1, 2, b"")
                elif ptype == 2:
                    out = self.reply(body).encode()
                    for i in range(0, max(len(out), 1), 4096):
                        self._send(conn, rid, 0, out[i:i + 4096])

    def close(self) -> None:
        self.sock.close()


@pytest.fixture
def fake_rcon(password: str) -> Iterator[FakeRcon]:
    want = {rule: value for rule, (value, _) in st.expected_gamerules("none").items()}
    server = FakeRcon(want)
    yield server
    server.close()


def test_check_gamerules_reads_every_rule(fake_rcon: FakeRcon, capsys: pytest.CaptureFixture) -> None:
    assert st.main(["check-gamerules", "--port", str(fake_rcon.port)]) == 0
    assert "PASS 52 checks, 0 problems" in capsys.readouterr().out


@pytest.mark.parametrize("rule, value", [("showDeathMessages", "false"), ("doImmediateRespawn", "true"),
                                         ("keepInventory", "true"), ("commandModificationBlockLimit", "32768")])
def test_check_gamerules_catches_a_changed_rule(fake_rcon: FakeRcon, rule: str, value: str) -> None:
    fake_rcon.rules[rule] = value
    assert st.main(["check-gamerules", "--port", str(fake_rcon.port)]) == 1


def test_check_gamerules_catches_difficulty_and_unknown_rule(fake_rcon: FakeRcon) -> None:
    fake_rcon.difficulty = "Hard"
    assert st.main(["check-gamerules", "--port", str(fake_rcon.port)]) == 1
    fake_rcon.difficulty = "Normal"
    del fake_rcon.rules["randomTickSpeed"]
    assert st.main(["check-gamerules", "--port", str(fake_rcon.port)]) == 1


def test_check_gamerules_takes_the_control_from_the_sidecar(tmp_path: Path, fake_rcon: FakeRcon) -> None:
    inst = render(tmp_path, "--control", "keep-inventory")
    fake_rcon.rules["keepInventory"] = "true"
    assert st.main(["check-gamerules", "--dir", str(inst), "--port", str(fake_rcon.port)]) == 0
    assert st.main(["check-gamerules", "--port", str(fake_rcon.port)]) == 1


def test_rcon_long_reply_and_wrong_password(fake_rcon: FakeRcon) -> None:
    with st.Rcon(fake_rcon.port, PASSWORD) as rc:
        assert rc.command("big") == "x" * 9000
        assert rc.command("difficulty") == "The difficulty is Normal"
    with pytest.raises(st.RconError, match="wrong password"):
        with st.Rcon(fake_rcon.port, "not-the-password-at-all"):
            pass


# ---------------------------------------------------------------------------------------------
# check-generator on a tiny synthetic generator output
# ---------------------------------------------------------------------------------------------

MINI_RULES = {
    "version": "1.21.1",
    "blocks": {"stone": {"id": 1, "min_state": 1, "max_state": 1, "default_state": 1},
               "oak_log": {"id": 46, "min_state": 130, "max_state": 132, "default_state": 131}},
    "items": {"stick": {"id": 5, "stack_size": 64, "max_damage": None, "food": None},
              "bread": {"id": 6, "stack_size": 64, "max_damage": None, "food": {"nutrition": 5, "saturation": 6.0}},
              "wooden_pickaxe": {"id": 7, "stack_size": 1, "max_damage": 59, "food": None}},
    "entities": {"zombie": {"id": 120}},
    "loot_tables": {"blocks/stone": {"type": "minecraft:block", "pools": [{"rolls": 1}]}},
    "recipes": [{"id": "stick", "result": {"item": "stick", "count": 4}, "cookingtime": None, "experience": None},
                {"id": "bread_smoking", "result": {"item": "bread", "count": 1}, "cookingtime": 100,
                 "experience": 0.35}],
    "tags": {"block": {"logs": ["oak_log"]}, "item": {"logs": ["oak_log"]}},
}


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def generated(tmp_path: Path) -> tuple[Path, Path, Path]:
    """(generator output, mcmeta copy, rules file) that agree with each other."""
    gen = tmp_path / "generated"
    mc = gen / "data" / "minecraft"
    _write(mc / "loot_table" / "blocks" / "stone.json", MINI_RULES["loot_tables"]["blocks/stone"])
    _write(mc / "recipe" / "stick.json", {"type": "minecraft:crafting_shaped", "result": {"id": "minecraft:stick",
                                                                                          "count": 4}})
    _write(mc / "recipe" / "bread_smoking.json", {"type": "minecraft:smoking", "result": {"id": "minecraft:bread"},
                                                  "cookingtime": 100, "experience": 0.35})
    _write(mc / "tags" / "block" / "logs.json", {"values": ["#minecraft:oak_logs"]})
    _write(mc / "tags" / "block" / "oak_logs.json", {"values": ["minecraft:oak_log"]})
    _write(mc / "tags" / "item" / "logs.json", {"values": ["minecraft:oak_log"]})
    _write(mc / "worldgen" / "placed_feature" / "ore_iron_upper.json", {"feature": "minecraft:ore_iron"})
    reports = gen / "reports"
    _write(reports / "blocks.json", {
        "minecraft:stone": {"states": [{"id": 1, "default": True}]},
        "minecraft:oak_log": {"properties": {"axis": ["x", "y", "z"]},
                              "states": [{"id": 130}, {"id": 131, "default": True}, {"id": 132}]}})
    _write(reports / "items.json", {
        "minecraft:stick": {"components": {"minecraft:max_stack_size": 64}},
        "minecraft:bread": {"components": {"minecraft:max_stack_size": 64,
                                           "minecraft:food": {"nutrition": 5, "saturation": 6.0}}},
        "minecraft:wooden_pickaxe": {"components": {"minecraft:max_stack_size": 1, "minecraft:max_damage": 59}}})
    _write(reports / "registries.json", {
        "minecraft:block": {"entries": {"minecraft:stone": {"protocol_id": 1}, "minecraft:oak_log": {"protocol_id": 46}}},
        "minecraft:item": {"entries": {"minecraft:stick": {"protocol_id": 5}, "minecraft:bread": {"protocol_id": 6},
                                       "minecraft:wooden_pickaxe": {"protocol_id": 7}}},
        "minecraft:entity_type": {"entries": {"minecraft:zombie": {"protocol_id": 120}}}})
    mcmeta = tmp_path / "mcmeta"
    shutil.copytree(gen / "data", mcmeta / "data")
    rules = tmp_path / "rules.json"
    _write(rules, MINI_RULES)
    return gen, mcmeta, rules


def check_generator(gen: Path, mcmeta: Path, rules: Path) -> int:
    return st.main(["check-generator", "--generated", str(gen), "--mcmeta", str(mcmeta), "--rules", str(rules)])


def test_check_generator_passes_on_agreeing_inputs(generated: tuple[Path, Path, Path],
                                                   capsys: pytest.CaptureFixture) -> None:
    assert check_generator(*generated) == 0
    out = capsys.readouterr().out
    assert out.count("PASS ") == 8 and "SKIP" not in out


def _edit(path: Path, fn) -> None:
    data = json.loads(path.read_text("utf-8"))
    fn(data)
    path.write_text(json.dumps(data), encoding="utf-8")


MUTATIONS = {
    "mcmeta file": lambda g, m: _edit(m / "data/minecraft/worldgen/placed_feature/ore_iron_upper.json",
                                      lambda d: d.update(feature="minecraft:ore_gold")),
    "loot table": lambda g, m: _edit(g / "data/minecraft/loot_table/blocks/stone.json",
                                     lambda d: d["pools"][0].update(rolls=2)),
    "recipe count": lambda g, m: _edit(g / "data/minecraft/recipe/stick.json", lambda d: d["result"].update(count=2)),
    "cooking time": lambda g, m: _edit(g / "data/minecraft/recipe/bread_smoking.json",
                                       lambda d: d.update(cookingtime=200)),
    "tag member": lambda g, m: _edit(g / "data/minecraft/tags/block/oak_logs.json", lambda d: d.update(values=[])),
    "state id": lambda g, m: _edit(g / "reports/blocks.json",
                                   lambda d: d["minecraft:stone"]["states"][0].update(id=999999)),
    "default state": lambda g, m: _edit(g / "reports/blocks.json",
                                        lambda d: d["minecraft:oak_log"]["states"][1].pop("default")),
    "stack size": lambda g, m: _edit(g / "reports/items.json",
                                     lambda d: d["minecraft:stick"]["components"].update({"minecraft:max_stack_size": 16})),
    "durability": lambda g, m: _edit(g / "reports/items.json",
                                     lambda d: d["minecraft:wooden_pickaxe"]["components"].pop("minecraft:max_damage")),
    "saturation": lambda g, m: _edit(g / "reports/items.json",
                                     lambda d: d["minecraft:bread"]["components"]["minecraft:food"].update(saturation=1)),
    "entity id": lambda g, m: _edit(g / "reports/registries.json",
                                    lambda d: d["minecraft:entity_type"]["entries"]["minecraft:zombie"].update(
                                        protocol_id=121)),
    "extra block": lambda g, m: _edit(g / "reports/registries.json",
                                      lambda d: d["minecraft:block"]["entries"].update(
                                          {"minecraft:dirt": {"protocol_id": 10}})),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_check_generator_catches_each_changed_value(generated: tuple[Path, Path, Path], name: str,
                                                    capsys: pytest.CaptureFixture) -> None:
    gen, mcmeta, rules = generated
    MUTATIONS[name](gen, mcmeta)
    assert check_generator(gen, mcmeta, rules) == 1
    out = capsys.readouterr().out
    # Generator data changes also differ from the mcmeta copy, so one or two checks fail.
    assert out.count("FAIL ") >= 2 and out.rstrip().splitlines()[-1].startswith("FAIL generator cross-check")


FLAT_PRESET = {"display": "minecraft:sand", "settings": {
    "biome": "minecraft:desert", "features": True, "lakes": False,
    "layers": [{"block": "minecraft:bedrock", "height": 1}, {"block": "minecraft:sand", "height": 8}],
    "structure_overrides": ["minecraft:desert_pyramids", "minecraft:mineshafts", "minecraft:villages"]}}
FLAT_REL = "data/minecraft/worldgen/flat_level_generator_preset/desert.json"


def _flat_presets(gen: Path, mcmeta: Path, change) -> None:
    """The generator's copy of a flat preset changed by `change`; mcmeta keeps the original."""
    mine = json.loads(json.dumps(FLAT_PRESET))
    change(mine["settings"])
    _write(gen / FLAT_REL, mine)
    _write(mcmeta / FLAT_REL, FLAT_PRESET)


def test_set_order_in_flat_presets_is_tolerated_and_reported(generated: tuple[Path, Path, Path],
                                                            capsys: pytest.CaptureFixture) -> None:
    """Seen on the owner's PC: the generator lists structure_overrides in another order than mcmeta."""
    gen, mcmeta, rules = generated
    _flat_presets(gen, mcmeta, lambda s: s["structure_overrides"].reverse())
    assert check_generator(gen, mcmeta, rules) == 0
    out = capsys.readouterr().out
    assert "1 tolerated" in out and "flat_level_generator_preset/desert.json: same up to the order" in out


@pytest.mark.parametrize("change", ["layer order", "structure set"])
def test_other_flat_preset_differences_still_fail(generated: tuple[Path, Path, Path], change: str,
                                                  capsys: pytest.CaptureFixture) -> None:
    gen, mcmeta, rules = generated
    if change == "layer order":  # the order of layers is the terrain: never tolerated
        _flat_presets(gen, mcmeta, lambda s: s["layers"].reverse())
    else:  # a different set, not a reordering
        _flat_presets(gen, mcmeta, lambda s: s["structure_overrides"].pop())
    assert check_generator(gen, mcmeta, rules) == 1
    assert "flat_level_generator_preset/desert.json differs" in capsys.readouterr().out


def test_set_order_is_tolerated_only_where_listed(generated: tuple[Path, Path, Path]) -> None:
    gen, mcmeta, rules = generated
    rel = "data/minecraft/worldgen/placed_feature/ore_iron_upper.json"
    _write(mcmeta / rel, {"feature": "minecraft:ore_iron", "placement": ["a", "b"]})
    _write(gen / rel, {"feature": "minecraft:ore_iron", "placement": ["b", "a"]})
    assert check_generator(gen, mcmeta, rules) == 1


def test_missing_reports_make_the_run_incomplete(generated: tuple[Path, Path, Path],
                                                 capsys: pytest.CaptureFixture) -> None:
    gen, mcmeta, rules = generated
    shutil.rmtree(gen / "reports")
    assert check_generator(gen, mcmeta, rules) == 3
    out = capsys.readouterr().out
    assert out.count("SKIP") == 3 and "INCOMPLETE" in out.splitlines()[-1]


def test_missing_mcmeta_makes_the_run_incomplete(generated: tuple[Path, Path, Path]) -> None:
    gen, _, rules = generated
    assert check_generator(gen, gen.parent / "nowhere", rules) == 3


def test_a_skip_never_hides_differences(generated: tuple[Path, Path, Path], capsys: pytest.CaptureFixture) -> None:
    gen, mcmeta, rules = generated

    def mutate(d: dict) -> None:
        d["minecraft:stone"]["states"][0]["id"] = 999999
        del d["minecraft:oak_log"]["states"]

    _edit(gen / "reports/blocks.json", mutate)
    assert check_generator(gen, mcmeta, rules) == 1
    out = capsys.readouterr().out
    assert "stone: (min, max, default) (999999, 999999, 999999)" in out and "oak_log: no 'states' list" in out

    c = st.Check("x", diffs=["a difference"], skipped="stopped")
    assert c.status == "FAIL"
    assert st.verdict(["PASS", "SKIP"]) == "INCOMPLETE" and st.verdict(["SKIP", "FAIL"]) == "FAIL"


def test_items_report_formats(generated: tuple[Path, Path, Path], capsys: pytest.CaptureFixture) -> None:
    gen, mcmeta, rules = generated
    # Components as a list of {type, value} (a later format [mem]) are read too.
    _edit(gen / "reports/items.json", lambda d: d.update({
        k: {"components": [{"type": t, "value": v} for t, v in e["components"].items()]} for k, e in d.items()}))
    assert check_generator(gen, mcmeta, rules) == 0
    # An unrecognised format is a SKIP (INCOMPLETE), never a crash, and the other checks still print.
    _edit(gen / "reports/items.json", lambda d: d.update({k: {"components": 3} for k in d}))
    capsys.readouterr()
    assert check_generator(gen, mcmeta, rules) == 3
    out = capsys.readouterr().out
    assert "SKIP reports/items.json" in out and out.count("PASS ") == 6
    # A crash inside a check stops only that check.
    _edit(gen / "reports/items.json", lambda d: d.update({"minecraft:bread": {"components": {
        "minecraft:max_stack_size": 64, "minecraft:food": {"nutrition": 5, "saturation": "lots"}}}}))
    assert check_generator(gen, mcmeta, rules) == 3


@pytest.mark.skipif(not (MCMETA / "data").is_dir() or not st.RULES.exists(), reason="data/sources/ not fetched")
def test_check_generator_on_the_pinned_sources(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The real rules against the pinned mcmeta data, with reports written in the remembered format [mem]."""
    rules = json.loads(st.RULES.read_text("utf-8"))
    gen = tmp_path / "generated"
    shutil.copytree(MCMETA / "data", gen / "data")
    blocks = {f"minecraft:{n}": {"states": [{"id": i, **({"default": True} if i == b["default_state"] else {})}
                                            for i in range(b["min_state"], b["max_state"] + 1)]}
              for n, b in rules["blocks"].items()}
    items = {}
    for n, it in rules["items"].items():
        comps = {"minecraft:max_stack_size": it["stack_size"]}
        if it["max_damage"] is not None:
            comps["minecraft:max_damage"] = it["max_damage"]
        if it["food"] is not None:
            comps["minecraft:food"] = it["food"]
        items[f"minecraft:{n}"] = {"components": comps}
    registries = {f"minecraft:{reg}": {"entries": {f"minecraft:{n}": {"protocol_id": e["id"]}
                                                   for n, e in rules[section].items()}}
                  for reg, section in (("block", "blocks"), ("item", "items"), ("entity_type", "entities"))}
    for name, data in (("blocks", blocks), ("items", items), ("registries", registries)):
        _write(gen / "reports" / f"{name}.json", data)
    assert st.main(["check-generator", "--generated", str(gen), "--mcmeta", str(MCMETA)]) == 0
    assert capsys.readouterr().out.count("PASS ") == 8
