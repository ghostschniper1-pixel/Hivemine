# Server setup (Windows 11, Java Edition 1.21.1)

How to set up the real Minecraft servers the agents are evaluated on (DESIGN.md 11.2, 11.3, 11.6). Target machine: the owner's PC (Threadripper 3970X, 64 GB, 2× RTX 3090, Windows 11).

**Status.** These files were written in a cloud container that cannot download the server jar or any mod (Mojang, Fabric maven and Modrinth are not reachable there). Nothing here has run against a real server yet. Steps marked **TODO-pin** need a version or hash filled in on first setup; anything marked [mem] is from memory and must be checked the first time it is used. The labels follow DESIGN.md: [n] is a reference there, [mem] is unverified memory, [est] is an estimate.

## What is frozen

The server config is frozen at M0, before any measurement (DESIGN.md 11.2, D13). Every statistic (spawns, ore exposure, furnace timing, despawns) is measured under exactly these files; changing one means measuring again.

| File | What |
|---|---|
| `server.properties` | The frozen properties template. Instances get a rendered copy that changes only the ports, the RCON password and the seed |
| `gamerules-1.21.1.mcfunction` | The frozen gamerules (three active rules), plus a commented block of proposals for the owner to decide |
| `teams-1.21.1.mcfunction` | Scoreboard teams for team worlds (M9 on): colours give the relation word `teammate` / `enemy` (DESIGN.md 6.3, 10.1) |
| `mods-1.21.1.json` | Pins for Java, the server jar, the Fabric loader and every mod (all TODO-pin until the first setup) |
| `servertool.py` | Renders instance configs and checks them: properties, gamerules over RCON, mod hashes, data-generator output |

Frozen values (DESIGN.md 11.2): `view-distance=10`, `simulation-distance=10`, `difficulty=normal`, `online-mode=false`, `spawn-protection=0`, RCON on port 25575 bound to localhost, `enforce-secure-profile=false`; gamerules `playersSleepingPercentage 100`, `commandModificationBlockLimit 8388608` (the design asks for at least 4,194,304), `keepInventory false`.

`servertool.py` uses only the Python standard library. Run it with the repository's venv (`.venv\Scripts\python`) or any Python 3.11+. All commands below are PowerShell, run from the repository root.

## 1. Prerequisites

1. **Java 21** (64-bit JDK). Minecraft 1.20.5 and later need Java 21 [mem]. For example `winget install EclipseAdoptium.Temurin.21.JDK` or `winget install Microsoft.OpenJDK.21` [mem]. Check with `java -version` and write the version into `mods-1.21.1.json` (`java.version`).
2. **Python 3.11+** with the repository venv (see the top-level README).
3. **Node.js 22 or newer** for the bridge hosts later (mineflayer 4.39.0 requires Node ≥ 22). Not needed for the server itself.
4. **Folders outside the repository**, so that worlds and the rendered RCON password never reach git:

   ```
   C:\hivemind\servers\inst-0 ... inst-3   one folder per server instance
   C:\hivemind\datagen                     vanilla data generator output
   C:\hivemind\backups                     full world copies for restores
   ```

## 2. RCON password

The committed template has an empty `rcon.password` on purpose. The render step fills it from the environment variable `HIVEMIND_RCON_PASSWORD` (at least 16 printable characters, no spaces).

```powershell
python -c "import secrets; print(secrets.token_urlsafe(24))"
setx HIVEMIND_RCON_PASSWORD "<the token>"        # for new shells (stored in the user's environment)
$env:HIVEMIND_RCON_PASSWORD = "<the token>"      # for this shell
```

RCON sends the password in plain text, which is fine only because it stays on loopback (section 5). The rendered `server.properties` in each instance folder contains the password; keep those folders out of git and out of anything you share.

## 3. Install one instance (Fabric 1.21.1 + mods)

Do this for `inst-0` first; the other instances are copies (section 8).

1. **Fabric server.** Open https://fabricmc.net/use/server/, choose Minecraft **1.21.1**, the current loader (**TODO-pin**) and installer (**TODO-pin**), and either
   - download the "executable server jar" (a launcher such as `fabric-server-mc.1.21.1-loader.<v>-launcher.<v>.jar`) into `C:\hivemind\servers\inst-0`; on first start it downloads the vanilla `server.jar` and the libraries [mem]; or
   - run the installer: `java -jar fabric-installer-<v>.jar server -mcversion 1.21.1 -loader <v> -downloadMinecraft -dir C:\hivemind\servers\inst-0` [mem], which produces `fabric-server-launch.jar` and `server.jar`.

   Record the loader and installer versions in `mods-1.21.1.json`.
2. **Vanilla jar check.** Compare `server.jar`'s SHA-1 with `downloads.server.sha1` in Mojang's 1.21.1 version JSON (linked from https://piston-meta.mojang.com/mc/game/version_manifest_v2.json). `servertool.py check-mods` prints both hashes. Telemetry records the jar hash with every session (DESIGN.md 11.4).
3. **Mods** into `C:\hivemind\servers\inst-0\mods\`, each the Fabric build for 1.21.1:
   - Lithium (**TODO-pin**): performance, with patches tested for vanilla parity [25].
   - Carpet (**TODO-pin**): spawn tracking, mob caps, TPS logging, `/profile`, `/player` fake players [27]. Keep `antiCheatDisabled` at its default, false.
   - Chunky (**TODO-pin**) for pregeneration (section 6) [mem]. It sits idle otherwise, so the mod set stays identical for every run.
   - Fabric API (**TODO-pin**) only if the pinned Chunky build requires it [mem].
   - The **harness mod** is planned (M2, DESIGN.md 11.3): server-side break-tick log, player-state restore, chunk-ticking report. Until it exists, Carpet's scarpet is the fallback.
4. **Pin the jars.** Run

   ```powershell
   python bridge\server\servertool.py check-mods --dir C:\hivemind\servers\inst-0
   ```

   It prints the SHA-256 of `server.jar` and of every jar in `mods\`. Copy the values and versions into `bridge/server/mods-1.21.1.json` and commit them. From then on `check-mods` fails on any missing, changed or extra jar. It reports `NOT FROZEN` until every entry is pinned.

## 4. Configure, start and verify

1. **Render the properties** (instance K gets game port 25565 + K and RCON port 25575 + K):

   ```powershell
   python bridge\server\servertool.py render --instance 0 --dir C:\hivemind\servers\inst-0 --seed <world seed>
   ```

   This writes `server.properties` and `hivemind-instance.json` (ports, seed, template and gamerule hashes, for telemetry). It never writes `eula.txt`: read the Minecraft EULA and set `eula=true` yourself.
2. **Start** the server, for example `java -Xms5G -Xmx5G -jar <launcher or fabric-server-launch.jar> nogui` (5 GB per instance, DESIGN.md 13 [est]). Record the JVM flags; they belong to the session record.
3. **Check the properties.** On start-up the server rewrites `server.properties` (comments removed, keys reordered, missing keys added) [mem]. The check compares values, not text:

   ```powershell
   python bridge\server\servertool.py check-properties --instance 0 --dir C:\hivemind\servers\inst-0 --seed <world seed>
   ```

   Any key the server added that the template lacks is reported: the template's key list for 1.21.1 is from memory, and this is how it gets verified. Add such keys (with the server's default value) to the template and commit.
4. **Apply and check the gamerules.** Gamerules live in the world's `level.dat`, so do this for every new or restored world:

   ```powershell
   python bridge\server\servertool.py rcon --instance 0 --file bridge\server\gamerules-1.21.1.mcfunction
   python bridge\server\servertool.py check-gamerules --instance 0
   ```

   `check-gamerules` also reads back the difficulty. The reply formats it parses (`Gamerule X is currently set to: Y`, `The difficulty is Normal`) are [mem]; if they differ, it prints the raw reply and fails, and the regexes in `servertool.py` need a fix.
5. **Carpet and tick checks:** `python bridge\server\servertool.py rcon --instance 0 "carpet antiCheatDisabled" "tick query"`. Expect `antiCheatDisabled` false and a target of 20 TPS [mem].
6. **Decide the proposed gamerules** in the commented block of `gamerules-1.21.1.mcfunction` before the M3 measurements. Two of them change what the agents meet and should be settled first: `doWeatherCycle false` (the simulator has no rain or storms) and `spawnChunkRadius 0` (otherwise spawn chunks tick with no agent nearby, unlike the simulator). Moving a proposal into the active part is a change of the frozen config: commit it before measuring.

## 5. Firewall and exposure (DESIGN.md 11.6)

Offline mode means anyone who can reach the game port can join under any name, including the human's. Three layers:

1. **Loopback binding.** The template sets `server-ip=127.0.0.1`. Vanilla RCON has no bind address of its own and listens on `server-ip` too [mem], so this keeps both the game and RCON local. The owner's own client on the same PC connects to `127.0.0.1`. Node hosts also connect to `127.0.0.1`, not `localhost`, which may resolve to `::1` first. Check what is listening:

   ```powershell
   Get-NetTCPConnection -State Listen -LocalPort 25565,25575 | Select-Object LocalAddress,LocalPort,OwningProcess
   ```

   Both should show `127.0.0.1`. If you ever set `server-ip=` (all interfaces) so that another PC on the LAN can join, RCON then listens on every interface as well, and the firewall rules below are the only protection.
2. **Windows Firewall block rules** (PowerShell as administrator). They cover instances 0–9, the conductor ↔ Node bridge port and the Toxiproxy API [28]:

   ```powershell
   New-NetFirewallRule -DisplayName "Hivemind: block Minecraft game ports" -Direction Inbound -Action Block -Protocol TCP -LocalPort 25565-25574
   New-NetFirewallRule -DisplayName "Hivemind: block Minecraft UDP"        -Direction Inbound -Action Block -Protocol UDP -LocalPort 25565-25574
   New-NetFirewallRule -DisplayName "Hivemind: block RCON"                 -Direction Inbound -Action Block -Protocol TCP -LocalPort 25575-25584
   New-NetFirewallRule -DisplayName "Hivemind: block bridge and Toxiproxy" -Direction Inbound -Action Block -Protocol TCP -LocalPort 7600-7609,8474
   ```

   Block rules take precedence over allow rules, and loopback traffic is not filtered, so local connections keep working [mem]. When Windows asks whether Java may accept connections, do not allow it. From another device, `Test-NetConnection <this PC's IP> -Port 25565` must fail.
3. **No port forwarding** on the router for any of these ports, and no public listing.

Also: never `/op` a player name. In offline mode, anyone connecting under that name would be an operator. Use RCON (`servertool.py rcon`) for every privileged command. Do not enable the whitelist as a security measure either: in offline mode it is keyed by names that anyone can claim. The EULA and usage guidelines govern servers [30]; their wording on bots and offline mode is unverified, so read them before sharing anything publicly. Public servers only with written permission (DESIGN.md 11.6).

## 6. Pregeneration with Chunky

Worlds are pregenerated under the frozen config so that nothing generates during an episode (DESIGN.md 4.4). Over RCON [mem]:

```powershell
python bridge\server\servertool.py rcon --instance 0 "chunky world minecraft:overworld" "chunky shape square" "chunky center <x> <z>" "chunky radius 256" "chunky start"
python bridge\server\servertool.py rcon --instance 0 "chunky progress"
```

- Radius per evaluation area: half the area (64) plus a bot's view distance and a margin, 64 + 16 × (10 + 2) = 256 blocks [est]. Generate the pristine copy's region (section 8) too.
- Stop the server and copy the whole world folder to `C:\hivemind\backups\<seed>` once pregeneration is done: that is the full-restore copy (DESIGN.md 11.3).
- Chunk generation also places the first animals in each new chunk [mem]. Section 9 lists what this means for resets.

## 7. Cross-check the rules with the vanilla data generator

`data/rules-1.21.1.json` is built from mcmeta's 1.21.1 data, which is itself the data generator's output [3]. Running the generator from the owner's own jar checks that chain end to end (DESIGN.md 4.1) [4]:

```powershell
mkdir C:\hivemind\datagen; cd C:\hivemind\datagen
copy C:\hivemind\servers\inst-0\server.jar .
java "-DbundlerMainClass=net.minecraft.data.Main" -jar server.jar --server --reports
```

Quote the `-D` argument in PowerShell. Use the vanilla `server.jar`, not the Fabric launcher. The output goes to `generated\` [mem]: `generated\data\minecraft\...` (recipes, loot tables, tags, worldgen) and `generated\reports\` (`blocks.json`, `items.json`, `registries.json`, ...). Then, from the repository root (the pinned sources must be in `data/sources/`; fetch them once with `python -m hivemind.rules.fetch_sources`):

```powershell
python bridge\server\servertool.py check-generator --generated C:\hivemind\datagen\generated
```

| Check | Compares |
|---|---|
| generator data == mcmeta 1.21.1-data | Every JSON file under `loot_table`, `recipe`, `tags`, `worldgen` (the pinned mcmeta paths), file set and content |
| loot tables == rules | Every loot table in the rules, verbatim |
| recipes == rules | Recipe ids, result item and count, cooking time and experience |
| tags == rules | Every block and item tag in the rules, fully resolved |
| `reports/blocks.json` | Each block's min, max and default state id (the ids Mineflayer reports and `data/block_classes-1.21.1.csv` maps) |
| `reports/items.json` | Stack size, durability, food nutrition and saturation |
| `reports/registries.json` | Block, item and entity-type protocol ids against the rules ids |

Exit code 0 means every check passed. A `SKIP` means an input was missing or its format was not recognised (the report formats are [mem]). A `FAIL` lists examples: report it, because either the pinned sources or the rules build is wrong for 1.21.1, and the rules must be rebuilt before any measurement. The entity ids deserve attention: the rules take entities from minecraft-data's 1.20.5 files (contracts section 1), and this is the first check against the 1.21.1 registry.

What was verified without a server: on the pinned mcmeta data and on reports synthesized from minecraft-data in the remembered format, all seven checks pass (3,894 data files, 1,082 loot tables, 1,290 recipes, 331 tags, 1,060 blocks, 1,333 items, 2,523 registry ids), and each check catches a deliberately changed value.

## 8. Evaluation instances: 4 × 10 bots

Forty bots on one server at simulation distance 10 would tick about 40 × 441 chunks. The design uses four instances with ten bots each instead [est], to be confirmed by the M7 ramp test (MSPT p95 < 40 ms, Node event-loop delay p99 < 20 ms) (DESIGN.md 11.3, 12).

| Instance | Folder | Game port | RCON port | Bots (agent slots) | Memory |
|---|---|---|---|---|---|
| 0 | `C:\hivemind\servers\inst-0` | 25565 | 25575 | 0–9 | `-Xmx5G` |
| 1 | `C:\hivemind\servers\inst-1` | 25566 | 25576 | 10–19 | `-Xmx5G` |
| 2 | `C:\hivemind\servers\inst-2` | 25567 | 25577 | 20–29 | `-Xmx5G` |
| 3 | `C:\hivemind\servers\inst-3` | 25568 | 25578 | 30–39 | `-Xmx5G` |

- **Making instances 1–3:** copy `inst-0` with its jars and `mods\` but without its world, render with `--instance K` and the instance's seed, start, then run `check-mods`, `check-properties` and the gamerule commands and check for each.
- **Areas:** each instance holds 10 single-agent areas of 128 × 128, one bot each, with centres at least 512 blocks apart (each bot ticks ±160 blocks, and 128 + 2 × 160 = 448). For example a 5 × 2 grid with 512-block spacing. Each area is walled with barrier blocks, matching the simulator's world edge. `/worldborder` is one square per world, so it is used only in team worlds (DESIGN.md 11.3).
- **Pristine copies** of every area live elsewhere in the same world, outside every bot's ticking square, and are the `/clone` source for resets. `/clone` needs both regions loaded [mem]; force-loaded chunks tick like spawn chunks [mem], so force-load the pristine region only for the duration of a reset.
- **Bots and hosts:** 10 Node host processes × 4 bots (at most 4–5 bots per process, DESIGN.md 11.1). Bot b plays on instance b ÷ 10, in area b mod 10. A host's bots may sit on different instances; every bot has its own connection.
- **CPU budget** for 40 bots (DESIGN.md 13 [est]): servers 12–16 threads, Node hosts about 6, conductor 1–2, inference on GPU 0.
- **Profiles** (DESIGN.md 8): A at 20 TPS on LAN latency, B with 120 ± 40 ms, C with 300 ± 120 ms plus spikes at `/tick rate 15`. Every bot measurement otherwise runs at 20 TPS (`tick query`).
- **Controls** (DESIGN.md 10.2), team stage only: the PvP-off control renders with `--control pvp-off` (sets `pvp=false`); the keepInventory control applies `gamerule keepInventory true` and is checked with `check-gamerules --control keep-inventory`. These are the only sanctioned departures from the frozen files, and they are recorded in `hivemind-instance.json`.

## 9. Known gaps (to settle in M2, M3 and M7)

- **Animals after resets.** `/clone` copies blocks and block entities, not entities [mem], and the design's reset ends with `/kill` on non-player entities (DESIGN.md 11.3). After a reset, an area therefore has none of the animals that chunk generation placed, while a fresh pregenerated area does. Snapshot replays need a recorded animal list to re-summon, or the simulator must start without generation-time animals. Decide before measuring animal rates.
- **Respawn and player data.** Bots reuse their names, so player data (position, inventory, statistics such as `time_since_rest`) persists across episodes unless reset. The harness (M2) should reset it.
- **The `allow-flight=false` kick.** If the lag shim triggers vanilla's "flying" kick, `allow-flight` may need to change. It does not affect mechanics.
- **Everything [mem] above:** key names in `server.properties`, RCON reply formats, Chunky commands, `/clone` and force-load behaviour, Windows Firewall precedence. Each is checked by a step in this guide the first time it runs.
