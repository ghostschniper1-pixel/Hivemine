# Minecraft Hivemind: FROZEN gamerules for Java Edition 1.21.1 (DESIGN.md 11.2, 10.1, D14).
#
# Every line that does not start with '#' is one server command, valid both in a datapack
# function and over RCON (no leading slash). Apply after a world is created and before any
# measurement:
#   python bridge/server/servertool.py rcon --instance K --file bridge/server/gamerules-1.21.1.mcfunction
#   python bridge/server/servertool.py check-gamerules --instance K
# Gamerules live in the world's level.dat, not in server.properties, so a new or restored
# world must be checked again.
#
# Rule NAMES are the 1.21.1 names, checked against the gamerule.<name> keys of the pinned
# minecraft-data pc/1.21.1 language.json [1]. Later versions renamed some of them
# (players_sleeping_percentage; max_block_modifications since 25w44a) [73].
#
# Only rules that DESIGN.md states are active here. Everything else stays at the vanilla
# default. Proposals are in the commented block at the end, for the owner to decide.

# Night skip only when 100% of players sleep. 100 is the vanilla default [73]; set explicitly
# because the simulator models exactly this rule (DESIGN.md 4.2, 10.1, D14).
gamerule playersSleepingPercentage 100

# /clone resets of a 128 x 128 x 256 area need >= 4,194,304 blocks per command; the vanilla
# default is 32,768 [73] (DESIGN.md 11.2). 8,388,608 (2^23) also covers a full-height reset of
# one area (128 x 128 x 384 = 6,291,456) or a reset that includes the barrier walls
# (130 x 130 x 256 = 4,326,400). The rule only limits /fill, /clone and /fillbiome.
gamerule commandModificationBlockLimit 8388608

# Death drops the inventory (DESIGN.md 10.1, D4). The keepInventory control (DESIGN.md 10.2)
# sets this to true on its own worlds: check-gamerules --control keep-inventory.
gamerule keepInventory false

# ------------------------------------------------------------------------------------------
# PROPOSED, NOT FROZEN. The owner decides; none of these is applied. Rules that change what
# the agents meet (weather, spawns, ticking) must be decided before the M3 measurements,
# because the measured statistics then include them. Defaults are vanilla [mem].
#
# Advancement messages ("red_0 has made the advancement ...") are system chat. The Node host
# forwards only player chat, so the agents never see them, but they clutter the logs and the
# human's chat. No effect on mechanics.
# gamerule announceAdvancements false
#
# The simulator skips rain and thunderstorms (DESIGN.md 4.2) and assumes clear weather for
# hostile spawning (4.3). Rain lowers sky light (more hostile spawns), thunderstorms allow
# daytime spawns and sleeping. Strongly recommended, together with "weather clear" once
# per world.
# gamerule doWeatherCycle false
#
# Spawn chunks (default radius 2 in 1.20.5+ [mem]) stay loaded and ticking with no player
# near them. The simulator ticks only chunks inside some agent's simulation-distance square
# (DESIGN.md 4.2). 0 removes the exception. Strongly recommended, or keep every evaluation
# area and pristine copy far from the world spawn.
# gamerule spawnChunkRadius 0
#
# Phantoms spawn for players who have not slept for 3 days (72,000 ticks of the
# time_since_rest statistic) [mem]. Bots that reuse a name keep that statistic across
# episodes unless player data is reset, and the simulator has no phantoms.
# gamerule doInsomnia false
#
# Pillager patrols and wandering traders spawn near players [mem]; neither is in the
# simulator (villagers are skipped, DESIGN.md 4.2).
# gamerule doPatrolSpawning false
# gamerule doTraderSpawning false
#
# Wardens spawn from sculk shriekers in the deep dark (around y -54, where the agents
# strip-mine) [mem]. They kill almost instantly and are not in the simulator.
# gamerule doWardenSpawning false
#
# Players with no bed or spawnpoint respawn at a random spot within 10 blocks of the world
# spawn [mem]. 0 makes respawn positions deterministic. The team stage sets a spawnpoint per
# agent anyway (respawn at base, D4).
# gamerule spawnRadius 0
#
# Fire spread is skipped in the simulator (DESIGN.md 4.2). Without lightning (weather off)
# fire comes only from lava next to flammable blocks, so this matters little.
# gamerule doFireTick false
