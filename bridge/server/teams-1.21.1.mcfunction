# Minecraft Hivemind: scoreboard teams for TEAM worlds only (M9 onwards; DESIGN.md 10.1, D5).
# Not used in single-agent worlds (M1-M7).
#
# Why this file matters for memory: an agent's notebook gives every player a relation word
# (teammate, enemy, human) from the VISIBLE team colour, as a real player sees it on a name tag
# (DESIGN.md 6.3). In the real game that colour comes from these scoreboard teams, which the
# server sends to every client, so the Node host can report each player's team without hidden
# information.
#
# Same format as gamerules-1.21.1.mcfunction: one command per non-comment line, valid over RCON
# (servertool.py rcon --instance K --file bridge/server/teams-1.21.1.mcfunction).
# Command syntax is from memory for 1.21.1 [mem]; check the first run's RCON replies.

team add red
team modify red color red
# Cross-team damage on (server.properties pvp=true), friendly fire off (D5).
team modify red friendlyFire false
team add blue
team modify blue color blue
team modify blue friendlyFire false

# The conductor adds each bot when it joins, using the in-game username (PROTOCOL.md,
# "Names"): team join red red_0 ... team join blue blue_19.
# The human stays on no team; the conductor knows the human's username from its config.
