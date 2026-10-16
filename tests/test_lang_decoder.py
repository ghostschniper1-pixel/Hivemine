"""Tests for masked word-by-word generation and the static automaton (hivemind.lang.decoder,
hivemind.lang.automaton; docs/contracts.md 4.5, DESIGN.md 5)."""

from __future__ import annotations

import json
import zlib
from collections.abc import Collection, Iterable, Mapping
from typing import Any

import numpy as np
import pytest

from hivemind.lang import Command, DeadEndError, EntityRef, Flags, FullProvider, Goal, Language, Sentence
from hivemind.lang import grammar as G
from hivemind.lang.automaton import AutomatonWalker
from hivemind.lang.decoder import Decoder, slot_domain


@pytest.fixture(scope="module")
def lang() -> Language:
    return Language.load()


@pytest.fixture(scope="module")
def auto(lang: Language):
    return lang.automaton(Flags.all())


class Restricted:
    """FullProvider with fewer verbs and, per slot, a fixed subset of values (None = all)."""

    def __init__(self, lang: Language, verbs: Iterable[str], keep: Mapping[str, Collection[Any]] | None = None):
        self.full = FullProvider(lang)
        self._verbs = tuple(verbs)
        self.keep = dict(keep or {})

    def verbs(self) -> tuple[str, ...]:
        return self._verbs

    def options(self, verb: str, slot: str, chosen: Mapping[str, Any]) -> list:
        vals = self.full.options(verb, slot, chosen)
        keep = self.keep.get(slot)
        return [v for v in vals if keep is None or v in keep]


class RandomProvider:
    """Deterministic pseudo-random non-empty subsets of every slot, depending on the slots
    already chosen (so there are no dead ends, but plenty of dependence on earlier choices)."""

    def __init__(self, lang: Language, seed: int, k: int = 3, pool: Mapping[str, Collection[Any]] | None = None):
        self.full = FullProvider(lang)
        self.seed = seed
        self.k = k
        self.pool = dict(pool or {})

    def _rng(self, *key: Any) -> np.random.Generator:
        return np.random.default_rng(zlib.crc32(repr((self.seed, *key)).encode()))

    def verbs(self) -> list[str]:
        rng = self._rng("verbs")
        return [v for v in G.VERBS if rng.random() < 0.5] or ["sleep"]

    def options(self, verb: str, slot: str, chosen: Mapping[str, Any]) -> list:
        full = self.full.options(verb, slot, chosen)
        if slot in self.pool:  # (never empty: an empty slot would be a dead end)
            full = [v for v in full if v in self.pool[slot]] or full
        rng = self._rng(verb, slot, tuple(chosen.items()))
        k = min(len(full), 1 + int(rng.integers(self.k)))
        return [full[i] for i in sorted(rng.choice(len(full), size=k, replace=False))]


# Values whose phrases are prefixes of one another or start alike (zombie / zombie villager,
# stone / stone pickaxe, iron ore / iron golem / iron ingot, red 1 / red 10).
CONFUSABLE = {
    "item": {"stone", "stone_pickaxe", "stone_sword", "iron_ingot", "iron_pickaxe", "raw_iron", "sweet_berries",
             "glow_berries", "cooked_beef", "beef", "flint", "flint_and_steel"},
    "block": {"stone", "iron_ore", "deepslate_iron_ore", "deepslate", "sand", "red_sand", "sandstone", "red_sandstone",
              "oak_leaves", "leaves"},
    "entity": {EntityRef(k, e) for k in (1, 2) for e in ("zombie", "zombie_villager", "spider", "cave_spider",
                                                        "iron_golem", ("dropped", "stone"), ("dropped", "stone_pickaxe"),
                                                        ("dropped", "iron_ingot"))},
    "player": {"red_1", "red_10", "red_11", "human", "blue_1"},
    "addressee": {"red_1", "red_10", "everyone", "team", "human"},
}


def prefix_table(lang: Language, cmds: Iterable[Command]) -> dict[tuple[int, ...], set[int]]:
    """prefix of word ids -> the set of word ids that can follow it, over a set of commands."""
    table: dict[tuple[int, ...], set[int]] = {}
    for c in cmds:
        ids = lang.encode(c)
        for k in range(len(ids)):
            table.setdefault(tuple(ids[:k]), set()).add(ids[k])
    return table


def walk(dec: Decoder, rng: np.random.Generator, check=None) -> Command:
    while not dec.done:
        mask = dec.allowed()
        if check is not None:
            check(tuple(dec.ids), mask)
        choices = np.flatnonzero(mask)
        assert dec.forced == (len(choices) == 1)
        dec.feed(int(rng.choice(choices)))
    return dec.command()


def dfs(dec: Decoder, out: list[Command], check=None) -> None:
    if dec.done:
        assert not dec.allowed().any()
        out.append(dec.command())
        return
    mask = dec.allowed()
    if check is not None:
        check(tuple(dec.ids), mask)
    for w in np.flatnonzero(mask):
        nxt = dec.fork()
        nxt.feed(int(w))
        dfs(nxt, out, check)


# -- masks allow exactly the valid continuations -------------------------------------------------


@pytest.mark.parametrize("seed", range(16))
def test_random_providers_masks_are_exact(lang: Language, seed: int) -> None:
    """Exhaustive: walking every allowed word reaches exactly the provider's commands, and at
    every prefix the mask is exactly the set of next words of those commands. Seeds 8-15 draw
    from confusable values, so phrases that are prefixes of others meet in one slot."""
    prov = RandomProvider(lang, seed) if seed < 8 else RandomProvider(lang, seed, k=6, pool=CONFUSABLE)
    flags = Flags.all()
    expected = list(lang.enumerate_commands(prov, flags))
    assert expected
    table = prefix_table(lang, expected)

    def check(prefix: tuple[int, ...], mask: np.ndarray) -> None:
        assert set(np.flatnonzero(mask)) == table[prefix], [lang.word(i) for i in prefix]

    got: list[Command] = []
    dfs(lang.decoder(prov, flags), got, check)
    assert sorted(map(repr, got)) == sorted(map(repr, expected))
    assert len(set(got)) == len(got)


@pytest.mark.parametrize("verbs,keep", [
    (("goto", "attack", "flee", "equip", "place"), {}),
    (("plan", "explore", "descend", "strip_mine", "mine", "wait", "sleep", "collect"), {}),
    (("give", "chat"), {"player": {"red_0", "blue_19", "human"}, "addressee": {"red_1", "team"}}),
    (("craft", "smelt_start", "eat", "store", "take"), {"chest": {1, 8}}),
])
def test_full_value_sets_masks_are_exact(lang: Language, verbs, keep) -> None:
    """Random walks over the full v1 value sets (a few slots cut), checking every mask exactly."""
    prov = Restricted(lang, verbs, keep)
    flags = Flags.all()
    expected = list(lang.enumerate_commands(prov, flags))
    table = prefix_table(lang, expected)
    exp_set = set(expected)
    rng = np.random.default_rng(len(expected))

    def check(prefix: tuple[int, ...], mask: np.ndarray) -> None:
        assert set(np.flatnonzero(mask)) == table[prefix], [lang.word(i) for i in prefix]

    for _ in range(150):
        cmd = walk(lang.decoder(prov, flags), rng, check)
        assert cmd in exp_set
        assert lang.parse(lang.render(cmd)) == cmd


def test_feeding_rendered_commands_reproduces_them(lang: Language) -> None:
    prov = FullProvider(lang)
    rng = np.random.default_rng(7)
    small = [v for v in G.VERBS if v not in ("give", "chat")]
    cmds = list(lang.enumerate_commands(prov, Flags.all(), verbs=small))
    items, players = lang.domains["item"], lang.domains["addressee"]
    for _ in range(60):
        item = items[int(rng.integers(len(items)))]
        cmds.append(Command("give", (("item", item), ("item_count", 64), ("player", "blue_3"))))
        cmds.append(Command("chat", (("addressee", players[int(rng.integers(len(players)))]),
                                     ("sentence", Sentence("have", item_count=2, item=item)))))
    for i in rng.choice(len(cmds), size=300, replace=False):
        c = cmds[i]
        dec = lang.decoder(prov, Flags.all())
        for w in lang.encode(c):
            assert dec.allowed()[w], (lang.render(c), dec.text())
            dec.feed(w)
        assert dec.done and dec.command() == c


def test_invalid_word_is_rejected(lang: Language) -> None:
    dec = lang.decoder(FullProvider(lang))
    dec.feed_word("mine")
    with pytest.raises(ValueError, match="not allowed"):
        dec.feed_word("iron")
    with pytest.raises(ValueError):
        dec.command()


# -- forced words -----------------------------------------------------------------------------


def test_forced_words(lang: Language) -> None:
    prov = FullProvider(lang)
    dec = lang.decoder(prov)
    assert not dec.forced
    dec.feed_word("dig")
    for w in ("down", "to", "y"):
        assert dec.forced and dec.allowed_words() == [w]
        dec.feed_word(w)
    assert not dec.forced
    dec.feed_word("-32")
    assert dec.forced and dec.allowed_words() == ["."]
    dec.feed_word(".")
    assert dec.done and not dec.allowed().any()
    assert dec.command() == Command("descend", (("y_band", -32),))

    dec = lang.decoder(prov)
    for w in ("give", "3", "bread"):
        dec.feed_word(w)
    assert dec.forced and dec.allowed_words() == ["to"]
    dec = lang.decoder(prov)
    for w in ("go", "to", "my"):
        dec.feed_word(w)
    assert sorted(dec.allowed_words()) == ["bed", "chest", "furnace"] and not dec.forced


def test_one_option_is_forced(lang: Language) -> None:
    prov = Restricted(lang, ["mine"], {"mine_count": {8}, "block": {"deepslate_iron_ore"}})
    dec = lang.decoder(prov)
    words = []
    while not dec.done:
        assert dec.forced
        w = int(dec.allowed().argmax())
        words.append(lang.word(w))
        dec.feed(w)
    assert " ".join(words) == "mine 8 deepslate iron ore ."


# -- phrases that are prefixes of other phrases -------------------------------------------------


def test_entity_prefix_phrases(lang: Language) -> None:
    refs = {EntityRef(1, "zombie"), EntityRef(1, "zombie_villager"), EntityRef(2, "zombie")}
    prov = Restricted(lang, ["attack"], {"entity": refs, "template": {"entity"}})
    for words, expected in (("attack the nearest zombie .", EntityRef(1, "zombie")),
                            ("attack the nearest zombie villager .", EntityRef(1, "zombie_villager")),
                            ("attack the second zombie .", EntityRef(2, "zombie"))):
        dec = lang.decoder(prov)
        for w in words.split():
            dec.feed_word(w)
        assert dec.command() == Command.of("attack", entity=expected)
    dec = lang.decoder(prov)
    for w in ["attack", "the", "nearest", "zombie"]:
        dec.feed_word(w)
    assert sorted(dec.allowed_words()) == [".", "villager"] and not dec.forced
    dec = lang.decoder(prov)
    for w in ["attack", "the", "second", "zombie"]:
        dec.feed_word(w)
    assert dec.allowed_words() == ["."]  # no second zombie villager is visible


def test_item_prefix_phrases(lang: Language) -> None:
    prov = Restricted(lang, ["craft", "place"], {"item": {"stone", "stone_pickaxe", "stone_sword"}})
    dec = lang.decoder(prov)
    for w in ["craft", "1", "stone"]:
        dec.feed_word(w)
    assert sorted(dec.allowed_words()) == [".", "pickaxe", "sword"]
    dec.feed_word(".")
    assert dec.command() == Command("craft", (("item", "stone"), ("item_count", 1)))
    dec = lang.decoder(prov)
    for w in ["craft", "1", "stone", "pickaxe", "."]:
        dec.feed_word(w)
    assert dec.command()["item"] == "stone_pickaxe"
    # place {item} {pattern}: after "stone" both the longer item and the patterns may follow
    dec = lang.decoder(prov)
    for w in ["place", "stone"]:
        dec.feed_word(w)
    assert sorted(dec.allowed_words()) == ["around", "for", "here", "in", "pickaxe", "sword", "under"]


def test_dependent_options_follow_the_tentative_value(lang: Language) -> None:
    """place's pattern depends on the item: after "stone" (which could end the item phrase) the
    patterns offered are stone's, asked for as if stone were chosen."""

    class Prov(Restricted):
        def options(self, verb, slot, chosen):
            if slot == "pattern":
                return ["here"] if chosen["item"] == "stone" else ["under_me"]
            return super().options(verb, slot, chosen)

    prov = Prov(lang, ["place"], {"item": {"stone", "stone_pickaxe"}})
    dec = lang.decoder(prov)
    dec.feed_word("place")
    dec.feed_word("stone")
    assert sorted(dec.allowed_words()) == ["here", "pickaxe"]
    dec.feed_word("pickaxe")
    assert dec.allowed_words() == ["under"]


def test_block_and_entity_after_go_to_the_nearest(lang: Language) -> None:
    prov = Restricted(lang, ["goto"], {"block": {"iron_ore"}, "entity": {EntityRef(1, "iron_golem")},
                                       "template": {"block", "entity"}})
    dec = lang.decoder(prov)
    for w in ["go", "to", "the", "nearest", "iron"]:
        dec.feed_word(w)
    assert sorted(dec.allowed_words()) == ["golem", "ore"]
    assert dec.n_threads == 2
    dec.feed_word("ore")
    dec.feed_word(".")
    assert dec.command() == Command.of("goto", block="iron_ore")


# -- flags, providers --------------------------------------------------------------------------


def test_flags_hide_store_take_and_plan(lang: Language) -> None:
    prov = FullProvider(lang)  # offers every verb
    first = {f: v for v, ts in G.COMMANDS.items() for f in {t.split()[0] for t in ts.values()}}

    def verbs_allowed(flags) -> set[str]:
        return {first[w] for w in lang.decoder(prov, flags).allowed_words()}

    assert verbs_allowed(None) == set(G.VERBS) - {"store", "take", "plan"}
    assert verbs_allowed(Flags(containers=True)) == set(G.VERBS) - {"plan"}
    assert verbs_allowed(Flags(goals=True)) == set(G.VERBS) - {"store", "take"}
    assert verbs_allowed(Flags.all()) == set(G.VERBS)
    off = {c.verb for c in lang.enumerate_commands(prov, verbs=["store", "take", "plan", "sleep"])}
    assert off == {"sleep"}
    dec = lang.decoder(prov)
    with pytest.raises(ValueError):
        dec.feed_word("plan")


def test_fallbacks_never_fill_a_slot(lang: Language) -> None:
    for slot, value, verb in (("block", "other_other_block", "mine"), ("item", "other_item_64", "craft"),
                              ("entity", EntityRef(1, "monster"), "attack"), ("block", "cobblestone", "mine"),
                              ("item", "dax", "craft"), ("entity", EntityRef(1, ("dropped", "other_item_1")), "goto")):
        class Bad(Restricted):
            def options(self, v, s, chosen, _slot=slot, _value=value):
                return [_value] if s == _slot else super().options(v, s, chosen)

        dec = Decoder(lang, Bad(lang, [verb]), Flags.all())
        with pytest.raises(ValueError, match="not a valid"):
            while not dec.done:
                dec.feed(int(np.flatnonzero(dec.allowed())[0]))


def test_bad_providers(lang: Language) -> None:
    with pytest.raises(ValueError, match="unknown verbs"):
        lang.decoder(Restricted(lang, ["teleport"]))
    with pytest.raises(DeadEndError):
        lang.decoder(Restricted(lang, [])).allowed()
    # the decoder looks one word ahead: "give 3 bread" is allowed (the forced "to" follows), but
    # with no player to give to, nothing is left after it
    prov = Restricted(lang, ["give"], {"player": set()})
    dec = lang.decoder(prov)
    for w in ("give", "3", "bread"):
        dec.feed_word(w)
    with pytest.raises(DeadEndError):
        dec.allowed()


def test_numpy_values_become_plain_and_bools_are_refused(lang: Language) -> None:
    """Simulator providers take counts and indices from NumPy arrays; commands must still hold
    plain ints (JSON for the bridge), and True must never pass as the number 1."""
    class Prov(Restricted):
        def options(self, verb, slot, chosen):
            vals = super().options(verb, slot, chosen)
            return np.array(vals) if slot in ("mine_count", "furnace") else vals

    for words, slot in ((("mine", "4", "iron", "ore", "."), "mine_count"),
                        (("collect", "from", "my", "furnace", "2", "."), "furnace")):
        dec = lang.decoder(Prov(lang, ["mine", "collect"], {"block": {"iron_ore"}}))
        for w in words:
            dec.feed_word(w)
        cmd = dec.command()
        assert type(cmd[slot]) is int and json.loads(json.dumps(cmd.to_json())) == cmd.to_json()
        wk = AutomatonWalker(lang.automaton(), lang, Prov(lang, ["mine", "collect"], {"block": {"iron_ore"}}))
        for w in words:
            wk.feed(lang.word_id(w))
        assert wk.command() == cmd

    class Bools(Restricted):
        def options(self, verb, slot, chosen):
            return [True] if slot == "furnace" else super().options(verb, slot, chosen)

    class Floats(Restricted):
        def options(self, verb, slot, chosen):
            return [4.0] if slot == "mine_count" else super().options(verb, slot, chosen)

    for prov, words, what in ((Bools(lang, ["collect"]), ("collect", "from", "my", "furnace", "1"), "furnace"),
                              (Floats(lang, ["mine"]), ("mine", "4"), "mine_count")):
        dec = lang.decoder(prov)  # the provider is asked one word ahead: feed or allowed raises
        with pytest.raises(ValueError, match=f"not a valid {what}"):
            for w in words:
                dec.feed_word(w)
            dec.allowed()


def test_plan_entity_slot_takes_entity_classes(lang: Language) -> None:
    """Inside ``plan to kill {mine_count} {entity}`` the slot ``entity`` names a kind of mob."""
    assert slot_domain("plan", "entity") == "entity_class" and slot_domain("attack", "entity") == "entity_ref"

    class Prov(Restricted):
        offer: Any = "zombie"

        def options(self, verb, slot, chosen):
            return [self.offer] if slot == "entity" else super().options(verb, slot, chosen)

    words = ("plan", "to", "kill", "4", "zombie", ".")
    prov = Prov(lang, ["plan"], {"goal": {"kill"}, "mine_count": {4}})
    dec = lang.decoder(prov, Flags.all())
    for w in words:
        dec.feed_word(w)
    assert dec.command() == Command("plan", (("goal", Goal("kill", mine_count=4, entity="zombie")),))
    prov.offer = EntityRef(1, "zombie")  # a command's entity value: refused, with a hint
    dec = lang.decoder(prov, Flags.all())
    with pytest.raises(ValueError, match="entity-class name"):
        for w in words:
            dec.feed_word(w)


def test_lookahead_masks_a_value_whose_next_slot_is_empty(lang: Language) -> None:
    class Prov(Restricted):
        def options(self, verb, slot, chosen):
            if slot == "pattern":
                return [] if chosen["item"] == "stone" else ["here"]
            return super().options(verb, slot, chosen)

    dec = lang.decoder(Prov(lang, ["place"], {"item": {"stone", "torch"}}))
    dec.feed_word("place")
    assert dec.allowed_words() == ["torch"]


def test_fork_is_independent(lang: Language) -> None:
    dec = lang.decoder(FullProvider(lang))
    dec.feed_word("mine")
    a, b = dec.fork(), dec.fork()
    a.feed_word("4")
    b.feed_word("16")
    assert a.text() == "mine 4" and b.text() == "mine 16" and dec.text() == "mine"


# -- the static automaton ------------------------------------------------------------------------


def test_automaton_shape(lang: Language, auto) -> None:
    assert auto.transition.shape == (len(auto.states), len(lang.words))
    assert auto.max_items <= 32  # live items fit a 32-bit mask per agent
    for name, dt in auto.domains.items():
        assert dt.incidence[0].all()  # every value passes through the root
        for i in range(len(dt.values)):
            node = int(np.flatnonzero(dt.terminal == i)[0])
            assert dt.incidence[node, i]
        # a node's values are the union of its children's (plus its own)
        for node, ch in enumerate(dt.children):
            below = np.zeros(len(dt.values), bool)
            for c in ch.values():
                below |= dt.incidence[c]
            if dt.terminal[node] >= 0:
                below[dt.terminal[node]] = True
            assert (below == dt.incidence[node]).all(), (name, node)
    start_words = {lang.word(int(w)) for w in np.flatnonzero(auto.transition[auto.start] >= 0)}
    assert start_words == set(G.first_words())
    off = lang.automaton()  # like the decoder, the default has every feature flag off
    assert off.flags == Flags()
    off_words = {lang.word(int(w)) for w in np.flatnonzero(off.transition[off.start] >= 0)}
    assert off_words == set(G.first_words()) - {"store", "take", "plan"}
    goals = lang.automaton(Flags(goals=True))
    goal_words = {lang.word(int(w)) for w in np.flatnonzero(goals.transition[goals.start] >= 0)}
    assert goal_words == set(G.first_words()) - {"store", "take"}
    assert "states x" in auto.summary()


@pytest.mark.parametrize("provider_kind", ["full", "random0", "random1", "random2", "random3"])
def test_automaton_walker_matches_decoder(lang: Language, auto, provider_kind: str) -> None:
    prov = FullProvider(lang) if provider_kind == "full" else RandomProvider(lang, int(provider_kind[-1]), k=4)
    flags = Flags.all()
    rng = np.random.default_rng(11)
    for _ in range(60):
        dec = lang.decoder(prov, flags)
        wk = AutomatonWalker(auto, lang, prov, flags)
        while not dec.done:
            m1, m2 = dec.allowed(), wk.allowed()
            assert (m1 == m2).all(), (dec.text(), [lang.word(int(i)) for i in np.flatnonzero(m1 != m2)])
            w = int(rng.choice(np.flatnonzero(m1)))
            dec.feed(w)
            wk.feed(w)
        assert wk.done and not wk.allowed().any()
        assert wk.command() == dec.command()


def test_automaton_walker_respects_flags(lang: Language, auto) -> None:
    wk = AutomatonWalker(auto, lang, FullProvider(lang), Flags())
    allowed = {lang.word(int(w)) for w in np.flatnonzero(wk.allowed())}
    assert allowed == set(G.first_words()) - {"store", "take", "plan"}
