"""Build the shared English vocabulary ``data/vocab-1.21.1.json`` (contracts 4.1, 4.2).

Usage::

    python -m hivemind.lang.vocab [--concepts FILE] [--out FILE] [--check]

The build reads only ``data/concepts-1.21.1.json`` and the fixed grammar text in
``hivemind.lang.grammar``; ``--check`` rebuilds in memory and compares with the committed file.

Word order. Ids 0-3 are reserved (``<pad>``, ``<unk>``, ``<bos>``, ``.``). Then every word used
anywhere, in order of first appearance over these sources, each in its own fixed order:

    block classes (curated, by id) · fallback block classes · item classes (by id) · fallback
    item classes · entity classes (by id) · fallback entity classes · players (red, blue, 0-19,
    human) · goal setters (me, llm) · addressees (everyone, team) · command templates · chat
    sentences · goal templates · result words · interrupt causes · stations · containers ·
    event sentences · relation words · place kinds · ordinals · directions · patterns ·
    entity phrases (dropped) · equipment slots · number tokens · nonce words

so the order depends only on the class tables and the grammar text. Changing either changes the
network's input and output layers, which is why the file carries the concepts hash and its own
sha256 (``hash``, over the canonical JSON without that field).

The file also stores what the language layer needs without reading the class tables again:

- ``concepts``: one row per named concept: ``kind`` (block, item, entity, player, setter,
  addressee, nonce, number, ordinal, dir, pattern, station, container, cause, relation,
  place_kind, equip_slot), ``name``, ``phrase`` and, for classes, ``id``, ``fallback`` and
  ``resource`` (blocks), ``v1`` and ``equip_slot`` (items) or ``entity_kind`` (entities).
- ``domains``: the full value list of every slot domain (see ``grammar.SLOT_DOMAINS``), in a
  fixed order. These are the "full v1 sets" the parser accepts and bitsets index.
- ``templates``: the command templates by verb and template id, the chat sentences, goals,
  result words and event sentences, verbatim from contracts 4.3, and the verb flags.
- ``sources``: the words each source uses (for audits; a word can belong to several).
- ``human_aliases``: per domain, extra names a human may type for a class, made from the
  vanilla ids of its members (``oak_log`` -> "oak log" for ``log``, ``beef`` -> "beef" for
  ``raw beef``). Only names true of the class: an alias that equals the phrase of any class of
  any kind (``chicken``: the mob's phrase, though ``minecraft:chicken`` is also the raw item),
  or names two classes, is left out.

Two hashes: ``hash`` covers the whole file (build provenance; it changes whenever the concepts
file does, even on a rules-only rebuild, through ``concepts_hash``). ``words_hash`` covers only
what the network and the grammar masks depend on (``WORDS_HASH_FIELDS``: version, words,
concept rows, domains, templates). Compare ``words_hash`` to decide whether a checkpoint's input
and output layers fit this vocabulary; record ``hash`` with runs and replays.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from hivemind.lang import grammar as G
from hivemind.lang.language import DEFAULT_VOCAB_PATH, vocab_hash, words_hash
from hivemind.rules.classes import Concepts, canonical_json, load_concepts
from hivemind.rules.loader import GAME_VERSION

# Entity classes that never fill an entity slot: "none" names nothing, players are named by
# their player names, and a dropped item is named "dropped {item}" with its item class.
UNNAMEABLE_ENTITIES = ("none", "player", "item")


def player_names(teams: Iterable[str] = G.TEAMS, size: int = G.TEAM_SIZE) -> list[tuple[str, tuple[str, ...]]]:
    """(name, phrase) of every player: red_0 .. red_19, blue_0 .. blue_19, human."""
    out = [(f"{t}_{i}", (t, str(i))) for t in teams for i in range(size)]
    out.append((G.HUMAN, (G.HUMAN,)))
    return out


def _rows(concepts: Concepts) -> list[dict]:
    rows: list[dict] = []
    for c in concepts.block_classes:
        rows.append({"kind": "block", "name": c.name, "id": c.id, "phrase": list(c.phrase), "fallback": c.fallback,
                     "resource": c.resource})
    for c in concepts.item_classes:
        rows.append({"kind": "item", "name": c.name, "id": c.id, "phrase": list(c.phrase), "fallback": c.fallback,
                     "v1": c.v1, "equip_slot": c.equip_slot})
    for c in concepts.entity_classes:
        rows.append({"kind": "entity", "name": c.name, "id": c.id, "phrase": list(c.phrase), "fallback": c.fallback,
                     "entity_kind": c.kind})
    for name, phrase in player_names():
        rows.append({"kind": "player", "name": name, "phrase": list(phrase)})
    for s in G.SETTERS:
        rows.append({"kind": "setter", "name": s, "phrase": [s]})
    for a in G.ADDRESSEE_GROUPS:
        rows.append({"kind": "addressee", "name": a, "phrase": [a]})
    for w in G.NONCE_WORDS:
        rows.append({"kind": "nonce", "name": w, "phrase": [w]})
    for n in G.NUMBER_TOKENS:
        rows.append({"kind": "number", "name": n, "phrase": [str(n)]})
    for i, w in enumerate(G.ORDINALS, start=1):
        rows.append({"kind": "ordinal", "name": i, "phrase": [w]})
    for d in G.DIR8:
        rows.append({"kind": "dir", "name": d, "phrase": [d]})
    for p, words in G.PATTERNS.items():
        rows.append({"kind": "pattern", "name": p, "phrase": list(words)})
    for s in G.STATIONS:
        rows.append({"kind": "station", "name": s, "phrase": list(concepts.block_class(s).phrase)})
    for c in G.CONTAINERS:
        rows.append({"kind": "container", "name": c, "phrase": [c]})
    for c in G.CAUSE_WORDS:
        rows.append({"kind": "cause", "name": c, "phrase": [c]})
    for r in G.RELATIONS:
        rows.append({"kind": "relation", "name": r, "phrase": [r]})
    for p in G.PLACE_KINDS:
        rows.append({"kind": "place_kind", "name": p, "phrase": [p]})
    for s, words in G.EQUIP_SLOTS.items():
        rows.append({"kind": "equip_slot", "name": s, "phrase": list(words)})
    return rows


def _domains(concepts: Concepts) -> dict[str, Any]:
    blocks = [c.name for c in concepts.block_classes if c.resource and not c.fallback]
    items = [c.name for c in concepts.item_classes if c.v1 and not c.fallback and c.name != "none"]
    entities = [c.name for c in concepts.entity_classes if not c.fallback and c.name not in UNNAMEABLE_ENTITIES]
    players = [name for name, _ in player_names()]
    return {
        "block": blocks,
        "item": items,
        "chat_item": items + list(G.NONCE_WORDS),
        "entity_class": entities,
        "entity_ref": {"ordinals": list(range(1, len(G.ORDINALS) + 1)), "entities": entities, "dropped": items},
        "player": players,
        "addressee": players + list(G.ADDRESSEE_GROUPS),
        "event_entity": [c.name for c in concepts.entity_classes if c.name not in UNNAMEABLE_ENTITIES],
        "event_item": [c.name for c in concepts.item_classes if c.name != "none"],
        "dir8": list(G.DIR8),
        "dir4": list(G.DIR4),
        "pattern": list(G.PATTERNS),
        "station": list(G.STATIONS),
        "container": list(G.CONTAINERS),
        "cause": list(G.CAUSE_WORDS) + [c.name for c in concepts.entity_classes if c.kind == "hostile"],
        **{k: list(v) for k, v in G.INT_DOMAINS.items()},
    }


def _sources(concepts: Concepts) -> dict[str, list[str]]:
    def words_of(phrases: Iterable[Iterable[str]]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for p in phrases:
            for w in p:
                if w not in seen:
                    seen.add(w)
                    out.append(w)
        return out

    lit = G.literal_words()
    hostile = [c.phrase for c in concepts.entity_classes if c.kind == "hostile"]
    return {
        "block classes": words_of(c.phrase for c in concepts.block_classes if not c.fallback),
        "fallback block classes": words_of(c.phrase for c in concepts.block_classes if c.fallback),
        "item classes": words_of(c.phrase for c in concepts.item_classes if not c.fallback),
        "fallback item classes": words_of(c.phrase for c in concepts.item_classes if c.fallback),
        "entity classes": words_of(c.phrase for c in concepts.entity_classes if not c.fallback),
        "fallback entity classes": words_of(c.phrase for c in concepts.entity_classes if c.fallback),
        "players": words_of(p for _, p in player_names()),
        "goal setters": list(G.SETTERS),
        "addressees": list(G.ADDRESSEE_GROUPS),
        "command templates": lit["command templates"],
        "chat sentences": lit["chat sentences"],
        "goal templates": lit["goal templates"],
        "result words": lit["result words"],
        "interrupt causes": words_of([*([w] for w in G.CAUSE_WORDS), *hostile]),
        "stations": words_of(concepts.block_class(s).phrase for s in G.STATIONS),
        "containers": list(G.CONTAINERS),
        "event sentences": lit["event sentences"],
        "relation words": list(G.RELATIONS),
        "place kinds": list(G.PLACE_KINDS),
        "ordinals": list(G.ORDINALS),
        "directions": list(G.DIR8),
        "patterns": words_of(G.PATTERNS.values()),
        "entity phrases": [G.DROPPED],
        "equipment slots": words_of(G.EQUIP_SLOTS.values()),
        "number tokens": [str(n) for n in G.NUMBER_TOKENS],
        "nonce words": list(G.NONCE_WORDS),
    }


def _member_words(member: str) -> tuple[str, ...]:
    return tuple(w for w in member.split("_") if w)


def _aliases(concepts: Concepts, domains: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Per domain: alias text -> class name, from the vanilla ids of the members (see module doc).

    An alias is never the phrase of any class of any kind (block, item or entity, curated or
    fallback): the vanilla id ``chicken`` is the raw chicken item, but the word ``chicken`` names
    the mob, so reading it as the item would name a thing with another thing's words."""
    phrases = {tuple(c.phrase) for cs in (concepts.block_classes, concepts.item_classes, concepts.entity_classes)
               for c in cs}
    out: dict[str, dict[str, str]] = {}
    for dom, getter in (("block", concepts.block_class), ("item", concepts.item_class),
                        ("entity_class", concepts.entity_class)):
        names = domains[dom]
        found: dict[tuple[str, ...], set[str]] = {}
        for n in names:
            c = getter(n)
            for m in c.members:
                a = _member_words(m)
                if a and a not in phrases:
                    found.setdefault(a, set()).add(n)
        out[dom] = {" ".join(a): next(iter(ns)) for a, ns in sorted(found.items()) if len(ns) == 1}
    return out


def build_vocab(concepts: Concepts | None = None) -> dict:
    """The vocabulary as a JSON-ready dict (with its hash)."""
    concepts = concepts or load_concepts()
    G.check_static_grammar()
    sources = _sources(concepts)
    words: list[str] = list(G.RESERVED)
    seen = set(words)
    for ws in sources.values():
        for w in ws:
            if w not in seen:
                seen.add(w)
                words.append(w)
    domains = _domains(concepts)
    vocab = {
        "version": GAME_VERSION,
        "concepts_hash": concepts.hash,
        "reserved": list(G.RESERVED),
        "words": words,
        "sources": sources,
        "concepts": _rows(concepts),
        "domains": domains,
        "templates": {
            "command": G.COMMANDS,
            "sentence": G.SENTENCES,
            "goal": G.GOALS,
            "result": G.RESULTS,
            "event": G.EVENTS,
            "verb_flags": G.VERB_FLAG,
        },
        "human_aliases": _aliases(concepts, domains),
    }
    missing = sorted({w for r in vocab["concepts"] for w in r["phrase"]} - seen)
    if missing:
        raise ValueError(f"phrase words missing from the vocabulary: {missing}")
    vocab["words_hash"] = words_hash(vocab)
    vocab["hash"] = vocab_hash(vocab)
    return vocab


def render(vocab: dict) -> bytes:
    """The exact bytes of the vocabulary file."""
    return (canonical_json(vocab) + "\n").encode("utf-8")


def report(vocab: dict) -> str:
    lines = [f"vocabulary: {len(vocab['words'])} words ({len(vocab['reserved'])} reserved), "
             f"hash {vocab['hash'][:12]}, words_hash {vocab['words_hash'][:12]}"]
    seen = set(vocab["reserved"])
    lines.append(f"  {'source':<26} {'new':>5} {'used':>5}")
    for src, ws in vocab["sources"].items():
        new = [w for w in ws if w not in seen]
        seen.update(ws)
        lines.append(f"  {src:<26} {len(new):>5} {len(ws):>5}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the shared English vocabulary (data/vocab-1.21.1.json).")
    ap.add_argument("--concepts", type=Path, default=None, help="concepts file (default data/concepts-1.21.1.json)")
    ap.add_argument("--out", type=Path, default=DEFAULT_VOCAB_PATH)
    ap.add_argument("--check", action="store_true", help="build in memory and compare with the file")
    args = ap.parse_args(argv)
    vocab = build_vocab(load_concepts(args.concepts))
    data = render(vocab)
    print(report(vocab))
    if args.check:
        same = args.out.exists() and args.out.read_bytes() == data
        print(f"{args.out}: {'up to date' if same else 'DIFFERS from a fresh build'}")
        return 0 if same else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"wrote {args.out} ({len(data):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
