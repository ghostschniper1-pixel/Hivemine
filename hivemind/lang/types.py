"""Value types of the language layer (contracts 4.4): commands, chat sentences, goals, events.

A ``Command`` is a verb id plus its slot values. Every value is a concept name (``"iron_ore"``,
``"red_0"``, ``"north"``), an int (counts, distances, furnace and chest numbers), or one of the
structured values below:

- ``EntityRef(ordinal, entity)``: ``the second zombie`` is ``EntityRef(2, "zombie")`` and
  ``the nearest dropped bread`` is ``EntityRef(1, ("dropped", "bread"))``.
- ``Sentence(intent, **slots)``: a chat sentence, the value of the ``sentence`` slot of ``chat``.
- ``Goal(kind, **slots)``: a goal, the value of the ``goal`` slot of ``plan``.
- ``Event(kind, **fields)``: an event sentence of the notebook (never part of a command).

All of them are frozen and hashable, and keep their slots sorted by name, so two values that
read the same compare equal. Values are stored as plain Python (``plain_value``: a NumPy
integer becomes an int; a bool is refused), so ``to_json`` always serializes.
``Command.args`` holds exactly the template's slot names, plus
``template`` for the verbs that have more than one template (``goto``, ``attack``, ``flee``,
``equip``), because some templates of a verb have the same slots (``go to my bed`` and
``go home`` have none; ``hold {item}`` and ``wear {item}`` both have ``item``).
``Command.of(verb, **slots)`` fills ``template`` in when the slots name exactly one template.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

MAX_ORDINAL = 16  # contracts 4.3: nearest, second, ... sixteenth


class ParseError(ValueError):
    """Raised by the strict parser (and by ``decode``) for text that is not a canonical command.

    ``position`` is the index of the first word that could not be read and ``expected`` lists
    what the grammar allowed there (words in quotes, slots in braces).
    """

    def __init__(self, message: str, *, words: tuple[str, ...] = (), position: int | None = None,
                 expected: tuple[str, ...] = ()):
        super().__init__(message)
        self.words = words
        self.position = position
        self.expected = expected


class DeadEndError(RuntimeError):
    """A slot provider offered a choice after which no value can follow (contracts 4.5 requires
    every offered value to have at least one complete command)."""


@dataclass(frozen=True)
class EntityRef:
    """A visible entity named by rank and phrase: ``the {ordinal} {entity}`` (contracts 4.3).

    ``ordinal`` is 1 (``nearest``) to 16 (``sixteenth``), the rank by distance among visible
    entities with the same phrase. ``entity`` is an entity-class name (``"zombie"``) or
    ``("dropped", item_class)`` for an item lying on the ground.
    """

    ordinal: int
    entity: str | tuple[str, str]

    def __post_init__(self) -> None:
        k = self.ordinal
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, numbers.Integral) or not 1 <= k <= MAX_ORDINAL:
            raise ValueError(f"EntityRef ordinal must be an int in 1..{MAX_ORDINAL}, got {k!r}")
        object.__setattr__(self, "ordinal", int(k))  # a NumPy integer becomes an int
        e = self.entity
        if isinstance(e, (list, tuple)):
            e = tuple(plain_value(x) for x in e)
        elif isinstance(e, str):
            e = plain_value(e)
        object.__setattr__(self, "entity", e)
        if isinstance(e, tuple):
            if len(e) != 2 or e[0] != "dropped" or not isinstance(e[1], str):
                raise ValueError(f"EntityRef entity must be a class name or ('dropped', item), got {e!r}")
        elif not isinstance(e, str):
            raise TypeError(f"EntityRef entity must be a class name or ('dropped', item), got {e!r}")

    @property
    def dropped_item(self) -> str | None:
        """The item class of a dropped item, or None for a mob or other entity."""
        return self.entity[1] if isinstance(self.entity, tuple) else None

    def sort_key(self) -> tuple:
        e = self.entity
        return (self.ordinal, 1, e[1]) if isinstance(e, tuple) else (self.ordinal, 0, e)

    def to_json(self) -> dict:
        if isinstance(self.entity, tuple):
            return {"ordinal": self.ordinal, "dropped": self.entity[1]}
        return {"ordinal": self.ordinal, "entity": self.entity}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> EntityRef:
        if "dropped" in d:
            return cls(int(d["ordinal"]), ("dropped", d["dropped"]))
        return cls(int(d["ordinal"]), d["entity"])


def plain_value(value: Any) -> Any:
    """A slot value as plain Python, so it compares, hashes and serializes to JSON as itself:
    a NumPy integer becomes an int, a NumPy string a str, a list a tuple. A bool is refused:
    no slot takes one, and ``True == 1`` would let it pass as the number 1."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"a slot value is never a bool, got {value!r}")
    if isinstance(value, list):
        return tuple(plain_value(v) for v in value)
    if type(value) is not int and isinstance(value, numbers.Integral):
        return int(value)
    if type(value) is not str and isinstance(value, str):
        return str(value)
    return value


def _sorted_args(args: Iterable[Any], what: str) -> tuple[tuple[str, Any], ...]:
    pairs = []
    for a in args:
        if not isinstance(a, (tuple, list)) or len(a) != 2 or not isinstance(a[0], str):
            raise ValueError(f"{what} args must be (slot name, value) pairs, got {a!r}")
        try:
            pairs.append((a[0], plain_value(a[1])))
        except ValueError as e:
            raise ValueError(f"{what} slot {a[0]!r}: {e}") from None
    pairs.sort(key=lambda kv: kv[0])
    names = [k for k, _ in pairs]
    if len(set(names)) != len(names):
        raise ValueError(f"{what} has a repeated slot: {names}")
    return tuple(pairs)


class _Frame:
    """Base of Sentence, Goal and Event: a head (intent or kind) plus named slots."""

    __slots__ = ("_head", "args")
    _head_name: ClassVar[str] = "kind"

    def __init__(self, head: str, /, **slots: Any):
        if not isinstance(head, str):
            raise TypeError(f"{type(self).__name__} {self._head_name} must be a str, got {head!r}")
        object.__setattr__(self, "_head", head)
        object.__setattr__(self, "args", _sorted_args(slots.items(), type(self).__name__))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is frozen")

    def __reduce__(self):
        return (_make_frame, (type(self), self._head, self.args))

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and other._head == self._head and other.args == self.args  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._head, self.args))

    def __repr__(self) -> str:
        inner = "".join(f", {k}={v!r}" for k, v in self.args)
        return f"{type(self).__name__}({self._head!r}{inner})"

    def __getitem__(self, slot: str) -> Any:
        for k, v in self.args:
            if k == slot:
                return v
        raise KeyError(slot)

    def get(self, slot: str, default: Any = None) -> Any:
        try:
            return self[slot]
        except KeyError:
            return default

    @property
    def slots(self) -> dict[str, Any]:
        return dict(self.args)

    def to_json(self) -> dict:
        return {self._head_name: self._head, **{k: _value_to_json(v) for k, v in self.args}}


def _make_frame(cls: type, head: str, args: tuple) -> _Frame:
    return cls(head, **dict(args))


class Sentence(_Frame):
    """A chat sentence (contracts 4.3, DESIGN.md 9): ``Sentence("need", item_count=3, item="iron_ingot")``."""

    __slots__ = ()
    _head_name = "intent"

    @property
    def intent(self) -> str:
        return self._head


class Goal(_Frame):
    """A goal (DESIGN.md 6.6): ``Goal("get", item_count=3, item="iron_ingot")``."""

    __slots__ = ()
    _head_name = "kind"

    @property
    def kind(self) -> str:
        return self._head


class Event(_Frame):
    """An event sentence of the notebook (contracts 4.3, DESIGN.md 6.3):
    ``Event("player_gave_me", player="blue_7", n=3, item="bread")``. ``n`` keeps the exact count;
    the words show it snapped to the nearest number token."""

    __slots__ = ()
    _head_name = "kind"

    @property
    def kind(self) -> str:
        return self._head


@dataclass(frozen=True)
class Command:
    """A parsed command: verb id plus slot values sorted by slot name (contracts 4.4).

    ``Command("mine", (("block", "iron_ore"), ("mine_count", 4)))``. Args given in any order are
    sorted; lists are turned into tuples.
    """

    verb: str
    args: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.verb, str):
            raise TypeError(f"Command verb must be a str, got {self.verb!r}")
        object.__setattr__(self, "args", _sorted_args(self.args, f"Command {self.verb!r}"))

    @classmethod
    def of(cls, verb: str, /, **slots: Any) -> Command:
        """Build a command from keyword slots, filling ``template`` in when the verb has several
        templates and exactly one of them has these slots: ``Command.of("goto", block="log")``."""
        from hivemind.lang.grammar import infer_template  # the grammar imports this module

        if "template" not in slots:
            tid = infer_template(verb, set(slots))
            if tid is not None:
                slots["template"] = tid
        return cls(verb, tuple(slots.items()))

    def __getitem__(self, slot: str) -> Any:
        for k, v in self.args:
            if k == slot:
                return v
        raise KeyError(slot)

    def get(self, slot: str, default: Any = None) -> Any:
        try:
            return self[slot]
        except KeyError:
            return default

    @property
    def slots(self) -> dict[str, Any]:
        return dict(self.args)

    def to_json(self) -> dict:
        """``{verb, args}`` with plain JSON values (bridge/PROTOCOL.md 6.1)."""
        return {"verb": self.verb, "args": {k: _value_to_json(v) for k, v in self.args}}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Command:
        return cls(d["verb"], tuple((k, _value_from_json(k, v)) for k, v in d["args"].items()))


def _value_to_json(v: Any) -> Any:
    if isinstance(v, (EntityRef, _Frame)):
        return v.to_json()
    return v


def _value_from_json(slot: str, v: Any) -> Any:
    if isinstance(v, Mapping):
        if "ordinal" in v:
            return EntityRef.from_json(v)
        if "intent" in v:
            return Sentence(v["intent"], **{k: _value_from_json(k, x) for k, x in v.items() if k != "intent"})
        if "kind" in v:
            return Goal(v["kind"], **{k: _value_from_json(k, x) for k, x in v.items() if k != "kind"})
    return v


@dataclass(frozen=True)
class Flags:
    """Feature flags of the grammar (contracts 4.3): ``containers`` enables ``store`` and
    ``take`` (M9), ``goals`` enables ``plan`` (M11). Both default off. The decoder never
    allows a verb whose flag is off; the parser and the renderer handle every verb always."""

    containers: bool = False
    goals: bool = False

    @classmethod
    def all(cls) -> Flags:
        return cls(containers=True, goals=True)

    @classmethod
    def coerce(cls, flags: Flags | Mapping[str, bool] | Iterable[str] | None) -> Flags:
        """Accept a Flags, a mapping of flag names to bools, an iterable of enabled flag names,
        or None (all off)."""
        if flags is None:
            return cls()
        if isinstance(flags, Flags):
            return flags
        if isinstance(flags, Mapping):
            unknown = set(flags) - {"containers", "goals"}
            if unknown:
                raise ValueError(f"unknown feature flags {sorted(unknown)}")
            return cls(**{k: bool(v) for k, v in flags.items()})
        names = set(flags)
        unknown = names - {"containers", "goals"}
        if unknown:
            raise ValueError(f"unknown feature flags {sorted(unknown)}")
        return cls(containers="containers" in names, goals="goals" in names)

    def enabled(self, flag: str | None) -> bool:
        return flag is None or bool(getattr(self, flag))
