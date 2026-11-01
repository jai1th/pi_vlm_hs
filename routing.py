"""Item -> bag mapping, editable from the dashboard's Routing tab.

The mapping lives in routing.json next to main.py (--routing); without that
file the built-in DEFAULT_ROUTING applies. main.py reads it once at startup,
so a saved change takes effect when main.py is restarted.

A GREEN item still needs a clean spectral reading to reach the GREEN bag,
unless it is listed as unchecked: then its label alone sends it GREEN, and
a dirty or full item can reach the GREEN bag. Only the dashboard's explicit
confirmation sets that.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass, field

from contamination_config import CONTAM_MODEL, GATE_NOT_IMPLEMENTED, SUBSTRATE_OF

GREEN = "green"
CLEAR = "clear"
BAGS = (GREEN, CLEAR)
ROUTING_PATH = "routing.json"
SCHEMA_VERSION = 1

# The model outputs 9 classes. togo_cup spans both bins in the original routing
# split (Paper cup, Foam cup -> CLEAR; Disposable plastic cup -> GREEN) and
# follows the majority -> CLEAR.
#
# togo_box is GREEN for the POC on the assumption of CARTON stock. The class also spans
# EPS foam and PET, which have different clean baselines; a foam box would be given the
# carton reference and produce a meaningless d. Showcase with carton boxes only.
DEFAULT_STREAMS = {
    "togo_box":       GREEN,
    "plastic_bottle": GREEN,
    "paper":          GREEN,
    "togo_cup":       CLEAR,
    "cup_lid":        CLEAR,
    "straw":          CLEAR,
    "tissue":         CLEAR,
    "wrapper":        CLEAR,
    "glass_bottle":   CLEAR,
}
LABELS = tuple(DEFAULT_STREAMS)


class RoutingError(Exception):
    pass


@dataclass(frozen=True)
class Routing:
    streams: dict                                  # label -> GREEN / CLEAR
    unchecked: frozenset = field(default_factory=frozenset)  # GREEN labels that skip the check
    updated: str | None = None                     # when it was saved; None = built-in

    @property
    def digest(self):
        """Short fingerprint of what routes where, logged with every decision."""
        body = json.dumps({"streams": self.streams, "unchecked": sorted(self.unchecked)},
                          sort_keys=True)
        return hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]

    def with_edit(self, label, bag=None, unchecked=None):
        """A copy with one item changed. A CLEAR item never skips the check."""
        streams = dict(self.streams)
        if bag is not None:
            streams[label] = bag
        skip = set(self.unchecked)
        if unchecked is not None:
            (skip.add if unchecked else skip.discard)(label)
        skip = {l for l in skip if streams[l] == GREEN}
        return Routing(streams, frozenset(skip), self.updated)


DEFAULT_ROUTING = Routing(dict(DEFAULT_STREAMS))


def from_dict(raw):
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise RoutingError("expected schema_version {}".format(SCHEMA_VERSION))
    items = raw.get("items")
    if not isinstance(items, dict):
        raise RoutingError("'items' must be an object of item -> settings")
    routing = Routing(dict(DEFAULT_STREAMS), frozenset(), raw.get("updated"))
    for label, item in items.items():
        if label not in DEFAULT_STREAMS:
            raise RoutingError("unknown item {!r} (the model's items are: {})".format(
                label, ", ".join(LABELS)))
        if not isinstance(item, dict) or item.get("bag") not in BAGS:
            raise RoutingError("{}: 'bag' must be one of {}".format(label, ", ".join(BAGS)))
        skip = item.get("skip_contamination_check", False)
        if not isinstance(skip, bool):
            raise RoutingError("{}: 'skip_contamination_check' must be true or false".format(label))
        routing = routing.with_edit(label, bag=item["bag"], unchecked=skip)
    return routing


def to_dict(routing):
    return {
        "schema_version": SCHEMA_VERSION,
        "updated": routing.updated,
        "items": {label: {"bag": routing.streams[label],
                          "skip_contamination_check": label in routing.unchecked}
                  for label in LABELS},
    }


def load(path=ROUTING_PATH):
    """(routing, problem). No file: the defaults, no problem. An unreadable or
    invalid file: the defaults plus why, so the caller can say so loudly."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return DEFAULT_ROUTING, None
    except (OSError, ValueError) as exc:
        return DEFAULT_ROUTING, "unreadable {}: {}".format(path, exc)
    try:
        return from_dict(raw), None
    except RoutingError as exc:
        return DEFAULT_ROUTING, "invalid {}: {}".format(path, exc)


def save(routing, path=ROUTING_PATH):
    """Write atomically, stamped with the save time; returns what was written."""
    stamped = Routing(dict(routing.streams), routing.unchecked,
                      dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_dict(stamped), f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return stamped


def outcome(routing, label):
    """What the decision node actually does with a frame of this item alone."""
    if routing.streams[label] == CLEAR:
        return "CLEAR"
    if label in routing.unchecked:
        return "GREEN on the label alone: no contamination check"
    if label in GATE_NOT_IMPLEMENTED:
        return "CLEAR: its {} check is not built".format(GATE_NOT_IMPLEMENTED[label].split(":")[0])
    substrate = SUBSTRATE_OF.get(label)
    if substrate in CONTAM_MODEL:
        return "GREEN if the {} spectral check is clean, else CLEAR".format(substrate)
    return "CLEAR: no spectral model for it"


def changes(old, new):
    """Human-readable differences between two mappings."""
    out = []
    for label in LABELS:
        if old.streams[label] != new.streams[label]:
            out.append("{}: {} → {}".format(label, old.streams[label].upper(),
                                            new.streams[label].upper()))
        if (label in old.unchecked) != (label in new.unchecked):
            out.append("{}: contamination check {}".format(
                label, "skipped" if label in new.unchecked else "required again"))
    return out
