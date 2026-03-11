"""
geography.py — Static knowledge about Vietnamese geography and transport.

Used by MockSlipnet for deterministic testing and by agents / detectors for
route validation.  RealSlipnet (LLM) may return locations not listed here;
that is intentional – the LLM can discover routes the static graph doesn't know.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

# ── Locations ─────────────────────────────────────────────────────────────────

LOCATIONS: List[str] = [
    "Hanoi",
    "Ninh Binh",
    "Ha Long Bay",
    "Vinh",
    "Dong Hoi",
    "Hue",
    "Hoi An",
    "Da Nang",
]

# ── Points of interest ────────────────────────────────────────────────────────

INTERESTING_STOPS: Dict[str, str] = {
    "Ha Long Bay": "UNESCO World Heritage – spectacular karst seascape",
    "Hue":         "Imperial Citadel, UNESCO heritage city",
    "Hoi An":      "Ancient Town, UNESCO World Heritage",
    "Ninh Binh":   "Inland 'Ha Long Bay' – Trang An caves, Tam Coc",
    "Dong Hoi":    "Gateway to Phong Nha-Ke Bang UNESCO cave system",
}

# ── Transport connections ─────────────────────────────────────────────────────
# (from_loc, to_loc) → list of (mode, duration_hours)

ConnectionList = List[Tuple[str, float]]
CONNECTIONS: Dict[Tuple[str, str], ConnectionList] = {
    # Direct long-haul
    ("Hanoi",      "Da Nang"):      [("flight",  1.50)],

    # Hanoi departures
    ("Hanoi",      "Hue"):          [("train",  13.00), ("flight",  1.25), ("bus", 14.00)],
    ("Hanoi",      "Ninh Binh"):    [("bus",     2.00), ("train",   2.50)],
    ("Hanoi",      "Ha Long Bay"):  [("bus",     3.50)],
    ("Hanoi",      "Vinh"):         [("train",   5.50), ("bus",     6.00)],

    # Ha Long Bay
    ("Ha Long Bay", "Hanoi"):       [("bus",     3.50)],

    # Ninh Binh onwards
    ("Ninh Binh",  "Hue"):          [("train",  10.00), ("bus",    11.00)],
    ("Ninh Binh",  "Vinh"):         [("train",   3.50), ("bus",     4.00)],

    # Vinh corridor
    ("Vinh",       "Dong Hoi"):     [("train",   2.00), ("bus",     2.50)],
    ("Vinh",       "Hue"):          [("train",   4.50), ("bus",     5.00)],

    # Dong Hoi
    ("Dong Hoi",   "Hue"):          [("train",   2.50), ("bus",     3.00)],

    # Hue connections
    ("Hue",        "Da Nang"):      [("train",   2.50), ("bus",     3.00), ("motorbike", 2.00)],
    ("Hue",        "Hoi An"):       [("bus",     3.50)],

    # Hoi An
    ("Hoi An",     "Da Nang"):      [("bus",     0.75), ("motorbike", 0.50)],
}

# ── Helper functions ──────────────────────────────────────────────────────────

def connections_from(loc: str) -> Dict[str, ConnectionList]:
    """Return all outbound connections from loc as {to_loc: [(mode, hours)]}."""
    result: Dict[str, ConnectionList] = {}
    for (frm, to), opts in CONNECTIONS.items():
        if frm == loc:
            result[to] = opts
    return result


def is_interesting(loc: str) -> bool:
    return loc in INTERESTING_STOPS


def all_modes_between(from_loc: str, to_loc: str) -> List[str]:
    opts = CONNECTIONS.get((from_loc, to_loc), [])
    return [mode for mode, _ in opts]


def duration(from_loc: str, to_loc: str, mode: str) -> float:
    opts = CONNECTIONS.get((from_loc, to_loc), [])
    for m, h in opts:
        if m == mode:
            return h
    return 0.0
