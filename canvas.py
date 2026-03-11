"""
canvas.py — Workspace element dataclasses for route representation.

ImCell  — a hypothetical, uncommitted route being explored in "imagination".
CanvasCell — a committed route leg written to the canvas (real state).
Leg     — a single travel segment between two locations.

All are frozen dataclasses so they are hashable and can live as workspace keys.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Leg:
    """One travel segment between two locations.

    mode: one of "flight", "train", "bus", "boat", "motorbike".
    duration_hours: approximate travel time.
    notes: free-text description (scenic highlights, service name, etc.).
    """
    from_loc: str
    to_loc: str
    mode: str
    duration_hours: float = 0.0
    notes: str = ""

    def __str__(self) -> str:
        return f"{self.from_loc} →[{self.mode}]→ {self.to_loc}"


@dataclass(frozen=True)
class ImCell:
    """Hypothetical (uncommitted) route being explored.

    Lives in the workspace as a speculative structure.  Agents work with
    ImCells during their .go() phase; legs are not yet on the canvas.

    path_id: stable identifier shared by all elements of the same path.
    legs:    tuple of Legs accumulated so far (immutable; extend via with_leg).
    current_loc: the location this path has reached so far.
    """
    path_id: str
    legs: Tuple[Leg, ...]
    current_loc: str

    def with_leg(self, leg: Leg) -> ImCell:
        """Return a new ImCell extended by one leg."""
        return ImCell(
            path_id=self.path_id,
            legs=self.legs + (leg,),
            current_loc=leg.to_loc,
        )

    def total_hours(self) -> float:
        return sum(l.duration_hours for l in self.legs)

    def location_sequence(self) -> Tuple[str, ...]:
        if not self.legs:
            return (self.current_loc,)
        return (self.legs[0].from_loc,) + tuple(l.to_loc for l in self.legs)

    def __str__(self) -> str:
        locs = " → ".join(self.location_sequence())
        return f"ImCell[{self.path_id}]: {locs}"


@dataclass(frozen=True)
class CanvasCell:
    """Committed route leg written to the canvas.

    Once an agent calls .act(), the proposed leg becomes a CanvasCell.
    Canvas cells are stored in Workspace.canvas (a plain list, ordered by
    position within each path_id).
    """
    path_id: str
    leg: Leg
    position: int       # 0-based index within this path

    def __str__(self) -> str:
        return f"Canvas[{self.path_id}#{self.position}]: {self.leg}"
