"""
logger.py — Tick-by-tick logging for the FARG Travel Path Finder.

Prints a human-readable summary each tick showing:
  • Tick number
  • The agent that ran and its activation
  • Top workspace elements by activation
  • Newly discovered complete paths
"""

from __future__ import annotations

import sys
from typing import Any, List, Optional


class TickLogger:
    """Prints structured tick summaries to stdout (or a custom stream)."""

    def __init__(self, stream=None, verbose: bool = True) -> None:
        self.stream = stream or sys.stdout
        self.verbose = verbose
        self._seen_paths: set = set()

    def log_tick(
        self,
        tick: int,
        ws,
        chosen: Optional[Any] = None,
        complete: Optional[List] = None,
    ) -> None:
        if not self.verbose:
            return

        out = self.stream
        out.write(f"\n-- Tick {tick:03d} {'-'*44}\n")

        if chosen is not None:
            a = ws.activation(chosen)
            label = _short(chosen)
            out.write(f"  > {label}  (a={a:.3f})\n")

        # Top elements
        top = ws.top_elements(n=6)
        if top:
            out.write("  Activations:\n")
            for elem, act in top:
                mark = _type_mark(elem)
                out.write(f"    {act:.3f} {mark} {_short(elem)}\n")

        # Canvas state
        if ws.canvas:
            out.write(f"  Canvas ({len(ws.canvas)} legs):\n")
            by_path: dict = {}
            for cell in ws.canvas:
                by_path.setdefault(cell.path_id, []).append(cell)
            for pid, cells in sorted(by_path.items()):
                locs = " ->".join(
                    ([cells[0].leg.from_loc] + [c.leg.to_loc for c in cells])
                )
                out.write(f"    [{pid}] {locs}\n")

        # Newly complete paths
        if complete:
            new_paths = [p for p in complete if id(p) not in self._seen_paths]
            for p in new_paths:
                self._seen_paths.add(id(p))
                route = " ->".join(p.path)
                out.write(f"  * PATH FOUND [{p.taggee.path_id}]: {route}\n")

        out.flush()

    def log_final(self, paths: List) -> None:
        out = self.stream
        sep = "=" * 54
        out.write(f"\n{sep}\n")
        out.write(f"  FARG discovered {len(paths)} competing paths\n")
        out.write(f"{sep}\n")
        for i, p in enumerate(paths, 1):
            route = " ->".join(p.path)
            modes = _path_modes(p)
            out.write(f"  Path {i}: {route}\n")
            if modes:
                out.write(f"          [{modes}]\n")
        out.write(f"{sep}\n")
        out.flush()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _short(elem: Any) -> str:
    """Single-line label for any workspace element."""
    from canvas import ImCell, CanvasCell, Leg
    from agents import Want, SuggestMode, SuggestRoute, SeekEvidence, Evaluate
    from tags import (
        GettingCloser, Blocked, GoIsDone, ActIsDone, PendingLLM, PathComplete, Tag
    )

    if isinstance(elem, Want):
        return f"Want({elem.from_loc}->{elem.to_loc})"
    if isinstance(elem, SuggestMode):
        return f"SuggestMode[{elem.path_id}] {elem.current_loc}/{elem.mode}"
    if isinstance(elem, SuggestRoute):
        return (
            f"SuggestRoute[{elem.path_id}] "
            f"{elem.proposed_leg.from_loc}->{elem.proposed_leg.to_loc}"
            f"({elem.proposed_leg.mode})"
        )
    if isinstance(elem, SeekEvidence):
        return f"SeekEvidence[{elem.path_id}]"
    if isinstance(elem, Evaluate):
        return f"Evaluate[{elem.path_id_a}<->{elem.path_id_b}]"
    if isinstance(elem, ImCell):
        locs = " ->".join(elem.location_sequence())
        return f"ImCell[{elem.path_id}]: {locs}"
    if isinstance(elem, GettingCloser):
        return f"GettingCloser(w={elem.weight:.2f})"
    if isinstance(elem, PathComplete):
        return f"PathComplete[{elem.taggee.path_id}]"
    if isinstance(elem, Tag):
        return f"{type(elem).__name__}"
    return repr(elem)[:60]


def _type_mark(elem: Any) -> str:
    from canvas import ImCell
    from agents import AGENT_TYPES
    from tags import Tag
    if isinstance(elem, ImCell):
        return "[M]"
    if isinstance(elem, AGENT_TYPES):
        return "[A]"
    if isinstance(elem, Tag):
        return "[T]"
    return " "


def _path_modes(p) -> str:
    try:
        modes = list({l.mode for l in p.taggee.legs})
        return ", ".join(sorted(modes))
    except Exception:
        return ""
