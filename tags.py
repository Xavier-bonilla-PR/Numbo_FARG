"""
tags.py — Frozen tag dataclasses for the FARG Travel Path Finder.

Tags annotate workspace elements without mutating them.  Every tag is a
frozen dataclass so it can live as a hashable key in the workspace dict.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple


# ── Base ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Tag:
    """Base class for all workspace tags."""
    pass


# ── Progress tags ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GettingCloser(Tag):
    """Marks an ImCell that is making progress toward the goal.

    weight: 0.0 (no progress) → 1.0 (reached goal).
    """
    taggee: Any
    weight: float


@dataclass(frozen=True)
class Blocked(Tag):
    """Marks an agent or ImCell that cannot make further progress.

    Agents tagged Blocked are excluded from the selection pool.
    """
    taggee: Any
    reason: str = ""


# ── Agent lifecycle tags ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class GoIsDone(Tag):
    """Marks an agent that has completed its .go() deliberation phase.

    Such an agent may now be eligible for .act() (commitment).
    """
    taggee: Any


@dataclass(frozen=True)
class ActIsDone(Tag):
    """Marks an agent that has already committed via .act().

    Such an agent should not be selected again for either phase.
    """
    taggee: Any


@dataclass(frozen=True)
class PendingLLM(Tag):
    """Marks an agent currently waiting for an async LLM response.

    PendingLLM agents are skipped during selection but still decay each tick.
    """
    taggee: Any


# ── Path outcome tags ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PathComplete(Tag):
    """Marks an ImCell whose current_loc equals the goal.

    path: ordered tuple of location strings from START to GOAL.
    """
    taggee: Any
    path: Tuple[str, ...]


@dataclass(frozen=True)
class GoalReached(Tag):
    """Marks a committed canvas chain (identified by path_id) that has reached GOAL.

    Applied per chain in SuggestRoute.act() the moment the final committed leg
    lands at the goal location — independently of ImCell feasibility scoring.
    This ensures every canvas path that completes is declared, replacing the
    single SolvedNumble-style halt with per-chain goal announcements.

    path_id: stable identifier for this canvas chain.
    path:    ordered tuple of location strings from START to GOAL.
    legs:    tuple of committed Leg objects for this chain.
    """
    path_id: str
    path: Tuple[str, ...]
    legs: Tuple  # Tuple[Leg, ...]
