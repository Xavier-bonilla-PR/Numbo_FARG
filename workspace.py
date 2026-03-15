"""
workspace.py — The FARG Workspace: activation graph + element registry.

The Workspace is the sole source of mutable state in the system.  Every
workspace element (agent, tag, ImCell, CanvasCell) lives here with an
activation level.  A networkx DiGraph stores support (+) and antipathy (−)
edges that drive activation propagation each tick.

Key design points
─────────────────
• elements: dict mapping hashable element → float activation
• graph:    nx.DiGraph with 'weight' edge attributes
              positive weight  →  support (builder↔built)
              negative weight  →  antipathy (competing paths)
• tags:     set of Tag instances (also present in elements)
• canvas:   list of CanvasCell (committed legs, ordered by path+position)
• Agent selection: weighted by activation², excludes PendingLLM / Blocked /
  ActIsDone-exhausted agents
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Type

import networkx as nx

from canvas import CanvasCell, ImCell
from config import (
    ANTIPATHY_WEIGHT,
    BOOST_FACTOR,
    DECAY_RATE,
    INIT_ACTIVATION,
    MIN_ACTIVATION,
    PROPAGATION_SCALE,
    SUPPORT_WEIGHT,
)
from tags import ActIsDone, Blocked, PendingLLM, Tag


# ── Agent type registry (import-time, avoids circular imports) ─────────────────

def _agent_types():
    from agents import AGENT_TYPES
    return AGENT_TYPES


class Workspace:
    """Central mutable state for the FARG travel path finder."""

    def __init__(self) -> None:
        self.elements: Dict[Any, float] = {}   # elem → activation
        self.graph: nx.DiGraph = nx.DiGraph()
        self.tags: Set[Tag] = set()
        self.canvas: List[CanvasCell] = []
        self._path_counter: int = 0

    # ── Path ID factory ───────────────────────────────────────────────────────

    def new_path_id(self) -> str:
        pid = f"path-{self._path_counter}"
        self._path_counter += 1
        return pid

    # ── Element registration ──────────────────────────────────────────────────

    def add(
        self,
        elem: Any,
        builder: Optional[Any] = None,
        init_a: Optional[float] = None,
    ) -> Any:
        """Register elem in the workspace; return the canonical instance.

        If elem is already present, return the existing instance unchanged.
        Otherwise:
          • Set initial activation (from builder if not provided).
          • Add a graph node.
          • If builder is given, add mutual support edge.
          • For every existing element that has antipathy to elem (or vice
            versa), add a mutual antipathy edge.
          • Call elem.on_build(self) if the method exists.
        """
        existing = self._find_existing(elem)
        if existing is not None:
            # Boost existing slightly when re-encountered
            self.elements[existing] = min(1.0, self.elements[existing] * 1.05)
            return existing

        # Determine initial activation
        if init_a is None:
            if builder is not None and builder in self.elements:
                init_a = min(1.0, self.elements[builder])
            else:
                init_a = INIT_ACTIVATION

        self.elements[elem] = init_a
        self.graph.add_node(elem)

        # Support edge with builder
        if builder is not None and builder in self.elements:
            self._add_support_edge(builder, elem)

        # Antipathy edges with conflicting existing elements
        for existing_elem in list(self.elements.keys()):
            if existing_elem is elem:
                continue
            # Check both directions
            elem_hates = (
                hasattr(elem, "has_antipathy_to")
                and elem.has_antipathy_to(existing_elem)
            )
            other_hates = (
                hasattr(existing_elem, "has_antipathy_to")
                and existing_elem.has_antipathy_to(elem)
            )
            if elem_hates or other_hates:
                self._add_antipathy_edge(elem, existing_elem)

        # Optional hook
        if hasattr(elem, "on_build"):
            elem.on_build(self)

        return elem

    def _find_existing(self, elem: Any) -> Optional[Any]:
        """Return the canonical workspace copy of elem if present.

        Iterates the elements dict to return the *stored* key rather than
        the caller's copy, making this the single deduplication chokepoint.
        Extend here (e.g. with elem.structurally_matches(stored)) to add
        fuzzy / structural dedup without touching add().
        """
        for stored in self.elements:
            if stored == elem:
                return stored
        return None

    # ── Edge management ───────────────────────────────────────────────────────

    def _add_support_edge(self, a: Any, b: Any) -> None:
        """Bidirectional positive-weight (support) edge between a and b."""
        if not self.graph.has_node(a):
            self.graph.add_node(a)
        if not self.graph.has_node(b):
            self.graph.add_node(b)
        self.graph.add_edge(a, b, weight=SUPPORT_WEIGHT)
        self.graph.add_edge(b, a, weight=SUPPORT_WEIGHT)

    def _add_antipathy_edge(self, a: Any, b: Any) -> None:
        """Bidirectional negative-weight (antipathy) edge between a and b."""
        if not self.graph.has_node(a):
            self.graph.add_node(a)
        if not self.graph.has_node(b):
            self.graph.add_node(b)
        self.graph.add_edge(a, b, weight=ANTIPATHY_WEIGHT)
        self.graph.add_edge(b, a, weight=ANTIPATHY_WEIGHT)

    # ── Activation dynamics ───────────────────────────────────────────────────

    def decay_all(self) -> None:
        """Multiply every element's activation by DECAY_RATE."""
        for elem in self.elements:
            self.elements[elem] *= DECAY_RATE

    def propagate(self) -> None:
        """Flow activation along all edges; clamp results to [0.0, 1.0].

        Each edge (u → v, weight w) contributes:
            delta_v += activation(u) * w * PROPAGATION_SCALE

        Negative deltas (from antipathy) reduce the target's activation.
        """
        deltas: Dict[Any, float] = {e: 0.0 for e in self.elements}

        for u, v, data in self.graph.edges(data=True):
            if u not in self.elements or v not in self.elements:
                continue
            weight = data.get("weight", 0.0)
            flow = self.elements[u] * weight * PROPAGATION_SCALE
            deltas[v] = deltas.get(v, 0.0) + flow

        for elem, delta in deltas.items():
            if elem in self.elements:
                new_a = self.elements[elem] + delta
                self.elements[elem] = max(0.0, min(1.0, new_a))

    def boost(self, elem: Any) -> None:
        """Increase activation: a += BOOST_FACTOR * a  (capped at 1.0)."""
        if elem in self.elements:
            a = self.elements[elem]
            self.elements[elem] = min(1.0, a + BOOST_FACTOR * a)

    def downboost(self, elem: Any) -> None:
        """Halve activation (used after an agent commits via .act())."""
        if elem in self.elements:
            self.elements[elem] /= 2.0

    # ── Tag management ────────────────────────────────────────────────────────

    def tag(self, t: Tag) -> None:
        """Add tag to the tag set and register it as a workspace element."""
        self.tags.add(t)
        self.add(t)   # tags compete for activation too

    def untag(self, elem: Any, tag_cls: Type[Tag]) -> None:
        """Remove all tags of type tag_cls whose .taggee == elem."""
        to_remove = {
            t for t in self.tags
            if isinstance(t, tag_cls) and getattr(t, "taggee", None) == elem
        }
        for t in to_remove:
            self.tags.discard(t)
            self.elements.pop(t, None)
            if self.graph.has_node(t):
                self.graph.remove_node(t)

    def has_tag(self, elem: Any, tag_cls: Type[Tag]) -> bool:
        return any(
            isinstance(t, tag_cls) and getattr(t, "taggee", None) == elem
            for t in self.tags
        )

    def tags_on(self, elem: Any) -> List[Tag]:
        return [t for t in self.tags if getattr(t, "taggee", None) == elem]

    # ── Agent queries ─────────────────────────────────────────────────────────

    def all_agents(self) -> List[Any]:
        """Return all workspace elements that are agents."""
        agent_types = _agent_types()
        return [e for e in self.elements if isinstance(e, agent_types)]

    def agents_not_pending(self) -> List[Any]:
        """Agents eligible for selection this tick.

        Excluded:
          • Tagged PendingLLM (waiting for LLM response)
          • Tagged Blocked (dead end)
          • Can neither .go() nor .act() right now (exhausted agents)
        """
        agent_types = _agent_types()
        result = []
        for elem in self.elements:
            if not isinstance(elem, agent_types):
                continue
            if self.has_tag(elem, PendingLLM):
                continue
            if self.has_tag(elem, Blocked):
                continue
            can_g = hasattr(elem, "can_go") and elem.can_go(self)
            can_a = hasattr(elem, "can_act") and elem.can_act(self)
            if not can_g and not can_a:
                continue   # exhausted: skip but let it decay
            result.append(elem)
        return result

    def choose_agent(self, agents: List[Any]) -> Optional[Any]:
        """Weighted random selection; weight = activation²."""
        if not agents:
            return None
        weights = [max(0.0, self.elements.get(a, 0.0)) ** 2 for a in agents]
        total = sum(weights)
        if total <= 0.0:
            return random.choice(agents)
        return random.choices(agents, weights=weights, k=1)[0]

    # ── ImCell queries ────────────────────────────────────────────────────────

    def imcell_for(self, path_id: str) -> Optional[ImCell]:
        """Return the best incomplete ImCell for path_id (highest activation,
        excluding ImCells already tagged PathComplete to prevent re-extension).
        """
        from tags import PathComplete
        candidates = [
            e for e in self.elements
            if isinstance(e, ImCell)
            and e.path_id == path_id
            and not self.has_tag(e, PathComplete)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda c: self.elements[c])

    def all_imcells(self) -> List[ImCell]:
        return [e for e in self.elements if isinstance(e, ImCell)]

    # ── Canvas ────────────────────────────────────────────────────────────────

    def commit_leg(self, path_id: str, leg) -> CanvasCell:
        """Append a committed leg to the canvas and return the CanvasCell."""
        pos = sum(1 for c in self.canvas if c.path_id == path_id)
        cell = CanvasCell(path_id=path_id, leg=leg, position=pos)
        self.canvas.append(cell)
        return cell

    def canvas_legs(self, path_id: str) -> List[CanvasCell]:
        return [c for c in self.canvas if c.path_id == path_id]

    # ── Pruning ───────────────────────────────────────────────────────────────

    def prune(self) -> None:
        """Remove elements whose activation has fallen below MIN_ACTIVATION."""
        dead = [e for e, a in list(self.elements.items()) if a < MIN_ACTIVATION]
        for e in dead:
            self.elements.pop(e, None)
            if self.graph.has_node(e):
                self.graph.remove_node(e)
            if isinstance(e, Tag):
                self.tags.discard(e)

    # ── Diagnostic helpers ────────────────────────────────────────────────────

    def activation(self, elem: Any) -> float:
        return self.elements.get(elem, 0.0)

    def top_elements(self, n: int = 8) -> List[tuple]:
        return sorted(self.elements.items(), key=lambda x: -x[1])[:n]

    def __repr__(self) -> str:
        return (
            f"Workspace({len(self.elements)} elems, "
            f"{len(self.canvas)} canvas cells, "
            f"{len(self.tags)} tags)"
        )
