"""
metrics.py — Structured metrics collection for the FARG Travel Path Finder dashboard.

MetricsCollector implements the same log_tick/log_final interface as TickLogger,
so it can be passed as the 'logger' argument to run_loop() and records structured
data at each tick without changing the core architecture.

Data collected
──────────────
TickSnapshot  : per-tick activations, type counts, act-probability, pending-LLM
LLMLegRecord  : lifecycle of each LLM-proposed leg (birth → survival → commit)
TempCall      : (tick, path_id, activation, temperature) for temperature tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class LLMLegRecord:
    """Tracks a single LLM-proposed leg from birth through to canvas commit."""
    birth_tick: int
    path_id: str
    leg_key: str          # "from_loc->to_loc(mode)"
    alive_at_10: Optional[bool] = None   # checked at birth_tick + 10
    committed: bool = False


@dataclass
class TickSnapshot:
    """All dashboard-relevant metrics at one tick."""
    tick: int
    # id_str -> (type_label, activation)
    activations: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    # type_name -> count of live workspace elements
    type_counts: Dict[str, int] = field(default_factory=dict)
    # fraction of eligible agents whose can_act() is True
    act_prob: float = 0.0
    # agents currently tagged PendingLLM
    pending_llm: int = 0
    # cumulative committed canvas legs
    canvas_total: int = 0
    # number of PathComplete tags
    complete_count: int = 0
    # (from_id, to_id, weight) for the competition graph
    edges: List[Tuple[str, str, float]] = field(default_factory=list)
    # type label of the agent chosen this tick (None if no agent chosen)
    chosen_type: Optional[str] = None


# ── MetricsCollector ──────────────────────────────────────────────────────────

class MetricsCollector:
    """Drop-in replacement / companion to TickLogger that records structured data.

    Usage::

        mc = MetricsCollector()
        run_loop(ws, slipnet, logger=mc, max_ticks=120)
        snapshots = mc.snapshots
        total, survived, committed = mc.survival_funnel()
    """

    def __init__(self) -> None:
        self.snapshots: List[TickSnapshot] = []
        self.llm_legs: List[LLMLegRecord] = []
        self._seen_suggest_routes: set = set()
        self._canvas_prev: int = 0

    # ── Element introspection helpers ─────────────────────────────────────────

    @staticmethod
    def _elem_type(elem: Any) -> str:
        from agents import Want, SuggestMode, SuggestRoute, SeekEvidence, Evaluate
        from canvas import ImCell
        from tags import PendingLLM, Blocked, GettingCloser, PathComplete, GoIsDone, ActIsDone
        if isinstance(elem, Want):           return "Want"
        if isinstance(elem, SuggestMode):   return "SuggestMode"
        if isinstance(elem, SuggestRoute):  return "SuggestRoute"
        if isinstance(elem, SeekEvidence):  return "SeekEvidence"
        if isinstance(elem, Evaluate):      return "Evaluate"
        if isinstance(elem, ImCell):        return "ImCell"
        if isinstance(elem, PendingLLM):    return "PendingLLM"
        if isinstance(elem, Blocked):       return "Blocked"
        if isinstance(elem, GettingCloser): return "GettingCloser"
        if isinstance(elem, PathComplete):  return "PathComplete"
        if isinstance(elem, GoIsDone):      return "GoIsDone"
        if isinstance(elem, ActIsDone):     return "ActIsDone"
        return "Other"

    @staticmethod
    def _elem_id(elem: Any) -> str:
        from agents import Want, SuggestMode, SuggestRoute, SeekEvidence, Evaluate
        from canvas import ImCell
        from tags import Tag
        if isinstance(elem, SuggestMode):
            return f"SM[{elem.path_id}/{elem.mode[:4]}]"
        if isinstance(elem, SuggestRoute):
            leg = elem.proposed_leg
            return f"SR[{elem.path_id}:{leg.from_loc[:3]}->{leg.to_loc[:3]}]"
        if isinstance(elem, ImCell):
            return f"IC[{elem.path_id}@{elem.current_loc[:5]}]"
        if isinstance(elem, SeekEvidence):
            return f"SE[{elem.path_id}]"
        if isinstance(elem, Evaluate):
            return f"EV[{elem.path_id_a}|{elem.path_id_b}]"
        if isinstance(elem, Want):
            return f"Want[{elem.from_loc}->{elem.to_loc}]"
        if isinstance(elem, Tag):
            taggee = getattr(elem, "taggee", "?")
            taggee_id = MetricsCollector._elem_id(taggee) if taggee != "?" else "?"
            return f"{type(elem).__name__}({taggee_id})"
        return repr(elem)[:24]

    # ── logger interface ──────────────────────────────────────────────────────

    def log_tick(self, tick: int, ws, chosen=None, complete=None) -> None:
        from agents import SuggestRoute, AGENT_TYPES
        from tags import PendingLLM

        # ── 1. Track new LLM-proposed legs (new SuggestRoute entries) ─────────
        for elem in ws.elements:
            if isinstance(elem, SuggestRoute) and id(elem) not in self._seen_suggest_routes:
                self._seen_suggest_routes.add(id(elem))
                leg = elem.proposed_leg
                key = f"{leg.from_loc}->{leg.to_loc}({leg.mode})"
                self.llm_legs.append(LLMLegRecord(
                    birth_tick=tick,
                    path_id=elem.path_id,
                    leg_key=key,
                ))

        # ── 2. Check 10-tick survival ──────────────────────────────────────────
        for record in self.llm_legs:
            if record.alive_at_10 is None and tick >= record.birth_tick + 10:
                record.alive_at_10 = any(
                    isinstance(e, SuggestRoute)
                    and f"{e.proposed_leg.from_loc}->{e.proposed_leg.to_loc}({e.proposed_leg.mode})" == record.leg_key
                    for e in ws.elements
                )

        # ── 3. Check new canvas commits ────────────────────────────────────────
        canvas_now = len(ws.canvas)
        if canvas_now > self._canvas_prev:
            for cell in ws.canvas[self._canvas_prev:]:
                key = f"{cell.leg.from_loc}->{cell.leg.to_loc}({cell.leg.mode})"
                for record in self.llm_legs:
                    if record.leg_key == key and not record.committed:
                        record.committed = True
                        break
            self._canvas_prev = canvas_now

        # ── 4. Activations snapshot ────────────────────────────────────────────
        activations: Dict[str, Tuple[str, float]] = {}
        type_counts: Dict[str, int] = {}
        for elem, a in ws.elements.items():
            eid = self._elem_id(elem)
            etype = self._elem_type(elem)
            activations[eid] = (etype, a)
            type_counts[etype] = type_counts.get(etype, 0) + 1

        # ── 5. Act-probability: fraction of eligible agents that can_act ───────
        eligible = ws.agents_not_pending()
        can_act_n = sum(1 for a in eligible if hasattr(a, "can_act") and a.can_act(ws))
        act_prob = can_act_n / max(1, len(eligible))

        # ── 6. PendingLLM count ────────────────────────────────────────────────
        pending_llm = sum(1 for t in ws.tags if isinstance(t, PendingLLM))

        # ── 7. Competition graph edges ─────────────────────────────────────────
        edges: List[Tuple[str, str, float]] = []
        for u, v, data in ws.graph.edges(data=True):
            if u in ws.elements and v in ws.elements:
                uid = self._elem_id(u)
                vid = self._elem_id(v)
                w = data.get("weight", 0.0)
                edges.append((uid, vid, w))

        self.snapshots.append(TickSnapshot(
            tick=tick,
            activations=activations,
            type_counts=type_counts,
            act_prob=act_prob,
            pending_llm=pending_llm,
            canvas_total=canvas_now,
            complete_count=len(complete) if complete else 0,
            edges=edges,
            chosen_type=self._elem_type(chosen) if chosen else None,
        ))

    def log_final(self, paths) -> None:
        pass  # data already in snapshots; dashboard reads paths separately

    # ── Derived metrics ───────────────────────────────────────────────────────

    def survival_funnel(self) -> Tuple[int, int, int]:
        """Return (total_proposed, survived_10_ticks, committed_to_canvas)."""
        total = len(self.llm_legs)
        survived = sum(1 for r in self.llm_legs if r.alive_at_10 is True)
        committed = sum(1 for r in self.llm_legs if r.committed)
        return total, survived, committed

    def temperature_series(self) -> List[Tuple[int, str, float, float]]:
        """Return [(tick, path_id, activation, temperature)] for all SuggestMode firings.

        Temperature is derived from the activation recorded at each tick where
        a SuggestMode was the chosen agent (activation -> temperature via slipnet formula).
        """
        from slipnet import temperature_for_activation
        result = []
        for snap in self.snapshots:
            if snap.chosen_type == "SuggestMode":
                # Find all SuggestMode activations at this tick
                for eid, (etype, a) in snap.activations.items():
                    if etype == "SuggestMode":
                        temp = temperature_for_activation(a)
                        result.append((snap.tick, eid, a, temp))
        return result
