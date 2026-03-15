"""
baseline.py — Fixed sequential pipeline with no FARG dynamics.

Calls the exact same four slipnet functions as the FARG system but in a
straight line: no activation, no competition, no antipathy, no workspace.

Pipeline per mode
─────────────────
1. query_modes(start, goal)
       → list of transport modes
2. query_route(current, goal, mode, ...) × N
       → build one complete path leg-by-leg until goal reached
3. evaluate_path(path_id, legs, ...)
       → 0–1 quality score for each complete path
4. compare_paths(a, b, ...)
       → bubble-sort all complete paths by head-to-head comparison

IMPORTANT: run_baseline() does NOT wrap slipnet internally.  The caller is
responsible for passing a CountingSlipnet if call counting is desired.  This
mirrors how travel_dashboard.run_simulation() works and avoids a double-wrap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from canvas import Leg

log = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class BaselineResult:
    """One complete path produced by the baseline pipeline."""
    mode: str
    legs: Tuple[Leg, ...]
    score: float
    path: Tuple[str, ...]       # (start, ..., goal) location sequence

    def total_hours(self) -> float:
        return sum(l.duration_hours for l in self.legs)


@dataclass
class BaselineRun:
    """All results from one baseline run, plus aggregate statistics."""
    paths: List[BaselineResult] = field(default_factory=list)
    llm_calls: int = 0                          # total calls (backward compat)
    modes_tried: int = 0                        # how many modes were attempted
    modes_complete: int = 0                     # how many modes reached goal
    calls_by_type: Dict[str, int] = field(default_factory=dict)  # per-type counts


# ── Mock imcell shim (compare_paths expects .legs and .current_loc) ───────────

class _Cell:
    def __init__(self, result: BaselineResult) -> None:
        self.legs = result.legs
        self.current_loc = result.path[-1]


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_baseline(
    slipnet,
    start: str,
    goal: str,
    max_legs: int = 8,
) -> BaselineRun:
    """Run the fixed sequential pipeline and return a BaselineRun.

    Pass a CountingSlipnet from the caller for transparent call accounting.
    The caller should set base_run.calls_by_type and base_run.llm_calls
    from the CountingSlipnet after this function returns.
    """
    run = BaselineRun()

    # ── Step 1: get modes ─────────────────────────────────────────────────────
    modes = slipnet.query_modes(start, goal)
    run.modes_tried = len(modes)
    log.info("Baseline: %d modes: %s", len(modes), modes)

    # ── Step 2: build one path per mode ──────────────────────────────────────
    for mode in modes:
        legs: List[Leg] = []
        current = start
        visited: List[str] = [start]

        for step in range(max_legs):
            leg_data = slipnet.query_route(
                current, goal, mode,
                activation=0.5,
                visited_locs=visited,
                depth=step,
            )

            if leg_data is None:
                log.debug("Baseline [%s]: no route returned at step %d", mode, step)
                break

            try:
                leg = Leg(
                    from_loc=str(leg_data["from_loc"]),
                    to_loc=str(leg_data["to_loc"]),
                    mode=str(leg_data.get("mode", mode)),
                    duration_hours=float(leg_data.get("duration_hours") or 0.0),
                    notes=str(leg_data.get("notes", "")),
                )
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("Baseline [%s]: bad leg_data %s: %s", mode, leg_data, exc)
                break

            if leg.to_loc in visited and leg.to_loc != goal:
                log.debug("Baseline [%s]: revisit %s — stopping", mode, leg.to_loc)
                break

            legs.append(leg)
            visited.append(leg.to_loc)
            current = leg.to_loc

            if current == goal:
                break

        if not legs or current != goal:
            log.info("Baseline [%s]: did not reach goal", mode)
            continue

        # ── Step 3: evaluate ─────────────────────────────────────────────────
        score = slipnet.evaluate_path(
            f"baseline-{mode}", tuple(legs), activation=0.5
        )

        path_locs = (start,) + tuple(l.to_loc for l in legs)
        result = BaselineResult(
            mode=mode,
            legs=tuple(legs),
            score=score,
            path=path_locs,
        )
        run.paths.append(result)
        run.modes_complete += 1
        log.info(
            "Baseline [%s]: %d legs, score=%.2f  %s",
            mode, len(legs), score, " -> ".join(path_locs),
        )

    # ── Step 4: rank by head-to-head comparison ───────────────────────────────
    if len(run.paths) >= 2:
        n = len(run.paths)
        for _ in range(n - 1):
            for j in range(n - 1):
                a, b = run.paths[j], run.paths[j + 1]
                winner = slipnet.compare_paths(_Cell(a), _Cell(b), activation=0.5)
                if winner == "b":
                    run.paths[j], run.paths[j + 1] = b, a

    log.info(
        "Baseline done: %d/%d modes complete",
        run.modes_complete, run.modes_tried,
    )
    return run
