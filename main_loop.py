"""
main_loop.py — The single stochastic heartbeat for the FARG Travel Path Finder.

One loop.  Each tick:
  1. Decay all activations          (workspace.decay_all)
  2. Propagate through edges         (workspace.propagate)
  3. Run detectors                   (detect completed paths, blocked states)
  4. Check termination condition
  5. Choose ONE agent probabilistically weighted by activation²
     — agents tagged PendingLLM are skipped but still decay
  6. Execute .go() or .act() on the chosen agent
  7. Prune dead elements

The loop terminates when MIN_COMPLETE_PATHS paths reach the goal, or after
MAX_TICKS ticks.

Design note
───────────
This is NOT a pipeline.  No external code decides which agent runs or what
it builds.  The activation dynamics are the only control structure.

Early ticks (< ACT_LATE_TICK) favour deliberation (.go) to build up a
population of competing ImCells before any commitments are made.
Later ticks shift probability toward commitment (.act).
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional

from agents import AGENT_TYPES, Evaluate, SeekEvidence
from canvas import ImCell
from config import ACT_EARLY_PROB, ACT_LATE_TICK, GOAL, MAX_TICKS, MIN_COMPLETE_PATHS
from tags import ActIsDone, Blocked, GettingCloser, GoIsDone, GoalReached, PathComplete

log = logging.getLogger(__name__)


# ── Detector helpers ──────────────────────────────────────────────────────────

_MIN_FEASIBILITY_SCORE: float = 0.35  # SeekEvidence score required before PathComplete


def _detect_complete_paths(ws) -> None:
    """Tag an ImCell at GOAL as PathComplete only after SeekEvidence stamps it feasible.

    If the ImCell has not yet been evaluated, a SeekEvidence agent is spawned
    and the decision is deferred to the next tick(s).  Paths scoring below
    _MIN_FEASIBILITY_SCORE are never marked complete, preventing degenerate
    hallucinated routes (e.g. wrong-direction island hops) from polluting results.
    """
    for elem in list(ws.elements.keys()):
        if not isinstance(elem, ImCell):
            continue
        if elem.current_loc != GOAL:
            continue
        if ws.has_tag(elem, PathComplete):
            continue

        # Require a quality evaluation before declaring complete.
        gc_tags = [t for t in ws.tags_on(elem) if isinstance(t, GettingCloser)]
        if not gc_tags:
            # Not yet evaluated — spawn SeekEvidence if none already pending.
            already = any(
                isinstance(e, SeekEvidence) and e.imcell == elem
                for e in ws.elements
            )
            if not already:
                ws.add(SeekEvidence(path_id=elem.path_id, imcell=elem))
            continue

        best_score = max(t.weight for t in gc_tags)
        if best_score < _MIN_FEASIBILITY_SCORE:
            log.debug(
                "_detect_complete_paths: %s scored %.2f < %.2f — not marking complete",
                elem.path_id, best_score, _MIN_FEASIBILITY_SCORE,
            )
            continue

        if elem.legs:
            path_locs = (elem.legs[0].from_loc,) + tuple(l.to_loc for l in elem.legs)
        else:
            path_locs = (elem.current_loc,)
        tag = PathComplete(taggee=elem, path=path_locs)
        ws.tag(tag)
        ws.boost(elem)
        log.info("Path complete (score=%.2f): %s", best_score, " -> ".join(path_locs))


def _detect_blocked_agents(ws) -> None:
    """Mark agents that have nothing left to do as Blocked."""
    for elem in list(ws.elements.keys()):
        if not isinstance(elem, AGENT_TYPES):
            continue
        if ws.has_tag(elem, Blocked):
            continue
        # Agent is exhausted if it's ActIsDone and cannot go
        if ws.has_tag(elem, ActIsDone):
            can_still_go = hasattr(elem, "can_go") and elem.can_go(ws)
            if not can_still_go:
                ws.tag(Blocked(taggee=elem, reason="ActIsDone and cannot go"))


def _spawn_evaluate_agents(ws) -> None:
    """After several ImCells exist, spawn Evaluate agents to compare pairs.

    An Evaluate(a, b) is only created once per ordered pair of path IDs.
    """
    existing_evals = {
        (e.path_id_a, e.path_id_b)
        for e in ws.elements
        if isinstance(e, Evaluate)
    }
    imcells = ws.all_imcells()
    path_ids = list({c.path_id for c in imcells})

    for i in range(len(path_ids)):
        for j in range(i + 1, len(path_ids)):
            pa, pb = path_ids[i], path_ids[j]
            if (pa, pb) not in existing_evals and (pb, pa) not in existing_evals:
                ev = Evaluate(path_id_a=pa, path_id_b=pb)
                ws.add(ev)
                existing_evals.add((pa, pb))


def run_detectors(ws) -> None:
    """Run all detectors once per tick."""
    _detect_complete_paths(ws)
    _detect_blocked_agents(ws)
    # Only spawn evaluators when we have at least 2 distinct ImCells
    if len(ws.all_imcells()) >= 2:
        _spawn_evaluate_agents(ws)


# ── Complete-path query ───────────────────────────────────────────────────────

def get_complete_paths(ws) -> List[GoalReached]:
    """Return all per-chain GoalReached tags (one per canvas chain at GOAL).

    Uses GoalReached instead of PathComplete so every committed canvas chain
    that finishes is declared — not just the first MIN_COMPLETE_PATHS.
    """
    return [t for t in ws.tags if isinstance(t, GoalReached)]


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_loop(ws, slipnet, logger=None, max_ticks: int = MAX_TICKS) -> List[GoalReached]:
    """Run the stochastic heartbeat.

    Returns a list of GoalReached tags (one per committed canvas chain that
    reached the goal).  PathComplete tags (ImCell-level) are a separate
    population used internally by detectors; they are not returned here.
    """
    complete: List[GoalReached] = []

    for tick in range(max_ticks):

        # ── 1. Decay ──────────────────────────────────────────────────────
        ws.decay_all()

        # ── 2. Propagate ──────────────────────────────────────────────────
        ws.propagate()

        # ── 3. Detectors ──────────────────────────────────────────────────
        run_detectors(ws)

        # ── 4. Check termination ──────────────────────────────────────────
        complete = get_complete_paths(ws)
        if len(complete) >= MIN_COMPLETE_PATHS:
            log.info("Tick %d: found %d paths, stopping.", tick, len(complete))
            break

        # ── 5. Collect eligible agents ────────────────────────────────────
        available = ws.agents_not_pending()
        if not available:
            log.debug("Tick %d: no eligible agents.", tick)
            if logger:
                logger.log_tick(tick, ws, chosen=None, complete=complete)
            ws.prune()
            continue

        # ── 6. Choose ONE agent weighted by activation² ───────────────────
        chosen = ws.choose_agent(available)
        if chosen is None:
            continue

        # ── 7. go() or act()? ─────────────────────────────────────────────
        # Early ticks strongly prefer go() to build up a rich ImCell population.
        # Later ticks allow act() to commit promising legs.
        acted = False
        can_a = hasattr(chosen, "can_act") and chosen.can_act(ws)
        can_g = hasattr(chosen, "can_go") and chosen.can_go(ws)

        if can_a:
            if tick >= ACT_LATE_TICK or random.random() < ACT_EARLY_PROB:
                try:
                    chosen.act(ws, slipnet)
                except Exception as exc:
                    log.warning("agent.act() error: %s – %s", chosen, exc)
                ws.downboost(chosen)
                acted = True

        if not acted and can_g:
            try:
                chosen.go(ws, slipnet)
            except Exception as exc:
                log.warning("agent.go() error: %s – %s", chosen, exc)

        # ── 8. Log ────────────────────────────────────────────────────────
        if logger:
            logger.log_tick(tick, ws, chosen=chosen, complete=complete)

        # ── 9. Prune ──────────────────────────────────────────────────────
        ws.prune()

    # Final pass
    complete = get_complete_paths(ws)
    return complete
