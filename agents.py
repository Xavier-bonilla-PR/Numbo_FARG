"""
agents.py — Agent dataclasses for the FARG Travel Path Finder.

All agents are frozen dataclasses (hashable workspace elements).  Each defines:

  can_go(ws)   → bool   — may this agent deliberate right now?
  go(ws, sl)            — deliberation: explore hypothetical states (ImCells)
  can_act(ws)  → bool   — may this agent commit right now?
  act(ws, sl)           — commitment: write to canvas, spawn successors

Mutual antipathy is declared via has_antipathy_to(other) → bool.

Agent lifecycle
───────────────
Want
  └─ go() → builds SuggestMode × N (one per transport mode)

SuggestMode
  └─ go() → [PendingLLM] → query slipnet → build SuggestRoute → [GoIsDone]

SuggestRoute
  ├─ go()  → build/extend ImCell → [GoIsDone]
  └─ act() → commit CanvasCell, spawn next SuggestMode (if not at goal)

SeekEvidence
  └─ go() → evaluate ImCell, tag GettingCloser, boost/downboost

Evaluate
  └─ go() → compare two ImCells, boost winner, downboost loser
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from canvas import ImCell, Leg
from tags import ActIsDone, GoIsDone, PendingLLM

log = logging.getLogger(__name__)


# ── Exported tuple used by workspace.py to identify agents ────────────────────

# (populated at module bottom after all classes are defined)
AGENT_TYPES: tuple = ()


# ── Want ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Want:
    """Initial desire: find a path from from_loc to to_loc.

    go():
      Query the slipnet for available transport modes.
      Build one SuggestMode per mode (they will compete via antipathy).
    """

    from_loc: str
    to_loc: str

    def can_go(self, ws) -> bool:
        return not ws.has_tag(self, GoIsDone)

    def can_act(self, ws) -> bool:
        return False

    def go(self, ws, slipnet) -> None:
        modes = slipnet.query_modes(self.from_loc, self.to_loc)
        log.debug("Want.go: modes=%s", modes)
        for mode in modes:
            pid = ws.new_path_id()
            sm = SuggestMode(
                path_id=pid,
                current_loc=self.from_loc,
                goal=self.to_loc,
                mode=mode,
            )
            ws.add(sm, builder=self)
        ws.tag(GoIsDone(taggee=self))

    def has_antipathy_to(self, other) -> bool:
        return False


# ── SuggestMode ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SuggestMode:
    """Suggest a transport mode for the next leg from current_loc toward goal.

    go():
      1. Tag self PendingLLM.
      2. Query slipnet for a concrete leg using self.mode.
      3. Untag PendingLLM.
      4. Build SuggestRoute with the returned leg.
      5. Tag self GoIsDone.

    Antipathy: SuggestMode agents at the SAME current_loc with DIFFERENT
    modes (and different path IDs) compete with each other.
    """

    path_id: str
    current_loc: str
    goal: str
    mode: str

    def can_go(self, ws) -> bool:
        return (
            not ws.has_tag(self, PendingLLM)
            and not ws.has_tag(self, GoIsDone)
        )

    def can_act(self, ws) -> bool:
        return False

    def go(self, ws, slipnet) -> None:
        ws.tag(PendingLLM(taggee=self))
        activation = ws.activation(self)
        log.debug(
            "SuggestMode.go: path=%s loc=%s mode=%s a=%.3f",
            self.path_id, self.current_loc, self.mode, activation,
        )

        # Build visited-location list from committed canvas so the LLM
        # does not suggest places the traveller has already been.
        canvas_cells = sorted(ws.canvas_legs(self.path_id), key=lambda c: c.position)
        if canvas_cells:
            visited = [canvas_cells[0].leg.from_loc] + [c.leg.to_loc for c in canvas_cells]
        else:
            visited = []

        leg_data = slipnet.query_route(
            self.current_loc, self.goal, self.mode, activation,
            visited_locs=visited,
            depth=len(canvas_cells),
        )

        ws.untag(self, PendingLLM)

        if leg_data is None:
            log.debug("SuggestMode.go: no route returned, staying alive")
            return

        try:
            leg = Leg(
                from_loc=str(leg_data["from_loc"]),
                to_loc=str(leg_data["to_loc"]),
                mode=str(leg_data.get("mode", self.mode)),
                duration_hours=float(leg_data.get("duration_hours") or 0.0),
                notes=str(leg_data.get("notes", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("SuggestMode.go: bad leg_data %s: %s", leg_data, exc)
            return

        sr = SuggestRoute(
            path_id=self.path_id,
            current_loc=self.current_loc,
            goal=self.goal,
            mode=self.mode,
            proposed_leg=leg,
        )
        ws.add(sr, builder=self)
        ws.tag(GoIsDone(taggee=self))

    def has_antipathy_to(self, other) -> bool:
        return (
            isinstance(other, SuggestMode)
            and other.current_loc == self.current_loc
            and other.mode != self.mode
            and other.path_id != self.path_id
        )


# ── SuggestRoute ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SuggestRoute:
    """Propose a specific route leg; deliberate into ImCell then commit.

    go():
      Build or extend the ImCell for this path with the proposed leg.
      Tag self GoIsDone.

    act():
      Write the leg to the canvas (CanvasCell).
      If not yet at goal, spawn a new SuggestMode from the new location.
      Tag self ActIsDone.

    Antipathy: two SuggestRoute agents for the SAME path at the SAME
    current_loc but with DIFFERENT modes compete.
    """

    path_id: str
    current_loc: str
    goal: str
    mode: str
    proposed_leg: Leg

    def can_go(self, ws) -> bool:
        return not ws.has_tag(self, GoIsDone)

    def can_act(self, ws) -> bool:
        return (
            ws.has_tag(self, GoIsDone)
            and not ws.has_tag(self, ActIsDone)
        )

    def go(self, ws, slipnet) -> None:
        log.debug(
            "SuggestRoute.go: path=%s %s", self.path_id, self.proposed_leg
        )
        existing = ws.imcell_for(self.path_id)

        if existing is None:
            # Reconstruct ImCell from committed canvas legs so the route
            # history isn't lost if the earlier ImCell was pruned.
            canvas_cells = sorted(
                ws.canvas_legs(self.path_id), key=lambda c: c.position
            )
            if canvas_cells:
                prior_legs = tuple(c.leg for c in canvas_cells)
                existing = ImCell(
                    path_id=self.path_id,
                    legs=prior_legs,
                    current_loc=prior_legs[-1].to_loc,
                )

        # Guard: if the path is already at or past the goal, nothing to do.
        if existing is not None and existing.current_loc == self.goal:
            ws.tag(GoIsDone(taggee=self))
            return

        # Guard: proposed leg must connect to where we currently are.
        # FARG fix: if from_loc is in the path history, branch instead of skipping.
        if existing is not None and existing.current_loc != self.proposed_leg.from_loc:
            seq = existing.location_sequence()
            if self.proposed_leg.from_loc in seq:
                cut = seq.index(self.proposed_leg.from_loc)
                branch_legs = existing.legs[:cut]
                new_pid = ws.new_path_id()
                branch_visited = set(seq[:cut + 1])
                # Only branch if the proposed destination is novel (no revisit)
                if (self.proposed_leg.to_loc not in branch_visited
                        or self.proposed_leg.to_loc == self.goal):
                    new_imcell = ImCell(
                        path_id=new_pid,
                        legs=branch_legs + (self.proposed_leg,),
                        current_loc=self.proposed_leg.to_loc,
                    )
                    # Commit historical legs + proposed leg to canvas for new_pid
                    # so future reconstruction from canvas works correctly.
                    for bl in branch_legs:
                        ws.commit_leg(new_pid, bl)
                    ws.commit_leg(new_pid, self.proposed_leg)
                    ws.add(new_imcell, builder=self)
                    ws.add(SeekEvidence(path_id=new_pid, imcell=new_imcell), builder=self)
                    if self.proposed_leg.to_loc != self.goal:
                        ws.add(
                            SuggestMode(
                                path_id=new_pid,
                                current_loc=self.proposed_leg.to_loc,
                                goal=self.goal,
                                mode=self.mode,
                            ),
                            builder=self,
                            init_a=0.5,
                        )
                    log.debug(
                        "SuggestRoute.go: branched %s -> %s at %s -> %s",
                        self.path_id, new_pid,
                        self.proposed_leg.from_loc, self.proposed_leg.to_loc,
                    )
            else:
                log.debug(
                    "SuggestRoute.go: leg mismatch (at %s, leg starts from %s) — skipping",
                    existing.current_loc, self.proposed_leg.from_loc,
                )
            # Either branched or irreconcilable — never commit wrong leg to wrong path.
            ws.tag(GoIsDone(taggee=self))
            ws.tag(ActIsDone(taggee=self))
            return

        # Guard: prevent revisiting a location already in the path.
        if existing is not None:
            visited = {l.from_loc for l in existing.legs} | {l.to_loc for l in existing.legs}
            if self.proposed_leg.to_loc in visited and self.proposed_leg.to_loc != self.goal:
                log.debug(
                    "SuggestRoute.go: proposed leg revisits %s — skipping",
                    self.proposed_leg.to_loc,
                )
                ws.tag(GoIsDone(taggee=self))
                return

        if existing is not None:
            new_imcell = existing.with_leg(self.proposed_leg)
        else:
            new_imcell = ImCell(
                path_id=self.path_id,
                legs=(self.proposed_leg,),
                current_loc=self.proposed_leg.to_loc,
            )

        # Final sanity: reject any ImCell whose location sequence has
        # internal cycles (same location visited twice before the goal).
        seq = new_imcell.location_sequence()
        non_goal = seq[:-1]  # all except last stop
        if len(non_goal) != len(set(non_goal)):
            log.debug("SuggestRoute.go: cycle detected in %s — skipping", seq)
            ws.tag(GoIsDone(taggee=self))
            return

        ws.add(new_imcell, builder=self)
        ws.tag(GoIsDone(taggee=self))

        # Also spawn a SeekEvidence agent to evaluate the new ImCell
        se = SeekEvidence(path_id=self.path_id, imcell=new_imcell)
        ws.add(se, builder=self)

    def act(self, ws, slipnet) -> None:
        log.debug(
            "SuggestRoute.act: path=%s committing %s", self.path_id, self.proposed_leg
        )
        ws.commit_leg(self.path_id, self.proposed_leg)
        ws.tag(ActIsDone(taggee=self))
        ws.downboost(self)

        if self.proposed_leg.to_loc == self.goal:
            # Declare this canvas chain complete — one GoalReached tag per chain,
            # so all paths reaching the goal are captured (no single-halt stop).
            from tags import GoalReached
            canvas_cells = sorted(ws.canvas_legs(self.path_id), key=lambda c: c.position)
            chain_legs = tuple(c.leg for c in canvas_cells)
            path_locs = (canvas_cells[0].leg.from_loc,) + tuple(c.leg.to_loc for c in canvas_cells)
            ws.tag(GoalReached(path_id=self.path_id, path=path_locs, legs=chain_legs))
            log.info(
                "GoalReached: %s  %s",
                self.path_id, " -> ".join(path_locs),
            )
        else:
            next_sm = SuggestMode(
                path_id=self.path_id,
                current_loc=self.proposed_leg.to_loc,
                goal=self.goal,
                mode=self.mode,
            )
            # Give continuation agents a healthy starting activation so they
            # can compete even after their builder has been downboosted.
            ws.add(next_sm, builder=self, init_a=0.5)

    def has_antipathy_to(self, other) -> bool:
        return (
            isinstance(other, SuggestRoute)
            and other.path_id == self.path_id
            and other.current_loc == self.current_loc
            and other.mode != self.mode
        )


# ── SeekEvidence ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SeekEvidence:
    """Evaluate how promising an ImCell is; tag it and adjust its activation.

    go():
      Query the slipnet to score the ImCell's route quality.
      Tag ImCell with GettingCloser(weight=score).
      Boost if score > 0.5, downboost otherwise.
    """

    path_id: str
    imcell: ImCell

    def can_go(self, ws) -> bool:
        return not ws.has_tag(self, GoIsDone)

    def can_act(self, ws) -> bool:
        return False

    def go(self, ws, slipnet) -> None:
        from tags import GettingCloser

        if self.imcell not in ws.elements:
            ws.tag(GoIsDone(taggee=self))
            return

        activation = ws.activation(self)
        score = slipnet.evaluate_path(
            self.path_id, self.imcell.legs, activation
        )
        weight = max(0.0, min(1.0, float(score)))

        # Chain coherence accumulation: blend this leg's score with the
        # predecessor ImCell's GettingCloser score so the weight compounds
        # across legs rather than resetting per ImCell.  This stabilises
        # act_prob and prevents shallow chains from extending indefinitely
        # on lucky high single-leg scores.
        if len(self.imcell.legs) > 1:
            pred_leg_count = len(self.imcell.legs) - 1
            pred_score = None
            for t in ws.tags:
                if (
                    isinstance(t, GettingCloser)
                    and isinstance(t.taggee, ImCell)
                    and t.taggee.path_id == self.path_id
                    and len(t.taggee.legs) == pred_leg_count
                ):
                    pred_score = t.weight
                    break
            if pred_score is not None:
                # 60 % own evaluation + 40 % predecessor's accumulated score
                weight = 0.6 * weight + 0.4 * pred_score

        log.debug(
            "SeekEvidence.go: path=%s score=%.3f (legs=%d)",
            self.path_id, weight, len(self.imcell.legs),
        )

        ws.tag(GettingCloser(taggee=self.imcell, weight=weight))

        if weight > 0.5:
            ws.boost(self.imcell)
        else:
            ws.downboost(self.imcell)

        ws.tag(GoIsDone(taggee=self))

    def has_antipathy_to(self, other) -> bool:
        return False


# ── Evaluate ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Evaluate:
    """Compare two competing ImCells; reinforce the winner.

    go():
      Retrieve both ImCells from the workspace.
      Ask slipnet which is better.
      Boost the winner, downboost the loser.
    """

    path_id_a: str
    path_id_b: str

    def can_go(self, ws) -> bool:
        return not ws.has_tag(self, GoIsDone)

    def can_act(self, ws) -> bool:
        return False

    def go(self, ws, slipnet) -> None:
        imcell_a = ws.imcell_for(self.path_id_a)
        imcell_b = ws.imcell_for(self.path_id_b)

        if imcell_a is None or imcell_b is None:
            ws.tag(GoIsDone(taggee=self))
            return

        activation = ws.activation(self)
        winner = slipnet.compare_paths(imcell_a, imcell_b, activation)

        log.debug(
            "Evaluate.go: %s vs %s → winner=%s",
            self.path_id_a, self.path_id_b, winner,
        )

        if winner == "a":
            ws.boost(imcell_a)
            ws.downboost(imcell_b)
        else:
            ws.boost(imcell_b)
            ws.downboost(imcell_a)

        ws.tag(GoIsDone(taggee=self))

    def has_antipathy_to(self, other) -> bool:
        return False


# ── AGENT_TYPES tuple (used by workspace.py) ──────────────────────────────────

AGENT_TYPES = (Want, SuggestMode, SuggestRoute, SeekEvidence, Evaluate)
