"""
baseline.py — Fixed sequential pipeline using the same LLM + same prompts as FARG,
but with zero cognitive architecture.

No activation, no competition, no antipathy, no stochastic selection.
Just:
  1. query_modes          → get transport modes
  2. query_route (loop)   → greedily extend each path until Da Nang or dead end
  3. evaluate_path        → score each complete path
  4. compare_paths        → rank pairs

All LLM calls use a fixed temperature of 0.5 (the midpoint of FARG's range).

Run:
    python baseline.py            # uses MockSlipnet
    python baseline.py --real     # uses RealSlipnet (needs OPENROUTER_API_KEY)
    python baseline.py --real --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from canvas import Leg
from config import GOAL, START

log = logging.getLogger(__name__)

FIXED_TEMPERATURE = 0.5   # no activation → no dynamic temperature
MAX_STEPS_PER_PATH = 8    # hard cap on greedy hops per mode


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class BaselinePath:
    mode: str
    legs: List[Leg]
    score: float = 0.0
    score_reason: str = ""

    @property
    def complete(self) -> bool:
        return bool(self.legs) and self.legs[-1].to_loc == GOAL

    @property
    def stops(self) -> List[str]:
        if not self.legs:
            return [START]
        return [self.legs[0].from_loc] + [l.to_loc for l in self.legs]

    @property
    def total_hours(self) -> float:
        return sum(l.duration_hours for l in self.legs)

    def route_str(self) -> str:
        return " -> ".join(self.stops)


@dataclass
class BaselineResult:
    modes: List[str]
    paths: List[BaselinePath]
    rankings: List[Tuple[str, str]]   # [(winner_mode, loser_mode), ...]
    elapsed_s: float
    n_llm_calls: int


# ── Pipeline ──────────────────────────────────────────────────────────────────

class BaselinePipeline:
    """Sequential LLM pipeline — same prompts as FARG, no dynamics."""

    def __init__(self, slipnet) -> None:
        self.slipnet = slipnet
        self.n_calls = 0

    def _query_modes(self) -> List[str]:
        self.n_calls += 1
        return self.slipnet.query_modes(START, GOAL)

    def _query_route(self, from_loc: str, visited: List[str]) -> Optional[Leg]:
        """One greedy step: ask LLM for the next hop.  Fixed temperature."""
        self.n_calls += 1
        # Patch: override the slipnet's dynamic temperature by temporarily
        # calling it at activation=0.57 which maps to T≈0.5 via the formula.
        # For RealSlipnet this is just activation=0.57; MockSlipnet ignores it.
        data = self.slipnet.query_route(
            from_loc, GOAL, self._current_mode,
            activation=0.57,          # T = 0.9 - 0.7*0.57 ≈ 0.5
            visited_locs=visited,
        )
        if data is None:
            return None
        try:
            return Leg(
                from_loc=str(data["from_loc"]),
                to_loc=str(data["to_loc"]),
                mode=str(data.get("mode", self._current_mode)),
                duration_hours=float(data.get("duration_hours", 0.0)),
                notes=str(data.get("notes", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("bad leg_data %s: %s", data, exc)
            return None

    def _build_path(self, mode: str) -> BaselinePath:
        """Greedily extend a single-mode path until goal or dead end."""
        self._current_mode = mode
        legs: List[Leg] = []
        current = START
        visited = [START]

        for step in range(MAX_STEPS_PER_PATH):
            leg = self._query_route(current, visited)
            if leg is None:
                log.debug("baseline: %s step %d returned None", mode, step)
                break

            # Reject revisits (same check as FARG)
            if leg.to_loc in visited and leg.to_loc != GOAL:
                log.debug("baseline: %s step %d revisit %s — stopping", mode, step, leg.to_loc)
                break

            legs.append(leg)
            visited.append(leg.to_loc)
            current = leg.to_loc

            if current == GOAL:
                break

        return BaselinePath(mode=mode, legs=legs)

    def _evaluate(self, path: BaselinePath) -> None:
        self.n_calls += 1
        try:
            raw = self.slipnet.evaluate_path(
                f"baseline-{path.mode}", path.legs, activation=0.57
            )
            path.score = float(raw)
        except Exception as exc:
            log.warning("evaluate failed: %s", exc)
            path.score = 0.5

    def _compare(self, a: BaselinePath, b: BaselinePath) -> str:
        """Return 'a' or 'b' — which mode's path is better?"""
        self.n_calls += 1

        class _FakeCel:
            def __init__(self, p):
                self.legs = p.legs
                self.current_loc = p.legs[-1].to_loc if p.legs else START

        winner = self.slipnet.compare_paths(_FakeCel(a), _FakeCel(b), activation=0.57)
        return winner

    def run(self) -> BaselineResult:
        t0 = time.monotonic()

        # Step 1 — modes
        modes = self._query_modes()
        print(f"[1] Modes returned: {modes}")

        # Step 2 — greedy path per mode
        paths: List[BaselinePath] = []
        for mode in modes:
            print(f"[2] Building path for mode: {mode} ...", end=" ", flush=True)
            path = self._build_path(mode)
            print(f"{'COMPLETE' if path.complete else 'INCOMPLETE'} -> {path.route_str()}")
            paths.append(path)

        complete = [p for p in paths if p.complete]

        # Step 3 — evaluate each complete path
        for path in complete:
            print(f"[3] Evaluating {path.mode} ...", end=" ", flush=True)
            self._evaluate(path)
            print(f"score={path.score:.2f}")

        # Step 4 — compare pairs (round-robin)
        rankings: List[Tuple[str, str]] = []
        for i in range(len(complete)):
            for j in range(i + 1, len(complete)):
                a, b = complete[i], complete[j]
                print(f"[4] Comparing {a.mode} vs {b.mode} ...", end=" ", flush=True)
                winner_key = self._compare(a, b)
                winner = a if winner_key == "a" else b
                loser  = b if winner_key == "a" else a
                rankings.append((winner.mode, loser.mode))
                print(f"winner={winner.mode}")

        elapsed = time.monotonic() - t0
        return BaselineResult(
            modes=modes,
            paths=paths,
            rankings=rankings,
            elapsed_s=elapsed,
            n_llm_calls=self.n_calls,
        )


# ── Comparison metrics ────────────────────────────────────────────────────────

def compute_metrics(result: BaselineResult) -> dict:
    complete = [p for p in result.paths if p.complete]
    all_stops = [stop for p in complete for stop in p.stops[1:-1]]  # intermediates
    unique_stops = set(all_stops)

    # Shared legs: legs that appear in >1 path
    leg_keys = [f"{l.from_loc}->{l.to_loc}" for p in complete for l in p.legs]
    from collections import Counter
    leg_counts = Counter(leg_keys)
    shared_legs = sum(1 for c in leg_counts.values() if c > 1)

    scores = [p.score for p in complete]
    import statistics
    return {
        "n_paths_complete":      len(complete),
        "n_modes_tried":         len(result.paths),
        "n_unique_intermediate": len(unique_stops),
        "unique_stops":          sorted(unique_stops),
        "shared_legs":           shared_legs,
        "mean_score":            round(statistics.mean(scores), 3) if scores else 0.0,
        "score_stdev":           round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0,
        "mean_legs":             round(statistics.mean(len(p.legs) for p in complete), 1) if complete else 0,
        "total_llm_calls":       result.n_llm_calls,
        "elapsed_s":             round(result.elapsed_s, 1),
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(result: BaselineResult) -> None:
    metrics = compute_metrics(result)
    complete = [p for p in result.paths if p.complete]
    complete.sort(key=lambda p: -p.score)

    sep = "-" * 60
    print(f"\n{sep}")
    print("BASELINE PIPELINE — RESULTS")
    print(sep)

    print(f"\nComplete paths ({metrics['n_paths_complete']}/{metrics['n_modes_tried']} modes reached {GOAL}):")
    for rank, p in enumerate(complete, 1):
        print(f"  #{rank}  [{p.mode}]  score={p.score:.2f}  {p.total_hours:.1f}h")
        print(f"       {p.route_str()}")
        for leg in p.legs:
            print(f"         {leg.from_loc} --[{leg.mode}]--> {leg.to_loc} ({leg.duration_hours:.1f}h)")
            if leg.notes:
                print(f"           {leg.notes}")

    print(f"\nPair comparisons:")
    for winner, loser in result.rankings:
        print(f"  {winner}  >  {loser}")

    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:30s} {v}")

    print(f"\n{sep}")
    print("WHAT TO COMPARE AGAINST FARG")
    print(sep)
    print(f"""
  n_unique_intermediate  : {metrics['n_unique_intermediate']}  stops
    FARG advantage: activation competition forces agents to explore
    distinct detours; baseline greedily re-converges on the same cities.

  shared_legs            : {metrics['shared_legs']}  legs appear in >1 path
    FARG advantage: antipathy edges penalise paths that copy each other;
    baseline has no such pressure.

  score_stdev            : {metrics['score_stdev']}
    FARG advantage: SeekEvidence + Compare create score spread.
    Baseline evaluates independently with no cross-path pressure.

  total_llm_calls        : {metrics['total_llm_calls']}  (baseline)
    Compare with FARG tick log to see overhead of dynamics.
""")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline sequential LLM pipeline")
    ap.add_argument("--real",    action="store_true", help="Use RealSlipnet (needs API key)")
    ap.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from travel_slipnet import MockSlipnet, RealSlipnet
    use_real = args.real and bool(os.getenv("OPENROUTER_API_KEY"))
    slipnet = RealSlipnet() if use_real else MockSlipnet()
    kind = "RealSlipnet" if use_real else "MockSlipnet"

    print(f"\nBaseline Pipeline: {START} -> {GOAL}  [{kind}]")
    print(f"Fixed temperature: {FIXED_TEMPERATURE}  (no activation dynamics)\n")

    pipeline = BaselinePipeline(slipnet)
    result = pipeline.run()
    print_report(result)


if __name__ == "__main__":
    main()
