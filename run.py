"""
run.py — Entry point for the FARG Travel Path Finder.

Usage
─────
  # With MockSlipnet (no API key required):
  python run.py

  # With real LLM (requires OpenRouter API key):
  OPENROUTER_API_KEY=sk-or-... python run.py

  # Verbose / quiet:
  python run.py --verbose
  python run.py --quiet

  # Override tick count:
  python run.py --ticks 80

  # Force mock even if key is set:
  python run.py --mock
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="FARG Travel Path Finder: Hanoi → Da Nang")
    p.add_argument("--ticks",   type=int,  default=None, help="Max heartbeat ticks")
    p.add_argument("--quiet",   action="store_true",     help="Suppress tick log")
    p.add_argument("--verbose", action="store_true",     help="Enable DEBUG logging")
    p.add_argument("--mock",    action="store_true",     help="Force MockSlipnet")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Logging ───────────────────────────────────────────────────────────────
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ── Imports (after logging config) ────────────────────────────────────────
    from agents import Want
    from config import GOAL, MAX_TICKS, START
    from logger import TickLogger
    from main_loop import run_loop
    from slipnet import MockSlipnet, RealSlipnet
    from workspace import Workspace

    # ── Select slipnet ────────────────────────────────────────────────────────
    use_real = bool(os.getenv("OPENROUTER_API_KEY")) and not args.mock
    if use_real:
        slipnet = RealSlipnet()
        slipnet_name = f"RealSlipnet (model={__import__('config').MODEL_ID})"
    else:
        slipnet = MockSlipnet()
        slipnet_name = "MockSlipnet (deterministic)"

    max_ticks = args.ticks if args.ticks is not None else MAX_TICKS
    verbose = not args.quiet

    print(f"\nFARG Travel Path Finder")
    print(f"  Route  : {START} → {GOAL}")
    print(f"  Slipnet: {slipnet_name}")
    print(f"  MaxTick: {max_ticks}")
    print()

    # ── Initialise workspace ──────────────────────────────────────────────────
    ws = Workspace()
    want = Want(from_loc=START, to_loc=GOAL)
    ws.add(want, init_a=1.0)

    # ── Run ───────────────────────────────────────────────────────────────────
    tick_logger = TickLogger(verbose=verbose)
    paths = run_loop(ws, slipnet, logger=tick_logger, max_ticks=max_ticks)

    # ── Final report ──────────────────────────────────────────────────────────
    tick_logger.log_final(paths)

    if not paths:
        print("No complete paths found.  Try --ticks 100 or check slipnet output.")
        sys.exit(1)


if __name__ == "__main__":
    main()
