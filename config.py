"""
config.py — Global constants for the FARG Travel Path Finder.

Hanoi → Da Nang via a stochastic FARG cognitive architecture.
"""

import os

# ── LLM / Slipnet ─────────────────────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
MODEL_ID: str = "qwen/qwen3.5-9b"
MAX_CONCURRENT_LLM: int = 2          # semaphore cap on async LLM calls

# ── Activation dynamics ────────────────────────────────────────────────────────
DECAY_RATE: float = 0.95             # per-tick multiplicative decay
SUPPORT_WEIGHT: float = 0.10         # positive edge weight (support flows)
ANTIPATHY_WEIGHT: float = -0.20      # negative edge weight (mutual inhibition)
PROPAGATION_SCALE: float = 0.10      # fraction of edge weight that flows per tick
BOOST_FACTOR: float = 0.50           # boost adds BOOST_FACTOR * current_a
MIN_ACTIVATION: float = 0.01         # prune elements below this threshold
INIT_ACTIVATION: float = 1.0         # default activation for freshly built elements

# ── Loop control ───────────────────────────────────────────────────────────────
MAX_TICKS: int = 120                 # hard upper bound on heartbeat ticks
MIN_COMPLETE_PATHS: int = 3          # stop when at least this many paths found
ACT_EARLY_PROB: float = 0.20         # probability of acting in early ticks (< 20)
ACT_LATE_TICK: int = 20              # after this tick, always prefer act() when possible

# ── Geography ─────────────────────────────────────────────────────────────────
START: str = "Hanoi"
GOAL: str = "Da Nang"
