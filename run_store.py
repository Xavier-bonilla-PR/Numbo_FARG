"""
run_store.py — Persistent pickle storage for FARG and Baseline run results.

Each completed run is saved to ./runs/ as a pickle file.  Files are named:

    farg_YYYYMMDD_HHMMSS_Np_A.AA.pkl
    baseline_YYYYMMDD_HHMMSS_Np_A.AA.pkl

where N = paths found and A.AA = average quality score.

Record format (dict stored in each .pkl):
    _type:     "farg" | "baseline"
    _saved_at: "YYYYMMDD_HHMMSS"
    _meta:     lightweight summary dict (survives without importing project modules)
    --- FARG records also contain: mc, paths, arch_calls, farg_scores ---
    --- Baseline records also contain: _run (BaselineRun object) -------
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

RUNS_DIR = Path(__file__).parent / "runs"


def _ensure_runs_dir() -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    return RUNS_DIR


# ── Save ──────────────────────────────────────────────────────────────────────

def save_farg_run(run_data: dict) -> Path:
    """Pickle a FARG run dict to ./runs/.  Returns the saved file path."""
    _ensure_runs_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    scores = list(run_data["farg_scores"].values())
    n = len(run_data["paths"])
    avg = sum(scores) / max(1, len(scores))
    fname = f"farg_{ts}_{n}p_{avg:.2f}.pkl"
    record = {
        **run_data,
        "_type":     "farg",
        "_saved_at": ts,
        "_meta": {
            "n_paths":   n,
            "avg_score": avg,
            "n_ticks":   len(run_data["mc"].snapshots),
            "arch_total": sum(run_data["arch_calls"].values()),
        },
    }
    path = RUNS_DIR / fname
    with open(path, "wb") as f:
        pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def save_baseline_run(base_run) -> Path:
    """Pickle a BaselineRun to ./runs/.  Returns the saved file path."""
    _ensure_runs_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    scores = [r.score for r in base_run.paths]
    n = len(base_run.paths)
    avg = sum(scores) / max(1, len(scores))
    fname = f"baseline_{ts}_{n}p_{avg:.2f}.pkl"
    record = {
        "_run":      base_run,
        "_type":     "baseline",
        "_saved_at": ts,
        "_meta": {
            "n_paths":       n,
            "avg_score":     avg,
            "modes_tried":   base_run.modes_tried,
            "modes_complete": base_run.modes_complete,
            "llm_calls":     base_run.llm_calls,
        },
    }
    path = RUNS_DIR / fname
    with open(path, "wb") as f:
        pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


# ── List ──────────────────────────────────────────────────────────────────────

def list_runs() -> list[dict]:
    """Return metadata for all saved runs, newest first.

    Each dict has at minimum:
        filename, path, type, saved_at, n_paths, avg_score
    plus any extra keys from _meta.
    """
    _ensure_runs_dir()
    results = []
    for pkl in sorted(RUNS_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime,
                      reverse=True):
        try:
            with open(pkl, "rb") as f:
                record = pickle.load(f)
            meta = record.get("_meta", {})
            results.append({
                "filename": pkl.name,
                "path":     pkl,
                "type":     record["_type"],
                "saved_at": record["_saved_at"],
                **meta,
            })
        except Exception:
            pass   # skip corrupt / incompatible files
    return results


# ── Load / Delete ─────────────────────────────────────────────────────────────

def load_run(path) -> dict:
    """Load a pickled run record.  Returns the full dict."""
    with open(path, "rb") as f:
        return pickle.load(f)


def delete_run(path) -> None:
    """Delete a saved run file."""
    Path(path).unlink(missing_ok=True)
