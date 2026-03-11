"""
slipnet.py — LLM-backed associative memory for the FARG Travel Path Finder.

Two implementations are provided:

RealSlipnet
    Calls qwen/qwen3-14b via the OpenRouter API using async httpx.
    • A module-level asyncio.Semaphore caps concurrent LLM calls at
      MAX_CONCURRENT_LLM (default 2).
    • Responses are constrained to JSON (no markdown).
    • Temperature is scaled by the calling agent's activation:
        high activation (1.0) → low temperature (0.2)  — reliable/factual
        low  activation (0.0) → high temperature (0.9) — exploratory/creative

MockSlipnet
    Deterministic mock used for unit-testing activation dynamics without
    incurring LLM latency or API cost.  Hard-codes a rich set of Vietnam
    routes so that multiple competing paths emerge naturally.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config import MAX_CONCURRENT_LLM, MODEL_ID, OPENROUTER_API_KEY

log = logging.getLogger(__name__)

# ── Semaphore (module-level so it is shared across all RealSlipnet instances) ──
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)


# ── Temperature helper ────────────────────────────────────────────────────────

def temperature_for_activation(activation: float) -> float:
    """Map agent activation [0.0, 1.0] → LLM temperature [0.9, 0.2].

    High-activation agents are "confident" → low temperature (precise).
    Low-activation agents are "exploring" → high temperature (creative).
    """
    clamped = min(1.0, max(0.0, activation))
    return round(0.9 - 0.7 * clamped, 3)


# ── Async LLM call ────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


async def _async_query(prompt: str, temperature: float) -> Dict[str, Any]:
    """Send a single chat-completion request; return parsed JSON dict."""
    # /no_think suppresses Qwen3 chain-of-thought tokens
    full_prompt = prompt + " /no_think"
    async with _semaphore:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/farg-travel",
                    "X-Title": "FARG Travel Path Finder",
                },
                json={
                    "model": MODEL_ID,
                    "messages": [{"role": "user", "content": full_prompt}],
                    "temperature": temperature,
                },
            )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    # Strip any residual <think>…</think> blocks before parsing
    raw = _THINK_RE.sub("", raw).strip()
    return json.loads(raw)


def _run_async(coro) -> Any:
    """Run an async coroutine synchronously, compatible with both fresh and
    existing event loops (avoids 'event loop already running' errors)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an already-running loop (e.g., Jupyter / nested asyncio)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


# ── Real Slipnet ──────────────────────────────────────────────────────────────

class RealSlipnet:
    """Queries qwen/qwen3-14b via OpenRouter for travel suggestions."""

    def query_modes(self, from_loc: str, to_loc: str) -> List[str]:
        prompt = (
            f"You are a Vietnam travel expert. "
            f"Return JSON ONLY (no markdown). "
            f"List all practical transport modes from {from_loc} to {to_loc}. "
            f'Format: {{"modes": ["flight", "train", "bus"]}}'
        )
        try:
            result = _run_async(_async_query(prompt, 0.5))
            modes = result.get("modes", [])
            if isinstance(modes, list) and modes:
                return modes
        except Exception as exc:
            log.warning("query_modes failed: %s", exc)
        return ["flight", "train", "bus"]

    def query_route(
        self,
        from_loc: str,
        to_loc: str,
        mode: str,
        activation: float = 0.5,
    ) -> Optional[Dict[str, Any]]:
        temp = temperature_for_activation(activation)
        prompt = (
            f"You are a Vietnam travel expert. "
            f"Return JSON ONLY (no markdown). "
            f"Suggest ONE travel leg from {from_loc} toward {to_loc} in Vietnam "
            f"using {mode} transport. "
            f"You may suggest an interesting intermediate stop (scenic, cultural, "
            f"non-obvious) rather than going directly to {to_loc}. "
            f"Be creative at low confidence levels. "
            f'Format: {{"from_loc": "...", "to_loc": "...", "mode": "{mode}", '
            f'"duration_hours": 0.0, "notes": "..."}}'
        )
        try:
            result = _run_async(_async_query(prompt, temp))
            if "from_loc" in result and "to_loc" in result:
                return result
        except Exception as exc:
            log.warning("query_route failed: %s", exc)
        return None

    def evaluate_path(
        self,
        path_id: str,
        legs: Any,
        activation: float = 0.5,
    ) -> float:
        temp = temperature_for_activation(activation)
        legs_desc = " → ".join(f"{l.from_loc}→{l.to_loc}({l.mode})" for l in legs)
        prompt = (
            f"You are a Vietnam travel expert. "
            f"Return JSON ONLY (no markdown). "
            f"Rate this Vietnam travel route on a scale 0.0 to 1.0 considering "
            f"scenic value, cultural richness, and practical feasibility: {legs_desc}. "
            f'Format: {{"score": 0.75, "reason": "..."}}'
        )
        try:
            result = _run_async(_async_query(prompt, temp))
            return float(result.get("score", 0.5))
        except Exception as exc:
            log.warning("evaluate_path failed: %s", exc)
        return 0.5

    def compare_paths(
        self,
        imcell_a: Any,
        imcell_b: Any,
        activation: float = 0.5,
    ) -> str:
        temp = temperature_for_activation(activation)
        desc_a = " → ".join(l.to_loc for l in imcell_a.legs) or imcell_a.current_loc
        desc_b = " → ".join(l.to_loc for l in imcell_b.legs) or imcell_b.current_loc
        prompt = (
            f"You are a Vietnam travel expert. "
            f"Return JSON ONLY (no markdown). "
            f"Which Vietnam travel route is more interesting and worthwhile? "
            f"Route A: Hanoi → {desc_a}  |  Route B: Hanoi → {desc_b}. "
            f'Format: {{"winner": "a"}} or {{"winner": "b"}}'
        )
        try:
            result = _run_async(_async_query(prompt, temp))
            winner = result.get("winner", "a").lower().strip()
            return winner if winner in ("a", "b") else "a"
        except Exception as exc:
            log.warning("compare_paths failed: %s", exc)
        return "a"


# ── Mock Slipnet ──────────────────────────────────────────────────────────────

class MockSlipnet:
    """Deterministic, no-LLM slipnet for testing.

    Encodes 6 distinct Vietnam routes so that activation competition can be
    observed without any API calls:

    path-0  Direct flight:          Hanoi → Da Nang
    path-1  Train via Hue:          Hanoi → Hue → Da Nang
    path-2  Scenic bus:             Hanoi → Ninh Binh → Hue → Da Nang
    path-3  Cultural detour:        Hanoi → Hue → Hoi An → Da Nang
    path-4  Ha Long loop + flight:  Hanoi → Ha Long Bay → (back) Hanoi → Da Nang
    path-5  Northern adventure:     Hanoi → Vinh → Dong Hoi → Hue → Da Nang
    """

    # ── Fixed mode list per origin/destination ─────────────────────────────

    _MODES: Dict[Tuple[str, str], List[str]] = {
        ("Hanoi", "Da Nang"): ["flight", "train", "bus", "scenic", "halong", "adventure"],
    }

    # ── Route table: (from, to, mode) → leg dict ──────────────────────────
    # The "mode" key in each dict is the actual transport for that leg,
    # which may differ from the path's nominal mode (e.g. bus-detour then train).

    _ROUTES: Dict[Tuple[str, str, str], Dict[str, Any]] = {
        # ── flight path (path-0) ──────────────────────────────────────────
        ("Hanoi", "Da Nang", "flight"): {
            "from_loc": "Hanoi", "to_loc": "Da Nang",
            "mode": "flight", "duration_hours": 1.5,
            "notes": "Direct VietJet / Vietnam Airlines",
        },
        # ── train path (path-1) ───────────────────────────────────────────
        ("Hanoi", "Da Nang", "train"): {
            "from_loc": "Hanoi", "to_loc": "Hue",
            "mode": "train", "duration_hours": 13.0,
            "notes": "Reunification Express SE3/SE5",
        },
        ("Hue", "Da Nang", "train"): {
            "from_loc": "Hue", "to_loc": "Da Nang",
            "mode": "train", "duration_hours": 2.5,
            "notes": "Stunning Hai Van Pass coastal stretch",
        },
        # ── bus/scenic path (path-2): Ninh Binh detour ────────────────────
        ("Hanoi", "Da Nang", "bus"): {
            "from_loc": "Hanoi", "to_loc": "Ninh Binh",
            "mode": "bus", "duration_hours": 2.0,
            "notes": "Inland Ha Long Bay – Trang An caves",
        },
        ("Ninh Binh", "Da Nang", "bus"): {
            "from_loc": "Ninh Binh", "to_loc": "Hue",
            "mode": "train", "duration_hours": 10.0,
            "notes": "Overnight sleeper train south",
        },
        ("Hue", "Da Nang", "bus"): {
            "from_loc": "Hue", "to_loc": "Da Nang",
            "mode": "bus", "duration_hours": 3.0,
            "notes": "Direct open-tour bus via Hai Van Pass",
        },
        # ── scenic path (path-3): cultural detour via Hoi An ──────────────
        ("Hanoi", "Da Nang", "scenic"): {
            "from_loc": "Hanoi", "to_loc": "Hue",
            "mode": "flight", "duration_hours": 1.25,
            "notes": "Fly to Hue for Imperial Citadel",
        },
        ("Hue", "Da Nang", "scenic"): {
            "from_loc": "Hue", "to_loc": "Hoi An",
            "mode": "bus", "duration_hours": 3.5,
            "notes": "Ancient Town UNESCO World Heritage",
        },
        ("Hoi An", "Da Nang", "scenic"): {
            "from_loc": "Hoi An", "to_loc": "Da Nang",
            "mode": "motorbike", "duration_hours": 0.5,
            "notes": "Scenic coastal road",
        },
        # ── Ha Long loop (path-4): detour then fly direct ────────────────
        ("Hanoi", "Da Nang", "halong"): {
            "from_loc": "Hanoi", "to_loc": "Ha Long Bay",
            "mode": "bus", "duration_hours": 3.5,
            "notes": "UNESCO World Heritage karst seascape",
        },
        ("Ha Long Bay", "Da Nang", "halong"): {
            "from_loc": "Ha Long Bay", "to_loc": "Da Nang",
            "mode": "flight", "duration_hours": 5.0,
            "notes": "Return to Hanoi then fly south to Da Nang",
        },
        # ── Northern adventure (path-5) ───────────────────────────────────
        ("Hanoi", "Da Nang", "adventure"): {
            "from_loc": "Hanoi", "to_loc": "Vinh",
            "mode": "train", "duration_hours": 5.5,
            "notes": "Vinh – gateway to Ho Chi Minh Trail history",
        },
        ("Vinh", "Da Nang", "adventure"): {
            "from_loc": "Vinh", "to_loc": "Dong Hoi",
            "mode": "train", "duration_hours": 2.0,
            "notes": "Dong Hoi – Phong Nha caves access",
        },
        ("Dong Hoi", "Da Nang", "adventure"): {
            "from_loc": "Dong Hoi", "to_loc": "Hue",
            "mode": "train", "duration_hours": 2.5,
            "notes": "Through DMZ landscape",
        },
        ("Hue", "Da Nang", "adventure"): {
            "from_loc": "Hue", "to_loc": "Da Nang",
            "mode": "train", "duration_hours": 2.5,
            "notes": "Final leg via Hai Van Pass",
        },
    }

    _INTERESTING: set = {"Hue", "Hoi An", "Ha Long Bay", "Ninh Binh", "Dong Hoi", "Vinh"}

    def query_modes(self, from_loc: str, to_loc: str) -> List[str]:
        return list(self._MODES.get((from_loc, to_loc), ["flight", "train", "bus"]))

    def query_route(
        self,
        from_loc: str,
        to_loc: str,
        mode: str,
        activation: float = 0.5,
    ) -> Optional[Dict[str, Any]]:
        key = (from_loc, to_loc, mode)
        return self._ROUTES.get(key)

    def evaluate_path(
        self,
        path_id: str,
        legs: Any,
        activation: float = 0.5,
    ) -> float:
        stops = {l.to_loc for l in legs}
        bonus = len(stops & self._INTERESTING) * 0.15
        return min(1.0, 0.40 + bonus)

    def compare_paths(
        self,
        imcell_a: Any,
        imcell_b: Any,
        activation: float = 0.5,
    ) -> str:
        score_a = len({l.to_loc for l in imcell_a.legs} & self._INTERESTING)
        score_b = len({l.to_loc for l in imcell_b.legs} & self._INTERESTING)
        return "a" if score_a >= score_b else "b"
