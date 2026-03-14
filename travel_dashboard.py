"""
travel_dashboard.py — Streamlit dashboard for the FARG Travel Path Finder.

Launch with:
    streamlit run travel_dashboard.py

Panels
──────
1. Activation Timeline       — per-element activation traces over ticks
2. Workspace Composition     — stacked area of element counts by type
3. Act Probability Curve     — exploration → exploitation shift
4. Competition Graph         — live network of agents (activation-sized nodes,
                               green support / red antipathy edges)
5. LLM Perturbation Funnel   — proposed → survived 10 ticks → committed
6. Dynamic Temperature       — agent activation mapped to LLM temperature
7. Async Bottleneck Monitor  — PendingLLM count per tick
8. Path Comparison Panel     — final complete routes side-by-side
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# ── Force UTF-8 output on Windows ────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import networkx as nx
import numpy as np

from config import GOAL, MAX_TICKS, MIN_COMPLETE_PATHS, START
from metrics import MetricsCollector


# ── Type colours (consistent across all charts) ───────────────────────────────
TYPE_COLORS = {
    "Want":          "#e377c2",
    "SuggestMode":   "#1f77b4",
    "SuggestRoute":  "#ff7f0e",
    "SeekEvidence":  "#2ca02c",
    "Evaluate":      "#9467bd",
    "ImCell":        "#8c564b",
    "PendingLLM":    "#d62728",
    "Blocked":       "#7f7f7f",
    "GettingCloser": "#bcbd22",
    "PathComplete":  "#17becf",
    "GoalReached":   "#00e676",   # bright green — canvas chain hit goal
    "GoIsDone":      "#aec7e8",
    "ActIsDone":     "#ffbb78",
    "Other":         "#c7c7c7",
}

STACKED_TYPES = [
    "Want", "SuggestMode", "SuggestRoute", "SeekEvidence",
    "Evaluate", "ImCell", "PendingLLM", "Blocked",
]


# ── Simulation runner ─────────────────────────────────────────────────────────

def run_simulation(use_mock: bool, max_ticks: int):
    from agents import Want
    from main_loop import run_loop
    from slipnet import MockSlipnet, RealSlipnet, CountingSlipnet
    from tags import GoalReached
    from workspace import Workspace

    inner = MockSlipnet() if use_mock else RealSlipnet()
    counting = CountingSlipnet(inner)
    ws = Workspace()
    ws.add(Want(from_loc=START, to_loc=GOAL), init_a=1.0)
    mc = MetricsCollector()
    paths = run_loop(ws, counting, logger=mc, max_ticks=max_ticks)

    # Snapshot architecture-only call counts BEFORE post-run scoring.
    arch_calls = dict(counting.calls)

    # Score each completed FARG path with the same evaluate_path(activation=0.5)
    # call that baseline uses, so scores are directly comparable.
    farg_scores: dict = {}
    for pc in paths:
        pid  = pc.path_id if isinstance(pc, GoalReached) else pc.taggee.path_id
        legs = pc.legs    if isinstance(pc, GoalReached) else pc.taggee.legs
        farg_scores[pid] = counting.evaluate_path(f"farg-{pid}", legs, activation=0.5)

    return mc, paths, arch_calls, farg_scores


# ── Chart builders ────────────────────────────────────────────────────────────

def chart_activation_timeline(mc: MetricsCollector, top_n: int = 20) -> go.Figure:
    """Line chart: X=tick, Y=activation, one trace per top-N element."""
    # Aggregate: max activation across all ticks for each element id
    max_a: dict[str, float] = {}
    series: dict[str, list] = {}   # eid -> [a or None per tick]
    ticks = [s.tick for s in mc.snapshots]

    for snap in mc.snapshots:
        for eid, (_, a) in snap.activations.items():
            max_a[eid] = max(max_a.get(eid, 0.0), a)

    top_ids = sorted(max_a, key=lambda k: -max_a[k])[:top_n]

    for snap in mc.snapshots:
        for eid in top_ids:
            if eid not in series:
                series[eid] = []
            val = snap.activations.get(eid, (None, None))[1]
            series[eid].append(val)

    fig = go.Figure()
    for eid in top_ids:
        # Get type from first snapshot where seen
        etype = "Other"
        for snap in mc.snapshots:
            if eid in snap.activations:
                etype = snap.activations[eid][0]
                break
        fig.add_trace(go.Scatter(
            x=ticks,
            y=series[eid],
            mode="lines",
            name=eid,
            line=dict(color=TYPE_COLORS.get(etype, "#999"), width=1.5),
            hovertemplate="%{y:.3f}<extra>" + eid + "</extra>",
        ))

    fig.update_layout(
        title="Activation Timeline (top-N elements)",
        xaxis_title="Tick",
        yaxis_title="Activation",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(font_size=10, orientation="h"),
        height=420,
    )
    return fig


def chart_workspace_composition(mc: MetricsCollector) -> go.Figure:
    """Stacked area chart: element counts by type per tick."""
    ticks = [s.tick for s in mc.snapshots]
    fig = go.Figure()
    for ttype in STACKED_TYPES:
        counts = [s.type_counts.get(ttype, 0) for s in mc.snapshots]
        if any(c > 0 for c in counts):
            fig.add_trace(go.Scatter(
                x=ticks, y=counts,
                mode="lines",
                name=ttype,
                stackgroup="one",
                line=dict(width=0.5, color=TYPE_COLORS.get(ttype, "#999")),
                fillcolor=TYPE_COLORS.get(ttype, "#999"),
                hovertemplate=ttype + ": %{y}<extra></extra>",
            ))

    fig.update_layout(
        title="Workspace Composition — Metabolic Rhythm",
        xaxis_title="Tick",
        yaxis_title="Element count",
        legend=dict(font_size=10, orientation="h"),
        height=360,
    )
    return fig


def chart_act_probability(mc: MetricsCollector) -> go.Figure:
    """Line chart: fraction of eligible agents that can_act per tick."""
    ticks = [s.tick for s in mc.snapshots]
    probs = [s.act_prob for s in mc.snapshots]

    # Smooth with rolling window for readability
    window = 5
    smoothed = pd.Series(probs).rolling(window, min_periods=1, center=True).mean().tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ticks, y=probs,
        mode="lines",
        name="raw",
        line=dict(color="#aaaaaa", width=1),
        opacity=0.4,
    ))
    fig.add_trace(go.Scatter(
        x=ticks, y=smoothed,
        mode="lines",
        name=f"smoothed (w={window})",
        line=dict(color="#ff7f0e", width=2),
    ))
    fig.update_layout(
        title="Act Probability — Exploration -> Exploitation Shift",
        xaxis_title="Tick",
        yaxis_title="P(can act | eligible)",
        yaxis=dict(range=[0, 1.05]),
        height=300,
    )
    return fig


def chart_competition_graph(mc: MetricsCollector, tick_idx: int) -> go.Figure:
    """Network graph of live agents at a given tick snapshot."""
    snap = mc.snapshots[tick_idx]

    # Build networkx graph from snapshot
    G = nx.DiGraph()
    node_act: dict[str, float] = {}
    node_type: dict[str, str] = {}
    for eid, (etype, a) in snap.activations.items():
        G.add_node(eid)
        node_act[eid] = a
        node_type[eid] = etype

    support_pairs: list[tuple] = []
    antipathy_pairs: list[tuple] = []
    seen_pairs: set = set()
    for u, v, w in snap.edges:
        if u not in G.nodes or v not in G.nodes:
            continue
        pair = (min(u, v), max(u, v))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        if w > 0:
            support_pairs.append((u, v))
        else:
            antipathy_pairs.append((u, v))

    if len(G.nodes) == 0:
        return go.Figure().update_layout(title="No nodes at this tick")

    pos = nx.spring_layout(G, seed=42, k=1.8)

    fig = go.Figure()

    # Support edges (green)
    for u, v in support_pairs:
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(color="rgba(44,160,44,0.4)", width=1),
                hoverinfo="none", showlegend=False,
            ))

    # Antipathy edges (red dashed)
    for u, v in antipathy_pairs:
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(color="rgba(214,39,40,0.5)", width=1, dash="dot"),
                hoverinfo="none", showlegend=False,
            ))

    # Nodes grouped by type for legend
    for ttype in set(node_type.values()):
        xs, ys, texts, sizes, colors = [], [], [], [], []
        for eid in G.nodes():
            if node_type.get(eid) != ttype:
                continue
            if eid not in pos:
                continue
            x, y = pos[eid]
            a = node_act.get(eid, 0.0)
            xs.append(x)
            ys.append(y)
            texts.append(f"{eid}<br>a={a:.3f}")
            sizes.append(max(6, 30 * a))
            colors.append(TYPE_COLORS.get(ttype, "#999"))

        if not xs:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            name=ttype,
            marker=dict(size=sizes, color=colors[0], opacity=0.85, line=dict(width=1, color="white")),
            text=[eid.split("[")[-1].rstrip("]") for eid in G.nodes() if node_type.get(eid) == ttype and eid in pos],
            textposition="top center",
            textfont=dict(size=8),
            hovertext=texts,
            hoverinfo="text",
        ))

    fig.update_layout(
        title=f"Competition Graph at Tick {snap.tick}",
        showlegend=True,
        legend=dict(font_size=9),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
    )
    return fig


def chart_perturbation_funnel(mc: MetricsCollector) -> go.Figure:
    """Horizontal funnel: proposed -> survived 10 ticks -> committed."""
    total, survived, committed = mc.survival_funnel()
    labels = ["Proposed by LLM", "Survived 10 ticks", "Committed to Canvas"]
    values = [total, survived, committed]
    pcts = [
        "100%",
        f"{100*survived/max(1,total):.0f}%",
        f"{100*committed/max(1,total):.0f}%",
    ]

    fig = go.Figure(go.Funnel(
        y=labels,
        x=values,
        textinfo="value+percent initial",
        marker=dict(color=["#1f77b4", "#ff7f0e", "#2ca02c"]),
        connector=dict(line=dict(color="#888", width=2)),
    ))
    fig.update_layout(
        title="LLM Perturbation Survival Funnel",
        height=300,
    )
    return fig


def chart_temperature(mc: MetricsCollector) -> go.Figure:
    """Scatter: SuggestMode activation vs LLM temperature at each tick."""
    from slipnet import temperature_for_activation

    ticks, activations, temps, labels = [], [], [], []
    for snap in mc.snapshots:
        for eid, (etype, a) in snap.activations.items():
            if etype == "SuggestMode":
                ticks.append(snap.tick)
                activations.append(a)
                temps.append(temperature_for_activation(a))
                labels.append(eid)

    if not ticks:
        return go.Figure().update_layout(title="No SuggestMode data yet")

    # Reference line: the formula
    a_range = np.linspace(0, 1, 100)
    t_range = [temperature_for_activation(float(a)) for a in a_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=activations, y=temps,
        mode="markers",
        marker=dict(color=ticks, colorscale="Viridis", size=6, opacity=0.7,
                    colorbar=dict(title="Tick")),
        text=labels,
        hovertemplate="a=%{x:.3f} -> T=%{y:.3f}<br>%{text}<extra></extra>",
        name="Agents",
    ))
    fig.add_trace(go.Scatter(
        x=list(a_range), y=t_range,
        mode="lines",
        line=dict(color="red", width=2, dash="dash"),
        name="Formula: T = (0.9-0.7a) × max(0.2, 1-0.15d)  [d=0 shown]",
    ))
    fig.update_layout(
        title="Dynamic Temperature Tracker — T = f(activation) × g(depth)",
        xaxis_title="Agent Activation",
        yaxis_title="LLM Temperature",
        xaxis=dict(range=[0, 1.05]),
        yaxis=dict(range=[0, 1.05]),
        height=360,
    )
    return fig


def chart_pending_llm(mc: MetricsCollector) -> go.Figure:
    """Bar chart: PendingLLM count per tick (async bottleneck monitor)."""
    ticks = [s.tick for s in mc.snapshots]
    counts = [s.pending_llm for s in mc.snapshots]

    fig = go.Figure(go.Bar(
        x=ticks, y=counts,
        marker_color="#d62728",
        opacity=0.7,
        name="PendingLLM",
    ))
    fig.update_layout(
        title="Async Bottleneck Monitor — PendingLLM count per tick",
        xaxis_title="Tick",
        yaxis_title="Agents waiting on LLM",
        height=260,
    )
    return fig


def panel_path_comparison(paths: list) -> None:
    """Side-by-side cards for each complete path.

    Accepts GoalReached tags (canvas-chain completions with .path_id, .legs,
    .path) as well as legacy PathComplete tags (ImCell-based, with .taggee).
    """
    from tags import GoalReached

    if not paths:
        st.warning("No complete paths found.")
        return

    cols = st.columns(min(len(paths), 3))
    for i, pc in enumerate(paths):
        col = cols[i % len(cols)]
        with col:
            # Resolve fields from either tag type
            if isinstance(pc, GoalReached):
                path_id = pc.path_id
                legs = pc.legs
            else:
                # Legacy PathComplete: taggee is an ImCell
                path_id = pc.taggee.path_id
                legs = pc.taggee.legs

            st.markdown(f"### Path {path_id}")
            total_h = sum(l.duration_hours for l in legs)
            st.metric("Total hours", f"{total_h:.1f} h")
            st.metric("Legs", len(legs))
            for leg in legs:
                is_goal_leg = leg.to_loc == GOAL
                badge = ":green[GOAL]" if is_goal_leg else ""
                st.markdown(
                    f"**{leg.from_loc}** --[{leg.mode}]--> **{leg.to_loc}** "
                    f"({leg.duration_hours:.1f}h) {badge}"
                )
                if leg.notes:
                    st.caption(leg.notes)
            route_str = " -> ".join(pc.path)
            st.code(route_str, language=None)


def _path_diversity(legs_list) -> tuple[int, int]:
    """Return (unique_modes, unique_intermediate_cities) across a list of leg sequences."""
    modes: set = set()
    cities: set = set()
    for legs in legs_list:
        for i, leg in enumerate(legs):
            modes.add(leg.mode)
            # Intermediate = not the final destination
            if i < len(legs) - 1:
                cities.add(leg.to_loc)
    return len(modes), len(cities)


def chart_score_vs_hours(farg_paths: list, farg_scores: dict, base_run) -> go.Figure:
    """Scatter plot: quality score (Y) vs total hours (X), one point per path.

    FARG: blue circles.  Baseline: orange diamonds.
    Each trace annotated with n= count.
    """
    from tags import GoalReached

    fig = go.Figure()

    # ── FARG trace ────────────────────────────────────────────────────────────
    f_x, f_y, f_text = [], [], []
    for pc in farg_paths:
        if isinstance(pc, GoalReached):
            pid, legs = pc.path_id, pc.legs
        else:
            pid, legs = pc.taggee.path_id, pc.taggee.legs
        hours = sum(l.duration_hours for l in legs)
        score = farg_scores.get(pid)
        if score is None:
            continue
        route = " -> ".join(pc.path)
        f_x.append(hours)
        f_y.append(score)
        f_text.append(
            f"<b>FARG</b> {pid}<br>"
            f"Score: {score:.2f}<br>"
            f"Hours: {hours:.1f}<br>"
            f"Legs: {len(legs)}<br>"
            f"Route: {route}"
        )

    n_farg = len(f_x)
    if f_x:
        fig.add_trace(go.Scatter(
            x=f_x, y=f_y,
            mode="markers+text",
            name=f"FARG (n={n_farg})",
            marker=dict(symbol="circle", size=14, color="#1f77b4",
                        line=dict(width=1.5, color="white")),
            text=[f"F{i+1}" for i in range(n_farg)],
            textposition="top center",
            textfont=dict(size=9, color="#1f77b4"),
            hovertext=f_text,
            hoverinfo="text",
        ))

    # ── Baseline trace ────────────────────────────────────────────────────────
    b_x, b_y, b_text = [], [], []
    for r in base_run.paths:
        hours = r.total_hours()
        route = " -> ".join(r.path)
        b_x.append(hours)
        b_y.append(r.score)
        b_text.append(
            f"<b>Baseline</b> [{r.mode}]<br>"
            f"Score: {r.score:.2f}<br>"
            f"Hours: {hours:.1f}<br>"
            f"Legs: {len(r.legs)}<br>"
            f"Route: {route}"
        )

    n_base = len(b_x)
    if b_x:
        fig.add_trace(go.Scatter(
            x=b_x, y=b_y,
            mode="markers+text",
            name=f"Baseline (n={n_base})",
            marker=dict(symbol="diamond", size=14, color="#ff7f0e",
                        line=dict(width=1.5, color="white")),
            text=[f"B{i+1}" for i in range(n_base)],
            textposition="top center",
            textfont=dict(size=9, color="#ff7f0e"),
            hovertext=b_text,
            hoverinfo="text",
        ))

    fig.update_layout(
        title=(
            "Score vs Travel Hours — same LLM, same 4 prompts<br>"
            "<sup>Both systems scored by evaluate_path(activation=0.5) for parity</sup>"
        ),
        xaxis_title="Total travel hours",
        yaxis_title="Quality score [0–1]",
        yaxis=dict(range=[0, 1.1]),
        legend=dict(orientation="h", y=1.12),
        height=420,
    )
    return fig


def chart_call_breakdown(farg_explore_calls: dict, farg_score_calls: dict,
                         base_run) -> go.Figure:
    """Horizontal stacked bar: LLM calls by type and purpose for each system.

    Segments: query_modes (yellow), query_route (blue),
              evaluate_path (green), compare_paths (purple).
    Two sub-bars per system label: 'Exploration calls' vs 'Scoring calls'.
    """
    CALL_COLORS = {
        "query_modes":    "#f9c74f",
        "query_route":    "#4895ef",
        "evaluate_path":  "#4cc9f0",
        "compare_paths":  "#7b2d8b",
    }
    CALL_TYPES = ["query_modes", "query_route", "evaluate_path", "compare_paths"]

    base_calls = getattr(base_run, "calls_by_type", {})

    # y-axis labels — two bars per system
    y_labels = [
        "FARG — Exploration calls",
        "FARG — Scoring calls",
        "Baseline — All calls",
    ]

    # Data rows: [farg_explore, farg_score, baseline]
    data_rows = [farg_explore_calls, farg_score_calls, base_calls]
    totals = [sum(r.values()) for r in data_rows]
    n_farg = len([p for p in st.session_state.get("paths", [])]) or 1
    n_base = len(base_run.paths) or 1

    fig = go.Figure()
    for ctype in CALL_TYPES:
        vals = [row.get(ctype, 0) for row in data_rows]
        fig.add_trace(go.Bar(
            name=ctype,
            y=y_labels,
            x=vals,
            orientation="h",
            marker_color=CALL_COLORS[ctype],
            hovertemplate=f"{ctype}: %{{x}}<extra></extra>",
        ))

    # Annotations: calls/path at end of each bar
    calls_per_path = [
        totals[0] / n_farg,
        totals[1] / n_farg,
        totals[2] / n_base,
    ]
    annotations = []
    for i, (label, total, cpp) in enumerate(zip(y_labels, totals, calls_per_path)):
        annotations.append(dict(
            x=total + 0.5,
            y=label,
            text=f"  {total} total ({cpp:.1f}/path)",
            showarrow=False,
            xanchor="left",
            font=dict(size=11),
        ))

    fig.update_layout(
        title="LLM Call Breakdown — Exploration vs Scoring",
        xaxis_title="Number of LLM calls",
        barmode="stack",
        legend=dict(orientation="h", y=1.12),
        annotations=annotations,
        height=300,
    )
    return fig


def panel_farg_vs_baseline(farg_paths: list, farg_scores: dict,
                            farg_explore_calls: dict, base_run) -> None:
    """Full FARG vs Baseline comparison section."""
    from tags import GoalReached

    st.subheader("FARG vs Baseline — Head-to-Head")
    st.caption(
        "Same LLM, same 4 prompts — FARG uses activation dynamics to prune "
        "unpromising branches early. Baseline exhaustively explores every mode.  \n"
        "**Scoring parity**: both systems scored with `evaluate_path(activation=0.5)` "
        "after the run, so scores are directly comparable."
    )

    # ── Derive secondary data ─────────────────────────────────────────────────
    farg_scored_paths = []
    for pc in farg_paths:
        if isinstance(pc, GoalReached):
            pid, legs = pc.path_id, pc.legs
        else:
            pid, legs = pc.taggee.path_id, pc.taggee.legs
        if pid in farg_scores:
            farg_scored_paths.append((pid, legs))

    farg_scores_list = [farg_scores[pid] for pid, _ in farg_scored_paths]
    base_scores_list  = [r.score for r in base_run.paths]
    farg_avg  = sum(farg_scores_list) / max(1, len(farg_scores_list))
    base_avg  = sum(base_scores_list) / max(1, len(base_scores_list))

    farg_explore_total = sum(farg_explore_calls.values())
    # Scoring calls = total counting.calls minus exploration calls
    farg_score_total = sum(farg_scores_extra.get(k, 0)
                           for k in farg_explore_calls
                           for farg_scores_extra in [{}])   # computed below
    # Build scoring call dict: evaluate_path calls made AFTER arch snapshot
    # = (current counting total) - (arch snapshot total).
    # We receive farg_explore_calls (arch snapshot) and must infer score calls.
    # evaluate_path is the only post-run call type; query_modes/route/compare stay 0.
    farg_score_calls: dict = {k: 0 for k in farg_explore_calls}
    n_farg_paths = len(farg_scored_paths)
    farg_score_calls["evaluate_path"] = n_farg_paths

    farg_score_total = n_farg_paths
    base_total = sum(getattr(base_run, "calls_by_type", {}).values()) or base_run.llm_calls
    n_farg_total_calls = farg_explore_total + farg_score_total

    n_farg = len(farg_paths) or 1
    n_base = len(base_run.paths) or 1

    farg_legs_list   = [legs for _, legs in farg_scored_paths]
    base_legs_list   = [r.legs for r in base_run.paths]
    farg_umodes, farg_ucities = _path_diversity(farg_legs_list)
    base_umodes, base_ucities = _path_diversity(base_legs_list)

    # ── 8-tile KPI grid ───────────────────────────────────────────────────────
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric(
        "FARG exploration calls",
        farg_explore_total,
        help="LLM calls made during the FARG run (query_modes + query_route + evaluate + compare). "
             "Does NOT include post-run scoring.",
    )
    r1c2.metric(
        "Baseline total calls",
        base_total,
        help="All LLM calls in the fixed sequential pipeline.",
    )
    r1c3.metric("FARG avg score", f"{farg_avg:.2f}",
                help="evaluate_path(activation=0.5) — same formula as baseline.")
    r1c4.metric("Baseline avg score", f"{base_avg:.2f}")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("FARG paths found", len(farg_paths))
    r2c2.metric(
        "Baseline paths found",
        f"{base_run.modes_complete}/{base_run.modes_tried} modes",
    )
    r2c3.metric(
        "FARG exploration calls/path",
        f"{farg_explore_total / n_farg:.1f}",
        help="Exploration calls divided by number of paths found — architectural efficiency.",
    )
    r2c4.metric(
        "Baseline calls/path",
        f"{base_total / n_base:.1f}",
    )

    # ── Diversity metrics ─────────────────────────────────────────────────────
    st.markdown("**Path Diversity**")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("FARG unique modes", farg_umodes,
              help="Transport modes appearing in at least one FARG path.")
    d2.metric("Baseline unique modes", base_umodes)
    d3.metric("FARG unique intermediate cities", farg_ucities,
              help="Distinct non-goal waypoints across all FARG paths.")
    d4.metric("Baseline unique intermediate cities", base_ucities)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.plotly_chart(
        chart_score_vs_hours(farg_paths, farg_scores, base_run),
        use_container_width=True,
    )
    st.plotly_chart(
        chart_call_breakdown(farg_explore_calls, farg_score_calls, base_run),
        use_container_width=True,
    )

    # ── Side-by-side path cards ────────────────────────────────────────────────
    st.divider()
    col_farg, col_base = st.columns(2)

    with col_farg:
        st.markdown("#### FARG paths")
        if not farg_paths:
            st.warning("No FARG paths — run the FARG simulation first.")
        for pc in farg_paths:
            if isinstance(pc, GoalReached):
                pid, legs, path = pc.path_id, pc.legs, pc.path
            else:
                pid, legs, path = pc.taggee.path_id, pc.taggee.legs, pc.path
            score = farg_scores.get(pid)
            score_str = f"  score: **{score:.2f}**" if score is not None else ""
            with st.container(border=True):
                st.markdown(
                    f"**{pid}** — {sum(l.duration_hours for l in legs):.1f} h{score_str}"
                )
                for leg in legs:
                    badge = " :green[GOAL]" if leg.to_loc == GOAL else ""
                    st.markdown(
                        f"`{leg.from_loc}` --[{leg.mode}]--> `{leg.to_loc}` "
                        f"({leg.duration_hours:.1f}h){badge}"
                    )
                st.code(" -> ".join(path), language=None)

    with col_base:
        st.markdown("#### Baseline paths")
        if not base_run.paths:
            st.warning("No baseline paths — run the baseline pipeline first.")
        for r in base_run.paths:
            with st.container(border=True):
                st.markdown(
                    f"**{r.mode}** — {r.total_hours():.1f} h  "
                    f"score: **{r.score:.2f}**"
                )
                for leg in r.legs:
                    badge = " :green[GOAL]" if leg.to_loc == GOAL else ""
                    st.markdown(
                        f"`{leg.from_loc}` --[{leg.mode}]--> `{leg.to_loc}` "
                        f"({leg.duration_hours:.1f}h){badge}"
                    )
                st.code(" -> ".join(r.path), language=None)


# ── Main Streamlit app ────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="FARG Travel Path Finder — Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("FARG Travel Path Finder — Live Dashboard")
    st.caption(f"Stochastic cognitive architecture: {START} -> {GOAL}")

    # ── Sidebar config ────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Simulation Config")
        use_mock = st.toggle("Use MockSlipnet (no API calls)", value=True)
        max_ticks = st.slider("Max ticks", 20, 200, MAX_TICKS, step=10)
        top_n = st.slider("Activation timeline: top-N elements", 5, 40, 15, step=5)
        run_btn = st.button("Run FARG Simulation", type="primary", use_container_width=True)
        base_btn = st.button("Run Baseline (no dynamics)", use_container_width=True)

        if not use_mock and not os.getenv("OPENROUTER_API_KEY"):
            st.warning("OPENROUTER_API_KEY not set — will fall back to mock.")

        st.divider()
        st.markdown("**Legend**")
        for ttype, color in TYPE_COLORS.items():
            if ttype in STACKED_TYPES + ["ImCell", "PathComplete", "GoalReached"]:
                st.markdown(
                    f'<span style="color:{color}">■</span> {ttype}',
                    unsafe_allow_html=True,
                )

    # ── Run / load state ──────────────────────────────────────────────────────
    _use_mock = use_mock or not os.getenv("OPENROUTER_API_KEY")

    if run_btn:
        with st.spinner("Running FARG simulation..."):
            mc, paths, arch_calls, farg_scores = run_simulation(use_mock=_use_mock, max_ticks=max_ticks)
        st.session_state["mc"] = mc
        st.session_state["paths"] = paths
        st.session_state["farg_calls"] = arch_calls
        st.session_state["farg_scores"] = farg_scores
        st.success(f"Done — {len(paths)} path(s) found in {len(mc.snapshots)} ticks.")

    if base_btn:
        from baseline import run_baseline
        from slipnet import MockSlipnet, RealSlipnet, CountingSlipnet
        with st.spinner("Running baseline pipeline..."):
            inner = MockSlipnet() if _use_mock else RealSlipnet()
            base_counting = CountingSlipnet(inner)   # caller owns the wrapper
            base_run = run_baseline(base_counting, START, GOAL)
            base_run.calls_by_type = dict(base_counting.calls)
            base_run.llm_calls = base_counting.total
        st.session_state["base_run"] = base_run
        st.success(
            f"Baseline done — {base_run.modes_complete}/{base_run.modes_tried} modes "
            f"reached goal in {base_run.llm_calls} LLM calls."
        )

    mc: MetricsCollector | None = st.session_state.get("mc")
    paths: list = st.session_state.get("paths", [])

    if mc is None:
        st.info("Configure and click **Run Simulation** to begin.")
        return

    ticks_available = len(mc.snapshots)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    total_p, surv_p, comm_p = mc.survival_funnel()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Ticks run", ticks_available)
    k2.metric("Paths found", len(paths))
    k3.metric("LLM legs proposed", total_p)
    k4.metric("Survived 10 ticks", surv_p, f"{100*surv_p//max(1,total_p)}%")
    k5.metric("Committed to canvas", comm_p, f"{100*comm_p//max(1,total_p)}%")

    st.divider()

    # ── Row 1: Activation + Composition ───────────────────────────────────────
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(chart_activation_timeline(mc, top_n), use_container_width=True)
    with c2:
        st.plotly_chart(chart_workspace_composition(mc), use_container_width=True)

    # ── Row 2: Act Probability + Temperature ──────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(chart_act_probability(mc), use_container_width=True)
    with c4:
        st.plotly_chart(chart_temperature(mc), use_container_width=True)

    # ── Row 3: Competition Graph ───────────────────────────────────────────────
    st.subheader("Competition Graph (Network Topology)")
    tick_idx = st.slider(
        "Select tick to visualise",
        0, ticks_available - 1,
        min(ticks_available - 1, 20),
        key="graph_tick",
    )
    st.plotly_chart(chart_competition_graph(mc, tick_idx), use_container_width=True)

    # ── Row 4: Funnel + Async Bottleneck ──────────────────────────────────────
    c5, c6 = st.columns([2, 3])
    with c5:
        st.plotly_chart(chart_perturbation_funnel(mc), use_container_width=True)
    with c6:
        st.plotly_chart(chart_pending_llm(mc), use_container_width=True)

    # ── Row 5: Path Comparison ────────────────────────────────────────────────
    st.divider()
    st.subheader("Path Comparison Panel")
    panel_path_comparison(paths)

    # ── Row 6: FARG vs Baseline comparison ────────────────────────────────────
    base_run    = st.session_state.get("base_run")
    farg_calls  = st.session_state.get("farg_calls", {})
    farg_scores = st.session_state.get("farg_scores", {})

    if base_run is not None or paths:
        st.divider()
        if base_run is None:
            st.info("Click **Run Baseline (no dynamics)** in the sidebar to enable head-to-head comparison.")
        elif not paths:
            st.info("Click **Run FARG Simulation** to add FARG paths to the comparison.")
        else:
            panel_farg_vs_baseline(paths, farg_scores, farg_calls, base_run)

    # ── Raw data expander ─────────────────────────────────────────────────────
    with st.expander("Raw snapshot data (DataFrame)"):
        rows = []
        for s in mc.snapshots:
            rows.append({
                "tick": s.tick,
                "act_prob": round(s.act_prob, 4),
                "pending_llm": s.pending_llm,
                "canvas_total": s.canvas_total,
                "complete": s.complete_count,
                "chosen_type": s.chosen_type or "",
                **{t: s.type_counts.get(t, 0) for t in STACKED_TYPES},
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


if __name__ == "__main__":
    main()
