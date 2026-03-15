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


def chart_exploration_depth(tree: dict) -> go.Figure:
    """Stacked bar: ImCells born at each depth, colored by first-stop direction.

    Answers: how many distinct first-legs did FARG create ImCells for, and
    how many survived to depth 2, depth 3, … ?
    Accepts the dict returned by MetricsCollector.exploration_tree() (or
    _aggregate_tree()) directly so it works for both single- and multi-run views.
    """
    if not tree["births"]:
        return go.Figure().update_layout(title="No ImCell births recorded yet")

    depth_counts = tree["depth_counts"]
    max_depth = max(depth_counts.keys())
    all_first_stops = sorted(tree["first_stop_counts"].keys())

    # For each first-stop, count how many ImCells at each depth started from it
    from collections import defaultdict
    depth_first: dict = defaultdict(lambda: defaultdict(int))
    for rec in tree["births"]:
        first = rec.legs[0].to_loc
        depth_first[rec.depth][first] += 1

    palette = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
               "#e377c2", "#17becf", "#bcbd22", "#aec7e8"]
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(all_first_stops)}

    x_labels = [f"Depth {d}" for d in range(1, max_depth + 1)]

    fig = go.Figure()
    for first_stop in all_first_stops:
        ys = [depth_first[d].get(first_stop, 0) for d in range(1, max_depth + 1)]
        fig.add_trace(go.Bar(
            name=f"via {first_stop}",
            x=x_labels,
            y=ys,
            marker_color=color_map[first_stop],
            hovertemplate=f"via {first_stop}<br>depth %{{x}}: %{{y}} ImCells<extra></extra>",
        ))

    # Annotate total per depth
    for d in range(1, max_depth + 1):
        total = depth_counts.get(d, 0)
        fig.add_annotation(
            x=f"Depth {d}", y=total,
            text=str(total),
            showarrow=False,
            yshift=8,
            font=dict(size=11, color="#333"),
        )

    n_unique_first = len(all_first_stops)
    fig.update_layout(
        title=(
            f"Exploration Breadth vs Survival — "
            f"{n_unique_first} distinct first-leg direction(s)<br>"
            f"<sup>Each bar = ImCells born at that depth; stacked by first-stop city</sup>"
        ),
        xaxis_title="Exploration depth (legs in ImCell)",
        yaxis_title="ImCells born",
        barmode="stack",
        legend=dict(orientation="h", y=1.14),
        height=360,
    )
    return fig


def chart_exploration_sankey(tree: dict) -> go.Figure:
    """Sankey diagram showing how FARG's exploration branched and was pruned.

    Nodes = (city, depth_level) pairs — the same city at different depths is
    a separate node so the tree structure is preserved.  GOAL collapses to one
    terminal node regardless of depth.
    Accepts an exploration_tree() dict directly (single- or multi-run).
    """
    if not tree["transitions"]:
        return go.Figure().update_layout(title="No exploration data yet")

    # ── Build node registry ────────────────────────────────────────────────────
    node_tuples: list = []       # ordered list of (city, depth) tuples
    node_index: dict = {}        # (city, depth) → int index

    def _nid(city: str, depth: int) -> int:
        key = (GOAL, 999) if city == GOAL else (city, depth)
        if key not in node_index:
            node_index[key] = len(node_tuples)
            node_tuples.append(key)
        return node_index[key]

    # Pre-register START at depth 0
    _nid(START, 0)

    # ── Build links from transition dict ──────────────────────────────────────
    link_sources, link_targets, link_values, link_labels = [], [], [], []
    for (from_city, to_city, depth), count in sorted(tree["transitions"].items()):
        src = _nid(from_city, depth - 1)
        tgt = _nid(to_city, depth)
        link_sources.append(src)
        link_targets.append(tgt)
        link_values.append(count)
        link_labels.append(f"{from_city} → {to_city} (depth {depth}): {count}")

    # ── Node display names and colours ────────────────────────────────────────
    node_labels = []
    node_colors = []
    for city, depth in node_tuples:
        if city == GOAL:
            node_labels.append(f"GOAL\n{city}")
            node_colors.append("#00e676")
        elif depth == 0:
            node_labels.append(f"START\n{city}")
            node_colors.append("#1f77b4")
        elif depth == 1:
            node_labels.append(f"{city}\n(leg 1)")
            node_colors.append("#ff7f0e")
        else:
            node_labels.append(f"{city}\n(leg {depth})")
            node_colors.append("#9467bd")

    total_births = len(tree["births"])
    unique_first = len(tree["first_stop_counts"])

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=24,
            thickness=22,
            line=dict(color="#444", width=0.6),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=link_sources,
            target=link_targets,
            value=link_values,
            label=link_labels,
            color="rgba(150,150,150,0.35)",
        ),
    ))
    fig.update_layout(
        title=(
            f"ImCell Exploration Tree — {total_births} ImCells born, "
            f"{unique_first} distinct first-leg direction(s)<br>"
            "<sup>Link width = number of ImCells that used that edge; "
            "pruned branches narrow as depth increases</sup>"
        ),
        height=500,
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
        key="cmp_score_vs_hours",
    )
    st.plotly_chart(
        chart_call_breakdown(farg_explore_calls, farg_score_calls, base_run),
        use_container_width=True,
        key="cmp_call_breakdown",
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


# ── Multi-run helpers ─────────────────────────────────────────────────────────

import statistics as _stats
from collections import Counter as _Counter


def _score_stats(scores: list) -> dict:
    """Descriptive statistics for a list of float scores."""
    if not scores:
        return {k: None for k in
                ["n", "mean", "median", "stdev", "variance", "min", "max", "iqr", "mode"]}
    n = len(scores)
    mean     = _stats.mean(scores)
    median   = _stats.median(scores)
    stdev    = _stats.pstdev(scores)      # population stdev (all observed paths = pop.)
    variance = _stats.pvariance(scores)
    s_min    = min(scores)
    s_max    = max(scores)
    srt      = sorted(scores)
    q1       = srt[n // 4]
    q3       = srt[min((3 * n) // 4, n - 1)]
    iqr      = q3 - q1
    # Mode: bucket to nearest 0.05 so it's meaningful for continuous scores
    buckets  = [round(round(s / 0.05) * 0.05, 2) for s in scores]
    mode_val = _Counter(buckets).most_common(1)[0][0]
    return dict(n=n, mean=mean, median=median, stdev=stdev, variance=variance,
                min=s_min, max=s_max, iqr=iqr, mode=mode_val)


def _aggregate_tree(farg_history: list) -> dict:
    """Merge ImCellBirthRecords from all FARG runs into one exploration_tree dict."""
    all_births = []
    for run in farg_history:
        all_births.extend(run["mc"].imcell_births)
    if not all_births:
        return {"births": [], "depth_counts": {}, "transitions": {}, "first_stop_counts": {}}

    depth_counts: dict = {}
    transitions: dict  = {}
    first_stop_counts: dict = {}
    for rec in all_births:
        depth_counts[rec.depth] = depth_counts.get(rec.depth, 0) + 1
        if rec.depth >= 1:
            first = rec.legs[0].to_loc
            first_stop_counts[first] = first_stop_counts.get(first, 0) + 1
        from_city = rec.legs[-2].to_loc if rec.depth > 1 else rec.legs[0].from_loc
        to_city   = rec.legs[-1].to_loc
        key       = (from_city, to_city, rec.depth)
        transitions[key] = transitions.get(key, 0) + 1

    return dict(births=all_births, depth_counts=depth_counts,
                transitions=transitions, first_stop_counts=first_stop_counts)


def chart_score_distribution(farg_history: list, base_history: list) -> go.Figure:
    """Violin + box plot of scores across all runs for each system."""
    farg_scores = [s for r in farg_history for s in r["farg_scores"].values()]
    base_scores = [p.score for r in base_history for p in r.paths]

    fig = go.Figure()
    if farg_scores:
        fig.add_trace(go.Violin(
            y=farg_scores,
            name=f"FARG (n={len(farg_scores)})",
            box_visible=True, meanline_visible=True,
            points="all", jitter=0.35, pointpos=-1.6,
            marker=dict(color="#1f77b4", size=5, opacity=0.55),
            line_color="#1f77b4",
            fillcolor="rgba(31,119,180,0.22)",
            hovertemplate="FARG score: %{y:.3f}<extra></extra>",
        ))
    if base_scores:
        fig.add_trace(go.Violin(
            y=base_scores,
            name=f"Baseline (n={len(base_scores)})",
            box_visible=True, meanline_visible=True,
            points="all", jitter=0.35, pointpos=1.6,
            marker=dict(color="#ff7f0e", size=5, opacity=0.55),
            line_color="#ff7f0e",
            fillcolor="rgba(255,127,14,0.22)",
            hovertemplate="Baseline score: %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        title=(
            f"Score Distribution — "
            f"{len(farg_history)} FARG / {len(base_history)} Baseline run(s)"
        ),
        yaxis=dict(title="Quality score [0–1]", range=[0, 1.1]),
        violinmode="group",
        height=420,
    )
    return fig


def chart_per_run_trend(farg_history: list, base_history: list) -> go.Figure:
    """Avg score per run with shaded min/max envelope — shows stability."""
    fig = go.Figure()

    _SYSTEMS = [
        # (history, label, line_color, fill_color, score_iter)
        (farg_history, "FARG",     "#1f77b4", "rgba(31,119,180,0.12)",
         lambda r: list(r["farg_scores"].values())),
        (base_history, "Baseline", "#ff7f0e", "rgba(255,127,14,0.12)",
         lambda r: [p.score for p in r.paths]),
    ]
    for history, label, lc, fc, scores_fn in _SYSTEMS:
        if not history:
            continue
        xs    = list(range(1, len(history) + 1))
        sc    = [scores_fn(r) for r in history]
        avgs  = [sum(s) / max(1, len(s)) for s in sc]
        maxes = [max(s, default=0.0) for s in sc]
        mins  = [min(s, default=0.0) for s in sc]
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1], y=maxes + mins[::-1],
            fill="toself", fillcolor=fc,
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{label} min/max range", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=avgs, mode="lines+markers",
            name=f"{label} avg",
            line=dict(color=lc, width=2),
            marker=dict(size=7, color=lc),
            hovertemplate=f"Run %{{x}}: avg=%{{y:.3f}}<extra>{label}</extra>",
        ))

    fig.update_layout(
        title="Score Stability — avg (± min/max) per run",
        xaxis_title="Run index",
        yaxis=dict(title="Quality score [0–1]", range=[0, 1.1]),
        legend=dict(orientation="h", y=1.14),
        height=360,
    )
    return fig


def panel_multi_run_stats(farg_history: list, base_history: list) -> None:
    """Full statistical comparison across all accumulated runs."""
    from tags import GoalReached

    st.subheader("Multi-Run Statistical Analysis")

    if not farg_history and not base_history:
        st.info(
            "Set **Runs per click ≥ 2** and click Run buttons to accumulate history. "
            "Stats appear here once you have at least one run of each system."
        )
        return

    farg_all_scores = [s for r in farg_history for s in r["farg_scores"].values()]
    base_all_scores = [p.score for r in base_history for p in r.paths]
    farg_st = _score_stats(farg_all_scores)
    base_st = _score_stats(base_all_scores)

    # ── Top KPI row ───────────────────────────────────────────────────────────
    kk1, kk2, kk3, kk4 = st.columns(4)
    kk1.metric("FARG runs", len(farg_history))
    kk2.metric("Baseline runs", len(base_history))
    kk3.metric(
        "FARG total paths",
        len(farg_all_scores),
        f"{len(farg_all_scores)/max(1,len(farg_history)):.1f}/run avg",
    )
    kk4.metric(
        "Baseline total paths",
        len(base_all_scores),
        f"{len(base_all_scores)/max(1,len(base_history)):.1f}/run avg",
    )

    # ── Descriptive stats table ───────────────────────────────────────────────
    st.markdown("**Descriptive Statistics (all paths, all runs)**")

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "—"

    stats_rows = [
        ("Mean score",         farg_st["mean"],     base_st["mean"]),
        ("Median score",       farg_st["median"],   base_st["median"]),
        ("Std deviation (σ)",  farg_st["stdev"],    base_st["stdev"]),
        ("Variance (σ²)",      farg_st["variance"], base_st["variance"]),
        ("Min (abs. lowest)",  farg_st["min"],      base_st["min"]),
        ("Max (abs. highest)", farg_st["max"],      base_st["max"]),
        ("IQR (Q3 − Q1)",      farg_st["iqr"],      base_st["iqr"]),
        ("Mode (±0.05)",       farg_st["mode"],     base_st["mode"]),
        ("N paths",            farg_st["n"],        base_st["n"]),
    ]
    stats_df = pd.DataFrame(
        [(m, _fmt(f), _fmt(b)) for m, f, b in stats_rows],
        columns=["Metric", "FARG", "Baseline"],
    )
    st.dataframe(stats_df.set_index("Metric"), use_container_width=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    cv1, cv2 = st.columns(2)
    with cv1:
        st.plotly_chart(
            chart_score_distribution(farg_history, base_history),
            use_container_width=True,
            key="multi_score_distribution",
        )
    with cv2:
        st.plotly_chart(
            chart_per_run_trend(farg_history, base_history),
            use_container_width=True,
            key="multi_per_run_trend",
        )

    # ── Novel routes ──────────────────────────────────────────────────────────
    if farg_history and base_history:
        st.divider()
        st.markdown("**Novel Routes — FARG directions not explored by Baseline**")
        st.caption(
            "A route is *novel* when its first intermediate city never appeared "
            "in any baseline run's paths."
        )

        baseline_first_cities: set = set()
        for run in base_history:
            for r in run.paths:
                if r.legs:
                    baseline_first_cities.add(r.legs[0].to_loc)

        novel_paths, total_farg_paths = [], 0
        farg_unique_routes: set = set()
        base_unique_routes: set = set()
        for run in farg_history:
            for pc in run["paths"]:
                total_farg_paths += 1
                farg_unique_routes.add(pc.path)
                legs = pc.legs if isinstance(pc, GoalReached) else pc.taggee.legs
                if legs and legs[0].to_loc not in baseline_first_cities:
                    novel_paths.append(pc)
        for run in base_history:
            for r in run.paths:
                base_unique_routes.add(tuple(r.path))

        novel_first_cities = set()
        for pc in novel_paths:
            legs = pc.legs if isinstance(pc, GoalReached) else pc.taggee.legs
            if legs:
                novel_first_cities.add(legs[0].to_loc)

        nr1, nr2, nr3, nr4 = st.columns(4)
        nr1.metric(
            "Novel FARG paths",
            f"{len(novel_paths)} / {total_farg_paths}",
            help="Paths whose first-leg destination was never tried by Baseline.",
        )
        nr2.metric(
            "Novel first-leg cities",
            len(novel_first_cities),
            help=", ".join(sorted(novel_first_cities)) or "none",
        )
        nr3.metric(
            "Unique FARG routes",
            len(farg_unique_routes),
            help="Distinct city-sequence signatures across all FARG runs.",
        )
        nr4.metric(
            "Unique Baseline routes",
            len(base_unique_routes),
            help="Distinct city-sequence signatures across all Baseline runs.",
        )
        if novel_first_cities:
            st.caption(
                f"Novel first-leg directions (FARG only): "
                f"**{', '.join(sorted(novel_first_cities))}**"
            )
        if baseline_first_cities:
            st.caption(
                f"Baseline first-leg directions seen: "
                f"{', '.join(sorted(baseline_first_cities))}"
            )

    # ── Aggregate exploration tree ────────────────────────────────────────────
    if farg_history:
        st.divider()
        st.markdown(
            f"**Aggregate Exploration Tree — {len(farg_history)} FARG run(s) combined**"
        )
        st.caption(
            "All ImCell births across every run merged into one tree. "
            "Wider Sankey links = that edge was taken more often across runs."
        )
        agg = _aggregate_tree(farg_history)
        if agg["births"]:
            max_d = max(agg["depth_counts"].keys(), default=0)
            at1, at2, at3 = st.columns(3)
            at1.metric("Combined ImCell births", len(agg["births"]))
            at2.metric("Distinct first-leg directions", len(agg["first_stop_counts"]))
            at3.metric("Max depth reached", max_d)

            ac1, ac2 = st.columns([2, 3])
            with ac1:
                st.plotly_chart(chart_exploration_depth(agg), use_container_width=True, key="multi_agg_depth")
            with ac2:
                st.plotly_chart(chart_exploration_sankey(agg), use_container_width=True, key="multi_agg_sankey")


def _panel_run_browser() -> None:
    """Browse every saved run; load into the active dashboard or add to comparison."""
    from run_store import list_runs, load_run, delete_run

    saved = list_runs()
    if not saved:
        st.caption("No runs saved yet. Click a Run button and runs will be auto-saved here.")
        return

    farg_saved    = [r for r in saved if r["type"] == "farg"]
    base_saved    = [r for r in saved if r["type"] == "baseline"]
    farg_in_cmp   = {r.get("_filename", "") for r in st.session_state.get("farg_run_history", [])}
    base_in_cmp   = {getattr(r, "_filename", "") for r in st.session_state.get("base_run_history", [])}
    active_farg   = st.session_state.get("_active_farg_file", "")
    active_base   = st.session_state.get("_active_base_file", "")

    col_farg, col_base = st.columns(2)

    with col_farg:
        st.markdown(f"**FARG — {len(farg_saved)} saved run(s)**")
        for rm in farg_saved:
            ts = rm["saved_at"]          # YYYYMMDD_HHMMSS
            label = (
                f"{ts[6:8]}/{ts[4:6]} {ts[9:11]}:{ts[11:13]}  ·  "
                f"{rm['n_paths']} path(s)  ·  avg {rm['avg_score']:.2f}  ·  "
                f"{rm.get('n_ticks', '?')} ticks"
            )
            is_active = rm["filename"] == active_farg
            is_in_cmp = rm["filename"] in farg_in_cmp
            with st.container(border=True):
                hdr_col, del_col = st.columns([8, 1])
                hdr_col.caption(
                    ("📌 **[active]** " if is_active else "") + label
                )
                if del_col.button("🗑", key=f"del_f_{rm['filename']}",
                                  help="Delete this run from disk"):
                    delete_run(rm["path"])
                    # Remove from history if present
                    hist = [r for r in st.session_state.get("farg_run_history", [])
                            if r.get("_filename") != rm["filename"]]
                    st.session_state["farg_run_history"] = hist
                    st.rerun()

                b_view, b_cmp = st.columns(2)
                if b_view.button(
                    "📂 View" + (" ✓" if is_active else ""),
                    key=f"view_f_{rm['filename']}", use_container_width=True,
                    help="Load this run into the main dashboard.",
                ):
                    data = load_run(rm["path"])
                    st.session_state["mc"]          = data["mc"]
                    st.session_state["paths"]       = data["paths"]
                    st.session_state["farg_calls"]  = data["arch_calls"]
                    st.session_state["farg_scores"] = data["farg_scores"]
                    st.session_state["_active_farg_file"] = rm["filename"]
                    st.rerun()

                cmp_label = ("✅ In comparison" if is_in_cmp else "+ Compare")
                if b_cmp.button(
                    cmp_label, key=f"cmp_f_{rm['filename']}", use_container_width=True,
                    disabled=is_in_cmp,
                    help="Add to multi-run comparison panel.",
                ):
                    data = load_run(rm["path"])
                    hist = list(st.session_state.get("farg_run_history", []))
                    entry = {
                        "run_idx":     len(hist) + 1,
                        "mc":          data["mc"],
                        "paths":       data["paths"],
                        "arch_calls":  data["arch_calls"],
                        "farg_scores": data["farg_scores"],
                        "_filename":   rm["filename"],
                    }
                    hist.append(entry)
                    st.session_state["farg_run_history"] = hist
                    st.rerun()

    with col_base:
        st.markdown(f"**Baseline — {len(base_saved)} saved run(s)**")
        for rm in base_saved:
            ts = rm["saved_at"]
            label = (
                f"{ts[6:8]}/{ts[4:6]} {ts[9:11]}:{ts[11:13]}  ·  "
                f"{rm['n_paths']} path(s)  ·  avg {rm['avg_score']:.2f}  ·  "
                f"{rm.get('modes_complete', '?')}/{rm.get('modes_tried', '?')} modes"
            )
            is_active = rm["filename"] == active_base
            is_in_cmp = rm["filename"] in base_in_cmp
            with st.container(border=True):
                hdr_col, del_col = st.columns([8, 1])
                hdr_col.caption(
                    ("📌 **[active]** " if is_active else "") + label
                )
                if del_col.button("🗑", key=f"del_b_{rm['filename']}",
                                  help="Delete this run from disk"):
                    delete_run(rm["path"])
                    hist = [r for r in st.session_state.get("base_run_history", [])
                            if getattr(r, "_filename", "") != rm["filename"]]
                    st.session_state["base_run_history"] = hist
                    st.rerun()

                b_view, b_cmp = st.columns(2)
                if b_view.button(
                    "📂 View" + (" ✓" if is_active else ""),
                    key=f"view_b_{rm['filename']}", use_container_width=True,
                    help="Load this run as the active baseline.",
                ):
                    data = load_run(rm["path"])
                    st.session_state["base_run"] = data["_run"]
                    st.session_state["_active_base_file"] = rm["filename"]
                    st.rerun()

                cmp_label = ("✅ In comparison" if is_in_cmp else "+ Compare")
                if b_cmp.button(
                    cmp_label, key=f"cmp_b_{rm['filename']}", use_container_width=True,
                    disabled=is_in_cmp,
                    help="Add to multi-run comparison panel.",
                ):
                    data = load_run(rm["path"])
                    base_run = data["_run"]
                    base_run._filename = rm["filename"]
                    hist = list(st.session_state.get("base_run_history", []))
                    hist.append(base_run)
                    st.session_state["base_run_history"] = hist
                    st.rerun()


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
        n_runs = st.number_input(
            "Runs per click", min_value=1, max_value=50, value=1, step=1,
            help="How many independent runs to add to history on each button click.",
        )
        run_btn = st.button(
            f"Run FARG × {n_runs}", type="primary", use_container_width=True,
        )
        base_btn = st.button(
            f"Run Baseline × {n_runs}", use_container_width=True,
        )

        if not use_mock and not os.getenv("OPENROUTER_API_KEY"):
            st.warning("OPENROUTER_API_KEY not set — will fall back to mock.")

        st.divider()
        n_farg_hist = len(st.session_state.get("farg_run_history", []))
        n_base_hist = len(st.session_state.get("base_run_history", []))
        st.caption(
            f"History: **{n_farg_hist}** FARG run(s) · **{n_base_hist}** Baseline run(s)"
        )
        cc1, cc2 = st.columns(2)
        clear_farg = cc1.button(
            "Clear FARG", use_container_width=True,
            help="Remove all accumulated FARG runs from history.",
        )
        clear_base = cc2.button(
            "Clear Baseline", use_container_width=True,
            help="Remove all accumulated Baseline runs from history.",
        )
        if clear_farg:
            for k in ("farg_run_history", "mc", "paths", "farg_calls", "farg_scores"):
                st.session_state.pop(k, None)
            st.rerun()
        if clear_base:
            for k in ("base_run_history", "base_run"):
                st.session_state.pop(k, None)
            st.rerun()

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
        from run_store import save_farg_run
        farg_history = list(st.session_state.get("farg_run_history", []))
        prog = st.progress(0.0, text=f"FARG run 1 / {n_runs}…")
        for i in range(n_runs):
            prog.progress(
                (i + 0.5) / n_runs,
                text=f"FARG run {i + 1} / {n_runs}…",
            )
            _mc, _paths, _arch, _scores = run_simulation(
                use_mock=_use_mock, max_ticks=max_ticks
            )
            run_entry = {
                "run_idx":    len(farg_history) + 1,
                "mc":         _mc,
                "paths":      _paths,
                "arch_calls": _arch,
                "farg_scores": _scores,
            }
            saved_path = save_farg_run(run_entry)
            run_entry["_filename"] = saved_path.name
            farg_history.append(run_entry)
            prog.progress(
                (i + 1.0) / n_runs,
                text=f"FARG run {i + 1} / {n_runs} — {len(_paths)} path(s) found",
            )
        prog.empty()
        st.session_state["farg_run_history"] = farg_history
        last = farg_history[-1]
        st.session_state["mc"] = last["mc"]
        st.session_state["paths"] = last["paths"]
        st.session_state["farg_calls"] = last["arch_calls"]
        st.session_state["farg_scores"] = last["farg_scores"]
        st.session_state["_active_farg_file"] = last.get("_filename", "")
        total_paths = sum(len(r["paths"]) for r in farg_history)
        st.success(
            f"FARG: {n_runs} run(s) saved and added to history. "
            f"Total: {len(farg_history)} run(s), {total_paths} paths."
        )

    if base_btn:
        from baseline import run_baseline
        from run_store import save_baseline_run
        from slipnet import MockSlipnet, RealSlipnet, CountingSlipnet
        base_history = list(st.session_state.get("base_run_history", []))
        prog = st.progress(0.0, text=f"Baseline run 1 / {n_runs}…")
        for i in range(n_runs):
            prog.progress(
                (i + 0.5) / n_runs,
                text=f"Baseline run {i + 1} / {n_runs}…",
            )
            inner = MockSlipnet() if _use_mock else RealSlipnet()
            base_counting = CountingSlipnet(inner)
            _base_run = run_baseline(base_counting, START, GOAL)
            _base_run.calls_by_type = dict(base_counting.calls)
            _base_run.llm_calls = base_counting.total
            saved_path = save_baseline_run(_base_run)
            _base_run._filename = saved_path.name   # tag for dedup
            base_history.append(_base_run)
            prog.progress(
                (i + 1.0) / n_runs,
                text=f"Baseline run {i + 1} / {n_runs} — {len(_base_run.paths)} path(s)",
            )
        prog.empty()
        st.session_state["base_run_history"] = base_history
        st.session_state["base_run"] = base_history[-1]
        st.session_state["_active_base_file"] = getattr(base_history[-1], "_filename", "")
        total_base_paths = sum(len(r.paths) for r in base_history)
        st.success(
            f"Baseline: {n_runs} run(s) saved and added to history. "
            f"Total: {len(base_history)} run(s), {total_base_paths} paths."
        )

    mc: MetricsCollector | None = st.session_state.get("mc")
    paths: list = st.session_state.get("paths", [])

    # ── Saved Run Browser (always visible) ────────────────────────────────────
    active_farg = st.session_state.get("_active_farg_file", "")
    active_base = st.session_state.get("_active_base_file", "")
    active_parts = []
    if active_farg:
        active_parts.append(f"FARG: `{active_farg}`")
    if active_base:
        active_parts.append(f"Baseline: `{active_base}`")
    browser_title = (
        "📂 Saved Run History"
        + (f" — viewing {', '.join(active_parts)}" if active_parts else "")
    )
    with st.expander(browser_title, expanded=(mc is None)):
        _panel_run_browser()

    if mc is None:
        st.info(
            "Click **Run FARG** or **Run Baseline** in the sidebar to start a new run, "
            "or load a saved run from the history above."
        )
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
        st.plotly_chart(chart_activation_timeline(mc, top_n), use_container_width=True, key="farg_activation_timeline")
    with c2:
        st.plotly_chart(chart_workspace_composition(mc), use_container_width=True, key="farg_workspace_composition")

    # ── Row 2: Act Probability + Temperature ──────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(chart_act_probability(mc), use_container_width=True, key="farg_act_probability")
    with c4:
        st.plotly_chart(chart_temperature(mc), use_container_width=True, key="farg_temperature")

    # ── Row 3: Competition Graph ───────────────────────────────────────────────
    st.subheader("Competition Graph (Network Topology)")
    tick_idx = st.slider(
        "Select tick to visualise",
        0, ticks_available - 1,
        min(ticks_available - 1, 20),
        key="graph_tick",
    )
    st.plotly_chart(chart_competition_graph(mc, tick_idx), use_container_width=True, key="farg_competition_graph")

    # ── Row 4: Funnel + Async Bottleneck ──────────────────────────────────────
    c5, c6 = st.columns([2, 3])
    with c5:
        st.plotly_chart(chart_perturbation_funnel(mc), use_container_width=True, key="farg_perturbation_funnel")
    with c6:
        st.plotly_chart(chart_pending_llm(mc), use_container_width=True, key="farg_pending_llm")

    # ── Row 4b: Exploration Tree ──────────────────────────────────────────────
    st.divider()
    st.subheader("Exploration Tree — ImCell Lineage")
    st.caption(
        "Tracks every ImCell from birth. "
        "Shows how many distinct first-leg directions FARG explored, "
        "how many survived to deeper legs, and which reached the goal."
    )

    tree = mc.exploration_tree()
    if tree["births"]:
        max_depth = max(tree["depth_counts"].keys()) if tree["depth_counts"] else 0
        n_unique_first = len(tree["first_stop_counts"])

        tk1, tk2, tk3, tk4 = st.columns(4)
        tk1.metric(
            "Distinct first-leg directions",
            n_unique_first,
            help="Unique first intermediate cities across all ImCells (= exploration breadth).",
        )
        tk2.metric(
            "ImCells born total",
            len(tree["births"]),
            help="Every speculative route fragment created during the run.",
        )
        tk3.metric(
            "Max depth reached",
            max_depth,
            help="Longest ImCell chain (number of legs) seen in the workspace.",
        )
        tk4.metric(
            "Paths reaching goal",
            len(paths),
            help="Canvas chains that committed all the way to the goal location.",
        )

        c_depth, c_sankey = st.columns([2, 3])
        with c_depth:
            st.plotly_chart(chart_exploration_depth(tree), use_container_width=True, key="farg_exploration_depth")
        with c_sankey:
            st.plotly_chart(chart_exploration_sankey(tree), use_container_width=True, key="farg_exploration_sankey")

        # Per-first-stop breakdown table
        with st.expander("First-leg breakdown table"):
            all_first_stops = sorted(tree["first_stop_counts"].keys())
            all_depths = sorted(tree["depth_counts"].keys())
            from collections import defaultdict
            depth_first: dict = defaultdict(lambda: defaultdict(int))
            goal_by_first: dict = defaultdict(int)
            for rec in tree["births"]:
                first = rec.legs[0].to_loc
                depth_first[rec.depth][first] += 1
                if rec.legs[-1].to_loc == GOAL:
                    goal_by_first[first] += 1
            table_rows = []
            for first in all_first_stops:
                row = {"First stop": first}
                for d in all_depths:
                    row[f"Depth {d}"] = depth_first[d].get(first, 0)
                row["Reached goal"] = goal_by_first.get(first, 0)
                table_rows.append(row)
            st.dataframe(pd.DataFrame(table_rows).set_index("First stop"),
                         use_container_width=True)
    else:
        st.info("Run the FARG simulation to see the exploration tree.")

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

    # ── Row 7: Multi-run statistical analysis ────────────────────────────────
    farg_history = st.session_state.get("farg_run_history", [])
    base_history = st.session_state.get("base_run_history", [])
    if farg_history or base_history:
        st.divider()
        panel_multi_run_stats(farg_history, base_history)

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
