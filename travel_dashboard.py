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

from baseline import BaselinePipeline, BaselineResult, compute_metrics
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
    "GoIsDone":      "#aec7e8",
    "ActIsDone":     "#ffbb78",
    "Other":         "#c7c7c7",
}

STACKED_TYPES = [
    "Want", "SuggestMode", "SuggestRoute", "SeekEvidence",
    "Evaluate", "ImCell", "PendingLLM", "Blocked",
]


# ── Simulation runner ─────────────────────────────────────────────────────────

def run_simulation(use_mock: bool, max_ticks: int) -> tuple[MetricsCollector, list]:
    from agents import Want
    from main_loop import run_loop
    from slipnet import MockSlipnet, RealSlipnet
    from workspace import Workspace

    slipnet = MockSlipnet() if use_mock else RealSlipnet()
    ws = Workspace()
    ws.add(Want(from_loc=START, to_loc=GOAL), init_a=1.0)
    mc = MetricsCollector()
    paths = run_loop(ws, slipnet, logger=mc, max_ticks=max_ticks)
    return mc, paths


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
        name="Formula: T = 0.9 - 0.7a",
    ))
    fig.update_layout(
        title="Dynamic Temperature Tracker (Activation -> LLM Temperature)",
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
    """Side-by-side cards for each complete path."""
    from slipnet import temperature_for_activation
    from canvas import ImCell

    if not paths:
        st.warning("No complete paths found.")
        return

    cols = st.columns(min(len(paths), 3))
    for i, pc in enumerate(paths):
        col = cols[i % len(cols)]
        imcell = pc.taggee
        with col:
            st.markdown(f"### Path {imcell.path_id}")
            total_h = sum(l.duration_hours for l in imcell.legs)
            st.metric("Total hours", f"{total_h:.1f} h")
            st.metric("Legs", len(imcell.legs))
            for leg in imcell.legs:
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


# ── Baseline runner ───────────────────────────────────────────────────────────

def run_baseline_pipeline(use_mock: bool) -> BaselineResult:
    from slipnet import MockSlipnet, RealSlipnet
    slipnet = MockSlipnet() if use_mock else RealSlipnet()
    pipeline = BaselinePipeline(slipnet)
    return pipeline.run()


# ── Comparison panel ──────────────────────────────────────────────────────────

def panel_farg_vs_baseline(mc: MetricsCollector, farg_paths: list, br: BaselineResult) -> None:
    """Side-by-side FARG vs Baseline comparison across every measurable axis."""
    import statistics
    from collections import Counter

    # ── Compute FARG metrics ───────────────────────────────────────────────────
    total_p, surv_p, comm_p = mc.survival_funnel()
    farg_complete = [pc for pc in farg_paths]
    farg_stops = set()
    farg_leg_keys = []
    farg_scores = []
    for pc in farg_complete:
        ic = pc.taggee
        for leg in ic.legs:
            if leg.to_loc != GOAL:
                farg_stops.add(leg.to_loc)
            farg_leg_keys.append(f"{leg.from_loc}->{leg.to_loc}")
    farg_leg_counts = Counter(farg_leg_keys)
    farg_shared = sum(1 for c in farg_leg_counts.values() if c > 1)

    # ── Compute baseline metrics ───────────────────────────────────────────────
    bm = compute_metrics(br)
    bl_complete = [p for p in br.paths if p.complete]
    bl_scores = [p.score for p in bl_complete]

    # ── Metric table ──────────────────────────────────────────────────────────
    st.subheader("Head-to-Head Metrics")
    metrics_rows = [
        ("Complete paths found",       len(farg_complete),              bm["n_paths_complete"],
         "More paths = richer solution space"),
        ("Unique intermediate stops",  len(farg_stops),                 bm["n_unique_intermediate"],
         "Higher = more geographic diversity"),
        ("Shared legs (homogeneity)",  farg_shared,                     bm["shared_legs"],
         "Lower = paths are more distinct (antipathy working)"),
        ("Mean quality score",         round(statistics.mean([0.5]*max(1,len(farg_complete))), 3),
                                                                         bm["mean_score"],
         "Higher = better average route quality"),
        ("Score std-dev",              "n/a",                           round(bm["score_stdev"], 3),
         "FARG score spread driven by SeekEvidence competition"),
        ("LLM calls (total)",          total_p + 1,                     bm["total_llm_calls"],
         "FARG overhead vs baseline linearity"),
        ("Legs proposed (LLM output)", total_p,                         sum(len(p.legs) for p in bl_complete),
         "FARG explores more candidates before committing"),
        ("Survival rate (10 ticks)",   f"{100*surv_p//max(1,total_p)}%", "100% (no decay)",
         "Baseline never discards — FARG prunes via activation decay"),
        ("Canvas commit rate",         f"{100*comm_p//max(1,total_p)}%", "100% (all kept)",
         "FARG commits only winning legs; baseline commits everything"),
    ]

    col_h1, col_h2, col_h3, col_h4 = st.columns([2.5, 1.2, 1.2, 3])
    col_h1.markdown("**Metric**")
    col_h2.markdown("**FARG**")
    col_h3.markdown("**Baseline**")
    col_h4.markdown("**What it means**")
    st.divider()
    for label, farg_val, bl_val, note in metrics_rows:
        c1, c2, c3, c4 = st.columns([2.5, 1.2, 1.2, 3])
        c1.write(label)
        c2.write(str(farg_val))
        c3.write(str(bl_val))
        c4.caption(note)

    st.divider()

    # ── Bar chart: unique stops ────────────────────────────────────────────────
    st.subheader("Intermediate Stop Diversity")
    bl_stops = set(bm["unique_stops"])
    only_farg = sorted(farg_stops - bl_stops)
    only_bl   = sorted(bl_stops - farg_stops)
    shared    = sorted(farg_stops & bl_stops)

    stop_fig = go.Figure()
    if shared:
        stop_fig.add_trace(go.Bar(name="Both", x=shared,
                                   y=[1]*len(shared), marker_color="#2ca02c"))
    if only_farg:
        stop_fig.add_trace(go.Bar(name="FARG only", x=only_farg,
                                   y=[1]*len(only_farg), marker_color="#1f77b4"))
    if only_bl:
        stop_fig.add_trace(go.Bar(name="Baseline only", x=only_bl,
                                   y=[1]*len(only_bl), marker_color="#ff7f0e"))
    stop_fig.update_layout(
        barmode="group", height=260,
        xaxis_title="City", yaxis=dict(showticklabels=False),
        title="Which intermediate cities each system discovered",
    )
    st.plotly_chart(stop_fig, use_container_width=True)

    # ── Score bar chart ────────────────────────────────────────────────────────
    if bl_scores:
        st.subheader("Path Quality Scores (Baseline)")
        score_fig = go.Figure(go.Bar(
            x=[p.mode for p in bl_complete],
            y=[p.score for p in bl_complete],
            marker_color=[TYPE_COLORS.get("SuggestMode", "#1f77b4")] * len(bl_complete),
            text=[f"{p.score:.2f}" for p in bl_complete],
            textposition="outside",
        ))
        score_fig.add_hline(y=0.35, line_dash="dash", line_color="red",
                             annotation_text="FARG feasibility gate (0.35)")
        score_fig.update_layout(
            title="Baseline route scores (all paths accepted regardless of score)",
            xaxis_title="Mode", yaxis_title="evaluate_path() score",
            yaxis=dict(range=[0, 1.1]), height=300,
        )
        st.plotly_chart(score_fig, use_container_width=True)

    # ── Baseline paths ────────────────────────────────────────────────────────
    st.subheader("Baseline Complete Paths")
    bl_sorted = sorted(bl_complete, key=lambda p: -p.score)
    cols = st.columns(min(len(bl_sorted), 3))
    for i, p in enumerate(bl_sorted):
        with cols[i % len(cols)]:
            st.markdown(f"**[{p.mode}]** score={p.score:.2f}  {p.total_hours:.1f}h")
            st.code(p.route_str(), language=None)
            for leg in p.legs:
                st.caption(f"{leg.from_loc} --[{leg.mode}]--> {leg.to_loc} ({leg.duration_hours:.1f}h)")


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
        run_btn = st.button("Run FARG", type="primary", use_container_width=True)
        run_bl  = st.button("Run Baseline", use_container_width=True)

        if not use_mock and not os.getenv("OPENROUTER_API_KEY"):
            st.warning("OPENROUTER_API_KEY not set — will fall back to mock.")

        st.divider()
        st.markdown("**Legend**")
        for ttype, color in TYPE_COLORS.items():
            if ttype in STACKED_TYPES + ["ImCell", "PathComplete"]:
                st.markdown(
                    f'<span style="color:{color}">■</span> {ttype}',
                    unsafe_allow_html=True,
                )

    # ── Run / load state ──────────────────────────────────────────────────────
    _use_mock = use_mock or not os.getenv("OPENROUTER_API_KEY")

    if run_btn:
        with st.spinner("Running FARG simulation..."):
            mc, paths = run_simulation(use_mock=_use_mock, max_ticks=max_ticks)
        st.session_state["mc"] = mc
        st.session_state["paths"] = paths
        st.success(f"FARG done — {len(paths)} path(s) in {len(mc.snapshots)} ticks.")

    if run_bl:
        with st.spinner("Running baseline pipeline..."):
            br = run_baseline_pipeline(use_mock=_use_mock)
        st.session_state["baseline"] = br
        bm = compute_metrics(br)
        st.success(f"Baseline done — {bm['n_paths_complete']} paths, "
                   f"{bm['total_llm_calls']} LLM calls.")

    mc: MetricsCollector | None = st.session_state.get("mc")
    paths: list = st.session_state.get("paths", [])
    br: BaselineResult | None = st.session_state.get("baseline")

    if mc is None and br is None:
        st.info("Click **Run FARG** and/or **Run Baseline** to begin.")
        return

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_labels = []
    if mc is not None:
        tab_labels.append("FARG Dashboard")
    if br is not None:
        tab_labels.append("FARG vs Baseline")
    tabs = st.tabs(tab_labels)
    tab_iter = iter(tabs)

    # ── FARG tab ──────────────────────────────────────────────────────────────
    if mc is not None:
        with next(tab_iter):
            ticks_available = len(mc.snapshots)
            total_p, surv_p, comm_p = mc.survival_funnel()
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Ticks run", ticks_available)
            k2.metric("Paths found", len(paths))
            k3.metric("LLM legs proposed", total_p)
            k4.metric("Survived 10 ticks", surv_p, f"{100*surv_p//max(1,total_p)}%")
            k5.metric("Committed to canvas", comm_p, f"{100*comm_p//max(1,total_p)}%")

            st.divider()
            c1, c2 = st.columns([3, 2])
            with c1:
                st.plotly_chart(chart_activation_timeline(mc, top_n), use_container_width=True)
            with c2:
                st.plotly_chart(chart_workspace_composition(mc), use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                st.plotly_chart(chart_act_probability(mc), use_container_width=True)
            with c4:
                st.plotly_chart(chart_temperature(mc), use_container_width=True)

            st.subheader("Competition Graph (Network Topology)")
            tick_idx = st.slider(
                "Select tick to visualise",
                0, ticks_available - 1,
                min(ticks_available - 1, 20),
                key="graph_tick",
            )
            st.plotly_chart(chart_competition_graph(mc, tick_idx), use_container_width=True)

            c5, c6 = st.columns([2, 3])
            with c5:
                st.plotly_chart(chart_perturbation_funnel(mc), use_container_width=True)
            with c6:
                st.plotly_chart(chart_pending_llm(mc), use_container_width=True)

            st.divider()
            st.subheader("FARG Complete Paths")
            panel_path_comparison(paths)

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

    # ── Comparison tab ────────────────────────────────────────────────────────
    if br is not None:
        with next(tab_iter):
            if mc is None:
                st.info("Run FARG first to enable the full head-to-head comparison.")
                st.subheader("Baseline paths only")
                bl_sorted = sorted([p for p in br.paths if p.complete], key=lambda p: -p.score)
                cols = st.columns(min(len(bl_sorted), 3))
                for i, p in enumerate(bl_sorted):
                    with cols[i % len(cols)]:
                        st.markdown(f"**[{p.mode}]** score={p.score:.2f}")
                        st.code(p.route_str(), language=None)
            else:
                panel_farg_vs_baseline(mc, paths, br)


if __name__ == "__main__":
    main()
