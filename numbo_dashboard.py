# numbo_dashboard.py
# Streamlit dashboard for visualizing every aspect of Numbo's cognitive process.
# Run with: streamlit run numbo_dashboard.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from copy import deepcopy

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np

from FARGish2 import (
    FARGModel, SeqCanvas, SeqState, CellRef, ImCell,
    Tag, TaggeeTag, NoGo, NoAct, GoIsDone, ActIsDone, Blocked,
    Agent, Elem, ElemInWS, BaseDetector, Detector, CanGo, CanAct,
    CallGo, CallAct, StateDelta
)
from Numbo1 import (
    Numbo, Want, Consume, ArithDelta, SolvedNumble, GettingCloser,
    Increase, Decrease, After
)
from Slipnet import FeatureWrapper


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class ElemRecord:
    name: str
    elem_type: str
    activation: float
    activation_delta: float
    degree: int
    builder: str
    tob: int
    tags: List[str]
    is_sleeping: bool
    operator: Optional[str] = None
    operands: Optional[tuple] = None
    source_addr: Optional[str] = None
    computed_result: Optional[int] = None


@dataclass
class CanvasRecord:
    addr: int
    avails: tuple
    last_move: str


@dataclass
class ImCellRecord:
    name: str
    avails: tuple
    last_move: str
    getting_closer_weight: float
    builder: str
    activation: float


@dataclass
class FlowRecord:
    from_elem: str
    to_elem: str
    amount: float


@dataclass
class SlipnetQueryRecord:
    features_in: Dict[str, float]
    direction: str
    target: Optional[int]
    returned_agents: List[str]


@dataclass
class StepSnapshot:
    t: int
    phase: str
    action_mode: str
    selected_agent: str
    selected_agent_type: str
    sum_a: float
    ws_size: int
    num_agents: int
    num_imcells: int
    num_tags: int
    elements: List[ElemRecord]
    canvas_states: List[CanvasRecord]
    imcells: List[ImCellRecord]
    flows: List[FlowRecord]
    graph_nodes: List[dict]
    graph_edges: List[dict]
    slipnet_queries: List[SlipnetQueryRecord]
    sleeping: List[dict]
    event_log_line: str
    solved: bool
    solution: Optional[str]
    target_in_canvas: bool = False   # derived: target value found in any committed canvas state
    type_counts: Dict[str, int] = field(default_factory=dict)
    problem_avails: tuple = field(default_factory=tuple)
    problem_target: Optional[int] = None


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def elem_type_name(elem) -> str:
    if isinstance(elem, Want):
        return "Want"
    if isinstance(elem, Consume):
        return "Consume"
    if isinstance(elem, ImCell):
        return "ImCell"
    if isinstance(elem, CellRef):
        return "CellRef"
    if isinstance(elem, GettingCloser):
        return "GettingCloser"
    if isinstance(elem, (GoIsDone, ActIsDone, Blocked, NoGo, NoAct)):
        return "Tag"
    if isinstance(elem, Tag):
        return "Tag"
    if isinstance(elem, BaseDetector):
        return "Detector"
    if isinstance(elem, SeqCanvas):
        return "Canvas"
    return type(elem).__name__


def tags_for_elem(fm, elem) -> List[str]:
    result = []
    for e in fm.ws.keys():
        if isinstance(e, TaggeeTag) and e.taggee == elem:
            result.append(type(e).__name__)
    return result


def gc_weight_for(fm, elem) -> float:
    for e in fm.ws.keys():
        if isinstance(e, GettingCloser) and e.taggee == elem:
            return e.weight or 0.0
    return 0.0


def safe_str(obj) -> str:
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def phase_for_t(t: int) -> str:
    if t < 20:
        return "Deliberate"
    elif t < 40:
        return "Transition"
    else:
        return "Commit"


# ═══════════════════════════════════════════════════════════════
# Snapshot Capture
# ═══════════════════════════════════════════════════════════════

def capture_snapshot(
    fm: 'ObservableNumbo',
    prev: Optional[StepSnapshot],
    solved: bool,
    solution: Optional[str]
) -> StepSnapshot:

    t = fm.t
    phase = phase_for_t(t)
    selected_agent = safe_str(fm._last_selected_agent) if fm._last_selected_agent else "none"
    selected_agent_type = elem_type_name(fm._last_selected_agent) if fm._last_selected_agent else "none"
    action_mode = fm._last_action_mode

    # Build previous activation lookup
    prev_activations: Dict[str, float] = {}
    if prev:
        for er in prev.elements:
            prev_activations[er.name] = er.activation

    # All workspace elements
    elements = []
    type_counts: Dict[str, int] = {}
    for elem, eiws in list(fm.ws.items()):
        etype = elem_type_name(elem)
        type_counts[etype] = type_counts.get(etype, 0) + 1
        activation = fm.a(elem)
        name = safe_str(elem)
        prev_a = prev_activations.get(name, activation)
        tags = tags_for_elem(fm, elem)
        sleeping = elem in fm.sleeping

        operator_s = None
        operands = None
        source_addr = None
        computed_result = None
        if isinstance(elem, Consume):
            operator_s = safe_str(elem.operator) if elem.operator else None
            operands = elem.operands
            source_addr = safe_str(elem.source) if elem.source else None
            if elem.operator and elem.operands:
                try:
                    computed_result = elem.operator.call(*elem.operands)
                except Exception:
                    pass

        try:
            deg = fm.degree(elem)
        except Exception:
            deg = 0

        elements.append(ElemRecord(
            name=name,
            elem_type=etype,
            activation=activation,
            activation_delta=activation - prev_a,
            degree=deg,
            builder=safe_str(eiws.builder),
            tob=eiws.tob,
            tags=tags,
            is_sleeping=sleeping,
            operator=operator_s,
            operands=operands,
            source_addr=source_addr,
            computed_result=computed_result,
        ))

    # Canvas states
    canvas_states = []
    canvas = None
    for elem in fm.ws.keys():
        if isinstance(elem, SeqCanvas):
            canvas = elem
            break
    if canvas:
        for addr, state in enumerate(canvas.states):
            if state is None:
                continue
            avails = state.avails or ()
            last_move = ""
            if state.last_move and hasattr(state.last_move, 'seq_str'):
                last_move = state.last_move.seq_str()
            elif state.last_move:
                last_move = safe_str(state.last_move)
            canvas_states.append(CanvasRecord(addr=addr, avails=avails, last_move=last_move))

    # ImCells
    imcells = []
    for elem in fm.ws.keys():
        if isinstance(elem, ImCell):
            contents = elem.contents
            if contents is None:
                continue
            avails = contents.avails if hasattr(contents, 'avails') else ()
            last_move = ""
            if hasattr(contents, 'last_move') and contents.last_move:
                lm = contents.last_move
                if hasattr(lm, 'seq_str'):
                    last_move = lm.seq_str()
                else:
                    last_move = safe_str(lm)
            gc_weight = gc_weight_for(fm, elem)
            builder = safe_str(fm.ws[elem].builder) if elem in fm.ws else "?"
            imcells.append(ImCellRecord(
                name=safe_str(elem),
                avails=avails or (),
                last_move=last_move,
                getting_closer_weight=gc_weight,
                builder=builder,
                activation=fm.a(elem),
            ))

    # Activation flows
    flows = []
    try:
        flow_dict = fm.activation_g.propagator.flows._fromto
        for (fromnode, tonode), amount in flow_dict.items():
            if abs(amount) > 0.001:
                flows.append(FlowRecord(
                    from_elem=safe_str(fromnode),
                    to_elem=safe_str(tonode),
                    amount=amount,
                ))
        flows.sort(key=lambda f: abs(f.amount), reverse=True)
    except Exception:
        pass

    # Support graph
    graph_nodes = []
    graph_edges = []
    try:
        for node in fm.activation_g.nodes:
            if node in fm.ws:
                graph_nodes.append({
                    'name': safe_str(node),
                    'type': elem_type_name(node),
                    'activation': fm.a(node),
                })
        for u, v, data in fm.activation_g.edges(data=True):
            weight = data.get('weight', 0.0)
            graph_edges.append({
                'from': safe_str(u),
                'to': safe_str(v),
                'weight': weight,
                'positive': weight > 0,
            })
    except Exception:
        pass

    # Sleeping
    sleeping_list = []
    for elem, wake_t in fm.sleeping.items():
        sleeping_list.append({
            'name': safe_str(elem),
            'wakes_at': wake_t,
        })

    # Counts
    num_agents = sum(1 for e in fm.ws if isinstance(e, Agent))
    num_imcells = sum(1 for e in fm.ws if isinstance(e, ImCell))
    num_tags = sum(1 for e in fm.ws if isinstance(e, Tag))

    # Event log line
    event = _build_event_line(
        t, phase, action_mode, selected_agent, selected_agent_type,
        fm, canvas_states, solved, solution
    )

    # Problem info
    problem_avails = fm._problem_avails or ()
    problem_target = fm._problem_target

    # Derived: check if target appears in any committed canvas state
    target_in_canvas = False
    if problem_target is not None and canvas:
        for state in canvas.states:
            if state is not None and hasattr(state, 'avails') and state.avails:
                if problem_target in state.avails:
                    target_in_canvas = True
                    break

    # If target reached but SolvedNumble wasn't formally raised, derive solution string
    if target_in_canvas and not solved:
        try:
            solution = canvas.as_solution()
            solved = True
        except Exception:
            pass

    return StepSnapshot(
        t=t,
        phase=phase,
        action_mode=action_mode,
        selected_agent=selected_agent,
        selected_agent_type=selected_agent_type,
        sum_a=fm.sum_a(),
        ws_size=len(fm.ws),
        num_agents=num_agents,
        num_imcells=num_imcells,
        num_tags=num_tags,
        elements=sorted(elements, key=lambda e: e.activation, reverse=True),
        canvas_states=canvas_states,
        imcells=sorted(imcells, key=lambda i: i.activation, reverse=True),
        flows=flows[:20],
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        slipnet_queries=list(fm._last_slipnet_queries),
        sleeping=sleeping_list,
        event_log_line=event,
        solved=solved,
        solution=solution,
        target_in_canvas=target_in_canvas,
        type_counts=type_counts,
        problem_avails=problem_avails,
        problem_target=problem_target,
    )


def _build_event_line(t, phase, mode, agent_name, agent_type, fm, canvas_states, solved, solution) -> str:
    parts = [f"t={t} [{phase}]"]
    if agent_name and agent_name != "none":
        parts.append(f"{agent_name}.{mode}()")
    if solved:
        parts.append(f"✅ SOLVED: {solution}")
    elif canvas_states:
        n = len(canvas_states)
        parts.append(f"canvas has {n} committed state(s)")
    return "  ".join(parts)


# ═══════════════════════════════════════════════════════════════
# ObservableNumbo
# ═══════════════════════════════════════════════════════════════

class ObservableNumbo(Numbo):
    """Subclass of Numbo that records a full snapshot after every timestep."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history: List[StepSnapshot] = []
        self._last_selected_agent = None
        self._last_action_mode = 'none'
        self._last_slipnet_queries: List[SlipnetQueryRecord] = []
        self._problem_avails: Optional[tuple] = None
        self._problem_target: Optional[int] = None

    def choose_agent_by_activation(self, pred):
        agent = super().choose_agent_by_activation(pred)
        if agent is not None:
            self._last_selected_agent = agent
            if pred is CanAct:
                self._last_action_mode = 'act'
            elif pred is CanGo:
                self._last_action_mode = 'go'
        return agent

    def pulse_slipnet(self, activations_in, type=None, k=20, num_get=1, filter=None):
        kwargs = dict(type=type, k=k, num_get=num_get)
        if filter is not None:
            kwargs['filter'] = filter
        result = super().pulse_slipnet(activations_in, **kwargs)

        direction = ''
        target = None
        for feat in activations_in:
            if isinstance(feat, Increase):
                direction = 'Increase'
            elif isinstance(feat, Decrease):
                direction = 'Decrease'
            if isinstance(feat, After) and hasattr(feat, 'feature'):
                target = feat.feature

        record = SlipnetQueryRecord(
            features_in={safe_str(k): v for k, v in activations_in.items()},
            direction=direction,
            target=target,
            returned_agents=[safe_str(a) for a in result],
        )
        self._last_slipnet_queries.append(record)
        return result

    def do_timestep(self, ag=None, num=1, act=False, until=None):
        if until is None:
            until = self.t + num
        while self.t < until:
            self._last_selected_agent = None
            self._last_action_mode = 'none'
            self._last_slipnet_queries = []
            solved = False
            solution = None
            try:
                FARGModel.do_timestep(self, ag=ag, num=1)
            except SolvedNumble as exc:
                solved = True
                solution = str(exc)
                self._record_snapshot(solved=True, solution=solution)
                raise
            self._record_snapshot()

    def _record_snapshot(self, solved=False, solution=None):
        prev = self.history[-1] if self.history else None
        snap = capture_snapshot(self, prev, solved, solution)
        self.history.append(snap)


# ═══════════════════════════════════════════════════════════════
# Run function
# ═══════════════════════════════════════════════════════════════

def run_numbo(seed=None, num_timesteps=40) -> ObservableNumbo:
    """Run Numbo with the standard 4,5,6→15 problem and return the model."""
    fm = ObservableNumbo(seed=seed)
    ca = fm.build(SeqCanvas([SeqState((4, 5, 6), None)]))
    wa = fm.build(Want(15, canvas=ca, addr=0))
    fm._problem_avails = (4, 5, 6)
    fm._problem_target = 15
    try:
        fm.do_timestep(num=num_timesteps)
    except SolvedNumble:
        pass
    return fm


# ═══════════════════════════════════════════════════════════════
# Color scheme
# ═══════════════════════════════════════════════════════════════

TYPE_COLORS = {
    "Want":         "#4C72B0",
    "Consume":      "#DD8452",
    "ImCell":       "#55A868",
    "CellRef":      "#937860",
    "Canvas":       "#8172B3",
    "Detector":     "#9B9B9B",
    "GettingCloser":"#64B5CD",
    "Tag":          "#DA8BC3",
    "Other":        "#CCCCCC",
}

PHASE_COLORS = {
    "Deliberate": "#4C72B0",
    "Transition": "#DD8452",
    "Commit":     "#55A868",
}

MODE_COLORS = {
    "go":   "#4CAFEA",
    "act":  "#F28B30",
    "none": "#AAAAAA",
}


# ═══════════════════════════════════════════════════════════════
# Dashboard Panels
# ═══════════════════════════════════════════════════════════════

def render_header(fm: ObservableNumbo):
    """Panel 0: Static run header."""
    final = fm.history[-1] if fm.history else None
    solved = (final.solved or final.target_in_canvas) if final else False
    solution = final.solution if final else None

    avails_str = ", ".join(str(n) for n in (fm._problem_avails or ()))
    target_str = str(fm._problem_target or "?")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Problem", f"{avails_str}  →  {target_str}")
    col2.metric("Seed", str(fm.seed))
    if solved:
        col3.markdown("**Status:** 🟢 SOLVED")
        col4.markdown(f"**Solution:** `{solution}`")
    else:
        col3.markdown("**Status:** 🟡 Not solved")
        col4.markdown("**Solution:** —")


def render_navigator(snap: StepSnapshot, total: int) -> int:
    """Panel 1: Timestep slider and metrics. Returns selected index."""
    st.markdown("---")
    idx = st.slider("Timestep", min_value=0, max_value=total - 1,
                    value=total - 1, key="ts_slider")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("t", snap.t)

    phase_color = PHASE_COLORS.get(snap.phase, "#888")
    col2.markdown(f"**Phase**<br><span style='color:{phase_color};font-weight:bold'>{snap.phase}</span>",
                  unsafe_allow_html=True)

    mode_color = MODE_COLORS.get(snap.action_mode, "#888")
    col3.markdown(f"**Mode**<br><span style='color:{mode_color};font-weight:bold'>.{snap.action_mode}()</span>",
                  unsafe_allow_html=True)

    col4.markdown(f"**Selected**<br>`{snap.selected_agent[:40]}`", unsafe_allow_html=True)
    col5.metric("Σ Activation", f"{snap.sum_a:.3f}")
    col6.metric("Workspace", snap.ws_size)
    return idx


def render_canvas(snap: StepSnapshot):
    """Panel 2: The committed solution path."""
    st.subheader("Canvas — Committed Solution Path")
    if not snap.canvas_states:
        st.info("No committed states yet.")
        return

    cols = st.columns(len(snap.canvas_states) + 1)
    # Initial state (canvas_states[0] has the initial avails)
    for i, cr in enumerate(snap.canvas_states):
        with cols[i]:
            avails_str = ", ".join(str(n) for n in cr.avails)
            label = cr.last_move if cr.last_move else "(start)"
            target = snap.problem_target
            is_solved = target is not None and target is not None and target in cr.avails
            border_color = "#55A868" if is_solved else "#888"
            bg = "#e8f5e9" if is_solved else "#f8f9fa"
            st.markdown(
                f"<div style='border:2px solid {border_color};border-radius:8px;"
                f"padding:8px;background:{bg};text-align:center'>"
                f"<b>{label}</b><br>"
                f"<span style='font-size:1.2em'>[{avails_str}]</span>"
                f"{'<br>🎯 TARGET!' if is_solved else ''}"
                f"</div>",
                unsafe_allow_html=True
            )
        if i < len(snap.canvas_states) - 1:
            cols[i + 1].markdown("<div style='text-align:center;font-size:2em;padding-top:20px'>→</div>",
                                 unsafe_allow_html=True)


def render_imcells(snap: StepSnapshot):
    """Panel 3: Hypothetical states (ImCells)."""
    st.subheader("Hypothetical States (ImCells) — What the Model is Considering")
    if not snap.imcells:
        st.info("No hypothetical states yet.")
        return

    rows = []
    for ic in snap.imcells:
        rows.append({
            "ImCell": ic.name,
            "Arithmetic": ic.last_move,
            "Avails After": str(ic.avails),
            "GettingCloser?": "✓" if ic.getting_closer_weight > 0.001 else "✗",
            "Weight": round(ic.getting_closer_weight, 3),
            "Activation": round(ic.activation, 4),
            "Builder": ic.builder,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True,
                 column_config={
                     "Weight": st.column_config.ProgressColumn("Weight", min_value=0, max_value=1),
                 })

    if snap.problem_target:
        st.caption(
            f"GettingCloser weight = (|target – closest operand| – |target – result|) "
            f"/ |target – closest operand|.  "
            f"Target = {snap.problem_target}"
        )


def render_workspace_table(snap: StepSnapshot):
    """Panel 4: Full workspace census."""
    st.subheader("Workspace — Every Element in the Model's Mind")

    type_filter = st.multiselect(
        "Filter by type",
        options=list(set(e.elem_type for e in snap.elements)),
        default=list(set(e.elem_type for e in snap.elements)),
        key="ws_type_filter"
    )

    rows = []
    for er in snap.elements:
        if er.elem_type not in type_filter:
            continue
        delta_str = f"+{er.activation_delta:.3f}" if er.activation_delta >= 0 else f"{er.activation_delta:.3f}"
        tags_str = ", ".join(er.tags) if er.tags else "—"
        details = ""
        if er.elem_type == "Consume":
            result_str = f"={er.computed_result}" if er.computed_result is not None else ""
            ops_str = f"{er.operator} {', '.join(str(o) for o in (er.operands or ()))}{result_str}"
            details = f"{ops_str}  src={er.source_addr}"
        elif er.elem_type == "Want":
            details = f"target={snap.problem_target}"
        elif er.elem_type == "ImCell":
            details = f"avails={er.name}"
        sleep_str = ""
        if er.is_sleeping:
            for s in snap.sleeping:
                if s['name'] == er.name:
                    sleep_str = f"💤 t={s['wakes_at']}"
                    break

        rows.append({
            "Element": er.name[:60],
            "Type": er.elem_type,
            "Activation": round(er.activation, 4),
            "Δa": delta_str,
            "Degree": er.degree,
            "Age (tob)": er.tob,
            "Tags": tags_str,
            "Builder": (er.builder or "—")[:30],
            "Details": details[:60],
            "Sleep": sleep_str,
        })

    if not rows:
        st.info("No elements match the filter.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True,
                 column_config={
                     "Activation": st.column_config.NumberColumn(format="%.4f"),
                 })


def render_activation_charts(history: List[StepSnapshot], current_idx: int):
    """Panel 5: Activation bar chart (current step) + line chart (all time)."""
    snap = history[current_idx]

    col_left, col_right = st.columns(2)

    # Left: bar chart of top 12 by activation at current step
    with col_left:
        st.subheader("Activation Ranking (this step)")
        top_elems = snap.elements[:12]
        if top_elems:
            names = [e.name[:35] for e in top_elems]
            acts = [e.activation for e in top_elems]
            types = [e.elem_type for e in top_elems]
            colors = [TYPE_COLORS.get(t, "#CCCCCC") for t in types]

            fig = go.Figure(go.Bar(
                x=acts,
                y=names,
                orientation='h',
                marker_color=colors,
                text=[f"{a:.3f}" for a in acts],
                textposition='outside',
            ))
            fig.update_layout(
                xaxis_title="Activation",
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=0, r=60, t=10, b=30),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Legend
            seen_types = list(dict.fromkeys(types))
            legend_parts = [
                f"<span style='color:{TYPE_COLORS.get(t, '#CCC')}'>■</span> {t}"
                for t in seen_types
            ]
            st.markdown("  ".join(legend_parts), unsafe_allow_html=True)

    # Right: activation over time for top elements
    with col_right:
        st.subheader("Activation History (all timesteps)")
        all_names = set()
        for h in history:
            for e in h.elements[:8]:
                all_names.add(e.name)

        time_data: Dict[str, List] = {name: [] for name in all_names}
        ts_list = [h.t for h in history]

        for h in history:
            present = {e.name: e.activation for e in h.elements}
            for name in all_names:
                time_data[name].append(present.get(name, 0.0))

        fig2 = go.Figure()
        for name in all_names:
            first_snap_elem = None
            for h in history:
                for e in h.elements:
                    if e.name == name:
                        first_snap_elem = e
                        break
                if first_snap_elem:
                    break
            color = TYPE_COLORS.get(first_snap_elem.elem_type if first_snap_elem else "Other", "#CCC")
            fig2.add_trace(go.Scatter(
                x=ts_list,
                y=time_data[name],
                mode='lines',
                name=name[:30],
                line=dict(color=color, width=1.5),
                hovertemplate=f"{name}<br>t=%{{x}}<br>a=%{{y:.3f}}"
            ))

        fig2.add_vline(x=snap.t, line_dash="dash", line_color="red", line_width=1)
        fig2.update_layout(
            xaxis_title="Timestep",
            yaxis_title="Activation",
            legend=dict(font=dict(size=8), x=1, y=1),
            margin=dict(l=0, r=0, t=10, b=30),
            height=380,
            showlegend=True,
        )
        st.plotly_chart(fig2, use_container_width=True)


def render_event(snap: StepSnapshot):
    """Panel 6: Plain-English description of this timestep."""
    st.subheader("What Happened This Step")

    phase_color = PHASE_COLORS.get(snap.phase, "#888")
    mode_color = MODE_COLORS.get(snap.action_mode, "#888")

    lines = [
        f"**t={snap.t}**  |  "
        f"Phase: <span style='color:{phase_color}'><b>{snap.phase}</b></span>  |  "
        f"Mode: <span style='color:{mode_color}'><b>.{snap.action_mode}()</b></span>",
    ]

    if snap.selected_agent != "none":
        agent_color = TYPE_COLORS.get(snap.selected_agent_type, "#888")
        lines.append(
            f"Selected agent: <span style='color:{agent_color}'><b>{snap.selected_agent}</b></span>  "
            f"(type: {snap.selected_agent_type})"
        )

    if snap.slipnet_queries:
        q = snap.slipnet_queries[-1]
        lines.append(
            f"**Slipnet queried** by {snap.selected_agent}:  "
            f"direction={q.direction}, target={q.target}  →  "
            f"returned: {', '.join(q.returned_agents) or 'nothing'}"
        )

    # New ImCells this step (GettingCloser ones)
    gc_imcells = [ic for ic in snap.imcells if ic.getting_closer_weight > 0.001]
    if gc_imcells:
        for ic in gc_imcells:
            lines.append(
                f"💡 Promising hypothetical: **{ic.name}**  "
                f"({ic.last_move})  GettingCloser weight = {ic.getting_closer_weight:.3f}"
            )

    if snap.solved:
        lines.append(f"✅ **SOLVED!** Solution: `{snap.solution}`")

    if snap.canvas_states:
        path = " → ".join(
            (cr.last_move if cr.last_move else f"[{','.join(str(n) for n in cr.avails)}]")
            for cr in snap.canvas_states
        )
        lines.append(f"Canvas: {path}")

    st.markdown("<br>".join(lines), unsafe_allow_html=True)


def render_slipnet_panel(snap: StepSnapshot):
    """Panel 7: Slipnet query details."""
    st.subheader("Slipnet Query — Analogical Reasoning")
    if not snap.slipnet_queries:
        st.info("No slipnet query this timestep. (Queries happen when Want.go() or Want.act() runs.)")
        return

    for i, q in enumerate(snap.slipnet_queries, 1):
        st.markdown(f"**Query {i}** — direction: `{q.direction}`, target: `{q.target}`")

        feat_rows = [{"Feature": k, "Weight": v} for k, v in q.features_in.items()]
        st.markdown("Features sent to slipnet:")
        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True)

        if q.returned_agents:
            st.markdown(f"Returned (sampled): `{', '.join(q.returned_agents)}`")

            st.caption(
                "The slipnet retrieves Consume templates whose feature profile best matches "
                "the input. Want then extracts the operator and applies it to the actual "
                "available numbers, creating new Consume agents in the workspace."
            )
        else:
            st.info("Slipnet returned nothing.")


def render_flow_panel(snap: StepSnapshot):
    """Panel 8: Activation flows this timestep."""
    st.subheader("Activation Flows — How Activation Moved This Step")
    if not snap.flows:
        st.info("No significant flows this timestep.")
        return

    rows = [{
        "From": f.from_elem[:40],
        "To": f.to_elem[:40],
        "Flow": round(f.amount, 5),
        "Direction": "→ in" if f.amount > 0 else "← out",
    } for f in snap.flows[:15]]

    df = pd.DataFrame(rows)

    fig = go.Figure(go.Bar(
        x=[r["Flow"] for r in rows],
        y=[f"{r['From'][:20]} → {r['To'][:20]}" for r in rows],
        orientation='h',
        marker_color=["#55A868" if r["Flow"] > 0 else "#CC4444" for r in rows],
        text=[f"{r['Flow']:.4f}" for r in rows],
        textposition='outside',
    ))
    fig.update_layout(
        xaxis_title="Activation flow amount",
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=0, r=80, t=10, b=30),
        height=min(400, max(200, len(rows) * 28)),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Positive = activation flowing into target node. Negative = flowing out.")


def render_support_graph(snap: StepSnapshot):
    """Panel 9: Support/Antipathy graph visualization."""
    st.subheader("Support & Antipathy Graph — Who is Helping/Hindering Whom")
    if not snap.graph_nodes:
        st.info("No graph data.")
        return

    # Build networkx graph from snapshot
    G = nx.DiGraph()
    node_activations = {n['name']: n['activation'] for n in snap.graph_nodes}
    node_types = {n['name']: n['type'] for n in snap.graph_nodes}

    # Only show top nodes by activation to keep readable
    top_names = set(n['name'] for n in sorted(snap.graph_nodes, key=lambda x: x['activation'], reverse=True)[:20])

    for n in snap.graph_nodes:
        if n['name'] in top_names:
            G.add_node(n['name'])

    for e in snap.graph_edges:
        if e['from'] in top_names and e['to'] in top_names:
            G.add_edge(e['from'], e['to'], weight=e['weight'], positive=e['positive'])

    if len(G.nodes) == 0:
        st.info("No nodes in graph.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    try:
        pos = nx.spring_layout(G, seed=42, k=2.0)
    except Exception:
        pos = nx.random_layout(G, seed=42)

    # Node sizes proportional to activation
    node_sizes = [max(300, node_activations.get(n, 0.1) * 2000) for n in G.nodes]
    node_colors = [TYPE_COLORS.get(node_types.get(n, 'Other'), '#CCC') for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.9)

    # Draw edges with color by positive/negative
    support_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('positive', True)]
    antipathy_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('positive', True)]

    if support_edges:
        nx.draw_networkx_edges(G, pos, edgelist=support_edges, ax=ax,
                               edge_color='#55A868', alpha=0.7, arrows=True,
                               arrowsize=15, width=1.5,
                               connectionstyle='arc3,rad=0.1')
    if antipathy_edges:
        nx.draw_networkx_edges(G, pos, edgelist=antipathy_edges, ax=ax,
                               edge_color='#CC4444', alpha=0.7, arrows=True,
                               arrowsize=15, width=1.5,
                               connectionstyle='arc3,rad=0.1')

    # Short labels
    labels = {}
    for n in G.nodes:
        parts = n.split('(')
        short = parts[0][:8]
        if len(parts) > 1:
            short += '(' + parts[1][:8]
        labels[n] = short

    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                             font_size=6, font_color='white')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=c, label=t)
        for t, c in TYPE_COLORS.items()
        if any(node_types.get(n) == t for n in G.nodes)
    ]
    legend_elements += [
        mpatches.Patch(facecolor='#55A868', label='Support edge'),
        mpatches.Patch(facecolor='#CC4444', label='Antipathy edge'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7,
              facecolor='#2a2a4e', labelcolor='white')
    ax.set_title(f"Support graph at t={snap.t} (top 20 nodes by activation)",
                 color='white', fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_heatmap(history: List[StepSnapshot], current_idx: int):
    """Panel 10: Activation heatmap across all timesteps."""
    st.subheader("Activation Heatmap — Lifecycle of Every Element")

    if len(history) < 2:
        st.info("Run more timesteps to see the heatmap.")
        return

    # Find top 15 elements by peak activation
    peak_a: Dict[str, float] = {}
    for h in history:
        for e in h.elements:
            peak_a[e.name] = max(peak_a.get(e.name, 0.0), e.activation)

    top_names = [n for n, _ in sorted(peak_a.items(), key=lambda x: x[1], reverse=True)[:15]]
    ts_list = [h.t for h in history]

    matrix = []
    for name in top_names:
        row = []
        for h in history:
            present = {e.name: e.activation for e in h.elements}
            row.append(present.get(name, 0.0))
        matrix.append(row)

    # Birth annotations
    birth_t: Dict[str, int] = {}
    for h in history:
        for e in h.elements:
            if e.name not in birth_t:
                birth_t[e.name] = e.tob

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=ts_list,
        y=[n[:40] for n in top_names],
        colorscale='Blues',
        showscale=True,
        hovertemplate="Element: %{y}<br>t=%{x}<br>activation=%{z:.4f}<extra></extra>"
    ))

    # Mark selected timestep
    fig.add_vline(x=history[current_idx].t, line_dash="dash", line_color="red", line_width=2)

    # Birth markers
    birth_xs, birth_ys, birth_texts = [], [], []
    for i, name in enumerate(top_names):
        if name in birth_t:
            birth_xs.append(birth_t[name])
            birth_ys.append(name[:40])
            birth_texts.append(f"born t={birth_t[name]}")

    if birth_xs:
        fig.add_trace(go.Scatter(
            x=birth_xs, y=birth_ys,
            mode='markers+text',
            marker=dict(symbol='triangle-up', size=10, color='yellow'),
            text=birth_texts,
            textposition='top center',
            textfont=dict(size=7, color='yellow'),
            name='Born',
            showlegend=True,
        ))

    fig.update_layout(
        xaxis_title="Timestep",
        yaxis_title="Element",
        margin=dict(l=0, r=0, t=10, b=30),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_agent_status(snap: StepSnapshot):
    """Panel 11: Agent status summary."""
    st.subheader("Agent Status Summary")

    agents = [e for e in snap.elements if e.elem_type in ("Want", "Consume", "Detector")]
    total = len(agents)

    active = sum(1 for e in agents if not e.is_sleeping and "GoIsDone" not in e.tags and "ActIsDone" not in e.tags and "Blocked" not in e.tags)
    sleeping = sum(1 for e in agents if e.is_sleeping)
    blocked = [e for e in agents if "Blocked" in e.tags]
    go_done = sum(1 for e in agents if "GoIsDone" in e.tags)
    act_done = sum(1 for e in agents if "ActIsDone" in e.tags)
    no_go = sum(1 for e in agents if "NoGo" in e.tags and "GoIsDone" not in e.tags and "ActIsDone" not in e.tags)
    no_act = sum(1 for e in agents if "NoAct" in e.tags and "ActIsDone" not in e.tags)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Agents", total)
    col2.metric("Active", active)
    col3.metric("Sleeping", sleeping)
    col4.metric("GoIsDone", go_done)
    col5.metric("ActIsDone", act_done)

    if blocked:
        st.markdown("**Blocked agents:**")
        for e in blocked:
            st.warning(f"🚫 `{e.name}` — tags: {', '.join(e.tags)}")

    if snap.sleeping:
        st.markdown("**Sleeping agents:**")
        for s in snap.sleeping:
            st.info(f"💤 `{s['name']}` — wakes at t={s['wakes_at']}")

    # Type distribution
    st.markdown("**Workspace type distribution:**")
    if snap.type_counts:
        tc_df = pd.DataFrame([
            {"Type": t, "Count": c, "Color": TYPE_COLORS.get(t, "#CCC")}
            for t, c in sorted(snap.type_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        fig = px.bar(tc_df, x="Type", y="Count", color="Type",
                     color_discrete_map=TYPE_COLORS)
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=30), height=200)
        st.plotly_chart(fig, use_container_width=True)


def render_event_log(history: List[StepSnapshot]):
    """Panel 12: Scrollable full event log."""
    st.subheader("Full Event Log")
    lines = []
    for h in history:
        prefix = "✅ " if h.solved else ""
        phase_badge = f"[{h.phase[:3]}]"
        line = f"**t={h.t}** {phase_badge} {prefix}{h.event_log_line}"
        lines.append(line)

    log_text = "\n\n".join(lines)
    st.markdown(
        f"<div style='height:300px;overflow-y:scroll;background:#1e1e2e;color:#cdd6f4;"
        f"padding:12px;border-radius:8px;font-family:monospace;font-size:0.85em'>"
        f"{'<br>'.join(lines)}"
        f"</div>",
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════
# Main Streamlit App
# ═══════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Numbo Dashboard",
        page_icon="🧠",
        layout="wide",
    )
    st.title("🧠 Numbo Cognitive Dashboard")
    st.caption(
        "A complete window into how Numbo (a FARG model) solves the puzzle [4, 5, 6] → 15. "
        "Every panel updates as you scrub through timesteps."
    )

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        seed_input = st.number_input("Seed (0 = random)", min_value=0, value=0, step=1)
        num_steps = st.slider("Max timesteps", 10, 100, 40)
        run_btn = st.button("▶ Run Numbo", type="primary")
        st.markdown("---")
        st.markdown("""
**Panel Guide:**
1. Run Header — problem & solution
2. Timestep Navigator — scrub through time
3. Canvas — committed arithmetic path
4. ImCells — what's being considered
5. Workspace Table — every element
6. Activation Charts — who's most active
7. This Step — plain-English event
8. Slipnet Query — analogical retrieval
9. Activation Flows — how activation moved
10. Support Graph — relationships
11. Activation Heatmap — element lifecycles
12. Agent Status — sleeping/blocked
13. Event Log — full history
        """)

    if run_btn or 'fm' not in st.session_state:
        with st.spinner("Running Numbo..."):
            seed = int(seed_input) if seed_input > 0 else None
            fm = run_numbo(seed=seed, num_timesteps=num_steps)
            st.session_state['fm'] = fm
            st.session_state['history'] = fm.history

    if 'fm' not in st.session_state or not st.session_state.get('history'):
        st.info("Click '▶ Run Numbo' to start.")
        return

    fm: ObservableNumbo = st.session_state['fm']
    history: List[StepSnapshot] = st.session_state['history']

    if not history:
        st.warning("No snapshots recorded.")
        return

    # Panel 0: Header
    render_header(fm)

    # Panel 1: Navigator (returns selected index)
    dummy_snap = history[-1]
    idx = render_navigator(dummy_snap, len(history))
    snap = history[idx]

    st.markdown("---")

    # Panel 6: This step event (early — most important at a glance)
    render_event(snap)
    st.markdown("---")

    # Panel 2: Canvas
    render_canvas(snap)
    st.markdown("---")

    # Panel 3: ImCells
    render_imcells(snap)
    st.markdown("---")

    # Panels 5: Activation charts (split)
    render_activation_charts(history, idx)
    st.markdown("---")

    # Panel 4: Workspace Table
    with st.expander("🗃 Workspace Table — Full Census of All Elements", expanded=False):
        render_workspace_table(snap)

    st.markdown("---")

    # Panel 7: Slipnet query
    with st.expander("🕸 Slipnet Query — Analogical Reasoning", expanded=True):
        render_slipnet_panel(snap)

    st.markdown("---")

    # Panel 8: Flows
    with st.expander("🌊 Activation Flows — How Activation Moved This Step", expanded=False):
        render_flow_panel(snap)

    st.markdown("---")

    # Panel 9: Support graph
    with st.expander("🕸 Support & Antipathy Graph", expanded=False):
        render_support_graph(snap)

    st.markdown("---")

    # Panel 10: Heatmap
    with st.expander("🔥 Activation Heatmap — Element Lifecycles", expanded=False):
        render_heatmap(history, idx)

    st.markdown("---")

    # Panel 11: Agent status
    with st.expander("📊 Agent Status & Type Counts", expanded=True):
        render_agent_status(snap)

    st.markdown("---")

    # Panel 12: Event log
    render_event_log(history)


if __name__ == '__main__':
    main()
