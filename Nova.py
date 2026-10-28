from __future__ import annotations

import copy
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core.assistant_engine import ask_openai, build_context, local_context_answer
from core.export_engine import (
    build_paper_docx,
    build_publication_package,
    build_session_archive,
    environment_snapshot,
    generate_paper_sections,
    parse_session_archive,
)
from core.ml_engine import (
    ablation_study,
    drift_study,
    mcnemar_exact,
    privacy_advisor,
    robotics_dataset,
    robustness_missing,
    robustness_noise,
    run_benchmark,
    run_dp_sweep,
)
from core.project_engine import analyze_pdf, build_gap_matrix, run_novelty_screen
from core.research_engine import (
    generate_experiment_plan,
    generate_research_story,
    journal_matches,
    next_best_action,
    reproducibility_passport,
    research_journey,
)
from core.visuals import (
    comparison_radar_figure,
    fig_to_png,
    novelty_heatmap_figure,
    novelty_radar_figure,
    research_graph_figure,
)
from core.xai_engine import permutation_feature_importance, shap_global_local


# =========================================================
# APP PATHS / CONFIG
# =========================================================

def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative


ICON_PATH = resource_path("assets/nova_icon.png")
LOGO_PATH = resource_path("assets/nova_logo.png")
page_icon: Any = "✦"
try:
    from PIL import Image
    if ICON_PATH.exists():
        page_icon = Image.open(ICON_PATH)
except Exception:
    pass

st.set_page_config(
    page_title="NovaResearch AI Premium",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PREMIUM UI / BRANDING
# =========================================================

st.markdown(
    """
<style>
:root {
    --nova-indigo:#4f46e5;
    --nova-violet:#7c3aed;
    --nova-cyan:#06b6d4;
    --nova-ink:#0f172a;
    --nova-muted:#64748b;
    --nova-border:rgba(148,163,184,.22);
    --nova-card:rgba(255,255,255,.90);
}
.stApp {
    background:
        radial-gradient(circle at 10% 8%, rgba(99,102,241,.12), transparent 28%),
        radial-gradient(circle at 92% 8%, rgba(6,182,212,.10), transparent 25%),
        linear-gradient(180deg,#f8fafc 0%,#ffffff 100%);
}
.block-container {max-width:1500px;padding-top:1.25rem;padding-bottom:5rem;}
.nova-hero {
    position:relative;overflow:hidden;padding:2.1rem 2.3rem;margin-bottom:1.35rem;border-radius:30px;color:white;
    background:linear-gradient(135deg,#111827 0%,#312e81 50%,#1d4ed8 100%);
    box-shadow:0 24px 70px rgba(15,23,42,.18);border:1px solid rgba(255,255,255,.12);
}
.nova-hero:before {content:"";position:absolute;width:290px;height:290px;right:-75px;top:-130px;border-radius:50%;background:rgba(34,211,238,.23);filter:blur(10px);}
.nova-hero:after {content:"";position:absolute;width:240px;height:240px;left:-95px;bottom:-135px;border-radius:50%;background:rgba(168,85,247,.20);filter:blur(12px);}
.nova-badge {position:relative;z-index:2;display:inline-flex;gap:.45rem;align-items:center;padding:.38rem .78rem;margin-bottom:.9rem;border-radius:999px;border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.10);font-size:.76rem;font-weight:800;letter-spacing:.06em;}
.nova-hero-title {position:relative;z-index:2;font-size:2.55rem;font-weight:850;letter-spacing:-.04em;margin:0;}
.nova-hero-subtitle {position:relative;z-index:2;margin-top:.7rem;color:rgba(255,255,255,.78);font-size:1rem;}
.nova-card {border-radius:22px;padding:1.15rem 1.25rem;border:1px solid var(--nova-border);background:var(--nova-card);box-shadow:0 12px 35px rgba(15,23,42,.055);min-height:126px;}
.nova-card-title {color:var(--nova-muted);font-size:.74rem;font-weight:850;letter-spacing:.075em;text-transform:uppercase;}
.nova-card-value {color:var(--nova-ink);font-size:1.9rem;font-weight:850;margin-top:.25rem;overflow-wrap:anywhere;}
.nova-card-sub {color:var(--nova-muted);font-size:.84rem;margin-top:.25rem;}
.next-action {border-radius:22px;padding:1.25rem 1.4rem;background:linear-gradient(135deg,rgba(79,70,229,.085),rgba(6,182,212,.085));border:1px solid rgba(79,70,229,.18);}
.next-action-label {color:#4f46e5;font-size:.75rem;font-weight:850;letter-spacing:.08em;text-transform:uppercase;}
.next-action-title {margin-top:.35rem;color:#0f172a;font-size:1.15rem;font-weight:800;}
.journey-step {padding:.72rem .9rem;margin-bottom:.48rem;border-radius:13px;border:1px solid var(--nova-border);background:rgba(255,255,255,.94);}
.journey-complete {border-left:4px solid #10b981;}
.journey-active {border-left:4px solid #6366f1;background:rgba(99,102,241,.055);}
.journey-pending {opacity:.58;}
.dna-pill {display:inline-block;padding:.35rem .7rem;margin:.18rem .12rem;border-radius:999px;border:1px solid rgba(79,70,229,.16);background:rgba(79,70,229,.055);color:#3730a3;font-size:.82rem;font-weight:700;}
.professor-banner {border:1px solid rgba(79,70,229,.18);border-radius:18px;padding:1rem 1.15rem;background:rgba(255,255,255,.86);}
section[data-testid="stSidebar"] {border-right:1px solid rgba(148,163,184,.16);}
section[data-testid="stSidebar"] > div {background:linear-gradient(180deg,#ffffff 0%,#f8fafc 100%);}
.stButton > button,.stDownloadButton > button {border-radius:12px;font-weight:700;}
[data-testid="stMetric"] {background:rgba(255,255,255,.82);padding:1rem;border-radius:18px;border:1px solid var(--nova-border);}
[data-testid="stDataFrame"] {border-radius:16px;overflow:hidden;}
/* ================================
   FORCE LIGHT / HIGH-CONTRAST UI
   Keeps Streamlit Cloud readable even when the browser/account is in dark mode.
   ================================ */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    color-scheme: light !important;
}
[data-testid="stAppViewContainer"] {
    background:#ffffff !important;
    color:#0f172a !important;
}
[data-testid="stMainBlockContainer"],
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stText"],
[data-testid="stCaptionContainer"],
h1,h2,h3,h4,h5,h6,p,li,label {
    color:#0f172a !important;
}
[data-testid="stCaptionContainer"] p,
small {
    color:#64748b !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:#ffffff !important;
}
section[data-testid="stSidebar"] > div {
    background:linear-gradient(180deg,#ffffff 0%,#f8fafc 100%) !important;
}
section[data-testid="stSidebar"] * {
    color:#0f172a !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color:#64748b !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] label {
    color:#0f172a !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background:rgba(79,70,229,.06) !important;
    border-radius:10px;
}

/* Inputs */
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="select"] input {
    color:#0f172a !important;
    -webkit-text-fill-color:#0f172a !important;
}
[data-baseweb="input"],
[data-baseweb="textarea"],
[data-baseweb="select"] > div {
    background:#ffffff !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background:#ffffff !important;
    box-shadow:0 8px 28px rgba(15,23,42,.06) !important;
}
[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"] * {
    color:#0f172a !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius:14px !important;
}
[data-testid="stAlert"] * {
    color:#0f172a !important;
}

/* Buttons: keep primary buttons legible */
.stButton > button, .stDownloadButton > button {
    border-radius:12px !important;
    font-weight:700 !important;
}
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    color:#ffffff !important;
    background:#4f46e5 !important;
    border-color:#4f46e5 !important;
}
.stButton > button[kind="secondary"],
.stDownloadButton > button[kind="secondary"] {
    color:#0f172a !important;
    background:#ffffff !important;
}

/* Custom Nova components override the global text rule */
.nova-hero, .nova-hero * {color:#ffffff !important;}
.nova-hero-subtitle {color:rgba(255,255,255,.82) !important;}
.nova-card-title,.nova-card-sub {color:#64748b !important;}
.nova-card-value,.next-action-title {color:#0f172a !important;}
.next-action-label {color:#4f46e5 !important;}
.dna-pill {color:#3730a3 !important;}

footer {visibility:hidden;}
@media (max-width:900px) {
    .nova-hero-title{font-size:1.95rem}
    .block-container{padding-left:.85rem;padding-right:.85rem}
    .nova-card{min-height:auto}
    .nova-hero{padding:1.5rem 1.35rem}
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# STATE
# =========================================================

DEFAULT_STATE = {
    "project": None,
    "active_idea_idx": 0,
    "novelty": None,
    "experiment_plan": None,
    "active_df": None,
    "active_target": None,
    "ml_results": None,
    "ml_cache": None,
    "dp_results": None,
    "dp_note": "",
    "robustness_results": None,
    "importance_results": None,
    "ablation_results": None,
    "drift_results": None,
    "shap_global": None,
    "shap_local": None,
    "statistical_validation": None,
    "selected_journal": None,
    "nova_messages": [],
    "research_story": None,
    "paper_sections": None,
    "publication_ready": False,
    "activity_log": [],
    "repo_url": "",
    "simulator_version": "",
    "availability_statement": "",
    "environment_snapshot": None,
    "professor_mode": False,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value)


def reset_project() -> None:
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = copy.deepcopy(value)


def log_event(event: str, details: str = "") -> None:
    st.session_state.activity_log.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        "details": details,
    })
    st.session_state.activity_log = st.session_state.activity_log[-100:]


def active_idea() -> dict[str, Any] | None:
    project = st.session_state.project
    if not project:
        return None
    ideas = project.get("ideas") or []
    if not ideas:
        return None
    idx = min(max(int(st.session_state.active_idea_idx), 0), len(ideas) - 1)
    return ideas[idx]


def state_payload() -> dict[str, Any]:
    return {k: st.session_state.get(k) for k in DEFAULT_STATE}


def research_state() -> dict[str, Any]:
    payload = state_payload()
    payload["active_idea"] = active_idea()
    return payload


def journey_and_action():
    state = research_state()
    return research_journey(state), next_best_action(state)


def current_context() -> dict[str, Any]:
    journey, action = journey_and_action()
    return build_context(research_state(), active_idea(), action, journey)


# =========================================================
# UI HELPERS
# =========================================================

def hero() -> None:
    project = st.session_state.project
    badge = "AI RESEARCH COPILOT"
    if project:
        badge = f"ACTIVE RESEARCH · {project['domain']['domain']}"
    st.markdown(
        f"""<div class="nova-hero"><div class="nova-badge">✦ {badge}</div><div class="nova-hero-title">NovaResearch AI</div><div class="nova-hero-subtitle">From project PDF to reproducible evidence, explainable ML and a publication-ready research package.</div></div>""",
        unsafe_allow_html=True,
    )


def card(title: str, value: str, sub: str) -> None:
    st.markdown(
        f"""<div class="nova-card"><div class="nova-card-title">{title}</div><div class="nova-card-value">{value}</div><div class="nova-card-sub">{sub}</div></div>""",
        unsafe_allow_html=True,
    )


def render_journey() -> None:
    (stages, completed), _ = journey_and_action()
    st.progress(completed / len(stages))
    st.caption(f"{completed} of {len(stages)} research stages completed")
    for index, (name, done) in enumerate(stages):
        if done:
            css, icon = "journey-complete", "✓"
        elif index == completed:
            css, icon = "journey-active", "●"
        else:
            css, icon = "journey-pending", "○"
        st.markdown(f'<div class="journey-step {css}"><b>{icon} {name}</b></div>', unsafe_allow_html=True)


def render_next_action() -> None:
    _, (title, text) = journey_and_action()
    st.markdown(
        f'<div class="next-action"><div class="next-action-label">✨ Nova Recommendation</div><div class="next-action-title">{title}</div><div style="margin-top:.35rem;color:#64748b">{text}</div></div>',
        unsafe_allow_html=True,
    )


def render_paper_dna() -> None:
    project = st.session_state.project
    if not project:
        return
    signals = project.get("signals", {})
    idea = active_idea() or {}
    novelty = st.session_state.novelty or {}
    elements = [
        ("Domain", project["domain"]["domain"]),
        ("Method", "Detected" if signals.get("methodology") else "Strengthen"),
        ("Data", "Detected" if signals.get("dataset") else "Planned"),
        ("ML", "Complete" if st.session_state.ml_results is not None else "Planned"),
        ("DP", "Complete" if st.session_state.dp_results is not None else "Planned"),
        ("XAI", "Complete" if st.session_state.shap_global is not None else "Planned"),
        ("Simulator", idea.get("simulator", "—")),
        ("Reproducibility", f"{project['scores']['Reproducibility']}/100"),
        ("Novelty", f"{novelty.get('Overall novelty', '—')}/100"),
    ]
    st.markdown("### Paper DNA")
    st.markdown("".join(f'<span class="dna-pill">{k}: {v}</span>' for k, v in elements), unsafe_allow_html=True)


def best_ml_summary() -> tuple[str, str]:
    df = st.session_state.ml_results
    if df is None or df.empty:
        return "—", "Not run"
    valid = df.dropna(subset=["F1"])
    if valid.empty:
        return "—", "No valid result"
    row = valid.iloc[0]
    return str(row["Model"]), f"F1 {row['F1']:.3f}"


def best_dp_summary() -> tuple[str, str]:
    df = st.session_state.dp_results
    if df is None or df.empty:
        return "—", "Not run"
    valid = df.dropna(subset=["F1"])
    if valid.empty:
        return "—", "No valid result"
    max_f1 = valid["F1"].max()
    candidate = valid[valid["F1"] >= max_f1 - 0.03].sort_values("Epsilon").iloc[0]
    return f"ε={candidate['Epsilon']:.3g}", f"F1 {candidate['F1']:.3f}"


def robustness_retention() -> str:
    rob = st.session_state.robustness_results or {}
    noise = rob.get("noise")
    if noise is None or noise.empty:
        return "—"
    base = float(noise.iloc[0]["F1"])
    final = float(noise.iloc[-1]["F1"])
    if base == 0:
        return "—"
    return f"{100 * final / base:.1f}%"


def idea_comparison_df() -> pd.DataFrame:
    project = st.session_state.project
    if not project:
        return pd.DataFrame()
    scores = project["scores"]
    rows = []
    for i, idea in enumerate(project.get("ideas", [])):
        active_novelty = (st.session_state.novelty or {}).get("Overall novelty") if i == st.session_state.active_idea_idx else None
        novelty_indicator = active_novelty if active_novelty is not None else max(45, int(idea.get("q1_potential", 70)) - 8)
        difficulty_penalty = {"Medium": 10, "High": 20, "Very high": 30}.get(str(idea.get("difficulty")), 15)
        rows.append({
            "Idea": idea.get("title"),
            "Level": idea.get("level"),
            "Research potential": idea.get("q1_potential", 0),
            "Readiness": scores.get("Overall", 0),
            "Novelty": novelty_indicator,
            "Reproducibility": scores.get("Reproducibility", 0),
            "Journal fit": scores.get("Journal fit", 0),
            "Difficulty": idea.get("difficulty"),
            "Cost": idea.get("estimated_cost", "—"),
            "Feasibility score": max(0, 100 - difficulty_penalty),
            "Active": "✓" if i == st.session_state.active_idea_idx else "",
        })
    return pd.DataFrame(rows)


def professor_banner() -> None:
    if st.session_state.professor_mode:
        st.markdown(
            '<div class="professor-banner"><b>Professor / Jury Mode</b><br>Focus on the Research Journey, Competition Dashboard, reproducibility, evidence and export package.</div>',
            unsafe_allow_html=True,
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=210)
    else:
        st.title("✦ NovaResearch AI")
    st.caption("Premium Research Copilot · Final V5")
    st.session_state.professor_mode = st.toggle("Professor / Jury Mode", value=st.session_state.professor_mode)

    pages = [
        "Dashboard",
        "Competition Dashboard",
        "Project AI",
        "Research Ideas",
        "Comparison Arena",
        "Data Intelligence",
        "Novelty & Gap Lab",
        "Experiment Planner",
        "Synthetic / Simulator Data",
        "ML & Privacy Lab",
        "Robustness & XAI",
        "Research Graph",
        "Journal Matcher",
        "Ask Nova",
        "Research Story",
        "Publication Studio",
        "Project Manager",
    ]
    page = st.radio("Navigation", pages)
    st.divider()
    if st.session_state.project:
        st.success(st.session_state.project["domain"]["domain"])
        idea = active_idea()
        if idea:
            st.caption(f"Active idea: {idea['level']}")
        (stages, completed), _ = journey_and_action()
        st.progress(completed / len(stages))
        st.caption(f"{completed}/{len(stages)} research stages")
    else:
        st.info("No active project")

    if st.button("New Project / Reset", use_container_width=True):
        reset_project()
        st.rerun()


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":
    hero(); professor_banner()
    st.markdown("## Research Command Center")
    st.caption("A single workspace for understanding, designing, validating and documenting a research project.")
    project = st.session_state.project
    if not project:
        st.info("Start by uploading a project PDF in **Project AI**.")
        st.markdown("### Research Journey")
        render_journey(); render_next_action()
    else:
        scores = project["scores"]
        idea = active_idea() or {}
        novelty = (st.session_state.novelty or {}).get("Overall novelty", "—")
        c1, c2, c3, c4 = st.columns(4)
        with c1: card("Research Readiness", f"{scores['Overall']}/100", "Current research maturity")
        with c2: card("Novelty Intelligence", f"{novelty}/100" if novelty != "—" else "—", "Literature screening")
        with c3: card("Reproducibility", f"{scores['Reproducibility']}/100", "Repeatability & documentation")
        with c4: card("Research Potential", f"{idea.get('q1_potential','—')}/100", "Internal potential indicator")
        st.markdown("### Project Snapshot")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Domain", project["domain"]["domain"])
        p2.metric("Domain confidence", f"{project['domain']['confidence']}%")
        p3.metric("Journal fit", f"{scores['Journal fit']}/100")
        p4.metric("Innovation", f"{scores['Innovation potential']}/100")
        render_next_action()
        left, right = st.columns([1.15, .85])
        with left:
            st.markdown("### Research Journey"); render_journey()
        with right:
            render_paper_dna()
            st.markdown("### Current gaps")
            if scores["Missing"]:
                for item in scores["Missing"]: st.write("•", item)
            else:
                st.success("No major structural gaps detected in the initial screening.")
        st.markdown("### Research Graph preview")
        fig = research_graph_figure(research_state(), active_idea()); st.pyplot(fig); plt.close(fig)


# =========================================================
# COMPETITION DASHBOARD
# =========================================================

elif page == "Competition Dashboard":
    hero(); professor_banner()
    st.subheader("Competition / Jury Dashboard")
    if not st.session_state.project:
        st.info("Upload a project first.")
    else:
        context = current_context(); project = st.session_state.project
        best_model, best_model_sub = best_ml_summary(); best_eps, best_eps_sub = best_dp_summary()
        top_feature = "—"
        if st.session_state.importance_results is not None and len(st.session_state.importance_results):
            top_feature = str(st.session_state.importance_results.iloc[0]["Feature"])
        passport = reproducibility_passport(research_state())
        passport_score = round(100 * passport["Complete"].mean()) if len(passport) else 0
        a,b,c,d = st.columns(4)
        with a: card("Readiness", f"{context['readiness_overall']}/100", "Research maturity")
        with b: card("Novelty", f"{(context.get('novelty') or {}).get('Overall novelty','—')}/100", "Heuristic screening")
        with c: card("Best ML", best_model, best_model_sub)
        with d: card("Repro Passport", f"{passport_score}/100", "Evidence & repeatability")
        e,f,g,h = st.columns(4)
        with e: card("Privacy candidate", best_eps, best_eps_sub)
        with f: card("Robustness retention", robustness_retention(), "Highest tested noise")
        with g: card("Top feature", top_feature, "Permutation importance")
        with h: card("Journal fit", f"{context['journal_fit']}/100", "Scope/readiness indicator")
        render_next_action()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Research Graph")
            fig = research_graph_figure(research_state(), active_idea()); st.pyplot(fig); plt.close(fig)
        with col2:
            st.markdown("### Reproducibility Passport")
            st.dataframe(passport.drop(columns=["Complete"]), hide_index=True, use_container_width=True)
        if st.session_state.novelty:
            st.markdown("### Novelty Radar & Heatmap")
            n1,n2 = st.columns(2)
            with n1:
                fig = novelty_radar_figure(st.session_state.novelty); st.pyplot(fig); plt.close(fig)
            with n2:
                fig = novelty_heatmap_figure(st.session_state.novelty); st.pyplot(fig); plt.close(fig)
        if st.session_state.dp_results is not None:
            st.markdown("### Privacy–Utility")
            fig, ax = plt.subplots(figsize=(7,3.5)); dp=st.session_state.dp_results
            ax.plot(dp["Epsilon"], dp["F1"], marker="o"); ax.set_xlabel("Epsilon"); ax.set_ylabel("Weighted F1"); ax.set_title("Differential Privacy · Utility trade-off")
            st.pyplot(fig); plt.close(fig)
        if st.session_state.robustness_results:
            st.markdown("### Robustness")
            noise=st.session_state.robustness_results["noise"]
            fig, ax = plt.subplots(figsize=(7,3.5)); ax.plot(noise["Noise level"],noise["F1"],marker="o"); ax.set_xlabel("Noise"); ax.set_ylabel("F1"); ax.set_title("Noise Robustness")
            st.pyplot(fig); plt.close(fig)
        st.markdown("### Jury-ready summary")
        st.write(f"**Project:** {context['project_title']}")
        st.write(f"**Active contribution:** {(active_idea() or {}).get('novelty_hook','—')}")
        st.write(f"**Current next action:** {context['next_action_title']} — {context['next_action_text']}")


# =========================================================
# PROJECT AI
# =========================================================

elif page == "Project AI":
    hero(); professor_banner(); st.subheader("1. Intelligent Project Understanding")
    upload = st.file_uploader("Upload the research project PDF", type=["pdf"])
    if upload and st.button("Analyze Project", type="primary"):
        with st.spinner("Extracting text and building the research profile..."):
            try:
                project = analyze_pdf(upload)
                if project.get("error"):
                    st.error(project["error"])
                else:
                    reset_project(); st.session_state.project = project
                    log_event("Project analyzed", project["title"]); st.success("Project analysis completed.")
            except Exception as exc:
                st.error(f"Project analysis failed: {exc}")
    project = st.session_state.project
    if project:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Domain", project["domain"]["domain"]); c2.metric("Confidence",f"{project['domain']['confidence']}%")
        c3.metric("Readiness",f"{project['scores']['Overall']}/100"); c4.metric("Words", project["stats"]["words"])
        st.markdown("### Project Intelligence Card")
        st.write("**Detected title:**", project["title"]); st.write("**Domain evidence:**", ", ".join(project["domain"]["evidence"]))
        st.write("**Recommended simulators:**", ", ".join(project["domain"]["profile"]["simulators"]))
        st.write("**Data strategies:**")
        for item in project["domain"]["profile"]["data_strategy"]: st.write("•",item)
        render_paper_dna()
        st.markdown("### Signals detected")
        sig = pd.DataFrame([{"Signal":k.replace('_',' ').title(),"Detected":"✓" if v else "—"} for k,v in project["signals"].items()])
        st.dataframe(sig,hide_index=True,use_container_width=True)


# =========================================================
# IDEAS / COMPARISON
# =========================================================

elif page == "Research Ideas":
    hero(); professor_banner(); st.subheader("2. Research Idea Portfolio")
    project=st.session_state.project
    if not project: st.info("Upload a project first.")
    else:
        for index, idea in enumerate(project["ideas"]):
            with st.container(border=True):
                st.markdown(f"### {index+1}. {idea['level']} — {idea['title']}")
                c1,c2,c3,c4,c5=st.columns(5)
                c1.metric("Potential",f"{idea['q1_potential']}/100"); c2.write("**Difficulty**");c2.write(idea["difficulty"])
                c3.write("**Time**");c3.write(idea["time"]); c4.write("**Hardware**");c4.write(idea["hardware"]); c5.write("**Cost**");c5.write(idea.get("estimated_cost","—"))
                st.write("**Research question:**",idea["question"]); st.write("**Inputs:**",", ".join(idea["inputs"])); st.write("**Target:**",idea["target"])
                st.write("**Method:**",idea["method"]); st.write("**Differential Privacy:**",idea["dp"]); st.write("**Simulator:**",idea["simulator"]); st.write("**Novelty hook:**",idea["novelty_hook"])
                if st.button("Select this idea",key=f"idea_{index}",type="primary" if index==st.session_state.active_idea_idx else "secondary"):
                    st.session_state.active_idea_idx=index; st.session_state.novelty=None; st.session_state.experiment_plan=None; st.session_state.research_story=None; st.session_state.paper_sections=None
                    log_event("Active idea changed",idea["title"]); st.success("Active idea updated across the application.")

elif page == "Comparison Arena":
    hero(); professor_banner(); st.subheader("3. Project & Idea Comparison Arena")
    if st.session_state.project:
        st.markdown("### Compare research ideas")
        df=idea_comparison_df(); st.dataframe(df,hide_index=True,use_container_width=True)
        if len(df):
            fig=comparison_radar_figure(df); st.pyplot(fig); plt.close(fig)
    else:
        st.info("Upload a project to compare its research ideas.")
    st.markdown("### Compare saved Nova projects")
    compare_files=st.file_uploader("Upload two or more `.novazip` saved projects",type=["novazip","zip"],accept_multiple_files=True,key="compare_archives")
    if compare_files:
        rows=[]
        for f in compare_files:
            try:
                data=parse_session_archive(f.getvalue()); project=data.get("project") or {}; scores=project.get("scores") or {}; novel=data.get("novelty") or {}
                rows.append({"Project":project.get("title",f.name),"Domain":(project.get("domain") or {}).get("domain","—"),"Readiness":scores.get("Overall",0),"Novelty":novel.get("Overall novelty",0),"Reproducibility":scores.get("Reproducibility",0),"Journal fit":scores.get("Journal fit",0),"Publication ready":"✓" if data.get("publication_ready") else ""})
            except Exception as exc:
                st.warning(f"Could not read {f.name}: {exc}")
        if rows: st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)


# =========================================================
# DATA / NOVELTY / EXPERIMENT
# =========================================================

elif page == "Data Intelligence":
    hero(); st.subheader("4. Open-Data Intelligence")
    project=st.session_state.project
    if not project: st.info("Upload a project first.")
    else:
        st.write("**Active idea:**",(active_idea() or {}).get("title","—"))
        for dataset in project["datasets"]:
            with st.container(border=True):
                st.markdown(f"### {dataset['name']}")
                c1,c2,c3,c4=st.columns(4); c1.metric("Fit",f"{dataset['fit']}/100"); c2.write("**Samples**");c2.write(dataset["samples"]);c3.write("**Features**");c3.write(dataset["features"]);c4.write("**Difficulty**");c4.write(dataset["difficulty"])
                st.write("**Inputs:**",dataset["inputs"]);st.write("**Target:**",dataset["target"]);st.write("**Missing values:**",dataset["missing"]);st.write("**Format:**",dataset["format"]);st.write("**License:**",dataset["license"]);st.write("**DOI / citation:**",dataset["doi"]);st.write("**Why it fits:**",dataset["why"])
                if str(dataset["url"]).startswith("http"): st.link_button("Open source",dataset["url"])

elif page == "Novelty & Gap Lab":
    hero(); st.subheader("5. Novelty Intelligence + Research Gap Matrix")
    project=st.session_state.project
    if not project: st.info("Upload a project first.")
    else:
        idea=active_idea() or {}; dataset_default=project["datasets"][0]["name"] if project["datasets"] else ""
        title=st.text_input("Research title",idea.get("title",""));method=st.text_area("Method",idea.get("method",""));dataset_name=st.text_input("Dataset",dataset_default);simulator=st.text_input("Simulator",idea.get("simulator",""))
        if st.button("Run full novelty analysis",type="primary"):
            with st.spinner("Searching Crossref literature candidates and computing the screening profile..."):
                novelty=run_novelty_screen(title,method,dataset_name,simulator);st.session_state.novelty=novelty;st.session_state.research_story=None;st.session_state.paper_sections=None
                log_event("Novelty screening completed",f"Overall={novelty['Overall novelty']}")
        novelty=st.session_state.novelty
        if novelty:
            c1,c2,c3=st.columns(3);c1.metric("Title novelty",f"{novelty['Title novelty']}/100");c2.metric("Method novelty",f"{novelty['Method novelty']}/100");c3.metric("Overall novelty",f"{novelty['Overall novelty']}/100")
            r,h=st.columns(2)
            with r: fig=novelty_radar_figure(novelty);st.pyplot(fig);plt.close(fig)
            with h: fig=novelty_heatmap_figure(novelty);st.pyplot(fig);plt.close(fig)
            if novelty.get("similar"):
                st.markdown("### Closest literature candidates");st.dataframe(pd.DataFrame(novelty["similar"]),hide_index=True,use_container_width=True)
            st.markdown("### Premium Research Gap Matrix")
            matrix=pd.DataFrame(build_gap_matrix(novelty.get("similar",[]),idea.get("title","YOUR ACTIVE IDEA")))
            st.dataframe(matrix,hide_index=True,use_container_width=True)
            st.warning(novelty["screening_note"])

elif page == "Experiment Planner":
    hero();st.subheader("6. AI Experiment Planner")
    project=st.session_state.project
    if not project: st.info("Upload a project first.")
    else:
        idea=active_idea() or {}; st.write("**Active idea:**",idea.get("title","—"))
        sim_version=st.text_input("Simulator / environment version",value=st.session_state.simulator_version,placeholder="e.g. CoppeliaSim 4.x / ROS version")
        st.session_state.simulator_version=sim_version
        if st.button("Generate complete experiment plan",type="primary"):
            st.session_state.experiment_plan=generate_experiment_plan(project,idea,sim_version);log_event("Experiment plan generated",idea.get("title",""))
        plan=st.session_state.experiment_plan
        if plan:
            for k,v in plan.items():
                with st.expander(k,expanded=k in ["Hypothesis","Split strategy","Acceptance criteria"]):
                    if isinstance(v,list):
                        for item in v: st.write("•",item)
                    else: st.write(v)
            st.download_button("Download experiment plan (.json)",json.dumps(plan,indent=2,ensure_ascii=False),file_name="nova_experiment_plan.json",mime="application/json")


# =========================================================
# SYNTHETIC / ML / DP
# =========================================================

elif page == "Synthetic / Simulator Data":
    hero();st.subheader("7. Synthetic / Simulator Data Generator")
    project=st.session_state.project
    if not project: st.info("Upload a project first.")
    elif project["domain"]["domain"]=="Robotics & Mechanism Dynamics":
        n=st.slider("Number of observations",500,20000,3000,500);seed=st.number_input("Random seed",value=42,step=1)
        if st.button("Generate robotics dataset",type="primary"):
            st.session_state.active_df=robotics_dataset(int(n),int(seed));st.session_state.active_target="state";st.session_state.ml_results=None;st.session_state.ml_cache=None;st.session_state.dp_results=None
            log_event("Synthetic dataset generated",f"{n} rows, seed={seed}")
        if st.session_state.active_df is not None:
            st.dataframe(st.session_state.active_df.head(25),use_container_width=True)
            st.download_button("Download synthetic CSV",st.session_state.active_df.to_csv(index=False).encode(),file_name="nova_robotics_synthetic_dataset.csv",mime="text/csv")
    else: st.info("The built-in synthetic generator is optimized for robotics. Upload a domain CSV in ML & Privacy Lab for other projects.")

elif page == "ML & Privacy Lab":
    hero();st.subheader("8. ML Benchmark & Differential Privacy Lab")
    csv_file=st.file_uploader("Upload CSV or use the active synthetic dataset",type=["csv"])
    if csv_file is not None:
        try: st.session_state.active_df=pd.read_csv(csv_file);log_event("CSV loaded",csv_file.name)
        except Exception as exc: st.error(f"Could not read CSV: {exc}")
    df=st.session_state.active_df
    if df is None: st.info("Upload a CSV or generate a synthetic dataset first.")
    else:
        st.dataframe(df.head(20),use_container_width=True)
        cols=list(df.columns);default_idx=cols.index(st.session_state.active_target) if st.session_state.active_target in cols else len(cols)-1
        target=st.selectbox("Target",cols,index=default_idx);st.session_state.active_target=target
        if st.button("Run Full ML Benchmark",type="primary"):
            with st.spinner("Training benchmark models..."):
                try:
                    results,fitted,split,preds,labels=run_benchmark(df,target);st.session_state.ml_results=results;st.session_state.ml_cache={"fitted":fitted,"split":split,"preds":preds,"labels":labels,"target":target};st.session_state.statistical_validation=None
                    log_event("ML benchmark completed",str(results.iloc[0]["Model"]) if len(results) else "")
                except Exception as exc: st.error(f"ML benchmark failed: {exc}")
        if st.session_state.ml_results is not None:
            st.markdown("### Non-private benchmark");st.dataframe(st.session_state.ml_results,hide_index=True,use_container_width=True);name,sub=best_ml_summary();st.success(f"Best model: {name} · {sub}")
        eps=st.multiselect("Epsilon sweep",[0.1,0.25,0.5,1,2,5,10],default=[0.1,0.5,1,2,5,10])
        if st.button("Run Differential Privacy Sweep"):
            with st.spinner("Training privacy-aware models..."):
                try:
                    dp,note=run_dp_sweep(df,target,eps);st.session_state.dp_results=dp;st.session_state.dp_note=note;log_event("Differential Privacy sweep completed")
                except Exception as exc: st.error(f"DP analysis failed: {exc}")
        if st.session_state.dp_results is not None:
            dp=st.session_state.dp_results;st.markdown("### Differential Privacy results");st.dataframe(dp,hide_index=True,use_container_width=True)
            fig,ax=plt.subplots(figsize=(7,3.8));ax.plot(dp["Epsilon"],dp["F1"],marker="o");ax.set_xlabel("Epsilon");ax.set_ylabel("Weighted F1");ax.set_title("Privacy–Utility Curve");st.pyplot(fig);plt.close(fig)
            st.info("Nova Privacy Advisor: "+privacy_advisor(dp));st.caption(st.session_state.dp_note)


# =========================================================
# ROBUSTNESS / XAI / STATS
# =========================================================

elif page == "Robustness & XAI":
    hero();st.subheader("9. Robustness, Ablation, Drift, SHAP & Statistical Validation")
    cache=st.session_state.ml_cache
    if cache is None: st.info("Run the ML benchmark first. If you loaded a saved project, use Project Manager → Rebuild ML cache.")
    else:
        fitted=cache["fitted"];X_train,X_test,y_train,y_test=cache["split"];valid=st.session_state.ml_results.dropna(subset=["F1"])
        best_name=str(valid.iloc[0]["Model"]);best=fitted[best_name];st.write("**Best model:**",best_name)
        if st.button("Run full robustness + XAI + statistics suite",type="primary"):
            with st.spinner("Running stress tests, ablation, drift, SHAP preparation and paired statistics..."):
                try:
                    noise=robustness_noise(best,X_test,y_test);missing=robustness_missing(best,X_test,y_test);importance=permutation_feature_importance(best,X_test,y_test)
                    ablation=ablation_study(best,X_train,X_test,y_train,y_test,importance);drift=drift_study(best,X_test,y_test)
                    st.session_state.robustness_results={"noise":noise,"missing":missing};st.session_state.importance_results=importance;st.session_state.ablation_results=ablation;st.session_state.drift_results=drift
                    names=[n for n in valid["Model"].tolist() if n in cache["preds"]]
                    if len(names)>=2: st.session_state.statistical_validation=mcnemar_exact(y_test,cache["preds"][names[0]],cache["preds"][names[1]],names[0],names[1])
                    log_event("Robustness/XAI/statistics suite completed",best_name)
                except Exception as exc: st.error(f"Suite failed: {exc}")
        rob=st.session_state.robustness_results
        if rob:
            c1,c2=st.columns(2)
            with c1:
                st.markdown("### Noise robustness");st.dataframe(rob["noise"],hide_index=True,use_container_width=True);fig,ax=plt.subplots();ax.plot(rob["noise"]["Noise level"],rob["noise"]["F1"],marker="o");ax.set_xlabel("Noise");ax.set_ylabel("F1");st.pyplot(fig);plt.close(fig)
            with c2:
                st.markdown("### Missing-data robustness");st.dataframe(rob["missing"],hide_index=True,use_container_width=True);fig,ax=plt.subplots();ax.plot(rob["missing"]["Missing ratio"],rob["missing"]["F1"],marker="o");ax.set_xlabel("Missing ratio");ax.set_ylabel("F1");st.pyplot(fig);plt.close(fig)
        if st.session_state.importance_results is not None:
            st.markdown("### Permutation importance");st.dataframe(st.session_state.importance_results.head(20),hide_index=True,use_container_width=True)
            top=st.session_state.importance_results.head(12).iloc[::-1];fig,ax=plt.subplots();ax.barh(top["Feature"],top["Importance"]);ax.set_title("Permutation Importance");st.pyplot(fig);plt.close(fig)
        if st.session_state.ablation_results is not None:
            st.markdown("### Automatic Ablation Study");st.dataframe(st.session_state.ablation_results,hide_index=True,use_container_width=True)
        if st.session_state.drift_results is not None:
            st.markdown("### Sensor Drift Simulator");st.dataframe(st.session_state.drift_results,hide_index=True,use_container_width=True);fig,ax=plt.subplots();ax.plot(st.session_state.drift_results["Drift level (σ)"],st.session_state.drift_results["F1"],marker="o");ax.set_xlabel("Drift level (σ)");ax.set_ylabel("F1");st.pyplot(fig);plt.close(fig)
        st.markdown("### Real SHAP · Global + Local")
        row_idx=st.number_input("Test-row index for local SHAP",min_value=0,max_value=max(0,len(X_test)-1),value=0,step=1)
        if st.button("Compute SHAP explanations"):
            with st.spinner("Computing real SHAP values..."):
                g,l,note=shap_global_local(best,X_test,int(row_idx));st.session_state.shap_global=g;st.session_state.shap_local=l
                if g is not None: log_event("SHAP computed",best_name);st.success(note)
                else: st.warning(note)
        if st.session_state.shap_global is not None:
            s1,s2=st.columns(2)
            with s1:
                st.markdown("#### Global SHAP");st.dataframe(st.session_state.shap_global.head(20),hide_index=True,use_container_width=True);top=st.session_state.shap_global.head(12).iloc[::-1];fig,ax=plt.subplots();ax.barh(top["Feature"],top["Mean |SHAP|"]);ax.set_title("Global SHAP Importance");st.pyplot(fig);plt.close(fig)
            with s2:
                st.markdown("#### Local SHAP");st.dataframe(st.session_state.shap_local.head(20),hide_index=True,use_container_width=True);top=st.session_state.shap_local.head(12).iloc[::-1];fig,ax=plt.subplots();ax.barh(top["Feature"],top["SHAP contribution"]);ax.axvline(0,linewidth=1);ax.set_title("Local prediction explanation");st.pyplot(fig);plt.close(fig)
        if st.session_state.statistical_validation:
            st.markdown("### Exact McNemar Statistical Validation");stat=st.session_state.statistical_validation
            st.dataframe(pd.DataFrame([stat]),hide_index=True,use_container_width=True)
            if stat["Significant at 0.05"]: st.success(f"p={stat['p_value']:.4g}: the two models show statistically different paired error patterns at α=0.05.")
            else: st.info(f"p={stat['p_value']:.4g}: no significant paired-error difference detected at α=0.05.")


# =========================================================
# GRAPH / JOURNAL / ASK NOVA
# =========================================================

elif page == "Research Graph":
    hero();st.subheader("10. Research Graph")
    if not st.session_state.project: st.info("Upload a project first.")
    else:
        fig=research_graph_figure(research_state(),active_idea());st.pyplot(fig);plt.close(fig)
        st.caption("The graph updates automatically as the research pipeline progresses from PDF to publication.")
        st.markdown("### Research Journey");render_journey()

elif page == "Journal Matcher":
    hero();st.subheader("11. Journal / Venue Matcher")
    project=st.session_state.project
    if not project: st.info("Upload a project first.")
    else:
        matches=journal_matches(project["domain"]["domain"]);st.dataframe(matches,hide_index=True,use_container_width=True)
        venue=st.selectbox("Select a candidate venue",matches["Venue"].tolist())
        if st.button("Save selected venue"):
            st.session_state.selected_journal=matches[matches["Venue"]==venue].iloc[0].to_dict();log_event("Venue selected",venue);st.success(f"Selected: {venue}")
        st.warning("This matcher is scope-based. Verify current quartile/indexing, APC, deadlines and author instructions on the venue's official website before submission.")

elif page == "Ask Nova":
    hero();st.subheader("12. Ask Nova — Context-Aware Research Copilot")
    if not st.session_state.project: st.info("Upload a project first.")
    else:
        context=current_context();c1,c2,c3,c4=st.columns(4);c1.metric("Domain",context["domain"]);c2.metric("Readiness",f"{context['readiness_overall']}/100");c3.metric("Novelty","Not analyzed" if not context.get("novelty") else f"{context['novelty'].get('Overall novelty','—')}/100");c4.metric("Stage",f"{context['research_stage']}/{context['research_stage_total']}")
        engine=st.radio("Nova engine",["Local contextual engine","OpenAI Responses API (optional)"],horizontal=True)
        api_key=""
        try: api_key=st.secrets.get("OPENAI_API_KEY","")
        except Exception: api_key=os.getenv("OPENAI_API_KEY","") or ""
        if engine.startswith("OpenAI") and not api_key: st.caption("No API key is configured; Nova will automatically fall back to the local contextual engine.")
        if not st.session_state.nova_messages:
            st.info("Try: *What should I do next?* · *Why?* · *What evidence in my PDF supports this?* · *Compare my ML and privacy results.* · *What should I show my professor?*")
        for m in st.session_state.nova_messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        prompt=st.chat_input("Ask Nova about the active project...")
        if prompt:
            st.session_state.nova_messages.append({"role":"user","content":prompt})
            with st.chat_message("user"): st.markdown(prompt)
            answer=None;err=None
            if engine.startswith("OpenAI") and api_key: answer,err=ask_openai(prompt,context,st.session_state.nova_messages,api_key=api_key)
            if answer is None:
                answer=local_context_answer(prompt,context,st.session_state.nova_messages)
                if err: answer += f"\n\n*Cloud engine fallback:* {err}"
            st.session_state.nova_messages.append({"role":"assistant","content":answer})
            with st.chat_message("assistant"): st.markdown(answer)


# =========================================================
# STORY / PUBLICATION
# =========================================================

elif page == "Research Story":
    hero();st.subheader("13. Research Story Generator")
    if not st.session_state.project: st.info("Upload a project first.")
    else:
        if st.button("Generate / Refresh Research Story",type="primary"):
            st.session_state.research_story=generate_research_story(current_context());log_event("Research Story generated")
        if st.session_state.research_story:
            st.markdown(st.session_state.research_story);st.download_button("Download Research Story (.md)",st.session_state.research_story,file_name="NovaResearch_Story.md",mime="text/markdown")

elif page == "Publication Studio":
    hero();professor_banner();st.subheader("14. Publication Studio V2")
    if not st.session_state.project: st.info("Upload a project first.")
    else:
        context=current_context();project=st.session_state.project;idea=active_idea() or {}
        if st.session_state.research_story is None: st.session_state.research_story=generate_research_story(context)
        if st.button("Generate / Refresh complete paper sections",type="primary"):
            st.session_state.environment_snapshot=environment_snapshot();st.session_state.paper_sections=generate_paper_sections(context,research_state());log_event("Paper sections generated")
        if st.session_state.paper_sections is None: st.session_state.paper_sections=generate_paper_sections(context,research_state())
        sections=st.session_state.paper_sections
        tabs=st.tabs(list(sections.keys()))
        for tab,(heading,body) in zip(tabs,sections.items()):
            with tab: st.markdown(f"### {heading}");st.write(body)
        st.markdown("### Reproducibility Passport")
        passport=reproducibility_passport(research_state());st.dataframe(passport.drop(columns=["Complete"]),hide_index=True,use_container_width=True);score=round(100*passport["Complete"].mean());st.metric("Passport score",f"{score}/100")
        if st.button("Mark publication stage ready"):
            st.session_state.publication_ready=True;log_event("Publication stage marked ready");st.success("Publication stage marked complete.")
        docx=build_paper_docx(sections,project["title"]);st.download_button("Download complete paper draft (.docx)",docx,file_name="NovaResearch_Final_Paper_Draft.docx",mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        plots={}
        if st.session_state.novelty:
            plots["novelty_radar"]=fig_to_png(novelty_radar_figure(st.session_state.novelty));plots["novelty_heatmap"]=fig_to_png(novelty_heatmap_figure(st.session_state.novelty))
        plots["research_graph"]=fig_to_png(research_graph_figure(research_state(),active_idea()))
        if st.session_state.ml_results is not None:
            ml=st.session_state.ml_results.dropna(subset=["F1"]);fig,ax=plt.subplots();ax.bar(ml["Model"],ml["F1"]);ax.set_ylim(0,1);ax.set_ylabel("Weighted F1");ax.set_title("ML Benchmark");ax.tick_params(axis="x",rotation=20);plots["ml_benchmark"]=fig_to_png(fig)
        if st.session_state.dp_results is not None:
            dp=st.session_state.dp_results;fig,ax=plt.subplots();ax.plot(dp["Epsilon"],dp["F1"],marker="o");ax.set_xlabel("Epsilon");ax.set_ylabel("F1");ax.set_title("Privacy–Utility");plots["privacy_utility"]=fig_to_png(fig)
        if st.session_state.robustness_results:
            noise=st.session_state.robustness_results["noise"];fig,ax=plt.subplots();ax.plot(noise["Noise level"],noise["F1"],marker="o");ax.set_xlabel("Noise");ax.set_ylabel("F1");ax.set_title("Noise Robustness");plots["robustness_noise"]=fig_to_png(fig)
        if st.session_state.ablation_results is not None and len(st.session_state.ablation_results):
            ab=st.session_state.ablation_results.head(12).iloc[::-1];fig,ax=plt.subplots();ax.barh(ab["Removed feature"],ab["F1 drop"]);ax.set_title("Ablation Impact");plots["ablation"]=fig_to_png(fig)
        if st.session_state.drift_results is not None:
            dr=st.session_state.drift_results;fig,ax=plt.subplots();ax.plot(dr["Drift level (σ)"],dr["F1"],marker="o");ax.set_xlabel("Drift level (σ)");ax.set_ylabel("F1");ax.set_title("Sensor Drift");plots["sensor_drift"]=fig_to_png(fig)
        if st.session_state.shap_global is not None:
            sg=st.session_state.shap_global.head(12).iloc[::-1];fig,ax=plt.subplots();ax.barh(sg["Feature"],sg["Mean |SHAP|"]);ax.set_title("Global SHAP");plots["shap_global"]=fig_to_png(fig)
        if st.session_state.shap_local is not None:
            sl=st.session_state.shap_local.head(12).iloc[::-1];fig,ax=plt.subplots();ax.barh(sl["Feature"],sl["SHAP contribution"]);ax.axvline(0,linewidth=1);ax.set_title("Local SHAP");plots["shap_local"]=fig_to_png(fig)
        package=build_publication_package(research_state(),context,sections,passport,plots=plots)
        st.download_button("Download COMPLETE publication package (.zip)",package,file_name="NovaResearch_COMPLETE_Publication_Package.zip",mime="application/zip",type="primary")
        st.caption("Includes paper draft, paper sections, project metadata, experiment plan, journal match, environment snapshot, reproducibility passport, result tables and generated plots.")


# =========================================================
# PROJECT MANAGER / SAVE-LOAD / PROFESSOR PACKAGE
# =========================================================

elif page == "Project Manager":
    hero();professor_banner();st.subheader("15. Project Manager, Session Recovery & Professor Mode")
    st.markdown("### Reproducibility metadata")
    st.session_state.repo_url=st.text_input("Code repository URL",value=st.session_state.repo_url,placeholder="https://github.com/...")
    st.session_state.simulator_version=st.text_input("Simulator / environment version",value=st.session_state.simulator_version)
    st.session_state.availability_statement=st.text_area("Data / Code Availability statement",value=st.session_state.availability_statement,placeholder="Describe where data and code will be available.")
    st.session_state.environment_snapshot=environment_snapshot()
    c1,c2=st.columns(2)
    with c1:
        st.markdown("### Save full session")
        archive=build_session_archive(research_state())
        st.download_button("Save project (.novazip)",archive,file_name="NovaResearch_Project.novazip",mime="application/zip",type="primary")
    with c2:
        st.markdown("### Load full session")
        f=st.file_uploader("Load `.novazip`",type=["novazip","zip"],key="load_archive")
        if f and st.button("Restore saved project"):
            try:
                restored=parse_session_archive(f.getvalue())
                reset_project()
                for key,value in restored.items():
                    if key in DEFAULT_STATE: st.session_state[key]=value
                log_event("Saved project restored",f.name);st.success("Project restored. ML model objects are rebuilt separately when required.")
            except Exception as exc: st.error(f"Could not restore project: {exc}")
    if st.session_state.active_df is not None and st.session_state.active_target in st.session_state.active_df.columns and st.session_state.ml_cache is None:
        if st.button("Rebuild ML cache after loading"):
            with st.spinner("Rebuilding models from restored data..."):
                results,fitted,split,preds,labels=run_benchmark(st.session_state.active_df,st.session_state.active_target);st.session_state.ml_results=results;st.session_state.ml_cache={"fitted":fitted,"split":split,"preds":preds,"labels":labels,"target":st.session_state.active_target};log_event("ML cache rebuilt")
                st.success("ML cache rebuilt.")
    st.markdown("### Live Analysis Timeline")
    if st.session_state.activity_log: st.dataframe(pd.DataFrame(st.session_state.activity_log),hide_index=True,use_container_width=True)
    else: st.caption("No activity recorded yet.")
    st.markdown("### Professor / Jury Demo Checklist")
    for item in [
        "Open Competition Dashboard and explain the Research Journey",
        "Show Project AI + Paper DNA",
        "Show Idea Comparison Arena",
        "Run Novelty Radar + Heatmap + Gap Matrix",
        "Show Experiment Planner and reproducible data",
        "Show ML + Differential Privacy + Privacy Advisor",
        "Show Robustness + Ablation + Drift + SHAP + McNemar",
        "Ask Nova a contextual follow-up question",
        "Show Research Graph and Reproducibility Passport",
        "Export the complete DOCX and publication ZIP",
        "Share the online URL and/or Windows executable",
    ]: st.write("✓",item)
    st.markdown("### Reset / New Project")
    if st.button("Reset entire project",type="secondary"):
        reset_project();st.rerun()


st.divider()
st.caption("NovaResearch AI Premium · Final V5 · research decision support · verify novelty, citations, privacy and venue claims before publication")
