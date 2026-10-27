from __future__ import annotations

from typing import Any

import pandas as pd


def research_journey(state: dict[str, Any]) -> tuple[list[tuple[str, bool]], int]:
    stages = [
        ("Project PDF", state.get("project") is not None),
        ("Research Idea", state.get("project") is not None and state.get("active_idea") is not None),
        ("Data Strategy", bool((state.get("project") or {}).get("datasets"))),
        ("Novelty Screening", state.get("novelty") is not None),
        ("Experiment Plan / Data", state.get("experiment_plan") is not None or state.get("active_df") is not None),
        ("ML Benchmark", state.get("ml_results") is not None),
        ("Differential Privacy", state.get("dp_results") is not None),
        ("Robustness & XAI", state.get("robustness_results") is not None and state.get("statistical_validation") is not None),
        ("Publication Package", bool(state.get("publication_ready"))),
    ]
    return stages, sum(1 for _, done in stages if done)


def next_best_action(state: dict[str, Any]) -> tuple[str, str]:
    if state.get("project") is None:
        return "Upload your first project", "Start with a PDF so Nova can understand the domain and research potential."
    if state.get("novelty") is None:
        return "Run novelty analysis", "Screen the title, method, data and simulator combination against literature candidates."
    if state.get("experiment_plan") is None:
        return "Generate the experiment plan", "Define hypotheses, variables, baselines, metrics, split, repeatability and acceptance criteria."
    if state.get("active_df") is None:
        return "Prepare experimental data", "Upload a CSV or generate a reproducible simulator dataset."
    if state.get("ml_results") is None:
        return "Run the ML benchmark", "Compare baseline classifiers before privacy and robustness analysis."
    if state.get("dp_results") is None:
        return "Run Differential Privacy", "Evaluate multiple epsilon values and inspect the privacy–utility trade-off."
    if state.get("robustness_results") is None or state.get("statistical_validation") is None:
        return "Run robustness, XAI and statistics", "Stress-test the best model, run ablation/drift/SHAP and compare model errors statistically."
    if not state.get("publication_ready"):
        return "Build the publication package", "Generate the research story, full paper draft, reproducibility passport and final exports."
    return "Final professor / competition review", "The research pipeline is complete. Review claims, citations, exports and demo flow."


def generate_experiment_plan(project: dict[str, Any], idea: dict[str, Any], simulator_version: str = "") -> dict[str, Any]:
    return {
        "Hypothesis": f"The proposed workflow can produce reproducible and robust evidence for: {idea.get('question', '')}",
        "Independent variables": idea.get("inputs", []),
        "Target": idea.get("target", "target"),
        "Baseline models": ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM-RBF"],
        "Primary metrics": ["Accuracy", "Precision", "Recall", "Weighted F1"],
        "Split strategy": "Stratified 75/25 hold-out with fixed random seed 42; add repeated cross-validation in the final paper.",
        "Random seed": 42,
        "Repeatability": "Repeat key experiments across multiple seeds and report mean ± variability for the final manuscript.",
        "Simulator": idea.get("simulator", "Domain-specific"),
        "Simulator version": simulator_version or "Record exact version before final experiment.",
        "Robustness tests": ["Gaussian sensor noise", "Missing-data stress", "Sensor drift"],
        "Explainability": ["Permutation importance", "Global SHAP", "Local SHAP", "Automatic ablation"],
        "Privacy": "Differentially private logistic regression when diffprivlib is available, with epsilon sweep and bounded/clipped features.",
        "Statistical validation": "Exact McNemar test for paired prediction errors of the two leading models.",
        "Recommended sample size": "At least 3,000 observations for the synthetic baseline; justify the final sample size for the paper.",
        "Acceptance criteria": "Best model should outperform the baseline and retain acceptable F1 under noise, missingness and drift.",
        "Expected outputs": ["Benchmark table", "Privacy–utility curve", "Robustness curves", "Ablation table", "SHAP analysis", "McNemar p-value"],
    }


def journal_matches(domain: str) -> pd.DataFrame:
    candidates = {
        "Robotics & Mechanism Dynamics": [
            ("Robotics", 94, "Robotics methods, modeling and experiments"),
            ("IEEE Robotics and Automation Letters", 91, "Robotics systems and concise experimental contributions"),
            ("Sensors", 88, "Sensor-driven robotics, monitoring and applied ML"),
            ("Applied Sciences", 82, "Applied engineering and reproducible computational studies"),
        ],
        "Cybersecurity & AI": [
            ("Computers & Security", 93, "Security analytics and defensive methods"),
            ("Journal of Information Security and Applications", 90, "Applied cybersecurity research"),
            ("IEEE Access", 82, "Broad engineering and computing studies"),
        ],
        "Computer Vision": [
            ("Pattern Recognition Letters", 91, "Computer vision and pattern recognition"),
            ("Machine Vision and Applications", 89, "Applied vision systems"),
            ("Sensors", 84, "Camera/sensor-driven applied vision"),
        ],
        "General Applied AI": [
            ("Expert Systems with Applications", 90, "Applied AI and decision-support research"),
            ("Applied Soft Computing", 88, "Applied ML / computational intelligence"),
            ("IEEE Access", 82, "Broad applied computing and engineering"),
        ],
    }
    rows = candidates.get(domain, candidates["General Applied AI"])
    return pd.DataFrame([
        {"Venue": name, "Scope fit": fit, "Why it may fit": why, "Verification": "Verify current scope, quartile/indexing, APC and author guidelines on the official site."}
        for name, fit, why in rows
    ])


def reproducibility_passport(state: dict[str, Any]) -> pd.DataFrame:
    project = state.get("project") or {}
    signals = project.get("signals") or {}
    items = [
        ("Documented dataset / generator", state.get("active_df") is not None or bool(project.get("datasets"))),
        ("Fixed random seed", True),
        ("Experiment plan", state.get("experiment_plan") is not None),
        ("Simulator version documented", bool(state.get("simulator_version"))),
        ("Hyperparameters / benchmark stored", state.get("ml_results") is not None),
        ("Formal or labelled privacy analysis", state.get("dp_results") is not None),
        ("Robustness tests", state.get("robustness_results") is not None),
        ("Ablation study", state.get("ablation_results") is not None),
        ("Drift study", state.get("drift_results") is not None),
        ("Global explainability", state.get("shap_global") is not None or state.get("importance_results") is not None),
        ("Local explainability", state.get("shap_local") is not None),
        ("Statistical validation", state.get("statistical_validation") is not None),
        ("Code repository documented", bool(state.get("repo_url")) or signals.get("code_availability", False)),
        ("Environment snapshot", bool(state.get("environment_snapshot"))),
        ("Data/code availability statements", bool(state.get("availability_statement"))),
    ]
    return pd.DataFrame([{"Requirement": name, "Status": "Complete" if ok else "Pending", "Complete": bool(ok)} for name, ok in items])


def generate_research_story(context: dict[str, Any]) -> str:
    idea = context.get("active_idea") or {}
    novelty = context.get("novelty") or {}
    missing = context.get("missing") or []
    return f"""# Research Story

## Problem
**{context.get('project_title', 'Untitled project')}** is currently classified in **{context.get('domain', 'Unknown')}**.

## Gap
Initial structural gaps: **{', '.join(missing) if missing else 'no major gaps detected in the automated screening'}**.

## Research question
{idea.get('question', 'Define a research question in Research Ideas.')}

## Proposed contribution
{idea.get('novelty_hook', 'Define the expected contribution.')}

## Method
{idea.get('method', 'Method not selected.')}

## Data strategy
Candidate sources: {', '.join(context.get('datasets') or []) or 'not selected'}.

## Novelty screening
Overall novelty indicator: **{novelty.get('Overall novelty', 'not analyzed')}**. This is a heuristic screening value and not proof of absolute novelty.

## Experimental evidence
The final workflow combines baseline ML, Differential Privacy, robustness, ablation, drift, explainability and paired statistical validation.

## Reproducibility
Current reproducibility indicator: **{context.get('reproducibility', '—')}/100**. The final package should contain seed, environment, data provenance, simulator version, hyperparameters and availability statements.

## Next work
**{context.get('next_action_title', 'Continue validation')}** — {context.get('next_action_text', '')}

## Claim discipline
Final novelty, journal, privacy and publication claims must be verified against primary literature, official venue information and the exact experimental mechanism.
"""
