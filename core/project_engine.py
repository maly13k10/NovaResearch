from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import requests
from pypdf import PdfReader

DOMAIN_PROFILES = {
    "Robotics & Mechanism Dynamics": {
        "keywords": [
            "robot", "robotic", "manipulator", "trajectory", "kinematic", "dynamic",
            "torque", "joint", "actuator", "scara", "coppeliasim", "gazebo", "ros",
            "sensor", "mechanism", "workspace", "end-effector", "pallet", "gripper",
        ],
        "simulators": ["CoppeliaSim", "Gazebo / ROS", "MATLAB / Simulink"],
        "data_strategy": [
            "Use reproducible simulator data when hardware data are limited.",
            "Store seeds, joint limits, controller settings and simulator version.",
            "Compare synthetic results against a real/open benchmark when possible.",
        ],
    },
    "Cybersecurity & AI": {
        "keywords": [
            "cyber", "security", "malware", "phishing", "intrusion", "attack", "network",
            "authentication", "threat", "vulnerability", "forensic", "anomaly", "ids", "siem",
        ],
        "simulators": ["Mininet", "Docker lab", "Network traffic replay"],
        "data_strategy": [
            "Prefer public, timestamped security datasets with documented labels.",
            "Use train/test separation that avoids temporal leakage.",
            "Report class imbalance and false-positive behavior.",
        ],
    },
    "Computer Vision": {
        "keywords": [
            "image", "vision", "opencv", "camera", "segmentation", "detection", "classification",
            "cnn", "yolo", "pixel", "object recognition",
        ],
        "simulators": ["CoppeliaSim", "Blender", "Synthetic image augmentation"],
        "data_strategy": [
            "Document image resolution, acquisition pipeline and augmentations.",
            "Keep subject/scenario leakage out of the test split.",
            "Include calibration or uncertainty analysis where relevant.",
        ],
    },
    "Industrial AI & Predictive Maintenance": {
        "keywords": [
            "predictive maintenance", "fault", "vibration", "bearing", "machine", "industrial",
            "condition monitoring", "failure", "maintenance", "remaining useful life",
        ],
        "simulators": ["MATLAB / Simulink", "Digital twin", "Signal generators"],
        "data_strategy": [
            "Preserve time order for validation.",
            "Document sensor sampling rates and operating regimes.",
            "Stress-test against noise, missingness and sensor drift.",
        ],
    },
    "General Applied AI": {
        "keywords": ["machine learning", "artificial intelligence", "dataset", "model", "classification", "regression"],
        "simulators": ["Domain-specific simulator"],
        "data_strategy": [
            "Use a documented data source.",
            "Define a reproducible evaluation split.",
            "Report baselines, uncertainty and limitations.",
        ],
    },
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_pdf_text(upload) -> str:
    upload.seek(0)
    reader = PdfReader(upload)
    chunks: list[str] = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            chunks.append("")
    return "\n".join(chunks)


def guess_title(text: str) -> str:
    lines = [clean_text(x) for x in (text or "").splitlines() if clean_text(x)]
    candidates = [x for x in lines[:35] if 8 <= len(x) <= 190]
    if candidates:
        return max(candidates[:12], key=lambda s: min(len(s), 135))
    return "Untitled Research Project"


def text_stats(text: str) -> dict[str, int]:
    words = re.findall(r"\b[\w-]+\b", text or "")
    return {
        "characters": len(text or ""),
        "words": len(words),
        "pages_estimate": max(1, round(len(words) / 450)) if words else 0,
    }


def detect_domain(text: str) -> dict[str, Any]:
    low = (text or "").lower()
    scores: dict[str, int] = {}
    evidence: dict[str, list[str]] = {}
    for domain, profile in DOMAIN_PROFILES.items():
        found = [kw for kw in profile["keywords"] if kw in low]
        scores[domain] = len(found)
        evidence[domain] = found[:12]
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = "General Applied AI"
    total = sum(scores.values()) or 1
    confidence = int(min(98, 55 + 43 * scores[best] / total))
    return {
        "domain": best,
        "confidence": confidence,
        "evidence": evidence.get(best) or ["general scientific / AI terminology"],
        "profile": DOMAIN_PROFILES[best],
    }


def extract_project_signals(text: str) -> dict[str, bool]:
    low = (text or "").lower()
    groups = {
        "problem_statement": ["problem", "objective", "aim", "goal", "research question"],
        "dataset": ["dataset", "data set", "data collection", "samples", "observations"],
        "methodology": ["method", "methodology", "algorithm", "model", "pipeline"],
        "experiment": ["experiment", "evaluation", "validation", "benchmark", "test"],
        "metrics": ["accuracy", "precision", "recall", "f1", "rmse", "mae", "auc"],
        "reproducibility": ["seed", "github", "repository", "reproduc", "environment", "version"],
        "privacy": ["privacy", "differential privacy", "epsilon", "gdpr"],
        "robustness": ["robust", "noise", "missing", "drift", "stress test"],
        "explainability": ["explain", "shap", "feature importance", "interpretability"],
        "limitations": ["limitation", "threats to validity", "future work"],
        "statistical_validation": ["p-value", "p value", "mcnemar", "confidence interval", "significant"],
        "code_availability": ["github", "gitlab", "code available", "repository"],
    }
    return {key: any(term in low for term in terms) for key, terms in groups.items()}


def readiness_scores(domain: dict[str, Any], signals: dict[str, bool]) -> dict[str, Any]:
    count = sum(bool(v) for v in signals.values())
    reproducibility = 40 + 9 * sum(int(signals.get(k, False)) for k in ["dataset", "experiment", "metrics", "reproducibility", "code_availability"])
    innovation = 50 + 8 * sum(int(signals.get(k, False)) for k in ["privacy", "robustness", "explainability", "methodology", "statistical_validation"])
    journal_fit = 50 + 6 * sum(int(signals.get(k, False)) for k in ["problem_statement", "methodology", "experiment", "metrics", "limitations", "statistical_validation"])
    open_data = 55 + 25 * int(signals.get("dataset", False)) + 12 * int(signals.get("reproducibility", False))
    overall = int(np.clip(round(50 + count * 3.2 + domain["confidence"] * 0.15), 0, 100))
    missing = [
        label
        for key, label in [
            ("problem_statement", "clear problem / research question"),
            ("dataset", "documented dataset or data strategy"),
            ("experiment", "experimental protocol"),
            ("metrics", "evaluation metrics"),
            ("reproducibility", "reproducibility details"),
            ("statistical_validation", "statistical validation"),
            ("limitations", "limitations / threats to validity"),
        ]
        if not signals.get(key, False)
    ]
    return {
        "Overall": int(np.clip(overall, 0, 100)),
        "Journal fit": int(np.clip(journal_fit, 0, 100)),
        "Reproducibility": int(np.clip(reproducibility, 0, 100)),
        "Innovation potential": int(np.clip(innovation, 0, 100)),
        "Open-data feasibility": int(np.clip(open_data, 0, 100)),
        "Missing": missing,
    }


def ideas_for_domain(domain: str) -> list[dict[str, Any]]:
    if domain == "Robotics & Mechanism Dynamics":
        return [
            {
                "level": "Balanced",
                "title": "Privacy-aware robust robotic state classification from simulated sensor data",
                "q1_potential": 86,
                "difficulty": "Medium",
                "time": "4–8 weeks",
                "hardware": "Optional",
                "estimated_cost": "Low",
                "question": "Can a reproducible ML pipeline retain robust classification performance under sensor noise while evaluating a privacy–utility trade-off?",
                "inputs": ["joint position", "velocity", "torque", "sensor noise", "payload"],
                "target": "state",
                "method": "Synthetic simulator data + LR/RF/GB/SVM benchmark + formal/fallback privacy analysis + robustness + XAI",
                "dp": "Epsilon sweep / formal DP when diffprivlib is available",
                "simulator": "CoppeliaSim-compatible synthetic generator",
                "novelty_hook": "Combines reproducibility, robustness, explainability and privacy-aware analysis in one robotic ML workflow.",
            },
            {
                "level": "Advanced",
                "title": "Sensor-drift-aware fault recognition for industrial manipulators",
                "q1_potential": 91,
                "difficulty": "High",
                "time": "6–12 weeks",
                "hardware": "Optional",
                "estimated_cost": "Low–Medium",
                "question": "How quickly does model performance degrade under realistic sensor drift, and which features drive the degradation?",
                "inputs": ["position", "velocity", "current", "vibration", "temperature"],
                "target": "fault_state",
                "method": "Benchmark + drift simulation + ablation + local/global explainability + statistical validation",
                "dp": "Optional privacy layer",
                "simulator": "CoppeliaSim / digital twin",
                "novelty_hook": "Turns robustness from a static test into a temporal drift study with interpretable degradation.",
            },
            {
                "level": "Ambitious",
                "title": "Reproducible digital-twin benchmark for privacy, robustness and explainability in robot monitoring",
                "q1_potential": 94,
                "difficulty": "Very high",
                "time": "8–16 weeks",
                "hardware": "Recommended for final validation",
                "estimated_cost": "Medium",
                "question": "Can one benchmark quantify predictive quality, privacy, robustness and explanation stability for robot monitoring models?",
                "inputs": ["multi-sensor telemetry", "controller state", "task context"],
                "target": "operating_condition",
                "method": "Digital twin + multi-model benchmark + formal/fallback privacy + robustness + local/global XAI + ablation",
                "dp": "Multi-epsilon evaluation",
                "simulator": "CoppeliaSim / ROS",
                "novelty_hook": "A unified four-axis evaluation framework: predictive quality, privacy, robustness and explainability.",
            },
        ]
    return [
        {
            "level": "Balanced", "title": f"Reproducible and explainable ML workflow for {domain}",
            "q1_potential": 82, "difficulty": "Medium", "time": "4–8 weeks", "hardware": "Domain-dependent",
            "estimated_cost": "Low", "question": "Can a reproducible ML benchmark improve transparency and robustness for this research problem?",
            "inputs": ["domain features"], "target": "research target",
            "method": "Open data + baseline ML + robustness + explainability", "dp": "Optional privacy analysis",
            "simulator": "Domain-specific", "novelty_hook": "Integrates reproducibility, robustness and explainability in a single research pipeline.",
        },
        {
            "level": "Advanced", "title": f"Privacy–utility and robustness benchmark for {domain}",
            "q1_potential": 88, "difficulty": "High", "time": "6–12 weeks", "hardware": "Optional",
            "estimated_cost": "Low–Medium", "question": "What trade-off exists among predictive quality, privacy and robustness?",
            "inputs": ["domain features"], "target": "research target",
            "method": "ML benchmark + privacy sweep + stress testing + XAI", "dp": "Multi-epsilon privacy evaluation",
            "simulator": "Domain-specific", "novelty_hook": "Evaluates accuracy, privacy, robustness and interpretability together instead of in isolation.",
        },
    ]


def recommend_datasets(domain: str) -> list[dict[str, Any]]:
    if domain == "Robotics & Mechanism Dynamics":
        return [
            {
                "name": "UCI Robot Execution Failures", "url": "https://archive.ics.uci.edu/",
                "fit": 88, "samples": "task-dependent", "features": "force / torque time-series descriptors",
                "difficulty": "Medium", "inputs": "force/torque signals", "target": "execution failure class",
                "missing": "verify source", "format": "tabular / time series", "license": "verify source terms",
                "doi": "verify current dataset citation", "why": "Useful for failure classification and robustness benchmarking in robotics.",
            },
            {
                "name": "NASA condition-monitoring collections", "url": "https://data.nasa.gov/",
                "fit": 80, "samples": "collection-dependent", "features": "sensor / condition-monitoring variables",
                "difficulty": "Medium–High", "inputs": "sensor telemetry", "target": "health / fault state",
                "missing": "collection-dependent", "format": "CSV / text", "license": "verify dataset page",
                "doi": "dataset-dependent", "why": "Useful for drift, degradation and robustness experiments when direct robot data are unavailable.",
            },
            {
                "name": "Nova Synthetic Robotics Benchmark", "url": "generated locally", "fit": 96,
                "samples": "500–20,000 configurable", "features": "joint/sensor/load variables", "difficulty": "Low",
                "inputs": "simulated robot telemetry", "target": "state", "missing": "controlled", "format": "CSV",
                "license": "generated by the user", "doi": "N/A", "why": "Fully reproducible and ideal for ablation, robustness, drift and privacy experiments.",
            },
        ]
    return [
        {
            "name": "OpenML dataset search", "url": "https://www.openml.org/", "fit": 80,
            "samples": "dataset-dependent", "features": "dataset-dependent", "difficulty": "Medium",
            "inputs": "domain-dependent", "target": "domain-dependent", "missing": "dataset-dependent",
            "format": "tabular", "license": "verify dataset page", "doi": "dataset-dependent",
            "why": "OpenML provides public ML datasets and metadata suitable for reproducible experiments.",
        },
        {
            "name": "Kaggle / public repository candidate", "url": "https://www.kaggle.com/datasets", "fit": 72,
            "samples": "dataset-dependent", "features": "dataset-dependent", "difficulty": "Medium",
            "inputs": "domain-dependent", "target": "domain-dependent", "missing": "dataset-dependent",
            "format": "varies", "license": "verify carefully", "doi": "often unavailable",
            "why": "Useful for discovery, but licensing and citation should be checked before publication.",
        },
    ]


def analyze_pdf(upload) -> dict[str, Any]:
    text = extract_pdf_text(upload)
    if len(clean_text(text)) < 120:
        return {"error": "Insufficient extractable text. The PDF may be scanned and require OCR."}
    domain = detect_domain(text)
    signals = extract_project_signals(text)
    return {
        "filename": upload.name,
        "text": text,
        "title": guess_title(text),
        "stats": text_stats(text),
        "signals": signals,
        "domain": domain,
        "scores": readiness_scores(domain, signals),
        "ideas": ideas_for_domain(domain["domain"]),
        "datasets": recommend_datasets(domain["domain"]),
    }


def crossref_search(title: str, rows: int = 10) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            "https://api.crossref.org/works",
            params={"query.title": title, "rows": rows, "select": "title,DOI,published,URL,author"},
            timeout=8,
            headers={"User-Agent": "NovaResearchAI/5.0 research prototype"},
        )
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])
        out: list[dict[str, Any]] = []
        for item in items:
            titles = item.get("title") or []
            work_title = titles[0] if titles else "Untitled"
            year: Any = "—"
            for key in ("published-print", "published-online", "published"):
                parts = item.get(key, {}).get("date-parts", [])
                if parts and parts[0]:
                    year = parts[0][0]
                    break
            out.append({
                "Title": work_title,
                "Year": year,
                "DOI": item.get("DOI") or "—",
                "URL": item.get("URL") or "",
                "Similarity": round(100 * SequenceMatcher(None, title.lower(), work_title.lower()).ratio(), 1),
            })
        return sorted(out, key=lambda x: x["Similarity"], reverse=True)
    except Exception:
        return []


def run_novelty_screen(title: str, method: str, dataset: str, simulator: str) -> dict[str, Any]:
    similar = crossref_search(title)
    max_similarity = max([x["Similarity"] for x in similar], default=25)
    method_terms = len(set(re.findall(r"[a-zA-Z]{4,}", method.lower())))
    title_score = int(np.clip(100 - max_similarity * 0.72, 20, 96))
    method_score = int(np.clip(56 + min(method_terms, 30) * 0.8 + 3 * bool(dataset) + 3 * bool(simulator), 35, 96))
    data_score = 82 if dataset.strip() else 45
    simulator_score = 84 if simulator.strip() else 45
    combo_score = int(np.clip(round(np.mean([title_score, method_score, data_score, simulator_score]) + 4), 20, 98))
    overall = int(round(0.30 * title_score + 0.34 * method_score + 0.16 * data_score + 0.10 * simulator_score + 0.10 * combo_score))
    return {
        "Title novelty": title_score,
        "Method novelty": method_score,
        "Data novelty": data_score,
        "Simulator novelty": simulator_score,
        "Combination novelty": combo_score,
        "Overall novelty": overall,
        "similar": similar,
        "screening_note": "Heuristic novelty screening only. Confirm final claims through a formal literature review and verified citations.",
    }


def build_gap_matrix(similar: list[dict[str, Any]], active_title: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    keywords = {
        "ML": ["machine learning", "learning", "classification", "prediction"],
        "Privacy": ["privacy", "differential"],
        "XAI": ["explain", "interpretable", "shap"],
        "Simulator": ["simulation", "simulator", "digital twin"],
        "Robustness": ["robust", "noise", "drift"],
    }
    for work in similar[:8]:
        title = str(work.get("Title", "")).lower()
        row = {"Paper": str(work.get("Title", ""))[:90]}
        for col, terms in keywords.items():
            row[col] = "Likely" if any(t in title for t in terms) else "Unknown"
        rows.append(row)
    rows.append({"Paper": active_title, "ML": "Planned", "Privacy": "Planned", "XAI": "Planned", "Simulator": "Planned", "Robustness": "Planned"})
    return rows
