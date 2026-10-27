from __future__ import annotations

import json
import os
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def dataframe_summary(df, rows: int = 8):
    if df is None:
        return None
    try:
        return {
            "shape": list(df.shape),
            "columns": [str(c) for c in df.columns],
            "head": df.head(rows).replace({np.nan: None}).to_dict("records"),
        }
    except Exception:
        return None


def build_context(state: dict[str, Any], active_idea: dict[str, Any] | None, next_action: tuple[str, str], journey: tuple[list[tuple[str, bool]], int]) -> dict[str, Any]:
    project = state.get("project")
    if not project:
        return {"has_project": False, "summary": "No active research project."}
    stages, completed = journey
    scores = project.get("scores", {})
    return {
        "has_project": True,
        "project_title": project.get("title", "Untitled Research Project"),
        "domain": project.get("domain", {}).get("domain", "Unknown"),
        "domain_confidence": project.get("domain", {}).get("confidence", 0),
        "readiness_overall": scores.get("Overall"),
        "journal_fit": scores.get("Journal fit"),
        "reproducibility": scores.get("Reproducibility"),
        "innovation_potential": scores.get("Innovation potential"),
        "missing": scores.get("Missing", []),
        "active_idea": active_idea,
        "datasets": [d.get("name", "Unknown") for d in project.get("datasets", [])],
        "novelty": state.get("novelty"),
        "ml_results": dataframe_summary(state.get("ml_results")),
        "dp_results": dataframe_summary(state.get("dp_results")),
        "robustness_noise": dataframe_summary((state.get("robustness_results") or {}).get("noise") if state.get("robustness_results") else None),
        "robustness_missing": dataframe_summary((state.get("robustness_results") or {}).get("missing") if state.get("robustness_results") else None),
        "ablation_results": dataframe_summary(state.get("ablation_results")),
        "drift_results": dataframe_summary(state.get("drift_results")),
        "importance_results": dataframe_summary(state.get("importance_results")),
        "shap_global": dataframe_summary(state.get("shap_global")),
        "shap_local": dataframe_summary(state.get("shap_local")),
        "statistical_validation": state.get("statistical_validation"),
        "experiment_plan": state.get("experiment_plan"),
        "selected_journal": state.get("selected_journal"),
        "research_story": state.get("research_story"),
        "research_stage": completed,
        "research_stage_total": len(stages),
        "next_action_title": next_action[0],
        "next_action_text": next_action[1],
        "project_text": project.get("text", "")[:12000],
    }


def retrieve_project_evidence(project_text: str, query: str, k: int = 3) -> list[str]:
    text = re.sub(r"\s+", " ", project_text or "").strip()
    if not text or not query.strip():
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < 700:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    chunks = [c for c in chunks if len(c) > 60][:120]
    if not chunks:
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        matrix = vectorizer.fit_transform(chunks + [query])
        sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        idx = np.argsort(sims)[::-1][:k]
        return [chunks[i] for i in idx if sims[i] > 0]
    except Exception:
        return chunks[:k]


def _previous_assistant(history: list[dict[str, str]]) -> str:
    for message in reversed(history[:-1]):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def local_context_answer(prompt: str, context: dict[str, Any], history: list[dict[str, str]]) -> str:
    p = prompt.lower().strip()
    idea = context.get("active_idea") or {}
    evidence = retrieve_project_evidence(context.get("project_text", ""), prompt, k=3)
    evidence_block = "\n\n".join(f"- {e}" for e in evidence)
    previous = _previous_assistant(history)

    if any(x in p for x in ["more details", "mai multe", "detalia", "why", "de ce", "explain that", "explica mai mult"]):
        return (
            "### Follow-up explanation\n\n"
            f"The current recommendation is **{context.get('next_action_title')}** because the project is at stage "
            f"**{context.get('research_stage')}/{context.get('research_stage_total')}**. {context.get('next_action_text')}\n\n"
            f"**Previous Nova answer:** {previous[:1100] if previous else 'No previous assistant answer.'}\n\n"
            + (f"**Relevant evidence from the uploaded PDF:**\n{evidence_block}\n\n" if evidence_block else "")
            + "Use the relevant module to generate measurable evidence before advancing to the next stage."
        )

    structured = []
    if "next" in p or "urm" in p:
        structured.append(f"**Next action:** {context.get('next_action_title')} — {context.get('next_action_text')}")
    if any(x in p for x in ["weak", "slab", "problem", "gap"]):
        gaps = context.get("missing") or []
        structured.append("**Current gaps:** " + (", ".join(gaps) if gaps else "no major structural gaps in the automated screening"))
    if "novel" in p:
        n = context.get("novelty") or {}
        structured.append(f"**Novelty:** overall={n.get('Overall novelty', 'not analyzed')}, title={n.get('Title novelty', '—')}, method={n.get('Method novelty', '—')}")
    if "dataset" in p or "data" in p:
        structured.append("**Data candidates:** " + ", ".join(context.get("datasets") or ["none selected"]))
    if "privacy" in p or "epsilon" in p or "dp" in p:
        dp = context.get("dp_results")
        structured.append("**DP status:** " + ("results available" if dp else "not yet run"))
    if "model" in p or "ml" in p:
        ml = context.get("ml_results") or {}
        head = ml.get("head") or []
        if head:
            structured.append(f"**Best ML result:** {head[0].get('Model')} with F1={head[0].get('F1')}")
        else:
            structured.append("**ML status:** benchmark not yet run")
    if "journal" in p or "venue" in p:
        j = context.get("selected_journal")
        structured.append(f"**Selected venue:** {j.get('Venue') if isinstance(j, dict) else 'not selected'}")
    if "idea" in p or "research question" in p:
        structured.append(f"**Active idea:** {idea.get('title', '—')}\n\n**Research question:** {idea.get('question', '—')}\n\n**Method:** {idea.get('method', '—')}")
    if "score" in p or "readiness" in p:
        structured.append(
            f"**Readiness:** {context.get('readiness_overall')}/100 · Reproducibility {context.get('reproducibility')}/100 · "
            f"Innovation {context.get('innovation_potential')}/100 · Journal fit {context.get('journal_fit')}/100"
        )

    if not structured:
        structured.append(
            f"I am analyzing **{context.get('project_title')}** in **{context.get('domain')}** with active idea **{idea.get('title', '—')}**."
        )
        structured.append(f"Current next action: **{context.get('next_action_title')}**.")

    if evidence_block:
        structured.append("**Relevant PDF evidence:**\n" + evidence_block)
    structured.append("Ask a follow-up such as *why?*, *compare the options*, *what evidence supports this?*, or *what should I show my professor?*")
    return "\n\n".join(structured)


def ask_openai(prompt: str, context: dict[str, Any], history: list[dict[str, str]], api_key: str | None = None, model: str | None = None) -> tuple[str | None, str | None]:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "No OpenAI API key configured."
    model = model or os.getenv("NOVA_OPENAI_MODEL", "gpt-6-luna")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        recent = history[-12:]
        conversation = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in recent)
        response = client.responses.create(
            model=model,
            instructions=(
                "You are NovaResearch AI, a rigorous research copilot. Use the supplied project context and uploaded-PDF evidence. "
                "Clearly separate evidence, inference and recommendation. Never claim publication acceptance or absolute novelty. "
                "Only describe Differential Privacy as formal when the context says a formal DP mechanism was used. "
                "Be concise, technical, actionable and suitable for a university research project."
            ),
            input=(
                "PROJECT CONTEXT:\n" + json.dumps(context, ensure_ascii=False, default=str)[:24000]
                + "\n\nRECENT CONVERSATION:\n" + conversation[:10000]
                + "\n\nUSER QUESTION:\n" + prompt
            ),
        )
        return response.output_text, None
    except Exception as exc:
        return None, str(exc)
