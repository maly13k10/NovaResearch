from __future__ import annotations

import io
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def fig_to_png(fig) -> bytes:
    buff = io.BytesIO()
    fig.savefig(buff, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    return buff.getvalue()


def novelty_radar_figure(novelty: dict[str, Any]):
    labels = ["Title", "Method", "Data", "Simulator", "Combination"]
    vals = [
        novelty.get("Title novelty", 0), novelty.get("Method novelty", 0), novelty.get("Data novelty", 0),
        novelty.get("Simulator novelty", 0), novelty.get("Combination novelty", 0),
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals2, angles2 = vals + vals[:1], angles + angles[:1]
    fig = plt.figure(figsize=(5.6, 4.6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles2, vals2, marker="o")
    ax.fill(angles2, vals2, alpha=0.12)
    ax.set_xticks(angles); ax.set_xticklabels(labels); ax.set_ylim(0, 100); ax.set_title("Novelty Radar")
    return fig


def novelty_heatmap_figure(novelty: dict[str, Any]):
    labels = ["Title", "Method", "Data", "Simulator", "Combination", "Overall"]
    vals = [[
        novelty.get("Title novelty", 0), novelty.get("Method novelty", 0), novelty.get("Data novelty", 0),
        novelty.get("Simulator novelty", 0), novelty.get("Combination novelty", 0), novelty.get("Overall novelty", 0),
    ]]
    fig, ax = plt.subplots(figsize=(8.2, 2.0))
    image = ax.imshow(vals, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks([0]); ax.set_yticklabels(["Score"])
    for j, value in enumerate(vals[0]):
        ax.text(j, 0, f"{value}", ha="center", va="center")
    ax.set_title("Novelty Heatmap")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04)
    return fig


def research_graph_figure(state: dict[str, Any], active_idea: dict[str, Any] | None):
    nodes = ["PDF", "Idea", "Data", "Novelty", "Experiment", "ML", "DP", "XAI", "Publication"]
    edges = list(zip(nodes[:-1], nodes[1:]))
    G = nx.DiGraph(); G.add_nodes_from(nodes); G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42, k=1.0)
    status = {
        "PDF": state.get("project") is not None,
        "Idea": active_idea is not None,
        "Data": state.get("active_df") is not None,
        "Novelty": state.get("novelty") is not None,
        "Experiment": state.get("experiment_plan") is not None,
        "ML": state.get("ml_results") is not None,
        "DP": state.get("dp_results") is not None,
        "XAI": state.get("robustness_results") is not None,
        "Publication": bool(state.get("publication_ready")),
    }
    fig, ax = plt.subplots(figsize=(10, 5.2))
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=18, width=1.5)
    completed = [n for n in nodes if status[n]]
    pending = [n for n in nodes if not status[n]]
    nx.draw_networkx_nodes(G, pos, nodelist=completed, ax=ax, node_size=1800)
    nx.draw_networkx_nodes(G, pos, nodelist=pending, ax=ax, node_size=1800, alpha=0.35)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
    ax.set_title("Research Graph · PDF → Publication")
    ax.axis("off")
    return fig


def comparison_radar_figure(df: pd.DataFrame):
    metrics = ["Research potential", "Readiness", "Novelty", "Reproducibility", "Journal fit"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, polar=True)
    for _, row in df.iterrows():
        vals = [float(row.get(m, 0) or 0) for m in metrics]; vals += vals[:1]
        ax.plot(angles, vals, marker="o", label=str(row.get("Idea", "Idea"))[:28])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics); ax.set_ylim(0, 100); ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15))
    ax.set_title("Idea Comparison Arena")
    return fig
