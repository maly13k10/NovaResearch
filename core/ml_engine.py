from __future__ import annotations

from math import comb
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def robotics_dataset(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    joint_pos = rng.normal(0, 1.1, n)
    joint_vel = rng.normal(0, 0.8, n)
    torque = 0.65 * joint_pos + 0.45 * joint_vel + rng.normal(0, 0.45, n)
    vibration = np.abs(rng.normal(0.35 + 0.15 * np.abs(joint_vel), 0.18, n))
    temperature = 38 + 5 * np.abs(torque) + rng.normal(0, 2.2, n)
    payload = rng.uniform(0.2, 8.0, n)
    sensor_noise = rng.uniform(0.0, 0.25, n)
    energy = np.abs(torque * joint_vel) + 0.08 * payload + rng.normal(0, 0.12, n)
    risk = 0.9 * vibration + 0.045 * (temperature - 40) + 0.15 * np.abs(torque) + 0.9 * sensor_noise + rng.normal(0, 0.18, n)
    state = np.where(risk > 1.2, "fault", np.where(risk > 0.72, "warning", "normal"))
    return pd.DataFrame({
        "joint_position": joint_pos, "joint_velocity": joint_vel, "torque": torque,
        "vibration": vibration, "temperature": temperature, "payload": payload,
        "sensor_noise": sensor_noise, "energy_proxy": energy, "state": state,
    })


def encode_dataframe(df: pd.DataFrame, target: str):
    data = df.copy()
    y_raw = data.pop(target)
    y_codes, uniques = pd.factorize(y_raw.astype(str))
    X = pd.get_dummies(data, dummy_na=True)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X.astype(float), y_codes, [str(x) for x in uniques]


def model_zoo() -> dict[str, Any]:
    return {
        "Logistic Regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2500, random_state=42))]),
        "Random Forest": RandomForestClassifier(n_estimators=280, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM-RBF": Pipeline([("scale", StandardScaler()), ("model", SVC(C=2.0, probability=True, random_state=42))]),
    }


def run_benchmark(df: pd.DataFrame, target: str):
    X, y, labels = encode_dataframe(df, target)
    counts = np.bincount(y) if len(y) else np.array([])
    stratify = y if len(np.unique(y)) > 1 and len(counts) and counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    rows, fitted, preds = [], {}, {}
    for name, model in model_zoo().items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
            rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred), "Precision": p, "Recall": r, "F1": f1})
            fitted[name] = model
            preds[name] = pred
        except Exception as exc:
            rows.append({"Model": name, "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan, "F1": np.nan, "Error": str(exc)})
    results = pd.DataFrame(rows).sort_values("F1", ascending=False, na_position="last").reset_index(drop=True)
    return results, fitted, (X_train, X_test, y_train, y_test), preds, labels


def run_dp_sweep(df: pd.DataFrame, target: str, epsilons: list[float]) -> tuple[pd.DataFrame, str]:
    X, y, _ = encode_dataframe(df, target)
    counts = np.bincount(y) if len(y) else np.array([])
    stratify = y if len(np.unique(y)) > 1 and len(counts) and counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_test)
    # Bound standardized values so data_norm is finite and explicit.
    clip = 5.0
    Xt = np.clip(Xt, -clip, clip)
    Xv = np.clip(Xv, -clip, clip)
    data_norm = float(np.sqrt(Xt.shape[1] * clip**2))
    rows: list[dict[str, Any]] = []
    engine = "fallback"
    try:
        from diffprivlib.models import LogisticRegression as DPLogisticRegression
        engine = "diffprivlib"
        for eps in epsilons:
            model = DPLogisticRegression(epsilon=float(eps), data_norm=data_norm, max_iter=2000, random_state=42)
            model.fit(Xt, y_train)
            pred = model.predict(Xv)
            rows.append({
                "Model": "DP Logistic Regression", "Epsilon": float(eps), "Mechanism": "diffprivlib",
                "Accuracy": accuracy_score(y_test, pred),
                "F1": f1_score(y_test, pred, average="weighted", zero_division=0),
                "Data norm": data_norm,
            })
    except Exception:
        rng = np.random.default_rng(42)
        for eps in epsilons:
            scale = 0.35 / max(float(eps), 0.05)
            noisy = Xt + rng.laplace(0.0, scale, Xt.shape)
            model = LogisticRegression(max_iter=2000, random_state=42)
            model.fit(noisy, y_train)
            pred = model.predict(Xv)
            rows.append({
                "Model": "Logistic Regression", "Epsilon": float(eps), "Mechanism": "privacy-utility fallback",
                "Accuracy": accuracy_score(y_test, pred),
                "F1": f1_score(y_test, pred, average="weighted", zero_division=0),
                "Data norm": data_norm,
            })
    note = (
        "Formal DP logistic-regression engine (diffprivlib) used with clipped standardized features and an explicit data norm."
        if engine == "diffprivlib"
        else "diffprivlib was unavailable; results are a privacy–utility noise simulation and must not be reported as a formal DP guarantee."
    )
    return pd.DataFrame(rows), note


def privacy_advisor(dp: pd.DataFrame | None) -> str:
    if dp is None or dp.empty or "F1" not in dp:
        return "Run Differential Privacy first."
    clean = dp.dropna(subset=["F1"]).copy()
    if clean.empty:
        return "No valid DP result is available."
    f1_max = float(clean["F1"].max())
    threshold = f1_max - 0.03
    eligible = clean[clean["F1"] >= threshold].sort_values("Epsilon")
    row = eligible.iloc[0] if len(eligible) else clean.sort_values("F1", ascending=False).iloc[0]
    return f"Balanced candidate: ε={row['Epsilon']:.3g}, F1={row['F1']:.3f}. It is the smallest tested epsilon within ~0.03 F1 of the best observed utility."


def robustness_noise(model, X_test: pd.DataFrame, y_test) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    std = np.asarray(X_test.std(axis=0).replace(0, 1))
    for level in [0, 0.03, 0.06, 0.10, 0.16, 0.24]:
        noisy = X_test.to_numpy() + rng.normal(0, level, X_test.shape) * std
        pred = model.predict(pd.DataFrame(noisy, columns=X_test.columns, index=X_test.index))
        rows.append({"Noise level": level, "F1": f1_score(y_test, pred, average="weighted", zero_division=0)})
    return pd.DataFrame(rows)


def robustness_missing(model, X_test: pd.DataFrame, y_test) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    med = X_test.median().to_numpy()
    rows = []
    for ratio in [0, 0.03, 0.06, 0.10, 0.15, 0.20]:
        arr = X_test.to_numpy().copy()
        mask = rng.random(arr.shape) < ratio
        for j in range(arr.shape[1]):
            arr[mask[:, j], j] = med[j]
        pred = model.predict(pd.DataFrame(arr, columns=X_test.columns, index=X_test.index))
        rows.append({"Missing ratio": ratio, "F1": f1_score(y_test, pred, average="weighted", zero_division=0)})
    return pd.DataFrame(rows)


def ablation_study(model, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train, y_test, importance: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    baseline = f1_score(y_test, model.predict(X_test), average="weighted", zero_division=0)
    rows = []
    for feature in importance.head(limit)["Feature"]:
        cols = [c for c in X_train.columns if c != feature]
        m = clone(model)
        try:
            m.fit(X_train[cols], y_train)
            f1 = f1_score(y_test, m.predict(X_test[cols]), average="weighted", zero_division=0)
            rows.append({"Removed feature": feature, "Baseline F1": baseline, "Ablated F1": f1, "F1 drop": baseline - f1})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["Removed feature", "Baseline F1", "Ablated F1", "F1 drop"])
    return pd.DataFrame(rows).sort_values("F1 drop", ascending=False).reset_index(drop=True)


def drift_study(model, X_test: pd.DataFrame, y_test) -> pd.DataFrame:
    std = X_test.std().replace(0, 1)
    rows = []
    for level in [0, 0.10, 0.20, 0.35, 0.50, 0.75]:
        drifted = X_test + level * std
        pred = model.predict(drifted)
        rows.append({"Drift level (σ)": level, "F1": f1_score(y_test, pred, average="weighted", zero_division=0)})
    return pd.DataFrame(rows)


def mcnemar_exact(y_true, pred_a, pred_b, name_a: str, name_b: str) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    a_correct = a == y_true
    b_correct = b == y_true
    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))
    discordant = n01 + n10
    p = 1.0 if discordant == 0 else float(binomtest(min(n01, n10), n=discordant, p=0.5, alternative="two-sided").pvalue)
    return {
        "Model A": name_a,
        "Model B": name_b,
        "A correct / B wrong": n01,
        "A wrong / B correct": n10,
        "Discordant": discordant,
        "p_value": p,
        "Significant at 0.05": bool(p < 0.05),
    }
