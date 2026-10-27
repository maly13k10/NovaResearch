from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def permutation_feature_importance(model, X_test: pd.DataFrame, y_test) -> pd.DataFrame:
    imp = permutation_importance(model, X_test, y_test, scoring="f1_weighted", n_repeats=8, random_state=42)
    return pd.DataFrame({"Feature": X_test.columns, "Importance": imp.importances_mean}).sort_values("Importance", ascending=False).reset_index(drop=True)


def shap_global_local(model, X_test: pd.DataFrame, row_index: int = 0) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    try:
        import shap
        sample = X_test.sample(min(140, len(X_test)), random_state=42)
        row_index = int(max(0, min(row_index, len(X_test) - 1)))
        selected = X_test.iloc[[row_index]]

        if isinstance(model, Pipeline) and "scale" in model.named_steps and "model" in model.named_steps:
            scaler = model.named_steps["scale"]
            estimator = model.named_steps["model"]
            background = scaler.transform(sample)
            selected_arr = scaler.transform(selected)
            explainer = shap.Explainer(estimator, background)
            global_exp = explainer(background)
            local_exp = explainer(selected_arr)
            feature_names = list(X_test.columns)
        else:
            explainer = shap.Explainer(model, sample)
            global_exp = explainer(sample)
            local_exp = explainer(selected)
            feature_names = list(X_test.columns)

        values = np.asarray(global_exp.values)
        if values.ndim == 3:
            mean_abs = np.mean(np.abs(values), axis=(0, 2))
        else:
            mean_abs = np.mean(np.abs(values), axis=0)
        global_df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs}).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)

        local_values = np.asarray(local_exp.values)
        if local_values.ndim == 3:
            local_values = np.mean(local_values[0], axis=1)
        elif local_values.ndim == 2:
            local_values = local_values[0]
        else:
            local_values = local_values.reshape(-1)
        local_df = pd.DataFrame({"Feature": feature_names, "SHAP contribution": local_values, "Feature value": selected.iloc[0].values}).sort_values("SHAP contribution", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
        return global_df, local_df, "Real SHAP global and local explanations computed."
    except Exception as exc:
        return None, None, f"SHAP could not be computed for this model: {exc}"
