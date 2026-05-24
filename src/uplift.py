"""Counterfactual uplift scoring and recommendations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import EXPERIENCE_IDS
from src.modeling import AuthModel, predict_proba


def score_counterfactuals(
  auth_model: AuthModel, feature_df: pd.DataFrame
) -> pd.DataFrame:
  """
  Score P(auth | X, experience=k) for all 7 experiences per session.

  Returns dataframe with columns p_exp_1 .. p_exp_7.
  """
  base = feature_df[auth_model.feature_cols].copy()
  n = len(base)
  prob_matrix = np.zeros((n, len(EXPERIENCE_IDS)))

  for i, exp_id in enumerate(EXPERIENCE_IDS):
    cf = base.copy()
    cf["experience_id_shown"] = exp_id
    prob_matrix[:, i] = predict_proba(auth_model, cf)

  prob_cols = {f"p_exp_{k}": prob_matrix[:, i] for i, k in enumerate(EXPERIENCE_IDS)}
  return pd.DataFrame(prob_cols)


def recommend_experiences(
  auth_model: AuthModel,
  feature_df: pd.DataFrame,
  meta_df: pd.DataFrame,
) -> pd.DataFrame:
  """
  Produce per-session recommendations and predicted uplift vs. shown experience.

  uplift = P(auth | X, best_experience) - P(auth | X, shown_experience)
  """
  probs = score_counterfactuals(auth_model, feature_df)
  prob_values = probs.values
  best_idx = prob_values.argmax(axis=1)
  recommended = np.array([EXPERIENCE_IDS[i] for i in best_idx])
  p_best = prob_values.max(axis=1)

  shown = meta_df["experience_id_shown"].values
  shown_idx = np.array([EXPERIENCE_IDS.index(int(s)) for s in shown])
  p_shown = prob_values[np.arange(len(shown)), shown_idx]

  result = meta_df[
    ["visitor_identifier", "visitor_session_id", "experience_id_shown", "authenticated"]
  ].copy()
  result["recommended_experience"] = recommended
  result["p_shown"] = p_shown
  result["p_best"] = p_best
  result["predicted_uplift"] = p_best - p_shown

  for i, exp_id in enumerate(EXPERIENCE_IDS):
    result[f"p_exp_{exp_id}"] = prob_values[:, i]
    result[f"uplift_exp_{exp_id}"] = prob_values[:, i] - p_shown

  return result


def uplift_summary(recommendations: pd.DataFrame) -> pd.Series:
  """Aggregate uplift statistics."""
  return pd.Series(
    {
      "n_sessions": len(recommendations),
      "pct_positive_uplift": (recommendations["predicted_uplift"] > 0).mean(),
      "mean_uplift": recommendations["predicted_uplift"].mean(),
      "median_uplift": recommendations["predicted_uplift"].median(),
      "pct_changed_experience": (
        recommendations["recommended_experience"] != recommendations["experience_id_shown"]
      ).mean(),
      "mean_p_shown": recommendations["p_shown"].mean(),
      "mean_p_best": recommendations["p_best"].mean(),
    }
  )
