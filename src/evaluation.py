"""Model and policy evaluation metrics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
  average_precision_score,
  brier_score_loss,
  log_loss,
  roc_auc_score,
)

from src.modeling import AuthModel, predict_proba


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> pd.Series:
  """Compute standard binary classification metrics."""
  return pd.Series(
    {
      "roc_auc": roc_auc_score(y_true, y_prob),
      "pr_auc": average_precision_score(y_true, y_prob),
      "log_loss": log_loss(y_true, y_prob),
      "brier_score": brier_score_loss(y_true, y_prob),
      "auth_rate": y_true.mean(),
      "mean_predicted_prob": y_prob.mean(),
    }
  )


def evaluate_model(
  auth_model: AuthModel, df: pd.DataFrame, split_name: str
) -> pd.Series:
  """Evaluate model on a dataset split."""
  y_true = df["authenticated"].values
  y_prob = predict_proba(auth_model, df)
  metrics = classification_metrics(y_true, y_prob)
  metrics.name = split_name
  return metrics


def observational_auth_by_experience(df: pd.DataFrame) -> pd.DataFrame:
  """Actual auth rate by shown experience."""
  return (
    df.groupby("experience_id_shown")["authenticated"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "auth_rate"})
    .reset_index()
  )


def policy_comparison(recommendations: pd.DataFrame) -> pd.DataFrame:
  """
  Compare historical vs. recommended policy using model-based estimates.

  Historical: actual auth outcomes for shown experience.
  Recommended: predicted auth prob under recommended experience.
  """
  historical_auth_rate = recommendations["authenticated"].mean()
  recommended_auth_rate_est = recommendations["p_best"].mean()
  shown_pred_rate = recommendations["p_shown"].mean()

  return pd.DataFrame(
    {
      "policy": ["historical_actual", "historical_model_pred", "recommended_model_est"],
      "auth_rate": [historical_auth_rate, shown_pred_rate, recommended_auth_rate_est],
    }
  )


def plot_calibration(
  y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> plt.Figure:
  """Reliability diagram for predicted auth probabilities."""
  prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
  fig, ax = plt.subplots(figsize=(6, 5))
  ax.plot(prob_pred, prob_true, marker="o", label="Model")
  ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
  ax.set_xlabel("Mean predicted probability")
  ax.set_ylabel("Fraction authenticated")
  ax.set_title("Calibration Curve (Validation)")
  ax.legend()
  fig.tight_layout()
  return fig


def plot_uplift_distribution(recommendations: pd.DataFrame) -> plt.Figure:
  """Histogram of predicted uplift on validation sessions."""
  fig, ax = plt.subplots(figsize=(8, 4))
  ax.hist(recommendations["predicted_uplift"], bins=30, edgecolor="white")
  ax.axvline(0, color="red", linestyle="--", label="Zero uplift")
  ax.set_xlabel("Predicted uplift (p_best - p_shown)")
  ax.set_ylabel("Sessions")
  ax.set_title("Predicted Uplift Distribution")
  ax.legend()
  fig.tight_layout()
  return fig
