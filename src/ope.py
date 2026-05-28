"""Offline policy evaluation with propensity scoring and doubly robust estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import EXPERIENCE_IDS, RANDOM_STATE
from src.modeling import AuthModel, train_auth_model
from src.uplift import score_counterfactuals


@dataclass
class PropensityModel:
  pipeline: Pipeline
  feature_cols: list[str]


def build_propensity_pipeline() -> Pipeline:
  return Pipeline(
    [
      ("scaler", StandardScaler()),
      (
        "model",
        LogisticRegression(
          max_iter=1000,
          solver="lbfgs",
          random_state=RANDOM_STATE,
        ),
      ),
    ]
  )


def train_propensity_model(
  train_df: pd.DataFrame, feature_cols: list[str]
) -> PropensityModel:
  """Fit multinomial propensity model pi(a | x)."""
  pipeline = build_propensity_pipeline()
  pipeline.fit(train_df[feature_cols], train_df["experience_id_shown"])
  return PropensityModel(pipeline=pipeline, feature_cols=feature_cols)


def predict_propensity(model: PropensityModel, df: pd.DataFrame) -> np.ndarray:
  """Return P(A=a|X) matrix with shape (n_sessions, n_experiences)."""
  proba = model.pipeline.predict_proba(df[model.feature_cols])
  classes = model.pipeline.named_steps["model"].classes_
  class_to_col = {int(exp_id): idx for idx, exp_id in enumerate(classes)}

  aligned = np.zeros((len(df), len(EXPERIENCE_IDS)))
  for j, exp_id in enumerate(EXPERIENCE_IDS):
    aligned[:, j] = proba[:, class_to_col[exp_id]]
  return aligned


def _arm_indices(actions: np.ndarray) -> np.ndarray:
  return np.array([EXPERIENCE_IDS.index(int(a)) for a in actions])


def stabilize_propensities(
  propensity_matrix: np.ndarray,
  min_propensity: float = 0.05,
) -> np.ndarray:
  """Clip propensities from below and renormalize rows to sum to 1."""
  clipped = np.clip(propensity_matrix, min_propensity, 1.0)
  return clipped / clipped.sum(axis=1, keepdims=True)


def cross_fitted_propensities(
  df: pd.DataFrame,
  feature_cols: list[str],
  n_folds: int = 5,
) -> np.ndarray:
  """Cross-fitted propensity scores pi(a | x) on df."""
  n = len(df)
  propensity_matrix = np.zeros((n, len(EXPERIENCE_IDS)))
  kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

  for train_idx, val_idx in kf.split(df):
    fold_train = df.iloc[train_idx]
    fold_val = df.iloc[val_idx]
    model = train_propensity_model(fold_train, feature_cols)
    propensity_matrix[val_idx] = predict_propensity(model, fold_val)

  return propensity_matrix


def cross_fitted_outcomes(
  df: pd.DataFrame,
  feature_cols: list[str],
  n_folds: int = 5,
) -> np.ndarray:
  """Cross-fitted counterfactual outcome estimates mu_a(x) on df."""
  n = len(df)
  outcome_matrix = np.zeros((n, len(EXPERIENCE_IDS)))
  kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

  for train_idx, val_idx in kf.split(df):
    fold_train = df.iloc[train_idx]
    fold_val = df.iloc[val_idx]
    auth_model = train_auth_model(fold_train, feature_cols)
    cf_probs = score_counterfactuals(auth_model, fold_val[feature_cols])
    outcome_matrix[val_idx] = cf_probs.values

  return outcome_matrix


def _clip_importance_weights(weights: np.ndarray, max_weight: float | None) -> np.ndarray:
  if max_weight is None:
    return weights
  return np.clip(weights, 0.0, max_weight)


def policy_value_dm(outcome_matrix: np.ndarray, policy_actions: np.ndarray) -> float:
  """Direct method: mean predicted outcome under deterministic policy."""
  policy_idx = _arm_indices(policy_actions)
  return outcome_matrix[np.arange(len(policy_actions)), policy_idx].mean()


def policy_value_ips(
  y: np.ndarray,
  treatment: np.ndarray,
  propensity_matrix: np.ndarray,
  policy_actions: np.ndarray,
  max_weight: float | None = 100.0,
) -> tuple[float, np.ndarray]:
  """IPS policy value for a deterministic target policy."""
  treatment_idx = _arm_indices(treatment)
  policy_idx = _arm_indices(policy_actions)
  prop_treated = propensity_matrix[np.arange(len(y)), treatment_idx]
  match = treatment == policy_actions
  weights = _clip_importance_weights(match / np.clip(prop_treated, 1e-6, None), max_weight)
  contributions = weights * y
  return contributions.mean(), contributions


def policy_value_snips(
  y: np.ndarray,
  treatment: np.ndarray,
  propensity_matrix: np.ndarray,
  policy_actions: np.ndarray,
  max_weight: float | None = 100.0,
) -> float:
  """Self-normalized IPS for a deterministic target policy."""
  treatment_idx = _arm_indices(treatment)
  prop_treated = propensity_matrix[np.arange(len(y)), treatment_idx]
  match = (treatment == policy_actions).astype(float)
  raw_weights = _clip_importance_weights(
    match / np.clip(prop_treated, 1e-6, None),
    max_weight,
  )
  return (raw_weights * y).sum() / np.clip(raw_weights.sum(), 1e-6, None)


def policy_value_dr(
  y: np.ndarray,
  treatment: np.ndarray,
  outcome_matrix: np.ndarray,
  propensity_matrix: np.ndarray,
  policy_actions: np.ndarray,
  max_weight: float | None = 100.0,
) -> tuple[float, np.ndarray]:
  """Doubly robust policy value for a deterministic target policy."""
  treatment_idx = _arm_indices(treatment)
  policy_idx = _arm_indices(policy_actions)
  mu_policy = outcome_matrix[np.arange(len(y)), policy_idx]
  mu_treated = outcome_matrix[np.arange(len(y)), treatment_idx]
  prop_treated = propensity_matrix[np.arange(len(y)), treatment_idx]
  match = treatment == policy_actions
  weights = _clip_importance_weights(
    match.astype(float) / np.clip(prop_treated, 1e-6, None),
    max_weight,
  )
  contributions = mu_policy + weights * (y - mu_treated)
  return contributions.mean(), contributions


def policy_value_uniform_dr(
  y: np.ndarray,
  treatment: np.ndarray,
  outcome_matrix: np.ndarray,
  propensity_matrix: np.ndarray,
  max_weight: float | None = 100.0,
) -> tuple[float, np.ndarray]:
  """Doubly robust value of uniform random policy pi(a|x) = 1/K."""
  n_arms = len(EXPERIENCE_IDS)
  treatment_idx = _arm_indices(treatment)
  prop_treated = propensity_matrix[np.arange(len(y)), treatment_idx]
  mu_treated = outcome_matrix[np.arange(len(y)), treatment_idx]
  correction = (y - mu_treated) / np.clip(prop_treated, 1e-6, None)
  if max_weight is not None:
    correction = np.clip(correction, -max_weight, max_weight)

  per_arm = outcome_matrix + correction[:, None]
  contributions = per_arm.mean(axis=1)
  return contributions.mean(), contributions


def bootstrap_ci(
  contributions: np.ndarray,
  n_bootstrap: int = 500,
  alpha: float = 0.05,
  random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
  """Percentile bootstrap CI for the mean of per-session contributions."""
  rng = np.random.default_rng(random_state)
  n = len(contributions)
  if n == 0:
    return np.nan, np.nan

  boot_means = np.empty(n_bootstrap)
  for b in range(n_bootstrap):
    sample = contributions[rng.integers(0, n, size=n)]
    boot_means[b] = sample.mean()

  lower = float(np.quantile(boot_means, alpha / 2))
  upper = float(np.quantile(boot_means, 1 - alpha / 2))
  return lower, upper


def overlap_diagnostics(
  propensity_matrix: np.ndarray,
  treatment: np.ndarray,
) -> pd.DataFrame:
  """Summarize propensity overlap for observed treatments."""
  treatment_idx = _arm_indices(treatment)
  prop_treated = propensity_matrix[np.arange(len(treatment)), treatment_idx]
  rows = []
  for j, exp_id in enumerate(EXPERIENCE_IDS):
    arm_mask = treatment == exp_id
    if not arm_mask.any():
      continue
    arm_props = propensity_matrix[arm_mask, j]
    rows.append(
      {
        "experience_id": exp_id,
        "n_sessions": int(arm_mask.sum()),
        "mean_propensity": arm_props.mean(),
        "min_propensity": arm_props.min(),
        "pct_below_0.05": (arm_props < 0.05).mean(),
        "pct_below_0.10": (arm_props < 0.10).mean(),
      }
    )
  summary = pd.DataFrame(rows)
  summary["overall_prop_treated_mean"] = prop_treated.mean()
  summary["overall_prop_treated_min"] = prop_treated.min()
  return summary


def evaluate_deterministic_policy(
  name: str,
  y: np.ndarray,
  treatment: np.ndarray,
  outcome_matrix: np.ndarray,
  propensity_matrix: np.ndarray,
  policy_actions: np.ndarray,
  n_bootstrap: int = 500,
  max_weight: float | None = 100.0,
) -> pd.Series:
  """Evaluate one deterministic policy with DM, IPS, SNIPS, and DR."""
  dm = policy_value_dm(outcome_matrix, policy_actions)
  ips, ips_contrib = policy_value_ips(
    y, treatment, propensity_matrix, policy_actions, max_weight=max_weight
  )
  snips = policy_value_snips(
    y, treatment, propensity_matrix, policy_actions, max_weight=max_weight
  )
  dr, dr_contrib = policy_value_dr(
    y, treatment, outcome_matrix, propensity_matrix, policy_actions, max_weight=max_weight
  )
  dr_lo, dr_hi = bootstrap_ci(dr_contrib, n_bootstrap=n_bootstrap)

  row = {
    "policy": name,
    "dm_auth_rate": dm,
    "ips_auth_rate": ips,
    "snips_auth_rate": snips,
    "dr_auth_rate": dr,
    "dr_ci_lower": dr_lo,
    "dr_ci_upper": dr_hi,
    "pct_policy_matches_logging": (policy_actions == treatment).mean(),
  }
  if name == "logging":
    row["actual_auth_rate"] = y.mean()
  else:
    row["actual_auth_rate"] = np.nan

  return pd.Series(row)


def compare_policies_oof(
  df: pd.DataFrame,
  feature_cols: list[str],
  policies: dict[str, np.ndarray],
  n_folds: int = 5,
  n_bootstrap: int = 500,
  max_weight: float | None = 100.0,
  min_propensity: float = 0.05,
  include_uniform: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
  """
  Cross-fitted offline policy evaluation on df.

  Returns (results_df, propensity_matrix, outcome_matrix).
  """
  y = df["authenticated"].values.astype(float)
  treatment = df["experience_id_shown"].values

  propensity_matrix = stabilize_propensities(
    cross_fitted_propensities(df, feature_cols, n_folds=n_folds),
    min_propensity=min_propensity,
  )
  outcome_matrix = cross_fitted_outcomes(df, feature_cols, n_folds=n_folds)

  return _compare_policies_from_matrices(
    y,
    treatment,
    outcome_matrix,
    propensity_matrix,
    policies,
    n_bootstrap=n_bootstrap,
    max_weight=max_weight,
    include_uniform=include_uniform,
  )


def compare_policies_holdout(
  train_df: pd.DataFrame,
  val_df: pd.DataFrame,
  feature_cols: list[str],
  policies: dict[str, np.ndarray],
  auth_model: AuthModel | None = None,
  n_bootstrap: int = 500,
  max_weight: float | None = 100.0,
  min_propensity: float = 0.05,
  include_uniform: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
  """
  Offline policy evaluation with models fit on train and scored on val.

  Preferred when a visitor-level train/validation split already exists.
  """
  if auth_model is None:
    auth_model = train_auth_model(train_df, feature_cols)

  prop_model = train_propensity_model(train_df, feature_cols)
  propensity_matrix = stabilize_propensities(
    predict_propensity(prop_model, val_df),
    min_propensity=min_propensity,
  )
  outcome_matrix = score_counterfactuals(auth_model, val_df[feature_cols]).values

  y = val_df["authenticated"].values.astype(float)
  treatment = val_df["experience_id_shown"].values

  return _compare_policies_from_matrices(
    y,
    treatment,
    outcome_matrix,
    propensity_matrix,
    policies,
    n_bootstrap=n_bootstrap,
    max_weight=max_weight,
    include_uniform=include_uniform,
  )


def _compare_policies_from_matrices(
  y: np.ndarray,
  treatment: np.ndarray,
  outcome_matrix: np.ndarray,
  propensity_matrix: np.ndarray,
  policies: dict[str, np.ndarray],
  n_bootstrap: int = 500,
  max_weight: float | None = 100.0,
  include_uniform: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
  rows = [
    evaluate_deterministic_policy(
      name,
      y,
      treatment,
      outcome_matrix,
      propensity_matrix,
      actions,
      n_bootstrap=n_bootstrap,
      max_weight=max_weight,
    )
    for name, actions in policies.items()
  ]

  if include_uniform:
    dr, dr_contrib = policy_value_uniform_dr(
      y, treatment, outcome_matrix, propensity_matrix, max_weight=max_weight
    )
    dr_lo, dr_hi = bootstrap_ci(dr_contrib, n_bootstrap=n_bootstrap)
    rows.append(
      pd.Series(
        {
          "policy": "uniform_random",
          "dm_auth_rate": outcome_matrix.mean(),
          "ips_auth_rate": np.nan,
          "snips_auth_rate": np.nan,
          "dr_auth_rate": dr,
          "dr_ci_lower": dr_lo,
          "dr_ci_upper": dr_hi,
          "pct_policy_matches_logging": np.nan,
        }
      )
    )

  results = pd.DataFrame(rows).set_index("policy")
  return results, propensity_matrix, outcome_matrix
