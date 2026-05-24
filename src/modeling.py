"""Logistic regression training and prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import EXPERIENCE_IDS, RANDOM_STATE


@dataclass
class AuthModel:
  pipeline: Pipeline
  feature_cols: list[str]
  experience_col: str = "experience_id_shown"


def build_pipeline(feature_cols: list[str]) -> Pipeline:
  """Pipeline: scale numeric features, one-hot encode experience, fit LR."""
  numeric_transformer = Pipeline([("scaler", StandardScaler())])
  experience_transformer = OneHotEncoder(
    categories=[EXPERIENCE_IDS],
    drop="first",
    sparse_output=False,
    handle_unknown="error",
  )

  preprocessor = ColumnTransformer(
    transformers=[
      ("num", numeric_transformer, feature_cols),
      ("exp", experience_transformer, ["experience_id_shown"]),
    ]
  )

  model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
  )

  return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_auth_model(
  train_df: pd.DataFrame, feature_cols: list[str]
) -> AuthModel:
  """Train logistic regression on features + shown experience."""
  pipeline = build_pipeline(feature_cols)
  X = train_df[feature_cols + ["experience_id_shown"]]
  y = train_df["authenticated"]
  pipeline.fit(X, y)
  return AuthModel(pipeline=pipeline, feature_cols=feature_cols)


def predict_proba(auth_model: AuthModel, df: pd.DataFrame) -> np.ndarray:
  """Predict P(authenticated=1) for rows that include experience_id_shown."""
  X = df[auth_model.feature_cols + ["experience_id_shown"]]
  return auth_model.pipeline.predict_proba(X)[:, 1]
