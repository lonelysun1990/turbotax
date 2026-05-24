"""Assemble model-ready decision dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE
from src.features import CookieVocabulary, build_feature_row, fit_cookie_vocabulary


def build_decision_dataset(
  decision_df: pd.DataFrame,
  activity_df: pd.DataFrame,
  cookie_vocab: CookieVocabulary | None = None,
  train_visitor_ids: set[str] | None = None,
) -> pd.DataFrame:
  """
  Build one row per homepage session with features, treatment, and label.

  If train_visitor_ids is provided, cookie vocabulary is fit on that subset only.
  """
  if cookie_vocab is None:
    if train_visitor_ids is None:
      cookie_vocab = fit_cookie_vocabulary(decision_df["cookie_ids"])
    else:
      train_cookies = decision_df.loc[
        decision_df["visitor_identifier"].isin(train_visitor_ids), "cookie_ids"
      ]
      cookie_vocab = fit_cookie_vocabulary(train_cookies)

  rows = [
    build_feature_row(row, activity_df, cookie_vocab)
    for _, row in decision_df.iterrows()
  ]
  feature_df = pd.DataFrame(rows)

  dataset = feature_df.merge(
    decision_df[
      [
        "visitor_identifier",
        "visitor_session_id",
        "homepage_event_ts",
        "experience_id_shown",
        "authenticated",
      ]
    ],
    on=["visitor_identifier", "visitor_session_id", "homepage_event_ts"],
  )

  return dataset


def split_visitors(
  decision_df: pd.DataFrame, test_size: float = TEST_SIZE
) -> tuple[set[str], set[str]]:
  """Visitor-level stratified train/validation split."""
  visitor_labels = decision_df.groupby("visitor_identifier")["authenticated"].max()
  train_ids, val_ids = train_test_split(
    visitor_labels.index,
    test_size=test_size,
    random_state=RANDOM_STATE,
    stratify=visitor_labels,
  )
  return set(train_ids), set(val_ids)


def get_feature_columns(dataset: pd.DataFrame) -> list[str]:
  """Return model feature columns (excluding ids, treatment, label)."""
  exclude = {
    "visitor_identifier",
    "visitor_session_id",
    "homepage_event_ts",
    "experience_id_shown",
    "authenticated",
  }
  return [c for c in dataset.columns if c not in exclude]
