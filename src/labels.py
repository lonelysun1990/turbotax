"""Label construction for homepage decision points."""

from __future__ import annotations

import pandas as pd


def build_decision_points(
  first_auth_df: pd.DataFrame, activity_df: pd.DataFrame
) -> pd.DataFrame:
  """
  Identify homepage experiment rows and attach auth labels.

  Label = 1 iff first auth occurs in the same session as the homepage impression,
  at or after the homepage event timestamp.
  """
  homepage = (
    activity_df.loc[activity_df["experience_id"].notna()]
    .sort_values(["visitor_identifier", "event_ts"])
    .rename(columns={"event_dt": "homepage_dt", "experience_id": "experience_id_shown"})
  )

  auth_by_visitor = first_auth_df.set_index("visitor_identifier")

  def session_authenticated(row: pd.Series) -> int:
    vid = row["visitor_identifier"]
    if vid not in auth_by_visitor.index:
      return 0
    auth = auth_by_visitor.loc[vid]
    if isinstance(auth, pd.DataFrame):
      auth = auth.iloc[0]
    if pd.isna(auth.get("visitor_session_id")):
      return 0
    return int(
      auth["first_auth_dt"] >= row["homepage_dt"]
      and auth["visitor_session_id"] == row["visitor_session_id"]
    )

  homepage["authenticated"] = homepage.apply(session_authenticated, axis=1)

  decision_df = homepage[
    [
      "visitor_identifier",
      "visitor_session_id",
      "cookie_ids",
      "homepage_dt",
      "event_ts",
      "experience_id_shown",
      "authenticated",
    ]
  ].rename(columns={"event_ts": "homepage_event_ts"})

  decision_df["homepage_hour"] = decision_df["homepage_dt"].dt.hour
  decision_df["homepage_dow"] = decision_df["homepage_dt"].dt.dayofweek

  return decision_df.reset_index(drop=True)
