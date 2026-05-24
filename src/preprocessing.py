"""Clickstream preprocessing before EDA and modeling."""

from __future__ import annotations

import pandas as pd


def enrich_first_auth_sessions(
  first_auth_df: pd.DataFrame, activity_df: pd.DataFrame
) -> pd.DataFrame:
  """
  Attach visitor_session_id to first-auth rows when missing.

  Infers the homepage session: latest homepage impression at or before first_auth_dt.
  """
  if "visitor_session_id" in first_auth_df.columns:
    return first_auth_df

  homepage = activity_df.loc[
    activity_df["experience_id"].notna(),
    ["visitor_identifier", "visitor_session_id", "event_dt"],
  ]

  session_ids: list[str | None] = []
  for _, auth_row in first_auth_df.iterrows():
    vid = auth_row["visitor_identifier"]
    auth_dt = auth_row["first_auth_dt"]
    candidates = homepage[
      (homepage["visitor_identifier"] == vid) & (homepage["event_dt"] <= auth_dt)
    ].sort_values("event_dt")
    session_ids.append(
      None if candidates.empty else candidates.iloc[-1]["visitor_session_id"]
    )

  enriched = first_auth_df.copy()
  enriched["visitor_session_id"] = session_ids
  return enriched


def truncate_activity_after_first_auth(
  activity_df: pd.DataFrame, first_auth_df: pd.DataFrame
) -> pd.DataFrame:
  """Drop clickstream events at or after each visitor's first auth timestamp."""
  auth_times = first_auth_df.set_index("visitor_identifier")["first_auth_dt"]

  merged = activity_df.merge(
    auth_times.rename("first_auth_dt"),
    left_on="visitor_identifier",
    right_index=True,
    how="left",
  )
  keep = merged["first_auth_dt"].isna() | (
    merged["event_dt"] < merged["first_auth_dt"]
  )
  return merged.loc[keep, activity_df.columns].reset_index(drop=True)
