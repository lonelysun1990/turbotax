"""Load and parse raw tables."""

from __future__ import annotations

import pandas as pd

from src.config import CLICKSTREAM_PATH, FIRST_AUTH_PATH
from src.preprocessing import enrich_first_auth_sessions, truncate_activity_after_first_auth


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
  """
  Load first-auth and clickstream tables with parsed timestamps.

  Clickstream is truncated to events strictly before each visitor's first auth.
  First-auth rows are enriched with visitor_session_id when missing.
  """
  first_auth_df = pd.read_csv(FIRST_AUTH_PATH)
  activity_df = pd.read_csv(CLICKSTREAM_PATH, compression="gzip")

  activity_df["event_dt"] = pd.to_datetime(activity_df["event_ts"], unit="s", utc=True)
  first_auth_df["first_auth_dt"] = pd.to_datetime(
    first_auth_df["first_auth_timestamp"], utc=True
  )

  first_auth_df = enrich_first_auth_sessions(first_auth_df, activity_df)
  activity_df = truncate_activity_after_first_auth(activity_df, first_auth_df)

  return first_auth_df, activity_df
