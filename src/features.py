"""Session-level feature engineering."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import TypedDict

import pandas as pd

from src.config import (
  ACTION_TYPES,
  COOKIE_VOCAB_TOP_K,
  EVENTS_LAST_MINUTES,
  FREQUENT_COOKIE_TOP_K,
  GLOBAL_TOP_COOKIE_HIT_K,
  SESSIONS_LAST_DAYS,
  TAX_FLOW_PATTERN,
  URL_BUCKETS,
  URL_PREFIXES,
)


class CookieVocabulary(TypedDict):
  frequent_ids: frozenset[int]
  vocab_ids: frozenset[int]
  top_hit_ids: list[int]


def parse_cookie_ids(raw: str) -> list[int]:
  if pd.isna(raw):
    return []
  return json.loads(raw)


def fit_cookie_vocabulary(
  cookie_series: pd.Series,
  frequent_k: int = FREQUENT_COOKIE_TOP_K,
  vocab_k: int = COOKIE_VOCAB_TOP_K,
  top_hit_k: int = GLOBAL_TOP_COOKIE_HIT_K,
) -> CookieVocabulary:
  """Fit frequent / vocab / top-hit cookie sets on training visitors only."""
  counts: Counter = Counter()
  for raw in cookie_series.dropna().unique():
    counts.update(parse_cookie_ids(raw))
  ranked = [cid for cid, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
  return CookieVocabulary(
    frequent_ids=frozenset(ranked[:frequent_k]),
    vocab_ids=frozenset(ranked[:vocab_k]),
    top_hit_ids=ranked[:top_hit_k],
  )


def _url_matches(urls: pd.Series, pattern: str) -> bool:
  if urls.empty:
    return False
  return urls.astype(str).str.contains(pattern, regex=True, na=False).any()


def _session_stats(visitor_prior: pd.DataFrame) -> dict:
  """Aggregate session-level stats across all prior activity for a visitor."""
  if visitor_prior.empty:
    return {
      "num_sessions": 0,
      "session_duration_avg": 0.0,
      "pages_per_session": 0.0,
      "actions_per_session": 0.0,
    }

  grouped = visitor_prior.groupby("visitor_session_id", sort=False)
  durations = grouped["event_dt"].agg(lambda s: (s.max() - s.min()).total_seconds())
  pages = grouped["url"].nunique()
  actions = grouped.size()

  return {
    "num_sessions": int(visitor_prior["visitor_session_id"].nunique()),
    "session_duration_avg": float(durations.mean()),
    "pages_per_session": float(pages.mean()),
    "actions_per_session": float(actions.mean()),
  }


def _recency_features(visitor_prior: pd.DataFrame, homepage_dt: pd.Timestamp) -> dict:
  first_seen = visitor_prior["event_dt"].min() if len(visitor_prior) else homepage_dt
  last_event = visitor_prior["event_dt"].max() if len(visitor_prior) else homepage_dt

  hours_since_first = (homepage_dt - first_seen).total_seconds() / 3600
  hours_since_last = (homepage_dt - last_event).total_seconds() / 3600

  return {
    "time_since_last_visit_hours": float(hours_since_last),
    "days_since_first_seen": float(hours_since_first / 24),
    "homepage_hour": homepage_dt.hour,
    "homepage_dow": homepage_dt.dayofweek,
  }


def _temporal_features(visitor_prior: pd.DataFrame, homepage_dt: pd.Timestamp) -> dict:
  cutoff_5min = homepage_dt - pd.Timedelta(minutes=EVENTS_LAST_MINUTES)
  cutoff_7d = homepage_dt - pd.Timedelta(days=SESSIONS_LAST_DAYS)

  recent_events = visitor_prior[visitor_prior["event_dt"] >= cutoff_5min]
  recent_sessions = visitor_prior[visitor_prior["event_dt"] >= cutoff_7d]

  return {
    f"events_last_{EVENTS_LAST_MINUTES}min": len(recent_events),
    f"sessions_last_{SESSIONS_LAST_DAYS}days": int(
      recent_sessions["visitor_session_id"].nunique() if len(recent_sessions) else 0
    ),
    "distinct_urls": int(visitor_prior["url"].nunique() if len(visitor_prior) else 0),
    "distinct_actions": int(visitor_prior["action"].nunique() if len(visitor_prior) else 0),
  }


def _url_bucket_features(urls: pd.Series, prefix: str = "") -> dict:
  """Binary flags for coarse URL page-type buckets."""
  label = f"{prefix}_" if prefix else ""
  return {
    f"{label}url_bucket_{name}": int(_url_matches(urls, pattern))
    for name, pattern in URL_BUCKETS.items()
  }


def _session_features(prior_session: pd.DataFrame, homepage_dt: pd.Timestamp) -> dict:
  features: dict = {
    "n_session_events": len(prior_session),
    "n_unique_urls_session": prior_session["url"].nunique() if len(prior_session) else 0,
    "n_unique_actions_session": prior_session["action"].nunique() if len(prior_session) else 0,
    "seconds_since_session_start": (
      (homepage_dt - prior_session["event_dt"].min()).total_seconds()
      if len(prior_session)
      else 0.0
    ),
  }

  for action in ACTION_TYPES:
    features[f"session_action_{action}"] = int((prior_session["action"] == action).sum())

  for url_prefix in URL_PREFIXES:
    col = f"session_url_contains_{url_prefix.replace('-', '_')}"
    features[col] = int(_url_matches(prior_session["url"], re.escape(url_prefix)))

  features.update(_url_bucket_features(prior_session["url"], prefix="session"))

  return features


def _intent_features(
  visitor_prior: pd.DataFrame, session_prior: pd.DataFrame
) -> dict:
  """Behavioral intent proxies — high-intent vs. friction signals."""
  repeated_same_page = 0
  if len(session_prior):
    url_counts = session_prior["url"].value_counts()
    repeated_same_page = int(url_counts.max()) if len(url_counts) else 0

  return {
    "viewed_help_content": int(_url_matches(visitor_prior["url"], URL_BUCKETS["help_page"])),
    "visited_pricing": int(_url_matches(visitor_prior["url"], URL_BUCKETS["pricing_page"])),
    "repeated_same_page": repeated_same_page,
    "started_tax_flow": int(_url_matches(visitor_prior["url"], TAX_FLOW_PATTERN)),
  }


def _cookie_features(cookies: set[int], vocab: CookieVocabulary) -> dict:
  """Compact cookie summary features (15 total)."""
  n = len(cookies)
  frequent_ids = vocab["frequent_ids"]
  vocab_ids = vocab["vocab_ids"]
  top_hit_ids = vocab["top_hit_ids"]

  n_frequent = sum(1 for cid in cookies if cid in frequent_ids)
  n_rare = sum(1 for cid in cookies if cid not in frequent_ids)
  n_in_vocab = sum(1 for cid in cookies if cid in vocab_ids)
  n_out_vocab = n - n_in_vocab

  features = {
    "has_cookies": int(n > 0),
    "n_cookies": n,
    "n_frequent_cookies": n_frequent,
    "n_rare_cookies": n_rare,
    "frequent_cookie_share": float(n_frequent / n) if n else 0.0,
    "n_in_vocab_cookies": n_in_vocab,
    "n_out_of_vocab_cookies": n_out_vocab,
    "in_vocab_share": float(n_in_vocab / n) if n else 0.0,
    "any_frequent_cookie": int(n_frequent > 0),
    "any_rare_cookie": int(n_rare > 0),
    "n_global_top_cookie_hits": sum(1 for cid in top_hit_ids if cid in cookies),
    "log1p_n_cookies": float(math.log1p(n)),
  }
  for idx, cid in enumerate(top_hit_ids, start=1):
    features[f"has_global_top_cookie_{idx}"] = int(cid in cookies)

  return features


def build_feature_row(
  row: pd.Series,
  activity_df: pd.DataFrame,
  cookie_vocab: CookieVocabulary,
) -> dict:
  """Build feature dict for one homepage decision point."""
  vid = row["visitor_identifier"]
  session_id = row["visitor_session_id"]
  homepage_dt = row["homepage_dt"]

  visitor_prior = activity_df[
    (activity_df["visitor_identifier"] == vid) & (activity_df["event_dt"] < homepage_dt)
  ]
  session_prior = visitor_prior[visitor_prior["visitor_session_id"] == session_id]

  cookies = set(parse_cookie_ids(row["cookie_ids"]))

  features = {
    "visitor_identifier": vid,
    "visitor_session_id": session_id,
    "homepage_event_ts": int(row["homepage_event_ts"]),
  }
  features.update(_session_stats(visitor_prior))
  features.update(_recency_features(visitor_prior, homepage_dt))
  features.update(_temporal_features(visitor_prior, homepage_dt))
  features.update(_url_bucket_features(visitor_prior["url"]))
  features.update(_session_features(session_prior, homepage_dt))
  features.update(_intent_features(visitor_prior, session_prior))
  features.update(_cookie_features(cookies, cookie_vocab))

  return features


COOKIE_FEATURE_NAMES = (
  "has_cookies",
  "n_cookies",
  "n_frequent_cookies",
  "n_rare_cookies",
  "frequent_cookie_share",
  "n_in_vocab_cookies",
  "n_out_of_vocab_cookies",
  "in_vocab_share",
  "any_frequent_cookie",
  "any_rare_cookie",
  "n_global_top_cookie_hits",
  "log1p_n_cookies",
  "has_global_top_cookie_1",
  "has_global_top_cookie_2",
  "has_global_top_cookie_3",
)


def feature_group_counts(feature_cols: list[str]) -> dict[str, int]:
  """Exact feature counts per category for documentation."""
  counts = {
    "session_agg": 0,
    "recency": 0,
    "temporal": 0,
    "visitor_url_bucket": 0,
    "session_volume": 0,
    "session_action": 0,
    "session_url_prefix": 0,
    "session_url_bucket": 0,
    "intent": 0,
    "cookie": 0,
    "other": 0,
  }
  session_volume = {
    "n_session_events",
    "n_unique_urls_session",
    "n_unique_actions_session",
    "seconds_since_session_start",
  }
  intent = {
    "viewed_help_content",
    "visited_pricing",
    "repeated_same_page",
    "started_tax_flow",
  }

  for col in feature_cols:
    if col in ("num_sessions", "session_duration_avg", "pages_per_session", "actions_per_session"):
      counts["session_agg"] += 1
    elif col in (
      "time_since_last_visit_hours",
      "days_since_first_seen",
      "homepage_hour",
      "homepage_dow",
    ):
      counts["recency"] += 1
    elif col.startswith("events_last_") or col.startswith("sessions_last_") or col in (
      "distinct_urls",
      "distinct_actions",
    ):
      counts["temporal"] += 1
    elif col.startswith("url_bucket_"):
      counts["visitor_url_bucket"] += 1
    elif col in session_volume:
      counts["session_volume"] += 1
    elif col.startswith("session_action_"):
      counts["session_action"] += 1
    elif col.startswith("session_url_contains_"):
      counts["session_url_prefix"] += 1
    elif col.startswith("session_url_bucket_"):
      counts["session_url_bucket"] += 1
    elif col in intent:
      counts["intent"] += 1
    elif col in COOKIE_FEATURE_NAMES:
      counts["cookie"] += 1
    else:
      counts["other"] += 1

  return counts


def summarize_feature_groups(feature_cols: list[str]) -> pd.DataFrame:
  """Summarize engineered features by category for notebook display."""
  counts = feature_group_counts(feature_cols)
  return (
    pd.Series(counts, name="feature_count")
    .reset_index()
    .rename(columns={"index": "group"})
  )
