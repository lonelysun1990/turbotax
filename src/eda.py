"""Exploratory data analysis helpers."""

from __future__ import annotations

import json
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def summary_stats(
  first_auth_df: pd.DataFrame,
  activity_df: pd.DataFrame,
  decision_df: pd.DataFrame | None = None,
) -> dict:
  """Return high-level dataset statistics (post-truncation clickstream)."""
  homepage = activity_df.loc[activity_df["experience_id"].notna()]
  auth_visitors = set(first_auth_df["visitor_identifier"])

  if decision_df is not None:
    observed_auth_rate = float(decision_df["authenticated"].mean())
  else:
    observed_auth_rate = float("nan")

  return {
    "n_activity_rows": len(activity_df),
    "n_visitors": activity_df["visitor_identifier"].nunique(),
    "n_auth_visitors": len(auth_visitors),
    "date_min": activity_df["event_dt"].min(),
    "date_max": activity_df["event_dt"].max(),
    "n_homepage_decisions": len(homepage),
    "observed_auth_rate": observed_auth_rate,
  }


def auth_rate_by_experience(decision_df: pd.DataFrame) -> pd.Series:
  """Observational auth rate grouped by shown experience."""
  return decision_df.groupby("experience_id_shown")["authenticated"].mean()


def events_per_session(activity_df: pd.DataFrame) -> pd.Series:
  """Count events per session."""
  return activity_df.groupby("visitor_session_id").size()


def sessions_per_visitor_before_homepage(
  activity_df: pd.DataFrame, decision_df: pd.DataFrame
) -> pd.Series:
  """Count distinct sessions per visitor before homepage decision."""
  merged = decision_df[["visitor_identifier", "homepage_event_ts"]].merge(
    activity_df[["visitor_identifier", "visitor_session_id", "event_ts"]],
    on="visitor_identifier",
  )
  prior = merged[merged["event_ts"] < merged["homepage_event_ts"]]
  return prior.groupby("visitor_identifier")["visitor_session_id"].nunique()


def cookie_id_counts(activity_df: pd.DataFrame) -> Counter:
  """Count frequency of each cookie id across visitors."""
  counts: Counter = Counter()
  for raw in activity_df["cookie_ids"].dropna().unique():
    counts.update(json.loads(raw))
  return counts


def plot_experience_distribution(decision_df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
  ax = ax or plt.gca()
  counts = decision_df["experience_id_shown"].value_counts().sort_index()
  sns.barplot(x=counts.index, y=counts.values, ax=ax)
  ax.set_title("Homepage Experience Distribution")
  ax.set_xlabel("experience_id")
  ax.set_ylabel("Count")
  return ax


def plot_auth_rate_by_experience(decision_df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
  ax = ax or plt.gca()
  rates = auth_rate_by_experience(decision_df).reset_index()
  sns.barplot(data=rates, x="experience_id_shown", y="authenticated", ax=ax)
  ax.set_title("Observational Auth Rate by Experience")
  ax.set_xlabel("experience_id")
  ax.set_ylabel("Auth rate")
  return ax


def plot_homepage_time_patterns(decision_df: pd.DataFrame) -> plt.Figure:
  fig, axes = plt.subplots(1, 2, figsize=(12, 4))
  sns.countplot(data=decision_df, x="homepage_hour", ax=axes[0])
  axes[0].set_title("Homepage Impressions by Hour")
  sns.countplot(data=decision_df, x="homepage_dow", ax=axes[1])
  axes[1].set_title("Homepage Impressions by Day of Week")
  fig.tight_layout()
  return fig


def plot_cookie_frequency(cookie_counts: Counter, top_n: int = 30) -> plt.Figure:
  top = cookie_counts.most_common(top_n)
  fig, ax = plt.subplots(figsize=(10, 4))
  ids, freqs = zip(*top)
  sns.barplot(x=list(ids), y=list(freqs), ax=ax)
  ax.set_title(f"Top {top_n} Cookie IDs by Visitor Frequency")
  ax.set_xlabel("Cookie ID")
  ax.set_ylabel("Visitors")
  plt.xticks(rotation=90)
  fig.tight_layout()
  return fig


def value_counts_table(series: pd.Series, name: str, top_n: int = 10) -> pd.DataFrame:
  """Return top-N value counts as a dataframe."""
  vc = series.value_counts().head(top_n)
  return pd.DataFrame({name: vc.index, "count": vc.values})


def visitor_conversion_frame(decision_df: pd.DataFrame) -> pd.DataFrame:
  """Visitor-level conversion flag and exposure counts (pre-auth clickstream)."""
  converted = (
    decision_df.groupby("visitor_identifier")["authenticated"]
    .max()
    .astype(bool)
    .rename("converted")
  )
  n_experiences = (
    decision_df.groupby("visitor_identifier")["experience_id_shown"]
    .nunique()
    .rename("n_experiences_seen")
  )
  return pd.concat([converted, n_experiences], axis=1).reset_index()


def visitor_session_counts(activity_df: pd.DataFrame) -> pd.Series:
  """Distinct sessions per visitor in the (truncated) clickstream."""
  return (
    activity_df.groupby("visitor_identifier")["visitor_session_id"]
    .nunique()
    .rename("n_sessions")
  )


def build_visitor_eda_frame(
  decision_df: pd.DataFrame, activity_df: pd.DataFrame
) -> pd.DataFrame:
  """Visitor-level table for converted vs not-converted comparisons."""
  visitor_df = visitor_conversion_frame(decision_df)
  visitor_df = visitor_df.merge(
    visitor_session_counts(activity_df).reset_index(),
    on="visitor_identifier",
  )
  visitor_df["converted_label"] = visitor_df["converted"].map(
    {True: "Converted", False: "Not converted"}
  )
  return visitor_df


def plot_visitor_metric_by_conversion(
  visitor_df: pd.DataFrame,
  metric: str,
  title: str,
  ax: plt.Axes | None = None,
) -> plt.Axes:
  """Density/histogram of a visitor metric split by eventual conversion."""
  ax = ax or plt.gca()
  sns.histplot(
    data=visitor_df,
    x=metric,
    hue="converted_label",
    hue_order=["Not converted", "Converted"],
    stat="density",
    common_norm=False,
    discrete=(metric == "n_experiences_seen"),
    multiple="dodge",
    shrink=0.85,
    ax=ax,
  )
  ax.set_title(title)
  ax.set_xlabel(metric)
  ax.set_ylabel("Density")
  return ax


def plot_visitor_conversion_distributions(
  decision_df: pd.DataFrame, activity_df: pd.DataFrame
) -> plt.Figure:
  """Compare experiences seen and session counts for converters vs non-converters."""
  visitor_df = build_visitor_eda_frame(decision_df, activity_df)
  fig, axes = plt.subplots(1, 2, figsize=(12, 4))
  plot_visitor_metric_by_conversion(
    visitor_df,
    "n_experiences_seen",
    "Distinct Homepage Experiences per Visitor",
    ax=axes[0],
  )
  plot_visitor_metric_by_conversion(
    visitor_df,
    "n_sessions",
    "Total Sessions per Visitor (pre-auth)",
    ax=axes[1],
  )
  fig.tight_layout()
  return fig


def visitor_metric_summary_by_conversion(visitor_df: pd.DataFrame) -> pd.DataFrame:
  """Mean / median visitor metrics by conversion status."""
  return (
    visitor_df.groupby("converted_label")[["n_experiences_seen", "n_sessions"]]
    .agg(["mean", "median", "max"])
    .round(2)
  )
