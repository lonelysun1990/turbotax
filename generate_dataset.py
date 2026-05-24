"""
Generate synthetic TurboTax craft-demo datasets:
  - first_auth_data.csv
  - visitors_activity.csv.gz

Run: python generate_dataset.py
"""

from __future__ import annotations

import gzip
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_VISITORS = 8_000
N_COOKIE_IDS = 850
START_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2024, 3, 31, 23, 59, 59, tzinfo=timezone.utc)
OUTPUT_DIR = Path(__file__).parent / "data"

SEGMENTS = [
  "new_visitor",
  "returning_customer",
  "price_sensitive",
  "expert_seeker",
  "mobile_first",
  "business_owner",
  "student",
]
SEGMENT_TO_BEST_EXPERIENCE = {
  "new_visitor": 1,
  "returning_customer": 2,
  "price_sensitive": 3,
  "expert_seeker": 4,
  "mobile_first": 5,
  "business_owner": 6,
  "student": 7,
}

URLS = [
  "/",
  "/personal-taxes",
  "/business-taxes",
  "/pricing",
  "/free-edition",
  "/deluxe",
  "/premier",
  "/self-employed",
  "/help",
  "/expert-review",
  "/mobile-app",
  "/sign-in",
  "/create-account",
  "/blog/tax-tips",
  "/blog/refund-tracker",
  "/tools/w4-calculator",
  "/tools/tax-bracket-calculator",
  "/compare/products",
  "/support/contact",
  "/about",
]

ACTIONS = [
  "",
  "click_cta",
  "click_nav",
  "scroll",
  "expand_faq",
  "play_video",
  "open_modal",
  "submit_form",
  "hover",
]


def _rng() -> np.random.Generator:
  return np.random.default_rng(SEED)


def _assign_segment(rng: np.random.Generator) -> str:
  weights = np.array([0.22, 0.18, 0.16, 0.12, 0.12, 0.10, 0.10])
  idx = rng.choice(len(SEGMENTS), p=weights)
  return SEGMENTS[idx]


def _cookie_pool_for_segment(segment: str, rng: np.random.Generator) -> set[int]:
  base = {
    "new_visitor": (1, 120),
    "returning_customer": (80, 220),
    "price_sensitive": (180, 320),
    "expert_seeker": (260, 420),
    "mobile_first": (340, 520),
    "business_owner": (460, 620),
    "student": (560, 720),
  }[segment]
  lo, hi = base
  n_cookies = int(rng.integers(8, 35))
  core = set(rng.integers(lo, hi + 1, size=n_cookies).tolist())
  noise = set(rng.integers(721, N_COOKIE_IDS + 1, size=int(rng.integers(2, 12))).tolist())
  return core | noise


def _auth_probability(segment: str, experience_id: int, rng: np.random.Generator) -> float:
  best = SEGMENT_TO_BEST_EXPERIENCE[segment]
  if experience_id == best:
    base = 0.42
  elif abs(experience_id - best) == 1:
    base = 0.22
  elif abs(experience_id - best) == 2:
    base = 0.12
  else:
    base = 0.06
  return float(np.clip(base + rng.normal(0, 0.03), 0.02, 0.75))


def _random_ts(rng: np.random.Generator, start: datetime, end: datetime) -> datetime:
  span = int((end - start).total_seconds())
  offset = int(rng.integers(0, max(span, 1)))
  return start + timedelta(seconds=offset)


def _unix_ts(dt: datetime) -> int:
  return int(dt.timestamp())


def _url_to_screen(url: str) -> str:
  if url == "/":
    return "homepage"
  if any(x in url for x in ("pricing", "free-edition", "deluxe", "premier")):
    return "pricing"
  if "help" in url or "support" in url or "expert" in url:
    return "help"
  if "blog" in url:
    return "blog"
  if "tools" in url:
    return "tools"
  if "sign-in" in url or "create-account" in url:
    return "sign_in"
  if "business" in url or "self-employed" in url:
    return "product_detail"
  return "other"


def _segment_url_bias(segment: str, rng: np.random.Generator) -> list[str]:
  """URLs a visitor is more likely to browse before homepage."""
  bias = {
    "new_visitor": ["/", "/personal-taxes", "/free-edition", "/blog/tax-tips", "/create-account"],
    "returning_customer": ["/", "/sign-in", "/deluxe", "/premier", "/pricing"],
    "price_sensitive": ["/pricing", "/free-edition", "/compare/products", "/deluxe", "/"],
    "expert_seeker": ["/expert-review", "/help", "/support/contact", "/", "/premier"],
    "mobile_first": ["/mobile-app", "/", "/personal-taxes", "/blog/refund-tracker"],
    "business_owner": ["/business-taxes", "/self-employed", "/pricing", "/", "/premier"],
    "student": ["/free-edition", "/blog/tax-tips", "/tools/w4-calculator", "/", "/personal-taxes"],
  }[segment]
  extra = rng.choice(URLS, size=int(rng.integers(3, 8)), replace=True).tolist()
  return list(dict.fromkeys(bias + extra))


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
  rng = _rng()
  random.seed(SEED)

  activity_rows: list[dict] = []
  first_auth_rows: list[dict] = []

  for i in range(1, N_VISITORS + 1):
    vid = f"vis_{i:06d}"
    segment = _assign_segment(rng)
    cookies = sorted(_cookie_pool_for_segment(segment, rng))
    cookie_str = json.dumps(cookies)

    n_sessions = int(rng.integers(1, 6))
    session_ids = [f"{vid}_sess_{s:02d}" for s in range(1, n_sessions + 1)]
    preferred_urls = _segment_url_bias(segment, rng)

    visit_start = _random_ts(rng, START_DATE, END_DATE - timedelta(days=7))
    visit_end = min(visit_start + timedelta(days=int(rng.integers(0, 5)), hours=int(rng.integers(1, 48))), END_DATE)

    n_pre_homepage_events = int(rng.integers(3, 28))
    event_times: list[datetime] = sorted(
      [_random_ts(rng, visit_start, visit_end - timedelta(minutes=30)) for _ in range(n_pre_homepage_events)]
    )

    current_session_idx = 0
    last_event_ts: datetime | None = None

    for j, ts in enumerate(event_times):
      if last_event_ts and (ts - last_event_ts).total_seconds() > 30 * 60:
        current_session_idx = min(current_session_idx + 1, n_sessions - 1)
      last_event_ts = ts

      url = preferred_urls[j % len(preferred_urls)] if j < len(preferred_urls) else rng.choice(URLS)
      if j > 0 and rng.random() < 0.35:
        url = rng.choice(URLS)

      action = rng.choice(ACTIONS, p=[0.35, 0.15, 0.12, 0.18, 0.05, 0.04, 0.04, 0.04, 0.03])
      screen = _url_to_screen(url)

      activity_rows.append(
        {
          "visitor_identifier": vid,
          "cookie_ids": cookie_str,
          "visitor_session_id": session_ids[current_session_idx],
          "action": action,
          "url": url,
          "event_ts": _unix_ts(ts),
          "screen": screen,
          "experience_id": np.nan,
        }
      )

    # Homepage impression — experience is assigned (historical policy: mostly random with slight skew)
    homepage_ts = visit_end if last_event_ts is None else max(last_event_ts + timedelta(minutes=int(rng.integers(5, 45))), visit_start)
    if homepage_ts > END_DATE:
      homepage_ts = END_DATE - timedelta(minutes=int(rng.integers(10, 120)))

    if last_event_ts and (homepage_ts - last_event_ts).total_seconds() > 30 * 60:
      current_session_idx = min(current_session_idx + 1, n_sessions - 1)

    experience_id = int(rng.integers(1, 8))
    if rng.random() < 0.15:
      experience_id = SEGMENT_TO_BEST_EXPERIENCE[segment]

    activity_rows.append(
      {
        "visitor_identifier": vid,
        "cookie_ids": cookie_str,
        "visitor_session_id": session_ids[current_session_idx],
        "action": "view_homepage",
        "url": "/",
        "event_ts": _unix_ts(homepage_ts),
        "screen": "homepage",
        "experience_id": experience_id,
      }
    )

    auth_prob = _auth_probability(segment, experience_id, rng)
    authenticated = rng.random() < auth_prob

    if authenticated:
      auth_delay_minutes = int(rng.integers(2, 180))
      auth_ts = homepage_ts + timedelta(minutes=auth_delay_minutes)
      if auth_ts > END_DATE:
        auth_ts = END_DATE - timedelta(minutes=int(rng.integers(1, 30)))

      first_auth_rows.append(
        {
          "visitor_identifier": vid,
          "first_auth_timestamp": auth_ts.strftime("%Y-%m-%d %H:%M:%S"),
          "visitor_session_id": session_ids[current_session_idx],
        }
      )

      post_auth_events = int(rng.integers(1, 8))
      post_times = sorted(
        [_random_ts(rng, auth_ts, min(auth_ts + timedelta(hours=6), END_DATE)) for _ in range(post_auth_events)]
      )
      post_session = session_ids[min(current_session_idx + 1, n_sessions - 1)]

      for pts in post_times:
        activity_rows.append(
          {
            "visitor_identifier": vid,
            "cookie_ids": cookie_str,
            "visitor_session_id": post_session,
            "action": rng.choice(["click_nav", "scroll", "click_cta", ""]),
            "url": rng.choice(["/sign-in", "/personal-taxes", "/pricing", "/help", "/"]),
            "event_ts": _unix_ts(pts),
            "screen": rng.choice(["sign_in", "pricing", "help", "homepage", "other"]),
            "experience_id": np.nan,
          }
        )

  activity_df = pd.DataFrame(activity_rows)
  activity_df["experience_id"] = activity_df["experience_id"].astype("Int64")

  first_auth_df = pd.DataFrame(first_auth_rows).drop_duplicates("visitor_identifier")
  first_auth_df = first_auth_df.sort_values("first_auth_timestamp").reset_index(drop=True)

  return first_auth_df, activity_df


def main() -> None:
  first_auth_df, activity_df = generate()

  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  auth_path = OUTPUT_DIR / "first_auth_data.csv"
  activity_path = OUTPUT_DIR / "visitors_activity.csv.gz"

  first_auth_df.to_csv(auth_path, index=False)
  with gzip.open(activity_path, "wt", encoding="utf-8") as f:
    activity_df.to_csv(f, index=False)

  n_auth = len(first_auth_df)
  n_visitors = activity_df["visitor_identifier"].nunique()
  n_homepage = (activity_df["screen"] == "homepage").sum()

  print(f"Wrote {auth_path}")
  print(f"Wrote {activity_path}")
  print(f"Visitors: {n_visitors:,}")
  print(f"First-time auths: {n_auth:,} ({100 * n_auth / n_visitors:.1f}%)")
  print(f"Activity rows: {len(activity_df):,}")
  print(f"Homepage impressions: {n_homepage:,}")
  print(f"Experience distribution:\n{activity_df.loc[activity_df['screen'] == 'homepage', 'experience_id'].value_counts().sort_index()}")


if __name__ == "__main__":
  main()
