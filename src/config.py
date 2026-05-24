"""Project configuration and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

FIRST_AUTH_PATH = DATA_DIR / "first_auth_data.csv"
CLICKSTREAM_PATH = DATA_DIR / "visitors_activity.csv.gz"

EXPERIENCE_IDS = list(range(1, 8))
FREQUENT_COOKIE_TOP_K = 10
COOKIE_VOCAB_TOP_K = 50
GLOBAL_TOP_COOKIE_HIT_K = 3
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Rolling temporal windows
EVENTS_LAST_MINUTES = 5
SESSIONS_LAST_DAYS = 7

# URL buckets for simple regex-based extraction (no NLP needed)
URL_BUCKETS = {
  "product_page": r"personal-taxes|deluxe|premier|business-taxes|self-employed|compare/products|free-edition",
  "pricing_page": r"pricing|free-edition|compare/products",
  "help_page": r"help|support|expert",
  "checkout_page": r"sign-in|create-account",
}

TAX_FLOW_PATTERN = r"personal-taxes|sign-in|create-account|deluxe|premier|self-employed"

URL_PREFIXES = (
  "pricing",
  "help",
  "blog",
  "tools",
  "sign-in",
  "create-account",
  "business",
  "mobile",
)

ACTION_TYPES = (
  "click_cta",
  "play_video",
  "scroll",
  "expand_faq",
  "submit_form",
  "click_nav",
)

