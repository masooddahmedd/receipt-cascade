# Single place for models, prices, thresholds and paths. Prices are OpenAI list prices in USD per
# million tokens at the time of writing [check them again before quoting a cost number].
from pathlib import Path

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
OUT_DIR = ROOT / "out"

TIER1_MODEL = "gpt-4o-mini"
TIER2_MODEL = "gpt-4o"
TIER1_RUNS = 3
TIER1_TEMPERATURE = 0.7
TIER2_TEMPERATURE = 0.0
PROMPT_VERSION = "v1"

# Longest image side sent to the models. Keeps the tile count (and cost) bounded.
MAX_IMAGE_SIDE = 1024

PRICE_PER_M = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
}

# Line items must add up to the subtotal within this fraction. CORD receipts often carry discounts
# and service charges, so a strict equality check would reject many correct extractions.
SUM_TOLERANCE = 0.02

# Fields that have ground truth in CORD and are scored. Merchant and date are extracted but CORD
# does not label them, so they are not part of the accuracy numbers.
SCORED_FIELDS = ["line_items", "subtotal", "tax", "total"]
ALL_FIELDS = ["merchant", "date", "line_items", "subtotal", "tax", "total"]
