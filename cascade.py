# Runs the extraction tiers for one receipt and collects everything the evaluation needs: the
# Tier 1 majority value per field, the Tier 2 value, their feature rows and the API usage. All
# model calls go through the cache, so sweeping thresholds later costs nothing.
from concurrent.futures import ThreadPoolExecutor

from config import (ALL_FIELDS, TIER1_MODEL, TIER1_RUNS, TIER1_TEMPERATURE, TIER2_MODEL,
                    TIER2_TEMPERATURE)
from confidence import canonical, feature_row, modal_run
from data import Receipt
from extract import extract
from ocr import ocr_text


def raw_field(r, field: str):
    value = getattr(r, field)
    return [i.model_dump() for i in value] if field == "line_items" else value


def run_receipt(rec: Receipt) -> dict:
    ocr = ocr_text(rec.image_bytes, rec.image_sha256)
    tier1 = [extract(rec.image_bytes, rec.image_sha256, TIER1_MODEL, TIER1_TEMPERATURE, i, ocr)
             for i in range(TIER1_RUNS)]
    tier2, usage2 = extract(rec.image_bytes, rec.image_sha256, TIER2_MODEL, TIER2_TEMPERATURE, 0)
    runs = [r for r, _ in tier1]

    fields = {}
    for f in ALL_FIELDS:
        best, agreement = modal_run(runs, f)
        agrees_with_t1 = float(canonical(tier2, f) == canonical(best, f))
        fields[f] = {
            "tier1_value": canonical(best, f),
            "tier1_raw": raw_field(best, f),
            "tier2_raw": raw_field(tier2, f),
            "tier1_features": feature_row(best, f, agreement, ocr),
            "tier2_value": canonical(tier2, f),
            "tier2_features": feature_row(tier2, f, agrees_with_t1, ocr),
        }
    return {"receipt": rec, "fields": fields, "tier1_usage": [u for _, u in tier1],
            "tier2_usage": usage2}


def run_split(receipts: list[Receipt]) -> list[dict]:
    # OCR first (CPU-bound, one engine), then the API calls in parallel.
    for r in receipts:
        ocr_text(r.image_bytes, r.image_sha256)
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(run_receipt, receipts))
