# Loads CORD v2 receipts from the local parquet files and turns the ground truth into the same
# normalized shape the extractor produces. Only dev (validation) and test are used, the cascade
# has nothing to train except a small confidence model, so the 800 train receipts are skipped.
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from schema import to_amount

DATA_DIR = Path(__file__).parent / "data" / "data"
SPLIT_FILES = {"dev": "validation", "test": "test"}


@dataclass
class Receipt:
    receipt_id: str
    image_bytes: bytes
    image_sha256: str
    truth: dict  # only fields CORD actually labels for this receipt


def name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def as_text(value) -> str:
    # A few CORD fields hold a list of strings when the label wrapped over several lines.
    if isinstance(value, list):
        return " ".join(as_text(v) for v in value)
    return "" if value is None else str(value)


def flatten_menu(menu) -> list[dict]:
    # CORD stores a single item as a dict and nests add-ons under "sub".
    if isinstance(menu, dict):
        menu = [menu]
    items = []
    for m in menu:
        items.append(m)
        items.extend(flatten_menu(m["sub"]) if "sub" in m else [])
    return items


def parse_truth(gt_parse: dict) -> dict:
    truth = {}
    items = [
        m
        for m in flatten_menu(gt_parse.get("menu", []))
        if as_text(m.get("nm")) and to_amount(as_text(m.get("price"))) is not None
    ]
    if items:
        truth["line_items"] = [
            (
                name_key(as_text(m["nm"])),
                to_amount(as_text(m.get("cnt"))) or 1,
                to_amount(as_text(m["price"])),
            )
            for m in items
        ]
    sub = gt_parse.get("sub_total") or {}
    if isinstance(sub, dict):
        for field, key in [("subtotal", "subtotal_price"), ("tax", "tax_price")]:
            if to_amount(as_text(sub.get(key))) is not None:
                truth[field] = to_amount(as_text(sub[key]))
    total = gt_parse.get("total") or {}
    if isinstance(total, dict) and to_amount(as_text(total.get("total_price"))) is not None:
        truth["total"] = to_amount(as_text(total["total_price"]))
    return truth


def load_split(split: str) -> list[Receipt]:
    path = next(DATA_DIR.glob(f"{SPLIT_FILES[split]}-*.parquet"))
    df = pd.read_parquet(path)
    receipts = []
    for i, row in df.iterrows():
        raw = row["image"]["bytes"]
        digest = hashlib.sha256(raw).hexdigest()
        truth = parse_truth(json.loads(row["ground_truth"])["gt_parse"])
        receipts.append(Receipt(f"{split}-{i:03d}", raw, digest, truth))
    return receipts
