# Per-field confidence. Three signals go into a small logistic regression fit on the dev split:
# how often the three Tier 1 runs agreed, whether the schema validators passed, and whether the
# value shows up in the OCR text. We fit weights instead of hand-tuning them because the
# validators are much more informative for some fields (total) than others (line item names).
import re
from collections import Counter

import numpy as np

from config import SCORED_FIELDS
from data import name_key
from ocr import number_runs
from schema import ExtractedReceipt, date_parses, to_amount, validator_flags


def canonical(r: ExtractedReceipt, field: str):
    # The comparable form of a field: amounts as ints, names stripped to lowercase alphanumerics.
    if field == "line_items":
        return tuple(
            sorted((name_key(i.name), i.quantity, to_amount(i.price)) for i in r.line_items)
        )
    if field in ("subtotal", "tax", "total"):
        return to_amount(getattr(r, field))
    if field == "merchant":
        return name_key(r.merchant) if r.merchant else None
    return r.date.strip() if r.date else None


def token_overlap(name: str, ocr_lower: str) -> float:
    tokens = [t for t in re.findall(r"[a-z0-9]+", name.lower()) if len(t) >= 3]
    return sum(t in ocr_lower for t in tokens) / len(tokens) if tokens else 0.0


def ocr_match(r: ExtractedReceipt, field: str, ocr: str) -> float:
    runs, low = number_runs(ocr), ocr.lower()
    if field in ("subtotal", "tax", "total"):
        value = to_amount(getattr(r, field))
        return 0.0 if value is None else float(str(value) in runs)
    if field == "line_items":
        if not r.line_items:
            return 0.0
        scores = [
            (float(str(to_amount(i.price)) in runs) + token_overlap(i.name, low)) / 2
            for i in r.line_items
        ]
        return float(np.mean(scores))
    if field == "merchant":
        return token_overlap(r.merchant, low) if r.merchant else 0.0
    if not date_parses(r.date):
        return 0.0
    return float(all(g in re.sub(r"\D", " ", ocr).split() for g in re.findall(r"\d+", r.date)))


def modal_run(runs: list[ExtractedReceipt], field: str):
    # Returns the most common value across runs, the run that produced it, and the vote share.
    canon = [canonical(r, field) for r in runs]
    value, votes = Counter(canon).most_common(1)[0]
    return runs[canon.index(value)], votes / len(runs)


def feature_row(receipt: ExtractedReceipt, field: str, agreement: float, ocr: str) -> list[float]:
    one_hot = [float(field == f) for f in SCORED_FIELDS]
    return [
        agreement,
        float(validator_flags(receipt)[field]),
        ocr_match(receipt, field, ocr),
        *one_hot,
    ]


class ConfidenceModel:
    # Plain L2-regularised logistic regression fit with Newton steps. scikit-learn is not used
    # because its compiled extensions are blocked on the machine this was built on, and the model
    # is tiny (a dozen weights, a few hundred rows).
    def __init__(self, l2: float = 1.0):
        self.l2 = l2
        self.w = None

    def fit(self, rows: list[list[float]], correct: list[bool]) -> "ConfidenceModel":
        X = np.hstack([np.array(rows), np.ones((len(rows), 1))])
        y = np.array(correct, dtype=float)
        w = np.zeros(X.shape[1])
        reg = self.l2 * np.eye(X.shape[1])
        reg[-1, -1] = 0.0
        for _ in range(50):
            p = 1 / (1 + np.exp(-X @ w))
            grad = X.T @ (p - y) + reg @ w
            hess = X.T @ (X * (p * (1 - p))[:, None]) + reg + 1e-8 * np.eye(X.shape[1])
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def predict(self, rows: list[list[float]]) -> np.ndarray:
        X = np.hstack([np.array(rows), np.ones((len(rows), 1))])
        return 1 / (1 + np.exp(-X @ self.w))
