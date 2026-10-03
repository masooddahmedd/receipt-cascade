# Tests for the confidence model and the scoring rules in evaluate.py: the model must rank a
# feature pattern that was right in training above one that was wrong, and line item matching
# must tolerate small spelling differences but not wrong prices.
import numpy as np

from confidence import ConfidenceModel, modal_run
from evaluate import is_correct, items_match
from schema import ExtractedReceipt


def test_model_learns_that_agreement_means_correct():
    rng = np.random.default_rng(1)
    agree = rng.uniform(0, 1, 300)
    rows = [[a, 1.0, 1.0] for a in agree]
    labels = list(agree > 0.5)
    model = ConfidenceModel().fit(rows, labels)
    low, high = model.predict([[0.1, 1.0, 1.0], [0.95, 1.0, 1.0]])
    assert high > 0.9 and low < 0.2


def test_modal_run_returns_majority_and_vote_share():
    def r(total):
        return ExtractedReceipt(
            merchant=None, date=None, line_items=[], subtotal=None, tax=None, total=total
        )

    best, share = modal_run([r("10"), r("12"), r("12")], "total")
    assert best.total == "12" and abs(share - 2 / 3) < 1e-9


def test_item_matching_tolerates_spelling_not_price():
    truth = [("eggtart", 1, 13000), ("pizzatoast", 2, 32000)]
    assert items_match([("eggtart", 1, 13000), ("pizzatoast", 2, 32000)], truth)
    assert items_match([("eggtarts", 1, 13000), ("pizzatoast", 2, 32000)], truth)
    assert not items_match([("eggtart", 1, 13500), ("pizzatoast", 2, 32000)], truth)
    assert not items_match([("eggtart", 1, 13000)], truth)


def test_amount_fields_compare_exactly():
    assert is_correct("total", 45500, {"total": 45500})
    assert not is_correct("total", 45000, {"total": 45500})
