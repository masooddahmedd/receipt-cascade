# Tests for the validators that double as confidence signals. These encode the receipt rules we
# actually rely on: Indonesian-style amounts, items vs subtotal, and the subtotal fallback.
from schema import ExtractedReceipt, LineItem, date_parses, to_amount, validator_flags


def receipt(items, subtotal=None, tax=None, total=None, date=None):
    return ExtractedReceipt(
        merchant="Shop",
        date=date,
        subtotal=subtotal,
        tax=tax,
        total=total,
        line_items=[LineItem(name=n, quantity=1, price=p) for n, p in items],
    )


def test_amounts_ignore_separators():
    assert to_amount("45.000") == 45000
    assert to_amount("Rp 16,500") == 16500
    assert to_amount(None) is None
    assert to_amount("n/a") is None


def test_consistent_receipt_passes_everything():
    r = receipt([("a", "10.000"), ("b", "5.000")], subtotal="15.000", tax="1.500", total="16.500")
    flags = validator_flags(r)
    assert all(v for k, v in flags.items() if k != "date")


def test_wrong_total_blames_total_and_tax():
    r = receipt([("a", "10.000")], subtotal="10.000", tax="1.000", total="99.000")
    flags = validator_flags(r)
    assert not flags["total"] and not flags["tax"] and flags["line_items"]


def test_missing_subtotal_falls_back_to_item_sum():
    r = receipt([("a", "10.000"), ("b", "5.000")], total="15.000")
    assert validator_flags(r)["total"]


def test_item_sum_mismatch_flags_line_items():
    r = receipt([("a", "10.000")], subtotal="20.000", total="20.000")
    assert not validator_flags(r)["line_items"]


def test_date_formats():
    assert date_parses("24/12/2019")
    assert date_parses("2019-12-24")
    assert not date_parses("yesterday")
    assert not date_parses(None)
