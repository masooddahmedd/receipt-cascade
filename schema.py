# Receipt schema and the validators that double as confidence signals. Amounts are kept as the
# strings printed on the receipt and turned into integers here, because CORD uses Indonesian
# formatting ("45.000" is forty-five thousand) and models love to "fix" that into 45.0.
import re
from datetime import datetime

from pydantic import BaseModel

from config import SUM_TOLERANCE


class LineItem(BaseModel):
    name: str
    quantity: int
    price: str


class ExtractedReceipt(BaseModel):
    merchant: str | None
    date: str | None
    line_items: list[LineItem]
    subtotal: str | None
    tax: str | None
    total: str | None


def to_amount(value: str | None) -> int | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", value)
    return int(digits) if digits else None


DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%y", "%d.%m.%Y", "%d %b %Y"]


def date_parses(value: str | None) -> bool:
    if not value:
        return False
    for fmt in DATE_FORMATS:
        try:
            datetime.strptime(value.strip(), fmt)
            return True
        except ValueError:
            continue
    return False


def within_tolerance(a: int, b: int) -> bool:
    return abs(a - b) <= SUM_TOLERANCE * max(a, b, 1)


def line_item_sum(r: ExtractedReceipt) -> int | None:
    prices = [to_amount(i.price) for i in r.line_items]
    if not prices or any(p is None for p in prices):
        return None
    return sum(prices)


def validator_flags(r: ExtractedReceipt) -> dict[str, bool]:
    # One pass/fail per field. The cross-field checks (items vs subtotal, subtotal + tax vs total)
    # blame every field involved, since from the numbers alone we cannot tell which one is wrong.
    items_sum = line_item_sum(r)
    sub, tax, tot = to_amount(r.subtotal), to_amount(r.tax), to_amount(r.total)
    items_ok = items_sum is not None and sub is not None and within_tolerance(items_sum, sub)
    # Many receipts print no subtotal line, so the items stand in for it when it is missing.
    base = sub if sub is not None else items_sum
    sum_ok = base is not None and tot is not None and within_tolerance(base + (tax or 0), tot)
    return {
        "merchant": bool(r.merchant and r.merchant.strip()),
        "date": date_parses(r.date),
        "line_items": items_ok,
        "subtotal": items_ok and sum_ok,
        "tax": sum_ok,
        "total": sum_ok,
    }
