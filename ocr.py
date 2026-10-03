# Local OCR (RapidOCR, runs on CPU, no system install) used only as an independent check on what
# the LLM read. The point is a second opinion that did not come from the same model, so the
# extracted strings are matched against OCR text but never fed back into the prompt.
import io
import json
import re

import numpy as np
from PIL import Image

from config import CACHE_DIR

_engine = None


def ocr_text(image_bytes: bytes, image_sha256: str) -> str:
    global _engine
    path = CACHE_DIR / "ocr" / f"{image_sha256}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
    result, _ = _engine(np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB")))
    text = "\n".join(line[1] for line in (result or []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def ocr_layout_text(image_bytes: bytes, image_sha256: str) -> str:
    # Same OCR, but boxes on the same visual row are joined on one line so a quantity, a name and
    # a price stay together. Plain OCR text lists boxes in reading order and often splits a row,
    # which is our guess for why Tier 1 does badly on line items.
    global _engine
    path = CACHE_DIR / "ocr_boxes" / f"{image_sha256}.json"
    if path.exists():
        boxes = json.loads(path.read_text(encoding="utf-8"))
    else:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR()
        result, _ = _engine(np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB")))
        boxes = [[b, t] for b, t, _ in (result or [])]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(boxes), encoding="utf-8")
    return rows_to_text(boxes)


def rows_to_text(boxes: list) -> str:
    items = []
    for quad, text in boxes:
        ys = [p[1] for p in quad]
        xs = [p[0] for p in quad]
        items.append((sum(ys) / 4, max(ys) - min(ys), min(xs), text))
    if not items:
        return ""
    items.sort()
    typical_height = float(np.median([h for _, h, _, _ in items]))
    rows, current, row_y = [], [], None
    for y, _, x, text in items:
        if row_y is not None and abs(y - row_y) > 0.6 * typical_height:
            rows.append(current)
            current = []
        current.append((x, text))
        row_y = y if len(current) == 1 else row_y
    rows.append(current)
    return "\n".join("    ".join(t for _, t in sorted(r)) for r in rows)


def number_runs(text: str) -> set[str]:
    # Digits only, so "16,500" and "16.500" both read as 16500.
    return {re.sub(r"\D", "", t) for t in re.findall(r"\d[\d.,]*", text)}
