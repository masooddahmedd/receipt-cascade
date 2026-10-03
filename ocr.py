# Local OCR (RapidOCR, runs on CPU, no system install) used only as an independent check on what
# the LLM read. The point is a second opinion that did not come from the same model, so the
# extracted strings are matched against OCR text but never fed back into the prompt.
import io
import re

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
    import numpy as np

    result, _ = _engine(np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB")))
    text = "\n".join(line[1] for line in (result or []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def number_runs(text: str) -> set[str]:
    # Digits only, so "16,500" and "16.500" both read as 16500.
    return {re.sub(r"\D", "", t) for t in re.findall(r"\d[\d.,]*", text)}
