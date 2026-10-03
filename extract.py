# Calls the vision models and caches every response on disk keyed by (prompt version, model, run,
# image hash). The cache is committed, so the eval reruns for free and anyone can check the
# numbers without an API key.
import base64
import hashlib
import io
import json
import time

from openai import OpenAI
from PIL import Image

from config import CACHE_DIR, MAX_IMAGE_SIDE, PROMPT_VERSION
from schema import ExtractedReceipt

PROMPT = (
    "Extract the data from this receipt photo. Copy amounts exactly as printed (keep the dots and "
    "commas, do not convert them). line_items has one entry per purchased product with its "
    "quantity and the line price. subtotal, tax and total are null if the receipt does not print "
    "them. Do not guess values you cannot read."
)

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        from dotenv import load_dotenv

        load_dotenv()
        _client = OpenAI(max_retries=5)
    return _client


def prepare_image(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def cache_path(model: str, temperature: float, run: int, image_sha256: str):
    key = hashlib.sha1(f"{PROMPT_VERSION}|{model}|{temperature}|{run}|{image_sha256}".encode())
    return CACHE_DIR / "llm" / f"{model}-{key.hexdigest()}.json"


def extract(image_bytes: bytes, image_sha256: str, model: str, temperature: float, run: int):
    path = cache_path(model, temperature, run, image_sha256)
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return ExtractedReceipt.model_validate(cached["parsed"]), cached["usage"]

    b64 = prepare_image(image_bytes)
    started = time.time()
    resp = client().chat.completions.parse(
        model=model,
        temperature=temperature,
        response_format=ExtractedReceipt,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                                                "detail": "high"}},
        ]}],
    )
    parsed = resp.choices[0].message.parsed
    usage = {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens,
             "seconds": round(time.time() - started, 2)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"parsed": parsed.model_dump(), "usage": usage}), encoding="utf-8")
    return parsed, usage
