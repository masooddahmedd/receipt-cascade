# Runs the cascade on dev and test, fits the confidence models on dev, sweeps the threshold and
# writes every number the README quotes: results.json, results_table.md, curve.png and one JSON
# per test receipt in out/test/. Rerunning is free once cache/ is filled.
import argparse
import json
from difflib import SequenceMatcher

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cascade import run_split
from confidence import ConfidenceModel
from config import OUT_DIR, PRICE_PER_M, ROOT, SCORED_FIELDS, TIER1_MODEL, TIER2_MODEL
from data import load_split

NAME_SIMILARITY = 0.8


def items_match(pred, truth) -> bool:
    # Greedy one-to-one matching. Exact name equality would punish harmless spelling differences,
    # so names only need to be close, but quantity and price must be exact.
    if len(pred) != len(truth):
        return False
    left = list(truth)
    for name, qty, price in pred:
        hit = next(
            (
                t
                for t in left
                if t[1] == qty
                and t[2] == price
                and SequenceMatcher(None, name, t[0]).ratio() >= NAME_SIMILARITY
            ),
            None,
        )
        if hit is None:
            return False
        left.remove(hit)
    return True


def is_correct(field: str, value, truth: dict) -> bool:
    if field == "line_items":
        return items_match(value, truth["line_items"])
    return value == truth[field]


def build_rows(results: list[dict]) -> list[dict]:
    rows = []
    for idx, res in enumerate(results):
        truth = res["receipt"].truth
        for f in SCORED_FIELDS:
            if f not in truth:
                continue
            d = res["fields"][f]
            rows.append(
                {
                    "receipt": idx,
                    "field": f,
                    "x1": d["tier1_features"],
                    "x2": d["tier2_features"],
                    "ok1": is_correct(f, d["tier1_value"], truth),
                    "ok2": is_correct(f, d["tier2_value"], truth),
                }
            )
    return rows


def call_cost(usage: dict, model: str) -> float:
    p = PRICE_PER_M[model]
    return (usage["in"] * p["in"] + usage["out"] * p["out"]) / 1e6


def score_policy(rows, results, policy: str, t: float) -> dict:
    # policy is tier1 (majority vote only), tier2 (stronger model only) or cascade. Anything below
    # the threshold after the last tier counts as needs_review, not as an error. A Tier 2 call is
    # charged once per receipt that has at least one escalated field.
    accepted = correct = 0
    escalated = set()
    for r in rows:
        if policy == "tier1":
            take, ok = r["c1"] >= t, r["ok1"]
        elif policy == "tier2":
            take, ok = r["c2"] >= t, r["ok2"]
        elif r["c1"] >= t:
            take, ok = True, r["ok1"]
        else:
            escalated.add(r["receipt"])
            take, ok = r["c2"] >= t, r["ok2"]
        accepted += int(take)
        correct += int(take and ok)
    n, m = len(rows), len(results)
    cost = latency = 0.0
    for i, res in enumerate(results):
        t1_cost = sum(call_cost(u, TIER1_MODEL) for u in res["tier1_usage"])
        t1_sec = max(u["seconds"] for u in res["tier1_usage"])
        t2_cost = call_cost(res["tier2_usage"], TIER2_MODEL)
        t2_sec = res["tier2_usage"]["seconds"]
        if policy == "tier1":
            cost, latency = cost + t1_cost, latency + t1_sec
        elif policy == "tier2":
            cost, latency = cost + t2_cost, latency + t2_sec
        else:
            esc = i in escalated
            cost += t1_cost + (t2_cost if esc else 0)
            latency += t1_sec + (t2_sec if esc else 0)
    return {
        "threshold": round(float(t), 3),
        "coverage": accepted / n,
        "accuracy_of_accepted": correct / accepted if accepted else float("nan"),
        "end_to_end_accuracy_with_review": (correct + (n - accepted)) / n,
        "needs_review_fields": n - accepted,
        "cost_per_1000": cost / m * 1000,
        "mean_latency_s": latency / m,
        "receipts_escalated": len(escalated),
    }


def validator_report(rows) -> dict:
    # How often each validator fires on a wrong Tier 1 field, and how often on a right one.
    out = {}
    for f in SCORED_FIELDS:
        sub = [r for r in rows if r["field"] == f]
        wrong = [r for r in sub if not r["ok1"]]
        right = [r for r in sub if r["ok1"]]
        out[f] = {
            "wrong_fields": len(wrong),
            "caught_by_validator": sum(r["x1"][1] == 0 for r in wrong),
            "correct_fields": len(right),
            "false_alarms_on_correct": sum(r["x1"][1] == 0 for r in right),
        }
    return out


def attach_confidence(results, rows, m1: ConfidenceModel, m2: ConfidenceModel) -> None:
    for res in results:
        for d in res["fields"].values():
            d["c1"] = float(m1.predict([d["tier1_features"]])[0])
            d["c2"] = float(m2.predict([d["tier2_features"]])[0])
    for r in rows:
        d = results[r["receipt"]]["fields"][r["field"]]
        r["c1"], r["c2"] = d["c1"], d["c2"]


def write_receipt_json(results, t: float, split: str) -> None:
    out_dir = OUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for res in results:
        rec = res["receipt"]
        fields = {}
        for f in SCORED_FIELDS + ["merchant", "date"]:
            d = res["fields"][f]
            if d["c1"] >= t:
                fields[f] = {
                    "value": d["tier1_raw"],
                    "confidence": round(d["c1"], 4),
                    "tier": 1,
                    "status": "accepted",
                }
            else:
                status = "accepted" if d["c2"] >= t else "needs_review"
                fields[f] = {
                    "value": d["tier2_raw"],
                    "confidence": round(d["c2"], 4),
                    "tier": 2,
                    "status": status,
                }
        overall = min(fields[f]["confidence"] for f in SCORED_FIELDS)
        doc = {
            "receipt_id": rec.receipt_id,
            "image_sha256": rec.image_sha256,
            "overall_confidence": overall,
            "fields": fields,
        }
        (out_dir / f"{rec.receipt_id}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")


def accepted_accuracy(rows, policy: str, t: float):
    kept = []
    for r in rows:
        if policy == "tier2":
            take, ok = r["c2"] >= t, r["ok2"]
        elif r["c1"] >= t:
            take, ok = True, r["ok1"]
        else:
            take, ok = r["c2"] >= t, r["ok2"]
        if take:
            kept.append(ok)
    return float(np.mean(kept)) if kept else None


def bootstrap_ci(rows, policy: str, t: float, n_boot: int = 1000) -> list[float]:
    # Resample receipts, not fields, because the fields of one receipt are not independent.
    rng = np.random.default_rng(0)
    by_receipt: dict[int, list] = {}
    for r in rows:
        by_receipt.setdefault(r["receipt"], []).append(r)
    ids = list(by_receipt)
    accs = []
    for _ in range(n_boot):
        sample = [r for i in rng.choice(ids, size=len(ids)) for r in by_receipt[i]]
        acc = accepted_accuracy(sample, policy, t)
        if acc is not None:
            accs.append(acc)
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


def markdown_table(op: dict, t: float) -> str:
    lines = [
        "| Policy | Fields automated | Accuracy of automated fields | Accuracy if reviewed "
        "fields are fixed | Cost per 1,000 receipts | Mean latency (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    names = {
        "tier1": f"Tier 1 only ({TIER1_MODEL}, 3 runs)",
        "tier2": f"Tier 2 only ({TIER2_MODEL})",
        "tier2_gated": f"Tier 2 only, gated at the same threshold ({t:.2f})",
        "cascade": f"Cascade (threshold {t:.2f})",
    }
    for p in ("tier1", "tier2", "tier2_gated", "cascade"):
        r = op[p]
        lines.append(
            f"| {names[p]} | {r['coverage']:.1%} | {r['accuracy_of_accepted']:.1%} | "
            f"{r['end_to_end_accuracy_with_review']:.1%} | ${r['cost_per_1000']:.2f} | "
            f"{r['mean_latency_s']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def plot(curves: dict, op: dict, t: float, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = {
        "tier1": "Tier 1 only (gated)",
        "tier2": "Tier 2 only (confidence-gated)",
        "cascade": "Cascade (gated)",
    }
    for p, pts in curves.items():
        pts = [c for c in pts if c["coverage"] > 0.05]
        ax.plot(
            [c["coverage"] * 100 for c in pts],
            [c["accuracy_of_accepted"] * 100 for c in pts],
            label=labels[p],
            linewidth=2,
        )
    ax.scatter(
        [op["cascade"]["coverage"] * 100],
        [op["cascade"]["accuracy_of_accepted"] * 100],
        color="black",
        zorder=5,
        label=f"cascade operating point ({t:.2f})",
    )
    ax.set_xlabel("% of fields accepted automatically")
    ax.set_ylabel("accuracy of accepted fields (%)")
    ax.set_title("Accuracy vs automation on the CORD v2 test split")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / f"curve_{mode}.png", dpi=150)


def main(limit: int | None, mode: str) -> None:
    dev, test = load_split("dev")[:limit], load_split("test")[:limit]
    dev_res, test_res = run_split(dev, mode), run_split(test, mode)
    dev_rows, test_rows = build_rows(dev_res), build_rows(test_res)

    m1 = ConfidenceModel().fit([r["x1"] for r in dev_rows], [r["ok1"] for r in dev_rows])
    m2 = ConfidenceModel().fit([r["x2"] for r in dev_rows], [r["ok2"] for r in dev_rows])
    attach_confidence(dev_res, dev_rows, m1, m2)
    attach_confidence(test_res, test_rows, m1, m2)

    grid = np.linspace(0.0, 1.0, 101)
    curves = {
        p: [score_policy(test_rows, test_res, p, t) for t in grid]
        for p in ("tier1", "tier2", "cascade")
    }

    # The operating point is chosen on dev: the highest-coverage threshold whose accepted-field
    # accuracy is within a point of Tier 2 run on everything (the 100%-automation baseline).
    dev_tier2_acc = float(np.mean([r["ok2"] for r in dev_rows]))
    dev_curve = [score_policy(dev_rows, dev_res, "cascade", t) for t in grid]
    eligible = [c for c in dev_curve if c["accuracy_of_accepted"] >= dev_tier2_acc - 0.01]
    op_t = (
        max(eligible, key=lambda c: (c["coverage"], -c["threshold"]))["threshold"]
        if eligible
        else 0.5
    )

    op = {
        "tier1": score_policy(test_rows, test_res, "tier1", 0.0),
        "tier2": score_policy(test_rows, test_res, "tier2", 0.0),
        "tier2_gated": score_policy(test_rows, test_res, "tier2", op_t),
        "cascade": score_policy(test_rows, test_res, "cascade", op_t),
    }
    op["tier2"]["accuracy_ci95"] = bootstrap_ci(test_rows, "tier2", 0.0)
    op["cascade"]["accuracy_ci95"] = bootstrap_ci(test_rows, "cascade", op_t)

    summary = {
        "n_dev_receipts": len(dev),
        "n_test_receipts": len(test),
        "n_dev_scored_fields": len(dev_rows),
        "n_test_scored_fields": len(test_rows),
        "operating_threshold": op_t,
        "dev_tier2_only_accuracy": dev_tier2_acc,
        "operating_point_test": op,
        "per_field_accuracy_test": {
            f: {
                "n": sum(r["field"] == f for r in test_rows),
                "tier1": float(np.mean([r["ok1"] for r in test_rows if r["field"] == f])),
                "tier2": float(np.mean([r["ok2"] for r in test_rows if r["field"] == f])),
            }
            for f in SCORED_FIELDS
        },
        "validators_on_tier1_test": validator_report(test_rows),
        "validators_on_tier1_dev": validator_report(dev_rows),
        "tier1_confidence_weights": dict(
            zip(
                ["agreement", "validator", "ocr_match", *SCORED_FIELDS],
                [round(float(w), 3) for w in m1.w[:-1]],
            )
        ),
        "models": {"tier1": TIER1_MODEL, "tier2": TIER2_MODEL},
    }
    (ROOT / f"results_{mode}.json").write_text(
        json.dumps({"summary": summary, "curves": curves}, indent=1)
    )
    (ROOT / f"results_table_{mode}.md").write_text(markdown_table(op, op_t), encoding="utf-8")
    plot(curves, op, op_t, mode)
    write_receipt_json(test_res, op_t, f"test_{mode}")
    print((ROOT / f"results_table_{mode}.md").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("n_test_scored_fields", "operating_threshold", "validators_on_tier1_test")
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="use only the first N receipts per split (for a cheap smoke run)",
    )
    ap.add_argument("--tier1-input", choices=["plain", "layout"], default="layout")
    args = ap.parse_args()
    main(args.limit, args.tier1_input)
