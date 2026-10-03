# Renders the headline results table as a PNG for the LinkedIn post. Numbers are read from
# results_layout.json, the same file the README table is copied from, so nothing is typed by hand.
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ROOT

op = json.loads((ROOT / "results_layout.json").read_text())["summary"]["operating_point_test"]
t = json.loads((ROOT / "results_layout.json").read_text())["summary"]["operating_threshold"]
rows = [
    ("Cheap model only\n(gpt-4o-mini on OCR text)", op["tier1"]),
    ("Strong model only\n(gpt-4o on the image)", op["tier2"]),
    (f"Strong model, gated by\nits own confidence ({t:.2f})", op["tier2_gated"]),
    (f"Cascade: cheap first, escalate\nshaky fields ({t:.2f})", op["cascade"]),
]
cells = [
    [name, f"{r['coverage']:.1%}", f"{r['accuracy_of_accepted']:.1%}", f"${r['cost_per_1000']:.2f}"]
    for name, r in rows
]
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.axis("off")
table = ax.table(
    cellText=cells,
    colLabels=["", "Fields automated", "Accuracy of\nautomated fields", "Cost per\n1,000 receipts"],
    loc="center",
    cellLoc="center",
    colWidths=[0.34, 0.22, 0.22, 0.22],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.6)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(weight="bold")
    if c == 0 and r > 0:
        cell.set_text_props(ha="left")
        cell._loc = "left"
ax.set_title("CORD v2 test split: 100 receipts, 300 fields", fontsize=12, pad=8)
fig.tight_layout()
fig.savefig(ROOT / "table_layout.png", dpi=170)
