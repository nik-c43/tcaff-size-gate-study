"""Figure for check 4 (docs/3_hypotheses.md) from results/segment_unit/unit_test/aggregate.json (layout only, no new experiment).

Usage: MPLBACKEND=Agg python plot_unit_test.py [OUT_DIR]   (default results/figures/publication)
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/segment_unit/unit_test/aggregate.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/figures/publication"
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e6e5e0", "#a3a29b"
COLOR = {"R0": "#35629e", "U3": "#c96729"}  # same blue/orange roles as the repo's earlier figures; validated
NAMES = {"B0": "B0, авторские детекции", "R0": "4.0, контроль", "U1": "4.1, без контейнеров",
         "U2": "4.2, без многослойных", "U3": "4.3, основной"}

agg = json.loads(SRC.read_text())
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})

# (a) Variants: mean and min-max over repeats.
order = ["B0", "R0", "U1", "U2", "U3"]
for y, v in enumerate(reversed(order)):
    a = agg["variants"][v]["correct_share_all_updates"]
    c = COLOR.get(v, MUTED)
    ax1.plot([100 * a["min"], 100 * a["max"]], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
    ax1.scatter([100 * a["mean"]], [y], s=70, color=c, edgecolor="white", linewidth=2, zorder=3)
    ax1.text(100 * a["max"] + 1.2, y, f"{100 * a['mean']:.1f}%", va="center", color=INK, fontsize=10)
ax1.set_yticks(range(len(order)), [NAMES[v] for v in reversed(order)])
ax1.set_xlim(45, 85)
ax1.set_xlabel("Доля всех обновлений пар с ошибкой ≤1 м и ≤5°, %")
ax1.set_title("(a) Варианты: среднее и размах 5 повторов", loc="left", fontsize=11, color=INK)
ax1.grid(axis="x", color=GRID, lw=1)
ax1.set_axisbelow(True)

# (b) Directed pairs: R0 vs U3, sorted by the change.
pp = agg["per_pair"]
pairs = sorted(pp["R0"], key=lambda p: pp["U3"][p]["correct_share_mean"] - pp["R0"][p]["correct_share_mean"], reverse=True)
for y, p in enumerate(pairs):
    r, u = 100 * pp["R0"][p]["correct_share_mean"], 100 * pp["U3"][p]["correct_share_mean"]
    ax2.plot([r, u], [y, y], color=GRID, lw=2, zorder=1)
    ax2.scatter([r], [y], s=70, color=COLOR["R0"], edgecolor="white", linewidth=2, zorder=3, label="4.0, контроль" if y == 0 else None)
    ax2.scatter([u], [y], s=70, color=COLOR["U3"], edgecolor="white", linewidth=2, zorder=3, label="4.3, основной" if y == 0 else None)
ax2.set_yticks(range(len(pairs)), [p.replace("_", " → ") for p in pairs])
ax2.set_xlim(-2, 100)
ax2.set_xlabel("Доля обновлений пары с ошибкой ≤1 м и ≤5°, %")
ax2.set_title("(b) Направленные пары роботов: 4.0 против 4.3", loc="left", fontsize=11, color=INK)
ax2.grid(axis="x", color=GRID, lw=1)
ax2.set_axisbelow(True)
ax2.legend(loc="lower left", frameon=False, labelcolor=INK)

fig.suptitle("Проверка 4: удаление контейнеров и многослойных масок ухудшает совмещение", fontsize=12, color=INK, x=0.01, ha="left")
fig.text(0.01, 0.015, "Одна запись, 337 обновлений × 12 направленных пар; пары зависимы. Неверное совмещение в (b) устойчиво: "
         "разброс доли между повторами не больше 0.05.", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
OUT.mkdir(parents=True, exist_ok=True)
outs = {}
for ext in ("png", "svg"):
    path = OUT / f"unit_hypothesis.{ext}"
    fig.savefig(path, dpi=160, metadata={"Date": None} if ext == "svg" else None)
    outs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
plt.close(fig)
prov = {"scope": "Layout only; no new experiment or changed numerical input",
        "command": "MPLBACKEND=Agg python scripts/segment_unit/plot_unit_test.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": {str(SRC.relative_to(ROOT)): hashlib.sha256(SRC.read_bytes()).hexdigest()},
        "outputs": outs}
(OUT / "unit_hypothesis_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
print(json.dumps(prov, indent=1))
