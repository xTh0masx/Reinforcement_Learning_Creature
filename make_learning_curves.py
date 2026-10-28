"""Learning curves of every walking brain version, one figure per species (for the presentation).

Every TensorBoard run runs/<prefix>_legs_* is assigned to its brain version (v1 ... v12; pilots, resumed stages and
fixes of one version share its colour). The reward was defined differently in almost every version (v1-v8 add up
penalties, up to ~3000 per episode; v9+ multiply them, 0-50) and often between the stages of one version, so every
version gets its own panel with its own scale (raw smoothed mean episode reward). The x axis is the brain's total
training steps (resumed runs continue where their start brain ended).

    python make_learning_curves.py            # docs/figures/locomotion/learning_curves_all_<species>.png
"""
import glob
import os
import re

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT = "docs/figures/locomotion"
PREFIX = {"myrranox": "myr", "dc": "dc"}
TITLE = {"myrranox": "Myrranox", "dc": "DC"}
COLORS = ["#8FA89B", "#5B6C8F", "#2E7D6B", "#C9B27C", "#B23A48", "#D98E04", "#7A5195", "#1F3A35", "#EF8A62",
          "#3E9BBF", "#A26769", "#000000"]


def version_of(name):
    """'dc_legs_v12z' -> 'v12', 'myr_legs_pilot2' -> 'Pilot', 'dc_legs_v9_s1f' -> 'v9'."""
    m = re.search(r"_legs_(v\d+|pilot)", name)
    if not m:
        return None
    return "Pilot" if m.group(1) == "pilot" else m.group(1)


def read(species):
    runs = {}
    for d in sorted(glob.glob(f"runs/{PREFIX[species]}_legs_*")):
        name = os.path.basename(d).rsplit("_", 1)[0]
        v = version_of(name)
        if v is None:
            continue
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        if "rollout/ep_rew_mean" not in ea.Tags().get("scalars", []):
            continue
        ev = ea.Scalars("rollout/ep_rew_mean")
        if len(ev) < 5:
            continue
        runs.setdefault(v, []).append((np.array([e.step for e in ev]) / 1e6, np.array([e.value for e in ev])))
    return runs


def smooth(y, n=15):
    if len(y) < n:
        return y
    k = np.ones(n) / n
    return np.convolve(np.pad(y, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


NOTES = {"myrranox": {"Pilot": "Stillstand", "v1": "Mittelbeine", "v2": "Dreifuß-Takt", "v3": "gescheitert",
                     "v4": "präzise", "v5": "Knie frei", "v9": "Schubkarre", "v10": "final"},
         "dc": {"Pilot": "Stillstand", "v1": "läuft", "v2": "Schritt / Trab", "v3": "Kopf aktiv", "v4": "präzise",
                "v5": "Panther-Schritt", "v6": "6 Glieder", "v8": "Handstand", "v9": "neue Belohnung",
                "v11": "Katzen-Gewichte", "v12": "Wolf → final"}}


def plot(species):
    """Small multiples: one panel per version with its real (raw) mean episode reward - the reward definition differs
    between versions, so each panel has its own scale. Runs of one version (pilots, stages, resumes) share the panel."""
    runs = read(species)
    order = sorted(runs, key=lambda v: (-1 if v == "Pilot" else int(v[1:])))
    cols = 3
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(9.6, 2.35 * rows + 0.6), squeeze=False)
    for i, v in enumerate(order):
        ax = axes[i // cols][i % cols]
        col = COLORS[i % len(COLORS)]
        curves = [(x, smooth(y)) for x, y in runs[v] if len(y) >= 10]
        for x, y in curves:
            ax.plot(x, y, color=col, lw=1.6)
        n = len(curves)
        runs_txt = "1 Lauf" if n == 1 else f"{n} Läufe"
        ax.set_title(f"{v} – {NOTES[species].get(v, '')} · {runs_txt}", fontsize=10, fontweight="bold", loc="left")
        # axis range without each run's first 10 % (a resumed run starts with episodes of the old reward) and robust to
        # single runs on another scale
        tail = np.concatenate([y[len(y) // 10:] for _, y in curves])
        lo = np.percentile(np.concatenate([y for _, y in curves]), 1)     # the low start is real learning
        hi = np.percentile(tail, 95)
        pad = 0.12 * max(hi - lo, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    for j in range(len(order), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"{TITLE[species]}: Lernkurven aller Gehirn-Versionen", fontsize=15, fontweight="bold")
    fig.supxlabel("Trainingsschritte des Gehirns [Mio.]", fontsize=11)
    fig.supylabel("Belohnung je Episode (eigene Skala je Version)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT, f"learning_curves_all_{species}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"{species}: {len(order)} versions, {sum(len(runs[v]) for v in order)} runs -> {path}")


if __name__ == "__main__":
    for sp in ("myrranox", "dc"):
        plot(sp)
