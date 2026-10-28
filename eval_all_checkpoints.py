"""One comparable measure for every saved walking brain of both creatures (the reward changed between versions, so
reward curves cannot be compared): command "walk at 0.6 m/s", 2 seeds x 8 s after a 2 s start.
Walking quality (0-100 %) = speed accuracy exp(-((speed - 0.6) / 0.3)^2) x steady trunk exp(-bob / 4 cm) x even
rhythm exp(-timing variation / 40 %) x 0.5 per fall; the speed accuracy alone is kept as well. Every checkpoint is tested on the body it was trained with
(LocomotionEnv versions), except the early DC v12 stages before the hand / arm remodel of 14 Oct (their body no longer
exists - marked as a gap).

    python eval_all_checkpoints.py        # docs/evaluations/all_checkpoints_walk.csv + docs/figures/locomotion/learning_curve_common.png
"""
import csv
import glob
import math
import os
import re
from multiprocessing import Pool

import numpy as np

CMD = 0.6
SKIP = {"dc_legs_v12", "dc_legs_v12b", "dc_legs_v12c", "dc_legs_v12d", "dc_legs_v12e", "dc_legs_v12f", "dc_legs_v12s"}
CSV = "docs/evaluations/all_checkpoints_walk.csv"
PNG = "docs/figures/locomotion/learning_curve_common.png"


def runs():
    """{run: [(steps, path), ...]} sorted by steps, and the runs in the order they were trained (first checkpoint time)."""
    out = {}
    for f in glob.glob("checkpoints/*_legs_*_steps.zip"):
        m = re.match(r"checkpoints[\\/](.+)_(\d+)_steps\.zip", f)
        if m:
            out.setdefault(m.group(1), []).append((int(m.group(2)), f[:-4]))
    for r in out:
        out[r].sort()
    order = sorted(out, key=lambda r: min(os.path.getmtime(p + ".zip") for _, p in out[r]))
    return out, order


def version(run):
    m = re.search(r"_legs_(v\d+|pilot)", run)
    return m.group(1) if m else "?"


def score(task):
    run, steps, path = task
    import torch
    torch.set_num_threads(1)
    from play_locomotion import Brain
    from sim.locomotion_env import CONTROL_DT, LocomotionEnv
    species = "myrranox" if run.startswith("myr") else "dc"
    try:
        b = Brain(path, species)
        env = LocomotionEnv(species, version=b.version, episode_seconds=1e9)
        res, acc = [], []
        legs = env.is_leg
        for seed in (0, 1):
            obs, _ = env.reset(seed=seed, options={"command": (CMD, 0.0)})
            sp, z, cont, falls = [], [], [], 0
            for k in range(int(10 / CONTROL_DT)):
                obs, _, term, _, info = env.step(b(obs))
                if term:
                    falls += 1
                    obs, _ = env.reset(seed=seed + k, options={"command": (CMD, 0.0)})
                if k * CONTROL_DT >= 2:
                    sp.append(info["speed"]); z.append(float(env.data.qpos[env.q0 + 2]))
                    cont.append(info["foot_contact"][legs].copy())
            v = float(np.mean(sp))
            a = math.exp(-((v - CMD) / 0.3) ** 2)
            z = np.array(z)
            bob = float(np.mean([np.ptp(z[i:i + 50]) for i in range(0, len(z) - 50, 50)]))     # per second
            c = np.array(cont)
            cvs = []
            for j in range(c.shape[1]):
                t = np.flatnonzero(c[1:, j] & ~c[:-1, j])
                if len(t) > 2:
                    cvs.append(np.std(np.diff(t)) / np.mean(np.diff(t)))
            cv = float(np.mean(cvs)) if cvs else 1.0            # no steps at all = no rhythm
            res.append(100 * a * math.exp(-bob / 0.04) * math.exp(-cv / 0.4) * 0.5 ** falls)
            acc.append(100 * a * 0.5 ** falls)
        return run, steps, species, b.version, float(np.mean(res)), float(np.mean(acc)), ""
    except Exception as e:                                       # noqa: BLE001
        return run, steps, species, -1, float("nan"), float("nan"), str(e)[:80]


def evaluate():
    out, order = runs()
    tasks = [(r, n, p) for r in order if r not in SKIP for n, p in out[r]]
    print(f"{len(tasks)} checkpoints in {len(order)} runs")
    with Pool(7) as pool:
        rows = pool.map(score, tasks, chunksize=2)
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    with open(CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "steps", "species", "version", "quality", "speed_accuracy", "error"])
        w.writerows(rows)
    return order


FINAL_NAME = {"myrranox": "v10", "dc": "v12z"}


def plot(order=None):
    """x = brain version, one line per creature: best checkpoint of each version (thick), every checkpoint (dots)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["error"]:
                continue
            per.setdefault((r["species"], version(r["run"])), []).append(float(r["quality"]))
    final = {}                                                  # released brain = last checkpoint of the final run
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["run"] in ("myr_legs_v10", "dc_legs_v12z") and not r["error"]:
                if int(r["steps"]) >= final.get(r["species"], (0, 0))[0]:
                    final[r["species"]] = (int(r["steps"]), float(r["quality"]))
    global FINAL
    FINAL = {"myrranox": ("v10", final["myrranox"][1]), "dc": ("v12", final["dc"][1])}
    key = lambda v: -1 if v == "pilot" else int(v[1:])
    versions = sorted({v for _, v in per}, key=key)
    xpos = {v: i for i, v in enumerate(versions)}
    fig, ax = plt.subplots(figsize=(12, 5.8))
    rng = np.random.default_rng(0)
    style = {"myrranox": ("#2E7D6B", "Myrranox", -0.12), "dc": ("#5B6C8F", "DC", 0.12)}
    for species, (col, name, dx) in style.items():
        vs = [v for v in versions if (species, v) in per]
        for v in vs:
            q = per[(species, v)]
            ax.scatter(xpos[v] + dx + rng.uniform(-0.07, 0.07, len(q)), q, s=12, color=col, alpha=0.35, lw=0)
        best = [max(per[(species, v)]) for v in vs]
        ax.plot([xpos[v] + dx for v in vs], best, color=col, lw=2.8, marker="o", ms=7, label=f"{name}: bestes Gehirn je Version")
        fv, fq = FINAL[species]
        ax.scatter([xpos[fv] + dx], [fq], marker="*", s=360, color=col, edgecolor="white", lw=1.2, zorder=5,
                   label=f"{name}: freigegebenes Gehirn ({FINAL_NAME[species]})")
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(["Pilot" if v == "pilot" else v for v in versions], fontsize=12)
    ax.set_xlabel("Gehirn-Version", fontsize=12)
    ax.set_ylabel("Lauf-Qualität beim Gehen (0,6 m/s) [%]", fontsize=12)
    ax.set_title("Lernkurve beider Kreaturen über alle Gehirn-Versionen", fontsize=15, fontweight="bold")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10.5, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(PNG, dpi=150)
    plt.close(fig)
    print("wrote", PNG)


if __name__ == "__main__":
    import sys
    if "--plot-only" not in sys.argv:
        evaluate()
    plot()
