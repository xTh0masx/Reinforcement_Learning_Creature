"""Video of a whole predator (navigation) training run: every checkpoint lives one 10-minute day in the habitat
(top-down, 30x speed) next to the learning curve and live values (hunger, fatigue, prey, raids).

Each run is played under the conditions it was trained with (RUNS below), so early brains look like they really were.

    python make_predator_video.py --run dc_brain_v4_r1
    python make_predator_video.py --all
    -> docs/videos/predator/<run>_training.mp4   (1280x720, 25 fps)
"""
import argparse
import glob
import io
import os
import re

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from make_training_video import ACCENT, BAD, BG, DIM, FG, font
from sim.creature_env import CreatureEnv
from sim.nav_brain import NavBrain

W, H, FPS = 1280, 720, 25
MAP = 720                       # habitat view on the left
STEPS_PER_FRAME = 12            # 1.2 s of the day per video frame -> a 10-min day in 20 s
TITLE = {"myrranox": "Myrranox", "dc": "DC_SKIN_RAW"}

# Training conditions per run (PROJECT_PLAN / docs/predator_training_log.md)
OLD = dict(action_mapping="legacy", tired_from=1.0, intrude=0.0, prey=(0.3, 0.7))
RUNS = {
    "myr_brain_v1_r1": dict(OLD, note="attempt 1 - plan reward"),
    "dc_brain_v1_r1": dict(OLD, note="attempt 1 - plan reward"),
    "myr_brain_v2_r1": dict(OLD, note="attempt 2 - reward v2 (stopped at 2.8M)"),
    "dc_brain_v2_r1": dict(OLD, note="attempt 2 - reward v2 (stopped at 2.8M)"),
    "myr_brain_v3_r1": dict(action_mapping="full", tired_from=1.0, intrude=0.0, prey=(0.3, 0.7),
                            note="round 1 - exploration fixes"),
    "dc_brain_v3_r1": dict(action_mapping="full", tired_from=1.0, intrude=0.0, prey=(0.3, 0.7),
                           note="round 1 - exploration fixes"),
    "myr_brain_v4_r1": dict(action_mapping="full", tired_from=0.7, intrude=0.5, prey=(0.3, 0.7),
                            note="option B - home stock, lean days, invading rival"),
    "dc_brain_v4_r1": dict(action_mapping="full", tired_from=0.7, intrude=0.5, prey=(0.3, 0.7),
                           note="option B - home stock, lean days, invading rival (released)"),
}


def checkpoints(run):
    out = []
    for p in glob.glob(f"checkpoints/{run}_*_steps.zip"):
        m = re.search(r"_(\d+)_steps\.zip$", p)
        if m and "vecnormalize" not in p:
            out.append((int(m.group(1)), p[:-4]))
    return sorted(out)


def curve(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    xs, ys = [], []
    for d in sorted(glob.glob(f"runs/{run}_*")):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        if "habitat/prey_per_10min" in ea.Tags().get("scalars", []):
            for e in ea.Scalars("habitat/prey_per_10min"):
                xs.append(e.step)
                ys.append(e.value)
    o = np.argsort(xs)
    return np.array(xs)[o], np.array(ys)[o]


def curve_img(xs, ys, marker, total, width):
    fig, ax = plt.subplots(figsize=(width / 100, 2.4), dpi=100)
    bg = np.array(BG) / 255
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    if len(xs):
        ax.plot(xs / 1e6, ys, color=np.array(DIM) / 255, lw=1)
        done = xs <= marker
        ax.plot(xs[done] / 1e6, ys[done], color=np.array(ACCENT) / 255, lw=2)
    ax.axvline(marker / 1e6, color=np.array(ACCENT) / 255, ls="--", lw=1)
    ax.set_xlim(0, max(total, 1) / 1e6)
    ax.set_xlabel("training steps [M]", color="white", fontsize=9)
    ax.set_ylabel("prey / 10 min", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#666")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=bg)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def bar(d, x, y, w, v, color, label):
    d.text((x, y - 2), label, fill=DIM, font=font(15))
    d.rectangle([x + 70, y, x + 70 + w, y + 12], fill=(60, 63, 70))
    d.rectangle([x + 70, y, x + 70 + w * float(np.clip(v, 0, 1)), y + 12], fill=color)


def card(lines, sizes):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = H // 2 - sum(s + 16 for s in sizes) // 2
    for text, size in zip(lines, sizes):
        f = font(size, bold=size >= 40)
        d.text(((W - d.textlength(text, font=f)) / 2, y), text, fill=FG if size >= 40 else DIM, font=f)
        y += size + 16
    return np.asarray(img)


def make(run, out_dir="docs/videos/predator"):
    cfg = RUNS[run]
    species = "myrranox" if run.startswith("myr") else "dc"
    rival = "dc" if species == "myrranox" else "myrranox"
    cks = checkpoints(run)
    xs, ys = curve(run)
    total = max([c[0] for c in cks] + [int(xs.max()) if len(xs) else 0])
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{run}_training.mp4")
    writer = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    title = card([f"{TITLE[species]} learns to hunt", f"predator brain {run}  -  {cfg['note']}",
                  f"{len(cks)} checkpoints, untrained -> {total / 1e6:.1f}M steps; each lives one 10-minute day (30x speed)",
                  f"rival: scripted {TITLE[rival]}; prey flee at {cfg['prey'][1]} m/s (training conditions)"],
                 [48, 24, 20, 20])
    for _ in range(FPS * 4):
        writer.append_data(title)

    for idx, (steps, path) in enumerate(cks):
        brain = NavBrain(path)
        obs_v = 1 if brain.model.observation_space.shape[0] == 81 else 2
        env = CreatureEnv(species, render_mode="rgb_array", obs_version=obs_v,
                          others=[{"species": rival, "intrude_prob": cfg["intrude"]}],
                          action_mapping=cfg["action_mapping"], tired_from=cfg["tired_from"])
        env.habitat.prey["wander_speed"], env.habitat.prey["flee_speed"] = cfg["prey"]
        obs, _ = env.reset(seed=1000 + idx)
        brain.reset()
        panel_bg = Image.new("RGB", (W - MAP, H), BG)
        panel_bg.paste(curve_img(xs, ys, steps, total, W - MAP), (0, 190))
        starved, n = False, 0
        while True:
            obs, _, term, trunc, _ = env.step(brain(obs))
            n += 1
            if n % STEPS_PER_FRAME and not (term or trunc):
                continue
            view = Image.fromarray(env.render()).resize((MAP, MAP))
            img = Image.new("RGB", (W, H), BG)
            img.paste(view, (0, 0))
            img.paste(panel_bg, (MAP, 0))
            d = ImageDraw.Draw(img)
            x0 = MAP + 24
            c, s = env.ego, env.ego.stats
            d.text((x0, 18), f"{TITLE[species]} - learning to hunt", fill=FG, font=font(22, True))
            d.text((x0, 50), f"checkpoint {idx + 1} / {len(cks)}", fill=DIM, font=font(16))
            d.text((x0, 72), "untrained" if steps == 0 else f"{steps / 1e6:.2f}M steps", fill=ACCENT, font=font(40, True))
            d.rectangle([x0, 130, W - 24, 137], fill=(60, 63, 70))
            d.rectangle([x0, 130, x0 + (W - 24 - x0) * (idx + env.t / 6000) / len(cks), 137], fill=ACCENT)
            d.text((x0, 150), f"day time {env.t / 600:4.1f} / 10 min", fill=DIM, font=font(16))
            y = 450
            bar(d, x0, y, 300, c.hunger, (217, 87, 80), "hunger")
            bar(d, x0, y + 26, 300, c.fatigue, (80, 130, 220), "fatigue")
            d.text((x0, y + 60), f"prey caught {s['prey_caught']}   raids {s['raid_prey_eaten']}   "
                                 f"intruders repelled {s['intruders_repelled']}", fill=FG, font=font(16))
            d.text((x0, y + 86), f"activity {c.activity}   hits {s['hits']}", fill=DIM, font=font(15))
            if term:
                starved = True
                d.text((x0, y + 120), "STARVED", fill=BAD, font=font(34, True))
            frame = np.asarray(img)
            writer.append_data(frame)
            if term or trunc:
                for _ in range(FPS // 2 if not starved else FPS):
                    writer.append_data(frame)
                break
        env.close()
        print(f"  {run} @ {steps / 1e6:.2f}M: prey {env.ego.stats['prey_caught']}, "
              f"{'starved at %.0f s' % (env.t / 10) if starved else 'survived the day'}", flush=True)
    writer.close()
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    for run in (RUNS if args.all else [args.run]):
        make(run)
