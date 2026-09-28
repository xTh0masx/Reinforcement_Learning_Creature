"""Turn a locomotion training run into presentation material: how walking developed from step 0 to the end.

For every checkpoint (checkpoints/<run>_<N>_steps.zip, step 0 = untrained) and the final model it runs
the same command script as play_locomotion.py (fixed seed, deterministic brain) and writes to
docs/figures/locomotion/<run>/:

    <N>_steps.gif        clip of that checkpoint, labelled with steps, command and actual speed
    progress.gif         all clips back to back (smaller), first attempt -> final brain
    contact_sheet.png    one frame per checkpoint in a grid
    metrics.csv / metrics.png   speed/turn tracking error, falls and foot slip per checkpoint
and docs/figures/locomotion/learning_curves_<species>.png with every TensorBoard run of the species
(including abandoned pilots).

    python locomotion_progress.py --run myr_legs_v1 --species myrranox
"""
import argparse
import csv
import glob
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from play_locomotion import Brain, command_script
from sim.locomotion_env import CONTROL_DT, LocomotionEnv

FRAME_EVERY = 3                 # control steps per GIF frame -> 16.7 fps
CLIP_SECONDS = 12.0             # walk 4 s, max speed 4 s, walk + left turn 4 s
WIDTH, HEIGHT = 640, 400
# What each training session was, for the learning-curve legend (see docs/locomotion_training_log.md)
RUN_NOTES = {
    "myr_legs_pilot": "pilot 1 - foot-slip penalty too strong (stopped)",
    "myr_legs_pilot2": "pilot 2 - learned to stand still (stopped)",
    "dc_legs_pilot": "pilot 1 - same reward as Myrranox pilot 2 (stopped)",
    "myr_legs_v1": "v1 - wider speed reward + penalty curriculum",
    "dc_legs_v1": "v1 - wider speed reward + penalty curriculum",
}


def checkpoints(run):
    found = []
    for path in glob.glob(f"checkpoints/{run}_*_steps.zip"):
        m = re.fullmatch(rf"{re.escape(run)}_(\d+)_steps\.zip", os.path.basename(path))   # not <run>_pilot_...
        if m:
            found.append((int(m.group(1)), path[:-4]))
    found.sort()
    if os.path.exists(f"models/{run}.zip"):
        steps = found[-1][0] if found else 0
        from stable_baselines3 import PPO
        steps = PPO.load(f"models/{run}", device="cpu").num_timesteps
        # the last checkpoint and the final model are only a rollout apart - keep the final model
        found = [c for c in found if c[0] < steps - 100_000] + [(steps, f"models/{run}")]
    return found


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def label(frame, lines):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    f_big, f_small = _font(22), _font(16)
    d.rectangle([0, 0, img.width, 58], fill=(0, 0, 0))
    d.text((10, 6), lines[0], fill=(255, 255, 255), font=f_big)
    d.text((10, 34), lines[1], fill=(220, 220, 220), font=f_small)
    return img


def fmt_steps(n):
    return "untrained" if n == 0 else (f"{n / 1e6:.2f}M steps" if n >= 1e6 else f"{n / 1e3:.0f}k steps")


def evaluate(path, species, seed=0, clip=True):
    """Run the command script; return metrics and labelled clip frames (first CLIP_SECONDS)."""
    brain = Brain(path, species)
    env = LocomotionEnv(species, episode_seconds=1e6, render_mode="rgb_array", width=WIDTH, height=HEIGHT,
                        version=brain.version)
    obs, _ = env.reset(seed=seed, options={"command": (0.0, 0.0)})
    frames, rows, t, fell_at = [], [], 0.0, None
    for seconds, v, w in command_script(species):
        env.set_command(v, w)
        for i in range(int(round(seconds / CONTROL_DT))):
            obs, _, term, _, info = env.step(brain(obs))
            t += CONTROL_DT
            rows.append((t, v, w, info["speed"], info["turn_rate"], info["terms"]["foot_slip"], i * CONTROL_DT))
            if clip and t <= CLIP_SECONDS and len(rows) % FRAME_EVERY == 0:
                frames.append((env.render().copy(), v, w, info["speed"]))
            if term:
                fell_at = t
                break
        if fell_at:
            break
    env.close()
    r = np.array(rows)
    settled = r[r[:, 6] >= 1.0]               # ignore the first second after each command change
    sp = env.species
    metrics = {
        "speed_error": float(np.mean(np.abs(settled[:, 3] - settled[:, 1]))) if len(settled) else float("nan"),
        "turn_error_deg": float(np.degrees(np.mean(np.abs(settled[:, 4] - settled[:, 2])))) if len(settled) else float("nan"),
        "walk_speed_at_walk_cmd": float(np.mean(settled[(settled[:, 1] == sp.walk_speed) & (settled[:, 2] == 0), 3]))
        if len(settled) else float("nan"),
        "restart_speed": float(np.mean(r[-int(3 / CONTROL_DT):, 3])) if fell_at is None else float("nan"),
        "fell_at_s": fell_at or "",
        "foot_slip": float(-np.mean(r[:, 5])),
    }
    return metrics, frames


def learning_curves(species, out_png):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    prefix = {"myrranox": "myr", "dc": "dc"}[species]
    fig, ax = plt.subplots(figsize=(9, 5))
    for d in sorted(glob.glob(f"runs/{prefix}_legs_*")):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        if "rollout/ep_rew_mean" not in ea.Tags().get("scalars", []):
            continue
        ev = ea.Scalars("rollout/ep_rew_mean")
        name = os.path.basename(d).rsplit("_", 1)[0]
        ax.plot([e.step / 1e6 for e in ev], [e.value for e in ev], label=f"{name}: {RUN_NOTES.get(name, '')}", lw=1.5)
    ax.set_xlabel("training steps [M]")
    ax.set_ylabel("mean episode reward (20 s episodes)")
    ax.set_title(f"{species}: locomotion training runs (reward definitions differ between runs)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--species", required=True)
    p.add_argument("--out", default="docs/figures/locomotion")
    args = p.parse_args()
    out = os.path.join(args.out, args.run)
    os.makedirs(out, exist_ok=True)

    learning_curves(args.species, os.path.join(args.out, f"learning_curves_{args.species}.png"))
    cks = checkpoints(args.run)
    print(f"{len(cks)} checkpoints: {[fmt_steps(n) for n, _ in cks]}")

    all_metrics, sheet, progress = [], [], []
    for n, path in cks:
        m, frames = evaluate(path, args.species)
        m["steps"] = n
        all_metrics.append(m)
        title = f"{args.species} - {fmt_steps(n)}"
        imgs = [label(f, [title, f"command {v:+.2f} m/s, {math.degrees(w):+.0f} deg/s  |  actual {a:+.2f} m/s"])
                for f, v, w, a in frames]
        imgs[0].save(os.path.join(out, f"{n:09d}_steps.gif"), save_all=True, append_images=imgs[1:],
                     duration=int(FRAME_EVERY * CONTROL_DT * 1000), loop=0)
        sheet.append(imgs[min(len(imgs) - 1, int(5.0 / (FRAME_EVERY * CONTROL_DT)))])
        progress += [im.resize((WIDTH // 2, HEIGHT // 2)) for im in imgs[::2]]
        print(f"{fmt_steps(n):>14}: speed err {m['speed_error']:.2f} m/s, turn err {m['turn_error_deg']:.0f} deg/s, "
              f"walk {m['walk_speed_at_walk_cmd']:+.2f} m/s, restart {m['restart_speed']:+.2f} m/s, "
              f"fell {m['fell_at_s'] or '-'}")

    progress[0].save(os.path.join(out, "progress.gif"), save_all=True, append_images=progress[1:],
                     duration=int(2 * FRAME_EVERY * CONTROL_DT * 1000), loop=0)
    cols = min(4, len(sheet))
    rows = math.ceil(len(sheet) / cols)
    grid = Image.new("RGB", (cols * WIDTH, rows * HEIGHT), "white")
    for i, im in enumerate(sheet):
        grid.paste(im, ((i % cols) * WIDTH, (i // cols) * HEIGHT))
    grid.save(os.path.join(out, "contact_sheet.png"))

    keys = ["steps", "speed_error", "turn_error_deg", "walk_speed_at_walk_cmd", "restart_speed", "fell_at_s",
            "foot_slip"]
    with open(os.path.join(out, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(all_metrics)
    x = [m["steps"] / 1e6 for m in all_metrics]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, [m["speed_error"] for m in all_metrics], "o-", label="speed error [m/s]")
    ax[0].plot(x, [m["walk_speed_at_walk_cmd"] for m in all_metrics], "s-", label="speed at walk command [m/s]")
    ax[0].plot(x, [m["restart_speed"] for m in all_metrics], "^-", label="speed when restarting from standstill")
    ax[0].axhline(LocomotionEnv(args.species).species.walk_speed, ls="--", c="gray", lw=1, label="walk command")
    ax[1].plot(x, [m["turn_error_deg"] for m in all_metrics], "o-", c="C2", label="turn error [deg/s]")
    for a in ax:
        a.set_xlabel("training steps [M]")
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(f"{args.species}: how well each checkpoint follows its commands")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "metrics.png"), dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
