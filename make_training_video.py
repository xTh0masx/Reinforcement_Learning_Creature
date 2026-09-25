"""Make an MP4 of a whole locomotion training run: every checkpoint from the untrained brain to the end,
each doing the same 12 s test (walk, full speed, walk + turn), next to the learning curve.

    python make_training_video.py --run myr_legs_v1 --species myrranox
    -> docs/videos/myr_legs_v1_training.mp4   (1280x720, 25 fps)
"""
import argparse
import glob
import io
import math
import os

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from locomotion_progress import checkpoints, fmt_steps
from play_locomotion import Brain, command_script
from sim.locomotion_env import CONTROL_DT, LocomotionEnv

W, H = 1280, 720
VIEW_W = 860                     # MuJoCo view on the left, info panel on the right
PANEL_W = W - VIEW_W
FPS = 25
STEPS_PER_FRAME = int(round(1 / (FPS * CONTROL_DT)))   # 2 control steps per frame
CLIP_SECONDS = 12.0
TITLE = {"myrranox": "Myrranox", "dc": "DC_SKIN_RAW"}
BG, FG, DIM, ACCENT, BAD = (24, 26, 30), (235, 235, 235), (150, 155, 165), (255, 170, 60), (235, 80, 70)


def font(size, bold=False):
    for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def reward_curve(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    dirs = sorted(glob.glob(f"runs/{run}_*"))
    xs, ys = [], []
    for d in dirs:
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        if "rollout/ep_rew_mean" in ea.Tags().get("scalars", []):
            ev = ea.Scalars("rollout/ep_rew_mean")
            xs += [e.step for e in ev]
            ys += [e.value for e in ev]
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]


def curve_image(xs, ys, marker_steps, total):
    fig, ax = plt.subplots(figsize=(PANEL_W / 100, 2.8), dpi=100)
    fig.patch.set_facecolor(np.array(BG) / 255)
    ax.set_facecolor(np.array(BG) / 255)
    ax.plot(xs / 1e6, ys, color=np.array(DIM) / 255, lw=1.2)
    done = xs <= marker_steps
    ax.plot(xs[done] / 1e6, ys[done], color=np.array(ACCENT) / 255, lw=2)
    ax.axvline(marker_steps / 1e6, color=np.array(ACCENT) / 255, lw=1, ls="--")
    ax.set_xlim(0, total / 1e6)
    ax.set_xlabel("training steps [M]", color="white", fontsize=9)
    ax.set_ylabel("reward / episode", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#666")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def card(lines, sizes):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = H // 2 - sum(s + 18 for s in sizes) // 2
    for text, size in zip(lines, sizes):
        f = font(size, bold=size >= 40)
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, fill=FG if size >= 40 else DIM, font=f)
        y += size + 18
    return np.asarray(img)


def compose(view, panel_bg, species, steps, idx, n, v_cmd, w_cmd, v, w, fell, t_in_clip):
    img = Image.new("RGB", (W, H), BG)
    img.paste(Image.fromarray(view), (0, 0))
    img.paste(panel_bg, (VIEW_W, 0))
    d = ImageDraw.Draw(img)
    x0 = VIEW_W + 24
    d.text((x0, 20), f"{TITLE[species]} - learning to walk", fill=FG, font=font(24, True))
    d.text((x0, 58), f"checkpoint {idx + 1} / {n}", fill=DIM, font=font(18))
    d.text((x0, 86), fmt_steps(steps), fill=ACCENT, font=font(46, True))
    # progress bar over all checkpoints
    d.rectangle([x0, 150, W - 24, 158], fill=(60, 63, 70))
    d.rectangle([x0, 150, x0 + (W - 24 - x0) * (idx + t_in_clip / CLIP_SECONDS) / n, 158], fill=ACCENT)

    y = 480
    d.text((x0, y), "command", fill=DIM, font=font(18))
    d.text((x0 + 150, y), f"{v_cmd:+.2f} m/s   {math.degrees(w_cmd):+4.0f} deg/s", fill=FG, font=font(20))
    d.text((x0, y + 34), "actual", fill=DIM, font=font(18))
    d.text((x0 + 150, y + 34), f"{v:+.2f} m/s   {math.degrees(w):+4.0f} deg/s", fill=FG, font=font(20))
    # speed gauge
    gx0, gx1, gy = x0, W - 24, y + 90
    d.text((x0, gy - 26), "speed (bar = actual, line = command)", fill=DIM, font=font(16))
    d.rectangle([gx0, gy, gx1, gy + 14], fill=(60, 63, 70))
    scale = (gx1 - gx0) / (2.8 if species == "dc" else 1.5)
    d.rectangle([gx0, gy, gx0 + max(0.0, v) * scale, gy + 14], fill=(90, 180, 120))
    cx = gx0 + max(0.0, v_cmd) * scale
    d.line([cx, gy - 6, cx, gy + 20], fill=ACCENT, width=3)
    if fell:
        d.text((x0, y + 140), "FELL", fill=BAD, font=font(36, True))
    return np.asarray(img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--species", required=True)
    p.add_argument("--out", default="docs/videos")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.run}_training.mp4")

    cks = checkpoints(args.run)
    xs, ys = reward_curve(args.run)
    total = max([c[0] for c in cks] + [int(xs.max()) if len(xs) else 1])
    script = command_script(args.species)
    clip_script, t = [], 0.0
    for seconds, v, w in script:
        if t >= CLIP_SECONDS:
            break
        clip_script.append((min(seconds, CLIP_SECONDS - t), v, w))
        t += seconds

    writer = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    title = card([f"{TITLE[args.species]} learns to walk",
                  "Reinforcement learning (PPO) in a MuJoCo physics body",
                  f"run {args.run}  -  {len(cks)} checkpoints from untrained to {fmt_steps(total)}",
                  "each checkpoint: walk 4 s, full speed 4 s, walk + turn 4 s"], [52, 26, 22, 22])
    for _ in range(FPS * 4):
        writer.append_data(title)

    for idx, (steps, path) in enumerate(cks):
        print(f"{fmt_steps(steps)} ...", flush=True)
        panel = curve_image(xs, ys, steps, total)
        panel_bg = Image.new("RGB", (PANEL_W, H), BG)
        panel_bg.paste(panel, (0, 180))
        env = LocomotionEnv(args.species, episode_seconds=1e9, render_mode="rgb_array", width=VIEW_W, height=H)
        env.model.vis.global_.offwidth, env.model.vis.global_.offheight = W, H
        brain = Brain(path, args.species)
        obs, _ = env.reset(seed=0, options={"command": (0.0, 0.0)})
        t_clip, fell_timer, n = 0.0, 0.0, 0
        info = {"speed": 0.0, "turn_rate": 0.0}
        for seconds, v_cmd, w_cmd in clip_script:
            env.set_command(v_cmd, w_cmd)
            for _ in range(int(round(seconds / CONTROL_DT))):
                obs, _, fell, _, info = env.step(brain(obs))
                t_clip += CONTROL_DT
                n += 1
                if fell:
                    fell_timer = 1.0
                    obs, _ = env.reset(seed=n, options={"command": (v_cmd, w_cmd)})
                fell_timer = max(0.0, fell_timer - CONTROL_DT)
                if n % STEPS_PER_FRAME == 0:
                    writer.append_data(compose(env.render(), panel_bg, args.species, steps, idx, len(cks),
                                               v_cmd, w_cmd, info["speed"], info["turn_rate"],
                                               fell_timer > 0, t_clip))
        env.close()

    end_lines = [f"{TITLE[args.species]} - {fmt_steps(total)}", "same brain from start to finish of training,",
                 "only the network weights change between checkpoints"]
    end = card(end_lines, [48, 24, 24])
    for _ in range(FPS * 3):
        writer.append_data(end)
    writer.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
