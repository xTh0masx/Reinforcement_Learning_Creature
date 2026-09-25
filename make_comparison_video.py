"""Side-by-side video of two locomotion brains on the same command script (e.g. v1 vs v2).

    python make_comparison_video.py --species myrranox --left models/myr_legs_v1 --right models/myr_legs_v2
    -> docs/videos/myr_legs_v1_vs_v2.mp4   (1280x720, 25 fps)
"""
import argparse
import math
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from make_training_video import BG, DIM, FG, ACCENT, TITLE, font
from play_locomotion import Brain, command_script
from sim.locomotion_env import CONTROL_DT, LocomotionEnv

W, H, FPS = 1280, 720, 25
HALF = W // 2
TOP = 70


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--species", required=True)
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--labels", nargs=2, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    labels = args.labels or [os.path.basename(args.left), os.path.basename(args.right)]
    short = {"myrranox": "myr", "dc": "dc"}[args.species]
    out = args.out or f"docs/videos/{short}_legs_v1_vs_v2.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    sides = []
    for path in (args.left, args.right):
        brain = Brain(path, args.species)
        env = LocomotionEnv(args.species, episode_seconds=1e9, render_mode="rgb_array", width=HALF, height=H - TOP,
                            version=brain.version)
        env.model.vis.global_.offwidth, env.model.vis.global_.offheight = W, H
        obs, _ = env.reset(seed=0, options={"command": (0.0, 0.0)})
        sides.append({"brain": brain, "env": env, "obs": obs, "speed": 0.0, "turn": 0.0, "power": 0.0, "fell": 0})

    writer = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    steps_per_frame = int(round(1 / (FPS * CONTROL_DT)))
    n = 0
    for seconds, v, w in command_script(args.species):
        for s in sides:
            s["env"].set_command(v, w)
        for _ in range(int(round(seconds / CONTROL_DT))):
            for s in sides:
                s["obs"], _, fell, _, info = s["env"].step(s["brain"](s["obs"]))
                s["speed"], s["turn"] = info["speed"], info["turn_rate"]
                s["power"] += (info["power"] - s["power"]) * CONTROL_DT / 1.0     # 1 s running average
                if fell:
                    s["fell"] += 1
                    s["obs"], _ = s["env"].reset(seed=n, options={"command": (v, w)})
            n += 1
            if n % steps_per_frame:
                continue
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)
            d.text((20, 8), f"{TITLE[args.species]} - same commands, two brains", fill=FG, font=font(24, True))
            d.text((20, 40), f"command  {v:+.2f} m/s   {math.degrees(w):+.0f} deg/s", fill=ACCENT, font=font(20))
            for i, s in enumerate(sides):
                img.paste(Image.fromarray(s["env"].render()), (i * HALF, TOP))
                x = i * HALF + 16
                d.rectangle([x - 8, TOP + 8, x + 330, TOP + 92], fill=BG)
                d.text((x, TOP + 12), labels[i], fill=FG, font=font(22, True))
                d.text((x, TOP + 42), f"{s['speed']:+.2f} m/s  {math.degrees(s['turn']):+4.0f} deg/s", fill=FG, font=font(18))
                d.text((x, TOP + 66), f"power {s['power']:5.0f} W (1 s avg)" + (f"   falls {s['fell']}" if s["fell"] else ""),
                       fill=DIM, font=font(16))
            d.line([HALF, TOP, HALF, H], fill=(60, 63, 70), width=3)
            writer.append_data(np.asarray(img))
    writer.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
