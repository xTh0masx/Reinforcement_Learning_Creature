"""Evaluate a locomotion brain on a fixed command script and save a GIF.

    python play_locomotion.py models/myr_legs_v1 --species myrranox --gif docs/figures/myr_legs_v1.gif
"""
import argparse
import math
import re

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.locomotion_env import CONTROL_DT, LocomotionEnv
from sim.species import SPECIES


def command_script(species):
    """(seconds, speed m/s, turn rate rad/s) segments covering the species' limits."""
    sp = SPECIES[species]
    s = [(4, sp.walk_speed, 0.0), (4, sp.max_speed, 0.0),
         (4, sp.walk_speed, sp.max_turn_rate * 0.6), (4, sp.walk_speed, -sp.max_turn_rate * 0.6)]
    if sp.turn_in_place:
        s.append((3, 0.0, sp.max_turn_rate * 0.8))
    if sp.max_backward_speed > 0:
        s.append((3, -sp.max_backward_speed, 0.0))
    s += [(3, 0.0, 0.0), (4, sp.walk_speed, 0.0)]     # stop, then start walking again from standstill
    return s


def vecnormalize_path(path):
    """models/<run> -> models/<run>_vecnormalize.pkl; checkpoints/<run>_<N>_steps -> <run>_vecnormalize_<N>_steps.pkl"""
    path = str(path).removesuffix(".zip")
    m = re.match(r"(.*)_(\d+)_steps$", path)
    return f"{m.group(1)}_vecnormalize_{m.group(2)}_steps.pkl" if m else f"{path}_vecnormalize.pkl"


class Brain:
    """Trained locomotion policy with its observation normalisation, usable step by step."""

    def __init__(self, path, species):
        self.model = PPO.load(path, device="cpu")
        venv = DummyVecEnv([lambda: LocomotionEnv(species)])
        self.norm = VecNormalize.load(vecnormalize_path(path), venv)
        self.norm.training = False

    def __call__(self, obs):
        return self.model.predict(self.norm.normalize_obs(obs[None]), deterministic=True)[0][0]


def run(path, species, gif=None, seed=0):
    env = LocomotionEnv(species, episode_seconds=1e6, render_mode="rgb_array" if gif else None)
    brain = Brain(path, species)
    obs, _ = env.reset(seed=seed, options={"command": (0.0, 0.0)})
    frames, rows = [], []
    fell = False
    for seconds, v, w in command_script(species):
        env.set_command(v, w)
        for i in range(int(seconds / CONTROL_DT)):
            obs, r, term, trunc, info = env.step(brain(obs))
            rows.append((v, w, info["speed"], info["turn_rate"], info["terms"]["foot_slip"]))
            if gif and i % 3 == 0:
                frames.append(env.render())
            if term:
                fell = True
                break
        if fell:
            break
        # Report per segment, skipping the first second (acceleration)
        seg = np.array(rows[-int(seconds / CONTROL_DT):][int(1 / CONTROL_DT):])
        print(f"cmd v={v:+.2f} m/s w={math.degrees(w):+5.0f} deg/s -> v={seg[:, 2].mean():+.2f}+-{seg[:, 2].std():.2f}"
              f"  w={math.degrees(seg[:, 3].mean()):+5.0f} deg/s  slip {seg[:, 4].mean():.3f}")
    print("FELL" if fell else "stayed upright")
    if gif:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(3 * CONTROL_DT * 1000), loop=0)
        print(f"wrote {gif}")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--species", default="myrranox")
    p.add_argument("--gif", default=None)
    args = p.parse_args()
    run(args.model, args.species, args.gif)
