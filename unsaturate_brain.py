"""Bring joints whose command is stuck beyond the action limit back into range, so they can learn again.

A policy output beyond +-1 is clipped: every sample then gives the same joint target, the reward cannot tell them
apart and the joint stops learning. This happened to the Myrranox knees between 1M and 2M steps of walking v2
(mean output about -1.4, so the knees stayed locked straight in v2-v4). The fix shifts the output bias of each
stuck joint so its mean output sits just inside the range (the pose barely changes) and widens its exploration.

    python unsaturate_brain.py models/myr_legs_v5_pilot --species myrranox --out models/myr_legs_v5_start
"""
import argparse
import shutil

import numpy as np
import torch

from play_locomotion import Brain, command_script, vecnormalize_path
from sim.locomotion_env import CONTROL_DT, LocomotionEnv


def mean_outputs(brain, species, seconds=4.0):
    """Mean raw policy output per joint over the command script's first segments (walk, full speed, turns)."""
    env = LocomotionEnv(species, version=brain.version, episode_seconds=1e9, passive=True)
    obs, _ = env.reset(seed=0, options={"command": (0.0, 0.0)})
    pol = brain.model.policy
    out = []
    for _, v, w in command_script(species)[:4]:
        env.set_command(v, w)
        for _ in range(int(seconds / CONTROL_DT)):
            o = pol.obs_to_tensor(brain.norm.normalize_obs(obs[None]))[0]
            with torch.no_grad():
                mu = pol.get_distribution(o).distribution.mean.numpy()[0]
            out.append(mu)
            obs, *_ = env.step(np.clip(mu, -1, 1))
    return np.array(out).mean(0), env.body.joint_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--species", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=float, default=1.0, help="joints with |mean output| above this are stuck")
    p.add_argument("--target", type=float, default=0.8, help="new |mean output| of a stuck joint")
    p.add_argument("--std", type=float, default=0.35, help="exploration noise of a stuck joint")
    args = p.parse_args()

    brain = Brain(args.model, args.species)
    mean, names = mean_outputs(brain, args.species)
    stuck = np.flatnonzero(np.abs(mean) > args.limit)
    pol = brain.model.policy
    with torch.no_grad():
        for j in stuck:
            shift = np.sign(mean[j]) * args.target - mean[j]
            pol.action_net.bias[j] += float(shift)
            pol.log_std[j] = max(float(pol.log_std[j]), float(np.log(args.std)))
            print(f"{names[j]}: mean output {mean[j]:+.2f} -> {mean[j] + shift:+.2f}")
    if not len(stuck):
        print("no stuck joints")
    brain.model.save(args.out)
    shutil.copy(vecnormalize_path(args.model), vecnormalize_path(args.out))
    after, _ = mean_outputs(Brain(args.out, args.species), args.species)
    print("after (mean output of the fixed joints):", np.round(after[stuck], 2), f"-> saved {args.out}")


if __name__ == "__main__":
    main()
