"""Analyse the gait of a locomotion brain: footfall diagram + per-leg statistics.

    python gait_analysis.py models/myr_legs_v1 --species myrranox
    -> docs/figures/gait/<name>_footfall.png and printed per-leg table

Footfall diagram: one row per leg, dark = foot on the ground (stance). A clean ant/insect tripod shows two
alternating groups of three legs (L1 R2 L3 / R1 L2 R3); a quadruped walk shows the sequence LH LF RH RF,
a trot diagonal pairs (LF+RH / RF+LH).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from play_locomotion import Brain
from sim.locomotion_env import CONTROL_DT, LocomotionEnv


def record(path, species, speed, turn=0.0, seconds=8.0, seed=0):
    env = LocomotionEnv(species, episode_seconds=1e9)
    brain = Brain(path, species) if path else None
    obs, _ = env.reset(seed=seed, options={"command": (speed, turn)})
    contacts, joints, slips, vs = [], [], [], []
    for _ in range(int(seconds / CONTROL_DT)):
        a = brain(obs) if brain else np.zeros(env.nj)
        obs, _, fell, _, info = env.step(a)
        contacts.append(info["foot_contact"].copy())
        joints.append(env.data.qpos[env.joint_qpos].copy())
        vs.append(info["speed"])
        if fell:
            break
    return env, np.array(contacts), np.array(joints), np.array(vs)


def leg_stats(contact):
    """Duty factor, steps per second and mean stance phase for one leg's contact series."""
    c = contact.astype(int)
    touchdowns = np.flatnonzero(np.diff(c) == 1)
    duration = len(c) * CONTROL_DT
    return c.mean(), len(touchdowns) / duration, touchdowns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--species", required=True)
    p.add_argument("--speed", type=float, default=None, help="default: the species' walk speed")
    p.add_argument("--turn", type=float, default=0.0)
    p.add_argument("--out", default="docs/figures/gait")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env0 = LocomotionEnv(args.species)
    speed = env0.species.walk_speed if args.speed is None else args.speed
    env, contact, joints, vs = record(args.model, args.species, speed, args.turn)
    names = env.body.foot_names
    skip = int(2.0 / CONTROL_DT)                      # ignore the first 2 s (start-up)
    c = contact[skip:]
    j = joints[skip:]

    print(f"{args.model} @ {speed:.2f} m/s, turn {np.degrees(args.turn):.0f} deg/s -> mean speed "
          f"{vs[skip:].mean():.2f} m/s")
    print(f"{'leg':<12}{'duty':>6}{'steps/s':>9}{'joint range (deg)':>28}")
    per_leg = {}
    k = 0
    for leg, fname in zip(env.body.legs, names):
        nj = len(leg.joints)
        duty, freq, _ = leg_stats(c[:, names.index(fname)])
        rng = np.degrees(j[:, k:k + nj].max(0) - j[:, k:k + nj].min(0))
        per_leg[leg.name] = (duty, freq, rng)
        print(f"{leg.name:<12}{duty:>6.2f}{freq:>9.2f}{'  ' + ' '.join(f'{r:5.0f}' for r in rng):>28}")
        k += nj
    # Left/right symmetry: compare joint ranges of mirrored legs
    asym = []
    for name, (_, _, rng) in per_leg.items():
        if name.endswith("_L"):
            other = per_leg.get(name[:-2] + "_R")
            if other is not None:
                asym.append(np.abs(rng - other[2]).mean())
    print(f"left/right asymmetry (mean |range_L - range_R|): {np.mean(asym):.1f} deg")
    # Correlation of contact between legs: +1 = move together, -1 = alternate
    corr = np.corrcoef(c.T.astype(float) + 1e-9 * np.random.default_rng(0).standard_normal(c.T.shape))
    print("contact correlation matrix (rows/cols: " + ", ".join(n.replace('_foot', '') for n in names) + ")")
    print(np.round(corr, 2))

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(names) + 1.2))
    t = np.arange(len(contact)) * CONTROL_DT
    for i, n in enumerate(names):
        on = contact[:, i]
        ax.fill_between(t, i - 0.4, i + 0.4, where=on, color="#333", step="mid")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_foot", "") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("time [s]  (dark = foot on the ground)")
    name = os.path.basename(args.model.rstrip("/")) if args.model else "zero_action"
    ax.set_title(f"{args.species} footfall - {name} @ {speed:.2f} m/s")
    fig.tight_layout()
    out = os.path.join(args.out, f"{name}_footfall_{speed:.2f}.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
