"""Measure how much mechanical power a locomotion brain uses at different speeds / turn rates and fit
P(v, w) = p0 + p1 |v| + p2 v^2 + p3 |w|  (W). Written to sim/energy.json and used by CreatureEnv to make
hunger and fatigue depend on real effort when no physics body is simulated (2D training).

    python calibrate_energy.py --loco models/myr_legs_v2 models/dc_legs_v2
"""
import argparse
import json
import os

import numpy as np

from play_locomotion import Brain
from sim.locomotion_env import CONTROL_DT, LocomotionEnv
from sim.species import SPECIES

OUT = os.path.join("sim", "energy.json")


def measure(path, species, seconds=4.0):
    sp = SPECIES[species]
    brain = Brain(path, species)
    env = LocomotionEnv(species, episode_seconds=1e9, version=brain.version)
    speeds = np.linspace(0.0, sp.max_speed, 6)
    turns = [0.0, 0.5 * sp.max_turn_rate]
    rows = []
    for v in speeds:
        for w in turns:
            if not sp.turn_in_place and v < sp.min_turn_speed and w:
                continue
            obs, _ = env.reset(seed=0, options={"command": (v, w)})
            p, n = [], int(seconds / CONTROL_DT)
            for k in range(n):
                obs, _, fell, _, info = env.step(brain(obs))
                if k > n // 4:                                # skip the start-up
                    p.append(info["power"])
                if fell:
                    break
            rows.append((abs(info["speed"]), abs(info["turn_rate"]), float(np.mean(p))))
            print(f"  {species} v={v:.2f} w={np.degrees(w):4.0f} deg/s -> {np.mean(p):7.1f} W")
    r = np.array(rows)
    A = np.column_stack([np.ones(len(r)), r[:, 0], r[:, 0] ** 2, r[:, 1]])
    coeffs, *_ = np.linalg.lstsq(A, r[:, 2], rcond=None)
    coeffs = np.maximum(coeffs, 0.0)
    return {"brain": path, "coeffs": [float(c) for c in coeffs],
            "p_max": float(coeffs @ [1, sp.max_speed, sp.max_speed ** 2, 0])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loco", nargs=2, required=True, help="locomotion brains for myrranox and dc")
    args = p.parse_args()
    data = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for species, path in zip(("myrranox", "dc"), args.loco):
        data[species] = measure(path, species)
        print(f"{species}: P(v,w) = {data[species]['coeffs']}, P(max speed) = {data[species]['p_max']:.0f} W")
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
