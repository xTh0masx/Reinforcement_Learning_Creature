"""How precisely does a walking brain follow its commands? Runs the fixed command script (play_locomotion.py) and
reports per segment the achieved speed / turn rate against the command, plus foot slip and falls.

    python precision_metrics.py models/dc_legs_v3 models/dc_legs_v4 --species dc
"""
import argparse
import json
import math
import os

import numpy as np

from play_locomotion import Brain, command_script
from sim.locomotion_env import CONTROL_DT, LocomotionEnv


def measure(path, species, seed=0):
    brain = Brain(path, species)
    env = LocomotionEnv(species, episode_seconds=1e9, version=brain.version, passive=True)
    obs, _ = env.reset(seed=seed, options={"command": (0.0, 0.0)})
    segments, falls = [], 0
    for seconds, v, w in command_script(species):
        env.set_command(v, w)
        rows = []
        for i in range(int(seconds / CONTROL_DT)):
            obs, _, fell, _, info = env.step(brain(obs))
            if i * CONTROL_DT >= 1.0:                       # skip the first second (acceleration)
                rows.append((info["speed"], info["turn_rate"], info["terms"]["foot_slip"]))
            if fell:
                falls += 1
                obs, _ = env.reset(seed=seed + falls, options={"command": (v, w)})
        r = np.array(rows)
        segments.append({"cmd_speed": v, "cmd_turn_deg": math.degrees(w),
                         "speed": float(r[:, 0].mean()), "turn_deg": math.degrees(float(r[:, 1].mean())),
                         "speed_err": float(np.abs(r[:, 0] - v).mean()),
                         "turn_err_deg": math.degrees(float(np.abs(r[:, 1] - w).mean())),
                         "slip": float(r[:, 2].mean())})
    return {"brain": path, "version": brain.version, "falls": falls, "segments": segments,
            "mean_speed_err": float(np.mean([s["speed_err"] for s in segments])),
            "mean_turn_err_deg": float(np.mean([s["turn_err_deg"] for s in segments]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--species", required=True)
    p.add_argument("--out", default="docs/evaluations")
    args = p.parse_args()
    rows = [measure(m, args.species) for m in args.models]
    for r in rows:
        print(f"{os.path.basename(r['brain'])} (v{r['version']}): mean speed error {r['mean_speed_err']:.3f} m/s, "
              f"mean turn error {r['mean_turn_err_deg']:.1f} deg/s, falls {r['falls']}")
        for s in r["segments"]:
            print(f"   cmd {s['cmd_speed']:+.2f} m/s {s['cmd_turn_deg']:+5.0f} deg/s -> {s['speed']:+.2f} m/s "
                  f"{s['turn_deg']:+5.0f} deg/s   slip {s['slip']:.3f}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"precision_{args.species}.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
