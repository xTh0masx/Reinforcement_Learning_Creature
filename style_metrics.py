"""How natural does a walking brain move? Walks at walk speed and measures per leg joint type the range of motion,
the foot lift, steps per second, stride length, duty factor and body bob; for the DC also how close each hind paw
lands to the front paw's print ("register", a panther trait).

    python style_metrics.py models/myr_legs_v4 models/myr_legs_v5 --species myrranox
"""
import argparse
import json
import os

import numpy as np

from play_locomotion import Brain
from sim.locomotion_env import CONTROL_DT, LocomotionEnv

JOINT_LABELS = {"myrranox": ["hip swing", "leg lift", "knee"], "dc": ["abduction", "hip/shoulder", "knee/elbow", "ankle/wrist"]}


def measure(path, species, seconds=10.0, seed=0):
    brain = Brain(path, species)
    env = LocomotionEnv(species, episode_seconds=1e9, version=brain.version, passive=True)
    v = env.species.walk_speed
    obs, _ = env.reset(seed=seed, options={"command": (v, 0.0)})
    d = env.data
    n_leg_j = sum(len(leg.joints) for leg in env.body.legs)
    q, lift, contact, z, feet, speed = [], [], [], [], [], []
    for i in range(int(seconds / CONTROL_DT)):
        obs, _, _, _, info = env.step(brain(obs))
        if i * CONTROL_DT < 2.0:
            continue
        q.append(d.qpos[env.joint_qpos][:n_leg_j].copy())
        lift.append(d.site_xpos[env.foot_sites][:, 2] - env.foot_radius)
        contact.append(info["foot_contact"])
        feet.append(d.site_xpos[env.foot_sites][:, :2].copy())
        z.append(d.qpos[env.q0 + 2])
        speed.append(info["speed"])
    q, lift, c, feet = np.array(q), np.array(lift), np.array(contact), np.array(feet)
    nj = len(env.body.legs[0].joints)
    rom = np.degrees(np.ptp(q, axis=0)).reshape(len(env.body.legs), nj).mean(0)
    duration = len(c) * CONTROL_DT
    steps_per_s = float((c[1:] & ~c[:-1]).sum() / len(env.body.legs) / duration)
    out = {"brain": path, "version": brain.version,
           "range_of_motion_deg": dict(zip(JOINT_LABELS[species], np.round(rom, 1).tolist())),
           "foot_lift_cm": float(lift.max(0).mean() * 100), "steps_per_s": steps_per_s,
           "stride_m": float(np.mean(speed) / steps_per_s) if steps_per_s > 0 else 0.0,
           "duty": float(c.mean()), "body_bob_cm": float(np.ptp(z) * 100), "speed": float(np.mean(speed))}
    names = [leg.name for leg in env.body.legs]
    if species == "dc":                        # distance from each hind touchdown to the last front-paw print
        last, dists = {}, []
        for t in range(1, len(c)):
            for i, n in enumerate(names):
                if c[t, i] and not c[t - 1, i]:
                    if n.startswith("front"):
                        last[n] = feet[t, i]
                    elif n.replace("hind", "front") in last:
                        dists.append(np.linalg.norm(feet[t, i] - last[n.replace("hind", "front")]))
        out["register_cm"] = float(np.median(dists) * 100) if dists else None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--species", required=True)
    p.add_argument("--out", default="docs/evaluations")
    args = p.parse_args()
    rows = [measure(m, args.species) for m in args.models]
    for r in rows:
        extra = f", hind paw {r['register_cm']:.0f} cm from the front print" if r.get("register_cm") is not None else ""
        print(f"{os.path.basename(r['brain'])} (v{r['version']}): range of motion {r['range_of_motion_deg']} deg, "
              f"foot lift {r['foot_lift_cm']:.1f} cm, {r['steps_per_s']:.2f} steps/s, stride {r['stride_m']:.2f} m, "
              f"duty {r['duty']:.2f}, body bob {r['body_bob_cm']:.1f} cm, speed {r['speed']:.2f} m/s{extra}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"style_{args.species}.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
