"""Record creatures driven by both brains: navigation (where to go) + locomotion (how to move the legs).

Every 0.1 s the navigation brain in CreatureEnv picks a speed/turn command; the locomotion brain
walks the MuJoCo body for 5 control steps (50 Hz) to follow it; the body's real position and heading
go back into CreatureEnv (prey, needs, territory). Output:

    recordings/<name>.csv              10 Hz, same columns as record.py (creatures + prey)
    recordings/<name>_<id>_pose.csv    50 Hz body pose: time, x, y, z, qw, qx, qy, qz, <joints...>, <feet xyz...>

    python record_physics.py --species myrranox dc --loco models/myr_legs_v1 models/dc_legs_v1 --seconds 120
    python record_physics.py --nav models/myr_brain_v1 models/dc_brain_v1 ...   # trained navigation brains
"""
import argparse
import csv
import math
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from play_locomotion import Brain as LocoBrain
from record import HeuristicPolicy
from sim.creature_env import DT, CreatureEnv
from sim.locomotion_env import CONTROL_DT, LocomotionEnv

SUBSTEPS = int(round(DT / CONTROL_DT))


class NavBrain:
    """Trained navigation brain with its 3-frame stack and observation normalisation."""

    def __init__(self, path, n_stack=3):
        import pickle
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")
        with open(f"{path}_vecnormalize.pkl", "rb") as f:
            self.norm = pickle.load(f)
        self.frames = deque(maxlen=n_stack)

    def __call__(self, obs):
        if not self.frames:
            self.frames.extend([np.zeros_like(obs)] * (self.frames.maxlen - 1))
        self.frames.append(obs)
        stacked = np.concatenate(self.frames)
        o = np.clip((stacked - self.norm.obs_rms.mean) / np.sqrt(self.norm.obs_rms.var + self.norm.epsilon),
                    -self.norm.clip_obs, self.norm.clip_obs)
        return self.model.predict(o[None].astype(np.float32), deterministic=True)[0][0]


class SineGait:
    """Open-loop test gait (not RL): every joint oscillates, legs alternate. Use --loco sine to check
    the recording and Blender playback without a trained locomotion brain."""

    def __init__(self, env, hz=1.0, amp=0.5):
        self.env, self.hz, self.amp, self.t = env, hz, amp, 0.0
        self.phase = np.array([(i % 2) * math.pi + (i // 2) * math.pi for i, leg in enumerate(env.body.legs)
                               for _ in leg.joints])
        self.offset = np.array([k * math.pi / 2 for leg in env.body.legs for k in range(len(leg.joints))])

    def __call__(self, obs):
        self.t += CONTROL_DT
        return self.amp * np.sin(2 * math.pi * self.hz * self.t + self.phase + self.offset)


def _yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def record_creature(species, cid, loco_path, nav_path, seconds, seed, nav_writer, pose_path):
    nav = CreatureEnv(species=species)
    nav_obs, _ = nav.reset(seed=seed)
    nav_policy = NavBrain(nav_path) if nav_path else HeuristicPolicy(nav, seed)

    loco = LocomotionEnv(species, episode_seconds=1e9)
    lobs, _ = loco.reset(seed=seed, options={"command": (0.0, 0.0), "yaw": nav.heading})
    loco_brain = SineGait(loco) if loco_path == "sine" else LocoBrain(loco_path, species)
    offset = nav.pos - loco.data.qpos[:2]

    joint_cols = loco.body.joint_names
    foot_cols = [f"{f}_{a}" for f in loco.body.foot_names for a in "xyz"]
    falls = 0
    with open(pose_path, "w", newline="") as pf:
        pw = csv.writer(pf)
        pw.writerow(["time", "x", "y", "z", "qw", "qx", "qy", "qz"] + joint_cols + foot_cols)

        def write_pose(t):
            p = loco.pose()
            feet = p["feet"].copy()
            feet[:, :2] += offset
            pw.writerow([f"{t:.3f}", *(f"{v:.5f}" for v in (p["pos"][0] + offset[0], p["pos"][1] + offset[1], p["pos"][2])),
                         *(f"{v:.6f}" for v in p["quat"]), *(f"{v:.5f}" for v in p["joints"]),
                         *(f"{v:.5f}" for v in feet.ravel())])

        write_pose(0.0)
        n_steps = int(round(seconds / DT))
        for step in range(n_steps + 1):
            t = step * DT
            nav_writer.writerow([round(t, 3), cid, species, f"{nav.pos[0]:.4f}", f"{nav.pos[1]:.4f}",
                                 f"{nav.heading:.5f}", f"{nav.speed:.4f}", f"{nav.turn_rate:.5f}", nav.activity])
            for j, (pos, h, alive) in enumerate(zip(nav.prey_pos, nav.prey_heading, nav.prey_alive)):
                nav_writer.writerow([round(t, 3), f"{cid}_prey_{j}", "prey", f"{pos[0]:.4f}", f"{pos[1]:.4f}",
                                     f"{h:.5f}", "", "", "walk" if alive else "hidden"])
            if step == n_steps:
                break

            action = nav_policy(nav_obs)
            v, w, _ = nav.command(action)
            loco.set_command(v, w)
            for k in range(SUBSTEPS):
                lobs, _, fell, _, linfo = loco.step(loco_brain(lobs))
                write_pose(t + (k + 1) * CONTROL_DT)
                if fell:
                    falls += 1
                    lobs, _ = loco.reset(seed=seed + falls, options={"command": (v, w), "yaw": nav.heading})
                    offset = nav.pos - loco.data.qpos[:2]
            body = {"pos": loco.data.qpos[:2] + offset, "heading": _yaw(loco.data.qpos[3:7]),
                    "speed": linfo["speed"], "turn_rate": linfo["turn_rate"]}
            nav_obs, _, terminated, truncated, info = nav.step(action, body=body)
            if info["hit"]:
                # Rocks and walls exist only in the habitat: push the physics body back.
                loco.data.qpos[:2] = nav.pos - offset
                loco.data.qvel[:2] = 0.0
                mujoco.mj_forward(loco.model, loco.data)
            if terminated:
                break
    s = nav.stats
    print(f"{cid}: {nav.t * DT:.0f} s, prey {s['prey_caught']}, hits {s['hits']}, falls {falls}, "
          f"home {100 * s['steps_in_home'] / max(nav.t, 1):.0f} %, reward {s['reward']:.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--species", nargs="+", default=["myrranox", "dc"])
    p.add_argument("--loco", nargs="+", required=True, help="locomotion brain per species (models/<run>)")
    p.add_argument("--nav", nargs="+", default=None, help="navigation brain per species; default: heuristic")
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="recordings/physics_preview.csv")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "id", "species", "x", "y", "heading", "speed", "turn_rate", "activity"])
        for k, species in enumerate(args.species):
            cid = f"{species}_{counts.get(species, 0)}"
            counts[species] = counts.get(species, 0) + 1
            nav_path = args.nav[k] if args.nav else None
            record_creature(species, cid, args.loco[k], nav_path, args.seconds, args.seed + k, w,
                            out.with_name(f"{out.stem}_{cid}_pose.csv"))
    print(f"wrote {out} (+ pose files)")


if __name__ == "__main__":
    main()
