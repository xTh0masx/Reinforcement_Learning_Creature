"""Record both predators living in the habitat, everything learned: predator brains decide where to go, walking
brains move the legs in one MuJoCo world built from habitat.json (solid rocks and walls).

    python record_world.py --minutes 5 --out recordings/showcase.csv
    python record_world.py --pick-seed 8 ...   # try 8 seeds in the fast 2D sim, record the one where both survive longest

Output (read by blender/scripts/import_trajectories.py):
    <out>.csv              10 Hz: time, id, species, x, y, heading, speed, turn_rate, activity, hunger, fatigue, (prey rows)
    <out>_<id>_pose.csv    50 Hz body pose + joint angles + feet per creature
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from sim.creature_env import CreatureEnv
from sim.habitat import Habitat
from sim.nav_brain import NavBrain
from sim.physics_world import PhysicsWorld

PREDATORS = {"myrranox": "models/myr_brain_v3_r1", "dc": "models/dc_brain_v4_r1"}
WALKERS = {"myrranox": "models/myr_legs_v2", "dc": "models/dc_legs_v2"}


def make_env(ego, other, seed):
    ego_brain = NavBrain(PREDATORS[ego])
    obs_version = 1 if ego_brain.model.observation_space.shape[0] == 81 else 2
    env = CreatureEnv(ego, others=[{"species": other, "brain": PREDATORS[other]}], obs_version=obs_version)
    obs, _ = env.reset(seed=seed)
    return env, ego_brain, obs


def survival_2d(ego, other, seed, steps):
    env, brain, obs = make_env(ego, other, seed)
    for t in range(steps):
        obs, _, term, _, _ = env.step(brain(obs))
        if term:
            return t
    return steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pick-seed", type=int, default=0, help="try this many seeds in 2D first, keep the best")
    p.add_argument("--out", default="recordings/showcase.csv")
    args = p.parse_args()
    steps = int(args.minutes * 600)
    ego, other = "myrranox", "dc"

    seed = args.seed
    if args.pick_seed:
        scores = {s: survival_2d(ego, other, s, steps) for s in range(args.pick_seed)}
        seed = max(scores, key=scores.get)
        print("2D survival per seed (steps):", scores, "-> seed", seed)

    env, ego_brain, obs = make_env(ego, other, seed)
    creatures = env.creatures
    world = PhysicsWorld([{"id": c.id, "species": c.species.name, "loco": WALKERS[c.species.name],
                           "pos": tuple(c.pos), "yaw": c.heading} for c in creatures], Habitat.load(), seed=seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pose_files, pose_writers = {}, {}
    for c in creatures:
        body = world.bodies[c.id]
        f = open(out.with_name(f"{out.stem}_{c.id}_pose.csv"), "w", newline="")
        w = csv.writer(f)
        w.writerow(["time", "x", "y", "z", "qw", "qx", "qy", "qz"] + body.body.joint_names +
                   [f"{n}_{a}" for n in body.body.foot_names for a in "xyz"])
        pose_files[c.id], pose_writers[c.id] = f, w

    def write_pose(cid, t, pz):
        pose_writers[cid].writerow([f"{t:.3f}", *(f"{v:.5f}" for v in pz["pos"]), *(f"{v:.6f}" for v in pz["quat"]),
                                    *(f"{v:.5f}" for v in pz["joints"]), *(f"{v:.5f}" for v in pz["feet"].ravel())])

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "id", "species", "x", "y", "heading", "speed", "turn_rate", "activity", "hunger", "fatigue"])
        for c in creatures:
            write_pose(c.id, 0.0, world.pose(c.id))
        for step in range(steps + 1):
            t = step * 0.1
            for c in creatures:
                w.writerow([round(t, 3), c.id, c.species.name, f"{c.pos[0]:.4f}", f"{c.pos[1]:.4f}", f"{c.heading:.5f}",
                            f"{c.speed:.4f}", f"{c.turn_rate:.5f}", c.activity, f"{c.hunger:.3f}", f"{c.fatigue:.3f}"])
            for j, (pos, h, alive) in enumerate(zip(env.prey_pos, env.prey_heading, env.prey_alive)):
                w.writerow([round(t, 3), f"prey_{j}", "prey", f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{h:.5f}",
                            "", "", "walk" if alive else "hidden", "", ""])
            if step == steps:
                break
            ego_action = ego_brain(obs)
            others = env.other_actions()
            for c in creatures:
                v, wr, _ = env.command(ego_action if c is env.ego else others[c.id], c)
                world.set_command(c.id, v, wr)
            state = world.step(record_poses=True)
            for c in creatures:
                for k, pz in enumerate(state[c.id]["poses"]):
                    write_pose(c.id, t + (k + 1) * 0.02, pz)
            obs, _, term, _, _ = env.step(ego_action, bodies=state, actions=others)
            if step % 600 == 0:
                print(f"  {t / 60:4.1f} min: " + ", ".join(
                    f"{c.id} hunger {c.hunger:.2f} prey {c.stats['prey_caught']}" for c in creatures), flush=True)
            if term:
                print(f"{env.ego.id} starved at {t:.0f} s - recording ends there")
                break
    for f in pose_files.values():
        f.close()
    print("done:", {c.id: dict(prey=c.stats["prey_caught"], raids=c.stats["raid_prey_eaten"], hits=c.stats["hits"])
                    for c in creatures}, "falls", world.falls)
    print(f"wrote {out} (+ pose files)")


if __name__ == "__main__":
    main()
