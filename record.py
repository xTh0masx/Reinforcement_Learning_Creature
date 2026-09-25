"""Run creatures in CreatureEnv and write recordings/trajectories_<date>.csv for Blender.

Columns (PROJECT_PLAN.md section 5): time, id, species, x, y, heading, speed, turn_rate, activity.
Prey are written with species "prey" and activity "hidden" while waiting to respawn.

Policies:
    heuristic   scripted stand-in brain (hunt when hungry, rest when tired, stay home)
    random      random actions
    <path.zip>  trained SB3 PPO brain (phase 2+)

    python record.py --species myrranox dc --seconds 120
"""
import argparse
import csv
import datetime
import math
from pathlib import Path

import numpy as np

from sim.creature_env import DT, CreatureEnv, _wrap
from sim.species import SPECIES


class HeuristicPolicy:
    """Hand-written controller, only for previewing paths before a brain is trained."""

    def __init__(self, env, seed=0):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.wander_target = None

    def __call__(self, obs):
        env, sp = self.env, self.env.species
        speed = sp.walk_speed / sp.max_speed
        target = None

        in_home = np.linalg.norm(env.prey_pos - env.home.center, axis=1) < env.home.r - 2
        visible = env.prey_alive & in_home & (np.linalg.norm(env.prey_pos - env.pos, axis=1) < 30)
        if env.fatigue > 0.7 or (env.fatigue > 0.2 and env.activity == "rest"):
            i, d = env.habitat.nearest_rest_spot(env.pos)
            if d < env.habitat.rest_spots[i].r * 0.5:
                return np.array([env.speed_action(env.ego, 0.0), 0.0])  # stop and rest
            target = env.habitat.rest_spots[i].center
        elif env.hunger > 0.3 and visible.any():
            d = np.linalg.norm(env.prey_pos - env.pos, axis=1)
            d[~visible] = np.inf
            target, speed = env.prey_pos[int(np.argmin(d))], 1.0
        elif not env.home.contains(env.pos):
            target = env.home.center
        else:
            if self.wander_target is None or np.hypot(*(self.wander_target - env.pos)) < 2:
                self.wander_target = env.habitat.random_free_point(self.rng, margin=2, region=env.home)
            target = self.wander_target

        err = _wrap(math.atan2(target[1] - env.pos[1], target[0] - env.pos[0]) - env.heading)
        # Obstacle rays: 0 = ahead, 1/7 = 45 deg left/right (normalised to RAY_RANGE)
        rays = obs[-8:]
        if min(rays[0], rays[1], rays[7]) < 0.3:
            err += 0.8 if rays[1] > rays[7] else -0.8
        turn = float(np.clip(err / math.radians(45), -1, 1))
        speed *= max(0.25, math.cos(min(abs(err), math.pi / 2)))
        return np.array([env.speed_action(env.ego, speed), turn])


def make_policy(name, env, seed):
    if name == "heuristic":
        return HeuristicPolicy(env, seed)
    if name == "random":
        env.action_space.seed(seed)
        return lambda obs: env.action_space.sample()
    from stable_baselines3 import PPO
    model = PPO.load(name)
    return lambda obs: model.predict(obs, deterministic=True)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--species", nargs="+", default=["myrranox", "dc"], choices=list(SPECIES))
    p.add_argument("--policy", default="heuristic")
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = Path(args.out or f"recordings/trajectories_{datetime.date.today():%Y-%m-%d}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    n_steps = int(round(args.seconds / DT))

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "id", "species", "x", "y", "heading", "speed", "turn_rate", "activity"])
        counts = {}
        for k, name in enumerate(args.species):
            env = CreatureEnv(species=name)
            obs, info = env.reset(seed=args.seed + k)
            policy = make_policy(args.policy, env, args.seed + k)
            cid = f"{name}_{counts.get(name, 0)}"
            counts[name] = counts.get(name, 0) + 1
            for step in range(n_steps + 1):
                t = round(step * DT, 3)
                w.writerow([t, cid, name, f"{env.pos[0]:.4f}", f"{env.pos[1]:.4f}", f"{env.heading:.5f}",
                            f"{env.speed:.4f}", f"{env.turn_rate:.5f}", env.activity])
                for j, (pos, h, alive) in enumerate(zip(env.prey_pos, env.prey_heading, env.prey_alive)):
                    w.writerow([t, f"{cid}_prey_{j}", "prey", f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{h:.5f}",
                                "", "", "walk" if alive else "hidden"])
                if step == n_steps:
                    break
                obs, _, terminated, truncated, _ = env.step(policy(obs))
                if terminated or truncated:
                    break
            s = env.stats
            print(f"{cid}: {env.t * DT:.0f} s, prey {s['prey_caught']}, hits {s['hits']}, "
                  f"home {100 * s['steps_in_home'] / max(env.t, 1):.0f} %, reward {s['reward']:.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
