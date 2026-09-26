"""Evaluate predator (navigation) brains against the PROJECT_PLAN.md section 12 checks.

Fixed seeds, deterministic brains, 10-minute days, rival scripted or a frozen brain. The scripted predator is
evaluated the same way as a baseline.

    python evaluate.py --species myrranox --brain models/myr_brain_v3_r1 --rival dc
    python evaluate.py --species myrranox --brain scripted --rival dc
"""
import argparse
import json
import os

import numpy as np

from sim.creature_env import EPISODE_STEPS, CreatureEnv
from sim.nav_brain import NavBrain
from sim.scripted import ScriptedPredator

CHECKS = [  # key, label, target, higher is better
    ("home", "time inside own zone", 0.90, True),
    ("starving", "time with hunger at 100 %", 0.05, False),
    ("prey", "prey caught per 10 min", 3.0, True),
    ("rest_visits", "rest-spot visits when fatigue > 50 % per 10 min", 1.0, True),
    ("hits", "obstacle/boundary hits per 10 min", 1.0, False),
]


def run_episode(species, brain, rival, rival_brain, seed, intrude):
    policy = ScriptedPredator(seed) if brain == "scripted" else NavBrain(brain)
    obs_version = 2 if brain == "scripted" or policy.model.observation_space.shape[0] != 81 else 1
    others = [{"species": rival, "brain": rival_brain, "intrude_prob": intrude}] if rival else None
    env = CreatureEnv(species, others=others, obs_version=obs_version)
    obs, _ = env.reset(seed=seed)
    c = env.ego
    rest_visits, was_resting, path = 0, False, []
    for _ in range(EPISODE_STEPS):
        a = policy(obs, c, env) if brain == "scripted" else policy(obs)
        obs, _, term, trunc, _ = env.step(a)
        resting = c.activity == "rest" and c.fatigue > 0.3
        rest_visits += resting and not was_resting
        was_resting = resting
        path.append(c.pos.copy())
        if term or trunc:
            break
    s, t = c.stats, env.t
    per10 = EPISODE_STEPS / max(t, 1)
    return {"survived_s": t / 10, "starved": bool(term), "home": s["steps_in_home"] / t,
            "starving": s["steps_starving"] / t, "prey": s["prey_caught"] * per10,
            "raids": s["raid_prey_eaten"] * per10, "rest_visits": rest_visits * per10, "hits": s["hits"] * per10,
            "rival_zone": s["steps_in_rival_zone"] / t, "intruders_repelled": s["intruders_repelled"] * per10,
            "path": np.array(path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--species", required=True)
    p.add_argument("--brain", required=True, help='models/<run> or "scripted"')
    p.add_argument("--rival", default=None)
    p.add_argument("--rival-brain", default=None)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--intrude", type=float, default=0.4, help="share of days the scripted rival invades")
    p.add_argument("--out", default="docs/evaluations")
    args = p.parse_args()

    eps = [run_episode(args.species, args.brain, args.rival, args.rival_brain, 1000 + s, args.intrude)
           for s in range(args.seeds)]
    mean = {k: float(np.mean([e[k] for e in eps])) for k in eps[0] if k != "path"}
    # different paths: mean distance between the first 2 minutes of two episodes' paths
    name = "scripted" if args.brain == "scripted" else os.path.basename(args.brain)
    print(f"{args.species} - {name} vs {args.rival or 'no rival'} "
          f"({args.rival_brain or 'scripted'}), {args.seeds} days of 10 min:")
    print(f"  survived {mean['survived_s']:.0f} s on average, starved in {sum(e['starved'] for e in eps)}/{len(eps)} days")
    passed = 0
    for key, label, target, higher in CHECKS:
        ok = mean[key] >= target if higher else mean[key] <= target
        passed += ok
        v = f"{100 * mean[key]:.0f} %" if key in ("home", "starving") else f"{mean[key]:.2f}"
        tgt = f"{'>=' if higher else '<='} {100 * target:.0f} %" if key in ("home", "starving") else \
            f"{'>=' if higher else '<='} {target:g}"
        print(f"  [{'PASS' if ok else 'fail'}] {label:48s} {v:>8}   (target {tgt})")
    print(f"  also: raids {mean['raids']:.2f} / 10 min, time in rival zone {100 * mean['rival_zone']:.0f} %, "
          f"intruders repelled {mean['intruders_repelled']:.2f} / 10 min")
    print(f"  {passed}/{len(CHECKS)} checks passed")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.species}_{name}_vs_{args.rival or 'none'}.json"), "w") as f:
        json.dump({"species": args.species, "brain": args.brain, "rival": args.rival, "rival_brain": args.rival_brain,
                   "seeds": args.seeds, "mean": mean, "checks_passed": passed}, f, indent=2)


if __name__ == "__main__":
    main()
