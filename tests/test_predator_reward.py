"""Does the predator reward prefer hunting? Run: python tests/test_predator_reward.py

Compares four behaviours under reward v1 (plan section 8) and v2 (fed + every catch + hunting guide):
a scripted hunter, a scripted non-hunter (patrols and rests, never hunts), standing still, random.
A reward that teaches hunting must rank the hunter clearly first.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.creature_env import CreatureEnv  # noqa: E402
from sim.scripted import ScriptedPredator  # noqa: E402


class NonHunter(ScriptedPredator):
    def __call__(self, obs, c, env):
        h = c.hunger
        c.hunger = 0.0                     # the scripted rules never see hunger -> never hunt
        try:
            return super().__call__(obs, c, env)
        finally:
            c.hunger = h


def episode(species, rival, version, policy, seed):
    env = CreatureEnv(species, others=[{"species": rival}], reward_version=version)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(6000):
        if policy == "hunter":
            a = hunter(obs, env.ego, env)
        elif policy == "non-hunter":
            a = non_hunter(obs, env.ego, env)
        elif policy == "stand still":
            a = np.zeros(2)
        else:
            a = rng.uniform(-1, 1, 2)
        obs, r, te, tr, _ = env.step(a)
        total += r
        if te or tr:
            break
    return total, env.t / 10, env.ego.stats["prey_caught"]


if __name__ == "__main__":
    ok = True
    for species, rival in (("myrranox", "dc"), ("dc", "myrranox")):
        for version in (1, 2):
            rows = {}
            for policy in ("hunter", "non-hunter", "stand still", "random"):
                res = []
                for seed in range(3):
                    hunter, non_hunter = ScriptedPredator(seed), NonHunter(seed)
                    res.append(episode(species, rival, version, policy, seed))
                rows[policy] = np.mean(res, axis=0)
            line = "  ".join(f"{p}: {v[0]:+6.1f} ({v[1]:3.0f}s, {v[2]:.1f} prey)" for p, v in rows.items())
            print(f"{species:9s} reward v{version}: {line}")
            if version == 2:
                best_other = max(v[0] for p, v in rows.items() if p != "hunter")
                ok &= rows["hunter"][0] > best_other + 5
    print("PASS: reward v2 ranks hunting clearly first" if ok else "FAIL")
