"""Watch a predator (navigation) training run live: a top-down habitat window plays the newest checkpoint
hunting against its rival and switches to each new checkpoint at the start of the next episode.

    python watch_predator.py --run myr_brain_v3_r1 --species myrranox --rival dc
Close the window to stop.
"""
import argparse
import functools
import glob
import re
import time

import matplotlib.pyplot as plt

from sim.creature_env import CreatureEnv
from sim.nav_brain import NavBrain

print = functools.partial(print, flush=True)


def newest(run):
    found = []
    for p in glob.glob(f"checkpoints/{run}_*_steps.zip"):
        m = re.search(r"_(\d+)_steps\.zip$", p)
        if m and "vecnormalize" not in p:
            found.append((int(m.group(1)), p[:-4]))
    return max(found) if found else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--species", required=True)
    p.add_argument("--rival", default=None)
    p.add_argument("--rival-brain", default=None)
    p.add_argument("--speedup", type=float, default=4.0, help="x real time")
    p.add_argument("--draw-every", type=int, default=3)
    args = p.parse_args()

    others = [{"species": args.rival, "brain": args.rival_brain}] if args.rival else None
    env = CreatureEnv(args.species, render_mode="human", others=others)
    episode = 0
    while True:
        steps, path = newest(args.run)
        brain = NavBrain(path)
        obs, _ = env.reset(seed=episode)
        env.render_mode = None                      # draw only every few steps (below)
        print(f"episode {episode}: {args.run} @ {steps / 1e6:.2f}M steps")
        t0, n = time.time(), 0
        while True:
            obs, _, term, trunc, _ = env.step(brain(obs))
            n += 1
            if n % args.draw_every == 0:
                env.render_mode = "human"
                env.render()
                env.render_mode = None
                env._viewer.ax.set_title(f"{args.run} @ {steps / 1e6:.2f}M steps  -  episode {episode}", fontsize=10)
                if not plt.fignum_exists(env._viewer.fig.number):
                    return
                lag = n * 0.1 / args.speedup - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)
            if term or trunc:
                s = env.ego.stats
                print(f"  ended after {env.t / 10:.0f} s: prey {s['prey_caught']} (raids {s['raid_prey_eaten']}), "
                      f"{'starved' if term else 'survived the day'}")
                break
        episode += 1


if __name__ == "__main__":
    main()
