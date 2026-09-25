"""Quick test for CreatureEnv: Gymnasium's check_env, then one episode with random actions.

    python check_env.py                     # both species checked, Myrranox episode in the viewer
    python check_env.py --species dc
    python check_env.py --no-render         # headless, prints the summary only
"""
import argparse
import time

from gymnasium.utils.env_checker import check_env

from sim.creature_env import CreatureEnv
from sim.species import SPECIES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--species", default="myrranox", choices=list(SPECIES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None, help="stop after N steps (default: full episode)")
    p.add_argument("--render-every", type=int, default=5, help="redraw the viewer every N steps")
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()

    for name in SPECIES:
        check_env(CreatureEnv(species=name), skip_render_check=True)
        print(f"check_env passed: {name}")

    env = CreatureEnv(species=args.species)
    env.action_space.seed(args.seed)
    obs, info = env.reset(seed=args.seed)
    viewer = None
    if not args.no_render:
        from sim.viewer import HabitatViewer
        viewer = HabitatViewer(env.habitat)

    start = time.perf_counter()
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if viewer and env.t % args.render_every == 0:
            viewer.update(env)
        if args.steps and env.t >= args.steps:
            break
    elapsed = time.perf_counter() - start

    s = env.stats
    print(f"\nRandom-action episode ({args.species}, seed {args.seed}):")
    print(f"  steps            {env.t} ({'starved' if terminated else 'full day' if truncated else 'stopped'})")
    print(f"  total reward     {s['reward']:.2f}")
    print(f"  prey caught      {s['prey_caught']}")
    print(f"  obstacle hits    {s['hits']}")
    print(f"  time in home     {100 * s['steps_in_home'] / env.t:.1f} %")
    print(f"  time starving    {100 * s['steps_starving'] / env.t:.1f} %")
    print(f"  sim speed        {env.t / elapsed:.0f} steps/s" + (" (incl. drawing)" if viewer else ""))

    if viewer:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
