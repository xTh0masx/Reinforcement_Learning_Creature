"""Watch a locomotion run live: a MuJoCo window plays the newest checkpoint and switches to each new one
as soon as training saves it (every 250k steps). Commands cycle through play_locomotion's script.

    python watch_locomotion.py --run myr_legs_v1 --species myrranox
    python watch_locomotion.py --run dc_legs_v1 --species dc

Mouse: drag to rotate, right-drag to pan, scroll to zoom. Close the window to stop.
"""
import argparse
import functools
import time

import mujoco
import mujoco.viewer

from locomotion_progress import checkpoints, fmt_steps
from play_locomotion import Brain, command_script
from sim.locomotion_env import CONTROL_DT, LocomotionEnv

print = functools.partial(print, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--species", required=True)
    p.add_argument("--poll", type=float, default=10.0, help="seconds between checks for a new checkpoint")
    p.add_argument("--version", type=int, default=None,
                   help="body / env version to show (a resumed run's first checkpoint still carries the old version)")
    p.add_argument("--arm-run", default=None,
                   help="DC: also show the newest checkpoint of this arm-brain run (sim/arm_brain.py) on the arms")
    args = p.parse_args()

    def load(path):
        if not args.arm_run or not checkpoints(args.arm_run):
            return Brain(path, args.species)
        from sim.arm_brain import CombinedBrain
        return CombinedBrain(path, checkpoints(args.arm_run)[-1][1])

    print(f"waiting for the first checkpoint of {args.run} ...")
    while not checkpoints(args.run):
        time.sleep(args.poll)
    steps, loaded = checkpoints(args.run)[-1]
    brain = load(loaded)
    env = LocomotionEnv(args.species, episode_seconds=1e9, version=args.version or brain.version)
    obs, _ = env.reset(seed=0, options={"command": (0.0, 0.0)})
    print(f"now showing {args.run} @ {fmt_steps(steps)}")
    script = command_script(args.species)
    seg, seg_left, last_poll = 0, 0.0, time.time()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.torso_body
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 3.0, -20, 135
        while viewer.is_running():
            now = time.time()
            if now - last_poll > args.poll:
                last_poll = now
                cks = checkpoints(args.run)
                if cks and cks[-1][1] != loaded:
                    steps, loaded = cks[-1]
                    brain = load(loaded)
                    obs, _ = env.reset(seed=0, options={"command": (0.0, 0.0)})
                    seg, seg_left = 0, 0.0
                    print(f"now showing {args.run} @ {fmt_steps(steps)}")
            if brain is None:
                time.sleep(0.5)
                continue
            if seg_left <= 0:
                seconds, v, w = script[seg % len(script)]
                env.set_command(v, w)
                seg_left, seg = seconds, seg + 1
                print(f"  command {v:+.2f} m/s, {w * 57.3:+.0f} deg/s")
            obs, _, fell, _, _ = env.step(brain(obs))
            seg_left -= CONTROL_DT
            if fell:
                print("  fell - resetting")
                obs, _ = env.reset(seed=0, options={"command": tuple(env.command)})
            viewer.sync()
            time.sleep(max(0.0, CONTROL_DT - (time.time() - now)))


if __name__ == "__main__":
    main()
