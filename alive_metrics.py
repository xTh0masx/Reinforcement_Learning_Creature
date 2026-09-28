"""How "alive" (and how fluid) does a walking brain move? Measures head steadiness, spine use and the swing of tail / abdomen / arms
on the fixed command script (walk, full speed, turns), for comparing walking v2 (with the stage-1 passive parts)
and v3 (actively controlled spine / neck / head).

    python alive_metrics.py models/dc_legs_v2 models/dc_legs_v3 --species dc
"""
import argparse
import json
import os

import mujoco
import numpy as np

from play_locomotion import Brain, command_script
from sim.locomotion_env import CONTROL_DT, LocomotionEnv

HEAD = {"myrranox": "head_s0", "dc": "neck_s1"}


def measure(path, species, seed=0):
    brain = Brain(path, species)
    env = LocomotionEnv(species, episode_seconds=1e9, version=brain.version, passive=True)
    m, d = env.model, env.data
    head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HEAD[species])
    names = [n for n in env.body.joint_names[sum(len(l.joints) for l in env.body.legs):]] + env.body.passive_joint_names
    qadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names}
    obs, _ = env.reset(seed=seed, options={"command": (0.0, 0.0)})
    head_w, head_tilt, torso_w, q, speed_err, falls, power = [], [], [], {n: [] for n in names}, [], 0, []
    jacc, jump = [], []                                            # fluidity: joint acceleration, target jumps
    prev_v, prev_ctrl = d.qvel[env.joint_qvel].copy(), d.ctrl[env.act_ids].copy()
    for seconds, v, w in command_script(species)[:4]:              # walk, full speed, turn left, turn right
        env.set_command(v, w)
        for i in range(int(seconds / CONTROL_DT)):
            obs, _, fell, _, info = env.step(brain(obs))
            qv, ctrl = d.qvel[env.joint_qvel].copy(), d.ctrl[env.act_ids].copy()
            acc, dctrl = (qv - prev_v) / CONTROL_DT, ctrl - prev_ctrl
            prev_v, prev_ctrl = qv, ctrl
            if i * CONTROL_DT < 1.0:
                continue
            hw = d.cvel[head][:3] - np.array([0, 0, w])
            tw = d.cvel[env.torso_body][:3] - np.array([0, 0, w])
            head_w.append(hw @ hw)
            torso_w.append(tw @ tw)
            head_tilt.append(np.degrees(np.arccos(np.clip(d.xmat[head].reshape(3, 3)[2, 2], -1, 1))))
            for n in names:
                q[n].append(np.degrees(d.qpos[qadr[n]]))
            speed_err.append(abs(info["speed"] - v))
            jacc.append(np.mean(acc ** 2))
            jump.append(np.mean(dctrl ** 2))
            power.append(info["power"])
            if fell:
                falls += 1
                obs, _ = env.reset(seed=falls, options={"command": (v, w)})
    return {
        "brain": path, "version": brain.version,
        "head_wobble_rad_s": float(np.sqrt(np.mean(head_w))),       # head rotation not explained by the turn
        "torso_wobble_rad_s": float(np.sqrt(np.mean(torso_w))),
        "head_calmer_than_torso": float(np.sqrt(np.mean(torso_w)) / max(np.sqrt(np.mean(head_w)), 1e-9)),
        "head_tilt_deg": float(np.mean(head_tilt)),
        "joint_swing_deg": {n: float(np.std(v)) for n, v in q.items()},
        "joint_acc_rms": float(np.sqrt(np.mean(jacc))),           # rad/s^2, lower = smoother
        "target_jump_deg": float(np.degrees(np.sqrt(np.mean(jump)))),  # joint target change per 20 ms
        "speed_error": float(np.mean(speed_err)), "power_w": float(np.mean(power)), "falls": falls,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--species", required=True)
    p.add_argument("--out", default="docs/evaluations")
    args = p.parse_args()
    rows = [measure(m, args.species) for m in args.models]
    for r in rows:
        print(f"{os.path.basename(r['brain'])} (v{r['version']}): head wobble {r['head_wobble_rad_s']:.2f} rad/s "
              f"(torso {r['torso_wobble_rad_s']:.2f}, head {r['head_calmer_than_torso']:.1f}x calmer), "
              f"head tilt {r['head_tilt_deg']:.1f} deg, joint acc {r['joint_acc_rms']:.0f} rad/s2, "
              f"target jump {r['target_jump_deg']:.2f} deg, speed error {r['speed_error']:.2f} m/s, "
              f"power {r['power_w']:.0f} W, falls {r['falls']}")
        print("   swing (std, deg): " + ", ".join(f"{n} {v:.1f}" for n, v in r["joint_swing_deg"].items()))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"alive_{args.species}.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
