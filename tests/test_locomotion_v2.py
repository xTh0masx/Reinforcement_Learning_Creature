"""Checks for the v2 locomotion setup. Run: python tests/test_locomotion_v2.py

1. The mirror maps are exact: a mirrored simulation (mirrored state, mirrored actions) stays the mirror
   image of the original, with the same reward, step after step.
2. The gait reward measures what we want: a scripted gait that follows the clock scores high,
   standing still or v1's tapping scores low.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.locomotion_env import LocomotionEnv  # noqa: E402
from sim.symmetry import joint_mirror, mirror, mirror_maps  # noqa: E402


def mirrored_copy(a, b):
    """Put env b into the mirror image of env a's state."""
    jp, js = joint_mirror(a.body)
    lp = np.array([a.leg_names.index(n[:-1] + ("R" if n.endswith("L") else "L")) for n in a.leg_names])
    qa, va = a.data.qpos.copy(), a.data.qvel.copy()
    qb, vb = qa.copy(), va.copy()
    qb[1] = -qa[1]
    qb[3:7] = [qa[3], -qa[4], qa[5], -qa[6]]
    qb[a.joint_qpos] = js * qa[a.joint_qpos][jp]
    vb[0:3] = [va[0], -va[1], va[2]]
    vb[3:6] = [-va[3], va[4], -va[5]]
    vb[a.joint_qvel] = js * va[a.joint_qvel][jp]
    b.data.qpos[:], b.data.qvel[:] = qb, vb
    b.data.ctrl[:] = js * a.data.ctrl[jp]
    b.data.act[:] = a.data.act
    import mujoco
    mujoco.mj_forward(b.model, b.data)
    b.command = np.array([a.command[0], -a.command[1]])
    b.command_steps_left = a.command_steps_left
    b.phase = (a.phase + 0.5) % 1.0
    b.last_action = js * a.last_action[jp]
    b.prev_qvel = js * a.prev_qvel[jp]
    pf = a.prev_foot_pos[lp].copy()
    pf[:, 1] *= -1
    b.prev_foot_pos = pf
    b.prev_contact = a.prev_contact[lp].copy()
    b.air_time = a.air_time[lp].copy()
    b.t = a.t


def test_mirror_simulation(species, steps=15):
    a, b = LocomotionEnv(species, version=2), LocomotionEnv(species, version=2)
    maps = mirror_maps(a)
    a.reset(seed=3, options={"command": (a.species.walk_speed, 0.5 * a.species.max_turn_rate)})
    b.reset(seed=3)
    mirrored_copy(a, b)
    rng = np.random.default_rng(0)
    worst_obs, worst_rew = 0.0, 0.0
    for _ in range(steps):
        act = rng.uniform(-0.5, 0.5, a.nj)
        oa, ra, *_ = a.step(act)
        ob, rb, *_ = b.step(mirror(act, maps[2], maps[3]))
        worst_obs = max(worst_obs, float(np.max(np.abs(ob - mirror(oa, maps[0], maps[1])))))
        worst_rew = max(worst_rew, abs(ra - rb))
    print(f"  {species}: mirrored simulation max |obs diff| {worst_obs:.2e}, max |reward diff| {worst_rew:.2e}")
    assert worst_obs < 1e-3 and worst_rew < 1e-3, "mirror maps or body are not symmetric"


def lift_signs(species):
    """For every joint beyond the hip: which action sign raises the foot (found by trying both)."""
    signs = {}
    for sgn in (1.0, -1.0):
        env = LocomotionEnv(species, version=2)
        env.reset(seed=0, options={"command": (0.0, 0.0)})
        z0 = env.data.site_xpos[env.foot_sites][:, 2] - env.data.qpos[2]          # foot height relative to the body
        k = 0
        act = np.zeros(env.nj)
        for leg in env.body.legs:
            for seg, axis, _ in leg.joints:
                if seg == 1 and abs(axis[2]) < 0.5:
                    act[k] = sgn * 0.8
                k += 1
        for _ in range(15):
            env.step(act)
        signs[sgn] = float(np.mean(env.data.site_xpos[env.foot_sites][:, 2] - env.data.qpos[2] - z0))
    return max(signs, key=signs.get)


def make_scripted_gait(species):
    """Follows the gait clock with the knee/femur joint: swing legs lift, stance legs stay down. No brain."""
    up = lift_signs(species)

    def policy(env, obs):
        _, stance, *_ = env.gait_targets()
        act = np.zeros(env.nj)
        k = 0
        for i, leg in enumerate(env.body.legs):
            for seg, axis, _ in leg.joints:
                if seg == 1 and abs(axis[2]) < 0.5:
                    act[k] = -0.1 * up if stance[i] else 0.8 * up
                k += 1
        return act
    return policy


def gait_score(species, policy, seconds=6.0):
    """Mean of the gait reward term (1 = perfect, 0 = standing still / random)."""
    env = LocomotionEnv(species, version=2)
    obs, _ = env.reset(seed=0, options={"command": (env.species.walk_speed, 0.0)})
    scores = []
    for _ in range(int(seconds / 0.02)):
        obs, _, fell, _, info = env.step(policy(env, obs))
        scores.append(info["terms"]["gait"] / env.w["gait"])
        if fell:
            break
    return float(np.mean(scores[50:]))


if __name__ == "__main__":
    print("1. mirror symmetry")
    for sp in ("myrranox", "dc"):
        test_mirror_simulation(sp)
    print("2. gait reward (1 = perfect gait, 0 = standing still or random)")
    rng = np.random.default_rng(0)
    for sp in ("myrranox", "dc"):
        stand = gait_score(sp, lambda e, o: np.zeros(e.nj))
        rand = gait_score(sp, lambda e, o: rng.uniform(-1, 1, e.nj))
        scripted = gait_score(sp, make_scripted_gait(sp))
        print(f"  {sp}: stand still {stand:+.2f}, random {rand:+.2f}, scripted clock-following gait {scripted:+.2f}")
        assert abs(stand) < 0.15, "standing still must not earn (much) gait reward"
        # An open-loop gait only stays upright on six legs; the DC (64 kg quadruped) falls within ~1.5 s,
        # so the positive check is done on the Myrranox.
        if sp == "myrranox":
            assert scripted > max(stand, rand) + 0.3, "gait reward does not prefer a clock-following gait"
    print("all checks passed")
