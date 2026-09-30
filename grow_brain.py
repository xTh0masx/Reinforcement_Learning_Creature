"""Transplant a trained walking brain into a body with more joints (DC walking v5c -> v6 with actuated arms).

Every weight that belongs to a joint / leg the old body had is copied to its place in the new observation and
action vectors (matched by name); new joints start neutral (zero output weights, observation statistics 0 / 1).
Joints listed in --recenter get their output bias shifted so their mean command is 0 (e.g. the DC's sideways leg
joints, which the old brain held at the old 25 deg limit - the new body allows only 8 deg).

    python grow_brain.py models/dc_legs_v5c --species dc --version 6 --out models/dc_legs_v6_start --recenter _j0
"""
import argparse

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from play_locomotion import Brain, vecnormalize_path
from sim.locomotion_env import LocomotionEnv
from sim.symmetry import SymmetricPPO, mirror_maps
from unsaturate_brain import mean_outputs


def index_maps(old_env, new_env):
    """new index -> old index (or -1) for observation and action entries."""
    lo, ln = old_env.obs_layout(), new_env.obs_layout()
    oj, nj = old_env.body.joint_names, new_env.body.joint_names
    act = np.array([oj.index(n) if n in oj else -1 for n in nj])
    obs = np.full(new_env.observation_space.shape[0], -1)
    for key, sl in ln.items():
        osl = lo.get(key)
        if osl is None:
            continue
        if key in ("qpos", "qvel", "last_action"):
            src = act
        elif key in ("clock_sin", "clock_cos"):
            src = np.array([old_env.leg_names.index(n) if n in old_env.leg_names else -1 for n in new_env.leg_names])
        else:
            src = np.arange(sl.stop - sl.start)
        obs[sl] = np.where(src >= 0, osl.start + src, -1)
    return obs, act


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--species", required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--recenter", nargs="*", default=[], help="joint name suffixes whose mean command is set to 0")
    p.add_argument("--reset-steps", action="store_true",
                   help="start the step counter at 0 (the penalty curriculum then ramps in again)")
    args = p.parse_args()

    old = Brain(args.model, args.species)
    old_env = LocomotionEnv(args.species, version=old.version)
    new_env = LocomotionEnv(args.species, version=args.version)
    obs_map, act_map = index_maps(old_env, new_env)
    print(f"obs {old_env.observation_space.shape[0]} -> {len(obs_map)} ({(obs_map >= 0).sum()} copied), "
          f"actions {old_env.action_space.shape[0]} -> {len(act_map)} ({(act_map >= 0).sum()} copied)")

    venv = VecNormalize(DummyVecEnv([lambda: LocomotionEnv(args.species, version=args.version)]),
                        norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    om = old.model
    new = SymmetricPPO("MlpPolicy", venv, n_steps=om.n_steps, batch_size=om.batch_size, n_epochs=om.n_epochs,
                       learning_rate=3e-4, gamma=om.gamma, gae_lambda=om.gae_lambda, clip_range=0.2, ent_coef=0.0,
                       max_grad_norm=om.max_grad_norm, target_kl=om.target_kl,
                       policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                          activation_fn=torch.nn.ELU, log_std_init=-1.0),
                       tensorboard_log="runs/", verbose=1, sym_weight=getattr(om, "sym_weight", 1.0))
    op, np_ = om.policy.state_dict(), new.policy.state_dict()
    keep_o, keep_n = obs_map >= 0, np.flatnonzero(obs_map >= 0)
    keep_a = np.flatnonzero(act_map >= 0)
    with torch.no_grad():
        for k, v in np_.items():
            src = op[k]
            if src.shape == v.shape:
                v.copy_(src)
            elif k.endswith("0.weight") and ("policy_net" in k or "value_net" in k or "shared_net" in k):
                v.zero_()
                v[:, keep_n] = src[:, obs_map[keep_o]]      # first layer: copy the columns of known inputs
            elif k.startswith("action_net.weight"):
                v.zero_()
                v[keep_a] = src[act_map[keep_a]]
            elif k.startswith("action_net.bias"):
                v.zero_()
                v[keep_a] = src[act_map[keep_a]]
            elif k == "log_std":
                v.fill_(-1.0)
                v[keep_a] = src[act_map[keep_a]]
            else:
                raise RuntimeError(f"unexpected shape change in {k}: {tuple(src.shape)} -> {tuple(v.shape)}")
        new.policy.load_state_dict(np_)

    # observation normalisation: copy the statistics of known inputs, neutral for new ones
    rms_o, rms_n = old.norm.obs_rms, venv.obs_rms
    rms_n.mean[:] = 0.0
    rms_n.var[:] = 1.0
    rms_n.mean[keep_n] = rms_o.mean[obs_map[keep_o]]
    rms_n.var[keep_n] = rms_o.var[obs_map[keep_o]]
    rms_n.count = rms_o.count
    venv.ret_rms = old.norm.ret_rms
    new.num_timesteps = 0 if args.reset_steps else om.num_timesteps
    new.loco_version = args.version
    new.set_mirror(mirror_maps(new_env))
    new.save(args.out)
    venv.save(vecnormalize_path(args.out))

    if args.recenter:
        brain = Brain(args.out, args.species)
        mean, names = mean_outputs(brain, args.species)
        with torch.no_grad():
            for j, n in enumerate(names):
                if any(n.endswith(s) for s in args.recenter):
                    brain.model.policy.action_net.bias[j] -= float(mean[j])
                    print(f"recentred {n}: mean output {mean[j]:+.2f} -> 0")
        brain.model.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
