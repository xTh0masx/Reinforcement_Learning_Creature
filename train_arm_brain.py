"""Train the DC arm brain (sim/arm_brain.py) with the walking brain frozen.

    python train_arm_brain.py --main models/dc_legs_v12q --steps 2e6 --run dc_arms_v1

Saves models/<run>.zip and models/<run>_vecnormalize.pkl and checkpoints every --ckpt-every steps.
"""
import argparse
import functools

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from sim.arm_brain import ArmEnv


def make(main, seed):
    env = ArmEnv(main)
    env.reset(seed=seed)
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", required=True, help="frozen walking brain (models/<run> or checkpoints/<run>_<N>_steps)")
    p.add_argument("--steps", type=float, default=2e6)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt-every", type=float, default=250_000)
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    env = SubprocVecEnv([functools.partial(make, args.main, i) for i in range(args.envs)])
    if args.resume:
        from play_locomotion import vecnormalize_path
        env = VecNormalize.load(vecnormalize_path(args.resume), env)
        model = PPO.load(args.resume, env=env, tensorboard_log="runs/")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO("MlpPolicy", env, n_steps=512, batch_size=1024, n_epochs=5, learning_rate=3e-4, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.2, target_kl=0.03, max_grad_norm=1.0,
                    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[256, 256]), activation_fn=torch.nn.ELU,
                                       log_std_init=-1.0),
                    tensorboard_log="runs/", seed=0, verbose=1)
    model.main_brain = args.main
    ckpt = CheckpointCallback(save_freq=max(int(args.ckpt_every) // args.envs, 1), save_path="checkpoints",
                              name_prefix=args.run, save_vecnormalize=True)
    model.learn(total_timesteps=int(args.steps), tb_log_name=args.run, callback=ckpt,
                reset_num_timesteps=not args.resume)
    model.save(f"models/{args.run}")
    env.save(f"models/{args.run}_vecnormalize.pkl")


if __name__ == "__main__":
    main()
