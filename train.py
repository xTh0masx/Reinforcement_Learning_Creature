"""Train a navigation brain (PPO) in CreatureEnv - PROJECT_PLAN.md section 8, training stage 1.

    python train.py --species myrranox --steps 3_000_000 --run myr_brain_v1
    tensorboard --logdir runs
"""
import argparse

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize

from sim.creature_env import CreatureEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--species", default="myrranox")
    p.add_argument("--steps", type=float, default=3e6)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--run", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run = args.run or f"{args.species[:3]}_brain_v1"
    torch.set_num_threads(2)

    env = make_vec_env(CreatureEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
                       env_kwargs={"species": args.species})
    env = VecFrameStack(env, n_stack=3)
    env = VecNormalize(env, norm_reward=True)

    model = PPO(
        "MlpPolicy", env,
        n_steps=2048, batch_size=512, n_epochs=10,
        learning_rate=3e-4, gamma=0.995, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.005,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log="runs/", seed=args.seed, verbose=1,
    )
    model.learn(total_timesteps=int(args.steps), tb_log_name=run,
                callback=CheckpointCallback(max(500_000 // args.envs, 1), "checkpoints/", name_prefix=run,
                                            save_vecnormalize=True))
    model.save(f"models/{run}")
    env.save(f"models/{run}_vecnormalize.pkl")
    print(f"saved models/{run}")


if __name__ == "__main__":
    main()
