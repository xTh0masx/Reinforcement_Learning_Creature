"""Train a locomotion brain (PPO) for one species in LocomotionEnv.

    python train_locomotion.py --species myrranox --steps 4e6 --run myr_legs_v1
    python train_locomotion.py --species myrranox --steps 4e6 --run myr_legs_v2 --version 2   # natural gait
    tensorboard --logdir runs

Saves models/<run>.zip and models/<run>_vecnormalize.pkl, plus checkpoints (step 0 = untrained brain,
then every --ckpt-every steps) that locomotion_progress.py turns into a progress video.
"""
import argparse

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from sim.locomotion_env import LocomotionEnv
from sim.symmetry import SymmetricPPO, mirror_maps


class PenaltyCurriculum(BaseCallback):
    """Scale the smoothness/effort penalties from `start` to 1 over the first `ramp` fraction of training,
    so the brain first discovers walking and then learns to walk cleanly."""

    def __init__(self, total, start=0.2, ramp=0.4):
        super().__init__()
        self.total, self.start, self.ramp = total, start, ramp

    def _on_rollout_start(self):
        frac = min(1.0, self.num_timesteps / (self.ramp * self.total))
        scale = self.start + (1 - self.start) * frac
        self.training_env.env_method("set_penalty_scale", scale)
        self.logger.record("train/penalty_scale", scale)

    def _on_step(self):
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--species", default="myrranox")
    p.add_argument("--steps", type=float, default=20e6)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--run", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None, help="models/<run> to continue from")
    p.add_argument("--ckpt-every", type=float, default=250_000)
    p.add_argument("--version", type=int, default=1, help="1 = original reward, 2 = natural gait + symmetry")
    p.add_argument("--sym-weight", type=float, default=1.0, help="mirror-symmetry loss weight (version 2)")
    args = p.parse_args()
    run = args.run or f"{args.species}_legs_v1"
    torch.set_num_threads(2)

    env = make_vec_env(LocomotionEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
                       env_kwargs={"species": args.species, "version": args.version})
    algo = SymmetricPPO if args.version >= 2 else PPO
    algo_kwargs = {"sym_weight": args.sym_weight} if args.version >= 2 else {}
    if args.resume:
        env = VecNormalize.load(f"{args.resume}_vecnormalize.pkl", env)
        model = algo.load(args.resume, env=env, tensorboard_log="runs/")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = algo(
            "MlpPolicy", env,
            n_steps=512, batch_size=1024, n_epochs=5,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.0, max_grad_norm=1.0, target_kl=0.03,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                               activation_fn=torch.nn.ELU, log_std_init=-1.0),
            tensorboard_log="runs/", seed=args.seed, verbose=1, **algo_kwargs,
        )
    if args.version >= 2:
        model.set_mirror(mirror_maps(LocomotionEnv(args.species, version=args.version)))

    if not args.resume:
        model.save(f"checkpoints/{run}_0_steps")
        env.save(f"checkpoints/{run}_vecnormalize_0_steps.pkl")
    ckpt = CheckpointCallback(max(int(args.ckpt_every) // args.envs, 1), "checkpoints/", name_prefix=run,
                              save_vecnormalize=True)
    callbacks = CallbackList([ckpt, PenaltyCurriculum(int(args.steps))])
    model.learn(total_timesteps=int(args.steps), tb_log_name=run, callback=callbacks,
                reset_num_timesteps=not args.resume)
    model.save(f"models/{run}")
    env.save(f"models/{run}_vecnormalize.pkl")
    print(f"saved models/{run}")


if __name__ == "__main__":
    main()
