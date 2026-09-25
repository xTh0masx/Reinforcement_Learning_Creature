"""Train a navigation (predator) brain with PPO in CreatureEnv - PROJECT_PLAN.md sections 8 and 15.

    # stage 3: against a scripted rival
    python train.py --species myrranox --rival dc --steps 3e6 --run myr_brain_v3_r1
    # stage 4: against the other species' frozen brain (alternating rounds)
    python train.py --species myrranox --rival dc --rival-brain models/dc_brain_v3_r1 --resume models/myr_brain_v3_r1 \
        --steps 3e6 --run myr_brain_v3_r2
    tensorboard --logdir runs        # habitat/* curves: prey caught, raids, starving, time at home, intruders repelled
"""
import argparse

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize

from sim.creature_env import CreatureEnv


class PreyCurriculum(BaseCallback):
    """Prey escape ability 0 -> 1 over the first `ramp` of training: at first prey do not flee, so a beginner can
    catch one and learn that hunting pays; at full difficulty they flee at habitat.json flee_speed."""

    def __init__(self, total, ramp=0.4, start=0.0):
        super().__init__()
        self.total, self.ramp, self.start = total, ramp, start

    def _on_rollout_start(self):
        x = self.start + (1 - self.start) * min(1.0, self.num_timesteps / (self.ramp * self.total))
        self.training_env.env_method("set_difficulty", x)
        self.logger.record("train/prey_difficulty", x)

    def _on_step(self):
        return True


class HabitatStats(BaseCallback):
    """Log per-episode habitat statistics (from info["episode_stats"]) as habitat/* in TensorBoard."""

    def __init__(self):
        super().__init__()
        self.buf = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode_stats" in info:
                self.buf.append(info["episode_stats"])
        return True

    def _on_rollout_end(self):
        if not self.buf:
            return
        st = self.buf[-50:]
        per10 = lambda k: float(np.mean([s[k] * 6000 / max(s["steps"], 1) for s in st]))
        frac = lambda k: float(np.mean([s[k] / max(s["steps"], 1) for s in st]))
        self.logger.record("habitat/prey_per_10min", per10("prey_caught"))
        self.logger.record("habitat/raids_per_10min", per10("raid_prey_eaten"))
        self.logger.record("habitat/starving_fraction", frac("steps_starving"))
        self.logger.record("habitat/home_fraction", frac("steps_in_home"))
        self.logger.record("habitat/rival_zone_fraction", frac("steps_in_rival_zone"))
        self.logger.record("habitat/hits_per_10min", per10("hits"))
        self.logger.record("habitat/intruders_repelled_per_10min", per10("intruders_repelled"))
        self.logger.record("habitat/survived_fraction", float(np.mean([s["steps"] / 6000 for s in st])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--species", default="myrranox")
    p.add_argument("--rival", default=None, help="species of the rival sharing the habitat (none = alone)")
    p.add_argument("--rival-brain", default=None, help="frozen navigation brain of the rival; default scripted")
    p.add_argument("--steps", type=float, default=3e6)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--run", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None, help="models/<run> to continue from (next round)")
    p.add_argument("--ckpt-every", type=float, default=500_000)
    p.add_argument("--no-curriculum", action="store_true", help="prey at full difficulty from the start")
    args = p.parse_args()
    run = args.run or f"{args.species[:3]}_brain_v1"
    torch.set_num_threads(2)

    others = [{"species": args.rival, "brain": args.rival_brain}] if args.rival else None
    env = make_vec_env(CreatureEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
                       env_kwargs={"species": args.species, "others": others})
    env = VecFrameStack(env, n_stack=3)
    if args.resume:
        env = VecNormalize.load(f"{args.resume}_vecnormalize.pkl", env)
        model = PPO.load(args.resume, env=env, tensorboard_log="runs/")
    else:
        env = VecNormalize(env, norm_reward=True)
        model = PPO(
            "MlpPolicy", env,
            n_steps=2048, batch_size=512, n_epochs=10,
            learning_rate=3e-4, gamma=0.995, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.005,
            policy_kwargs=dict(net_arch=[128, 128]),
            tensorboard_log="runs/", seed=args.seed, verbose=1,
        )
        model.save(f"checkpoints/{run}_0_steps")
        env.save(f"checkpoints/{run}_vecnormalize_0_steps.pkl")
    ckpt = CheckpointCallback(max(int(args.ckpt_every) // args.envs, 1), "checkpoints/", name_prefix=run,
                              save_vecnormalize=True)
    callbacks = [ckpt, HabitatStats()]
    if not args.no_curriculum:
        callbacks.append(PreyCurriculum(int(args.steps), start=0.5 if args.resume else 0.0))
    model.learn(total_timesteps=int(args.steps), tb_log_name=run, callback=CallbackList(callbacks),
                reset_num_timesteps=not args.resume)
    model.save(f"models/{run}")
    env.save(f"models/{run}_vecnormalize.pkl")
    print(f"saved models/{run}")


if __name__ == "__main__":
    main()
