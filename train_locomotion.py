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


class TrackingCurriculum(BaseCallback):
    """Version 4: narrow the speed / turn tracking tolerance from 1 to `end` between 20 % and 80 % of this run
    (counted from where the run starts, also when resuming)."""

    def __init__(self, steps, end=0.5, start=1.0):
        super().__init__()
        self.steps, self.end, self.start, self.t0 = steps, end, start, None

    def _on_rollout_start(self):
        if self.t0 is None:
            self.t0 = self.num_timesteps
        frac = min(1.0, max(0.0, ((self.num_timesteps - self.t0) / self.steps - 0.2) / 0.6))
        scale = self.start + (self.end - self.start) * frac
        self.training_env.env_method("set_track_scale", scale)
        self.logger.record("train/track_scale", scale)

    def _on_step(self):
        return True


class WalkHelp(BaseCallback):
    """Start-up help: the forward-progress bonus (LocomotionEnv.set_walk_help) fades from 1 to 0 over `steps`."""

    def __init__(self, steps):
        super().__init__()
        self.steps, self.t0 = steps, None

    def _on_rollout_start(self):
        if self.t0 is None:
            self.t0 = self.num_timesteps
        scale = max(0.0, 1.0 - (self.num_timesteps - self.t0) / self.steps)
        self.training_env.env_method("set_walk_help", scale)
        self.logger.record("train/walk_help", scale)

    def _on_step(self):
        return True


class CommandRamp(BaseCallback):
    """Sprint stage: the highest commanded speed grows from `start` to `end` over the first 80 % of `steps`, so the
    creature learns each speed range before the next (straight to 4 m/s it would mostly fall)."""

    def __init__(self, steps, start, end):
        super().__init__()
        self.steps, self.start, self.end, self.t0 = steps, start, end, None

    def _on_rollout_start(self):
        if self.t0 is None:
            self.t0 = self.num_timesteps
        frac = min(1.0, (self.num_timesteps - self.t0) / (0.8 * self.steps))
        limit = self.start + frac * (self.end - self.start)
        self.training_env.env_method("set_command_limit", limit)
        self.logger.record("train/command_limit", limit)

    def _on_step(self):
        return True


class SprintPenaltyRamp(BaseCallback):
    """Version 12: the sprint's quality costs count `start` at first and full (1.0) after 80 % of the run - while the
    DC cannot sprint yet, full costs left it nothing to gain and it unlearned walking (14 Oct)."""

    def __init__(self, steps, start):
        super().__init__()
        self.steps, self.start, self.t0 = steps, start, None

    def _on_rollout_start(self):
        if self.t0 is None:
            self.t0 = self.num_timesteps
        frac = min(1.0, (self.num_timesteps - self.t0) / (0.8 * self.steps))
        scale = self.start + frac * (1.0 - self.start)
        self.training_env.env_method("set_sprint_penalty", scale)
        self.logger.record("train/sprint_penalty", scale)

    def _on_step(self):
        return True


class StuckJoints(BaseCallback):
    """Warn when a joint's mean command drifts beyond the action limit (clipped -> no learning signal; this froze the
    Myrranox knees in v2-v4). Logs the number of stuck joints and prints their names."""

    def __init__(self, names, limit=1.0):
        super().__init__()
        self.names, self.limit, self.last = names, limit, ()

    def _on_rollout_end(self):
        mean = self.model.rollout_buffer.actions.reshape(-1, len(self.names)).mean(0)
        stuck = tuple(n for n, m in zip(self.names, mean) if abs(m) > self.limit)
        self.logger.record("train/stuck_joints", len(stuck))
        if stuck != self.last:
            print(f"stuck joints at {self.num_timesteps}: {', '.join(stuck) or 'none'}", flush=True)
            self.last = stuck

    def _on_step(self):
        return True


class PenaltyCurriculum(BaseCallback):
    """Scale the smoothness/effort penalties from `start` to 1 over the first `ramp` fraction of training,
    so the brain first discovers walking and then learns to walk cleanly."""

    def __init__(self, total, start=0.2, ramp=0.4, fixed=None):
        super().__init__()
        self.total, self.start, self.ramp, self.fixed = total, start, ramp, fixed

    def _on_rollout_start(self):
        frac = min(1.0, self.num_timesteps / (self.ramp * self.total))
        scale = self.fixed if self.fixed is not None else self.start + (1 - self.start) * frac
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
    p.add_argument("--version", type=int, default=1,
                   help="1 = original reward, 2 = natural gait + symmetry, 3 = alive body, 4 = precise + fluid, "
                        "5 = natural style")
    p.add_argument("--track-end", type=float, default=0.5, help="version 4+: final tracking tolerance scale")
    p.add_argument("--track-start", type=float, default=1.0,
                   help="version 4+: starting tolerance scale (0.5 when continuing a v4 brain)")
    p.add_argument("--sym-weight", type=float, default=1.0, help="mirror-symmetry loss weight (version 2)")
    p.add_argument("--head-weight", type=float, default=1.0,
                   help="scale of the version-3 head steadiness/level penalties (0.1 = ten times weaker)")
    p.add_argument("--reset-reward-norm", action="store_true",
                   help="when resuming into a different reward: restart the reward normalisation statistics (the old "
                        "ones would make the new rewards look like noise - 13 Oct, v9 stage 1)")
    p.add_argument("--walk-help", type=float, default=None,
                   help="start-up help: forward-progress bonus faded out over this many steps")
    p.add_argument("--max-command", type=float, default=None,
                   help="training stage: commanded speeds up to this (walk first, then trot, run, sprint)")
    p.add_argument("--max-command-end", type=float, default=None,
                   help="sprint stage: raise the command limit from --max-command to this over the run")
    p.add_argument("--walk-share", type=float, default=None,
                   help="version 12: share of moving commands kept at walking speed (default 0.5)")
    p.add_argument("--sprint-penalty", type=float, default=None,
                   help="version 12: start weight of the sprint's quality costs, ramped to 1 over the run")
    p.add_argument("--speed-sigma", type=float, default=None,
                   help="fixed speed tracking tolerance in m/s (default 0.5 x max speed x track scale)")
    p.add_argument("--penalty-fixed", type=float, default=None,
                   help="hold the penalty scale at this value instead of ramping it (e.g. 0.6)")
    p.add_argument("--curriculum-total", type=float, default=None,
                   help="length the penalty curriculum refers to (default --steps; set to the original run's length "
                        "when resuming so the ramp continues where it was)")
    args = p.parse_args()
    run = args.run or f"{args.species}_legs_v1"
    torch.set_num_threads(2)

    env = make_vec_env(LocomotionEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
                       env_kwargs={"species": args.species, "version": args.version,
                                   "head_weight": args.head_weight, "speed_sigma": args.speed_sigma})
    algo = SymmetricPPO if args.version >= 2 else PPO
    algo_kwargs = {"sym_weight": args.sym_weight} if args.version >= 2 else {}
    if args.resume:
        from play_locomotion import vecnormalize_path          # models/<run> or checkpoints/<run>_<N>_steps
        env = VecNormalize.load(vecnormalize_path(args.resume), env)
        if args.reset_reward_norm:
            from stable_baselines3.common.running_mean_std import RunningMeanStd
            env.ret_rms = RunningMeanStd(shape=())
            env.returns[:] = 0.0
            print("reward normalisation statistics reset", flush=True)
        model = algo.load(args.resume, env=env, tensorboard_log="runs/")
        print(f"resumed {args.resume} at {model.num_timesteps} steps", flush=True)
        model.save(f"checkpoints/{run}_{model.num_timesteps}_steps")      # the starting brain, for the progress video
        env.save(f"checkpoints/{run}_vecnormalize_{model.num_timesteps}_steps.pkl")
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
    model.loco_version = args.version          # saved with the brain (play_locomotion.brain_version)
    if args.max_command:
        env.env_method("set_command_limit", args.max_command)
    if args.walk_share is not None:
        env.set_attr("walk_share", args.walk_share)
    if args.version >= 2:
        model.set_mirror(mirror_maps(LocomotionEnv(args.species, version=args.version)))

    if not args.resume:
        model.save(f"checkpoints/{run}_0_steps")
        env.save(f"checkpoints/{run}_vecnormalize_0_steps.pkl")
    ckpt = CheckpointCallback(max(int(args.ckpt_every) // args.envs, 1), "checkpoints/", name_prefix=run,
                              save_vecnormalize=True)
    callbacks = [ckpt, PenaltyCurriculum(int(args.curriculum_total or args.steps), fixed=args.penalty_fixed),
                 StuckJoints(LocomotionEnv(args.species, version=args.version).body.joint_names)]
    if args.version >= 4:
        callbacks.append(TrackingCurriculum(int(args.steps), end=args.track_end, start=args.track_start))
    if args.walk_help:
        callbacks.append(WalkHelp(int(args.walk_help)))
    if args.sprint_penalty is not None:
        callbacks.append(SprintPenaltyRamp(int(args.steps), args.sprint_penalty))
    if args.max_command_end:
        callbacks.append(CommandRamp(int(args.steps), args.max_command, args.max_command_end))
    callbacks = CallbackList(callbacks)
    model.learn(total_timesteps=int(args.steps), tb_log_name=run, callback=callbacks,
                reset_num_timesteps=not args.resume)
    model.save(f"models/{run}")
    env.save(f"models/{run}_vecnormalize.pkl")
    print(f"saved models/{run}")


if __name__ == "__main__":
    main()
