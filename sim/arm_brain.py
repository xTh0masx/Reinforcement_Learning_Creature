"""DC arm brain (walking v12, owner 14 Oct: "a sub brain only for the arms"): a small policy that controls only the
8 arm joints (spread, shoulder, elbow, wrist per side); the walking brain keeps legs, spine, neck and head.

ArmEnv wraps LocomotionEnv: every step the frozen walking brain computes the whole body's action, the arm brain's
action replaces its arm part. The arm brain is rewarded for the body doing well (the walking reward, so the arms
learn to help with balance) and for its own job:
  walk   - T-rex pose, hands extended (the arm-pose score);
  sprint - each hand on the ground when the gait wants it there and in the air otherwise, palm flat on the ground.

    python train_arm_brain.py --main models/dc_legs_v12q --steps 2e6 --run dc_arms_v1
"""
import gymnasium as gym
import numpy as np

from sim.locomotion_env import LocomotionEnv

ARM_BONUS = 0.5          # weight of the arm's own score next to the walking reward
ARM_WALK_SHARE = 0.25    # arm training: most moving commands are runs / sprints (the arms only run along from 2 m/s)
ARM_COMMAND_LIMIT = 3.0  # m/s


class ArmEnv(gym.Env):
    def __init__(self, main_path, command_limit=ARM_COMMAND_LIMIT, **env_kwargs):
        from play_locomotion import Brain
        self.env = LocomotionEnv("dc", version=12, **env_kwargs)
        self.env.set_command_limit(command_limit)
        self.env.walk_share = ARM_WALK_SHARE
        self.main = Brain(main_path, "dc")
        self.main.version = 12
        self.arm = self.env.arm_joint_mask
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(int(self.arm.sum()),), dtype=np.float32)
        self.obs = None

    def reset(self, *, seed=None, options=None):
        self.obs, info = self.env.reset(seed=seed, options=options)
        return self.obs, info

    def full_action(self, arm_action):
        a = np.array(self.main(self.obs), dtype=float)
        a[self.arm] = arm_action
        return a

    def step(self, arm_action):
        obs, r, term, trunc, info = self.env.step(self.full_action(arm_action))
        self.obs = obs
        score = self.arm_score(info)
        info["arm_score"] = score
        return obs, float(r + ARM_BONUS * score), term, trunc, info

    def arm_score(self, info):
        """0..1: walk - arm pose; sprint - hands follow the gait (down / up) with flat palms."""
        e = self.env
        arms = e.arm_legs
        if e.engage <= 0:
            duty = np.broadcast_to(np.asarray(e.gait_targets()[2], dtype=float), (e.nlegs,))
            return e._arm_tuck(e.data.site_xpos[e.foot_sites], duty)
        want, got = info["desired_stance"][arms], info["foot_contact"][arms]
        match = float(np.mean(want == got))
        flat = np.exp(-info["terms"].get("q_palm_flat", 0.0))
        return match * flat

    def set_command_limit(self, v):
        self.env.set_command_limit(v)

    def set_penalty_scale(self, s):
        self.env.set_penalty_scale(s)

    def set_track_scale(self, s):
        self.env.set_track_scale(s)


class CombinedBrain:
    """Walking brain + arm brain, used like play_locomotion.Brain: action = walking brain's, arm part from the arm
    brain (same observation)."""

    def __init__(self, main_path, arm_path):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from play_locomotion import Brain, vecnormalize_path
        self.main = Brain(main_path, "dc")
        self.version = 12
        self.arm_model = PPO.load(arm_path, device="cpu")
        venv = DummyVecEnv([lambda: LocomotionEnv("dc", version=12)])
        self.arm_norm = VecNormalize.load(vecnormalize_path(arm_path), venv)
        self.arm_norm.training = False
        self.arm = LocomotionEnv("dc", version=12).arm_joint_mask

    def __call__(self, obs):
        a = np.array(self.main(obs), dtype=float)
        a[self.arm] = self.arm_model.predict(self.arm_norm.normalize_obs(obs[None]), deterministic=True)[0][0]
        return a
