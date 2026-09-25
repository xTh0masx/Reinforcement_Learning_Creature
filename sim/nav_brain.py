"""A trained navigation brain (train.py: PPO + 3-frame stack + VecNormalize) usable step by step."""
import pickle
from collections import deque

import numpy as np


class NavBrain:
    def __init__(self, path, n_stack=3):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")
        with open(f"{path}_vecnormalize.pkl", "rb") as f:
            self.norm = pickle.load(f)
        self.frames = deque(maxlen=n_stack)

    def reset(self):
        self.frames.clear()

    def __call__(self, obs):
        if not self.frames:
            self.frames.extend([np.zeros_like(obs)] * (self.frames.maxlen - 1))
        self.frames.append(obs)
        stacked = np.concatenate(self.frames)
        o = np.clip((stacked - self.norm.obs_rms.mean) / np.sqrt(self.norm.obs_rms.var + self.norm.epsilon),
                    -self.norm.clip_obs, self.norm.clip_obs)
        return self.model.predict(o[None].astype(np.float32), deterministic=True)[0][0]
