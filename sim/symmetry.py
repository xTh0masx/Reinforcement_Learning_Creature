"""Left/right mirror symmetry for the v2 locomotion brains.

Both bodies are mirror-symmetric (y -> -y). Mirroring a situation swaps the left and right legs and
flips sideways quantities; a symmetric brain must answer the mirrored situation with the mirrored action.
`SymmetricPPO` adds this as an extra loss (Yu, Turk & Liu 2018, "Learning Symmetric and Low-Energy
Locomotion"): after every PPO update, one pass over the rollout minimises
    sym_weight * || mean_action(mirror(obs)) - mirror(mean_action(obs)) ||^2

Mirror maps are derived from the body definition (sim/bodies.py), not typed by hand: a hinge on a left
leg with axis a maps to the matching right-leg hinge with axis a_R, and the angle keeps its sign if
a_R == -M a (M = diag(1, -1, 1)) and flips if a_R == M a.
"""
import numpy as np
import torch as th
from stable_baselines3 import PPO

M = np.diag([1.0, -1.0, 1.0])


def _other_side(name):
    return name[:-1] + ("R" if name.endswith("L") else "L")


def _partner(name):
    """Mirror partner of a joint: legs / arms swap side (front_L_j1 <-> front_R_j1), midline joints map to themselves."""
    for a, b in (("_L_", "_R_"), ("_R_", "_L_")):
        if a in name:
            return name.replace(a, b)
    return name


def joint_mirror(body, names=None):
    """(perm, sign) with mirrored_joints = sign * joints[perm] for the given joints (default: actuated, in actuator
    order). Works for legs, spine, neck, head, abdomen and passive parts alike."""
    names = list(names or body.joint_names)
    axes = {n: body.joint_axis(n) / np.linalg.norm(body.joint_axis(n)) for n in names}
    perm, sign = [], []
    for n in names:
        src = _partner(n)
        s = -float(np.dot(M @ axes[src], axes[n]))
        assert abs(abs(s) - 1) < 1e-6, f"{n}: mirrored axis does not match ({s})"
        perm.append(names.index(src))
        sign.append(round(s))
    return np.array(perm), np.array(sign, dtype=float)


def mirror_maps(env):
    """(obs_perm, obs_sign, act_perm, act_sign) for a LocomotionEnv (version 2 layout)."""
    lay = env.obs_layout()
    n = env.observation_space.shape[0]
    perm, sign = np.arange(n), np.ones(n)
    jp, js = joint_mirror(env.body)
    lp = np.array([env.leg_names.index(_other_side(l)) for l in env.leg_names])

    def put(key, p, s):
        sl = lay[key]
        perm[sl] = sl.start + np.asarray(p)
        sign[sl] = s

    put("grav", [0, 1, 2], [1, -1, 1])
    put("lin", [0, 1, 2], [1, -1, 1])
    put("ang", [0, 1, 2], [-1, 1, -1])          # angular velocity is a pseudo-vector
    for key in ("qpos", "qvel", "last_action"):
        put(key, jp, js)
    put("command", [0, 1], [1, -1])             # speed stays, turn rate flips
    if "clock_sin" in lay:
        put("clock_sin", lp, 1)
        put("clock_cos", lp, 1)
    return perm, sign, jp, js


def mirror(x, perm, sign):
    return x[..., perm] * sign


class SymmetricPPO(PPO):
    def __init__(self, *args, sym_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sym_weight = sym_weight
        self.mirror_maps = None

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["mirror_maps"]

    def set_mirror(self, maps):
        op, os_, ap, as_ = maps
        dev = self.device
        self.mirror_maps = (th.as_tensor(op, device=dev), th.as_tensor(os_, dtype=th.float32, device=dev),
                            th.as_tensor(ap, device=dev), th.as_tensor(as_, dtype=th.float32, device=dev))

    def _mirror_normalized_obs(self, obs):
        """Rollout observations are VecNormalize-normalised: un-normalise, mirror, normalise again."""
        op, os_, _, _ = self.mirror_maps
        venv = self.get_env()
        rms = getattr(venv, "obs_rms", None)
        if rms is None:
            return obs[:, op] * os_
        mean = th.as_tensor(rms.mean, dtype=th.float32, device=self.device)
        std = th.sqrt(th.as_tensor(rms.var, dtype=th.float32, device=self.device) + venv.epsilon)
        raw = obs * std + mean
        return th.clamp((raw[:, op] * os_ - mean) / std, -venv.clip_obs, venv.clip_obs)

    def train(self):
        super().train()
        if self.mirror_maps is None or self.sym_weight <= 0:
            return
        _, _, ap, as_ = self.mirror_maps
        losses = []
        for data in self.rollout_buffer.get(self.batch_size):
            obs = data.observations
            mu = self.policy.get_distribution(obs).distribution.mean
            mu_m = self.policy.get_distribution(self._mirror_normalized_obs(obs)).distribution.mean
            loss = self.sym_weight * th.mean((mu_m - mu[:, ap] * as_) ** 2)
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(loss.item())
        self.logger.record("train/symmetry_loss", float(np.mean(losses)))
