"""Gymnasium environment: a creature's MuJoCo body learns to walk (the "locomotion brain").

The brain gets a command (forward speed, turn rate) - later produced by the navigation brain in
CreatureEnv - and sets joint position targets for every leg joint at 50 Hz. It is rewarded for
tracking the command while staying upright, keeping feet planted and moving smoothly.
Command limits come from sim/species.py, so the navigation brain only asks for what is walkable.
"""
import math

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from sim.bodies import BODIES, build_mjcf
from sim.species import SPECIES

FRAME_SKIP = 4                  # physics 200 Hz -> control 50 Hz
CONTROL_DT = 0.005 * FRAME_SKIP

# Reward weights
W_LIN = 2.0
W_ANG = 0.5
W_VZ = -0.5
W_ROLL_PITCH_RATE = -0.05
W_TILT = -1.0
W_ACTION_RATE = -0.02
W_TORQUE = -0.02                # on (torque / max_torque)^2, summed
W_FOOT_SLIP = -0.1              # on foot speed^2 (capped at 1) while in contact, summed
W_BODY_CONTACT = -1.0           # torso geoms (thorax, head, abdomen, ...) touching the floor
W_JOINT_LIMIT = -0.2            # fraction of joints within 5 deg of a limit


class LocomotionEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / CONTROL_DT)}

    def __init__(self, species="myrranox", episode_seconds=20.0, command_seconds=(2.0, 5.0),
                 stand_prob=0.15, render_mode=None, width=640, height=400):
        self.species = SPECIES[species]
        self.body = BODIES[species]()
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.body))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.episode_steps = int(round(episode_seconds / CONTROL_DT))
        self.command_seconds = command_seconds
        self.stand_prob = stand_prob

        m = self.model
        self.nj = m.nu
        self.joint_qpos = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(self.nj)])
        self.joint_qvel = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(self.nj)])
        jids = m.actuator_trnid[:, 0]
        self.joint_lo, self.joint_hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]

        self.foot_geoms = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in self.body.foot_names])
        self.foot_sites = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in self.body.foot_names])
        self.floor_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.torso_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.torso_geoms = set(np.flatnonzero(m.geom_bodyid == self.torso_body).tolist())

        mujoco.mj_forward(m, self.data)
        self.stand_z = float(self.data.qpos[2])
        self.init_qpos = self.data.qpos.copy()

        obs_size = 3 + 3 + 3 + 1 + 3 * self.nj + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nj,), dtype=np.float32)
        self._renderer = None
        self._size = (height, width)
        self.penalty_scale = 1.0        # ramped up from ~0.2 during training (train_locomotion.py)

    # ------------------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        mujoco.mj_resetData(self.model, self.data)
        qpos = self.init_qpos.copy()
        yaw = options["yaw"] if options and "yaw" in options else rng.uniform(-math.pi, math.pi)
        qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        qpos[self.joint_qpos] += rng.uniform(-0.05, 0.05, size=self.nj)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        # Let the body settle onto the floor with the rest pose held.
        for _ in range(40):
            self.data.ctrl[:] = qpos[self.joint_qpos]
            mujoco.mj_step(self.model, self.data)

        self.t = 0
        self.last_action = np.zeros(self.nj)
        self.prev_foot_pos = self.data.site_xpos[self.foot_sites].copy()
        self.command = np.zeros(2)
        self.command_steps_left = 0
        if options and "command" in options:
            self.set_command(*options["command"], hold=True)
        else:
            self._resample_command()
        return self._observe(), {}

    def set_command(self, speed, turn_rate, hold=True):
        """Forward speed (m/s) and turn rate (rad/s) for the brain to follow; clipped to species limits."""
        sp = self.species
        self.command = np.array([np.clip(speed, -sp.max_backward_speed, sp.max_speed),
                                 np.clip(turn_rate, -sp.max_turn_rate, sp.max_turn_rate)])
        self.command_steps_left = 10 ** 9 if hold else self.command_steps_left

    def _resample_command(self):
        rng, sp = self.np_random, self.species
        if rng.uniform() < self.stand_prob:
            v, w = 0.0, 0.0
        else:
            v = rng.uniform(-sp.max_backward_speed, sp.max_speed)
            w = rng.uniform(-sp.max_turn_rate, sp.max_turn_rate) * (rng.uniform() < 0.7)
            if not sp.turn_in_place and abs(v) < sp.min_turn_speed:
                w *= abs(v) / sp.min_turn_speed
        self.command = np.array([v, w])
        self.command_steps_left = int(rng.uniform(*self.command_seconds) / CONTROL_DT)

    # ------------------------------------------------------------------------------------------
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        m, d = self.model, self.data
        target = np.clip(self.init_qpos[self.joint_qpos] + a * self.body.action_scale, self.joint_lo, self.joint_hi)
        d.ctrl[:] = target
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(m, d)

        # Body-frame velocities and gravity
        R = d.xmat[self.torso_body].reshape(3, 3)          # body -> world
        lin = R.T @ d.qvel[0:3]
        ang = d.qvel[3:6]                                   # free joint angular velocity is in body frame
        grav = R.T @ np.array([0, 0, -1.0])

        # Contacts
        foot_contact = np.zeros(len(self.foot_geoms), dtype=bool)
        body_contact = False
        for i in range(d.ncon):
            c = d.contact[i]
            if self.floor_geom not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.floor_geom else c.geom1
            hit = np.flatnonzero(self.foot_geoms == other)
            if len(hit):
                foot_contact[hit[0]] = True
            elif other in self.torso_geoms:
                body_contact = True

        foot_pos = d.site_xpos[self.foot_sites]
        foot_vel = (foot_pos - self.prev_foot_pos)[:, :2] / CONTROL_DT
        self.prev_foot_pos = foot_pos.copy()

        sp = self.species
        sig_v = 0.5 * sp.max_speed      # wide enough that moving a little already pays off
        sig_w = 0.5 * sp.max_turn_rate
        lin_err = (lin[0] - self.command[0]) ** 2 + lin[1] ** 2
        ang_err = (ang[2] - self.command[1]) ** 2
        torque = d.actuator_force / self.body.max_torque
        qj = d.qpos[self.joint_qpos]
        near_limit = np.mean((qj < self.joint_lo + 0.087) | (qj > self.joint_hi - 0.087))
        terms = {
            "lin": W_LIN * math.exp(-lin_err / sig_v ** 2),
            "ang": W_ANG * math.exp(-ang_err / sig_w ** 2),
            "vz": W_VZ * lin[2] ** 2,
            "roll_pitch": W_ROLL_PITCH_RATE * (ang[0] ** 2 + ang[1] ** 2),
            "tilt": W_TILT * (grav[0] ** 2 + grav[1] ** 2),
            "action_rate": W_ACTION_RATE * float(np.sum((a - self.last_action) ** 2)),
            "torque": W_TORQUE * float(np.sum(torque ** 2)) / self.nj * 4,
            "foot_slip": W_FOOT_SLIP * float(np.sum(np.minimum((foot_vel ** 2).sum(axis=1), 1.0) * foot_contact)),
            "body_contact": W_BODY_CONTACT * body_contact,
            "joint_limit": W_JOINT_LIMIT * near_limit,
        }
        reward = terms["lin"] + terms["ang"] + self.penalty_scale * float(
            sum(v for k, v in terms.items() if k not in ("lin", "ang")))
        self.last_action = a

        self.t += 1
        self.command_steps_left -= 1
        if self.command_steps_left <= 0:
            self._resample_command()

        fallen = grav[2] > -0.5 or d.qpos[2] < self.stand_z - 0.5 * self._leg_height()
        terminated = bool(fallen)
        if terminated:
            reward -= 10.0
        truncated = self.t >= self.episode_steps
        info = {"terms": terms, "command": self.command.copy(), "speed": float(lin[0]), "turn_rate": float(ang[2]),
                "foot_contact": foot_contact}
        return self._observe(lin, ang, grav), reward, terminated, truncated, info

    def set_penalty_scale(self, scale):
        self.penalty_scale = float(scale)

    def _leg_height(self):
        return float(np.mean([leg.points[0, 2] for leg in self.body.legs]))

    def _observe(self, lin=None, ang=None, grav=None):
        d = self.data
        if lin is None:
            R = d.xmat[self.torso_body].reshape(3, 3)
            lin, ang, grav = R.T @ d.qvel[0:3], d.qvel[3:6], R.T @ np.array([0, 0, -1.0])
        sp = self.species
        obs = np.concatenate([
            grav, lin, ang * 0.25, [(d.qpos[2] - self.stand_z) * 5],
            d.qpos[self.joint_qpos] - self.init_qpos[self.joint_qpos],
            d.qvel[self.joint_qvel] * 0.05,
            self.last_action,
            [self.command[0] / sp.max_speed, self.command[1] / sp.max_turn_rate],
        ])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------------------------------
    def pose(self):
        """Current body pose for recording: position, quaternion (w, x, y, z), joint angles, foot positions."""
        d = self.data
        mujoco.mj_kinematics(self.model, d)   # site positions after mj_step are one substep old
        return {"pos": d.qpos[0:3].copy(), "quat": d.qpos[3:7].copy(),
                "joints": d.qpos[self.joint_qpos].copy(), "feet": d.site_xpos[self.foot_sites].copy()}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self._size)
            self._cam = mujoco.MjvCamera()
            self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._cam.trackbodyid = self.torso_body
            self._cam.distance = 3.0 * max(1.0, self.body.mass / 20) ** 0.33
            self._cam.elevation = -20
            self._cam.azimuth = 135
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
