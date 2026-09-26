"""Gymnasium environment: a creature's MuJoCo body learns to walk (the "locomotion brain").

The brain gets a command (forward speed, turn rate) - later produced by the navigation brain in
CreatureEnv - and sets joint position targets for every leg joint at 50 Hz. It is rewarded for
tracking the command while staying upright, keeping feet planted and moving smoothly.
Command limits come from sim/species.py, so the navigation brain only asks for what is walkable.

version=1  original brains (models/*_legs_v1): command tracking + generic penalties.
version=2  natural gait: a gait clock per leg is added to the observation and the brain is rewarded
           for following the species' gait (sim/gaits.py: insect tripod / quadruped walk -> trot),
           lifting swinging feet, proper step length in the air, low energy and smooth joints.
           Trained with the mirror-symmetry loss (sim/symmetry.py). More turn / backwards commands.

Shared world: pass model/data/prefix to drive one creature inside a bigger MuJoCo world
(sim/physics_world.py). Then call apply_action(), step the world, and after_physics() - step() does
exactly that for the single-creature training world.
"""
import math

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from sim import gaits
from sim.bodies import BODIES, build_mjcf
from sim.species import SPECIES

FRAME_SKIP = 4                  # physics 200 Hz -> control 50 Hz
CONTROL_DT = 0.005 * FRAME_SKIP

# Reward weights. "lin"/"ang" and the v2 gait terms are task rewards; everything else is a penalty that
# train_locomotion.py ramps in with the penalty curriculum.
WEIGHTS = {
    1: dict(lin=2.0, ang=0.5, vz=-0.5, roll_pitch=-0.05, tilt=-1.0, action_rate=-0.02, torque=-0.02,
            foot_slip=-0.1, body_contact=-1.0, joint_limit=-0.2),
    2: dict(lin=2.0, ang=1.0, vz=-0.5, roll_pitch=-0.05, tilt=-1.0, action_rate=-0.05, torque=-0.02,
            foot_slip=-0.3, body_contact=-1.0, joint_limit=-0.2,
            gait=1.0,           # fraction of legs whose contact matches the gait clock
            clearance=0.3,      # swing feet lifted to the target height
            air_time=0.5,       # per touchdown: +0.5 for a full swing, -0.5 for a tap
            energy=-0.1,        # mechanical power |torque * joint speed| / (m g v_walk)
            joint_acc=-0.05),   # mean (joint acceleration / 500 rad/s^2)^2
}
TASK_TERMS = ("lin", "ang", "gait", "clearance", "air_time")


class LocomotionEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / CONTROL_DT)}

    def __init__(self, species="myrranox", episode_seconds=20.0, command_seconds=(2.0, 5.0),
                 stand_prob=0.15, render_mode=None, width=640, height=400, version=1,
                 model=None, data=None, prefix="", passive=False):
        self.species_name = species
        self.species = SPECIES[species]
        self.body = BODIES[species](passive=passive)       # passive=True: tail/abdomen/head/arms on springs
        self.version = version
        self.w = WEIGHTS[version]
        self.shared = model is not None
        self.prefix = prefix
        self.model = model if self.shared else mujoco.MjModel.from_xml_string(build_mjcf(self.body))
        self.data = data if self.shared else mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.episode_steps = int(round(episode_seconds / CONTROL_DT))
        self.command_seconds = command_seconds
        self.stand_prob = stand_prob

        m = self.model

        def nid(kind, name):
            i = mujoco.mj_name2id(m, kind, prefix + name)
            assert i >= 0, f"{prefix + name} not in model"
            return i

        self.act_ids = np.array([nid(mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in self.body.joint_names])
        self.nj = len(self.act_ids)
        jids = m.actuator_trnid[self.act_ids, 0]
        self.joint_qpos = m.jnt_qposadr[jids]
        self.joint_qvel = m.jnt_dofadr[jids]
        self.joint_lo, self.joint_hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
        root = nid(mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.q0, self.v0 = int(m.jnt_qposadr[root]), int(m.jnt_dofadr[root])     # free joint: pos 3 + quat 4

        self.leg_names = [leg.name for leg in self.body.legs]
        self.nlegs = len(self.leg_names)
        self.foot_geoms = np.array([nid(mujoco.mjtObj.mjOBJ_GEOM, n) for n in self.body.foot_names])
        self.foot_sites = np.array([nid(mujoco.mjtObj.mjOBJ_SITE, n) for n in self.body.foot_names])
        self.foot_radius = m.geom_size[self.foot_geoms, 0]
        self.floor_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.torso_body = nid(mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.torso_geoms = set(np.flatnonzero(m.geom_bodyid == self.torso_body).tolist())
        self.passive_qpos = np.array([m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, n)]
                                      for n in self.body.passive_joint_names], dtype=int)

        mujoco.mj_forward(m, self.data)
        self.stand_z = float(self.data.qpos[self.q0 + 2])
        self.init_qpos = self.data.qpos.copy()
        self.rest_joints = self.init_qpos[self.joint_qpos].copy()
        self.power_ref = self.body.mass * 9.81 * self.species.walk_speed

        obs_size = 3 + 3 + 3 + 1 + 3 * self.nj + 2 + (2 * self.nlegs + 1 if version >= 2 else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nj,), dtype=np.float32)
        self._renderer = None
        self._size = (height, width)
        self.penalty_scale = 1.0        # ramped up from ~0.2 during training (train_locomotion.py)

    # ------------------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        yaw = options["yaw"] if options and "yaw" in options else rng.uniform(-math.pi, math.pi)
        noise = rng.uniform(-0.05, 0.05, size=self.nj)
        q0, v0 = self.q0, self.v0
        if self.shared:
            # Only this creature is re-placed (at options["pos"]); the world is stepped by its owner.
            d = self.data
            x, y = options.get("pos", (0.0, 0.0)) if options else (0.0, 0.0)
            d.qpos[q0:q0 + 7] = [x, y, self.init_qpos[q0 + 2], math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            d.qpos[self.joint_qpos] = self.rest_joints + noise
            d.qvel[v0:v0 + 6] = 0.0
            d.qvel[self.joint_qvel] = 0.0
            d.ctrl[self.act_ids] = d.qpos[self.joint_qpos]
            mujoco.mj_forward(self.model, d)
        else:
            mujoco.mj_resetData(self.model, self.data)
            qpos = self.init_qpos.copy()
            qpos[q0 + 3:q0 + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            qpos[self.joint_qpos] += noise
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            # Let the body settle onto the floor with the rest pose held.
            for _ in range(40):
                self.data.ctrl[self.act_ids] = qpos[self.joint_qpos]
                mujoco.mj_step(self.model, self.data)

        self.t = 0
        self.last_action = np.zeros(self.nj)
        self.prev_foot_pos = self.data.site_xpos[self.foot_sites].copy()
        self.prev_qvel = self.data.qvel[self.joint_qvel].copy()
        self.phase = float(rng.uniform())
        self.air_time = np.zeros(self.nlegs)
        self.prev_contact = np.ones(self.nlegs, dtype=bool)
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
        u = rng.uniform()
        if u < self.stand_prob:
            v, w = 0.0, 0.0
        elif self.version >= 2 and sp.turn_in_place and u < self.stand_prob + 0.10:
            v, w = 0.0, rng.uniform(-sp.max_turn_rate, sp.max_turn_rate)            # turn on the spot
        elif self.version >= 2 and sp.max_backward_speed > 0 and u < self.stand_prob + 0.20:
            v, w = -rng.uniform(0.1, sp.max_backward_speed), 0.0                      # walk backwards
        else:
            v = rng.uniform(-sp.max_backward_speed, sp.max_speed) if self.version == 1 else rng.uniform(0.1, sp.max_speed)
            w = rng.uniform(-sp.max_turn_rate, sp.max_turn_rate) * (rng.uniform() < (0.7 if self.version == 1 else 0.8))
            if not sp.turn_in_place and abs(v) < sp.min_turn_speed:
                w *= abs(v) / sp.min_turn_speed
        self.command = np.array([v, w])
        self.command_steps_left = int(rng.uniform(*self.command_seconds) / CONTROL_DT)

    # ------------------------------------------------------------------------------------------
    def gait_targets(self):
        """(per-leg phase in [0,1), desired stance mask, duty, frequency Hz, standing?) for the current command."""
        v, w = self.command
        offsets, duty = gaits.pattern(self.species_name, self.leg_names, v)
        f = gaits.frequency(self.species_name, v, w, self.species.max_turn_rate)
        standing = gaits.is_standing(v, w)
        leg_phase = (self.phase + offsets) % 1.0
        stance = np.ones(self.nlegs, dtype=bool) if standing else leg_phase < duty
        return leg_phase, stance, duty, f, standing

    def step(self, action):
        a = self.apply_action(action)
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        return self.after_physics(a)

    def apply_action(self, action):
        """Set this creature's joint targets; returns the clipped action (pass it to after_physics)."""
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.data.ctrl[self.act_ids] = np.clip(self.rest_joints + a * self.body.action_scale,
                                               self.joint_lo, self.joint_hi)
        return a

    def after_physics(self, a):
        """Read the body after FRAME_SKIP physics steps: reward terms, observation, termination."""
        m, d = self.model, self.data
        v0 = self.v0
        # Body-frame velocities and gravity
        R = d.xmat[self.torso_body].reshape(3, 3)          # body -> world
        lin = R.T @ d.qvel[v0:v0 + 3]
        ang = d.qvel[v0 + 3:v0 + 6]                         # free joint angular velocity is in body frame
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
        qvel = d.qvel[self.joint_qvel]
        qacc = (qvel - self.prev_qvel) / CONTROL_DT
        self.prev_qvel = qvel.copy()
        power = float(np.sum(np.abs(d.actuator_force[self.act_ids] * qvel)))        # W, mechanical

        sp, w = self.species, self.w
        sig_v = 0.5 * sp.max_speed      # wide enough that moving a little already pays off
        sig_w = 0.5 * sp.max_turn_rate
        lin_err = (lin[0] - self.command[0]) ** 2 + lin[1] ** 2
        ang_err = (ang[2] - self.command[1]) ** 2
        torque = d.actuator_force[self.act_ids] / self.body.max_torque
        qj = d.qpos[self.joint_qpos]
        near_limit = np.mean((qj < self.joint_lo + 0.087) | (qj > self.joint_hi - 0.087))
        terms = {
            "lin": w["lin"] * math.exp(-lin_err / sig_v ** 2),
            "ang": w["ang"] * math.exp(-ang_err / sig_w ** 2),
            "vz": w["vz"] * lin[2] ** 2,
            "roll_pitch": w["roll_pitch"] * (ang[0] ** 2 + ang[1] ** 2),
            "tilt": w["tilt"] * (grav[0] ** 2 + grav[1] ** 2),
            "action_rate": w["action_rate"] * float(np.sum((a - self.last_action) ** 2)),
            "torque": w["torque"] * float(np.sum(torque ** 2)) / self.nj * 4,
            "foot_slip": w["foot_slip"] * float(np.sum(np.minimum((foot_vel ** 2).sum(axis=1), 1.0) * foot_contact)),
            "body_contact": w["body_contact"] * body_contact,
            "joint_limit": w["joint_limit"] * near_limit,
        }

        leg_phase = stance = None
        if self.version >= 2:
            leg_phase, stance, duty, f, standing = self.gait_targets()
            swing = ~stance
            # Stance legs down + swing legs up, scored separately: perfect gait 1, standing still or random 0
            # (a plain "fraction of legs matching" would give standing still 55-65 % for free).
            stance_ok = float(np.mean(foot_contact[stance])) if stance.any() else 1.0
            swing_ok = float(np.mean(~foot_contact[swing])) if swing.any() else 1.0
            terms["gait"] = w["gait"] * (stance_ok + swing_ok - 1.0)
            if swing.any():
                height = foot_pos[:, 2] - self.foot_radius
                target_h = gaits.FOOT_CLEARANCE[self.species_name]
                terms["clearance"] = w["clearance"] * float(np.mean(np.clip(height[swing] / target_h, 0, 1)))
            else:
                terms["clearance"] = 0.0
            # Air time: reward full swings, punish taps (only while walking)
            touchdown = foot_contact & ~self.prev_contact
            t_swing = (1 - duty) / f
            if not standing and touchdown.any():
                terms["air_time"] = w["air_time"] * float(np.sum(np.minimum(self.air_time[touchdown] / t_swing, 1.0) - 0.5))
            else:
                terms["air_time"] = 0.0
            self.air_time = np.where(foot_contact, 0.0, self.air_time + CONTROL_DT)
            self.prev_contact = foot_contact.copy()
            terms["energy"] = w["energy"] * power / self.power_ref
            terms["joint_acc"] = w["joint_acc"] * float(np.mean((qacc / 500.0) ** 2))
            if not standing:
                self.phase = (self.phase + f * CONTROL_DT) % 1.0

        reward = sum(v for k, v in terms.items() if k in TASK_TERMS) + self.penalty_scale * float(
            sum(v for k, v in terms.items() if k not in TASK_TERMS))
        self.last_action = a

        self.t += 1
        self.command_steps_left -= 1
        if self.command_steps_left <= 0:
            self._resample_command()

        fallen = grav[2] > -0.5 or d.qpos[self.q0 + 2] < self.stand_z - 0.5 * self._leg_height()
        terminated = bool(fallen)
        if terminated:
            reward -= 10.0
        truncated = self.t >= self.episode_steps
        info = {"terms": terms, "command": self.command.copy(), "speed": float(lin[0]), "turn_rate": float(ang[2]),
                "foot_contact": foot_contact, "power": power, "desired_stance": stance}
        return self._observe(lin, ang, grav), float(reward), terminated, truncated, info

    def set_penalty_scale(self, scale):
        self.penalty_scale = float(scale)

    def _leg_height(self):
        return float(np.mean([leg.points[0, 2] for leg in self.body.legs]))

    def obs_layout(self):
        """Slices of the observation vector (used by sim/symmetry.py)."""
        nj, i = self.nj, 0
        out = {}
        for name, n in (("grav", 3), ("lin", 3), ("ang", 3), ("height", 1), ("qpos", nj), ("qvel", nj),
                        ("last_action", nj), ("command", 2)):
            out[name] = slice(i, i + n)
            i += n
        if self.version >= 2:
            out["clock_sin"] = slice(i, i + self.nlegs)
            out["clock_cos"] = slice(i + self.nlegs, i + 2 * self.nlegs)
            out["standing"] = slice(i + 2 * self.nlegs, i + 2 * self.nlegs + 1)
        return out

    def _observe(self, lin=None, ang=None, grav=None):
        d = self.data
        if lin is None:
            R = d.xmat[self.torso_body].reshape(3, 3)
            v0 = self.v0
            lin, ang, grav = R.T @ d.qvel[v0:v0 + 3], d.qvel[v0 + 3:v0 + 6], R.T @ np.array([0, 0, -1.0])
        sp = self.species
        parts = [
            grav, lin, ang * 0.25, [(d.qpos[self.q0 + 2] - self.stand_z) * 5],
            d.qpos[self.joint_qpos] - self.rest_joints,
            d.qvel[self.joint_qvel] * 0.05,
            self.last_action,
            [self.command[0] / sp.max_speed, self.command[1] / sp.max_turn_rate],
        ]
        if self.version >= 2:
            leg_phase, _, _, _, standing = self.gait_targets()
            parts += [np.sin(2 * np.pi * leg_phase), np.cos(2 * np.pi * leg_phase), [1.0 if standing else 0.0]]
        return np.concatenate(parts).astype(np.float32)

    # ------------------------------------------------------------------------------------------
    def pose(self):
        """Current body pose for recording: position, quaternion (w, x, y, z), joint angles, foot positions."""
        d = self.data
        mujoco.mj_kinematics(self.model, d)   # site positions after mj_step are one substep old
        q0 = self.q0
        return {"pos": d.qpos[q0:q0 + 3].copy(), "quat": d.qpos[q0 + 3:q0 + 7].copy(),
                "passive": d.qpos[self.passive_qpos].copy(),
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
