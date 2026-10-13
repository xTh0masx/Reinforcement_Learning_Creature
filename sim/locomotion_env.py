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
version=3  "alive" bodies (stage 2): as version 2, plus spring body parts (tail, arms) and actuated spine / neck /
           head / abdomen (sim/bodies.py active=True); rewards a steady, level head (the "panther gaze").
version=4  precise and fluid (28 Sep): the brain's joint targets pass a low-pass filter (no jumps at 50 Hz),
           speed / turn tracking gets stricter during training (set_track_scale), turning weighs as much as
           speed, soft foot landings, commands ramp like the navigation brain's. Body per species (ACTIVE):
           DC keeps the v3 active spine / neck / head, the Myrranox the v2 body with passive parts (its v3 failed).
           Head style terms are paid only in proportion to how well the command is tracked, so standing still
           can no longer dodge them (walking log lesson 8). On spot-turn commands the speed reward is scaled by the
           turn tracking, so standing still no longer earns it.
version=5  natural style (29 Sep): as version 4, plus a leg rhythm per species from the gait clock ("style"
           reward): each leg reaches forward in the swing and pushes back in stance by exactly the stride the
           commanded speed needs, and folds its knee mid-swing (insect: ~45 deg, panther: ~35 deg) - measured as
           swing around each joint's running mean, so the posture stays free. The knee is
           mostly exempt from the energy / joint-acceleration penalties (in v2-v4 they froze the Myrranox knee).
           DC: long, slow panther strides (lower step frequency), hind paw placed into the front paw's print
           ("register"), tail and extra arms damped harder (less jelly), steady trunk (STEADY_TRUNK_V5).
version=6  DC "six-limbed panther" (30 Sep): body with the extra arms as actuated limbs, legs +-8 deg sideways, mesh-
           fitted segment radii (sim/bodies.py dc(arms=True)). Walk / trot on four legs with the arms folded against
           the chest ("arm_tuck"), from 1.4 m/s the arms join the run (sim/gaits.py). Ankles / wrists join the leg
           rhythm (paws flat in stance, no flicking). Stricter rhythm and fluency weights (gait, air time, joint
           acceleration, action rate, "surge" = speeding up / slowing down within a stride).
version=7  as version 6, but the legs follow a built-in step generator (owner, 30 Sep: "let the DC walk directly"):
           joint target = rest pose + gait-clock reference (reach / push / knee and paw fold, folded arms while
           walking) + the brain's correction (CPG_RESIDUAL rad per action unit). The DC steps on the beat from the
           first step; the brain learns balance, steering and fine adjustments. (Experiment: the generator alone
           does not balance the DC.)
version=8  DC on the mesh-based body (owner, 30 Sep: "enhance the meshes of DC as highly detailed as possible"): every
           part is a convex hull cut from the real DC mesh (sim/dc_hulls.py), paws with claws are the feet, the
           mass is distributed by volume. Rules as version 6 (no step generator), plus no unnecessary movement:
           energy weighs double and joint movement while commanded to stand ("fidget") is penalised. Sprint up to
           SPRINT_SPEED; every command change ramps (speed up at max_accel, slow down gently at SLOW_DOWN).
version=9  both creatures on their mesh-based bodies (sim/hulls.py), new reward structure (owner, 13 Oct):
               reward = tracking x (0.4 + 0.6 x animal-likeness) x quality      (falls end the episode, -10)
           tracking: speed and turn as commanded (0..1) - no points without it, so standing still never pays while
           walking is commanded. animal-likeness: gait beat, leg rhythm and each animal's traits (0..1). quality:
           exp(-sum of normalised penalties) (0..1) - jerks, energy, sliding feet, a rocking trunk, ... only shrink
           the points, they can never make standing still better than walking (the trap of v3, v8).
version=10 Myrranox: as version 9, plus the step generator (as in v7, owner 13 Oct: "the Myrranox is not using the
           legs correctly"): joint target = rest + standing pose + tripod rhythm from the gait clock (hip sweep, knee
           fold) + the brain's correction (CPG_RESIDUAL_V10 x action). A hexapod tripod is statically stable, so the
           rhythm alone already walks (0.37 m/s, no falls, tested); the brain learns balance, steering and details.
version=11 DC: as version 9 on the "lean" body (sim/bodies.py dc(lean=True)): big-cat mass distribution instead of mass
           by volume (it was head-heavy, centre of mass almost over the front paws) and rounded paw pads as feet.

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
WEIGHTS[3] = dict(WEIGHTS[2],
                  head_steady=-0.05,    # |head angular velocity - commanded turn|^2 (rad/s): a calm head
                  head_level=-2.0)      # 1 - cos(head tilt): keep the gaze level
WEIGHTS[4] = dict(WEIGHTS[3],
                  ang=1.5,              # turning counts as much as speed (lesson 4: turning lagged)
                  landing=-0.05)        # per touchdown: (foot vertical speed, m/s)^2 - soft steps
WEIGHTS[5] = dict(WEIGHTS[4],
                  style=0.8,            # leg rhythm: exp(-mean(joint error)^2 / 0.35^2) against the gait-clock reference
                  register=0.3)         # DC, per hind touchdown: hind paw near the same-side front paw's last print
WEIGHTS[6] = dict(WEIGHTS[5],
                  arm_tuck=0.4,         # arms not in the gait: claws folded near the chest
                  # stricter rhythm and fluency (owner, 30 Sep): v5c walked with 15 % step-timing variation
                  gait=2.0,             # contacts on the beat of the gait clock (v2-v5: 1.0)
                  air_time=1.0,         # full swings, no taps / double steps (v2-v5: 0.5)
                  joint_acc=-0.15,      # (v2-v5: -0.05)
                  action_rate=-0.1,     # (v2-v5: -0.05)
                  surge=-0.2)           # ((forward acceleration) / 2 m/s^2)^2: no speeding up and slowing down per stride
TASK_TERMS = ("lin", "ang", "gait", "clearance", "air_time", "style", "register", "arm_tuck")
ARM_TUCK = (0.80, 0.20, 0.42)   # version 6: folded claw position in the torso frame (y mirrored per side)
ANKLE_FOLD = 0.3                # version 6: ankle / wrist fold in mid-swing (rad); flat (no rhythm) in stance
# version 9: natural standing pose = the neutral position the brain's commands work around (13 Oct: on the mesh bodies
# the brains pushed joints to their limits just to hold the right stance, leaving no range for walking)
# m the body is raised per leg pair (feet stay put): the Myrranox carries its heavy abdomen (hind legs brace higher,
# else it tips nose-up and the front feet lift; tested 13 Oct: level, all six feet down, abdomen clear)
STAND_TALLER = {"myrranox": {"front": 0.06, "mid": 0.06, "back": 0.12},
                "dc": {"front": 0.03, "hind": 0.0}}       # DC: trunk level (nose-down 5.5 -> 4.1 deg), arms folded
WEIGHTS[7] = dict(WEIGHTS[6])
CPG_RESIDUAL_V10 = 0.45      # version 10: rad per action unit for the brain's correction on top of the tripod rhythm
WEIGHTS[8] = dict(WEIGHTS[6],
                  # no unnecessary movement (owner, 30 Sep)
                  energy=-0.2,          # (v2-v6: -0.1) every movement that does not help costs twice as much
                  fidget=-0.5,          # standing command: mean (joint speed)^2 - standing means standing still
                  arm_ground=-1.0)      # per folded arm touching the ground (owner: no handstand on front legs + arms)
MAX_TILT_V8 = 25.0           # deg: a trunk tilted further counts as a fall (the "handstand" had ~20 deg nose down)
WEIGHTS[9] = dict(WEIGHTS[8])
WEIGHTS[10] = dict(WEIGHTS[9])
WEIGHTS[11] = WEIGHTS[12] = dict(WEIGHTS[9])  # (the term values are still computed for analysis; v9 scores with V9_QUALITY)
# version 9 quality factor: exp(-penalty_scale * sum(c * p)); p = normalised penalty, c = how much it shrinks the points
V9_QUALITY = dict(
    energy=0.08,        # mechanical power / (m g v_walk) (knee mostly exempt)
    jerk=1.0,           # mean (joint acceleration / 500 rad/s^2)^2
    action_rate=1.0,    # mean (change of the smoothed joint targets)^2 - measured after the filter, so the brain's
                        # exploration noise does not wipe out the points (13 Oct: quality 0.06 in training)
    slip=1.0,           # feet sliding while on the ground (x3 on 13 Oct: the Myrranox scrubbed its middle legs)
    trunk_rock=0.3,     # trunk roll / pitch rate^2
    bounce=3.0,         # vertical trunk speed^2 (x3 on 13 Oct: 4 cm bob cost only ~5 % of the points)
    tilt=6.0,           # static trunk tilt (x3 on 13 Oct: 11 deg tilt cost only ~5 %)
    surge=0.2,          # speeding up / slowing down within a stride
    landing=0.05,       # hard touchdowns (vertical foot speed^2)
    taps=1.0,           # touchdowns after less than half a swing (x3 on 13 Oct: the DC's hind legs tripped at 4 steps/s)
    head_steady=0.02,   # DC: head rotation not explained by the turn
    head_level=2.0,     # DC: head tilt
    fidget=0.5,         # joint movement while commanded to stand
    joint_limit=0.5,    # joints near their limits
    belly=1.0,          # Myrranox: abdomen on the ground (an ant carries it)
    arm_ground=1.0,     # DC: folded arm on the ground at walking speed
    body_contact=2.0,   # trunk on the ground
    track=1.0,          # DC: paws on the ground wider than the standing pose ((excess / 5 cm)^2)
    extra_step=0.0,     # per touchdown less than 0.6 gait cycles after the same leg's last one. Switched off: at 1.0
                        # the DC stopped stepping with its hind legs altogether and then stopped walking (13 Oct)
    idle_leg=0.5)       # per leg that has not touched the ground for 2 step cycles while walking (13 Oct: the
                        # Myrranox carried both front legs in the air and pushed like a wheelbarrow)
# version 12 (DC): the owner's reference movement, a wolf walk cycle (13 Oct). Seen in it: each leg steps once per cycle
# (front and hind at the same rate), 2-3 paws always on the ground, paws lifted low with the front wrist folding under in
# the swing, back level, head low and forward. Added to the likeness (0..1 scores) and the quality costs.
V12_QUALITY = dict(V9_QUALITY,
    support=1.0,        # walking with fewer than two paws on the ground (a wolf's walk never leaves the ground)
    high_step=0.5)      # swing paw higher than WOLF_LIFT ((excess / 5 cm)^2)
WOLF_LIFT = 0.08             # m: highest paw lift in the wolf's walk scaled to the DC (~12 % of leg height)
WOLF_HEAD_DZ = 0.75          # m: head (neck_s1 origin) above the torso origin (which is at floor level for the DC);
                             # the DC holds it at 0.87 m - the wolf carries its head lower, about at its back's height
CPG_RESIDUAL = 0.35          # version 7: rad per action unit for the brain's correction on top of the step generator
ACTIVE = {7: {"myrranox": False, "dc": True}, 8: {"myrranox": False, "dc": True}, 9: {"myrranox": False, "dc": True},
          10: {"myrranox": False, "dc": True}, 11: {"myrranox": False, "dc": True}, 12: {"myrranox": False, "dc": True}, 3: {"myrranox": True, "dc": True}, 4: {"myrranox": False, "dc": True},
          5: {"myrranox": False, "dc": True}, 6: {"myrranox": False, "dc": True}}   # actuated spine/neck/head/abdomen
# version 5 leg style per species: (sweep joint index, knee joint index, knee fold rad, max sweep rad)
STYLE = {"myrranox": (0, 2, 0.8, 0.45), "dc": (1, 2, 0.6, 0.7)}
KNEE_EFFORT = 0.25           # version 5: the knee pays a quarter of the energy / joint-acceleration penalty
PASSIVE_DAMPING_V5 = {"dc": {"tail": 0.5, "arm_L": 0.5, "arm_R": 0.5}}   # damping ratio (v1-v4: 0.15 / 0.25)
# version 5 steady trunk (owner, 29 Sep: "DC does not look stable"): the long strides made the torso rock and bounce;
# these weights replace roll_pitch / vz and are paid in proportion to the command tracking (like the head terms)
STEADY_TRUNK_V5 = {"dc": dict(roll_pitch=-0.5, vz=-1.5)}
ACTION_FILTER = 0.5          # version 4: target += 0.5 * (action - target) every 20 ms, about 5.5 Hz cut-off
COMMAND_RAMP_S = (1.5, 1.0)  # version 4: seconds to ramp from 0 to max speed / max turn rate (half the commands)
# version 8 (owner, 30 Sep): the DC can sprint, and every command change is a ramp - speeding up at the species'
# max_accel, slowing down gently at SLOW_DOWN, so a sprint eases back into the walk (training and playback alike)
SPRINT_SPEED = {"dc": 4.0}   # m/s (the navigation brain's max_speed stays 2.5 until it is retrained)
SLOW_DOWN = 1.0              # m/s^2


def body_for(species, version, passive=False):
    """The body a walking brain of this version controls."""
    if version >= 6 and species == "dc":
        body = BODIES[species](passive=True, active=True, arms=True, hulls=version >= 8, lean=version >= 11,
                                detailed=version >= 12)
    elif version >= 9:
        body = BODIES[species](passive=True, hulls=True)
    elif ACTIVE.get(version, {}).get(species, False):
        body = BODIES[species](passive=True, active=True)
    else:
        body = BODIES[species](passive=passive or version >= 4)
    if version >= 5:
        for ch in body.passive:
            ch.damping_ratio = PASSIVE_DAMPING_V5.get(species, {}).get(ch.name, ch.damping_ratio)
    return body


class LocomotionEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / CONTROL_DT)}

    def __init__(self, species="myrranox", episode_seconds=20.0, command_seconds=(2.0, 5.0),
                 stand_prob=0.15, render_mode=None, width=640, height=400, version=1,
                 model=None, data=None, prefix="", passive=False, head_weight=1.0, speed_sigma=None):
        self.species_name = species
        self.species = SPECIES[species]
        self.body = body_for(species, version, passive)     # passive=True: tail/abdomen/head/arms on springs
        self.version = version
        self.speed_sigma = speed_sigma      # training option: fixed speed tolerance (m/s) instead of 0.5 x max speed
        self.top_speed = SPRINT_SPEED.get(species, SPECIES[species].max_speed) if version >= 8 else SPECIES[species].max_speed
        self.ramp_all = version >= 8        # every command change ramps (speed up / slow down gradually)
        self.command_limit = None           # training stage: highest commanded speed (set_command_limit)
        self.walk_help = 0.0                # training start-up help: bonus for forward progress (set_walk_help)
        self.w = dict(WEIGHTS[version])
        if version >= 5:
            self.w.update(STEADY_TRUNK_V5.get(species, {}))
        for k in ("head_steady", "head_level"):              # scale of the v3 head penalties (train_locomotion.py)
            if k in self.w:
                self.w[k] *= head_weight
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
        self.is_leg = np.array([leg.mass_scale == 1.0 for leg in self.body.legs])     # False: the DC's arms
        self.nlegs = len(self.leg_names)
        self.front_legs = np.array([n.startswith("front") for n in self.leg_names])      # version 12 (DC)
        self.hind_legs = np.array([n.startswith("hind") for n in self.leg_names])
        self.paw_bodies = np.array([nid(mujoco.mjtObj.mjOBJ_BODY, f"{n}_s2") for n in self.leg_names])             if self.front_legs.any() else np.zeros(0, int)
        self.foot_geoms = np.array([nid(mujoco.mjtObj.mjOBJ_GEOM, n) for n in self.body.foot_names])
        self.foot_sites = np.array([nid(mujoco.mjtObj.mjOBJ_SITE, n) for n in self.body.foot_names])
        self.foot_radius = np.where(m.geom_type[self.foot_geoms] == mujoco.mjtGeom.mjGEOM_MESH, 0.0,
                                    m.geom_size[self.foot_geoms, 0])     # mesh paws: the site is on the sole
        self.floor_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.torso_body = nid(mujoco.mjtObj.mjOBJ_BODY, "torso")
        trunk = [self.torso_body]
        if self.body.rear is not None:
            trunk.append(nid(mujoco.mjtObj.mjOBJ_BODY, f"{self.body.rear.name}_rear"))
        self.torso_geoms = set(np.flatnonzero(np.isin(m.geom_bodyid, trunk)).tolist())
        belly = [i for i in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(prefix + "abdomen")]
        self.belly_geoms = set(np.flatnonzero(np.isin(m.geom_bodyid, belly)).tolist())
        self.head_body = nid(mujoco.mjtObj.mjOBJ_BODY, self.body.head_body) if self.body.head_body else None
        self.action_scales = np.array([sc for _, _, sc in self.body.actuator_params()])
        self.passive_qpos = np.array([m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, n)]
                                      for n in self.body.passive_joint_names], dtype=int)

        mujoco.mj_forward(m, self.data)
        self.stand_z = float(self.data.qpos[self.q0 + 2])
        self.init_qpos = self.data.qpos.copy()
        self.rest_joints = self.init_qpos[self.joint_qpos].copy()
        self.power_ref = self.body.mass * 9.81 * self.species.walk_speed
        self.neutral = np.zeros(self.nj)       # version 9: joint offsets of the natural standing pose
        if version >= 5:
            self._init_style()
        if version >= 9:
            self._init_neutral()

        obs_size = 3 + 3 + 3 + 1 + 3 * self.nj + 2 + (2 * self.nlegs + 1 if version >= 2 else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nj,), dtype=np.float32)
        self._renderer = None
        self._size = (height, width)
        self.penalty_scale = 1.0        # ramped up from ~0.2 during training (train_locomotion.py)
        self.track_scale = 1.0          # version 4: tracking tolerance, lowered to 0.5 during training

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
            d.qpos[self.joint_qpos] = self.init_qpos[self.joint_qpos] + noise
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
        self.last_action = np.zeros(self.nj)     # version 4: the filtered target the joints are driven to
        self.last_raw = self.raw = np.zeros(self.nj)
        self.prev_foot_pos = self.data.site_xpos[self.foot_sites].copy()
        self.prev_qvel = self.data.qvel[self.joint_qvel].copy()
        self.phase = float(rng.uniform())
        self.air_time = np.zeros(self.nlegs)
        self.prev_contact = np.ones(self.nlegs, dtype=bool)
        self.prev_vx = self.prev_vx_v9 = 0.0
        self.since_ground = np.zeros(self.nlegs)
        self.since_touchdown = np.full(self.nlegs, 10.0)
        self.register_ema = 0.5
        self.step_rate = np.zeros(self.nlegs)        # version 12: touchdowns per second per leg (EMA, ~2 s)
        if self.version >= 5:
            self.last_print = np.full((self.nlegs, 2), np.nan)
            self.q_mean = self.data.qpos[self.joint_qpos].copy()
        self.command = np.zeros(2)
        self.command_target = np.zeros(2)
        self.command_rate = None
        self.command_steps_left = 0
        if options and "command" in options:
            self.set_command(*options["command"], hold=True)
        else:
            self._resample_command()
        return self._observe(), {}

    def set_command(self, speed, turn_rate, hold=True):
        """Forward speed (m/s) and turn rate (rad/s) for the brain to follow; clipped to species limits."""
        sp = self.species
        target = np.array([np.clip(speed, -sp.max_backward_speed, self.top_speed),
                           np.clip(turn_rate, -sp.max_turn_rate, sp.max_turn_rate)])
        if self.ramp_all:                   # version 8: the command glides to the new value
            self.command_target = target
            self.command_rate = self._ramp_rates()
        else:
            self.command = target
            self.command_target = self.command.copy()
            self.command_rate = None
        self.command_steps_left = 10 ** 9 if hold else self.command_steps_left

    def _ramp_rates(self):
        sp = self.species
        return np.array([sp.max_accel, sp.max_turn_rate / COMMAND_RAMP_S[1]]) * CONTROL_DT

    def _advance_command(self):
        """Move the command toward its target: speeding up at max_accel, slowing down at SLOW_DOWN (version 8)."""
        rate = self.command_rate.copy()
        if self.ramp_all and abs(self.command_target[0]) < abs(self.command[0]):
            rate[0] = SLOW_DOWN * CONTROL_DT
        self.command = self.command + np.clip(self.command_target - self.command, -rate, rate)

    def _resample_command(self):
        rng, sp = self.np_random, self.species
        u = rng.uniform()
        if u < self.stand_prob:
            v, w = 0.0, 0.0
        elif self.version >= 2 and sp.turn_in_place and u < self.stand_prob + (0.20 if self.version >= 4 else 0.10):
            v, w = 0.0, rng.uniform(-sp.max_turn_rate, sp.max_turn_rate)            # turn on the spot
        elif (self.version >= 2 and sp.max_backward_speed > 0
              and u < self.stand_prob + (0.30 if self.version >= 4 else 0.20)):
            v, w = -rng.uniform(0.1, sp.max_backward_speed), 0.0                      # walk backwards
        else:
            top = min(self.top_speed, self.command_limit or self.top_speed)
            v = rng.uniform(-sp.max_backward_speed, sp.max_speed) if self.version == 1 else rng.uniform(0.1, top)
            w = rng.uniform(-sp.max_turn_rate, sp.max_turn_rate) * (rng.uniform() < (0.7 if self.version == 1 else 0.8))
            if not sp.turn_in_place and abs(v) < sp.min_turn_speed:
                w *= abs(v) / sp.min_turn_speed
        if self.ramp_all:                                    # version 8: always a ramp (gradual speed changes)
            self.command_target = np.array([v, w])
            self.command_rate = self._ramp_rates()
        elif self.version >= 4 and rng.uniform() < 0.5:      # ramp to the new command, like the navigation brain
            self.command_target = np.array([v, w])
            self.command_rate = np.array([sp.max_speed / COMMAND_RAMP_S[0],
                                          sp.max_turn_rate / COMMAND_RAMP_S[1]]) * CONTROL_DT
        else:
            self.command = self.command_target = np.array([v, w])
            self.command_rate = None
        self.command_steps_left = int(rng.uniform(*self.command_seconds) / CONTROL_DT)

    # ------------------------------------------------------------------------------------------
    def gait_targets(self):
        """(per-leg phase in [0,1), desired stance mask, duty, frequency Hz, standing?) for the current command."""
        v, w = self.command
        offsets, duty = gaits.pattern(self.species_name, self.leg_names, v)
        f = gaits.frequency(self.species_name, v, w, self.species.max_turn_rate, style=self.version >= 5,
                            sprint=self.version >= 8)
        standing = gaits.is_standing(v, w)
        leg_phase = (self.phase + offsets) % 1.0
        stance = np.ones(self.nlegs, dtype=bool) if standing else leg_phase < duty
        if standing and np.ndim(duty):           # six limbs: the arms stay folded while standing
            stance &= self.is_leg
        return leg_phase, stance, duty, f, standing

    def step(self, action):
        a = self.apply_action(action)
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        return self.after_physics(a)

    def apply_action(self, action):
        """Set this creature's joint targets; returns the clipped action (pass it to after_physics)."""
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.version >= 4:                   # low-pass: the joint targets glide instead of jumping
            self.raw = a
            a = self.last_action + ACTION_FILTER * (a - self.last_action)
        if self.version == 7:                   # step generator + the brain's correction
            target = self.rest_joints + self.cpg_targets() + a * CPG_RESIDUAL
        elif self.version == 10:                # standing pose + tripod rhythm + the brain's correction
            target = self.rest_joints + self.neutral + self.cpg_targets() + a * CPG_RESIDUAL_V10
        else:
            target = self.rest_joints + self.neutral + a * self.action_scales
        self.data.ctrl[self.act_ids] = np.clip(target, self.joint_lo, self.joint_hi)
        return a

    def cpg_targets(self):
        """Version 7: joint offsets from the rest pose given by the gait clock (the step generator)."""
        out = np.zeros(self.nj)
        leg_phase, stance, duty, f, standing = self.gait_targets()
        idle = self.arm_legs & (np.broadcast_to(np.asarray(duty, dtype=float), (self.nlegs,)) <= 0)
        ref = self.style_reference(leg_phase, duty, f, standing)
        if ref is not None:
            sweep, knee = ref
            fold_mean = (1 - np.broadcast_to(np.asarray(duty, dtype=float), (self.nlegs,))) * 2 / np.pi
            out[self.sweep_idx] = sweep
            out[self.knee_idx] = knee + fold_mean * self.knee_fold * self.knee_dir    # stance straight, fold in swing
            if self.ankle_ref is not None:
                out[self.ankle_idx] = self.ankle_ref + fold_mean * ANKLE_FOLD * self.ankle_dir
        for i in np.flatnonzero(self.arm_legs):   # arms not in the gait: folded against the chest
            nj = len(self.body.legs[i].joints)
            if idle[i] or standing:
                out[i * nj:(i + 1) * nj] = self.tuck_pose[i]
        return out

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
        body_contact = belly_contact = False
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
            elif other in self.belly_geoms:
                belly_contact = True

        foot_pos = d.site_xpos[self.foot_sites]
        foot_vel3 = (foot_pos - self.prev_foot_pos) / CONTROL_DT
        foot_vel = foot_vel3[:, :2]
        self.prev_foot_pos = foot_pos.copy()
        qvel = d.qvel[self.joint_qvel]
        qacc = (qvel - self.prev_qvel) / CONTROL_DT
        self.prev_qvel = qvel.copy()
        power = float(np.sum(np.abs(d.actuator_force[self.act_ids] * qvel)))        # W, mechanical

        sp, w = self.species, self.w
        sig_v = self.speed_sigma or 0.5 * sp.max_speed * self.track_scale   # wide enough that moving a little pays off
        sig_w = 0.5 * sp.max_turn_rate * self.track_scale
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
            "action_rate": w["action_rate"] * float(np.sum((self.raw - self.last_raw) ** 2 if self.version >= 4
                                                           else (a - self.last_action) ** 2)),
            "torque": w["torque"] * float(np.sum(torque ** 2)) / self.nj * 4,
            "foot_slip": w["foot_slip"] * float(np.sum(np.minimum((foot_vel ** 2).sum(axis=1), 1.0) * foot_contact)),
            "body_contact": w["body_contact"] * body_contact,
            "joint_limit": w["joint_limit"] * near_limit,
        }
        if self.version >= 4 and abs(self.command[0]) < 0.05 and abs(self.command[1]) > 0.1:
            # Turn on the spot: standing still matches the speed command perfectly, so the speed reward only
            # counts as far as the turn is achieved (the Myrranox never learned spot turns in v1-v3).
            terms["lin"] *= math.exp(-ang_err / sig_w ** 2)

        leg_phase = stance = None
        if self.version >= 2:
            leg_phase, stance, duty, f, standing = self.gait_targets()
            swing = ~stance
            # Stance legs down + swing legs up, scored separately: perfect gait 1, standing still or random 0
            # (a plain "fraction of legs matching" would give standing still 55-65 % for free).
            stance_ok = float(np.mean(foot_contact[stance])) if stance.any() else 1.0
            swing_ok = float(np.mean(~foot_contact[swing])) if swing.any() else 1.0
            gait_score = max(stance_ok + swing_ok - 1.0, 0.0)
            terms["gait"] = w["gait"] * (stance_ok + swing_ok - 1.0)
            if swing.any():
                height = foot_pos[:, 2] - self.foot_radius
                target_h = gaits.FOOT_CLEARANCE[self.species_name]
                terms["clearance"] = w["clearance"] * float(np.mean(np.clip(height[swing] / target_h, 0, 1)))
            else:
                terms["clearance"] = 0.0
            # Air time: reward full swings, punish taps (only while walking)
            touchdown = foot_contact & ~self.prev_contact
            if "landing" in w:
                terms["landing"] = w["landing"] * float(np.sum(np.minimum(foot_vel3[touchdown, 2] ** 2, 4.0)))
            t_swing = np.broadcast_to((1 - np.asarray(duty)) / f, (self.nlegs,))
            taps = int(np.sum(self.air_time[touchdown] < 0.5 * t_swing[touchdown])) if not standing else 0
            if not standing and touchdown.any():
                terms["air_time"] = w["air_time"] * float(np.sum(np.minimum(self.air_time[touchdown] / t_swing[touchdown], 1.0) - 0.5))
            else:
                terms["air_time"] = 0.0
            self.air_time = np.where(foot_contact, 0.0, self.air_time + CONTROL_DT)
            self.since_ground = np.where(foot_contact, 0.0, self.since_ground + CONTROL_DT)
            self.since_touchdown = self.since_touchdown + CONTROL_DT
            self.prev_contact = foot_contact.copy()
            if "fidget" in w:
                terms["fidget"] = w["fidget"] * min(float(np.mean(qvel ** 2)), 1.0) if standing else 0.0   # capped
            if "surge" in w:
                ax = (lin[0] - self.prev_vx) / CONTROL_DT
                terms["surge"] = w["surge"] * min((ax / 2.0) ** 2, 1.0)   # capped: untrained jitter must not swamp the task
            self.prev_vx = float(lin[0])
            if self.version >= 5:                   # the knee may work (lesson: the penalties froze it)
                ew = self.effort_weight
                terms["energy"] = w["energy"] * float(np.sum(ew * np.abs(d.actuator_force[self.act_ids] * qvel))) / self.power_ref
                terms["joint_acc"] = w["joint_acc"] * float(np.mean(ew * (qacc / 500.0) ** 2))
                style_score = self._style(leg_phase, duty, f, standing)
                terms["style"] = w["style"] * style_score
                if self.species_name in STEADY_TRUNK_V5:  # steady trunk, not dodged by standing still
                    track = math.exp(-lin_err / sig_v ** 2)
                    terms["roll_pitch"] *= track
                    terms["vz"] *= track
                terms["register"] = w["register"] * self._register(foot_pos, touchdown, standing)
                if "arm_ground" in w:
                    idle_arms = self.arm_legs & (np.broadcast_to(np.asarray(duty, dtype=float), (self.nlegs,)) <= 0)
                    terms["arm_ground"] = w["arm_ground"] * float(np.sum(foot_contact & idle_arms))
                if "arm_tuck" in w:
                    terms["arm_tuck"] = w["arm_tuck"] * self._arm_tuck(foot_pos, duty)
            else:
                terms["energy"] = w["energy"] * power / self.power_ref
                terms["joint_acc"] = w["joint_acc"] * float(np.mean((qacc / 500.0) ** 2))
            if not standing:
                self.phase = (self.phase + f * CONTROL_DT) % 1.0
            if self.version >= 3 and self.head_body is not None:
                w_head = d.cvel[self.head_body][:3]                  # world-frame angular velocity of the head
                w_err = w_head - np.array([0.0, 0.0, self.command[1]])
                terms["head_steady"] = w["head_steady"] * float(w_err @ w_err)
                terms["head_level"] = w["head_level"] * (1.0 - float(d.xmat[self.head_body].reshape(3, 3)[2, 2]))
                if self.version >= 4:            # style only counts while the command is tracked (lesson 8)
                    track = math.exp(-lin_err / sig_v ** 2)
                    terms["head_steady"] *= track
                    terms["head_level"] *= track

        reward = sum(v for k, v in terms.items() if k in TASK_TERMS) + self.penalty_scale * float(
            sum(v for k, v in terms.items() if k not in TASK_TERMS))
        if self.version >= 9:
            ew = self.effort_weight
            idle_arms = self.arm_legs & (np.broadcast_to(np.asarray(duty, dtype=float), (self.nlegs,)) <= 0)
            w_head = (d.cvel[self.head_body][:3] - np.array([0.0, 0.0, self.command[1]])) if self.head_body is not None else None
            p = dict(
                energy=float(np.sum(ew * np.abs(d.actuator_force[self.act_ids] * qvel))) / self.power_ref,
                jerk=float(np.mean(ew * (qacc / 500.0) ** 2)),
                action_rate=float(np.mean((a - self.last_action) ** 2)),
                slip=float(np.sum(np.minimum((foot_vel ** 2).sum(axis=1), 1.0) * foot_contact)),
                trunk_rock=float(ang[0] ** 2 + ang[1] ** 2),
                bounce=float(lin[2] ** 2),
                tilt=float(grav[0] ** 2 + grav[1] ** 2),
                surge=min(((lin[0] - self.prev_vx_v9) / CONTROL_DT / 2.0) ** 2, 1.0),
                landing=float(np.sum(np.minimum(foot_vel3[touchdown, 2] ** 2, 4.0))),
                taps=float(taps),
                head_steady=float(w_head @ w_head) if w_head is not None else 0.0,
                head_level=(1.0 - float(d.xmat[self.head_body].reshape(3, 3)[2, 2])) if w_head is not None else 0.0,
                fidget=min(float(np.mean(qvel ** 2)), 1.0) if standing else 0.0,
                joint_limit=float(near_limit),
                belly=float(belly_contact),
                arm_ground=float(np.sum(foot_contact & idle_arms)),
                body_contact=float(body_contact),
                track=self._track_width(foot_pos, foot_contact),
                idle_leg=float(np.sum(self.is_leg & (self.since_ground > 2.0 / f))) if not standing else 0.0,
                extra_step=float(np.sum(touchdown & self.is_leg & (self.since_touchdown < 0.6 / f))) if not standing else 0.0)
            wolf = self.version >= 12 and self.front_legs.any()
            if wolf:
                walking = not standing and abs(self.command[0]) > 0.1
                self.step_rate += (CONTROL_DT / 2.0) * (touchdown / CONTROL_DT - self.step_rate)
                lift = foot_pos[:, 2] - self.foot_radius
                swing_legs = self.is_leg & ~foot_contact
                p["support"] = float(walking and np.sum(foot_contact & self.is_leg) < 2)
                p["high_step"] = float(np.sum((np.maximum(lift[swing_legs] - WOLF_LIFT, 0.0) / 0.05) ** 2)) if walking else 0.0
            self.since_touchdown = np.where(touchdown, 0.0, self.since_touchdown)
            self.prev_vx_v9 = float(lin[0])
            qw = V12_QUALITY if self.version >= 12 else V9_QUALITY
            quality = math.exp(-self.penalty_scale * sum(qw[k] * v for k, v in p.items()))
            # tolerance grows with the commanded speed / turn: moving a little already earns something, standing
            # still while walking is commanded earns almost nothing (a fixed wide tolerance let it pay, 13 Oct test)
            tol_v = (0.5 * abs(self.command[0]) + 0.1) * max(self.track_scale, 0.5)
            tol_w = (0.5 * abs(self.command[1]) + 0.15) * max(self.track_scale, 0.5)
            tracking = math.exp(-lin_err / tol_v ** 2) * math.exp(-ang_err / tol_w ** 2)
            likeness = [gait_score, 1.0 if standing else style_score]
            if self.arm_legs.any():                         # DC: panther traits
                likeness.append(self.register_ema)
                if idle_arms.any():
                    likeness.append(self._arm_tuck(foot_pos, duty))
                if wolf:
                    likeness += self._wolf(foot_pos, foot_contact, walking)
            likeness = float(np.mean(likeness))
            # moving like the animal counts: a creature on speed but in the wrong rhythm keeps 20 % (was 40 %)
            reward = tracking * (0.2 + 0.8 * likeness) * quality
            if self.walk_help > 0 and abs(self.command[0]) > 0.1:
                # start-up help (owner, 13 Oct: "help the Myrranox a little bit for walking"): every bit of progress in
                # the commanded direction pays, whatever the tempo or style; standing still never earns it. Faded out.
                progress = float(np.clip(lin[0] * np.sign(self.command[0]) / abs(self.command[0]), 0.0, 1.0))
                reward += self.walk_help * 0.3 * progress
                terms["v9_walk_help"] = self.walk_help * 0.3 * progress
            terms.update({"v9_tracking": tracking, "v9_likeness": likeness, "v9_quality": quality})
            terms.update({f"q_{k}": qw[k] * v for k, v in p.items()})
        self.last_action = a
        if self.version >= 4:
            self.last_raw = self.raw
            if self.command_rate is not None:
                self._advance_command()

        self.t += 1
        self.command_steps_left -= 1
        if self.command_steps_left <= 0:
            self._resample_command()

        max_tilt = MAX_TILT_V8 if self.version >= 8 else 60.0
        fallen = (grav[2] > -math.cos(math.radians(max_tilt))
                  or d.qpos[self.q0 + 2] < self.stand_z - 0.5 * self._leg_height())
        terminated = bool(fallen)
        if terminated:
            reward -= 10.0
        truncated = self.t >= self.episode_steps
        info = {"terms": terms, "command": self.command.copy(), "speed": float(lin[0]), "turn_rate": float(ang[2]),
                "foot_contact": foot_contact, "power": power, "desired_stance": stance}
        return self._observe(lin, ang, grav), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------------------------------
    def _init_style(self):
        """Version 5: per leg the joint that sweeps the foot forward/back and the knee, with their directions,
        measured on the model at the rest pose (so no signs are typed by hand)."""
        m, d = self.model, self.data
        sweep_k, knee_k, self.knee_fold, self.max_sweep = STYLE[self.species_name]
        nj = len(self.body.legs[0].joints)
        R = d.xmat[self.torso_body].reshape(3, 3)
        hips = np.array([leg.points[0] for leg in self.body.legs])

        def foot_body():
            mujoco.mj_kinematics(m, d)
            return (d.site_xpos[self.foot_sites] - d.xpos[self.torso_body]) @ R

        base = foot_body()
        self.sweep_idx = np.array([i * nj + sweep_k for i in range(self.nlegs)])
        self.knee_idx = np.array([i * nj + knee_k for i in range(self.nlegs)])
        self.sweep_dir, self.sweep_gain, self.knee_dir = np.zeros(self.nlegs), np.zeros(self.nlegs), np.zeros(self.nlegs)
        for i in range(self.nlegs):
            for idx, kind in ((self.sweep_idx[i], "sweep"), (self.knee_idx[i], "knee")):
                qa = self.joint_qpos[idx]
                d.qpos[qa] += 0.05
                delta = (foot_body()[i] - base[i]) / 0.05
                d.qpos[qa] -= 0.05
                if kind == "sweep":             # +1: positive joint angle moves the foot forward
                    self.sweep_dir[i], self.sweep_gain[i] = np.sign(delta[0]), abs(delta[0])
                else:                           # +1: positive joint angle shortens the leg (folds it)
                    dist = np.linalg.norm(base[i] + 0.05 * delta - hips[i]) - np.linalg.norm(base[i] - hips[i])
                    self.knee_dir[i] = -np.sign(dist)
        foot_body()
        self.effort_weight = np.ones(self.nj)
        self.effort_weight[self.knee_idx] = KNEE_EFFORT
        self.front_of = {i: self.leg_names.index(n.replace("hind", "front"))
                         for i, n in enumerate(self.leg_names) if n.startswith("hind")}
        # version 6: ankles / wrists (4th joint of a limb) join the rhythm; fold direction measured like the knee
        self.ankle_idx, self.ankle_dir = np.array([], dtype=int), np.array([])
        if self.version >= 6 and nj >= 4:
            self.ankle_idx = np.array([i * nj + 3 for i in range(self.nlegs)])
            dirs = []
            for i in range(self.nlegs):
                qa = self.joint_qpos[self.ankle_idx[i]]
                d.qpos[qa] += 0.05
                moved = foot_body()[i]
                d.qpos[qa] -= 0.05
                dirs.append(-np.sign(np.linalg.norm(moved - hips[i]) - np.linalg.norm(base[i] - hips[i])))
            self.ankle_dir = np.array(dirs)
            foot_body()
        self.arm_legs = np.array([n.startswith("arm") for n in self.leg_names])
        self.tuck_target = np.array([[ARM_TUCK[0], ARM_TUCK[1] * np.sign(leg.points[0, 1]), ARM_TUCK[2]]
                                     for leg in self.body.legs])
        # folded arm pose (joint offsets) that brings the claws to the tuck target - small coordinate search
        self.tuck_pose = {}
        for i in np.flatnonzero(self.arm_legs):
            idx = [i * nj + k for k in range(1, nj)]              # pitch joints (shoulder, elbow, wrist)
            q = np.zeros(len(idx))

            def cost(qq):
                for k, j in enumerate(idx):
                    d.qpos[self.joint_qpos[j]] = self.rest_joints[j] + qq[k]
                return float(np.linalg.norm(foot_body()[i] - self.tuck_target[i]))
            best = cost(q)
            for step in (0.4, 0.2, 0.1, 0.05, 0.02):
                improved = True
                while improved:
                    improved = False
                    for k in range(len(idx)):
                        for sgn in (1, -1):
                            trial = q.copy()
                            trial[k] += sgn * step
                            lo, hi = self.joint_lo[idx[k]] - self.rest_joints[idx[k]], self.joint_hi[idx[k]] - self.rest_joints[idx[k]]
                            trial[k] = np.clip(trial[k], lo, hi)
                            c = cost(trial)
                            if c < best - 1e-4:
                                best, q, improved = c, trial, True
            for j in idx:
                d.qpos[self.joint_qpos[j]] = self.rest_joints[j]
            pose = np.zeros(nj)
            pose[1:] = q
            self.tuck_pose[i] = pose
            self.tuck_error = best
        foot_body()
        self.last_print = np.full((self.nlegs, 2), np.nan)

    def _init_neutral(self):
        """Version 9: the natural standing pose. Myrranox: every leg's lift / knee joints chosen so the foot sits
        STAND_TALLER lower relative to the body (the body rises, the feet stay put). DC: arms folded (tuck pose).
        The episode start pose and the standing height follow."""
        m, d = self.model, self.data
        nj = len(self.body.legs[0].joints)
        R = d.xmat[self.torso_body].reshape(3, 3)

        def foot_body():
            mujoco.mj_kinematics(m, d)
            return (d.site_xpos[self.foot_sites] - d.xpos[self.torso_body]) @ R

        def set_pose(offset):
            d.qpos[self.joint_qpos] = self.rest_joints + offset

        base = foot_body()
        rise = STAND_TALLER.get(self.species_name, 0.0)
        for i, leg in enumerate(self.body.legs):
            if self.arm_legs[i]:
                self.neutral[i * nj:(i + 1) * nj] = self.tuck_pose[i]
                continue
            if not rise:
                continue
            idx = [i * nj + 1, i * nj + 2]                       # lift + knee
            r = rise[leg.name.split("_")[0]] if isinstance(rise, dict) else rise
            goal = base[i].copy()                                # every foot on the same floor level (the
            goal[2] = base[self.is_leg, 2].min() - r             # model's front claws sit 3 cm higher)
            q = np.zeros(2)

            def cost(qq):
                off = self.neutral.copy()
                off[idx] = qq
                set_pose(off)
                return float(np.linalg.norm(foot_body()[i] - goal))
            best = cost(q)
            for step in (0.2, 0.1, 0.05, 0.02, 0.01):
                improved = True
                while improved:
                    improved = False
                    for k in range(2):
                        for sgn in (1, -1):
                            trial = q.copy()
                            trial[k] += sgn * step
                            c = cost(trial)
                            if c < best - 1e-5:
                                best, q, improved = c, trial, True
            self.neutral[idx] = q
        for i, leg in enumerate(self.body.legs):             # mirror: the right side copies the left side
            if leg.name.endswith("_R"):
                src = self.leg_names.index(leg.name[:-1] + "L")
                self.neutral[i * nj:(i + 1) * nj] = self.neutral[src * nj:(src + 1) * nj]
        if self.version >= 12 and "neck_p0x" in self.body.joint_names:     # wolf head posture (bodies.WOLF_NECK_DEG)
            from sim.bodies import WOLF_NECK_DEG
            self.neutral[self.body.joint_names.index("neck_p0x")] = math.radians(WOLF_NECK_DEG)
        set_pose(np.zeros(self.nj))
        z_rest = foot_body()[self.is_leg, 2].min()
        set_pose(self.neutral)
        feet_neutral = foot_body()
        z_neutral = feet_neutral[self.is_leg, 2].min()
        self.foot_rest_y = np.abs(feet_neutral[:, 1])            # paw track of the standing pose
        self.init_qpos[self.joint_qpos] = self.rest_joints + self.neutral
        self.init_qpos[self.q0 + 2] += z_rest - z_neutral            # feet stay on the floor
        if not self.shared:
            # Reference height for falls / the height observation: where the body really settles in the pose
            # (under its weight the Myrranox sits ~4 cm below the computed pose; using the computed height made an
            # upright creature count as fallen - 13 Oct)
            d.qpos[:] = self.init_qpos
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            for _ in range(int(1.0 / m.opt.timestep)):
                d.ctrl[self.act_ids] = self.init_qpos[self.joint_qpos]
                mujoco.mj_step(m, d)
            self.stand_z = float(d.qpos[self.q0 + 2])
        else:
            # shared world (habitat): the world cannot be stepped here - measure the settled height on a private copy
            # of this creature (else an upright creature counts as fallen, 13 Oct: 98 "falls" per minute)
            self.stand_z = LocomotionEnv(self.species_name, version=self.version).stand_z
        mujoco.mj_resetData(m, d) if not self.shared else None
        d.qpos[:] = self.init_qpos
        mujoco.mj_forward(m, d)

    def style_reference(self, leg_phase, duty, f, standing):
        """Reference offsets from the rest pose for the sweep and knee joints (rad), or None while standing."""
        if standing:
            return None
        v = self.command[0]
        duty = np.broadcast_to(np.asarray(duty, dtype=float), (self.nlegs,))
        active = duty > 0                                            # six limbs: folded arms have no rhythm
        stride = abs(v) * duty / f                                   # stance foot travel that matches the speed
        amp = np.minimum(stride / 2 / self.sweep_gain, self.max_sweep) * np.sign(v)
        in_stance = leg_phase < duty
        p = np.where(in_stance, leg_phase / np.maximum(duty, 1e-6), (leg_phase - duty) / (1 - duty))
        fwd = np.where(in_stance, 1 - 2 * p, -np.cos(np.pi * p))    # stance: front -> back, swing: back -> front
        swing_bump = (np.where(in_stance, 0.0, np.sin(np.pi * p)) - (1 - duty) * 2 / np.pi) * active  # zero mean
        self.ankle_ref = swing_bump * ANKLE_FOLD * self.ankle_dir if len(self.ankle_idx) else None
        return amp * fwd * self.sweep_dir, swing_bump * self.knee_fold * self.knee_dir

    def _style(self, leg_phase, duty, f, standing):
        # Rhythm, not posture: compare each joint's swing around its own running mean (1 s), so the brain keeps
        # its stance (e.g. insect front legs angled forward, back legs backward) and only the motion is shaped.
        q_abs = self.data.qpos[self.joint_qpos]
        self.q_mean += (q_abs - self.q_mean) * (CONTROL_DT / 1.0)
        ref = self.style_reference(leg_phase, duty, f, standing)
        if ref is None:
            return 0.0
        q = q_abs - self.q_mean
        err = np.concatenate([q[self.sweep_idx] - ref[0], q[self.knee_idx] - ref[1]])
        if abs(self.command[0]) < 0.1:                                # turning on the spot: knees only
            err = q[self.knee_idx] - ref[1]
        if self.ankle_ref is not None:                                # version 6: paws flat, no flicking
            err = np.concatenate([err, q[self.ankle_idx] - self.ankle_ref])
        return math.exp(-float(np.mean(err ** 2)) / 0.35 ** 2)

    def _track_width(self, foot_pos, foot_contact):
        """Version 9, DC: paws on the ground further out than in the standing pose (a panther walks narrow)."""
        if not self.arm_legs.any():
            return 0.0
        legs = self.is_leg & foot_contact
        if not legs.any():
            return 0.0
        R = self.data.xmat[self.torso_body].reshape(3, 3)
        y = np.abs(((foot_pos - self.data.xpos[self.torso_body]) @ R)[:, 1])
        excess = np.maximum(y - self.foot_rest_y, 0.0)[legs]
        return float(np.mean((excess / 0.05) ** 2))

    def _arm_tuck(self, foot_pos, duty):
        """Version 6: arms that are not part of the gait keep their claws folded near the chest."""
        idle = self.arm_legs & (np.broadcast_to(np.asarray(duty, dtype=float), (self.nlegs,)) <= 0)
        if not idle.any():
            return 0.0
        R = self.data.xmat[self.torso_body].reshape(3, 3)
        rel = (foot_pos[idle] - self.data.xpos[self.torso_body]) @ R
        dist = np.linalg.norm(rel - self.tuck_target[idle], axis=1)
        return float(np.mean(np.exp(-(dist / 0.25) ** 2)))

    def _wolf(self, foot_pos, foot_contact, walking):
        """Version 12, DC: likeness scores (0..1) from the wolf reference - same step rate front and hind, front wrist
        folded under in the swing, head low and forward."""
        d = self.data
        R = d.xmat[self.torso_body].reshape(3, 3)
        head_dz = float(((d.xpos[self.head_body] - d.xpos[self.torso_body]) @ R)[2]) if self.head_body is not None else WOLF_HEAD_DZ
        head = math.exp(-((head_dz - WOLF_HEAD_DZ) / 0.08) ** 2)
        if not walking:
            return [head]
        front, hind = self.step_rate[self.front_legs].mean(), self.step_rate[self.hind_legs].mean()
        rhythm = math.exp(-(math.log(max(hind, 0.1) / max(front, 0.1)) / 0.3) ** 2)
        swing = self.front_legs & ~foot_contact & (self.air_time > 0.06)
        if swing.any():
            # paw tip behind the wrist (body frame x) = wrist folded under; flat paw (tip in front) = 0
            rel = (foot_pos[swing] - d.xpos[self.paw_bodies[swing]]) @ R
            fold = float(np.mean(np.clip(-rel[:, 0] / 0.08 + 0.5, 0.0, 1.0)))
        else:
            fold = 0.5
        return [rhythm, fold, head]

    def _register(self, foot_pos, touchdown, standing):
        """Panther walk: each hind paw lands close to where the front paw of the same side last stood."""
        if not self.front_of:
            return 0.0
        for i in np.flatnonzero(touchdown):
            self.last_print[i] = foot_pos[i, :2]
        if standing or abs(self.command[0]) > 1.2:                   # walking only (not in the trot)
            return 0.0
        r = 0.0
        for hind, front in self.front_of.items():
            if touchdown[hind] and not np.isnan(self.last_print[front, 0]):
                dist = float(np.linalg.norm(foot_pos[hind, :2] - self.last_print[front]))
                r += math.exp(-(dist / 0.15) ** 2)
                self.register_ema += 0.2 * (math.exp(-(dist / 0.15) ** 2) - self.register_ema)
        return r

    def set_penalty_scale(self, scale):
        self.penalty_scale = float(scale)

    def set_walk_help(self, scale):
        """Training start-up help: weight of the forward-progress bonus (1 at the start, faded to 0)."""
        self.walk_help = float(scale)

    def set_command_limit(self, speed):
        """Training stage: commands up to this speed only (walk first, then trot, run, sprint)."""
        self.command_limit = float(speed)

    def set_track_scale(self, scale):
        self.track_scale = float(scale)

    def _leg_height(self):
        return float(np.mean([leg.points[0, 2] for leg in self.body.legs if leg.mass_scale == 1.0]))

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
