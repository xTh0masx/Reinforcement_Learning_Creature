"""MuJoCo bodies for the locomotion brains, built from measurements of the Blender models.

Frame: +X forward, +Y left, +Z up, metres, at real size (Myrranox 1.5 m long, DC 1.3 m tall).
Joint angle 0 is the pose the model has in the .blend file, so Blender can apply joint angles
directly as bone rotations from the rest pose.

Myrranox: hexapod, 6 legs x 3 hinges (coxa yaw, femur lift, claw lift), measured joint positions.
DC_SKIN_RAW: quadruped, 4 legs x 4 hinges (hip abduction + hip pitch, knee, ankle), symmetric
approximation measured from a side render; the raised front arms are part of the torso.
Blender follows its feet with IK.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Leg:
    name: str                   # e.g. "front_L"
    points: np.ndarray          # (4, 3): segment 0/1/2 start points and the foot tip - in the body frame
    joints: list                # (segment, axis (body frame), (lo, hi) degrees); several per segment allowed
    radius: float

    def joint_names(self):
        return [f"{self.name}_j{k}" for k in range(len(self.joints))]


@dataclass
class PassiveChain:
    """Body parts moved only by physics (spring-damper joints, no motor, not controlled by a brain):
    tail, abdomen, neck/head, extra arms. points: joint positions of each segment + the chain's end point (body frame).
    joints[i]: list of (axis, stiffness Nm/rad, springref deg) of segment i; bones[i]: Blender bone of segment i."""
    name: str
    points: list
    joints: list
    masses: list
    radius: list
    bones: list
    damping_ratio: float = 0.35
    range_deg: float = 45.0

    def joint_names(self):
        return [f"{self.name}_p{i}{'xyz'[k]}" for i, seg in enumerate(self.joints) for k in range(len(seg))]


@dataclass
class BodySpec:
    name: str
    mass: float                 # kg, total
    torso_geoms: list           # MJCF geom strings for the torso (body frame)
    legs: list = field(default_factory=list)
    kp: float = 30.0            # position actuator gain, Nm/rad
    max_torque: float = 15.0    # Nm
    joint_damping: float = 0.5
    action_scale: float = 0.6   # rad, action +-1 -> +-this around the rest pose
    stand_height: float = 0.0   # torso origin height when standing in the rest pose
    passive: list = field(default_factory=list)     # PassiveChain parts (stage 1 "alive" bodies)

    @property
    def joint_names(self):
        return [n for leg in self.legs for n in leg.joint_names()]

    @property
    def foot_names(self):
        return [f"{leg.name}_foot" for leg in self.legs]

    @property
    def passive_joint_names(self):
        return [n for ch in self.passive for n in ch.joint_names()]


def _mirror(p):
    p = np.array(p, dtype=float)
    p[..., 1] *= -1
    return p


def _lift_axis(points, side):
    """Horizontal axis perpendicular to the leg's outward direction; positive rotation lifts the foot."""
    d = points[3, :2] - points[0, :2]
    d /= np.linalg.norm(d)
    return np.array([d[1], -d[0], 0.0])


# ------------------------------------------------------------------------------------------------
# Myrranox - joint positions measured on the mesh (left side), see blender measurement in the chat log
# ------------------------------------------------------------------------------------------------
_MYR_LEFT = {
    "front": [(0.335, 0.134, 0.122), (0.338, 0.207, 0.173), (0.442, 0.320, 0.053), (0.520, 0.416, 0.026)],
    "mid":   [(0.209, 0.132, 0.152), (0.223, 0.222, 0.177), (0.225, 0.375, 0.046), (0.254, 0.483, 0.007)],
    "back":  [(0.141, 0.102, 0.145), (0.095, 0.146, 0.179), (-0.014, 0.256, 0.060), (-0.094, 0.362, 0.020)],
}


PITCH, YAW = np.array([0, 1.0, 0]), np.array([0, 0, 1.0])


def myrranox(passive=False):
    """passive=True: abdomen and head hang on spring-damper joints and swing with the gait (stage 1)."""
    legs = []
    for pair, pts in _MYR_LEFT.items():
        for side, sgn in (("L", 1), ("R", -1)):
            p = np.array(pts) if sgn > 0 else _mirror(pts)
            lift = _lift_axis(p, sgn)
            legs.append(Leg(f"{pair}_{side}", p, [(0, np.array([0, 0, 1.0]), (-35, 35)),
                                                  (1, lift, (-40, 60)), (2, lift, (-50, 50))], 0.018))
    torso = [
        '<geom name="thorax" type="capsule" fromto="0.06 0 0.19 0.62 0 0.19" size="0.1" mass="3.5"/>',
    ]
    chains = []
    if passive:
        chains = [
            # petiole joint measured between Abdomen_low and the torso; the abdomen bobs and sways behind the body
            PassiveChain("abdomen", [(0.068, 0, 0.174), (-0.66, 0, 0.17)],
                         [[(PITCH, 16.0, MYR_SPRINGREF["abdomen_p0x"]), (YAW, 8.0, 0.0)]],
                         masses=[1.2], radius=[0.09], bones=["abdomen"], damping_ratio=0.2),
            # head + mandibles on the neck joint between Head_low and the torso
            PassiveChain("head", [(0.507, 0, 0.165), (0.75, 0, 0.13)],
                         [[(PITCH, 12.0, MYR_SPRINGREF["head_p0x"]), (YAW, 6.0, 0.0)]],
                         masses=[1.0], radius=[0.05], bones=["head"], damping_ratio=0.25),
        ]
    else:
        torso += [
            '<geom name="head" type="ellipsoid" pos="0.60 0 0.15" size="0.1 0.08 0.06" mass="0.8"/>',
            '<geom name="mandibles" type="box" pos="0.70 0 0.095" size="0.05 0.08 0.02" mass="0.2"/>',
            '<geom name="abdomen" type="ellipsoid" pos="-0.27 0 0.18" size="0.40 0.16 0.11" mass="1.2"/>',
        ]
    return BodySpec("myrranox", mass=8.0, torso_geoms=torso, legs=legs,
                    kp=60.0, max_torque=20.0, joint_damping=0.5, action_scale=0.6, stand_height=0.0, passive=chains)


# ------------------------------------------------------------------------------------------------
# DC_SKIN_RAW - symmetric quadruped approximation, joint points read off an orthographic side render
# ------------------------------------------------------------------------------------------------
_DC_LEFT = {
    #          hip / shoulder         knee / elbow           ankle / wrist          foot
    "hind":  [(-0.15, 0.17, 0.50), (-0.38, 0.20, 0.30), (-0.45, 0.20, 0.12), (-0.36, 0.20, 0.04)],
    "front": [(0.40, 0.20, 0.56), (0.48, 0.24, 0.28), (0.56, 0.26, 0.12), (0.68, 0.26, 0.04)],
}


def dc(passive=False):
    """passive=True: tail (2 segments), neck + head and the two extra arms on spring-damper joints (stage 1).
    The chain points are the DC_rig bones tail_1/2, neck, head, arm_L/R, so the proxy mesh follows directly."""
    legs = []
    for pair, pts in _DC_LEFT.items():
        for side, sgn in (("L", 1), ("R", -1)):
            p = np.array(pts) if sgn > 0 else _mirror(pts)
            abd = np.array([1.0, 0, 0]) * sgn          # positive = leg outward
            pitch = np.array([0, 1.0, 0])              # positive = foot backwards
            legs.append(Leg(f"{pair}_{side}", p, [(0, abd, (-25, 25)), (0, pitch, (-60, 60)),
                                                  (1, pitch, (-70, 70)), (2, pitch, (-60, 60))], 0.045))
    torso = [
        '<geom name="chest" type="capsule" fromto="-0.25 0 0.66 0.50 0 0.74" size="0.21" mass="35"/>',
    ]
    chains = []
    if passive:
        chains = [
            PassiveChain("tail", [(-0.30, 0, 0.60), (-0.80, 0, 0.50), (-1.35, 0, 0.42)],
                         [[(PITCH, 60.0, DC_SPRINGREF["tail_p0x"]), (YAW, 25.0, 0.0)],
                          [(PITCH, 18.0, DC_SPRINGREF["tail_p1x"]), (YAW, 6.0, 0.0)]],
                         masses=[2.5, 1.5], radius=[0.07, 0.05], bones=["tail_1", "tail_2"], damping_ratio=0.15),
            PassiveChain("neck", [(0.50, 0, 0.80), (1.00, 0, 0.90), (1.32, 0, 0.80)],
                         [[(PITCH, 400.0, DC_SPRINGREF["neck_p0x"]), (YAW, 250.0, 0.0)],
                          [(PITCH, 60.0, DC_SPRINGREF["neck_p1x"])]],
                         masses=[5.0, 4.0], radius=[0.10, 0.12], bones=["neck", "head"]),
        ]
        for side, sgn in (("L", 1), ("R", -1)):
            chains.append(PassiveChain(f"arm_{side}", [(0.62, 0.20 * sgn, 0.62), (0.98, 0.22 * sgn, 0.24)],
                                       [[(PITCH, 10.0, DC_SPRINGREF["arm_L_p0x"]), (np.array([1.0, 0, 0]) * sgn, 6.0, 0.0)]],
                                       masses=[1.0], radius=[0.05], bones=[f"arm_{side}"], damping_ratio=0.25))
    else:
        torso += [
            '<geom name="arms" type="capsule" fromto="0.76 0 0.44 0.98 0 0.24" size="0.07" mass="2"/>',
            '<geom name="neck" type="capsule" fromto="0.65 0 0.85 1.00 0 0.92" size="0.10" mass="5"/>',
            '<geom name="head" type="ellipsoid" pos="1.15 0 0.82" size="0.16 0.09 0.12" mass="4"/>',
            '<geom name="tail" type="capsule" fromto="-0.45 0 0.55 -1.30 0 0.44" size="0.06" mass="4"/>',
        ]
    return BodySpec("dc", mass=60.0, torso_geoms=torso, legs=legs,
                    kp=300.0, max_torque=150.0, joint_damping=3.0, action_scale=0.6, stand_height=0.0, passive=chains)


# Spring rest angles (deg) that cancel the sag under gravity, so the parts rest in the modelled pose.
# Found with calibrate_springref() (python -m sim.bodies --springref).
MYR_SPRINGREF = {"abdomen_p0x": 15.34, "head_p0x": -5.68}
DC_SPRINGREF = {"tail_p0x": 16.85, "tail_p1x": 12.91, "neck_p0x": -5.5, "neck_p1x": -5.85, "arm_L_p0x": -9.19}

BODIES = {"myrranox": myrranox, "dc": dc}


def _fmt(v):
    return " ".join(f"{x:.4f}" for x in v)


def _passive_xml(ch):
    """Nested bodies with spring-damper joints (no actuators). Damping from the damping ratio and a rough inertia."""
    xml, closes = "", ""
    pts = [np.asarray(p, dtype=float) for p in ch.points]
    for i, seg in enumerate(ch.joints):
        origin = pts[i] - (pts[i - 1] if i else 0)
        vec = pts[i + 1] - pts[i]
        length = float(np.linalg.norm(vec))
        mass_out = sum(ch.masses[i:])                       # this joint carries all segments beyond it
        inertia = mass_out * (0.5 * length) ** 2 + ch.masses[i] * length ** 2 / 12
        joints = ""
        for k, (axis, stiff, ref) in enumerate(seg):
            damp = 2 * ch.damping_ratio * np.sqrt(stiff * inertia)
            joints += (f'<joint name="{ch.name}_p{i}{"xyz"[k]}" axis="{_fmt(axis)}" stiffness="{stiff}" '
                       f'springref="{ref}" damping="{damp:.4f}" armature="0.005" limited="true" '
                       f'range="{-ch.range_deg} {ch.range_deg}"/>')
        xml += (f'<body name="{ch.name}_s{i}" pos="{_fmt(origin)}">{joints}'
                f'<geom type="capsule" fromto="0 0 0 {_fmt(vec)}" size="{ch.radius[i]}" mass="{ch.masses[i]}" '
                f'class="creature"/>')
        closes += "</body>"
    return xml + closes


def calibrate_springref(species):
    """Settle the body with the rest pose held and read how far each passive joint sags; springref = -sag."""
    import mujoco
    spec = BODIES[species](passive=True)
    m = mujoco.MjModel.from_xml_string(build_mjcf(spec))
    d = mujoco.MjData(m)
    for _ in range(1500):
        mujoco.mj_step(m, d)
    out = {}
    for ch in spec.passive:
        for i, seg in enumerate(ch.joints):
            if abs(seg[0][0][1]) > 0.5:                     # first joint of each segment is the pitch joint
                name = f"{ch.name}_p{i}x"
                q = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]]
                out[name] = round(-float(np.degrees(q)), 2)
    return out


def _prefix_names(xml, prefix):
    return xml.replace('name="', f'name="{prefix}') if prefix else xml


def start_height(spec):
    """Torso height at which the rest-pose feet just touch the floor."""
    foot_bottom = min(leg.points[3, 2] - leg.radius * 1.2 for leg in spec.legs)
    return spec.stand_height - foot_bottom + 0.005


def creature_xml(spec: BodySpec, prefix="", pos=(0.0, 0.0), yaw=0.0, leg_mass_fraction=0.25):
    """(body XML, actuator XML) for one creature. All names get `prefix`, so several creatures fit in one world.
    The creature's own defaults (joint damping, collision classes) are set on its elements."""
    leg_mass = spec.mass * leg_mass_fraction / len(spec.legs) / 3
    damp = f'damping="{spec.joint_damping}" armature="0.01" limited="true"'
    legs_xml = []
    for leg in spec.legs:
        p = leg.points
        # Nested bodies: each segment's body sits at its joint; geometry uses the measured points.
        xml = ""
        closes = ""
        for i in range(3):
            origin = p[i] - (p[i - 1] if i else 0)
            seg_from, seg_to = np.zeros(3), p[i + 1] - p[i]
            joints = "".join(f'<joint name="{leg.name}_j{k}" axis="{_fmt(axis)}" range="{lo} {hi}" {damp}/>'
                             for k, (seg, axis, (lo, hi)) in enumerate(leg.joints) if seg == i)
            xml += (f'<body name="{leg.name}_s{i}" pos="{_fmt(origin)}">{joints}'
                    f'<geom type="capsule" fromto="{_fmt(seg_from)} {_fmt(seg_to)}" size="{leg.radius}" '
                    f'mass="{leg_mass:.4f}" class="creature"/>')
            closes += "</body>"
        foot = p[3] - p[2]
        xml += (f'<geom name="{leg.name}_foot" type="sphere" pos="{_fmt(foot)}" size="{leg.radius * 1.2:.4f}" '
                f'mass="{leg_mass * 0.2:.4f}" friction="1.2 0.02 0.001" class="creature"/>'
                f'<site name="{leg.name}_foot" pos="{_fmt(foot)}" size="0.01"/>')
        legs_xml.append(xml + closes)

    torso = [g.replace("<geom ", '<geom class="creature" ') for g in spec.torso_geoms]
    torso += [_passive_xml(ch) for ch in spec.passive]
    quat = f"{np.cos(yaw / 2):.6f} 0 0 {np.sin(yaw / 2):.6f}"
    body = (f'<body name="torso" pos="{pos[0]:.4f} {pos[1]:.4f} {start_height(spec):.4f}" quat="{quat}">'
            f'<freejoint name="root"/><site name="torso_site" pos="0 0 0" size="0.01"/>'
            f'{"".join(torso)}{"".join(legs_xml)}</body>')
    actuators = "".join(
        f'<position name="{prefix}{j}" joint="{prefix}{j}" kp="{spec.kp}" '
        f'forcerange="{-spec.max_torque} {spec.max_torque}"/>' for j in spec.joint_names)
    return _prefix_names(body, prefix), actuators


def _habitat_xml(habitat):
    """Walls and rocks of sim/habitat.json as static collision geometry (same shapes as build_habitat.py)."""
    if habitat is None:
        return ""
    xml = []
    t, h = 1.0, 3.0
    cx, cy = (habitat.xmax + habitat.xmin) / 2, (habitat.ymax + habitat.ymin) / 2
    w, d = habitat.xmax - habitat.xmin, habitat.ymax - habitat.ymin
    for name, (x, y), (sx, sy) in (("wall_N", (cx, habitat.ymax + t / 2), (w / 2 + t, t / 2)),
                                   ("wall_S", (cx, habitat.ymin - t / 2), (w / 2 + t, t / 2)),
                                   ("wall_E", (habitat.xmax + t / 2, cy), (t / 2, d / 2)),
                                   ("wall_W", (habitat.xmin - t / 2, cy), (t / 2, d / 2))):
        xml.append(f'<geom name="{name}" type="box" pos="{x} {y} {h / 2}" size="{sx} {sy} {h / 2}" '
                   f'class="scenery" rgba="0.28 0.25 0.22 1"/>')
    for i, o in enumerate(habitat.obstacles):
        xml.append(f'<geom name="rock_{i}" type="ellipsoid" pos="{o.x} {o.y} {0.2 * o.r:.3f}" '
                   f'size="{o.r} {o.r} {0.8 * o.r:.3f}" class="scenery" rgba="0.35 0.35 0.35 1"/>')
    return "\n    ".join(xml)


def build_world_mjcf(creatures, habitat=None, name="world"):
    """MJCF world: floor (+ walls and rocks from a Habitat) + several creatures.
    creatures: list of (BodySpec, prefix, (x, y), yaw). Creatures collide with floor and scenery, not with
    each other (contype 2 / conaffinity 1)."""
    bodies, acts = [], []
    for spec, prefix, pos, yaw in creatures:
        b, a = creature_xml(spec, prefix, pos, yaw)
        bodies.append(b)
        acts.append(a)
    return f"""
<mujoco model="{name}">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.005" integrator="implicitfast"/>
  <default>
    <default class="creature"><geom contype="2" conaffinity="1" friction="0.9 0.02 0.001" rgba="0.7 0.6 0.5 1"/></default>
    <default class="scenery"><geom contype="1" conaffinity="1" friction="0.9 0.02 0.001"/></default>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.45 0.40 0.33" rgb2="0.36 0.32 0.26"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.1" class="scenery" material="grid"/>
    {_habitat_xml(habitat)}
    {"".join(bodies)}
  </worldbody>
  <actuator>{"".join(acts)}</actuator>
</mujoco>
"""


def build_mjcf(spec: BodySpec):
    """Training world: flat floor + one creature (no prefix) - used by LocomotionEnv."""
    return build_world_mjcf([(spec, "", (0.0, 0.0), 0.0)], name=spec.name)


if __name__ == "__main__":
    import mujoco
    for name, make in BODIES.items():
        spec = make()
        m = mujoco.MjModel.from_xml_string(build_mjcf(spec))
        d = mujoco.MjData(m)
        # Hold the rest pose for 3 s and check the body stays up.
        for _ in range(600):
            d.ctrl[:] = 0
            mujoco.mj_step(m, d)
        print(f"{name}: mass {m.body_subtreemass[1]:.1f} kg, nu {m.nu}, torso z {d.qpos[2]:.3f} m, "
              f"upright {d.qpos[3]:.3f} (quat w)")
