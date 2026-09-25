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

    @property
    def joint_names(self):
        return [n for leg in self.legs for n in leg.joint_names()]

    @property
    def foot_names(self):
        return [f"{leg.name}_foot" for leg in self.legs]


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


def myrranox():
    legs = []
    for pair, pts in _MYR_LEFT.items():
        for side, sgn in (("L", 1), ("R", -1)):
            p = np.array(pts) if sgn > 0 else _mirror(pts)
            lift = _lift_axis(p, sgn)
            legs.append(Leg(f"{pair}_{side}", p, [(0, np.array([0, 0, 1.0]), (-35, 35)),
                                                  (1, lift, (-40, 60)), (2, lift, (-50, 50))], 0.018))
    torso = [
        '<geom name="thorax" type="capsule" fromto="0.06 0 0.19 0.62 0 0.19" size="0.1" mass="3.5"/>',
        '<geom name="head" type="ellipsoid" pos="0.60 0 0.15" size="0.1 0.08 0.06" mass="0.8"/>',
        '<geom name="mandibles" type="box" pos="0.70 0 0.095" size="0.05 0.08 0.02" mass="0.2"/>',
        '<geom name="abdomen" type="ellipsoid" pos="-0.27 0 0.18" size="0.40 0.16 0.11" mass="1.2"/>',
    ]
    return BodySpec("myrranox", mass=8.0, torso_geoms=torso, legs=legs,
                    kp=60.0, max_torque=20.0, joint_damping=0.5, action_scale=0.6, stand_height=0.0)


# ------------------------------------------------------------------------------------------------
# DC_SKIN_RAW - symmetric quadruped approximation, joint points read off an orthographic side render
# ------------------------------------------------------------------------------------------------
_DC_LEFT = {
    #          hip / shoulder         knee / elbow           ankle / wrist          foot
    "hind":  [(-0.15, 0.17, 0.50), (-0.38, 0.20, 0.30), (-0.45, 0.20, 0.12), (-0.36, 0.20, 0.04)],
    "front": [(0.40, 0.20, 0.56), (0.48, 0.24, 0.28), (0.56, 0.26, 0.12), (0.68, 0.26, 0.04)],
}


def dc():
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
        '<geom name="arms" type="capsule" fromto="0.76 0 0.44 0.98 0 0.24" size="0.07" mass="2"/>',
        '<geom name="neck" type="capsule" fromto="0.65 0 0.85 1.00 0 0.92" size="0.10" mass="5"/>',
        '<geom name="head" type="ellipsoid" pos="1.15 0 0.82" size="0.16 0.09 0.12" mass="4"/>',
        '<geom name="tail" type="capsule" fromto="-0.45 0 0.55 -1.30 0 0.44" size="0.06" mass="4"/>',
    ]
    return BodySpec("dc", mass=60.0, torso_geoms=torso, legs=legs,
                    kp=300.0, max_torque=150.0, joint_damping=3.0, action_scale=0.6, stand_height=0.0)


BODIES = {"myrranox": myrranox, "dc": dc}


def _fmt(v):
    return " ".join(f"{x:.4f}" for x in v)


def build_mjcf(spec: BodySpec, leg_mass_fraction=0.25):
    """MJCF string: floor + torso with free joint + 3-segment legs (nested bodies) + position actuators."""
    leg_mass = spec.mass * leg_mass_fraction / len(spec.legs) / 3
    legs_xml = []
    for leg in spec.legs:
        p = leg.points
        # Nested bodies: each segment's body sits at its joint; geometry uses the measured points.
        xml = ""
        closes = ""
        for i in range(3):
            origin = p[i] - (p[i - 1] if i else 0)
            seg_from, seg_to = np.zeros(3), p[i + 1] - p[i]
            joints = "".join(f'<joint name="{leg.name}_j{k}" axis="{_fmt(axis)}" range="{lo} {hi}"/>'
                             for k, (seg, axis, (lo, hi)) in enumerate(leg.joints) if seg == i)
            xml += (f'<body name="{leg.name}_s{i}" pos="{_fmt(origin)}">{joints}'
                    f'<geom type="capsule" fromto="{_fmt(seg_from)} {_fmt(seg_to)}" size="{leg.radius}" '
                    f'mass="{leg_mass:.4f}"/>')
            closes += "</body>"
        foot = p[3] - p[2]
        xml += (f'<geom name="{leg.name}_foot" type="sphere" pos="{_fmt(foot)}" size="{leg.radius * 1.2:.4f}" '
                f'mass="{leg_mass * 0.2:.4f}" friction="1.2 0.02 0.001"/>'
                f'<site name="{leg.name}_foot" pos="{_fmt(foot)}" size="0.01"/>')
        legs_xml.append(xml + closes)

    foot_bottom = min(leg.points[3, 2] - leg.radius * 1.2 for leg in spec.legs)
    start_z = spec.stand_height - foot_bottom + 0.005
    actuators = "\n".join(
        f'<position name="{j}" joint="{j}" kp="{spec.kp}" forcerange="{-spec.max_torque} {spec.max_torque}"/>'
        for j in spec.joint_names)

    return f"""
<mujoco model="{spec.name}">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.005" integrator="implicitfast"/>
  <default>
    <joint damping="{spec.joint_damping}" armature="0.01" limited="true"/>
    <geom contype="2" conaffinity="1" friction="0.9 0.02 0.001" rgba="0.7 0.6 0.5 1"/>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.45 0.40 0.33" rgb2="0.36 0.32 0.26"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.1" contype="1" conaffinity="1" material="grid"/>
    <body name="torso" pos="0 0 {start_z:.4f}">
      <freejoint name="root"/>
      <site name="torso_site" pos="0 0 0" size="0.01"/>
      {''.join(spec.torso_geoms)}
      {''.join(legs_xml)}
    </body>
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


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
