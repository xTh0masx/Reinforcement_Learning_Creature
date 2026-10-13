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
    seg_radius: tuple = None    # per-segment capsule radius (fitted to the mesh); default: radius
    mass_scale: float = 1.0     # segment mass relative to a leg (the DC's arms are lighter)
    action_scale: float = None  # rad per action unit for this limb's joints; default: the body's

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
    # Stage 2: an actuated chain keeps its springs (they carry the weight) and gets a position motor per joint
    actuated: bool = False
    kp: float = 0.0
    max_torque: float = 0.0
    action_scale: float = 0.3
    parent: str = "torso"          # body the chain hangs on ("torso" or the rear segment)

    def joint_names(self):
        return [f"{self.name}_p{i}{'xyz'[k]}" for i, seg in enumerate(self.joints) for k in range(len(seg))]

    def joint_axes(self):
        return [np.asarray(a, float) for seg in self.joints for (a, _, _) in seg]


@dataclass
class RearSegment:
    """Stage 2 DC spine: the rear body (hips, hind legs, tail) hangs on an actuated spine joint behind the chest.
    joints: (name suffix, axis, stiffness Nm/rad, springref deg, range deg); geoms: (from, to, radius, mass), body frame."""
    name: str
    point: tuple
    joints: list
    geoms: list
    legs: list
    kp: float
    max_torque: float
    action_scale: float
    bone: str = "spine_rear"
    damping: float = 20.0

    def joint_names(self):
        return [f"{self.name}_{j[0]}" for j in self.joints]

    def joint_axes(self):
        return [np.asarray(j[1], float) for j in self.joints]


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
    rear: object = None                              # RearSegment (stage 2 DC spine)
    head_body: str = ""                              # body whose steadiness is rewarded (stage 2)
    hulls: dict = None      # walking v8 (DC): {bone: {"vertices": body-frame points, "volume": m^3}} from the mesh
    lean: bool = False      # walking v11 (DC): big-cat mass distribution + rounded paw pads (see dc(lean=True))

    @property
    def joint_names(self):
        """Actuated joints in actuator order: legs, spine, actuated chains (head, neck, abdomen)."""
        names = [n for leg in self.legs for n in leg.joint_names()]
        if self.rear is not None:
            names += self.rear.joint_names()
        return names + [n for ch in self.passive if ch.actuated for n in ch.joint_names()]

    def actuator_params(self):
        """Per actuated joint: (kp, max torque, action scale rad), same order as joint_names."""
        out = [(self.kp, self.max_torque, leg.action_scale or self.action_scale) for leg in self.legs for _ in leg.joints]
        if self.rear is not None:
            out += [(self.rear.kp, self.rear.max_torque, self.rear.action_scale)] * len(self.rear.joints)
        for ch in self.passive:
            if ch.actuated:
                out += [(ch.kp, ch.max_torque, ch.action_scale)] * len(ch.joint_names())
        return out

    def joint_axis(self, name):
        for leg in self.legs:
            for n, (_, axis, _) in zip(leg.joint_names(), leg.joints):
                if n == name:
                    return np.asarray(axis, float)
        if self.rear is not None and name in self.rear.joint_names():
            return self.rear.joint_axes()[self.rear.joint_names().index(name)]
        for ch in self.passive:
            if name in ch.joint_names():
                return ch.joint_axes()[ch.joint_names().index(name)]
        raise KeyError(name)

    @property
    def foot_names(self):
        return [f"{leg.name}_foot" for leg in self.legs]

    @property
    def passive_joint_names(self):
        """Spring joints without a motor (recorded and played back, not controlled)."""
        return [n for ch in self.passive if not ch.actuated for n in ch.joint_names()]


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


def myrranox(passive=False, active=False, hulls=False):
    """passive=True: abdomen and head hang on spring-damper joints and swing with the gait (stage 1).
    active=True (stage 2, walking v3): head and abdomen also get motors the brain controls; springs stay.
    hulls=True (walking v9): every part is a convex hull cut from the real Myrranox mesh (sim/hulls.py) instead of
    capsules; the claws are the feet; the 8 kg are distributed by hull volume."""
    passive = passive or active or hulls
    SR = MYR_HULL_SPRINGREF if hulls else MYR_SPRINGREF      # spring rest angles for the part masses in use
    legs = []
    for pair, pts in _MYR_LEFT.items():
        for side, sgn in (("L", 1), ("R", -1)):
            p = np.array(pts) if sgn > 0 else _mirror(pts)
            lift = _lift_axis(p, sgn)
            legs.append(Leg(f"{pair}_{side}", p, [(0, np.array([0, 0, 1.0]), (-35, 35)),
                                                  (1, lift, (-40, 60)), (2, lift, (-50, 50))], 0.018,
                            # mesh body (v9): the brain may use the knees' full range (+-50 deg) - to carry the
                            # heavy abdomen the knees wanted more than +-34 deg and got stuck at the limit
                            action_scale=0.87 if hulls else None))
    torso = [
        '<geom name="thorax" type="capsule" fromto="0.06 0 0.19 0.62 0 0.19" size="0.1" mass="3.5"/>',
    ]
    chains = []
    if passive:
        chains = [
            # petiole joint measured between Abdomen_low and the torso; the abdomen bobs and sways behind the body
            PassiveChain("abdomen", [(0.068, 0, 0.174), (-0.66, 0, 0.17)],
                         [[(PITCH, 16.0, SR["abdomen_p0x"]), (YAW, 8.0, 0.0)]],
                         masses=[1.2], radius=[0.09], bones=["abdomen"], damping_ratio=0.2),
            # head + mandibles on the neck joint between Head_low and the torso
            PassiveChain("head", [(0.507, 0, 0.165), (0.75, 0, 0.13)],
                         [[(PITCH, 12.0, SR["head_p0x"]), (YAW, 6.0, 0.0)]],
                         masses=[1.0], radius=[0.05], bones=["head"], damping_ratio=0.25),
        ]
        if active:
            chains[0].actuated, chains[0].kp, chains[0].max_torque, chains[0].action_scale = True, 20.0, 8.0, 0.3
            chains[1].actuated, chains[1].kp, chains[1].max_torque, chains[1].action_scale = True, 10.0, 4.0, 0.4
    else:
        torso += [
            '<geom name="head" type="ellipsoid" pos="0.60 0 0.15" size="0.1 0.08 0.06" mass="0.8"/>',
            '<geom name="mandibles" type="box" pos="0.70 0 0.095" size="0.05 0.08 0.02" mass="0.2"/>',
            '<geom name="abdomen" type="ellipsoid" pos="-0.27 0 0.18" size="0.40 0.16 0.11" mass="1.2"/>',
        ]
    hull_data = None
    if hulls:
        from sim.hulls import load
        hull_data = load("myrranox")
        torso = []                                   # thorax, head and abdomen shapes come from the hulls
    return BodySpec("myrranox", mass=8.0, torso_geoms=torso, legs=legs, hulls=hull_data,
                    kp=60.0, max_torque=20.0, joint_damping=0.5, action_scale=0.6, stand_height=0.0, passive=chains,
                    head_body="head_s0" if active else "")


# ------------------------------------------------------------------------------------------------
# DC_SKIN_RAW - symmetric quadruped approximation, joint points read off an orthographic side render
# ------------------------------------------------------------------------------------------------
_DC_LEFT = {
    #          hip / shoulder         knee / elbow           ankle / wrist          foot
    "hind":  [(-0.15, 0.17, 0.50), (-0.38, 0.20, 0.30), (-0.45, 0.20, 0.12), (-0.36, 0.20, 0.04)],
    "front": [(0.40, 0.20, 0.56), (0.48, 0.24, 0.28), (0.56, 0.26, 0.12), (0.68, 0.26, 0.04)],
}


# DC walking v6 ("six-limbed panther"): the extra arms as real limbs - points traced along the DC mesh arm
# (shoulder, elbow, wrist, claw tip), measured on DC_body_proxy on 30 Sep 2026
_DC_ARM_LEFT = [(0.62, 0.20, 0.61), (0.79, 0.19, 0.44), (1.02, 0.20, 0.32), (1.04, 0.20, 0.17)]
DC_LEG_RADII = {"hind": (0.075, 0.05, 0.04), "front": (0.07, 0.05, 0.04), "arm": (0.055, 0.045, 0.04)}


def dc(passive=False, active=False, arms=False, hulls=False, lean=False):
    """passive=True: tail (2 segments), neck + head and the two extra arms on spring-damper joints (stage 1).
    The chain points are the DC_rig bones tail_1/2, neck, head, arm_L/R, so the proxy mesh follows directly.
    active=True (stage 2, walking v3): a spine joint splits the chest - the rear body carries hind legs and tail -
    and spine, neck and head get motors the brain controls; tail and arms stay passive.
    arms=True (walking v6): the extra arms become actuated limbs (shoulder 2, elbow, wrist, claws touch the ground),
    legs sway sideways only +-8 deg (paws under the body like a panther), segment radii fitted to the mesh.
    hulls=True (walking v8): every part is a convex hull cut from the real DC mesh (sim/hulls.py) instead of
    capsules; the paws (with claws) are the feet; the 60 kg are distributed by hull volume."""
    passive = passive or active or arms
    SR = DC_LEAN_SPRINGREF if lean else DC_HULL_SPRINGREF if hulls else DC_SPRINGREF   # spring rest angles
    active = active or arms
    legs = []
    abd_range = (-8, 8) if arms else (-25, 25)
    for pair, pts in _DC_LEFT.items():
        for side, sgn in (("L", 1), ("R", -1)):
            p = np.array(pts) if sgn > 0 else _mirror(pts)
            abd = np.array([1.0, 0, 0]) * sgn          # positive = leg outward
            pitch = np.array([0, 1.0, 0])              # positive = foot backwards
            legs.append(Leg(f"{pair}_{side}", p, [(0, abd, abd_range), (0, pitch, (-60, 60)),
                                                  (1, pitch, (-70, 70)), (2, pitch, (-60, 60))], 0.045,
                            seg_radius=DC_LEG_RADII[pair] if arms else None))
    if arms:
        for side, sgn in (("L", 1), ("R", -1)):
            p = np.array(_DC_ARM_LEFT) if sgn > 0 else _mirror(_DC_ARM_LEFT)
            abd = np.array([1.0, 0, 0]) * sgn
            pitch = np.array([0, 1.0, 0])
            legs.append(Leg(f"arm_{side}", p, [(0, abd, (-15, 15)), (0, pitch, (-100, 80)),
                                               (1, pitch, (-110, 110)), (2, pitch, (-90, 90))], 0.04,
                            seg_radius=DC_LEG_RADII["arm"], mass_scale=0.5,
                            # mesh body: +-103 deg - folding the arms needs elbow 93 / wrist 66 deg (13 Oct)
                            action_scale=1.8 if hulls else 1.0))
    torso = [
        '<geom name="chest" type="capsule" fromto="-0.25 0 0.66 0.50 0 0.74" size="0.21" mass="35"/>',
    ]
    if active:
        torso = ['<geom name="chest" type="capsule" fromto="0.12 0 0.70 0.50 0 0.74" size="0.21" mass="17.5"/>']
    chains = []
    if passive:
        chains = [
            PassiveChain("tail", [(-0.30, 0, 0.60), (-0.80, 0, 0.50), (-1.35, 0, 0.42)],
                         [[(PITCH, 60.0, SR["tail_p0x"]), (YAW, 25.0, 0.0)],
                          [(PITCH, 18.0, SR["tail_p1x"]), (YAW, 6.0, 0.0)]],
                         masses=[2.5, 1.5], radius=[0.07, 0.05], bones=["tail_1", "tail_2"], damping_ratio=0.15),
            PassiveChain("neck", [(0.50, 0, 0.80), (1.00, 0, 0.90), (1.32, 0, 0.80)],
                         [[(PITCH, 400.0, SR["neck_p0x"]), (YAW, 250.0, 0.0)],
                          [(PITCH, 60.0, SR["neck_p1x"])]],
                         masses=[5.0, 4.0], radius=[0.10, 0.12], bones=["neck", "head"]),
        ]
        for side, sgn in (("L", 1), ("R", -1)) if not arms else ():
            chains.append(PassiveChain(f"arm_{side}", [(0.62, 0.20 * sgn, 0.62), (0.98, 0.22 * sgn, 0.24)],
                                       [[(PITCH, 10.0, SR["arm_L_p0x"]), (np.array([1.0, 0, 0]) * sgn, 6.0, 0.0)]],
                                       masses=[1.0], radius=[0.05], bones=[f"arm_{side}"], damping_ratio=0.25))
        if active:
            neck = chains[1]
            neck.actuated, neck.kp, neck.max_torque, neck.action_scale = True, 300.0, 150.0, 0.3
            chains[0].parent = "spine"                   # the tail hangs on the rear body
    else:
        torso += [
            '<geom name="arms" type="capsule" fromto="0.76 0 0.44 0.98 0 0.24" size="0.07" mass="2"/>',
            '<geom name="neck" type="capsule" fromto="0.65 0 0.85 1.00 0 0.92" size="0.10" mass="5"/>',
            '<geom name="head" type="ellipsoid" pos="1.15 0 0.82" size="0.16 0.09 0.12" mass="4"/>',
            '<geom name="tail" type="capsule" fromto="-0.45 0 0.55 -1.30 0 0.44" size="0.06" mass="4"/>',
        ]
    hull_data = None
    if hulls:
        from sim.hulls import load
        hull_data = load("dc")
        torso = []                                   # the torso / rear / neck / tail shapes come from the hulls
    return BodySpec("dc", mass=60.0, torso_geoms=torso, legs=legs, hulls=hull_data, lean=lean and hulls,
                    kp=300.0, max_torque=150.0, joint_damping=3.0, action_scale=0.6, stand_height=0.0, passive=chains,
                    rear=RearSegment("spine", (0.12, 0, 0.70),
                                     [("px", PITCH, 1500.0, SR.get("spine_px", 0.0), 20.0),
                                      ("py", YAW, 800.0, 0.0, 20.0)],
                                     [((0.0, 0, 0.0), (-0.37, 0, -0.04), 0.21, 17.5)],
                                     ["hind_L", "hind_R"], kp=600.0, max_torque=400.0, action_scale=0.25)
                    if active else None,
                    head_body="neck_s1" if active else "")


# Relative density of body parts (default 1). An insect's abdomen (gaster: air sacs, crop) is much lighter than its
# muscle-filled thorax; mass by plain volume gave the Myrranox abdomen 54 % of the weight, the centre of mass sat over
# the hind legs and even a perfect tripod rhythm tipped it back with the front feet in the air (13 Oct).
# 13 Oct (2): at half density the centre of mass still sat so far back that the front legs carried almost nothing and
# hung in the air during the tripod gait (front claws 1.4 cm above the ground in stance); at 0.25 the front feet are
# down 40 % of their stance time with the rhythm alone (was 19 %), body level, abdomen 23 % of the weight.
HULL_DENSITY = {"myrranox": {"abdomen": 0.25}}


# DC walking v11 ("lean" body, owner 13 Oct: "the DC mesh is too clunky to move fluidly"): mass by volume made the DC
# extremely head-heavy (neck + head 21 %, trunk 37 %, centre of mass 32 cm in front of mid-body, almost over the front
# paws). Big-cat distribution instead: trunk 55 %, neck + head 9 %, front legs 12 %, hind legs 15 %, arms 6 %, tail 3 %
# (each limb: upper 60 %, middle 28 %, paw 12 %). The hull shapes stay; only the masses change.
_LIMB = (0.60, 0.28, 0.12)
DC_LEAN_MASS = {"torso": 0.32, "spine_rear": 0.23, "neck": 0.05, "head": 0.04, "tail_1": 0.02, "tail_2": 0.01}
for _side in "LR":
    for _limb, _share in (("hind", 0.075), ("front", 0.06), ("arm", 0.03)):
        for _i, _f in enumerate(_LIMB):
            DC_LEAN_MASS[f"{_limb}_{_side}_s{_i}"] = _share * _f
PAW_PAD_RADIUS = 0.035      # m: rounded pad along the paw sole (contact geometry instead of the claw hull)


def hull_mass(spec, bone):
    """Mass of a hull part: the creature's mass shared by volume x relative density (DC v11: fixed share)."""
    if getattr(spec, "lean", False):
        return spec.mass * DC_LEAN_MASS[bone]
    dens = HULL_DENSITY.get(spec.name, {})
    weighted = {b: h["volume"] * dens.get(b, 1.0) for b, h in spec.hulls.items()}
    return spec.mass * weighted[bone] / sum(weighted.values())


def _hull_geom(spec, bone, origin, name=None, prefix_mesh=True):
    """Mesh geom for a hull: vertices relative to the body origin, mass by volume x density share of the body."""
    mass = hull_mass(spec, bone)
    gname = f' name="{name}"' if name else ""
    return f'<geom{gname} type="mesh" mesh="@MESH@hull_{bone}" mass="{mass:.4f}" class="creature"/>'


def hull_origins(spec):
    """Body-frame origin of the body each hull belongs to (all bodies are unrotated at the rest pose)."""
    o = {"torso": np.zeros(3)}
    for leg in spec.legs:
        for i in range(3):
            o[f"{leg.name}_s{i}"] = np.asarray(leg.points[i], float)
    if spec.rear is not None:
        o[spec.rear.bone] = np.asarray(spec.rear.point, float)
    for ch in spec.passive:
        for i, bone in enumerate(ch.bones):
            o[bone] = np.asarray(ch.points[i], float)
    return o


def creature_assets(spec, prefix=""):
    """Mesh assets of a creature's hull shapes (empty for capsule bodies)."""
    if not spec.hulls:
        return ""
    origins = hull_origins(spec)
    out = []
    for bone, h in spec.hulls.items():
        if bone not in origins:
            continue
        v = np.asarray(h["vertices"]) - origins[bone]
        out.append(f'<mesh name="{prefix}hull_{bone}" vertex="{" ".join(f"{x:.4f}" for x in v.ravel())}"/>')
    return "".join(out)


def _paw_pad(spec, leg):
    """DC v11: rounded pad along the paw sole (from the paw's middle to its tip), its bottom on the paw hull's lowest
    point - the paw rolls over it instead of catching on the claw tips. Returns (from, to, site) in the paw frame."""
    p = np.asarray(leg.points, float)
    a = p[2] + 0.35 * (p[3] - p[2]) - p[2]
    b = p[3] - p[2]
    low = spec.hulls[f"{leg.name}_s2"]["min"][2] - p[2][2]
    dz = (low + PAW_PAD_RADIUS) - min(a[2], b[2])
    a, b = a + [0, 0, dz], b + [0, 0, dz]
    tip = (a if a[2] < b[2] else b) - [0, 0, PAW_PAD_RADIUS]
    return a, b, tip


def _foot_site(spec, leg):
    """Lowest point of the paw hull, relative to the paw body (the foot's contact point at rest)."""
    if getattr(spec, "lean", False):
        return _paw_pad(spec, leg)[2]
    v = np.asarray(spec.hulls[f"{leg.name}_s2"]["vertices"])
    return v[np.argmin(v[:, 2])] - np.asarray(leg.points[2], float)


# Spring rest angles (deg) that cancel the sag under gravity, so the parts rest in the modelled pose.
# Found with calibrate_springref() (python -m sim.bodies --springref).
MYR_SPRINGREF = {"abdomen_p0x": 15.34, "head_p0x": -5.68}
# mesh bodies (walking v8/v9): calibrate_springref(..., hulls=True) residuals added on 13 Oct 2026
MYR_HULL_SPRINGREF = {"abdomen_p0x": 19.30, "head_p0x": -3.86}   # abdomen: 14.3 deg gravity sag (4.32 kg, COM 0.34 m behind the joint, 57.6 Nm/rad) + 5 deg carried up
DC_SPRINGREF = {"spine_px": -1.89, "tail_p0x": 16.85, "tail_p1x": 12.91, "neck_p0x": -5.5, "neck_p1x": -5.85, "arm_L_p0x": -9.19}
# walking v8 mesh body (heavier neck, rear body with its real volume): calibrate_springref("dc", active=True,
# arms=True, hulls=True) residuals added to DC_SPRINGREF on 30 Sep 2026
DC_LEAN_SPRINGREF = {"spine_px": -2.72, "tail_p0x": 14.33, "tail_p1x": 10.61, "neck_p0x": -5.72, "neck_p1x": -6.5, "arm_L_p0x": -9.19}      # recalibrated below
DC_HULL_SPRINGREF = {"spine_px": -1.42, "tail_p0x": 12.68, "tail_p1x": 10.72, "neck_p0x": -4.08, "neck_p1x": -6.32,
                     "arm_L_p0x": -9.19}

BODIES = {"myrranox": myrranox, "dc": dc}


def _fmt(v):
    return " ".join(f"{x:.4f}" for x in v)


def _passive_xml(ch, offset=(0.0, 0.0, 0.0), spec=None):
    """Nested bodies with spring-damper joints (motors, if any, are added as actuators). Damping from the damping
    ratio and a rough inertia. offset: position of the parent body's origin in the body frame."""
    xml, closes = "", ""
    pts = [np.asarray(p, dtype=float) for p in ch.points]
    masses = list(ch.masses)
    if spec is not None and spec.hulls:                     # walking v8: the real part masses (by hull volume)
        masses = [hull_mass(spec, bone) for bone in ch.bones]
    for i, seg in enumerate(ch.joints):
        origin = pts[i] - (pts[i - 1] if i else np.asarray(offset, dtype=float))
        vec = pts[i + 1] - pts[i]
        length = float(np.linalg.norm(vec))
        mass_out = sum(masses[i:])                          # this joint carries all segments beyond it
        inertia = mass_out * (0.5 * length) ** 2 + masses[i] * length ** 2 / 12
        scale = mass_out / sum(ch.masses[i:])               # mesh bodies: springs scaled to the real mass
        joints = ""
        for k, (axis, stiff, ref) in enumerate(seg):
            stiff = stiff * scale
            damp = 2 * ch.damping_ratio * np.sqrt(stiff * inertia)
            joints += (f'<joint name="{ch.name}_p{i}{"xyz"[k]}" axis="{_fmt(axis)}" stiffness="{stiff}" '
                       f'springref="{ref}" damping="{damp:.4f}" armature="0.005" limited="true" '
                       f'range="{-ch.range_deg} {ch.range_deg}"/>')
        if spec is not None and spec.hulls:
            geom = _hull_geom(spec, ch.bones[i], origin)
        else:
            geom = (f'<geom type="capsule" fromto="0 0 0 {_fmt(vec)}" size="{ch.radius[i]}" mass="{ch.masses[i]}" '
                    f'class="creature"/>')
        xml += f'<body name="{ch.name}_s{i}" pos="{_fmt(origin)}">{joints}{geom}'
        closes += "</body>"
    return xml + closes


def calibrate_springref(species, active=False, **body_kwargs):
    """Settle the body with the leg rest pose held and read how far each spring joint sags; springref = -sag.
    Motors of spine/neck/head/abdomen are switched off, so the springs alone carry the weight."""
    import mujoco
    spec = BODIES[species](passive=True, active=active, **body_kwargs)
    m = mujoco.MjModel.from_xml_string(build_mjcf(spec))
    n_leg = sum(len(leg.joints) for leg in spec.legs)
    m.actuator_gainprm[n_leg:, 0] = 0.0
    m.actuator_biasprm[n_leg:, :] = 0.0
    d = mujoco.MjData(m)
    for _ in range(1500):
        mujoco.mj_step(m, d)
    out = {}
    if spec.rear is not None:
        name = f"{spec.rear.name}_px"
        q = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]]
        out[name] = round(-float(np.degrees(q)), 2)
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
    if spec.hulls:
        foot_bottom = min(spec.hulls[f"{leg.name}_s2"]["min"][2] for leg in spec.legs if leg.mass_scale == 1.0)
    else:
        foot_bottom = min(leg.points[3, 2] - leg.radius * 1.2 for leg in spec.legs if leg.mass_scale == 1.0)
    return spec.stand_height - foot_bottom + 0.005


def _leg_xml(leg, leg_mass, damp, offset=(0.0, 0.0, 0.0), spec=None):
    """Nested leg bodies; each segment's body sits at its joint. offset: parent body origin in the body frame."""
    p = leg.points
    xml, closes = "", ""
    for i in range(3):
        origin = p[i] - (p[i - 1] if i else np.asarray(offset, dtype=float))
        seg_from, seg_to = np.zeros(3), p[i + 1] - p[i]
        joints = "".join(f'<joint name="{leg.name}_j{k}" axis="{_fmt(axis)}" range="{lo} {hi}" {damp}/>'
                         for k, (seg, axis, (lo, hi)) in enumerate(leg.joints) if seg == i)
        rad = leg.seg_radius[i] if leg.seg_radius else leg.radius
        if spec is not None and spec.hulls and i == 2 and spec.lean:
            a, b, _ = _paw_pad(spec, leg)
            geom = (f'<geom name="{leg.name}_foot" type="capsule" fromto="{_fmt(a)} {_fmt(b)}" size="{PAW_PAD_RADIUS}" '
                    f'mass="{hull_mass(spec, f"{leg.name}_s2"):.4f}" friction="1.2 0.02 0.001" class="creature"/>')
        elif spec is not None and spec.hulls:
            geom = _hull_geom(spec, f"{leg.name}_s{i}", origin, name=f"{leg.name}_foot" if i == 2 else None)
        else:
            geom = (f'<geom type="capsule" fromto="{_fmt(seg_from)} {_fmt(seg_to)}" size="{rad}" '
                    f'mass="{leg_mass * leg.mass_scale:.4f}" class="creature"/>')
        xml += f'<body name="{leg.name}_s{i}" pos="{_fmt(origin)}">{joints}{geom}'
        closes += "</body>"
    if spec is not None and spec.hulls:           # the paw hull is the foot; the site marks its lowest point
        xml += f'<site name="{leg.name}_foot" pos="{_fmt(_foot_site(spec, leg))}" size="0.01"/>'
    else:
        foot = p[3] - p[2]
        xml += (f'<geom name="{leg.name}_foot" type="sphere" pos="{_fmt(foot)}" size="{leg.radius * 1.2:.4f}" '
                f'mass="{leg_mass * leg.mass_scale * 0.2:.4f}" friction="1.2 0.02 0.001" class="creature"/>'
                f'<site name="{leg.name}_foot" pos="{_fmt(foot)}" size="0.01"/>')
    return xml + closes


def _rear_xml(spec, leg_mass, damp):
    """Stage 2 spine: rear body on the spine joint with its geometry, the hind legs and the chains hanging on it."""
    r = spec.rear
    pt = np.asarray(r.point, dtype=float)
    joints = "".join(f'<joint name="{r.name}_{sfx}" axis="{_fmt(axis)}" stiffness="{stiff}" springref="{ref}" '
                     f'damping="{r.damping}" armature="0.02" limited="true" range="{-rng} {rng}"/>'
                     for sfx, axis, stiff, ref, rng in r.joints)
    geoms = "".join(f'<geom type="capsule" fromto="{_fmt(a)} {_fmt(b)}" size="{rad}" mass="{mass}" class="creature"/>'
                    for a, b, rad, mass in r.geoms)
    legs = "".join(_leg_xml(leg, leg_mass, damp, pt, spec) for leg in spec.legs if leg.name in r.legs)
    chains = "".join(_passive_xml(ch, pt, spec) for ch in spec.passive if ch.parent == r.name)
    if spec.hulls:
        geoms = _hull_geom(spec, r.bone, pt)
    return f'<body name="{r.name}_rear" pos="{_fmt(pt)}">{joints}{geoms}{legs}{chains}</body>'


def creature_xml(spec: BodySpec, prefix="", pos=(0.0, 0.0), yaw=0.0, leg_mass_fraction=0.25):
    """(body XML, actuator XML) for one creature. All names get `prefix`, so several creatures fit in one world.
    The creature's own defaults (joint damping, collision classes) are set on its elements."""
    leg_mass = spec.mass * leg_mass_fraction / sum(leg.mass_scale for leg in spec.legs) / 3
    damp = f'damping="{spec.joint_damping}" armature="0.01" limited="true"'
    rear_legs = set(spec.rear.legs) if spec.rear is not None else set()
    legs_xml = [_leg_xml(leg, leg_mass, damp, spec=spec) for leg in spec.legs if leg.name not in rear_legs]

    torso = [g.replace("<geom ", '<geom class="creature" ') for g in spec.torso_geoms]
    if spec.hulls:
        torso.append(_hull_geom(spec, "torso", np.zeros(3)))
    torso += [_passive_xml(ch, spec=spec) for ch in spec.passive if ch.parent == "torso"]
    if spec.rear is not None:
        torso.append(_rear_xml(spec, leg_mass, damp))
    quat = f"{np.cos(yaw / 2):.6f} 0 0 {np.sin(yaw / 2):.6f}"
    body = (f'<body name="torso" pos="{pos[0]:.4f} {pos[1]:.4f} {start_height(spec):.4f}" quat="{quat}">'
            f'<freejoint name="root"/><site name="torso_site" pos="0 0 0" size="0.01"/>'
            f'{"".join(torso)}{"".join(legs_xml)}</body>')
    actuators = "".join(
        f'<position name="{prefix}{j}" joint="{prefix}{j}" kp="{kp}" forcerange="{-tq} {tq}"/>'
        for j, (kp, tq, _) in zip(spec.joint_names, spec.actuator_params()))
    return _prefix_names(body, prefix).replace('mesh="@MESH@', f'mesh="{prefix}'), actuators


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
    bodies, acts, meshes = [], [], []
    for spec, prefix, pos, yaw in creatures:
        b, a = creature_xml(spec, prefix, pos, yaw)
        bodies.append(b)
        acts.append(a)
        meshes.append(creature_assets(spec, prefix))
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
    {"".join(meshes)}
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
