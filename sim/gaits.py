"""Target gaits for the v2 locomotion brains: per-leg phase offsets, duty factor and step frequency.

A gait clock phi in [0, 1) advances with the step frequency. Leg i is meant to be on the ground
(stance) while (phi + offset_i) mod 1 < duty, and in the air (swing) otherwise. While the command is
"stand still" the clock pauses and every foot should stay down.

Myrranox: insect tripod - {front_L, mid_R, back_L} and {front_R, mid_L, back_R} alternate.
DC:       lateral-sequence walk (hind_L, front_L, hind_R, front_R, a quarter cycle apart) at low speed,
          blending into a trot (diagonal pairs) between 1.0 and 1.6 m/s.
Mirroring left <-> right maps every gait onto itself shifted by half a cycle, so the gaits are
compatible with the mirror-symmetry loss (sim/symmetry.py).
"""
import numpy as np

TRIPOD = {"front_L": 0.0, "mid_R": 0.0, "back_L": 0.0, "front_R": 0.5, "mid_L": 0.5, "back_R": 0.5}
DC_WALK = {"hind_L": 0.0, "front_L": 0.25, "hind_R": 0.5, "front_R": 0.75}
DC_TROT = {"hind_L": 0.0, "front_L": 0.5, "hind_R": 0.5, "front_R": 1.0}   # front_R 1.0 == 0.0, kept for blending
# DC walking v6 (six limbs): the arms stay folded while walking and join the run from ARMS_FROM m/s, landing a quarter
# cycle after the trot pairs (mirror-compatible: arm_L 0.25 <-> arm_R 0.75), so something touches down every quarter.
DC_ARMS = {"arm_L": 0.25, "arm_R": 0.75}
ARMS_FROM, ARMS_FULL, ARM_DUTY = 1.4, 1.8, 0.35
# DC walking v12 (owner, 14 Oct: "front arms only for sprinting, walking like a T-rex"): the arms join from 2.0 m/s
ARMS_SPRINT = (2.0, 2.6)

GAITS = {
    # species: (min Hz, max Hz, reference speed for max Hz)
    "myrranox": (1.5, 3.0, 1.2),
    "dc": (1.2, 2.8, 2.5),
}
# Walking v5 (natural style): the DC walks like a panther - slow, long strides (about 1.1 steps/s at 0.8 m/s instead
# of 1.7); frequency grows with speed^1.5, so the trot still reaches the old top frequency.
GAITS_STYLE = {"myrranox": (1.5, 3.0, 1.2, 1.0), "dc": (0.8, 2.6, 2.5, 1.5)}   # min Hz, max Hz, v ref, exponent
# Walking v8 (DC sprint up to 4 m/s): same slow panther walk (~1 step/s at 0.8 m/s), ~1.9 at 2.5 m/s, 3.0 at 4 m/s
GAITS_SPRINT = {"myrranox": GAITS_STYLE["myrranox"], "dc": (0.8, 3.0, 4.0, 1.5)}
FOOT_CLEARANCE = {"myrranox": 0.03, "dc": 0.08}     # m, target swing height of a foot
STAND_SPEED = 0.05                                  # |v| below this and |w| small -> standing


def frequency(species, speed, turn_rate, max_turn_rate, style=False, sprint=False):
    """Step frequency (Hz): grows with speed; turning on the spot also needs steps. style: walking v5 curve,
    sprint: walking v8 curve (DC up to 4 m/s)."""
    lo, hi, v_ref, expo = (GAITS_SPRINT[species] if sprint else GAITS_STYLE[species] if style
                           else (*GAITS[species], 1.0))
    effort = max(abs(speed) / v_ref, 0.5 * abs(turn_rate) / max_turn_rate)
    return lo + (hi - lo) * min(1.0, effort) ** expo


def pattern(species, leg_names, speed, arms=(ARMS_FROM, ARMS_FULL)):
    """(offsets per leg in leg_names order, duty factor) for the given speed. arms: speeds (m/s) at which the DC's arms
    start to join the run / are fully part of it."""
    if species == "myrranox":
        return np.array([TRIPOD[n] for n in leg_names]), 0.55
    b = float(np.clip((abs(speed) - 1.0) / 0.6, 0.0, 1.0))    # 0 = walk, 1 = trot
    legs = [n for n in leg_names if n in DC_WALK]
    walk = np.array([DC_WALK[n] for n in legs])
    trot = np.array([DC_TROT[n] for n in legs])
    duty = (1 - b) * 0.65 + b * 0.45
    if len(legs) == len(leg_names):
        return (1 - b) * walk + b * trot, duty
    # six limbs: per-limb duty; arms 0 (never on the ground) while walking, ARM_DUTY in the run
    a = float(np.clip((abs(speed) - arms[0]) / (arms[1] - arms[0]), 0.0, 1.0))
    offsets = np.array([(1 - b) * DC_WALK[n] + b * DC_TROT[n] if n in DC_WALK else DC_ARMS[n] for n in leg_names])
    duties = np.array([duty if n in DC_WALK else a * ARM_DUTY for n in leg_names])
    return offsets, duties


def is_standing(speed, turn_rate):
    return abs(speed) < STAND_SPEED and abs(turn_rate) < 0.05
