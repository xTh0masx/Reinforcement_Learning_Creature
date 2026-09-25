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

GAITS = {
    # species: (min Hz, max Hz, reference speed for max Hz)
    "myrranox": (1.5, 3.0, 1.2),
    "dc": (1.2, 2.8, 2.5),
}
FOOT_CLEARANCE = {"myrranox": 0.03, "dc": 0.08}     # m, target swing height of a foot
STAND_SPEED = 0.05                                  # |v| below this and |w| small -> standing


def frequency(species, speed, turn_rate, max_turn_rate):
    """Step frequency (Hz): grows with speed; turning on the spot also needs steps."""
    lo, hi, v_ref = GAITS[species]
    effort = max(abs(speed) / v_ref, 0.5 * abs(turn_rate) / max_turn_rate)
    return lo + (hi - lo) * min(1.0, effort)


def pattern(species, leg_names, speed):
    """(offsets per leg in leg_names order, duty factor) for the given speed."""
    if species == "myrranox":
        return np.array([TRIPOD[n] for n in leg_names]), 0.55
    b = float(np.clip((abs(speed) - 1.0) / 0.6, 0.0, 1.0))    # 0 = walk, 1 = trot
    walk = np.array([DC_WALK[n] for n in leg_names])
    trot = np.array([DC_TROT[n] for n in leg_names])
    return (1 - b) * walk + b * trot, (1 - b) * 0.65 + b * 0.45


def is_standing(speed, turn_rate):
    return abs(speed) < STAND_SPEED and abs(turn_rate) < 0.05
