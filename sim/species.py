"""Per-species movement limits and need rates (PROJECT_PLAN.md sections 7 and 8).

All speeds in m/s, angles in rad, rates per second. Need values live in [0, 1].
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Species:
    name: str
    home_zone: str                # key in habitat.json "home_zones"
    body_radius: float            # m, used for collisions with rocks/boundary

    # --- Movement model (section 7) ---
    walk_speed: float             # reference speed of the Blender Walk cycle
    max_speed: float
    max_backward_speed: float     # 0 = cannot walk backwards
    max_turn_rate: float          # rad/s
    turn_in_place: bool           # False: turn limit scales with speed up to min_turn_speed
    min_turn_speed: float         # speed at which the full turn rate is available
    max_accel: float              # m/s^2
    max_turn_accel: float         # rad/s^2, keeps paths smooth enough to animate
    catch_radius: float           # prey closer than this (centre to centre) is caught

    # --- Needs (section 8), base rates before the +-20 % per-individual randomisation ---
    hunger_rate: float            # per s, always
    hunger_move_rate: float       # per s, extra at max speed (scales with speed_frac^2)
    hunger_per_prey: float        # hunger removed per caught prey
    fatigue_walk_rate: float      # per s, scales with speed_frac
    fatigue_run_rate: float       # per s, scales with speed_frac^2
    fatigue_recovery: float       # per s while standing still
    fatigue_rest_recovery: float  # per s while standing still on a rest spot
    tired_speed_cap: float        # fraction of max_speed allowed at fatigue 100 %
    alert_rate: float             # per s while a rival is within alert_radius
    alert_home_factor: float      # multiplier when the rival is inside own home zone
    alert_decay: float            # per s otherwise
    alert_radius: float           # m


MYRRANOX = Species(
    name="myrranox",
    home_zone="myrranox",
    body_radius=0.75,             # 1.5 m body length
    walk_speed=0.6,
    max_speed=1.2,
    max_backward_speed=0.3,
    max_turn_rate=math.radians(90),
    turn_in_place=True,
    min_turn_speed=0.0,
    max_accel=1.0,
    max_turn_accel=math.radians(180),
    catch_radius=1.0,
    hunger_rate=1 / 360,          # idle: empty -> starving in 6 min
    hunger_move_rate=1 / 180,
    hunger_per_prey=0.5,
    fatigue_walk_rate=0.004,
    fatigue_run_rate=0.012,       # ~60 s at max speed from rested to exhausted
    fatigue_recovery=0.01,
    fatigue_rest_recovery=0.03,
    tired_speed_cap=0.3,
    alert_rate=0.2,
    alert_home_factor=1.5,
    alert_decay=0.05,
    alert_radius=10.0,
)

DC_SKIN_RAW = Species(
    name="dc",
    home_zone="dc",
    body_radius=0.6,              # 1.0 m shoulder height
    walk_speed=0.8,
    max_speed=2.5,                # Trot
    max_backward_speed=0.0,
    max_turn_rate=math.radians(60),
    turn_in_place=False,
    min_turn_speed=0.4,
    max_accel=1.5,
    max_turn_accel=math.radians(120),
    catch_radius=1.2,
    hunger_rate=1 / 360,
    hunger_move_rate=1 / 150,
    hunger_per_prey=0.5,
    fatigue_walk_rate=0.003,
    fatigue_run_rate=0.015,
    fatigue_recovery=0.01,
    fatigue_rest_recovery=0.03,
    tired_speed_cap=0.3,
    alert_rate=0.2,
    alert_home_factor=1.5,
    alert_decay=0.05,
    alert_radius=10.0,
)

SPECIES = {s.name: s for s in (MYRRANOX, DC_SKIN_RAW)}
