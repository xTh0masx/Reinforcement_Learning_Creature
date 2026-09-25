"""Gymnasium environment: one predator creature in the 2D habitat (PROJECT_PLAN.md sections 6-8).

Stage 1 of the training order: a single creature with prey, rest spots and its home zone.
The "nearest other creature" observation slots are already part of the 27-value layout and
report "none seen" until rivals are added in later stages.
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.habitat import Habitat
from sim.species import SPECIES

DT = 0.1                    # s per step (10 Hz)
EPISODE_STEPS = 6000        # one simulated day of 10 min
STARVE_LIMIT_STEPS = 600    # episode ends after 60 s at hunger 100 %

N_RAYS = 8
RAY_RANGE = 10.0            # m
PREY_SENSE_RANGE = 30.0     # m, prey further away are not seen
DIST_SCALE = 50.0           # m, normalisation for rest-spot / home-zone / other-creature distance
MOVING_SPEED = 0.05         # m/s, below this the creature counts as standing still

# Rewards (section 8)
R_CATCH = 1.0               # x hunger, only when hunger > 0.5
R_REST = 0.02
R_OUTSIDE_HOME = -0.01      # x distance factor
R_RIVAL_ZONE = -0.03
R_STARVING = -0.02
R_HIT = -0.5
R_STARVED = -5.0            # terminal; without it ending the episode early would dodge penalties

OBS_SIZE = 27


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class CreatureEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, species="myrranox", render_mode=None, habitat_path=None):
        if species not in SPECIES:
            raise ValueError(f"Unknown species {species!r}, choose from {list(SPECIES)}")
        self.species = SPECIES[species]
        self.habitat = Habitat.load(habitat_path)
        self.home = self.habitat.home_zones[self.species.home_zone]
        self.rival_zones = [z for k, z in self.habitat.home_zones.items() if k != self.species.home_zone]
        self.render_mode = render_mode

        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._ray_offsets = np.arange(N_RAYS) * (2 * math.pi / N_RAYS)
        self._viewer = None

    # ------------------------------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        sp = self.species

        # Per-individual need rates, +-20 %
        self.hunger_mult, self.fatigue_mult, self.alert_mult = rng.uniform(0.8, 1.2, size=3)

        self.pos = self.habitat.random_free_point(rng, margin=sp.body_radius + 0.5, region=self.home)
        self.heading = rng.uniform(-math.pi, math.pi)
        self.speed = 0.0
        self.turn_rate = 0.0
        self.hunger = rng.uniform(0.2, 0.5)
        self.fatigue = rng.uniform(0.0, 0.3)
        self.alertness = 0.0

        self._reset_prey(rng)

        self.t = 0
        self.starve_steps = 0
        self.activity = "idle"
        self.trail = [self.pos.copy()]
        self.stats = {"reward": 0.0, "prey_caught": 0, "hits": 0, "steps_in_home": 0,
                      "steps_starving": 0, "rest_steps": 0}
        return self._observe(), self._info({})

    def command(self, action):
        """Brain action -> (target speed m/s, target turn rate rad/s) within the species limits."""
        sp = self.species
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        max_fwd = sp.max_speed * (sp.tired_speed_cap if self.fatigue >= 1.0 else 1.0)
        target_v = a[0] * max_fwd if a[0] >= 0 else a[0] * sp.max_backward_speed
        turn_limit = sp.max_turn_rate
        if not sp.turn_in_place:
            turn_limit *= min(1.0, abs(self.speed) / sp.min_turn_speed)
        return float(target_v), float(a[1] * turn_limit), turn_limit

    def step(self, action, body=None):
        """Advance 0.1 s. Without `body`, the kinematic movement model (section 7) moves the creature.
        With `body` = dict(pos, heading, speed, turn_rate) from the physics body (record_physics.py),
        that pose is used instead; info["hit"] tells the caller to push the body back."""
        sp = self.species
        hunger_before, fatigue_before = self.hunger, self.fatigue
        rewards = {}

        if body is None:
            target_v, target_w, turn_limit = self.command(action)
            self.speed += np.clip(target_v - self.speed, -sp.max_accel * DT, sp.max_accel * DT)
            self.turn_rate += np.clip(target_w - self.turn_rate, -sp.max_turn_accel * DT, sp.max_turn_accel * DT)
            self.turn_rate = float(np.clip(self.turn_rate, -turn_limit, turn_limit))
            heading = _wrap(self.heading + self.turn_rate * DT)
            new_pos = self.pos + self.speed * DT * np.array([math.cos(heading), math.sin(heading)])
        else:
            heading, new_pos = float(body["heading"]), np.asarray(body["pos"], dtype=float)[:2]
            self.speed, self.turn_rate = float(body["speed"]), float(body["turn_rate"])
        self.heading = heading
        hit = not self.habitat.is_free(new_pos, sp.body_radius)
        if hit:
            self.speed = 0.0
            rewards["hit"] = R_HIT
            self.stats["hits"] += 1
        else:
            self.pos = new_pos
        self.speed = float(self.speed)

        # --- prey ---
        self._step_prey()
        caught = self._try_catch()
        if caught:
            self.hunger = max(0.0, self.hunger - sp.hunger_per_prey)
            self.stats["prey_caught"] += 1
            if hunger_before > 0.5:
                rewards["catch"] = R_CATCH * hunger_before

        # --- needs ---
        speed_frac = abs(self.speed) / sp.max_speed
        moving = abs(self.speed) > MOVING_SPEED
        on_rest = self.habitat.on_rest_spot(self.pos)

        self.hunger += (sp.hunger_rate + sp.hunger_move_rate * speed_frac ** 2) * self.hunger_mult * DT
        if moving:
            self.fatigue += (sp.fatigue_walk_rate * speed_frac + sp.fatigue_run_rate * speed_frac ** 2) \
                            * self.fatigue_mult * DT
        else:
            self.fatigue -= (sp.fatigue_rest_recovery if on_rest else sp.fatigue_recovery) * DT
        # No rival yet: alertness only decays.
        self.alertness -= sp.alert_decay * self.alert_mult * DT
        self.hunger, self.fatigue, self.alertness = (float(np.clip(v, 0.0, 1.0))
                                                     for v in (self.hunger, self.fatigue, self.alertness))

        # --- rewards (section 8) ---
        if on_rest and not moving and fatigue_before > 0.5:
            rewards["rest"] = R_REST
            self.stats["rest_steps"] += 1

        d_out = self.home.distance(self.pos) - self.home.r
        if d_out > 0:
            rewards["outside_home"] = R_OUTSIDE_HOME * min(1.0 + d_out / 10.0, 3.0)
        else:
            self.stats["steps_in_home"] += 1
        if any(z.contains(self.pos) for z in self.rival_zones):
            rewards["rival_zone"] = R_RIVAL_ZONE

        if self.hunger >= 1.0:
            rewards["starving"] = R_STARVING
            self.starve_steps += 1
            self.stats["steps_starving"] += 1
        else:
            self.starve_steps = 0

        # --- episode end ---
        self.t += 1
        terminated = self.starve_steps >= STARVE_LIMIT_STEPS
        if terminated:
            rewards["starved"] = R_STARVED
        truncated = self.t >= EPISODE_STEPS

        if caught:
            self.activity = "eat"
        elif not moving:
            self.activity = "rest" if on_rest else "idle"
        else:
            self.activity = "walk"

        reward = float(sum(rewards.values()))
        self.stats["reward"] += reward
        self.trail.append(self.pos.copy())

        if self.render_mode == "human":
            self.render()
        return self._observe(), reward, terminated, truncated, self._info(rewards)

    def render(self):
        if self.render_mode is None:
            return None
        if self._viewer is None:
            from sim.viewer import HabitatViewer
            self._viewer = HabitatViewer(self.habitat, interactive=self.render_mode == "human")
        self._viewer.update(self)
        if self.render_mode == "rgb_array":
            return self._viewer.rgb_array()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------------------------------
    # Prey
    # ------------------------------------------------------------------------------------------
    def _reset_prey(self, rng):
        cfg = self.habitat.prey
        n = int(cfg["count"])
        self.prey_pos = np.array([self.habitat.random_free_point(rng, margin=cfg["radius"]) for _ in range(n)])
        self.prey_heading = rng.uniform(-math.pi, math.pi, size=n)
        self.prey_alive = np.ones(n, dtype=bool)
        self.prey_respawn = np.zeros(n)

    def _step_prey(self):
        cfg, rng = self.habitat.prey, self.np_random
        alive = self.prey_alive

        # Respawn caught prey after their timer, away from the predator.
        self.prey_respawn[~alive] -= DT
        for i in np.flatnonzero(~alive & (self.prey_respawn <= 0)):
            self.prey_pos[i] = self.habitat.random_free_point(
                rng, margin=cfg["radius"], avoid=self.pos, avoid_dist=cfg["respawn_min_distance"])
            self.prey_heading[i] = rng.uniform(-math.pi, math.pi)
            self.prey_alive[i] = True
        alive = self.prey_alive

        away = self.prey_pos - self.pos
        dist = np.linalg.norm(away, axis=1)
        flee = alive & (dist < cfg["flee_radius"])
        wander = alive & ~flee

        self.prey_heading[wander] += rng.normal(0.0, 1.0, size=wander.sum()) * math.sqrt(DT)
        self.prey_heading[flee] = np.arctan2(away[flee, 1], away[flee, 0]) + rng.normal(0.0, 0.3, size=flee.sum())
        speed = np.where(flee, cfg["flee_speed"], cfg["wander_speed"])

        step = (speed * DT)[:, None] * np.stack([np.cos(self.prey_heading), np.sin(self.prey_heading)], axis=1)
        cand = self.prey_pos + step
        free = self.habitat.is_free_many(cand, cfg["radius"])
        move = alive & free
        self.prey_pos[move] = cand[move]
        # Blocked prey turn around.
        blocked = alive & ~free
        self.prey_heading[blocked] += math.pi + rng.normal(0.0, 0.5, size=blocked.sum())
        self.prey_heading = (self.prey_heading + math.pi) % (2 * math.pi) - math.pi

    def _try_catch(self):
        if not self.prey_alive.any():
            return False
        dist = np.linalg.norm(self.prey_pos - self.pos, axis=1)
        dist[~self.prey_alive] = np.inf
        i = int(np.argmin(dist))
        if dist[i] > self.species.catch_radius:
            return False
        lo, hi = self.habitat.prey["respawn_seconds"]
        self.prey_alive[i] = False
        self.prey_respawn[i] = self.np_random.uniform(lo, hi)
        return True

    # ------------------------------------------------------------------------------------------
    # Observation (section 8, 27 values, egocentric: x = forward, y = left)
    # ------------------------------------------------------------------------------------------
    def _ego(self, target, scale):
        """Unit direction to `target` in the creature's frame and distance / scale."""
        v = np.asarray(target, dtype=float) - self.pos
        d = float(np.hypot(*v))
        if d < 1e-6:
            return [0.0, 0.0, 0.0]
        c, s = math.cos(-self.heading), math.sin(-self.heading)
        return [(c * v[0] - s * v[1]) / d, (s * v[0] + c * v[1]) / d, min(d / scale, 1.0)]

    def _observe(self):
        sp = self.species
        obs = [self.hunger, self.fatigue, self.alertness,
               self.speed / sp.max_speed, self.turn_rate / sp.max_turn_rate]

        # Nearest visible prey
        prey = [0.0, 0.0, 1.0]
        if self.prey_alive.any():
            dist = np.linalg.norm(self.prey_pos - self.pos, axis=1)
            dist[~self.prey_alive] = np.inf
            i = int(np.argmin(dist))
            if dist[i] <= PREY_SENSE_RANGE:
                prey = self._ego(self.prey_pos[i], PREY_SENSE_RANGE)
        obs += prey

        # Nearest rest spot
        i, _ = self.habitat.nearest_rest_spot(self.pos)
        obs += self._ego(self.habitat.rest_spots[i].center, DIST_SCALE)

        # Home zone
        obs += self._ego(self.home.center, DIST_SCALE)
        obs.append(1.0 if self.home.contains(self.pos) else 0.0)

        # Nearest other creature: none in stage 1 (direction 0, distance 1, rival no)
        obs += [0.0, 0.0, 1.0, 0.0]

        # Obstacle rays over 360 deg, starting straight ahead, counter-clockwise
        rays = self.habitat.ray_distances(self.pos, self.heading + self._ray_offsets, RAY_RANGE)
        obs += list(rays / RAY_RANGE)

        return np.clip(np.asarray(obs, dtype=np.float32), -1.0, 1.0)

    def _info(self, rewards):
        return {
            "time": self.t * DT,
            "species": self.species.name,
            "x": float(self.pos[0]), "y": float(self.pos[1]),
            "heading": float(self.heading), "speed": self.speed, "turn_rate": self.turn_rate,
            "activity": self.activity,
            "hunger": self.hunger, "fatigue": self.fatigue, "alertness": self.alertness,
            "reward_terms": dict(rewards),
            "hit": "hit" in rewards,
        }
