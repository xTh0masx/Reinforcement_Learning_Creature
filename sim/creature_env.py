"""Gymnasium environment: predators in the 2D habitat (PROJECT_PLAN.md sections 6-8 and 15).

The brain controls one creature (the "ego"). Optional other creatures share the habitat and the prey;
each is driven by a scripted policy or a frozen trained navigation brain. Section 15 rules:
- every prey belongs to the area it spawned in: territory A, territory B or neutral (overlap = neutral)
  and stays there;
- prey eaten in the eater's own territory is gone for good;
- prey eaten in a rival territory re-appears at once in the eater's own territory (raids bring food home);
- prey eaten in the neutral area respawns there after 60-120 s.
Hunger and fatigue rise with mechanical effort: the power of the physics body when one is simulated
(step(..., bodies=...)), otherwise a power curve measured from the locomotion brain (sim/energy.json).

Observation: 27 values, egocentric (layout unchanged since stage 1). Action: forward speed, turn rate.
"""
import json
import math
import os

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
OTHER_SENSE_RANGE = 30.0    # m, other creatures further away are not seen
DIST_SCALE = 50.0           # m, normalisation for rest-spot / home-zone / other-creature distance
MOVING_SPEED = 0.05         # m/s, below this the creature counts as standing still

# Rewards (section 8, rebalanced in section 15)
R_CATCH = 1.0               # x hunger, only when hunger > 0.5
R_REST = 0.02
R_OUTSIDE_HOME = -0.01      # x distance factor; not applied inside a rival territory (R_RIVAL_ZONE instead)
R_RIVAL_ZONE = -0.01        # per step inside a rival territory (was -0.03; raids must stay possible)
R_STARVING = -0.02
R_HIT = -0.5
R_STARVED = -5.0            # terminal; without it ending the episode early would dodge penalties
R_CONFRONT = 0.02           # per step moving towards a rival that is inside the own territory, within 8 m
R_INTRUDER_LEFT = 0.5       # a confronted rival leaves the own territory
R_FIGHT_RISK = -0.1         # per step closer than 2 m to a rival
CONFRONT_RANGE = 8.0
FIGHT_RANGE = 2.0

OBS_SIZE = 27
NEUTRAL = "neutral"
ENERGY_FILE = os.path.join(os.path.dirname(__file__), "energy.json")


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _load_energy():
    try:
        with open(ENERGY_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


class Creature:
    """State of one predator in the habitat."""

    def __init__(self, species, cid, habitat):
        self.species = SPECIES[species]
        self.id = cid
        self.zone = self.species.home_zone
        self.home = habitat.home_zones[self.zone]
        self.policy = None

    def reset(self, rng, habitat):
        sp = self.species
        self.hunger_mult, self.fatigue_mult, self.alert_mult = rng.uniform(0.8, 1.2, size=3)
        self.pos = habitat.random_free_point(rng, margin=sp.body_radius + 0.5, region=self.home)
        self.heading = rng.uniform(-math.pi, math.pi)
        self.speed = 0.0
        self.turn_rate = 0.0
        self.hunger = rng.uniform(0.2, 0.5)
        self.fatigue = rng.uniform(0.0, 0.3)
        self.alertness = 0.0
        self.starve_steps = 0
        self._contact = False
        self.activity = "idle"
        self.trail = [self.pos.copy()]
        self.confronted = set()          # ids of rivals this creature confronted inside its territory
        self.stats = {"reward": 0.0, "prey_caught": 0, "hits": 0, "steps_in_home": 0, "steps_starving": 0,
                      "rest_steps": 0, "home_prey_eaten": 0, "raid_prey_eaten": 0, "neutral_prey_eaten": 0,
                      "steps_in_rival_zone": 0, "intruders_repelled": 0}
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()


class CreatureEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, species="myrranox", render_mode=None, habitat_path=None, others=None):
        """others: list of {"species": "dc", "brain": None | "models/dc_brain_v1"} sharing the habitat;
        brain None = scripted policy (sim/scripted.py)."""
        if species not in SPECIES:
            raise ValueError(f"Unknown species {species!r}, choose from {list(SPECIES)}")
        self.habitat = Habitat.load(habitat_path)
        self.render_mode = render_mode
        self.energy = _load_energy()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._ray_offsets = np.arange(N_RAYS) * (2 * math.pi / N_RAYS)
        self._viewer = None

        counts = {}

        def new(sp):
            counts[sp] = counts.get(sp, 0) + 1
            return Creature(sp, f"{sp}_{counts[sp] - 1}", self.habitat)

        self.ego = new(species)
        self.others_cfg = list(others or [])
        self.others = [new(o["species"]) for o in self.others_cfg]
        self.creatures = [self.ego] + self.others

    # ------------------------------------------------------------------------------------------
    # Ego shortcuts (viewer, record.py, record_physics.py)
    # ------------------------------------------------------------------------------------------
    species = property(lambda self: self.ego.species)
    home = property(lambda self: self.ego.home)
    rival_zones = property(lambda self: [z for k, z in self.habitat.home_zones.items() if k != self.ego.zone])
    pos = property(lambda self: self.ego.pos)
    heading = property(lambda self: self.ego.heading)
    speed = property(lambda self: self.ego.speed)
    turn_rate = property(lambda self: self.ego.turn_rate)
    hunger = property(lambda self: self.ego.hunger)
    fatigue = property(lambda self: self.ego.fatigue)
    alertness = property(lambda self: self.ego.alertness)
    activity = property(lambda self: self.ego.activity)
    trail = property(lambda self: self.ego.trail)
    stats = property(lambda self: self.ego.stats)

    def _other_policy(self, i):
        c, cfg = self.others[i], self.others_cfg[i]
        if c.policy is None:
            if cfg.get("brain"):
                from sim.nav_brain import NavBrain
                c.policy = NavBrain(cfg["brain"])
            else:
                from sim.scripted import ScriptedPredator
                c.policy = ScriptedPredator()
        return c.policy

    # ------------------------------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        for i, c in enumerate(self.creatures):
            if c is not self.ego:
                self._other_policy(i - 1)
            c.reset(rng, self.habitat)
        self._reset_prey(rng)
        self.t = 0
        return self._observe(self.ego), self._info(self.ego, {})

    def command(self, action, c=None):
        """Brain action -> (target speed m/s, target turn rate rad/s, turn limit) within the species limits."""
        c = c or self.ego
        sp = c.species
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        max_fwd = sp.max_speed * (sp.tired_speed_cap if c.fatigue >= 1.0 else 1.0)
        target_v = a[0] * max_fwd if a[0] >= 0 else a[0] * sp.max_backward_speed
        turn_limit = sp.max_turn_rate
        if not sp.turn_in_place:
            turn_limit *= min(1.0, abs(c.speed) / sp.min_turn_speed)
        return float(target_v), float(a[1] * turn_limit), turn_limit

    def other_actions(self):
        """Actions of the non-ego creatures from their own observations."""
        acts = {}
        for i, c in enumerate(self.others):
            pol = self._other_policy(i)
            obs = self._observe(c)
            acts[c.id] = pol(obs, c, self) if not hasattr(pol, "model") else pol(obs)
        return acts

    def step(self, action, body=None, bodies=None, actions=None):
        """Advance 0.1 s.
        Kinematic mode (default): the section 7 movement model moves every creature.
        Physics mode: bodies = {creature id: dict(pos, heading, speed, turn_rate, energy)} from sim/physics_world.py
        (`body` = the ego's entry only, kept for record_physics.py). actions: precomputed other-creature actions."""
        if body is not None:
            bodies = {**(bodies or {}), self.ego.id: body}
        acts = dict(actions or self.other_actions())
        acts[self.ego.id] = action
        before = {c.id: (c.hunger, c.fatigue) for c in self.creatures}
        rewards = {c.id: {} for c in self.creatures}

        # --- movement ---
        for c in self.creatures:
            b = bodies.get(c.id) if bodies else None
            power = self._move(c, acts[c.id], b)
            c._power = power
            if c._hit:
                rewards[c.id]["hit"] = R_HIT
                c.stats["hits"] += 1

        # --- prey ---
        self._step_prey()
        caught = self._catches(rewards, before)

        # --- needs + rewards ---
        for c in self.creatures:
            self._needs(c, caught.get(c.id, False))
            self._rewards(c, rewards[c.id], before[c.id][1])

        self.t += 1
        ego, r = self.ego, rewards[self.ego.id]
        terminated = ego.starve_steps >= STARVE_LIMIT_STEPS
        if terminated:
            r["starved"] = R_STARVED
        truncated = self.t >= EPISODE_STEPS
        for c in self.creatures:
            c.trail.append(c.pos.copy())
            c.stats["reward"] += float(sum(rewards[c.id].values()))
        reward = float(sum(r.values()))
        if self.render_mode == "human":
            self.render()
        return self._observe(ego), reward, terminated, truncated, self._info(ego, r)

    # ------------------------------------------------------------------------------------------
    def _power_estimate(self, c):
        cal = self.energy.get(c.species.name)
        if not cal:
            return None
        p0, p1, p2, p3 = cal["coeffs"]
        return p0 + p1 * abs(c.speed) + p2 * c.speed ** 2 + p3 * abs(c.turn_rate)

    def _move(self, c, action, body):
        """Move one creature; returns mechanical power (W) or None if unknown."""
        sp = c.species
        if body is None:
            target_v, target_w, turn_limit = self.command(action, c)
            c.speed += np.clip(target_v - c.speed, -sp.max_accel * DT, sp.max_accel * DT)
            c.turn_rate += np.clip(target_w - c.turn_rate, -sp.max_turn_accel * DT, sp.max_turn_accel * DT)
            c.turn_rate = float(np.clip(c.turn_rate, -turn_limit, turn_limit))
            heading = _wrap(c.heading + c.turn_rate * DT)
            step = c.speed * DT * np.array([math.cos(heading), math.sin(heading)])
            c.heading = heading
            contact = not self.habitat.is_free(c.pos + step, sp.body_radius)
            if contact:
                # Slide along the obstacle (like the physics body does) instead of freezing: a creature that
                # can only turn while moving (DC) would otherwise be stuck at a rock forever.
                n = self._obstacle_normal(c.pos)
                slide = step - min(0.0, float(step @ n)) * n
                if self.habitat.is_free(c.pos + slide, sp.body_radius):
                    c.pos = c.pos + slide
                c.speed *= 0.5
            else:
                c.pos = c.pos + step
            c._hit = contact and not getattr(c, "_contact", False)     # penalise entering contact, once
            c._contact = contact
            c.speed = float(c.speed)
            return self._power_estimate(c)
        # Physics body: rocks and walls are solid there, so the pose is always accepted; touching one counts as a hit.
        c.heading, c.pos = float(body["heading"]), np.asarray(body["pos"], dtype=float)[:2].copy()
        c.speed, c.turn_rate = float(body["speed"]), float(body["turn_rate"])
        contact = not self.habitat.is_free(c.pos, sp.body_radius * 0.6)
        c._hit = contact and not getattr(c, "_contact", False)
        c._contact = contact
        return body["energy"] / DT if "energy" in body else self._power_estimate(c)

    def _obstacle_normal(self, p):
        """Unit vector pointing away from the nearest rock or wall at p."""
        h = self.habitat
        cands = [(p[0] - h.xmin, np.array([1.0, 0.0])), (h.xmax - p[0], np.array([-1.0, 0.0])),
                 (p[1] - h.ymin, np.array([0.0, 1.0])), (h.ymax - p[1], np.array([0.0, -1.0]))]
        for o in h.obstacles:
            v = p - o.center
            d = float(np.hypot(*v))
            cands.append((d - o.r, v / max(d, 1e-9)))
        return min(cands, key=lambda x: x[0])[1]

    def _needs(self, c, ate):
        sp = c.species
        moving = abs(c.speed) > MOVING_SPEED
        on_rest = self.habitat.on_rest_spot(c.pos)
        cal = self.energy.get(sp.name)
        if c._power is not None and cal:
            effort = min(c._power / cal["p_max"], 1.5)             # 1 = effort of running at max speed
        else:
            effort = abs(c.speed) / sp.max_speed                    # fallback: speed-based (stage 1 model)
        c.hunger += (sp.hunger_rate + sp.hunger_move_rate * effort ** 2) * c.hunger_mult * DT
        if moving:
            c.fatigue += (sp.fatigue_walk_rate * effort + sp.fatigue_run_rate * effort ** 2) * c.fatigue_mult * DT
        else:
            c.fatigue -= (sp.fatigue_rest_recovery if on_rest else sp.fatigue_recovery) * DT

        # Alertness: a rival within alert_radius raises it (more inside the own territory)
        near = [o for o in self.creatures if o is not c and o.species.name != sp.name
                and np.hypot(*(o.pos - c.pos)) < sp.alert_radius]
        if near:
            factor = sp.alert_home_factor if any(c.home.contains(o.pos) for o in near) else 1.0
            c.alertness += sp.alert_rate * factor * c.alert_mult * DT
        else:
            c.alertness -= sp.alert_decay * c.alert_mult * DT
        c.hunger, c.fatigue, c.alertness = (float(np.clip(v, 0.0, 1.0)) for v in (c.hunger, c.fatigue, c.alertness))

        if ate:
            c.activity = "eat"
        elif not moving:
            c.activity = "rest" if on_rest else "idle"
        elif any(o.id in c.confronted and c.home.contains(o.pos) for o in self.creatures):
            c.activity = "confront"
        else:
            c.activity = "walk"
        c._on_rest, c._moving = on_rest, moving

    def _rewards(self, c, r, fatigue_before):
        if c._on_rest and not c._moving and fatigue_before > 0.5:
            r["rest"] = R_REST
            c.stats["rest_steps"] += 1

        in_rival = [k for k, z in self.habitat.home_zones.items() if k != c.zone and z.contains(c.pos)]
        d_out = c.home.distance(c.pos) - c.home.r
        if in_rival:
            r["rival_zone"] = R_RIVAL_ZONE
            c.stats["steps_in_rival_zone"] += 1
        elif d_out > 0:
            r["outside_home"] = R_OUTSIDE_HOME * min(1.0 + d_out / 10.0, 3.0)
        if d_out <= 0:
            c.stats["steps_in_home"] += 1

        # Territory defence (section 8)
        for o in self.creatures:
            if o is c or o.species.name == c.species.name:
                continue
            v = o.pos - c.pos
            dist = float(np.hypot(*v))
            if dist < FIGHT_RANGE:
                r["fight_risk"] = r.get("fight_risk", 0.0) + R_FIGHT_RISK
            if c.home.contains(o.pos):
                toward = math.cos(math.atan2(v[1], v[0]) - c.heading) > 0.7 and c.speed > MOVING_SPEED
                if dist < CONFRONT_RANGE:
                    c.confronted.add(o.id)
                    if toward:
                        r["confront"] = r.get("confront", 0.0) + R_CONFRONT
            elif o.id in c.confronted:
                c.confronted.discard(o.id)
                r["intruder_left"] = r.get("intruder_left", 0.0) + R_INTRUDER_LEFT
                c.stats["intruders_repelled"] += 1

        if c.hunger >= 1.0:
            r["starving"] = R_STARVING
            c.starve_steps += 1
            c.stats["steps_starving"] += 1
        else:
            c.starve_steps = 0

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
    # Prey (section 15)
    # ------------------------------------------------------------------------------------------
    def area_of(self, p):
        """Territory key if p lies in exactly one home zone, else "neutral" (the overlap counts as neutral)."""
        return self.areas_of(np.asarray(p, dtype=float)[None])[0]

    def areas_of(self, pts):
        keys = list(self.habitat.home_zones)
        centers = np.array([[z.x, z.y] for z in self.habitat.home_zones.values()])
        radii = np.array([z.r for z in self.habitat.home_zones.values()])
        inside = np.linalg.norm(pts[:, None, :] - centers[None], axis=2) <= radii
        out = np.full(len(pts), NEUTRAL, dtype=object)
        one = inside.sum(axis=1) == 1
        out[one] = np.array(keys, dtype=object)[np.argmax(inside[one], axis=1)]
        return out

    def _spawn_point(self, rng, area, avoid=None):
        cfg = self.habitat.prey
        region = self.habitat.home_zones.get(area)
        for _ in range(2000):
            p = self.habitat.random_free_point(rng, margin=cfg["radius"], region=region, avoid=avoid,
                                               avoid_dist=cfg["respawn_min_distance"] if avoid is not None else 0.0)
            if self.area_of(p) == area:
                return p
        raise RuntimeError(f"no free spawn point in {area}")

    def _reset_prey(self, rng):
        cfg = self.habitat.prey
        areas = [k for k in self.habitat.home_zones for _ in range(int(cfg["per_territory"]))]
        areas += [NEUTRAL] * int(cfg["neutral"])
        self.prey_area = np.array(areas, dtype=object)
        self.prey_pos = np.array([self._spawn_point(rng, a) for a in areas])
        self.prey_heading = rng.uniform(-math.pi, math.pi, size=len(areas))
        self.prey_alive = np.ones(len(areas), dtype=bool)
        self.prey_respawn = np.full(len(areas), np.inf)

    def _step_prey(self):
        cfg, rng = self.habitat.prey, self.np_random
        # Neutral prey respawn after their timer, away from every predator
        self.prey_respawn[~self.prey_alive] -= DT
        for i in np.flatnonzero(~self.prey_alive & (self.prey_respawn <= 0)):
            self.prey_pos[i] = self._spawn_point(rng, self.prey_area[i], avoid=self.ego.pos)
            self.prey_heading[i] = rng.uniform(-math.pi, math.pi)
            self.prey_alive[i] = True
            self.prey_respawn[i] = np.inf
        alive = self.prey_alive

        # Flee from the nearest predator within flee_radius, otherwise wander
        preds = np.array([c.pos for c in self.creatures])
        diff = self.prey_pos[:, None, :] - preds[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        nearest = np.argmin(dist, axis=1)
        away = diff[np.arange(len(self.prey_pos)), nearest]
        flee = alive & (dist.min(axis=1) < cfg["flee_radius"])
        wander = alive & ~flee
        self.prey_heading[wander] += rng.normal(0.0, 1.0, size=wander.sum()) * math.sqrt(DT)
        self.prey_heading[flee] = np.arctan2(away[flee, 1], away[flee, 0]) + rng.normal(0.0, 0.3, size=flee.sum())
        speed = np.where(flee, cfg["flee_speed"], cfg["wander_speed"])

        step = (speed * DT)[:, None] * np.stack([np.cos(self.prey_heading), np.sin(self.prey_heading)], axis=1)
        cand = self.prey_pos + step
        free = self.habitat.is_free_many(cand, cfg["radius"])
        same_area = self.areas_of(cand) == self.prey_area
        ok = free & same_area                                   # prey stay inside their own area
        move = alive & ok
        self.prey_pos[move] = cand[move]
        blocked = alive & ~ok
        self.prey_heading[blocked] += math.pi + rng.normal(0.0, 0.5, size=blocked.sum())
        self.prey_heading = (self.prey_heading + math.pi) % (2 * math.pi) - math.pi

    def _catches(self, rewards, before):
        """Each creature eats the nearest prey within its catch radius (random order). Section 15 rules."""
        rng, cfg = self.np_random, self.habitat.prey
        caught = {}
        for idx in rng.permutation(len(self.creatures)):
            c = self.creatures[idx]
            if not self.prey_alive.any():
                break
            d = np.linalg.norm(self.prey_pos - c.pos, axis=1)
            d[~self.prey_alive] = np.inf
            i = int(np.argmin(d))
            if d[i] > c.species.catch_radius:
                continue
            area = self.prey_area[i]
            if area == c.zone:                                  # own territory: gone for good
                self.prey_alive[i] = False
                self.prey_respawn[i] = np.inf
                c.stats["home_prey_eaten"] += 1
            elif area == NEUTRAL:                               # neutral: respawns later
                self.prey_alive[i] = False
                self.prey_respawn[i] = rng.uniform(*cfg["respawn_seconds"])
                c.stats["neutral_prey_eaten"] += 1
            else:                                               # raid: a new prey appears at the eater's home
                self.prey_area[i] = c.zone
                self.prey_pos[i] = self._spawn_point(rng, c.zone, avoid=c.pos)
                self.prey_heading[i] = rng.uniform(-math.pi, math.pi)
                c.stats["raid_prey_eaten"] += 1
            c.hunger = max(0.0, c.hunger - c.species.hunger_per_prey)
            c.stats["prey_caught"] += 1
            caught[c.id] = True
            if before[c.id][0] > 0.5:
                rewards[c.id]["catch"] = R_CATCH * before[c.id][0]
        return caught

    def prey_stock(self):
        """Alive prey per area, e.g. {"myrranox": 3, "dc": 5, "neutral": 2}."""
        out = {k: 0 for k in list(self.habitat.home_zones) + [NEUTRAL]}
        for a, alive in zip(self.prey_area, self.prey_alive):
            out[a] += int(alive)
        return out

    # ------------------------------------------------------------------------------------------
    # Observation (section 8, 27 values, egocentric: x = forward, y = left)
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _ego_vec(c, target, scale):
        """Unit direction to `target` in creature c's frame and distance / scale."""
        v = np.asarray(target, dtype=float) - c.pos
        d = float(np.hypot(*v))
        if d < 1e-6:
            return [0.0, 0.0, 0.0]
        co, si = math.cos(-c.heading), math.sin(-c.heading)
        return [(co * v[0] - si * v[1]) / d, (si * v[0] + co * v[1]) / d, min(d / scale, 1.0)]

    def _observe(self, c):
        sp = c.species
        obs = [c.hunger, c.fatigue, c.alertness, c.speed / sp.max_speed, c.turn_rate / sp.max_turn_rate]

        prey = [0.0, 0.0, 1.0]
        if self.prey_alive.any():
            dist = np.linalg.norm(self.prey_pos - c.pos, axis=1)
            dist[~self.prey_alive] = np.inf
            i = int(np.argmin(dist))
            if dist[i] <= PREY_SENSE_RANGE:
                prey = self._ego_vec(c, self.prey_pos[i], PREY_SENSE_RANGE)
        obs += prey

        i, _ = self.habitat.nearest_rest_spot(c.pos)
        obs += self._ego_vec(c, self.habitat.rest_spots[i].center, DIST_SCALE)

        obs += self._ego_vec(c, c.home.center, DIST_SCALE)
        obs.append(1.0 if c.home.contains(c.pos) else 0.0)

        # Nearest other creature within sight: direction, distance, rival (other species) yes/no
        other = [0.0, 0.0, 1.0, 0.0]
        seen = [(np.hypot(*(o.pos - c.pos)), o) for o in self.creatures if o is not c]
        seen = [s for s in seen if s[0] <= OTHER_SENSE_RANGE]
        if seen:
            _, o = min(seen, key=lambda s: s[0])
            other = self._ego_vec(c, o.pos, DIST_SCALE) + [1.0 if o.species.name != sp.name else 0.0]
        obs += other

        rays = self.habitat.ray_distances(c.pos, c.heading + self._ray_offsets, RAY_RANGE)
        obs += list(rays / RAY_RANGE)
        return np.clip(np.asarray(obs, dtype=np.float32), -1.0, 1.0)

    def observe(self, cid):
        return self._observe(next(c for c in self.creatures if c.id == cid))

    def _info(self, c, rewards):
        return {
            "time": self.t * DT, "id": c.id, "species": c.species.name,
            "x": float(c.pos[0]), "y": float(c.pos[1]),
            "heading": float(c.heading), "speed": c.speed, "turn_rate": c.turn_rate,
            "activity": c.activity, "hunger": c.hunger, "fatigue": c.fatigue, "alertness": c.alertness,
            "reward_terms": dict(rewards), "hit": "hit" in rewards, "prey_stock": self.prey_stock(),
        }
