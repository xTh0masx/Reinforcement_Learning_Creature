"""Scripted (hand-written, not learned) predator: the rival while a species trains alone (training stage 3),
and a baseline to compare trained brains with. Knows the map, unlike a trained brain.

Rules: rest when tired; when hungry hunt the nearest prey at home, and raid the rival territory once the home
stock is down to RAID_BELOW (a raided prey re-appears at home); otherwise patrol the home zone.
Steers around rocks with the obstacle rays.
"""
import math

import numpy as np

from sim.creature_env import NEUTRAL, RAYS, _wrap

RAID_BELOW = 2          # raid once the home stock is down to this many prey
SPRINT_RANGE = 8.0      # m, sprint only this close to the prey (sprinting all the way exhausts the creature)


class ScriptedPredator:
    def __init__(self, seed=0, intrude_prob=0.0):
        """intrude_prob: share of days on which it patrols inside the rival's territory (training defence)."""
        self.rng = np.random.default_rng(seed)
        self.intrude_prob = intrude_prob
        self.intruding = False
        self.patrol = None

    def reset(self):
        self.patrol = None
        self.intruding = self.rng.uniform() < self.intrude_prob

    def __call__(self, obs, c, env):
        sp = c.species
        speed = sp.walk_speed / sp.max_speed
        alive = env.prey_alive
        d = np.linalg.norm(env.prey_pos - c.pos, axis=1)

        if c.fatigue > 0.7 or (c.fatigue > 0.2 and c.activity == "rest"):
            spots = [s for s in env.habitat.rest_spots if c.home.contains(s.center)] or env.habitat.rest_spots
            s = min(spots, key=lambda s: s.distance(c.pos))
            if s.distance(c.pos) < s.r * 0.5:
                return np.array([env.speed_action(c, 0.0), 0.0])
            target = s.center
        elif c.hunger > 0.3 and alive.any():
            home = alive & (env.prey_area == c.zone)
            raid = alive & (env.prey_area != c.zone) & (env.prey_area != NEUTRAL)
            if home.sum() > RAID_BELOW or not raid.any():
                pool = home if home.any() else alive
            else:
                pool = raid
            dd = np.where(pool, d, np.inf)
            i = int(np.argmin(dd))
            target = env.prey_pos[i]
            if dd[i] < SPRINT_RANGE:                    # walk there, sprint only for the final catch
                speed = 1.0
        elif self.intruding:
            rival_zone = next(z for k, z in env.habitat.home_zones.items() if k != c.zone)
            if self.patrol is None or np.hypot(*(self.patrol - c.pos)) < 2 or not rival_zone.contains(self.patrol):
                self.patrol = env.habitat.random_free_point(self.rng, margin=2, region=rival_zone)
            target = self.patrol
        elif not c.home.contains(c.pos):
            target = c.home.center
        else:
            if self.patrol is None or np.hypot(*(self.patrol - c.pos)) < 2:
                self.patrol = env.habitat.random_free_point(self.rng, margin=2, region=c.home)
            target = self.patrol

        err = _wrap(math.atan2(target[1] - c.pos[1], target[0] - c.pos[0]) - c.heading)
        rays = obs[RAYS]                                  # 0 = ahead, 1 / 7 = 45 deg left / right
        if min(rays[0], rays[1], rays[7]) < 0.3:
            err += 0.8 if rays[1] > rays[7] else -0.8
        turn = float(np.clip(err / math.radians(45), -1, 1))
        speed *= max(0.25, math.cos(min(abs(err), math.pi / 2)))
        return np.array([env.speed_action(c, speed), turn])
