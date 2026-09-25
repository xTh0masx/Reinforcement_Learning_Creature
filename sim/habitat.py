"""Loads sim/habitat.json and answers geometry questions (free space, zones, ray casts)."""
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

DEFAULT_PATH = Path(__file__).with_name("habitat.json")


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    r: float
    kind: str = ""

    @classmethod
    def from_json(cls, d, kind_key="kind"):
        return cls(float(d["center"][0]), float(d["center"][1]), float(d["radius"]), d.get(kind_key, ""))

    @property
    def center(self):
        return np.array([self.x, self.y])

    def distance(self, p):
        """Distance from the centre to point p."""
        return float(np.hypot(p[0] - self.x, p[1] - self.y))

    def contains(self, p):
        return self.distance(p) <= self.r


class Habitat:
    def __init__(self, data):
        b = data["boundary"]
        self.xmin, self.xmax = float(b["xmin"]), float(b["xmax"])
        self.ymin, self.ymax = float(b["ymin"]), float(b["ymax"])
        self.home_zones = {k: Circle.from_json(v, "terrain") for k, v in data["home_zones"].items()}
        self.water = Circle.from_json(data["water"])
        self.rest_spots = [Circle.from_json(d) for d in data["rest_spots"]]
        self.obstacles = [Circle.from_json(d) for d in data["obstacles"]]
        self.prey = dict(data["prey"])

        self._obs_xy = np.array([[o.x, o.y] for o in self.obstacles]).reshape(-1, 2)
        self._obs_r = np.array([o.r for o in self.obstacles])
        self._rest_xy = np.array([[s.x, s.y] for s in self.rest_spots]).reshape(-1, 2)
        self._rest_r = np.array([s.r for s in self.rest_spots])

    @classmethod
    def load(cls, path=None):
        with open(path or DEFAULT_PATH, encoding="utf-8") as f:
            return cls(json.load(f))

    # --- free space -------------------------------------------------------------------------
    def is_free_many(self, pts, margin=0.0):
        """Bool mask: which points (N, 2) are inside the boundary and outside every rock (+margin)."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        ok = ((pts[:, 0] >= self.xmin + margin) & (pts[:, 0] <= self.xmax - margin)
              & (pts[:, 1] >= self.ymin + margin) & (pts[:, 1] <= self.ymax - margin))
        if len(self._obs_r):
            d = np.linalg.norm(pts[:, None, :] - self._obs_xy[None], axis=2)
            ok &= (d > self._obs_r + margin).all(axis=1)
        return ok

    def is_free(self, p, margin=0.0):
        return bool(self.is_free_many(p, margin)[0])

    def random_free_point(self, rng, margin=0.0, region=None, avoid=None, avoid_dist=0.0, tries=1000):
        """Uniform random free point, optionally inside a Circle `region` and >= avoid_dist from `avoid`."""
        for _ in range(tries):
            if region is None:
                p = rng.uniform([self.xmin, self.ymin], [self.xmax, self.ymax])
            else:
                a, rad = rng.uniform(0, 2 * np.pi), region.r * np.sqrt(rng.uniform())
                p = np.array([region.x + rad * np.cos(a), region.y + rad * np.sin(a)])
            if avoid is not None and np.hypot(*(p - avoid)) < avoid_dist:
                continue
            if self.is_free(p, margin):
                return p
        raise RuntimeError("No free point found - check habitat.json")

    # --- rest spots ---------------------------------------------------------------------------
    def nearest_rest_spot(self, p):
        """(index, distance to centre) of the nearest rest spot."""
        d = np.linalg.norm(self._rest_xy - p, axis=1)
        i = int(np.argmin(d))
        return i, float(d[i])

    def on_rest_spot(self, p):
        return bool((np.linalg.norm(self._rest_xy - p, axis=1) <= self._rest_r).any())

    # --- ray casting --------------------------------------------------------------------------
    def ray_distances(self, p, angles, max_range):
        """Distance along each world-frame angle to the nearest rock or boundary, capped at max_range."""
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)            # (R, 2)
        dist = np.full(len(angles), max_range)

        # Boundary: axis-aligned walls.
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dirs[:, 0] > 0, (self.xmax - p[0]) / dirs[:, 0],
                          np.where(dirs[:, 0] < 0, (self.xmin - p[0]) / dirs[:, 0], np.inf))
            ty = np.where(dirs[:, 1] > 0, (self.ymax - p[1]) / dirs[:, 1],
                          np.where(dirs[:, 1] < 0, (self.ymin - p[1]) / dirs[:, 1], np.inf))
        dist = np.minimum(dist, np.maximum(np.minimum(tx, ty), 0.0))

        # Rocks: ray-circle intersection, nearest hit in front of the origin.
        if len(self._obs_r):
            f = p - self._obs_xy                                             # (O, 2)
            b = dirs @ f.T                                                   # (R, O)
            c = (f ** 2).sum(axis=1) - self._obs_r ** 2                      # (O,)
            disc = b ** 2 - c
            with np.errstate(invalid="ignore"):
                t = -b - np.sqrt(disc)
            t = np.where((disc >= 0) & (t >= 0), t, np.inf)
            dist = np.minimum(dist, t.min(axis=1))
        return dist
