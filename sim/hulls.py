"""Mesh-based physics shapes: one convex hull per bone, cut from the creature's real mesh (DC walking v8+,
Myrranox walking v9+).

sim/<species>_parts.json (blender/scripts/export_parts.py) holds the mesh vertices of each bone in the body frame
at the rest pose. Points are first moved to the body part that physically carries them (the skin weights do not
follow the physics joints exactly, e.g. the DC's "neck" weights include the shoulder hump behind the neck joint).
The DC's trunk, neck, head and tail carry thin spikes and horns on top; a hull around them would add a lot of
volume that is not there, so points that stick up further than the part's own half-width above its bone are
dropped (the belly and the jaw stay). The Myrranox has no such spikes. Legs, arms and paws (claws) stay whole.

    python -m sim.hulls dc          # builds sim/dc_hulls.json and prints the shapes
    python -m sim.hulls myrranox
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

HERE = Path(__file__).parent
MAX_HULL_VERTS = 120            # MuJoCo is fast with small hulls; keep the most extreme points
UP_FACTOR = 1.1                 # keep points up to 1.1 x the part's half-width above the bone axis

# spiky parts with their bone axis in the body frame (rig bones) - trimmed on top
SPIKY = {
    "dc": {"torso": ((-0.25, 0, 0.66), (0.50, 0, 0.74)), "spine_rear": ((0.12, 0, 0.70), (-0.25, 0, 0.66)),
           "neck": ((0.50, 0, 0.80), (1.00, 0, 0.90)), "head": ((1.00, 0, 0.90), (1.32, 0, 0.80)),
           "tail_1": ((-0.30, 0, 0.60), (-0.80, 0, 0.50)), "tail_2": ((-0.80, 0, 0.50), (-1.35, 0, 0.42))},
    "myrranox": {},
}
# (from bone, to bone, x limit, side): points of `from` on `side` of the physics joint at x move to `to`
REASSIGN = {
    "dc": [("torso", "spine_rear", 0.12, "behind"),       # the spine joint splits the trunk
           ("spine_rear", "torso", 0.12, "front"),
           ("neck", "torso", 0.50, "behind")],            # the shoulder hump behind the neck joint is chest
    "myrranox": [],
}


def _trim_spikes(p, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
    axis = a + t[:, None] * (b - a)
    half_width = np.percentile(np.abs(p[:, 1] - axis[:, 1]), 95)
    return p[(p[:, 2] - axis[:, 2]) <= UP_FACTOR * half_width]


def _reduce(p, n):
    """Hull vertices; if too many, keep the n most extreme (greedy farthest-point sampling on the hull)."""
    v = p[ConvexHull(p).vertices]
    if len(v) <= n:
        return v
    keep = [int(np.argmax(np.linalg.norm(v - v.mean(0), axis=1)))]
    d = np.linalg.norm(v - v[keep[0]], axis=1)
    while len(keep) < n:
        i = int(np.argmax(d))
        keep.append(i)
        d = np.minimum(d, np.linalg.norm(v - v[i], axis=1))
    return v[keep]


def build(species):
    parts = {k: np.asarray(v, dtype=float) for k, v in json.loads((HERE / f"{species}_parts.json").read_text()).items()}
    for src, dst, x, side in REASSIGN[species]:
        p = parts[src]
        move = p[:, 0] < x if side == "behind" else p[:, 0] >= x
        parts[dst] = np.vstack([parts[dst], p[move]])
        parts[src] = p[~move]
    out = {}
    for bone, p in parts.items():
        if bone in SPIKY[species]:
            p = _trim_spikes(p, *SPIKY[species][bone])
        v = _reduce(p, MAX_HULL_VERTS)
        hull = ConvexHull(v)
        out[bone] = {"vertices": np.round(v, 4).tolist(), "volume": float(hull.volume),
                     "min": v.min(0).round(4).tolist(), "max": v.max(0).round(4).tolist()}
    (HERE / f"{species}_hulls.json").write_text(json.dumps(out))
    return out


def load(species):
    path = HERE / f"{species}_hulls.json"
    return json.loads(path.read_text()) if path.exists() else build(species)


if __name__ == "__main__":
    species = sys.argv[1] if len(sys.argv) > 1 else "dc"
    hulls = build(species)
    total = sum(h["volume"] for h in hulls.values())
    for bone, h in sorted(hulls.items(), key=lambda kv: -kv[1]["volume"]):
        print(f"{bone:12s} {len(h['vertices']):4d} verts  volume {h['volume'] * 1000:6.2f} L ({100 * h['volume'] / total:4.1f} %)  "
              f"x {h['min'][0]:+.2f}..{h['max'][0]:+.2f}  z {h['min'][2]:+.2f}..{h['max'][2]:+.2f}")
    print(f"total volume {total * 1000:.1f} L")
