"""Mesh-based physics shapes for the DC (walking v8): one convex hull per bone, cut from the real DC mesh.

sim/dc_parts.json (blender/scripts/export_dc_parts.py) holds the mesh vertices of each bone in the body frame at
the rest pose. The trunk, neck, head and tail parts carry thin spikes and horns on top; a hull around them would
add a lot of volume that is not there, so points that stick up further than the part's own half-width above its
bone are dropped (the belly and the jaw stay). Legs, arms and paws (with claws) are kept whole.

    python -m sim.dc_hulls          # builds sim/dc_hulls.json and prints the shapes
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

HERE = Path(__file__).parent
PARTS = HERE / "dc_parts.json"
HULLS = HERE / "dc_hulls.json"
MAX_HULL_VERTS = 120            # MuJoCo is fast with small hulls; keep the most extreme points
# bones of the trunk / neck / head / tail (the spiky parts) with their axis in the body frame (rig bones)
AXES = {"torso": ((-0.25, 0, 0.66), (0.50, 0, 0.74)), "spine_rear": ((0.12, 0, 0.70), (-0.25, 0, 0.66)),
        "neck": ((0.50, 0, 0.80), (1.00, 0, 0.90)), "head": ((1.00, 0, 0.90), (1.32, 0, 0.80)),
        "tail_1": ((-0.30, 0, 0.60), (-0.80, 0, 0.50)), "tail_2": ((-0.80, 0, 0.50), (-1.35, 0, 0.42))}
UP_FACTOR = 1.1                 # keep points up to 1.1 x the part's half-width above the bone axis
SPINE_X = 0.12                  # physics spine joint (sim/bodies.py RearSegment point): chest in front, rear behind


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


def build():
    parts = json.loads(PARTS.read_text())
    # The skin weights give the "torso" bone most of the mid-body; in physics the body bends at the spine joint,
    # so everything behind it belongs to the rear body.
    trunk = np.asarray(parts["torso"] + parts["spine_rear"], dtype=float)
    parts["torso"] = trunk[trunk[:, 0] >= SPINE_X].tolist()
    parts["spine_rear"] = trunk[trunk[:, 0] < SPINE_X].tolist()
    out = {}
    for bone, pts in parts.items():
        p = np.asarray(pts, dtype=float)
        if bone in AXES:
            p = _trim_spikes(p, *AXES[bone])
        v = _reduce(p, MAX_HULL_VERTS)
        hull = ConvexHull(v)
        out[bone] = {"vertices": np.round(v, 4).tolist(), "volume": float(hull.volume),
                     "min": v.min(0).round(4).tolist(), "max": v.max(0).round(4).tolist()}
    HULLS.write_text(json.dumps(out))
    return out


def load():
    if not HULLS.exists():
        return build()
    return json.loads(HULLS.read_text())


if __name__ == "__main__":
    hulls = build()
    total = sum(h["volume"] for h in hulls.values())
    for bone, h in sorted(hulls.items(), key=lambda kv: -kv[1]["volume"]):
        print(f"{bone:12s} {len(h['vertices']):4d} verts  volume {h['volume'] * 1000:6.1f} L ({100 * h['volume'] / total:4.1f} %)  "
              f"x {h['min'][0]:+.2f}..{h['max'][0]:+.2f}  z {h['min'][2]:+.2f}..{h['max'][2]:+.2f}")
    print(f"total volume {total * 1000:.0f} L -> density for 60 kg: {60 / total:.0f} kg/m^3")
