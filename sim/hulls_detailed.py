"""Detailed physics shapes for the DC (walking v12): every bone is decomposed into several convex pieces so the physics
body follows the real model closely - concave places (neck / shoulder, belly, between the toes) stay concave and the
spikes and horns are kept. A single hull per bone (sim/hulls.py) fills every concavity.

Source: the full-detail sculpt with its claws, teeth and other rigid parts, every vertex labelled with its bone
(blender/scripts/export_sculpt_parts.py -> sim/dc_sculpt_mesh.npz). Per bone: the bone's triangles (torso / spine and
shoulder hump reassigned as in sim/hulls.py), decimated to DECIMATE_FACES, decomposed with CoACD (approximate convex
decomposition, Wei et al. 2022 - the tool used for MuJoCo's model collection). The right limbs are mirror images of
the left ones.

Fidelity: points on the outer surface of the pieces (inside no other piece) are compared with the real surface.

    python -m sim.hulls_detailed           # builds sim/dc_hulls_detailed.json and prints the fit per bone
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from sim.hulls import REASSIGN

HERE = Path(__file__).parent
TOL = 0.015                 # m: allowed concavity per piece; CoACD's threshold is relative to the part's size, so it is
                            # set per bone to TOL / size (a fixed 0.02 gave a lower leg 24 pieces at 6 mm)
MAX_PIECES = {"torso": 16, "spine_rear": 16, "neck": 16, "head": 12, "tail_1": 8, "tail_2": 6}   # limbs: 8
LIMB_PIECES = 8
MAX_VERTS = 64              # vertices per piece (MuJoCo speed)
DECIMATE_FACES = 10000      # triangles per bone fed to CoACD (30000: > 45 min without finishing the first bones)
CACHE = HERE / "_coacd_cache"  # one file per finished bone, so an interrupted build resumes


def bone_meshes(species="dc"):
    d = np.load(HERE / f"{species}_sculpt_mesh.npz")
    V, F, L, names = d["vertices"].astype(float), d["faces"], d["labels"], [str(b) for b in d["bones"]]
    fl = L[F]
    face_bone = np.where((fl[:, 0] == fl[:, 1]) | (fl[:, 0] == fl[:, 2]), fl[:, 0], fl[:, 1])
    cx = V[F].mean(1)[:, 0]
    for src, dst, x, side in REASSIGN[species]:
        move = (face_bone == names.index(src)) & ((cx < x) if side == "behind" else (cx >= x))
        face_bone[move] = names.index(dst)
    out = {}
    for i, bone in enumerate(names):
        if "_R_s" in bone:
            continue
        f = F[face_bone == i]
        used, inv = np.unique(f, return_inverse=True)
        out[bone] = (V[used], inv.reshape(-1, 3))
    for bone in [b for b in names if "_R_s" in b]:
        v, f = out[bone.replace("_R_s", "_L_s")]
        out[bone] = (v * np.array([1.0, -1.0, 1.0]), f[:, ::-1])
    return out


def _decimate(v, f, n):
    if len(f) <= n:
        return v, f
    import fast_simplification
    v2, f2 = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=1 - n / len(f))
    return v2.astype(float), f2


def _reduce(v, n):
    v = v[ConvexHull(v).vertices]
    if len(v) <= n:
        return v
    keep = [int(np.argmax(np.linalg.norm(v - v.mean(0), axis=1)))]
    d = np.linalg.norm(v - v[keep[0]], axis=1)
    while len(keep) < n:
        i = int(np.argmax(d))
        keep.append(i)
        d = np.minimum(d, np.linalg.norm(v - v[i], axis=1))
    return v[keep]


def _surface_samples(v, n, rng):
    h = ConvexHull(v)
    tri = v[h.simplices]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    idx = rng.choice(len(tri), n, p=area / area.sum())
    w = rng.dirichlet(np.ones(3), n)
    return np.einsum("nk,nkd->nd", w, tri[idx])


def fit(pieces, mesh_v, mesh_f, rng, n=4000):
    """95th percentile and max distance (m) of the pieces' outer surface from the real surface, and the share of the
    real surface (vertices) farther than 1.5 cm from any piece surface (parts the pieces miss)."""
    tri = mesh_v[mesh_f]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    idx = rng.choice(len(tri), 30000, p=area / area.sum())
    surf = np.einsum("nk,nkd->nd", rng.dirichlet(np.ones(3), 30000), tri[idx])
    tree = cKDTree(surf)
    eqs = [ConvexHull(p).equations for p in pieces]
    outer = []
    for k, p in enumerate(pieces):
        s = _surface_samples(p, max(200, n // len(pieces)), rng)
        inside_other = np.zeros(len(s), bool)
        for j, e in enumerate(eqs):
            if j != k:
                inside_other |= np.all(s @ e[:, :3].T + e[:, 3] < -1e-4, axis=1)
        outer.append(s[~inside_other])
    outer = np.vstack(outer)
    d = tree.query(outer)[0]
    miss = cKDTree(outer).query(surf)[0]
    return float(np.percentile(d, 95)), float(d.max()), float(np.mean(miss > 0.015))


def build(species="dc", tol=TOL, only=None):
    """only: compute just these bones' pieces into the cache (parallel workers: --only head,neck)."""
    import coacd
    coacd.set_log_level("error")
    rng = np.random.default_rng(0)
    meshes = bone_meshes(species)
    out, report = {}, []
    order = sorted(meshes, key=lambda b: ("_R_s" in b, len(meshes[b][1])))     # small parts first, mirrors last
    if only:
        order = [b for b in order if b in only]
    for bone in order:
        v, f = meshes[bone]
        t0 = time.time()
        if "_R_s" in bone:
            left = out[bone.replace("_R_s", "_L_s")]
            out[bone] = [{"vertices": (np.asarray(h["vertices"]) * [1, -1, 1]).round(4).tolist(), "volume": h["volume"]}
                         for h in left]
            report.append((bone,) + next(r for r in report if r[0] == bone.replace("_R_s", "_L_s"))[1:])
            continue
        v, f = _decimate(v, f, DECIMATE_FACES)
        size = float(np.linalg.norm(v.max(0) - v.min(0)))
        threshold = min(max(tol / size, 0.01), 0.1)
        n_max = MAX_PIECES.get(bone, LIMB_PIECES)
        cache = CACHE / f"{species}_{bone}_{tol}_{n_max}.json"
        if cache.exists():
            pieces = [np.asarray(p) for p in json.loads(cache.read_text())]
        else:
            parts = coacd.run_coacd(coacd.Mesh(v, f), threshold=threshold, max_convex_hull=n_max,
                                    preprocess_mode="auto", resolution=1000, mcts_iterations=60, mcts_max_depth=2,
                                    mcts_nodes=12, seed=0)
            pieces = []
            for pv, _ in parts:
                pv = np.asarray(pv, float)
                if len(pv) >= 4:
                    try:
                        pieces.append(_reduce(pv, MAX_VERTS))
                    except Exception:
                        pass
            CACHE.mkdir(exist_ok=True)
            cache.write_text(json.dumps([p.round(5).tolist() for p in pieces]))
        out[bone] = [{"vertices": p.round(4).tolist(), "volume": float(ConvexHull(p).volume)} for p in pieces]
        # (mass_volume: added by weight_spikes(), run after the build)
        if only:
            print(f"  {bone:12s} cached", flush=True)
            continue
        one = _reduce(v, 120)
        report.append((bone, len(pieces), fit([one], v, f, rng), fit(pieces, v, f, rng),
                       float(ConvexHull(one).volume), sum(h["volume"] for h in out[bone]), time.time() - t0))
        r = report[-1]
        print(f"  {bone:12s} {r[1]:3d} pieces  {r[6]:5.0f} s  fit {100 * r[3][0]:.1f} cm (one hull {100 * r[2][0]:.1f} cm)",
              flush=True)
    if not only:
        (HERE / f"{species}_hulls_detailed.json").write_text(json.dumps(out))
    return out, report


SPIKE_DENSITY = 1 / 3       # spikes, horns and crest are horn / bone, much lighter than the muscle of the body


def weight_spikes(species="dc", n=6000):
    """Owner, 13 Oct: make the spikes lighter. The spike-trimmed single hulls of sim/hulls.py mark each part's core
    body; every piece gets a mass volume = volume inside the core + SPIKE_DENSITY x volume outside (Monte Carlo)."""
    from sim.hulls import SPIKY, load as load_single
    rng = np.random.default_rng(0)
    core = {b: ConvexHull(np.asarray(h["vertices"])).equations for b, h in load_single(species).items()}
    path = HERE / f"{species}_hulls_detailed.json"
    data = json.loads(path.read_text())
    for bone, pieces in data.items():
        for p in pieces:
            if bone not in SPIKY[species]:                  # limbs have no spikes: full density
                p["mass_volume"] = p["volume"]
                continue
            v = np.asarray(p["vertices"])
            eq = ConvexHull(v).equations
            pts = rng.uniform(v.min(0), v.max(0), size=(n, 3))
            pts = pts[np.all(pts @ eq[:, :3].T + eq[:, 3] <= 0, axis=1)]
            inside = np.all(pts @ core[bone][:, :3].T + core[bone][:, 3] <= 1e-4, axis=1).mean() if len(pts) else 1.0
            p["mass_volume"] = p["volume"] * (inside + SPIKE_DENSITY * (1 - inside))
    path.write_text(json.dumps(data))
    return data


def load(species="dc"):
    path = HERE / f"{species}_hulls_detailed.json"
    return json.loads(path.read_text()) if path.exists() else build(species)[0]


if __name__ == "__main__":
    if "--spikes" in sys.argv:
        data = weight_spikes("dc")
        for bone, pieces in data.items():
            v, m = sum(p["volume"] for p in pieces), sum(p["mass_volume"] for p in pieces)
            print(f"{bone:12s} mass volume {100 * m / v:4.0f} % of the volume")
        sys.exit()
    if "--only" in sys.argv:
        build("dc", only=sys.argv[sys.argv.index("--only") + 1].split(","))
        sys.exit()
    out, report = build(sys.argv[1] if len(sys.argv) > 1 else "dc")
    print(f"{'bone':12s} pieces | overfill 95% / max, surface missed >1.5cm: one hull -> pieces | volume one -> pieces")
    for bone, n, a, b, v1, v2, _ in sorted(report):
        print(f"{bone:12s} {n:5d}  | {100 * a[0]:4.1f}/{100 * a[1]:4.1f} cm {100 * a[2]:3.0f}% -> "
              f"{100 * b[0]:4.1f}/{100 * b[1]:4.1f} cm {100 * b[2]:3.0f}% | {1000 * v1:6.1f} L -> {1000 * v2:6.1f} L")
    print("total pieces", sum(r[1] for r in report),
          f"| volume {1000 * sum(r[4] for r in report):.0f} L -> {1000 * sum(r[5] for r in report):.0f} L")
