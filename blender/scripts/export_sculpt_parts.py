"""Export the DC's full-detail sculpt (PM3D_MainBody1, ~970k faces) split by bone for the detailed physics body.

The sculpt has no skin weights; each sculpt vertex takes the bone of the nearest vertex of the weighted proxy
(DC_body_proxy). Rigid parts (claws, teeth, tongue, eyes) go to their bone. Writes sim/dc_parts_sculpt.json
(at most MAX_POINTS points per bone, randomly sampled).

    blender -b blender/habitat_showcase.blend --python blender/scripts/export_sculpt_parts.py
"""
import json
import os

import bpy
import numpy as np
from mathutils import kdtree

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAX_POINTS = 20000

arm = bpy.data.objects["DC_rig"]
arm.data.pose_position = "REST"
bpy.context.view_layer.update()
proxy, sculpt = bpy.data.objects["DC_body_proxy"], bpy.data.objects["PM3D_MainBody1"]
names = {g.index: g.name for g in proxy.vertex_groups}
# The proxy is a decimated copy of the sculpt's mesh data (same local coordinates); the sculpt's parent was moved
# afterwards, so the sculpt is placed with the proxy's matrix to land in the body frame.
Mp = Ms = proxy.matrix_world
tree = kdtree.KDTree(len(proxy.data.vertices))
bone_of = []
for i, v in enumerate(proxy.data.vertices):
    tree.insert(Mp @ v.co, i)
    bone_of.append(names[max(v.groups, key=lambda g: g.weight).group] if v.groups else None)
tree.balance()
parts = {}
for v in sculpt.data.vertices:
    p = Ms @ v.co
    _, idx, _ = tree.find(p)
    b = bone_of[idx]
    if b:
        parts.setdefault(b, []).append(tuple(p))
for obj in bpy.data.objects:
    if obj.type == "MESH" and obj.parent == arm and obj.parent_type == "BONE" and obj.parent_bone:
        Mo = obj.matrix_world
        parts.setdefault(obj.parent_bone, []).extend(tuple(Mo @ v.co) for v in obj.data.vertices)
rng = np.random.default_rng(0)
out = {}
for bone, pts in parts.items():
    pts = np.array(pts)
    if len(pts) > MAX_POINTS:
        pts = pts[rng.choice(len(pts), MAX_POINTS, replace=False)]
    out[bone] = np.round(pts, 4).tolist()
    print(f"SCULPT {bone:12s} {len(parts[bone]):7d} vertices")
print(f"SCULPT total sculpt vertices {len(sculpt.data.vertices)}, proxy {len(proxy.data.vertices)}")
with open(os.path.join(REPO, "sim", "dc_parts_sculpt.json"), "w") as f:
    json.dump(out, f)
print("SCULPT wrote sim/dc_parts_sculpt.json")

# Triangle mesh for the convex decomposition (sim/hulls_detailed.py): the sculpt decimated to ~MESH_FACES faces plus
# the rigid parts, every vertex labelled with its bone. Written to sim/dc_sculpt_mesh.npz.
MESH_FACES = 150000
dup = sculpt.copy()
dup.data = sculpt.data.copy()
bpy.context.scene.collection.objects.link(dup)
mod = dup.modifiers.new("dec", "DECIMATE")
mod.ratio = MESH_FACES / max(len(sculpt.data.polygons), 1)
mod.use_collapse_triangulate = True
dg = bpy.context.evaluated_depsgraph_get()
me = dup.evaluated_get(dg).to_mesh()
me.calc_loop_triangles()
V, F, L = [], [], []
bones = sorted({b for b in bone_of if b} | {o.parent_bone for o in bpy.data.objects if o.parent == arm and o.parent_bone})
bidx = {b: i for i, b in enumerate(bones)}


def add(mesh, M, bone=None):
    base = len(V)
    for v in mesh.vertices:
        p = M @ v.co
        V.append(tuple(p))
        L.append(bidx[bone] if bone else (bidx[bone_of[tree.find(p)[1]]] if bone_of[tree.find(p)[1]] else -1))
    for t in mesh.loop_triangles:
        F.append(tuple(base + i for i in t.vertices))


add(me, Ms)
for obj in bpy.data.objects:
    if obj.type == "MESH" and obj.parent == arm and obj.parent_type == "BONE" and obj.parent_bone:
        obj.data.calc_loop_triangles()
        add(obj.data, obj.matrix_world, obj.parent_bone)
np.savez_compressed(os.path.join(REPO, "sim", "dc_sculpt_mesh.npz"), vertices=np.array(V, np.float32),
                    faces=np.array(F, np.int32), labels=np.array(L, np.int16), bones=np.array(bones))
print(f"SCULPT mesh {len(V)} vertices, {len(F)} triangles -> sim/dc_sculpt_mesh.npz")
