"""Export the DC mesh split by bone (rest pose, body frame) for the mesh-based physics body (sim/dc_hulls.py).

Each vertex of DC_body_proxy goes to the bone with its highest weight; claws, teeth, tongue and eyes (rigidly
parented to bones) go to their bone. Writes sim/dc_parts.json: {bone: [[x, y, z], ...]} (at most 4000 points each).

    blender -b blender/habitat_showcase.blend --python blender/scripts/export_dc_parts.py
"""
import json
import os

import bpy
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAX_POINTS = 4000

arm = bpy.data.objects["DC_rig"]
arm.data.pose_position = "REST"
bpy.context.view_layer.update()
proxy = bpy.data.objects["DC_body_proxy"]
names = {g.index: g.name for g in proxy.vertex_groups}
parts = {}
M = proxy.matrix_world
for v in proxy.data.vertices:
    if not v.groups:
        continue
    g = max(v.groups, key=lambda g: g.weight)
    parts.setdefault(names[g.group], []).append(tuple(M @ v.co))
for obj in bpy.data.objects:                       # rigid parts parented to a bone (claws, teeth, eyes, tongue)
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
    print(f"PARTS {bone:12s} {len(parts[bone]):6d} vertices")
with open(os.path.join(REPO, "sim", "dc_parts.json"), "w") as f:
    json.dump(out, f)
print("PARTS wrote sim/dc_parts.json")
