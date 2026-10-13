"""Export a creature's mesh split by bone (rest pose, body frame) for its mesh-based physics body (sim/hulls.py).

Skinned mesh (DC_body_proxy): each vertex goes to the bone with its highest weight. Rigid parts parented to a bone
(the Myrranox's body parts, the DC's claws, teeth, tongue, eyes) go to that bone.
Writes sim/<species>_parts.json: {bone: [[x, y, z], ...]} (at most 4000 points each).

    blender -b blender/habitat_showcase.blend --python blender/scripts/export_parts.py -- dc
    blender -b blender/habitat_showcase.blend --python blender/scripts/export_parts.py -- myrranox
"""
import json
import os
import sys

import bpy
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAX_POINTS = 4000
RIGS = {"dc": ("DC_rig", "DC_body_proxy"), "myrranox": ("Myrranox_rig", None)}

species = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "dc"
rig_name, skinned = RIGS[species]
arm = bpy.data.objects[rig_name]
arm.data.pose_position = "REST"
bpy.context.view_layer.update()
parts = {}
if skinned:
    proxy = bpy.data.objects[skinned]
    names = {g.index: g.name for g in proxy.vertex_groups}
    M = proxy.matrix_world
    for v in proxy.data.vertices:
        if v.groups:
            g = max(v.groups, key=lambda g: g.weight)
            parts.setdefault(names[g.group], []).append(tuple(M @ v.co))
for obj in bpy.data.objects:                       # rigid parts parented to a bone
    if obj.type == "MESH" and obj.parent == arm and obj.parent_type == "BONE" and obj.parent_bone:
        Mo = obj.matrix_world
        parts.setdefault(obj.parent_bone, []).extend(tuple(Mo @ v.co) for v in obj.data.vertices)
        print(f"PARTS   {obj.name:24s} -> {obj.parent_bone}")
rng = np.random.default_rng(0)
out = {}
for bone, pts in parts.items():
    pts = np.array(pts)
    if len(pts) > MAX_POINTS:
        pts = pts[rng.choice(len(pts), MAX_POINTS, replace=False)]
    out[bone] = np.round(pts, 4).tolist()
    print(f"PARTS {bone:12s} {len(parts[bone]):6d} vertices")
with open(os.path.join(REPO, "sim", f"{species}_parts.json"), "w") as f:
    json.dump(out, f)
print(f"PARTS wrote sim/{species}_parts.json")
