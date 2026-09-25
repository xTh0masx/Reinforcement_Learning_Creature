"""Build a simple habitat in Blender from sim/habitat.json (flat ground, zones, water, rest spots, rocks).

Run inside Blender, e.g. from the Text Editor:
    HABITAT_JSON = r"C:/.../sim/habitat.json"; exec(open(r"C:/.../build_habitat.py").read())
or call build(habitat_json_path). Re-running replaces the "Habitat" collection.
"""
import json
import math

import bpy

COLLECTION = "Habitat"
COLORS = {
    "ground": (0.42, 0.36, 0.28, 1),
    "myrranox": (0.78, 0.62, 0.35, 1),
    "dc": (0.30, 0.50, 0.25, 1),
    "water": (0.15, 0.35, 0.60, 1),
    "rest": (0.30, 0.20, 0.12, 1),
    "rock": (0.35, 0.35, 0.35, 1),
    "wall": (0.28, 0.25, 0.22, 1),
}


def _material(name, color, roughness=0.9):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    mat.diffuse_color = color
    return mat


def _collection():
    col = bpy.data.collections.get(COLLECTION)
    if col is None:
        col = bpy.data.collections.new(COLLECTION)
        bpy.context.scene.collection.children.link(col)
    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    return col


def _add(col, name, mesh_op, mat, location, scale=(1, 1, 1), rotation=(0, 0, 0), **kw):
    mesh_op(location=location, rotation=rotation, **kw)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)
    return obj


def build(habitat_json):
    with open(habitat_json, encoding="utf-8") as f:
        h = json.load(f)
    col = _collection()
    mats = {k: _material(f"RL_{k}", c) for k, c in COLORS.items()}
    b = h["boundary"]
    w, d = b["xmax"] - b["xmin"], b["ymax"] - b["ymin"]
    cx, cy = (b["xmax"] + b["xmin"]) / 2, (b["ymax"] + b["ymin"]) / 2

    _add(col, "Ground", bpy.ops.mesh.primitive_plane_add, mats["ground"], (cx, cy, 0), size=1, scale=(w, d, 1))

    for i, (name, z) in enumerate(h["home_zones"].items()):   # stacked slightly so overlaps don't flicker
        _add(col, f"Zone_{name}", bpy.ops.mesh.primitive_cylinder_add, mats.get(name, mats["ground"]),
             (z["center"][0], z["center"][1], 0.01 + 0.005 * i), vertices=96, radius=z["radius"], depth=0.02)
    wz = h["water"]
    _add(col, "Water", bpy.ops.mesh.primitive_cylinder_add, mats["water"],
         (wz["center"][0], wz["center"][1], 0.02), vertices=64, radius=wz["radius"], depth=0.04)
    for i, s in enumerate(h["rest_spots"]):
        _add(col, f"Rest_{i}_{s.get('kind', '')}", bpy.ops.mesh.primitive_cylinder_add, mats["rest"],
             (s["center"][0], s["center"][1], 0.03), vertices=48, radius=s["radius"], depth=0.06)
    for i, o in enumerate(h["obstacles"]):
        r = o["radius"]
        _add(col, f"Rock_{i}", bpy.ops.mesh.primitive_ico_sphere_add, mats["rock"],
             (o["center"][0], o["center"][1], 0.2 * r), subdivisions=2, radius=1, scale=(r, r, 0.8 * r))

    t, hgt = 1.0, 3.0
    for name, loc, sc in (("Wall_N", (cx, b["ymax"] + t / 2, hgt / 2), (w + 2 * t, t, hgt)),
                          ("Wall_S", (cx, b["ymin"] - t / 2, hgt / 2), (w + 2 * t, t, hgt)),
                          ("Wall_E", (b["xmax"] + t / 2, cy, hgt / 2), (t, d, hgt)),
                          ("Wall_W", (b["xmin"] - t / 2, cy, hgt / 2), (t, d, hgt))):
        _add(col, name, bpy.ops.mesh.primitive_cube_add, mats["wall"], loc, size=1, scale=sc)

    sun = bpy.data.objects.new("Sun", bpy.data.lights.get("RL_sun") or bpy.data.lights.new("RL_sun", "SUN"))
    sun.data.energy = 4.0
    sun.data.angle = math.radians(3)
    sun.rotation_euler = (math.radians(40), 0.0, math.radians(30))
    col.objects.link(sun)
    return col


if __name__ == "__main__" and "HABITAT_JSON" in globals():
    build(HABITAT_JSON)  # noqa: F821
