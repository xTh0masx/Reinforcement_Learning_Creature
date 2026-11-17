"""Turn the Poly Haven downloads (game/fetch_assets.py) into light game models: game/assets/env/<asset>.glb.

Every model is imported on its own, its meshes are decimated to a face budget (alpha-card foliage is decimated less,
cards collapse badly) and it is exported as one GLB with its 1k textures, Z-up kept as Blender's (the game loads with
panda3d-gltf, which converts back to Z-up). Prints the size of each model (for placing it in the world).

    blender -b --factory-startup --python blender/scripts/prepare_game_env.py
"""
import json
import os

import bpy

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(REPO, "game", "assets", "polyhaven")
OUT = os.path.join(REPO, "game", "assets", "env")
BUDGET = {  # faces per model
    "quiver_tree_01": 14000, "quiver_tree_02": 12000, "dead_quiver_trunk": 5000, "wild_rooibos_bush": 6000,
    "namaqualand_boulder_02": 4000, "namaqualand_boulder_04": 4000, "namaqualand_boulder_05": 4000,
    "namaqualand_rocks_01": 4000, "jacaranda_tree": 20000, "island_tree_02": 24000, "shrub_02": 6000,
    "shrub_03": 5000, "shrub_04": 6000, "fern_02": 3000, "rock_moss_set_01": 6000, "tree_stump_01": 4000,
    "dead_tree_trunk": 5000, "grass_medium_01": 4000, "rock_face_01": 8000, "rock_face_02": 8000,
    "namaqualand_cliff_01": 10000, "street_rat": 3000,
}


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def uses_alpha(obj):
    """A material whose Principled BSDF alpha is driven by a texture (leaf cards)."""
    for slot in obj.material_slots:
        m = slot.material
        if m and m.node_tree and any(l.to_socket.name == "Alpha" and l.to_node.type == "BSDF_PRINCIPLED"
                                     for l in m.node_tree.links):
            return True
    return False


def main():
    os.makedirs(OUT, exist_ok=True)
    sizes = {}
    for name, budget in BUDGET.items():
        src = os.path.join(RAW, name, f"{name}.gltf")
        if not os.path.exists(src):
            print("missing", src)
            continue
        clear()
        bpy.ops.import_scene.gltf(filepath=src)
        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        total = sum(len(o.data.polygons) for o in meshes)
        ratio = min(1.0, budget / max(total, 1))
        for o in meshes:
            r = ratio
            if uses_alpha(o):
                r = min(1.0, ratio * 3)                 # foliage cards: keep more
            if r < 0.999:
                m = o.modifiers.new("dec", "DECIMATE")
                m.ratio = r
                m.use_collapse_triangulate = True
                bpy.context.view_layer.objects.active = o
                bpy.ops.object.modifier_apply(modifier=m.name)
        after = sum(len(o.data.polygons) for o in meshes)
        xs = [(o.matrix_world @ v.co) for o in meshes for v in o.data.vertices]
        lo = [min(p[i] for p in xs) for i in range(3)]
        hi = [max(p[i] for p in xs) for i in range(3)]
        sizes[name] = {"faces": after, "min": [round(v, 3) for v in lo], "max": [round(v, 3) for v in hi]}
        for o in bpy.context.scene.objects:
            o.select_set(o.type == "MESH" or o.type == "EMPTY")
        bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, f"{name}.glb"), export_format="GLB", use_selection=True,
                                  export_image_format="JPEG", export_jpeg_quality=88, export_yup=True)
        print(f"ENV {name}: {total} -> {after} faces, size {[round(h - l, 2) for l, h in zip(lo, hi)]}")
    with open(os.path.join(OUT, "sizes.json"), "w") as f:
        json.dump(sizes, f, indent=1)


main()
