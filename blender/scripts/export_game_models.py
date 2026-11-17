"""Export both creatures for the game (game/): one skinned glTF per creature in the rest pose.

Rigid parts that hang on a bone (claws, teeth, eyes, the Myrranox's armour pieces) are bound to that bone with
weight 1, so each creature is a single skinned mesh the game drives bone by bone from the physics bodies. Heavy
parts are decimated for real time. Writes game/assets/creatures/<species>.glb and <species>_bones.json (bone parents).

    blender -b blender/habitat_showcase.blend --python blender/scripts/export_game_models.py
"""
import json
import os

import bpy

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "game", "assets", "creatures")
RIGS = {"dc": "DC_rig", "myrranox": "Myrranox_rig"}
MAX_FACES = 2500          # per rigid part (teeth, claws, eyes: a few pixels in the game; the DC teeth have 131k faces)
MAX_BODY_FACES = 24000    # per skinned mesh (the DC's body: ~150k; display only - the physics uses its own hulls)


def decimate(obj, faces):
    n = len(obj.data.polygons)
    if n > faces:
        m = obj.modifiers.new("dec", "DECIMATE")
        m.ratio = faces / n
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=m.name)


def export(species, rig_name):
    rig = bpy.data.objects[rig_name]
    rig.animation_data_clear()
    rig.data.pose_position = "REST"
    for pb in rig.pose.bones:
        pb.location, pb.scale = (0, 0, 0), (1, 1, 1)
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
    rig.matrix_world.identity()
    bpy.context.view_layer.update()
    for o in bpy.context.selected_objects:
        o.select_set(False)
    parts = [o for o in bpy.data.objects if o.type == "MESH" and o.parent == rig and not o.hide_render]
    keep = []
    for o in parts:
        c = o.copy()
        c.data = o.data.copy()
        bpy.context.scene.collection.objects.link(c)
        mw = o.matrix_world.copy()
        c.parent = None
        c.matrix_world = mw
        c.animation_data_clear()
        if o.parent_type == "BONE":                    # rigid part: weight 1 on its bone
            for m in list(c.modifiers):
                c.modifiers.remove(m)
            decimate(c, MAX_FACES)
            c.vertex_groups.clear()
            g = c.vertex_groups.new(name=o.parent_bone)
            g.add(list(range(len(c.data.vertices))), 1.0, "REPLACE")
        else:                                          # skinned mesh: decimate, the vertex weights are kept
            for m in [m for m in c.modifiers if m.type == "ARMATURE"]:
                c.modifiers.remove(m)
            decimate(c, MAX_BODY_FACES)
        bpy.context.view_layer.objects.active = c
        c.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        c.select_set(False)
        for m in [m for m in c.modifiers if m.type == "ARMATURE"]:
            c.modifiers.remove(m)
        mod = c.modifiers.new("Armature", "ARMATURE")
        mod.object = rig
        c.parent = rig
        c.matrix_parent_inverse.identity()
        keep.append(c)
    for o in keep + [rig]:
        o.select_set(True)
    bpy.context.view_layer.objects.active = rig
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{species}.glb")
    bpy.ops.export_scene.gltf(filepath=path, use_selection=True, export_format="GLB", export_skins=True,
                              export_animations=False, export_apply=False, export_yup=True)
    bones = {b.name: (b.parent.name if b.parent else None) for b in rig.data.bones}
    with open(os.path.join(OUT, f"{species}_bones.json"), "w") as f:
        json.dump(bones, f, indent=1)
    for o in keep:
        print(f"  part {o.name}: {len(o.data.polygons)} faces")
    faces = sum(len(o.data.polygons) for o in keep)
    print(f"GAME {species}: {len(keep)} parts, {faces} faces, {len(bones)} bones -> {path}")
    for o in keep:
        bpy.data.objects.remove(o, do_unlink=True)


for sp, rig in RIGS.items():
    export(sp, rig)
