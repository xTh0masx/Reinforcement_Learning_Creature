"""Rig the Myrranox for physics playback (PROJECT_PLAN.md section 9, asset prep).

- Places the model in the body frame used by sim/bodies.py (+X forward, feet on z = 0, 1.5 m long).
- Splits every leg mesh (left + right in one object) into _L and _R objects.
- Builds armature "Myrranox_rig": bone "torso" + 6 legs x 3 bones whose heads sit exactly on the
  MuJoCo hinge points, so a physics joint angle is a pure rotation of its bone.
- Parents each mesh part rigidly to its bone.

Run inside Blender:  rig(repo_root)
"""
import sys

import bpy

ASSET = "Myrranox_low_2_0"
RIG = "Myrranox_rig"
# mesh part per leg segment (coxa, femur, claw) for each leg pair in sim/bodies.py
# mesh parts that ride on the passive (spring) bones of stage 1
PASSIVE_PARTS = {
    "abdomen": ["Abdomen_low", "Fins_01_low", "Fins_02_low"],
    "head": ["Head_low", "Eyes_low", "Mandibles_low", "MandiblesStart_low", "MandiblesCon_low", "InnerMandible_low"],
}
LEG_PARTS = {
    "front": ("TopArms_low", "longTopArms_low", "topclaw_low"),
    "mid": ("MidArms_low", "longmidarms_low", "midclaw_low"),
    "back": ("LowerArms_low", "longbotarms_low", "Botclaw_low"),
}


def rig(repo_root):
    for p in (repo_root, f"{repo_root}/blender/scripts"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from rig_common import (add_passive_bones, attach_to_bone, body_frame_matrix, build_armature, split_left_right,
                            store_joint_metadata)
    from sim.bodies import myrranox
    spec = myrranox(passive=True, active=True)      # stage 2: head/abdomen actuated (same joints, same bones)

    existing = bpy.data.objects.get(RIG)
    if existing is not None:
        # Already rigged: rebuilding would orphan the parts, so only refresh the joint data (+ passive bones).
        store_joint_metadata(existing, spec)
        add_passive_bones(existing, spec, PASSIVE_PARTS)
        return existing

    asset = bpy.data.objects[ASSET]
    asset.matrix_world = body_frame_matrix(asset, axis=1, size=1.5)
    bpy.context.view_layer.update()

    for parts in LEG_PARTS.values():
        for name in parts:
            if name in bpy.data.objects:
                split_left_right(bpy.data.objects[name])
    bpy.context.view_layer.update()

    arm = build_armature(RIG, asset.users_collection[0], spec, (0, 0, 0), (0.25, 0, 0))

    leg_objs = set()
    for leg in spec.legs:
        pair, side = leg.name.split("_")
        for i, part in enumerate(LEG_PARTS[pair]):
            obj = bpy.data.objects.get(f"{part}_{side}")
            if obj is not None:
                attach_to_bone(obj, arm, f"{leg.name}_s{i}")
                leg_objs.add(obj.name)
    for k in list(asset.children_recursive):
        if k.type == "MESH" and k.name not in leg_objs:
            attach_to_bone(k, arm, "torso")
    add_passive_bones(arm, spec, PASSIVE_PARTS)
    bpy.context.view_layer.update()
    return arm
