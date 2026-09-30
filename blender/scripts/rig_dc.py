"""Rig DC_SKIN_RAW for physics playback (stand-in until the planned retopology).

- Places the model in the body frame used by sim/bodies.py (+X forward, feet on z = 0, 1.3 m tall).
- Makes a decimated copy of the 970k-face sculpt ("DC_body_proxy", ~40k faces) and hides the original.
- Builds armature "DC_rig": torso + 4 legs x 3 bones on the MuJoCo joint points, plus static
  tail / neck / head / arm bones, and binds the proxy with automatic weights.
- Splits the claw meshes into _L/_R and parents claws, teeth, tongue and eyes rigidly to bones.

Run inside Blender:  rig(repo_root)
"""
import sys

import bpy

ASSET = "DC_SKIN_RAW"
RIG = "DC_rig"
BODY = "PM3D_MainBody1"
PROXY = "DC_body_proxy"
PROXY_FACES = 40_000
CLAWS = {  # mesh -> bone (per side) ; None side = torso
    "BackClaw_Front_01": "hind_{}_s2", "BackClaw_Front_02": "hind_{}_s2", "BackClaw_Front_03": "hind_{}_s2",
    "MidClaw_Back": "front_{}_s2", "MidClaw_Front_01": "front_{}_s2", "MidClaw_Front_02": "front_{}_s2",
    "FrontClaws": "arm_{}_s2",
}
HEAD_PARTS = ("UpperTeeth", "LowerTeeth", "Tounge", "Eyes_01", "Eyes_02")
EXTRA_BONES = [
    ("neck", (0.50, 0, 0.80), (1.00, 0, 0.90), "torso"),
    ("head", (1.00, 0, 0.90), (1.32, 0, 0.80), "neck"),
    ("tail_1", (-0.30, 0, 0.60), (-0.80, 0, 0.50), "torso"),
    ("tail_2", (-0.80, 0, 0.50), (-1.35, 0, 0.42), "tail_1"),
    ("arm_L", (0.62, 0.20, 0.62), (0.98, 0.22, 0.24), "torso"),
    ("arm_R", (0.62, -0.20, 0.62), (0.98, -0.22, 0.24), "torso"),
]


def _decimated_proxy(body):
    proxy = bpy.data.objects.get(PROXY)
    if proxy is not None:
        return proxy
    proxy = body.copy()
    proxy.data = body.data.copy()
    proxy.name = proxy.data.name = PROXY
    for c in body.users_collection:
        c.objects.link(proxy)
    mod = proxy.modifiers.new("decimate", "DECIMATE")
    mod.ratio = min(1.0, PROXY_FACES / max(len(body.data.polygons), 1))
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = proxy
    proxy.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    body.hide_viewport = body.hide_render = True
    return proxy


def rig(repo_root):
    for p in (repo_root, f"{repo_root}/blender/scripts"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from rig_common import (add_passive_bones, add_spine, body_frame_matrix, build_armature, split_left_right,
                            store_joint_metadata)
    from sim.bodies import dc
    spec = dc(passive=True, active=True, arms=True)  # walking v6: spine, neck/head and the arms as limbs

    existing = bpy.data.objects.get(RIG)
    if existing is not None:
        add_arm_limbs(existing, spec, bpy.data.objects[PROXY])
        store_joint_metadata(existing, spec)
        add_passive_bones(existing, spec)          # tail/neck/head/arm bones exist already; adds their joints
        if add_spine(existing, spec):              # new spine bone -> re-weight the proxy so the back can bend
            bind(bpy.data.objects[PROXY], existing)
        return existing

    asset = bpy.data.objects[ASSET]
    asset.matrix_world = body_frame_matrix(asset, axis=2, size=1.3)
    bpy.context.view_layer.update()

    proxy = _decimated_proxy(bpy.data.objects[BODY])
    for name in CLAWS:
        if name in bpy.data.objects:
            split_left_right(bpy.data.objects[name])
    bpy.context.view_layer.update()

    arm = build_armature(RIG, asset.users_collection[0], spec, (-0.25, 0, 0.66), (0.50, 0, 0.74), EXTRA_BONES)

    # Automatic (heat) weights on the proxy. Unparent first (keeping the world placement) - otherwise
    # parent_set drops the transform inherited from the asset empty and weights the mesh in the wrong place.
    mw = proxy.matrix_world.copy()
    proxy.parent = None
    proxy.matrix_world = mw
    add_spine(arm, spec)
    bind(proxy, arm)
    _attach_parts(arm)
    add_passive_bones(arm, spec)
    return arm


def add_arm_limbs(arm, spec, proxy):
    """Walking v6: replace the single passive arm bone per side by three bones on the physics arm points
    (arm_<side>_s0..s2 = upper arm, forearm, hand), split the old arm weights along the new bones (smooth blend by
    distance to each segment), move the claws to the hand bone and switch on volume-preserving skinning.
    Returns True if the rig was changed."""
    import numpy as np
    from mathutils import Vector
    limbs = [leg for leg in spec.legs if leg.name.startswith("arm")]
    if not limbs or all(f"{leg.name}_s0" in arm.data.bones for leg in limbs):
        return False
    pose_position = arm.data.pose_position
    arm.data.pose_position = "REST"
    bpy.context.view_layer.update()
    # remember the claws' placement before their bone disappears (they would jump to the armature origin)
    claws = {leg.name: bpy.data.objects.get(f"FrontClaws_{leg.name[-1]}") for leg in limbs}
    claw_mw = {k: o.matrix_world.copy() for k, o in claws.items() if o is not None}
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    for leg in limbs:
        parent = eb["torso"]
        for i in range(3):
            b = eb.new(f"{leg.name}_s{i}")
            b.head, b.tail = Vector(leg.points[i]), Vector(leg.points[i + 1])
            b.parent = parent
            parent = b
        if leg.name in eb:                              # the old single passive bone
            eb.remove(eb[leg.name])
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"

    M = proxy.matrix_world
    for leg in limbs:
        old = proxy.vertex_groups.get(leg.name)
        new = [proxy.vertex_groups.get(f"{leg.name}_s{i}") or proxy.vertex_groups.new(name=f"{leg.name}_s{i}")
               for i in range(3)]
        if old is None:
            continue
        a, b = np.array(leg.points[:3]), np.array(leg.points[1:])
        for v in proxy.data.vertices:
            w = next((g.weight for g in v.groups if g.group == old.index), 0.0)
            if w <= 0:
                continue
            p = np.array(M @ v.co)
            t = np.clip(np.einsum("ij,ij->i", p - a, b - a) / np.einsum("ij,ij->i", b - a, b - a), 0, 1)
            dist = np.linalg.norm(p - (a + t[:, None] * (b - a)), axis=1)
            share = np.exp(-((dist - dist.min()) / 0.04) ** 2)
            share /= share.sum()
            for g, s in zip(new, share):
                if s > 1e-3:
                    g.add([v.index], float(w * s), "ADD")
        proxy.vertex_groups.remove(old)
        if leg.name in claw_mw:
            o = claws[leg.name]
            o.parent, o.parent_type, o.parent_bone = arm, "BONE", f"{leg.name}_s2"
            bpy.context.view_layer.update()
            o.matrix_world = claw_mw[leg.name]
    for m in proxy.modifiers:
        if m.type == "ARMATURE":
            m.use_deform_preserve_volume = True       # no shrinking / rubber look where limbs bend
    arm.data.pose_position = pose_position
    return True


def bind(proxy, arm):
    proxy.vertex_groups.clear()
    for m in [m for m in proxy.modifiers if m.type == "ARMATURE"]:
        proxy.modifiers.remove(m)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    proxy.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    pose_position = arm.data.pose_position
    arm.data.pose_position = "REST"          # weight against the rest pose, whatever frame is shown
    bpy.ops.object.parent_set(type="ARMATURE_AUTO", keep_transform=True)
    arm.data.pose_position = pose_position


def _attach_parts(arm):
    from rig_common import attach_to_bone
    for name, bone in CLAWS.items():
        for side in "LR":
            obj = bpy.data.objects.get(f"{name}_{side}")
            if obj is not None:
                attach_to_bone(obj, arm, bone.format(side))
    for name in HEAD_PARTS:
        if name in bpy.data.objects:
            attach_to_bone(bpy.data.objects[name], arm, "head")
    bpy.context.view_layer.update()
    return arm
