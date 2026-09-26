"""Helpers shared by rig_myrranox.py and rig_dc.py (run inside Blender)."""
import json
import math

import bmesh
import bpy
from mathutils import Matrix, Vector


def body_frame_matrix(asset, axis, size):
    """World matrix that puts `asset` in the sim/bodies.py body frame: centred, feet on z = 0, facing +X,
    and scaled so its extent along `axis` (0/1/2, before rotation) equals `size` metres."""
    if "rl_orig_matrix" not in asset:
        asset["rl_orig_matrix"] = [v for row in asset.matrix_world for v in row]
    orig = Matrix([asset["rl_orig_matrix"][i:i + 4] for i in range(0, 16, 4)])
    asset.parent = None
    asset.matrix_world = orig
    bpy.context.view_layer.update()
    pts = [k.matrix_world @ Vector(c) for k in asset.children_recursive if k.type == "MESH" for c in k.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    s = size / (hi[axis] - lo[axis])
    base = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    return Matrix.Rotation(math.pi / 2, 4, "Z") @ Matrix.Scale(s, 4) @ Matrix.Translation(-base) @ orig


def split_left_right(obj):
    """Split a mesh holding both sides into obj_L (world +Y) and obj_R (world -Y). Returns (L, R)."""
    if obj.name.endswith(("_L", "_R")):
        return None
    right = obj.copy()
    right.data = obj.data.copy()
    for c in obj.users_collection:
        c.objects.link(right)
    base = obj.name
    for o, keep_left in ((obj, True), (right, False)):
        bm = bmesh.new()
        bm.from_mesh(o.data)
        mw = o.matrix_world
        doomed = [v for v in bm.verts if ((mw @ v.co).y > 0) != keep_left]
        bmesh.ops.delete(bm, geom=doomed, context="VERTS")
        bm.to_mesh(o.data)
        bm.free()
        o.data.update()
    obj.name, right.name = f"{base}_L", f"{base}_R"
    obj.data.name, right.data.name = obj.name, right.name
    return obj, right


def build_armature(name, collection, spec, torso_head, torso_tail, extra_bones=()):
    """Armature with bone "torso" and one bone per leg segment (<leg>_s0..s2) on the physics joint points.
    extra_bones: (name, head, tail, parent) for bones that are not driven by physics (tail, head, ...)."""
    arm = bpy.data.objects.new(name, bpy.data.armatures.new(name))
    collection.objects.link(arm)
    arm.show_in_front = True
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    torso = eb.new("torso")
    torso.head, torso.tail = torso_head, torso_tail
    for leg in spec.legs:
        parent = torso
        for i in range(3):
            b = eb.new(f"{leg.name}_s{i}")
            b.head, b.tail = Vector(leg.points[i]), Vector(leg.points[i + 1])
            b.parent = parent
            parent = b
    for bname, head, tail, parent in extra_bones:
        b = eb.new(bname)
        b.head, b.tail = Vector(head), Vector(tail)
        b.parent = eb[parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    store_joint_metadata(arm, spec)
    return arm


def store_joint_metadata(arm, spec):
    """Each leg bone gets rl_joints = JSON [[joint name, ax, ay, az], ...]: hinge axes in the body frame,
    in the order MuJoCo applies them (import_trajectories.py turns them into bone rotations)."""
    for leg in spec.legs:
        names = leg.joint_names()
        for i in range(3):
            arm.data.bones[f"{leg.name}_s{i}"]["rl_joints"] = json.dumps(
                [[names[k], *map(float, axis)] for k, (seg, axis, _) in enumerate(leg.joints) if seg == i])
    arm["rl_species"] = spec.name


def attach_to_bone(obj, arm, bone):
    mw = obj.matrix_world.copy()
    obj.parent = arm
    obj.parent_type = "BONE"
    obj.parent_bone = bone
    obj.matrix_world = mw


def add_passive_bones(arm, spec, parts=None):
    """Stage 1 "alive" bodies: make sure every passive physics segment (sim/bodies.py PassiveChain) has a bone on its
    joint points, store its spring joints as rl_joints (played back like the leg joints) and parent the given mesh
    parts to it. parts: {bone name: [mesh object names]}. Existing bones (DC tail/neck/head/arms) are reused."""
    missing = [(ch, i) for ch in spec.passive for i in range(len(ch.joints)) if ch.bones[i] not in arm.data.bones]
    if missing:
        for o in bpy.context.selected_objects:
            o.select_set(False)
        bpy.context.view_layer.objects.active = arm
        arm.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        eb = arm.data.edit_bones
        for ch, i in missing:
            b = eb.new(ch.bones[i])
            b.head, b.tail = Vector(ch.points[i]), Vector(ch.points[i + 1])
            b.parent = eb[ch.bones[i - 1]] if i else eb["torso"]
        bpy.ops.object.mode_set(mode="OBJECT")
        for pb in arm.pose.bones:
            pb.rotation_mode = "QUATERNION"
    for ch in spec.passive:
        names = ch.joint_names()
        k = 0
        for i, seg in enumerate(ch.joints):
            arm.data.bones[ch.bones[i]]["rl_joints"] = json.dumps(
                [[names[k + j], *map(float, axis)] for j, (axis, _, _) in enumerate(seg)])
            k += len(seg)
    for bone, objs in (parts or {}).items():
        for n in objs:
            if n in bpy.data.objects:
                attach_to_bone(bpy.data.objects[n], arm, bone)
    return arm


def add_spine(arm, spec):
    """Stage 2 DC spine: bone on the physics spine joint (RearSegment) carrying the hind legs and the tail, with its
    joints stored as rl_joints. Returns True if the bone was created (the mesh must then be re-weighted)."""
    r = spec.rear
    if r is None:
        return False
    created = r.bone not in arm.data.bones
    if created:
        for o in bpy.context.selected_objects:
            o.select_set(False)
        bpy.context.view_layer.objects.active = arm
        arm.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        eb = arm.data.edit_bones
        b = eb.new(r.bone)
        b.head = Vector(r.point)
        a, c, _, _ = r.geoms[0]
        b.tail = Vector(r.point) + Vector(c)
        b.parent = eb["torso"]
        for leg in r.legs:
            eb[f"{leg}_s0"].parent = b
        for ch in spec.passive:
            if ch.parent == r.name:
                eb[ch.bones[0]].parent = b
        bpy.ops.object.mode_set(mode="OBJECT")
        for pb in arm.pose.bones:
            pb.rotation_mode = "QUATERNION"
    arm.data.bones[r.bone]["rl_joints"] = json.dumps(
        [[n, *map(float, ax)] for n, ax in zip(r.joint_names(), r.joint_axes())])
    return created
