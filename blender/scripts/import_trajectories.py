"""Play a recordings/*.csv in Blender (PROJECT_PLAN.md section 9, playback steps 1 and 6).

For every creature a root empty is keyed from the 10 Hz CSV (X/Y + heading); the follow camera
rides on it. If a physics pose file <csv stem>_<id>_pose.csv exists (record_physics.py) and the
species is rigged (rig_myrranox.py / rig_dc.py), the rig is animated from it: torso bone = physics
body pose, leg bones = physics joint angles, resampled from 50 Hz to 30 fps. Otherwise the unrigged
model glides on the root. Prey are moved and hidden while waiting to respawn.

Run inside Blender:
    CSV_PATH = r"C:/.../recordings/xyz.csv"; exec(open(r"C:/.../import_trajectories.py").read())
or call import_csv(path). Re-running replaces the "Creatures" collection.
"""
import csv
import json
import math
import os
from collections import defaultdict

import bpy
import numpy as np
from bpy_extras import anim_utils
from mathutils import Matrix, Quaternion, Vector

FPS = 30
COLLECTION = "Creatures"
PREY_KEY_EVERY = 5          # key prey every 5 sim steps (0.5 s); they move slowly

# Unrigged asset per species. Models face -Y in the .blend; "size" scales the chosen axis.
ASSETS = {
    "myrranox": {"object": "Myrranox_low_2_0", "axis": 1, "size": 1.5},   # 1.5 m body length
    "dc":       {"object": "DC_SKIN_RAW",      "axis": 2, "size": 1.3},   # overall height, ~1.0 m shoulder
}
RIGS = {"myrranox": "Myrranox_rig", "dc": "DC_rig"}
# Display-only head stabilization (owner, 30 Sep: "the head should not wobble at all"). The physics and the brains
# are unchanged; in the animation the head keeps its modelled (level) orientation and turns only with the body's
# smoothed heading, and the neck angles are smoothed without lag, so the head does not bob with each step.
HEAD_STABILIZE = {"DC_rig": {"bone": "head", "neck_joints": ("neck_p0x", "neck_p0y", "neck_p1x"),
                             "neck_sigma_s": 0.2, "yaw_sigma_s": 0.3}}


def _collection():
    col = bpy.data.collections.get(COLLECTION)
    if col is None:
        col = bpy.data.collections.new(COLLECTION)
        bpy.context.scene.collection.children.link(col)
    for obj in list(col.objects):
        if obj.animation_data and obj.animation_data.action:
            bpy.data.actions.remove(obj.animation_data.action)
        bpy.data.objects.remove(obj, do_unlink=True)
    return col


def _link(col, obj):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)
    return obj


def _fcurves(obj):
    ad = obj.animation_data_create()
    act = bpy.data.actions.new(f"{obj.name}_action")
    ad.action = act
    slot = act.slots.new(id_type="OBJECT", name=obj.name)
    ad.action_slot = slot
    return anim_utils.action_ensure_channelbag_for_slot(act, slot).fcurves


def _key(fcurves, data_path, index, frames, values, interpolation=2):
    """interpolation: 0 constant, 1 linear, 2 bezier."""
    fc = fcurves.new(data_path, index=index)
    kps = fc.keyframe_points
    kps.add(len(frames))
    kps.foreach_set("co", np.column_stack([frames, values]).ravel().astype(np.float32))
    kps.foreach_set("interpolation", [interpolation] * len(frames))
    fc.update()


def _read(path):
    tracks = defaultdict(lambda: defaultdict(list))
    species = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tr = tracks[row["id"]]
            species[row["id"]] = row["species"]
            tr["time"].append(float(row["time"]))
            tr["x"].append(float(row["x"]))
            tr["y"].append(float(row["y"]))
            tr["heading"].append(float(row["heading"]))
            tr["activity"].append(row["activity"])
            for k in ("hunger", "fatigue"):                  # record_world.py recordings carry the needs
                if row.get(k):
                    tr[k].append(float(row[k]))
    return tracks, species


def _attach_asset(root, species_name):
    """Parent the unrigged model to `root`: centred, feet on z = 0, facing +X, scaled to real size."""
    cfg = ASSETS.get(species_name)
    asset = bpy.data.objects.get(cfg["object"]) if cfg else None
    if asset is None:
        return None
    # Remember the untouched transform once, so re-runs and undo-by-hand start from the original.
    if "rl_orig_matrix" not in asset:
        asset["rl_orig_matrix"] = [v for row in asset.matrix_world for v in row]
    orig = Matrix([asset["rl_orig_matrix"][i:i + 4] for i in range(0, 16, 4)])

    asset.parent = None
    asset.matrix_world = orig
    bpy.context.view_layer.update()
    pts = [k.matrix_world @ Vector(c) for k in asset.children_recursive if k.type == "MESH" for c in k.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    s = cfg["size"] / (hi[cfg["axis"]] - lo[cfg["axis"]])
    base = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))

    asset.parent = root
    asset.matrix_parent_inverse = Matrix.Identity(4)
    asset.matrix_basis = (Matrix.Rotation(math.pi / 2, 4, "Z")      # model -Y -> root +X
                          @ Matrix.Scale(s, 4) @ Matrix.Translation(-base) @ orig)
    return asset


def _read_pose(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], np.array(rows[1:], dtype=float)
    return {name: data[:, i] for i, name in enumerate(header)}


def _gauss(values, sigma_s, dt):
    """Zero-phase Gaussian smoothing (no lag), edges reflected."""
    n = max(int(round(3 * sigma_s / dt)), 1)
    k = np.exp(-0.5 * (np.arange(-n, n + 1) * dt / sigma_s) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(values, n, mode="reflect"), k, mode="valid")


def _stabilize_head(col, arm, cfg, t, frames, qi):
    """Hold the head bone at its rest orientation, turned only by the smoothed body heading (Copy Rotation from a
    keyed empty, world space)."""
    yaw = np.unwrap(np.arctan2(2 * (qi[:, 0] * qi[:, 3] + qi[:, 1] * qi[:, 2]), 1 - 2 * (qi[:, 2] ** 2 + qi[:, 3] ** 2)))
    yaw = _gauss(yaw, cfg["yaw_sigma_s"], 1.0 / FPS)
    rest = arm.matrix_world.to_3x3() @ arm.data.bones[cfg["bone"]].matrix_local.to_3x3()
    aim = _link(col, bpy.data.objects.new(f"HeadAim_{arm.name}", None))
    aim.empty_display_size = 0.1
    aim.hide_render = True
    aim.rotation_mode = "QUATERNION"
    rots, prev = [], None
    for a in yaw:
        rq = (Matrix.Rotation(float(a), 3, "Z") @ rest).to_quaternion()
        if prev is not None and rq.dot(prev) < 0:
            rq.negate()
        prev = rq
        rots.append(tuple(rq))
    rots = np.array(rots)
    fcs = _fcurves(aim)
    for i in range(4):
        _key(fcs, "rotation_quaternion", i, frames, rots[:, i], 1)
    pb = arm.pose.bones[cfg["bone"]]
    for c in [c for c in pb.constraints if c.name == "RL_head_stable"]:
        pb.constraints.remove(c)
    c = pb.constraints.new("COPY_ROTATION")
    c.name, c.target = "RL_head_stable", aim
    c.owner_space = c.target_space = "WORLD"


def _animate_rig(arm, pose, last_time, col=None):
    """Key the rig from a 50 Hz physics pose table, resampled to FPS."""
    t = np.arange(0.0, min(last_time, pose["time"][-1]) + 1e-9, 1.0 / FPS)
    frames = 1 + t * FPS
    stab = HEAD_STABILIZE.get(arm.name) if col is not None else None
    dt_pose = float(pose["time"][1] - pose["time"][0])

    def lerp(name):
        values = pose[name]
        if stab and name in stab["neck_joints"]:
            values = _gauss(values, stab["neck_sigma_s"], dt_pose)
        return np.interp(t, pose["time"], values)

    # Quaternion: nlerp between neighbouring samples (dense enough at 50 Hz), keep sign continuous.
    q = np.column_stack([pose[k] for k in ("qw", "qx", "qy", "qz")])
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] *= -1
    qi =np.column_stack([np.interp(t, pose["time"], q[:, i]) for i in range(4)])
    qi /= np.linalg.norm(qi, axis=1, keepdims=True)
    pos = np.column_stack([lerp(k) for k in ("x", "y", "z")])

    if arm.animation_data and arm.animation_data.action:
        bpy.data.actions.remove(arm.animation_data.action)
    arm.animation_data_clear()
    fcs = _fcurves(arm)
    bones = arm.data.bones

    # Torso: pose = B^-1 @ T @ B, T = physics body transform relative to the rest pose at the origin.
    B = bones["torso"].matrix_local
    Binv = B.inverted()
    locs, rots, prev = [], [], None
    for p, qq in zip(pos, qi):
        T = Matrix.Translation(Vector(p)) @ Quaternion(qq).to_matrix().to_4x4()
        m = Binv @ T @ B
        rq = m.to_quaternion()
        if prev is not None and rq.dot(prev) < 0:
            rq.negate()
        prev = rq
        locs.append(m.to_translation())
        rots.append(rq)
    locs, rots = np.array(locs), np.array([tuple(r) for r in rots])
    for i in range(3):
        _key(fcs, 'pose.bones["torso"].location', i, frames, locs[:, i], 1)
    for i in range(4):
        _key(fcs, 'pose.bones["torso"].rotation_quaternion', i, frames, rots[:, i], 1)

    # Legs: each bone rotates by the product of its hinge rotations (body-frame axes, MuJoCo order).
    for bone in bones:
        if "rl_joints" not in bone:
            continue
        joints = json.loads(bone["rl_joints"])
        if not joints or any(j[0] not in pose for j in joints):
            continue
        R0 = bone.matrix_local.to_3x3()
        R0inv = R0.inverted()
        angles = [lerp(j[0]) for j in joints]
        out, prev = [], None
        for f in range(len(t)):
            R = Matrix.Identity(3)
            for (name, ax, ay, az), ang in zip(joints, angles):
                R = R @ Quaternion(Vector((ax, ay, az)), ang[f]).to_matrix()
            rq = (R0inv @ R @ R0).to_quaternion()
            if prev is not None and rq.dot(prev) < 0:
                rq.negate()
            prev = rq
            out.append(tuple(rq))
        out = np.array(out)
        for i in range(4):
            _key(fcs, f'pose.bones["{bone.name}"].rotation_quaternion', i, frames, out[:, i], 1)
    for pb in arm.pose.bones:                       # constraints from an earlier import
        for c in [c for c in pb.constraints if c.name == "RL_head_stable"]:
            pb.constraints.remove(c)
    if stab:
        _stabilize_head(col, arm, stab, t, frames, qi)
    return int(frames[-1])


def _placeholder(col, name, size):
    bpy.ops.mesh.primitive_cone_add(vertices=3, radius=size * 0.4, depth=size, location=(0, 0, 0),
                                    rotation=(0, math.pi / 2, 0))
    obj = bpy.context.active_object
    obj.name = name
    return _link(col, obj)


HUD_BARS = (("hunger", (0.85, 0.15, 0.12, 1)), ("fatigue", (0.20, 0.45, 0.90, 1)))


def _flat_material(name, color):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = color
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], out.inputs[0])
    mat.diffuse_color = color
    return mat


def _bar_mesh():
    """Unit bar from x=0 to x=1 so that scaling x fills it from the left."""
    me = bpy.data.meshes.get("RL_hud_bar")
    if me is None:
        me = bpy.data.meshes.new("RL_hud_bar")
        me.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2, 3)])
    return me


def _hud(col, cam, cid, species_name, frames, tr):
    """Name + hunger/fatigue bars in front of the camera (top left of the image), keyed from the recording."""
    # Place it in the top-left corner of the image for this camera's lens (16:9 image, sensor width 36 mm)
    d = 1.0                                              # distance in front of the camera (m)
    half_w = 0.5 * cam.data.sensor_width / cam.data.lens * d
    half_h = half_w * 9 / 16
    w, h = 0.55 * half_w, 0.04 * half_h                  # bar length and height
    x0, y0 = -0.95 * half_w, 0.62 * half_h
    white = _flat_material("RL_hud_text", (0.95, 0.95, 0.95, 1))
    grey = _flat_material("RL_hud_back", (0.15, 0.15, 0.15, 1))

    def child(obj, loc, scale=(1, 1, 1)):
        col.objects.link(obj)
        obj.parent = cam
        obj.location = (loc[0], loc[1], -d)
        obj.scale = scale
        return obj

    title = bpy.data.curves.new(f"HUD_{cid}_title", "FONT")
    title.body = f"{species_name}  ({cid})"
    title.size = 0.065 * half_h
    t = child(bpy.data.objects.new(f"HUD_{cid}_title", title), (x0, y0 + 0.15 * half_h))
    t.data.materials.append(white)
    for i, (need, color) in enumerate(HUD_BARS):
        y = y0 - i * 0.14 * half_h
        label = bpy.data.curves.new(f"HUD_{cid}_{need}_label", "FONT")
        label.body = need
        label.size = 0.045 * half_h
        lo = child(bpy.data.objects.new(f"HUD_{cid}_{need}_label", label), (x0, y))
        lo.data.materials.append(white)
        back = child(bpy.data.objects.new(f"HUD_{cid}_{need}_back", _bar_mesh().copy()), (x0 + 0.3 * half_w, y), (w, h, 1))
        back.data.materials.append(grey)
        bar = child(bpy.data.objects.new(f"HUD_{cid}_{need}", _bar_mesh().copy()), (x0 + 0.3 * half_w, y), (w, h, 1))
        bar.location.z = -d + 0.001
        bar.data.materials.append(_flat_material(f"RL_hud_{need}", color))
        if tr.get(need):
            vals = np.clip(np.asarray(tr[need]), 0.001, 1.0)
            _key(_fcurves(bar), "scale", 0, frames[:len(vals)], w * vals, 1)


# Side camera per species: distance to the side, height, lens (mm), body-centre height for the focus point
SIDE_VIEW = {"myrranox": (3.2, 0.6, 35, 0.2), "dc": (5.5, 1.2, 35, 0.7)}
POS_SMOOTH_S = 0.7          # camera position follows the creature with this delay
HEADING_SMOOTH_S = 3.0      # camera direction follows the creature's heading only slowly -> steady side view


def _smooth(values, tau_s, dt):
    """Zero-phase exponential smoothing (forward + backward), no lag in the result."""
    a = dt / (tau_s + dt)
    v = np.asarray(values, dtype=float).copy()
    for rng in (range(1, len(v)), range(len(v) - 2, -1, -1)):
        prev = v[rng[0] - 1] if rng.step == 1 else v[rng[0] + 1]
        for i in rng:
            v[i] = prev = prev + a * (v[i] - prev)
    return v


def _occlusion_offsets(t, rx, ry, hs, dist, obstacles, bounds, hold_s=3.0, turn_s=1.5, look_s=1.5):
    """Extra camera orbit angle per sample: 0 = creature's left side, pi = right side. The camera changes side when a
    rock (or the wall) would block the view within the next look_s seconds and the other side is free; it orbits
    around smoothly over turn_s and stays at least hold_s on a side."""
    n = len(t)
    dt = float(t[1] - t[0]) if n > 1 else 0.1

    def blocked(side):
        cx, cy = rx - side * dist * np.sin(hs), ry + side * dist * np.cos(hs)
        out = (cx < bounds[0] + 0.5) | (cx > bounds[1] - 0.5) | (cy < bounds[2] + 0.5) | (cy > bounds[3] - 0.5)
        dx, dy = rx - cx, ry - cy
        L2 = np.maximum(dx * dx + dy * dy, 1e-9)
        for ox, oy, r in obstacles:
            u = np.clip(((ox - cx) * dx + (oy - cy) * dy) / L2, 0, 1)
            out |= np.hypot(cx + u * dx - ox, cy + u * dy - oy) < r + 0.3
        look = max(int(look_s / dt), 1)
        ahead = np.array([out[i:i + look].any() for i in range(n)])
        return ahead

    b = {1: blocked(1), -1: blocked(-1)}
    side, last, target = 1, -1e9, np.zeros(n)
    switches = 0
    for i in range(n):
        if b[side][i] and not b[-side][i] and t[i] - last >= hold_s:
            side, last = -side, t[i]
            switches += 1
        target[i] = 0.0 if side == 1 else np.pi
    # smooth orbit: rate-limit the angle change to pi per turn_s
    off, rate = np.zeros(n), np.pi / turn_s * dt
    for i in range(1, n):
        off[i] = off[i - 1] + np.clip(target[i] - off[i - 1], -rate, rate)
    return off, switches


def _side_camera(col, cid, species_name, frames, t, x, y, heading, obstacles=(), bounds=(-1e9, 1e9, -1e9, 1e9)):
    """Steady camera beside the creature, following position and heading smoothly; it orbits to the other side when a
    rock or the wall would block the view. Also a focus empty on the body for the "follow + free orbit" mode."""
    dist, height, lens, body_z = SIDE_VIEW.get(species_name, (5.0, 1.0, 35, 0.5))
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.1

    focus = _link(col, bpy.data.objects.new(f"Focus_{cid}", None))
    focus.empty_display_type = "SPHERE"
    focus.empty_display_size = 0.1
    focus["rl_view_distance"] = dist * 1.4
    fx, fy = _smooth(x, 0.3, dt), _smooth(y, 0.3, dt)
    fcs = _fcurves(focus)
    _key(fcs, "location", 0, frames, fx)
    _key(fcs, "location", 1, frames, fy)
    _key(fcs, "location", 2, frames, np.full(len(t), body_z))

    rig = _link(col, bpy.data.objects.new(f"SideRig_{cid}", None))
    rig.rotation_mode = "XYZ"
    fcs = _fcurves(rig)
    rx, ry = _smooth(x, POS_SMOOTH_S, dt), _smooth(y, POS_SMOOTH_S, dt)
    hs = _smooth(np.unwrap(np.asarray(heading)), HEADING_SMOOTH_S, dt)
    off, switches = _occlusion_offsets(np.asarray(t), rx, ry, hs, dist, obstacles, bounds)
    print(f"  {cid}: side camera changes side {switches} times to keep rocks out of view")
    _key(fcs, "location", 0, frames, rx)
    _key(fcs, "location", 1, frames, ry)
    _key(fcs, "location", 2, frames, np.zeros(len(t)))
    _key(fcs, "rotation_euler", 2, frames, hs + off)

    cam = _link(col, bpy.data.objects.new(f"SideCam_{cid}", bpy.data.cameras.new(f"SideCam_{cid}")))
    cam.data.lens = lens
    cam.parent = rig
    cam.location = (0.0, dist, height)                                   # on the creature's left side
    cam.rotation_euler = (math.radians(90), 0.0, math.radians(180))      # looking along -Y, Z up
    tr = cam.constraints.new("DAMPED_TRACK")
    tr.target, tr.track_axis = focus, "TRACK_NEGATIVE_Z"
    return cam


def _overview_camera(col, habitat_size=100.0):
    cam = _link(col, bpy.data.objects.new("Overview_cam", bpy.data.cameras.new("Overview_cam")))
    cam.data.lens = 28
    cam.location = (0.0, -0.85 * habitat_size, 0.75 * habitat_size)
    cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    return cam


def _install_view_panel(csv_path):
    """Store blender/scripts/view_panel.py in the .blend as a self-registering text block and register it now.
    The script is found relative to the recording (<repo>/recordings/x.csv -> <repo>/blender/scripts)."""
    src = os.path.join(os.path.dirname(os.path.dirname(csv_path)), "blender", "scripts", "view_panel.py")
    if not os.path.exists(src):
        print(f"view panel not installed: {src} not found")
        return
    code = open(src, encoding="utf-8").read()
    text = bpy.data.texts.get("rl_view_panel.py") or bpy.data.texts.new("rl_view_panel.py")
    text.clear()
    text.write(code)
    text.use_module = True                    # runs automatically when the .blend is opened (if scripts are allowed)
    exec(compile(code, "rl_view_panel.py", "exec"), {"__name__": "__main__"})


def import_csv(path, habitat_path=None):
    path = os.path.abspath(path)
    scene = bpy.context.scene
    scene.render.fps = FPS
    col = _collection()
    tracks, species = _read(path)

    prey_mesh = bpy.data.meshes.get("RL_prey")
    if prey_mesh is None:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=0.25)
        tmp = bpy.context.active_object
        prey_mesh = tmp.data
        prey_mesh.name = "RL_prey"
        bpy.data.objects.remove(tmp)
        mat = bpy.data.materials.get("RL_prey") or bpy.data.materials.new("RL_prey")
        mat.use_nodes = True
        bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = (0.75, 0.15, 0.12, 1)
        mat.diffuse_color = (0.75, 0.15, 0.12, 1)
        prey_mesh.materials.append(mat)

    # rocks and walls for the side cameras' occlusion check (<repo>/recordings/x.csv -> <repo>/sim/habitat.json)
    import json
    hab_path = habitat_path or os.path.join(os.path.dirname(os.path.dirname(path)), "sim", "habitat.json")
    obstacles, bounds = [], (-1e9, 1e9, -1e9, 1e9)
    if os.path.exists(hab_path):
        hab = json.load(open(hab_path, encoding="utf-8"))
        obstacles = [(o["center"][0], o["center"][1], o["radius"]) for o in hab["obstacles"]]
        b = hab["boundary"]
        bounds = (b["xmin"], b["xmax"], b["ymin"], b["ymax"])

    last_frame, cameras = 1, []
    for cid, tr in tracks.items():
        t = np.asarray(tr["time"])
        frames = 1 + t * FPS
        last_frame = max(last_frame, int(frames[-1]))
        x, y = np.asarray(tr["x"]), np.asarray(tr["y"])

        if species[cid] == "prey":
            obj = _link(col, bpy.data.objects.new(cid, prey_mesh))
            idx = np.arange(0, len(t), PREY_KEY_EVERY)
            fcs = _fcurves(obj)
            _key(fcs, "location", 0, frames[idx], x[idx], 1)
            _key(fcs, "location", 1, frames[idx], y[idx], 1)
            _key(fcs, "location", 2, frames[idx], np.full(len(idx), 0.25), 1)
            vis = np.array([0.0 if a == "hidden" else 1.0 for a in tr["activity"]])
            change = np.flatnonzero(np.r_[True, vis[1:] != vis[:-1]])
            for i in range(3):
                _key(fcs, "scale", i, frames[change], vis[change], 0)
            continue

        root = _link(col, bpy.data.objects.new(f"{cid}_root", None))
        root.empty_display_type = "SINGLE_ARROW"
        root.rotation_mode = "XYZ"
        fcs = _fcurves(root)
        _key(fcs, "location", 0, frames, x)
        _key(fcs, "location", 1, frames, y)
        _key(fcs, "location", 2, frames, np.zeros(len(t)))
        _key(fcs, "rotation_euler", 2, frames, np.unwrap(np.asarray(tr["heading"])))

        # First individual of a species gets the real model, further ones a placeholder.
        model = None
        rig = bpy.data.objects.get(RIGS.get(species[cid], ""))
        pose_path = f"{os.path.splitext(path)[0]}_{cid}_pose.csv"
        if cid.endswith("_0") and rig is not None and os.path.exists(pose_path):
            _animate_rig(rig, _read_pose(pose_path), float(t[-1]), col)
            model = rig
        elif cid.endswith("_0") and rig is None:
            model = _attach_asset(root, species[cid])
        if model is None:
            _placeholder(col, f"{cid}_body", ASSETS.get(species[cid], {}).get("size", 1.0)).parent = root
        cameras.append(_side_camera(col, cid, species[cid], frames, t, x, y, tr["heading"], obstacles, bounds))
        if tr.get("hunger"):
            _hud(col, cameras[-1], cid, species[cid], frames, tr)

    scene.frame_start, scene.frame_end = 1, last_frame
    # No automatic camera switching: the viewer chooses (panel "Creatures" in the 3D view sidebar, N key).
    for m in [m for m in scene.timeline_markers if m.name.startswith("cam_")]:
        scene.timeline_markers.remove(m)
    _overview_camera(col)
    if cameras:
        scene.camera = cameras[0]
    _install_view_panel(path)
    scene.frame_set(1)
    return col


if __name__ == "__main__" and "CSV_PATH" in globals():
    import_csv(CSV_PATH)  # noqa: F821
