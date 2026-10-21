"""Render one camera of the showcase to a PNG sequence, resumable (existing frames are skipped).

Run with Blender in the background:
    blender -b blender/habitat_showcase.blend --python blender/scripts/render_camera.py -- \
        --camera SideCam_myrranox_0 --out renders/showcase/myrranox --samples 16 --size 960 540
Then combine: python make_side_by_side.py
"""
import argparse
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
p = argparse.ArgumentParser()
p.add_argument("--camera", required=True)
p.add_argument("--out", required=True)
p.add_argument("--samples", type=int, default=16)
p.add_argument("--size", type=int, nargs=2, default=(960, 540))
p.add_argument("--frames", type=int, nargs=2, default=None, help="start end (default: scene range)")
p.add_argument("--sharp-shadows", action="store_true",
               help="hard shadow edges: about 20 %% faster on the laptop's integrated GPU (21 Oct)")
args = p.parse_args(argv)

scene = bpy.context.scene
scene.camera = bpy.data.objects[args.camera]
for m in list(scene.timeline_markers):          # no marker may switch the camera
    m.camera = None
r = scene.render
r.resolution_x, r.resolution_y = args.size
r.resolution_percentage = 100
scene.eevee.taa_render_samples = args.samples
if args.sharp_shadows:
    for light in bpy.data.lights:
        if hasattr(light, "use_soft_shadows"):
            light.use_soft_shadows = False
r.image_settings.file_format = "PNG"
r.use_overwrite = False                          # resumable: skip frames that exist
r.use_placeholder = True                         # lets two processes share work safely
os.makedirs(args.out, exist_ok=True)
r.filepath = os.path.join(os.path.abspath(args.out), "frame_")
if args.frames:
    scene.frame_start, scene.frame_end = args.frames
print(f"rendering {args.camera}: frames {scene.frame_start}-{scene.frame_end} -> {args.out}", flush=True)
bpy.ops.render.render(animation=True)
print("RENDER_DONE", flush=True)
