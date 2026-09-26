"""Sidebar panel "Creatures" (3D view, N key) to switch between the views of the showcase.

import_trajectories.py stores this script in the .blend as a text block that registers itself when the file is
opened (Blender asks to allow scripts; if they are blocked: Text Editor -> rl_view_panel.py -> Run Script).

Buttons:
    Both side by side         with the 3D view split in two (View > Area > Vertical Split): the left half looks
                              through the Myrranox side camera, the right half through the DC side camera
    <creature> side view      look through that creature's steady side camera
    Overview                  camera over the whole habitat
    Follow <creature>         viewport stays centred on the creature; orbit / zoom freely with the mouse
    Free view                 unlock the viewport
"""
import bpy


def _view3d(context):
    for area in context.screen.areas:
        if area.type == "VIEW_3D":
            return area.spaces.active
    return None


class RL_OT_view(bpy.types.Operator):
    bl_idname = "rl.view"
    bl_label = "Creature view"
    bl_options = {"REGISTER"}
    mode: bpy.props.StringProperty()
    target: bpy.props.StringProperty()

    def execute(self, context):
        space = _view3d(context)
        if space is None:
            self.report({"WARNING"}, "no 3D view open")
            return {"CANCELLED"}
        r3d = space.region_3d
        space.lock_object = None
        space.use_local_camera = False
        if self.mode == "camera":
            cam = bpy.data.objects.get(self.target)
            if cam is None:
                return {"CANCELLED"}
            context.scene.camera = cam
            r3d.view_perspective = "CAMERA"
        elif self.mode == "follow":
            focus = bpy.data.objects.get(self.target)
            if focus is None:
                return {"CANCELLED"}
            r3d.view_perspective = "PERSP"
            space.lock_object = focus
            r3d.view_distance = focus.get("rl_view_distance", 6.0)
        else:
            r3d.view_perspective = "PERSP"
        return {"FINISHED"}


class RL_OT_split(bpy.types.Operator):
    """Split the 3D view into two halves, each looking through one creature's side camera (both play together)."""
    bl_idname = "rl.split"
    bl_label = "Both side by side"

    def execute(self, context):
        cams = sorted((o for o in bpy.data.objects if o.name.startswith("SideCam_")), key=lambda o: o.name,
                      reverse=True)          # myrranox left, dc right
        views = [a for a in context.screen.areas if a.type == "VIEW_3D" and a.width > 0]
        if len(views) < 2:
            # Blender only splits areas interactively; one click by the user: View > Area > Vertical Split
            self.report({"WARNING"}, "First split the 3D view: View menu > Area > Vertical Split, then press again")
            return {"CANCELLED"}
        views = sorted(views, key=lambda a: a.x)[:2]
        for area, cam in zip(views, cams):
            space = area.spaces.active
            space.lock_object = None
            space.use_local_camera = True
            space.camera = cam
            space.region_3d.view_perspective = "CAMERA"
            space.shading.type = "SOLID"
            space.shading.color_type = "MATERIAL"
            space.overlay.show_overlays = False           # clean picture: no grid, outlines or empties
        return {"FINISHED"}


class RL_PT_creatures(bpy.types.Panel):
    bl_label = "Creatures"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Creatures"

    def draw(self, context):
        col = self.layout.column(align=True)
        cams = sorted(o for o in bpy.data.objects if o.name.startswith("SideCam_"))
        foci = sorted(o for o in bpy.data.objects if o.name.startswith("Focus_"))
        col.operator("rl.split", text="Both side by side", icon="WINDOW")
        col.separator()
        col.label(text="Cameras")
        for cam in (o.name for o in bpy.data.objects if o.name.startswith("SideCam_")):
            op = col.operator("rl.view", text=f"{cam[8:]} side view", icon="CAMERA_DATA")
            op.mode, op.target = "camera", cam
        if "Overview_cam" in bpy.data.objects:
            op = col.operator("rl.view", text="Overview", icon="WORLD")
            op.mode, op.target = "camera", "Overview_cam"
        col.separator()
        col.label(text="Follow + free orbit")
        for f in (o.name for o in bpy.data.objects if o.name.startswith("Focus_")):
            op = col.operator("rl.view", text=f"Follow {f[6:]}", icon="VIEW_PAN")
            op.mode, op.target = "follow", f
        col.separator()
        op = col.operator("rl.view", text="Free view", icon="VIEW3D")
        op.mode = "free"
        col.separator()
        col.prop(context.scene, "frame_current", text="Frame")
        col.operator("screen.animation_play", text="Play / pause", icon="PLAY")


CLASSES = (RL_OT_view, RL_OT_split, RL_PT_creatures)


def register():
    for c in CLASSES:
        try:
            bpy.utils.register_class(c)
        except ValueError:          # already registered (script run twice)
            bpy.utils.unregister_class(c)
            bpy.utils.register_class(c)


if __name__ == "__main__":
    register()
