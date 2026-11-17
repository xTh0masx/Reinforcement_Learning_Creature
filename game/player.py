"""The player: a free-flying camera like Minecraft's creative mode.

W A S D move (horizontally, in the looking direction), Space up, Ctrl down, Shift fast, mouse looks around.
F follows a creature (again: the other one, again: free flight), Esc frees the mouse.
"""
import math

from panda3d.core import KeyboardButton, Vec3, WindowProperties

WALK, FAST = 9.0, 30.0           # m/s
EYE_MIN = 0.8                    # m above the ground
BOUND = 380.0


class Player:
    def __init__(self, base, terrain, pos=(-6.0, -14.0, 4.0), yaw=20.0, pitch=-12.0):
        self.base = base
        self.terrain = terrain
        self.pos = Vec3(*pos)
        self.vel = Vec3(0, 0, 0)
        self.yaw, self.pitch = yaw, pitch
        self.sens = 0.12
        self.captured = False
        self.follow = None             # creature id or None
        self.orbit_yaw, self.orbit_pitch, self.orbit_dist = 0.0, -22.0, 6.0
        self.k = {name: KeyboardButton.asciiKey(name) for name in "wasd"}

    def capture(self, on):
        wp = WindowProperties()
        wp.setCursorHidden(on)
        wp.setMouseMode(WindowProperties.M_confined if on else WindowProperties.M_absolute)
        self.base.win.requestProperties(wp)
        self.captured = on
        if on:
            self._recenter()

    def _recenter(self):
        w = self.base.win
        w.movePointer(0, w.getXSize() // 2, w.getYSize() // 2)

    def _down(self, b):
        mw = self.base.mouseWatcherNode
        return bool(mw is not None and mw.isButtonDown(b))

    def update(self, dt, creature_pos=None, creature_heading=0.0):
        """creature_pos / heading: the followed creature (world) when following."""
        base = self.base
        dx = dy = 0.0
        if self.captured and base.mouseWatcherNode is not None and base.win.getPointer(0).getInWindow():
            md = base.win.getPointer(0)
            cx, cy = base.win.getXSize() // 2, base.win.getYSize() // 2
            dx, dy = md.getX() - cx, md.getY() - cy
            self._recenter()
        keys = {n: self._down(b) for n, b in self.k.items()}
        up = self._down(KeyboardButton.space())
        down = self._down(KeyboardButton.control()) or self._down(KeyboardButton.lcontrol())
        fast = self._down(KeyboardButton.shift()) or self._down(KeyboardButton.lshift())
        moving = any(keys.values()) or up or down

        if self.follow is not None and moving:
            self.follow = None                      # any movement key returns to free flight
        if self.follow is not None and creature_pos is not None:
            self.orbit_yaw -= dx * self.sens
            self.orbit_pitch = max(-70.0, min(10.0, self.orbit_pitch - dy * self.sens))
            h = math.degrees(creature_heading) - 90.0 + 180.0 + self.orbit_yaw
            p = math.radians(self.orbit_pitch)
            back = Vec3(-math.sin(math.radians(h)) * math.cos(p), math.cos(math.radians(h)) * math.cos(p), -math.sin(p))
            target = Vec3(creature_pos[0], creature_pos[1], creature_pos[2] + 0.8)
            want = target + back * self.orbit_dist
            want.z = max(want.z, self.terrain.height_at(want.x, want.y) + EYE_MIN)
            a = min(1.0, dt * 4.0)
            self.pos += (want - self.pos) * a
            base.camera.setPos(self.pos)
            base.camera.lookAt(target)
            self.yaw, self.pitch = base.camera.getH(), base.camera.getP()
            self.vel = Vec3(0, 0, 0)
            return

        self.yaw -= dx * self.sens
        self.pitch = max(-89.0, min(89.0, self.pitch - dy * self.sens))
        yr = math.radians(self.yaw)
        fwd = Vec3(-math.sin(yr), math.cos(yr), 0)
        right = Vec3(math.cos(yr), math.sin(yr), 0)
        want = Vec3(0, 0, 0)
        want += fwd * (keys["w"] - keys["s"])
        want += right * (keys["d"] - keys["a"])
        if want.length() > 0:
            want.normalize()
        want.z = float(up) - float(down)
        want *= FAST if fast else WALK
        self.vel += (want - self.vel) * min(1.0, dt * 7.0)
        self.pos += self.vel * dt
        self.pos.x = max(-BOUND, min(BOUND, self.pos.x))
        self.pos.y = max(-BOUND, min(BOUND, self.pos.y))
        floor = self.terrain.height_at(self.pos.x, self.pos.y) + EYE_MIN
        if self.pos.z < floor:
            self.pos.z = floor
            self.vel.z = max(self.vel.z, 0)
        self.pos.z = min(self.pos.z, 300.0)
        base.camera.setPos(self.pos)
        base.camera.setHpr(self.yaw, self.pitch, 0)
