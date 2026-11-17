"""On-screen display: name plates over the creatures (name, what it does, hunger and fatigue bars), a minimap with
territories / water / creatures / prey / the player, the clock and the controls."""
import math
import os

import numpy as np
from panda3d.core import CardMaker, Point2, Point3, TextNode, TransparencyAttrib
from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText

from game.world import ASSETS, HALF, fpath

NAMES = {"myrranox": "Myrranox", "dc": "DC"}
ACTIVITY = {"walk": "streift umher", "idle": "steht still", "rest": "ruht sich aus", "eat": "frisst",
            "confront": "verteidigt sein Revier"}
COLOR = {"myrranox": (0.95, 0.62, 0.28, 1), "dc": (0.55, 0.75, 0.95, 1)}
HEAD_UP = {"myrranox": 1.3, "dc": 2.3}
GOLD = (0.93, 0.83, 0.58, 1)
HELP = ("W A S D  bewegen     Maus  umsehen     Leertaste  hoch     Strg  runter     Shift  schnell\n"
        "F  Kreatur folgen     T  Zeit x1 / x2 / x4     P  Pause     H  Hilfe     Esc  Maus freigeben")


def _card(parent, l, r, b, t, color):
    cm = CardMaker("c")
    cm.setFrame(l, r, b, t)
    n = parent.attachNewNode(cm.generate())
    n.setColor(*color)
    n.setTransparency(TransparencyAttrib.M_alpha)
    return n


class Bar:
    def __init__(self, parent, y, color, label):
        self.root = parent.attachNewNode("bar")
        self.root.setZ(y)
        _card(self.root, -0.16, 0.16, -0.012, 0.012, (0, 0, 0, 0.55))
        self.fill = _card(self.root, 0, 1, -0.008, 0.008, color)
        self.fill.setX(-0.157)
        OnscreenText(label, parent=self.root, pos=(-0.175, -0.008), scale=0.022, fg=(1, 1, 1, 0.85),
                     align=TextNode.ARight, mayChange=False)

    def set(self, v):
        self.fill.setSx(max(0.001, 0.314 * float(v)))


class Plate:
    def __init__(self, parent, species):
        self.root = parent.attachNewNode("plate")
        self.name = OnscreenText(NAMES[species], parent=self.root, pos=(0, 0.06), scale=0.042, fg=COLOR[species],
                                 shadow=(0, 0, 0, 0.8), mayChange=True)
        self.act = OnscreenText("", parent=self.root, pos=(0, 0.027), scale=0.026, fg=(1, 1, 1, 0.9),
                                shadow=(0, 0, 0, 0.8), mayChange=True)
        self.hunger = Bar(self.root, 0.0, (0.86, 0.35, 0.22, 0.95), "Hunger")
        self.fatigue = Bar(self.root, -0.03, (0.35, 0.6, 0.95, 0.95), "Müdigkeit")


class Hud:
    def __init__(self, base, world, sim):
        self.base, self.world, self.sim = base, world, sim
        a2 = base.aspect2d
        self.plates = {}
        self.title = OnscreenText("ENDEMIC CREATURES", parent=base.a2dTopLeft, pos=(0.06, -0.08), scale=0.05, fg=GOLD,
                                  align=TextNode.ALeft, shadow=(0, 0, 0, 0.8))
        self.clock = OnscreenText("", parent=base.a2dTopLeft, pos=(0.06, -0.135), scale=0.034, fg=(1, 1, 1, 0.9),
                                  align=TextNode.ALeft, shadow=(0, 0, 0, 0.8), mayChange=True)
        self.score = OnscreenText("", parent=base.a2dTopLeft, pos=(0.06, -0.18), scale=0.03, fg=(1, 1, 1, 0.85),
                                  align=TextNode.ALeft, shadow=(0, 0, 0, 0.8), mayChange=True)
        self.help = OnscreenText(HELP, parent=base.a2dBottomLeft, pos=(0.06, 0.09), scale=0.03, fg=(1, 1, 1, 0.8),
                                 align=TextNode.ALeft, shadow=(0, 0, 0, 0.9))
        self.message = OnscreenText("", parent=a2, pos=(0, 0.3), scale=0.06, fg=GOLD, shadow=(0, 0, 0, 0.9),
                                    mayChange=True)
        self.hint = OnscreenText("Klicke ins Fenster, um zu steuern", parent=a2, pos=(0, -0.1), scale=0.045,
                                 fg=(1, 1, 1, 0.9), shadow=(0, 0, 0, 0.9), mayChange=True)
        self.dot = OnscreenText("+", parent=a2, pos=(0, -0.012), scale=0.04, fg=(1, 1, 1, 0.45))
        self._minimap()

    # ------------------------------------------------------------------------------------------------ minimap
    def _minimap(self):
        path = os.path.join(ASSETS, "textures", "minimap.png")
        if not os.path.exists(path):
            from PIL import Image
            terr = self.world.terrain
            g = np.linspace(-HALF - 6, HALF + 6, 256)
            X, Y = np.meshgrid(g, g[::-1])
            ix = np.clip(np.searchsorted(terr.xs, X), 0, len(terr.xs) - 1)
            iy = np.clip(np.searchsorted(terr.ys, Y), 0, len(terr.ys) - 1)
            w = terr.splat[iy, ix]
            pal = np.array([[96, 128, 62], [118, 94, 66], [200, 172, 120], [120, 116, 110]], float)
            img = w @ pal
            r = np.hypot(X, Y)
            img[r < 4.0] = [70, 120, 140]
            for s in self.world.hab.rest_spots:
                img[np.hypot(X - s.x, Y - s.y) < s.r] *= 0.7
            outside = np.maximum(np.abs(X), np.abs(Y)) > HALF
            img[outside] *= 0.55
            a = np.full(X.shape + (1,), 235.0)
            Image.fromarray(np.concatenate([img, a], -1).clip(0, 255).astype(np.uint8), "RGBA").save(path)
        self.map_size = 0.36
        self.map_root = self.base.a2dTopRight.attachNewNode("minimap")
        self.map_root.setPos(-0.06 - self.map_size, 0, -0.06 - self.map_size)
        _card(self.map_root, -self.map_size - 0.008, self.map_size + 0.008, -self.map_size - 0.008, self.map_size + 0.008,
              (0.05, 0.05, 0.04, 0.8))
        img = OnscreenImage(fpath(path).getFullpath(), parent=self.map_root, scale=self.map_size)
        img.setTransparency(TransparencyAttrib.M_alpha)
        OnscreenText("N", parent=self.map_root, pos=(0, self.map_size - 0.05), scale=0.035, fg=GOLD, shadow=(0, 0, 0, 1))
        self.marks = {}
        self.prey_marks = []
        self.player_mark = OnscreenText("^", parent=self.map_root, scale=0.06, fg=(1, 1, 1, 1), shadow=(0, 0, 0, 1))

    def _map_xy(self, x, y):
        k = self.map_size / (HALF + 6)
        return max(-1, min(1, x / (HALF + 6))) * self.map_size, max(-1, min(1, y / (HALF + 6))) * self.map_size

    # ------------------------------------------------------------------------------------------------ per frame
    def update(self, status, creature_head, player):
        base = self.base
        sim = self.sim
        t = sim.clock
        self.clock.setText(f"Tag {sim.generation}   {int(t // 60):02d}:{int(t % 60):02d}   "
                           f"Zeit x{sim.speed:g}" + ("   PAUSE" if sim.paused else ""))
        self.message.setText(sim.message)
        self.hint.setText("" if player.captured else "Klicke ins Fenster, um zu steuern")
        lens = base.cam.node().getLens()
        parts = []
        for cid, c in status["creatures"].items():
            sp = c["species"]
            parts.append(f"{NAMES[sp]}: {c['prey']} Beute")
            pl = self.plates.get(cid)
            if pl is None:
                pl = self.plates[cid] = Plate(base.aspect2d, sp)
            pl.hunger.set(c["hunger"])
            pl.fatigue.set(c["fatigue"])
            act = ACTIVITY.get(c["activity"], c["activity"])
            if c["starving"]:
                act += "  -  verhungert!"
            pl.act.setText(act)
            head = creature_head.get(cid)
            shown = False
            if head is not None:
                p_cam = base.cam.getRelativePoint(base.render, Point3(head[0], head[1], head[2] + HEAD_UP[sp]))
                p2 = Point2()
                dist = p_cam.length()
                if p_cam.y > 0.5 and dist < 90 and lens.project(p_cam, p2):
                    pl.root.setPos(p2.x * base.getAspectRatio(), 0, p2.y)
                    pl.root.setScale(max(0.55, min(1.15, 9.0 / max(dist, 1e-3))))
                    shown = True
            pl.root.show() if shown else pl.root.hide()
            # minimap marker
            m = self.marks.get(cid)
            if m is None:
                m = self.marks[cid] = OnscreenText("o", parent=self.map_root, scale=0.055, fg=COLOR[sp],
                                                   shadow=(0, 0, 0, 1))
            mx, my = self._map_xy(*c["pos"])
            m.setPos(mx, my - 0.015)
        self.score.setText("     ".join(parts))
        prey = status["prey"]
        while len(self.prey_marks) < len(prey):
            self.prey_marks.append(OnscreenText(".", parent=self.map_root, scale=0.06, fg=(0.9, 0.2, 0.2, 1)))
        for mk, (p, _, alive) in zip(self.prey_marks, prey):
            if alive:
                mx, my = self._map_xy(*p)
                mk.setPos(mx, my)
                mk.show()
            else:
                mk.hide()
        px, py = self._map_xy(player.pos.x, player.pos.y)
        self.player_mark.setPos(px, py - 0.02)
        self.player_mark.setR(-player.yaw)
