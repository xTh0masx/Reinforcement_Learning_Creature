"""matplotlib top-down live view of the habitat and one creature."""
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle as CirclePatch, Polygon, Rectangle

ZONE_COLORS = {"myrranox": "#d9a441", "dc": "#5a9e4b"}
SPECIES_COLORS = {"myrranox": "#b5561c", "dc": "#2f6b2a"}
MARKER_LENGTH = 3.0   # m, drawn larger than the real body so it stays visible on a 100 m map


class HabitatViewer:
    def __init__(self, habitat, interactive=True):
        self.habitat = habitat
        self.interactive = interactive
        if interactive:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
        else:
            self.fig = Figure(figsize=(8, 8))
            FigureCanvasAgg(self.fig)
            self.ax = self.fig.add_subplot()
        self._draw_static()

        ax = self.ax
        self.trail_line, = ax.plot([], [], "-", lw=0.8, alpha=0.6, color="k")
        self.prey_dots = ax.scatter([], [], s=12, c="#c0392b", zorder=4, label="prey")
        self.body = Polygon(np.zeros((3, 2)), closed=True, zorder=5, ec="k")
        ax.add_patch(self.body)
        self.text = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left",
                            family="monospace", fontsize=8,
                            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    def _draw_static(self):
        h, ax = self.habitat, self.ax
        ax.set_aspect("equal")
        ax.set_xlim(h.xmin - 2, h.xmax + 2)
        ax.set_ylim(h.ymin - 2, h.ymax + 2)
        ax.add_patch(Rectangle((h.xmin, h.ymin), h.xmax - h.xmin, h.ymax - h.ymin,
                               fill=False, lw=2, ec="#555"))
        for name, z in h.home_zones.items():
            ax.add_patch(CirclePatch((z.x, z.y), z.r, fc=ZONE_COLORS.get(name, "#ccc"), alpha=0.2,
                                     ec=ZONE_COLORS.get(name, "#999"), lw=1.5))
            ax.text(z.x, z.y + z.r - 3, f"home: {name}", ha="center", fontsize=8, color="#444")
        ax.add_patch(CirclePatch((h.water.x, h.water.y), h.water.r, fc="#6fa8dc", alpha=0.7, ec="none"))
        for s in h.rest_spots:
            ax.add_patch(CirclePatch((s.x, s.y), s.r, fill=False, ls="--", ec="#6d4c41", lw=1.2))
        for o in h.obstacles:
            ax.add_patch(CirclePatch((o.x, o.y), o.r, fc="#8e8e8e", ec="#444"))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    def update(self, env):
        trail = np.asarray(env.trail)
        self.trail_line.set_data(trail[:, 0], trail[:, 1])
        self.prey_dots.set_offsets(env.prey_pos[env.prey_alive] if env.prey_alive.any() else np.empty((0, 2)))

        x, y, h = env.pos[0], env.pos[1], env.heading
        L, W = MARKER_LENGTH, MARKER_LENGTH * 0.5
        fwd, left = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])
        tip = env.pos + fwd * L * 0.6
        back = env.pos - fwd * L * 0.4
        self.body.set_xy([tip, back + left * W / 2, back - left * W / 2])
        self.body.set_facecolor(SPECIES_COLORS.get(env.species.name, "#333"))

        s = env.stats
        self.text.set_text(
            f"{env.species.name}   t = {env.t * 0.1:6.1f} s\n"
            f"hunger   {_bar(env.hunger)} {env.hunger:4.2f}\n"
            f"fatigue  {_bar(env.fatigue)} {env.fatigue:4.2f}\n"
            f"alert    {_bar(env.alertness)} {env.alertness:4.2f}\n"
            f"speed {env.speed:5.2f} m/s  turn {math.degrees(env.turn_rate):6.1f} deg/s\n"
            f"activity {env.activity}\n"
            f"prey {s['prey_caught']}  hits {s['hits']}  reward {s['reward']:8.2f}"
        )
        if self.interactive:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

    def rgb_array(self):
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        if self.interactive:
            plt.close(self.fig)


def _bar(v, n=10):
    k = int(round(v * n))
    return "#" * k + "." * (n - k)
