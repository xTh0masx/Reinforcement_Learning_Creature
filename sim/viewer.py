"""matplotlib top-down live view of the habitat and one creature."""
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle as CirclePatch, Polygon, Rectangle

ZONE_COLORS = {"myrranox": "#d9a441", "dc": "#5a9e4b"}
SPECIES_COLORS = {"myrranox": "#b5561c", "dc": "#2f6b2a"}
PREY_COLORS = {"myrranox": "#e74c3c", "dc": "#c0392b", "neutral": "#f5b041"}   # neutral prey drawn orange
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
        self.trails, self.bodies = {}, {}
        self.prey_dots = ax.scatter([], [], s=14, c="#c0392b", zorder=4, label="prey", edgecolors="k", linewidths=0.3)
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
        for c in getattr(env, "creatures", [env]):
            cid = getattr(c, "id", "ego")
            if cid not in self.bodies:
                color = SPECIES_COLORS.get(c.species.name, "#333")
                self.trails[cid], = self.ax.plot([], [], "-", lw=0.8, alpha=0.6, color=color)
                self.bodies[cid] = Polygon(np.zeros((3, 2)), closed=True, zorder=5, ec="k", fc=color,
                                           lw=2.0 if c is getattr(env, "ego", c) else 0.5)
                self.ax.add_patch(self.bodies[cid])
            trail = np.asarray(c.trail)
            self.trails[cid].set_data(trail[:, 0], trail[:, 1])
            h, L, W = c.heading, MARKER_LENGTH, MARKER_LENGTH * 0.5
            fwd, left = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])
            tip, back = c.pos + fwd * L * 0.6, c.pos - fwd * L * 0.4
            self.bodies[cid].set_xy([tip, back + left * W / 2, back - left * W / 2])
        alive = env.prey_alive
        self.prey_dots.set_offsets(env.prey_pos[alive] if alive.any() else np.empty((0, 2)))
        if hasattr(env, "prey_area") and alive.any():
            self.prey_dots.set_facecolor([PREY_COLORS.get(a, "#c0392b") for a in env.prey_area[alive]])

        s = env.stats
        self.text.set_text(
            f"{env.species.name}   t = {env.t * 0.1:6.1f} s\n"
            f"hunger   {_bar(env.hunger)} {env.hunger:4.2f}\n"
            f"fatigue  {_bar(env.fatigue)} {env.fatigue:4.2f}\n"
            f"alert    {_bar(env.alertness)} {env.alertness:4.2f}\n"
            f"speed {env.speed:5.2f} m/s  turn {math.degrees(env.turn_rate):6.1f} deg/s\n"
            f"activity {env.activity}\n"
            f"prey {s['prey_caught']}  hits {s['hits']}  reward {s['reward']:8.2f}"
            + (f"\nraids {s.get('raid_prey_eaten', 0)}  stock " +
               " ".join(f"{k[:3]}:{v}" for k, v in env.prey_stock().items()) if hasattr(env, "prey_stock") else "")
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
