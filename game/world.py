"""The game world around the 100 x 100 m habitat: terrain (flat inside the walls, exactly where the physics floor is;
cliffs and mountains outside), ground textures per territory, water hole, rest spots, plants and rocks (Poly Haven,
CC0), grass, sky and sun with shadows.

Territories (sim/habitat.json): Myrranox south-west (sand and rock, desert plants, burrows), DC north-east (grass,
trees, shade trees over its rest spots), water hole in the middle.
"""
import json
import math
import os

import numpy as np
from panda3d.core import (BitMask32, CompassEffect, DirectionalLight, Filename, Geom, GeomNode, GeomTriangles,
                          GeomVertexArrayFormat, GeomVertexData, GeomVertexFormat, InternalName, Material, NodePath,
                          SamplerState, Shader, TextureAttrib, TransparencyAttrib, Vec3)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
SHADERS = os.path.join(HERE, "shaders")
HALF = 50.0                     # habitat walls at +-50 m
SHADOW_MASK = BitMask32.bit(3)  # what the sun's shadow camera renders


def fpath(p):
    return Filename.fromOsSpecific(p)


def shader(vert, frag):
    return Shader.load(Shader.SL_GLSL, vertex=fpath(os.path.join(SHADERS, vert)), fragment=fpath(os.path.join(SHADERS, frag)))


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------------------------------- noise
def _hash(ix, iy, seed):
    return np.modf(np.abs(np.sin(ix * 127.1 + iy * 311.7 + seed * 74.7) * 43758.5453))[0]


def value_noise(x, y, seed=0):
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    ux, uy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = _hash(ix, iy, seed), _hash(ix + 1, iy, seed)
    c, d = _hash(ix, iy + 1, seed), _hash(ix + 1, iy + 1, seed)
    return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy


def fbm(x, y, octaves=4, seed=0):
    s, amp, tot = 0.0, 0.5, 0.0
    for o in range(octaves):
        s = s + amp * value_noise(x * 2 ** o, y * 2 ** o, seed + o * 13)
        tot += amp
        amp *= 0.5
    return s / tot


# ---------------------------------------------------------------------------------------------------- terrain
def height(x, y):
    """Ground height. 0 inside the walls (the physics floor), rising cliffs and mountains outside."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = np.maximum(np.abs(x), np.abs(y)) - HALF
    cliff = 15.0 * smoothstep(0.3, 10.0, d) * (0.6 + 0.8 * fbm(x / 22.0, y / 22.0, 3, 5))
    far = 120.0 * smoothstep(15.0, 360.0, d) ** 1.25 * (0.45 + 1.1 * fbm(x / 90.0, y / 90.0, 4, 9))
    detail = 3.0 * (fbm(x / 6.0, y / 6.0, 3, 2) - 0.5) * smoothstep(0.0, 14.0, d)
    z = np.where(d > 0, cliff + far + detail, 0.0)
    r = np.hypot(x, y)
    z = z - 0.4 * (1.0 - smoothstep(3.0, 4.8, r))          # the water hole's bed
    return z


def _axis():
    return np.unique(np.concatenate([np.linspace(-430, -150, 29), np.arange(-150, -72, 3.0), np.arange(-72, 72.01, 1.0),
                                     np.arange(75, 150.01, 3.0), np.linspace(150, 430, 29)]))


def _mesh(name, data, idx, columns):
    """columns: [(InternalName, n, contents)] all float32; data (N, sum n) float32; idx (M, 3)."""
    arr = GeomVertexArrayFormat()
    for nm, n, contents in columns:
        arr.addColumn(nm, n, Geom.NT_float32, contents)
    fmt = GeomVertexFormat.registerFormat(arr)
    vd = GeomVertexData(name, fmt, Geom.UH_static)
    vd.uncleanSetNumRows(len(data))
    vd.modifyArrayHandle(0).copyDataFrom(np.ascontiguousarray(data, np.float32))
    tris = GeomTriangles(Geom.UH_static)
    tris.setIndexType(Geom.NT_uint32)
    tris.modifyVertices().modifyHandle().copyDataFrom(np.ascontiguousarray(idx, np.uint32).ravel())
    g = Geom(vd)
    g.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(g)
    return NodePath(node)


V = InternalName.getVertex()
N = InternalName.getNormal()
C = InternalName.getColor()
T = InternalName.getTexcoord()


def grid_indices(nx, ny):
    i = np.arange((ny - 1) * (nx - 1))
    r, c = i // (nx - 1), i % (nx - 1)
    a = r * nx + c
    return np.stack([np.stack([a, a + 1, a + nx + 1], 1), np.stack([a, a + nx + 1, a + nx], 1)], 1).reshape(-1, 3)


class Terrain:
    def __init__(self, habitat):
        self.xs = self.ys = _axis()
        X, Y = np.meshgrid(self.xs, self.ys)
        self.Z = height(X, Y)
        gy, gx = np.gradient(self.Z, self.ys, self.xs)
        n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        self.splat = self._splat(X, Y, n, habitat)
        data = np.concatenate([np.stack([X, Y, self.Z], -1), n, self.splat], -1).reshape(-1, 10)
        self.np = _mesh("terrain", data, grid_indices(len(self.xs), len(self.ys)), [(V, 3, Geom.C_point),
                                                                                   (N, 3, Geom.C_normal),
                                                                                   (C, 4, Geom.C_color)])

    @staticmethod
    def _splat(X, Y, n, hab):
        noise = fbm(X / 9.0, Y / 9.0, 4, 3)
        mz, dz = hab.home_zones["myrranox"], hab.home_zones["dc"]
        dm = np.hypot(X - mz.x, Y - mz.y) - mz.r + 10 * (noise - 0.5)
        dd = np.hypot(X - dz.x, Y - dz.y) - dz.r + 10 * (noise - 0.5)
        myr = 1 - smoothstep(-5, 5, dm)
        dc = 1 - smoothstep(-5, 5, dd)
        outside = smoothstep(0.0, 3.0, np.maximum(np.abs(X), np.abs(Y)) - HALF)
        side = smoothstep(-30, 30, X + Y + 25 * (noise - 0.5))            # 1 = DC side (north-east)
        neutral = np.clip(1 - myr - dc, 0, 1) * (1 - outside)
        patches = fbm(X / 4.0, Y / 4.0, 3, 8)
        grass = dc * (0.75 + 0.5 * patches) + neutral * 0.6 * patches + outside * side
        sand = myr * (0.8 + 0.4 * patches) + outside * (1 - side) + neutral * 0.25
        dirt = neutral * (1 - patches) * 0.8 + dc * 0.25 * (1 - patches) + myr * 0.15
        r = np.hypot(X, Y)
        dirt += 2.5 * (1 - smoothstep(4.0, 8.5, r))                          # muddy shore
        for s in hab.rest_spots:                                             # worn ground on the rest spots
            dirt += 1.6 * (1 - smoothstep(1.0, 3.2, np.hypot(X - s.x, Y - s.y)))
        slope = 1 - n[..., 2]
        rock = smoothstep(0.12, 0.35, slope) + myr * smoothstep(0.72, 0.85, fbm(X / 7.0, Y / 7.0, 3, 21)) * (1 - outside)
        rock = np.clip(rock, 0, 1)
        w = np.stack([grass, dirt, sand], -1)
        w /= np.maximum(w.sum(-1, keepdims=True), 1e-6)
        w *= (1 - rock)[..., None]
        return np.concatenate([w, rock[..., None]], -1)

    def height_at(self, x, y):
        """Bilinear height from the grid (camera collision, placing things)."""
        i = int(np.clip(np.searchsorted(self.xs, x) - 1, 0, len(self.xs) - 2))
        j = int(np.clip(np.searchsorted(self.ys, y) - 1, 0, len(self.ys) - 2))
        u = (x - self.xs[i]) / (self.xs[i + 1] - self.xs[i])
        v = (y - self.ys[j]) / (self.ys[j + 1] - self.ys[j])
        u, v = min(max(u, 0), 1), min(max(v, 0), 1)
        z = self.Z
        return float((1 - u) * (1 - v) * z[j, i] + u * (1 - v) * z[j, i + 1] + (1 - u) * v * z[j + 1, i] + u * v * z[j + 1, i + 1])


# ---------------------------------------------------------------------------------------------------- textures
def _make_textures():
    """Procedural helper textures, cached: grass blades (alpha) and a large-scale variation map."""
    from PIL import Image, ImageDraw
    tex = os.path.join(ASSETS, "textures")
    blades = os.path.join(tex, "grass_blades.png")
    if not os.path.exists(blades):
        rng = np.random.default_rng(1)
        im = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        dr = ImageDraw.Draw(im)
        for _ in range(46):
            x0 = rng.uniform(20, 236)
            h = rng.uniform(120, 250)
            lean = rng.uniform(-40, 40)
            w = rng.uniform(3, 7)
            shade = rng.uniform(0.7, 1.0)
            for k in range(24):
                t0, t1 = k / 24, (k + 1) / 24
                x_a, x_b = x0 + lean * t0 ** 1.6, x0 + lean * t1 ** 1.6
                y_a, y_b = 255 - h * t0, 255 - h * t1
                ww = w * (1 - t0) + 0.6
                c = int(255 * shade * (0.55 + 0.45 * t0))
                dr.polygon([(x_a - ww, y_a), (x_a + ww, y_a), (x_b + ww * 0.9, y_b), (x_b - ww * 0.9, y_b)],
                           fill=(c, c, c, 255))
        im.save(blades)
    macro = os.path.join(tex, "macro.png")
    if not os.path.exists(macro):
        g = np.linspace(0, 8, 256, endpoint=False)
        X, Y = np.meshgrid(g, g)
        m = fbm(X, Y, 5, 31)            # not tileable at the seam - it is sampled across 90 m, the seam is invisible
        m = (m - m.min()) / (m.max() - m.min())
        Image.fromarray((m * 255).astype(np.uint8)).save(macro)
    return blades, macro


def _sky_info(sky_path):
    """Sun direction from the brightest spot of the sky image, the horizon colour and the sky's mean colour."""
    from PIL import Image
    im = np.asarray(Image.open(sky_path).convert("RGB").resize((512, 256)), np.float32) / 255
    lum = im.mean(-1)
    py, px = np.unravel_index(np.argmax(lum[:128]), lum[:128].shape)
    u, v = (px + 0.5) / 512, 1 - (py + 0.5) / 256
    az, el = (u - 0.5) * 2 * math.pi, (v - 0.5) * math.pi
    el = max(el, math.radians(35))       # a low sun makes long shadows; keep it high enough to read the creatures
    sun = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    horizon = im[120:126].reshape(-1, 3).mean(0)
    upper = (im[:110] ** 2.2).reshape(-1, 3).mean(0)
    return sun, horizon, upper


def only_base_color(np_):
    """Keep only the colour map on every geom (glTF also brings normal and metal-roughness maps, and the shader's
    p3d_Texture0 would otherwise pick whichever stage comes first)."""
    for gnp in np_.findAllMatches("**/+GeomNode"):
        gn = gnp.node()
        for i in range(gn.getNumGeoms()):
            st = gn.getGeomState(i)
            ta = st.getAttrib(TextureAttrib)
            if ta is None:
                continue
            new = TextureAttrib.make()
            for k in range(ta.getNumOnStages()):
                stg = ta.getOnStage(k)
                if stg.getName() == "Base Color":
                    new = new.addOnStage(stg, ta.getOnTexture(stg))
            gn.setGeomState(i, st.setAttrib(new))


# ---------------------------------------------------------------------------------------------------- world
class World:
    def __init__(self, base, habitat, seed=3, off=()):
        self.base = base
        self.hab = habitat
        self.rng = np.random.default_rng(seed)
        self.root = base.render.attachNewNode("world")
        loader = base.loader
        tex_dir = os.path.join(ASSETS, "textures")
        blades, macro = _make_textures()
        sky_path = os.path.join(tex_dir, "sky.jpg")
        sun, horizon, upper = _sky_info(sky_path)
        self.sun_dir = Vec3(*sun)
        self.fog_color = Vec3(*horizon)

        # shared lighting inputs (every shader reads them)
        r = base.render
        self.lit = shader("lit.vert", "lit.frag")
        r.setShader(self.lit)
        r.setShaderInput("sun_dir", self.sun_dir)
        r.setShaderInput("sun_color", Vec3(2.3, 2.15, 1.9))
        r.setShaderInput("sky_ambient", Vec3(*(upper * 0.9)))
        r.setShaderInput("ground_ambient", Vec3(0.16, 0.14, 0.11))
        r.setShaderInput("fog_color", self.fog_color)
        r.setShaderInput("fog_density", 1 / 330.0)
        r.setShaderInput("cam_pos", Vec3(0, 0, 0))
        r.setShaderInput("wind_time", 0.0)
        r.setShaderInput("wind_strength", 0.0)
        r.setShaderInput("mode", 0)
        r.setShaderInput("back_color", Vec3(0.5, 0.5, 0.5))
        r.setShaderInput("belly_color", Vec3(0.7, 0.7, 0.7))

        # sun + shadows
        dl = DirectionalLight("sun")
        dl.setColor((1, 1, 1, 1))
        dl.setShadowCaster("shadows" not in off, 4096, 4096)
        dl.setCameraMask(SHADOW_MASK)
        lens = dl.getLens()
        lens.setFilmSize(70, 70)
        lens.setNearFar(1, 400)
        self.sun_np = r.attachNewNode(dl)
        r.setLight(self.sun_np)
        self.shadow_size = 70.0

        # sky dome (follows the camera, keeps world orientation)
        self.sky_tex = loader.loadTexture(fpath(sky_path))
        self.sky_tex.setWrapU(SamplerState.WM_repeat)
        self.sky_tex.setWrapV(SamplerState.WM_clamp)
        sky = self._sphere(32, 16)
        sky.reparentTo(base.camera)
        sky.setEffect(CompassEffect.make(r))
        sky.setScale(800)
        sky.setShader(shader("sky.vert", "sky.frag"), 10)
        sky.setShaderInput("sky_tex", self.sky_tex)
        sky.setBin("background", 0)
        sky.setDepthWrite(False)
        sky.setDepthTest(False)
        sky.setTwoSided(True)
        sky.setLightOff(10)
        sky.hide(SHADOW_MASK)
        self.sky = sky

        # terrain
        self.terrain = Terrain(habitat)
        t = self.terrain.np
        t.reparentTo(self.root)
        t.setShader(shader("terrain.vert", "terrain.frag"), 5)
        for key in ("grass", "dirt", "sand", "rock"):
            tx = loader.loadTexture(fpath(os.path.join(tex_dir, f"{key}_diff.jpg")))
            tx.setMinfilter(SamplerState.FT_linear_mipmap_linear)
            tx.setAnisotropicDegree(4)
            t.setShaderInput(f"{key}_tex", tx)
        t.setShaderInput("macro_tex", loader.loadTexture(fpath(macro)))
        t.hide(SHADOW_MASK)          # flat-ish ground: receives shadows, casting them would only cost time

        self._water()
        self.models = {}
        self.sizes = json.load(open(os.path.join(ASSETS, "env", "sizes.json")))
        if "plants" not in off:
            self._place_everything()
        if "grass" not in off:
            self._grass(blades)

    # ------------------------------------------------------------------------------------------------ helpers
    def _sphere(self, nu, nv):
        u = np.linspace(0, 2 * math.pi, nu + 1)
        v = np.linspace(-math.pi / 2, math.pi / 2, nv + 1)
        U, Vv = np.meshgrid(u, v)
        p = np.stack([np.cos(Vv) * np.cos(U), np.cos(Vv) * np.sin(U), np.sin(Vv)], -1).reshape(-1, 3)
        return _mesh("sky", p, grid_indices(nu + 1, nv + 1), [(V, 3, Geom.C_point)])

    def _water(self):
        rad = 4.6
        ang = np.linspace(0, 2 * math.pi, 65)[:-1]
        rings = np.linspace(0, 1, 7)
        pts, cols = [], []
        for rr in rings:
            for a in ang:
                pts.append((rr * rad * math.cos(a), rr * rad * math.sin(a), 0.07))
                cols.append((1 - rr, 0, 0, 1))
        pts, cols = np.array(pts), np.array(cols)
        n = len(ang)
        idx = []
        for i in range(len(rings) - 1):
            for k in range(n):
                a, b = i * n + k, i * n + (k + 1) % n
                idx += [(a, b, b + n), (a, b + n, a + n)]
        data = np.concatenate([pts, np.tile([0, 0, 1], (len(pts), 1)), np.zeros((len(pts), 2)), cols], 1)
        w = _mesh("water", data, np.array(idx), [(V, 3, Geom.C_point), (N, 3, Geom.C_normal), (T, 2, Geom.C_texcoord),
                                                 (C, 4, Geom.C_color)])
        w.reparentTo(self.root)
        w.setShader(shader("lit.vert", "water.frag"), 5)
        w.setShaderInput("sky_tex", self.sky_tex)
        w.setTransparency(TransparencyAttrib.M_alpha)
        w.setBin("transparent", 10)
        w.setDepthWrite(False)
        w.hide(SHADOW_MASK)
        self.water = w

    def model(self, name):
        if name not in self.models:
            m = self.base.loader.loadModel(fpath(os.path.join(ASSETS, "env", f"{name}.glb")))
            m.setTwoSided(True)
            only_base_color(m)
            m.clearModelNodes()
            m.flattenStrong()
            self.models[name] = m
        return self.models[name]

    def place(self, name, x, y, scale=1.0, yaw=None, sink=0.05, tilt=0.0, parent=None):
        m = self.model(name).copyTo(parent or self.root)
        s = self.sizes[name]
        m.setScale(scale)
        m.setHpr(self.rng.uniform(0, 360) if yaw is None else yaw, self.rng.uniform(-tilt, tilt), self.rng.uniform(-tilt, tilt))
        z = self.terrain.height_at(x, y) if abs(x) > HALF - 0.5 or abs(y) > HALF - 0.5 else 0.0
        m.setPos(x, y, z - s["min"][2] * scale - sink * scale * max(s["max"][2] - s["min"][2], 0.2))
        return m

    def _scatter(self, n, region, min_dist, avoid=()):
        """n random points from region(rng) -> (x, y) or None, kept min_dist apart and away from avoid [(x, y, r)]."""
        pts = []
        tries = 0
        while len(pts) < n and tries < n * 60:
            tries += 1
            p = region(self.rng)
            if p is None:
                continue
            if any(math.hypot(p[0] - q[0], p[1] - q[1]) < min_dist for q in pts):
                continue
            if any(math.hypot(p[0] - a[0], p[1] - a[1]) < a[2] for a in avoid):
                continue
            pts.append(p)
        return pts

    def _zone(self, p):
        mz, dz = self.hab.home_zones["myrranox"], self.hab.home_zones["dc"]
        if math.hypot(p[0] - dz.x, p[1] - dz.y) < dz.r:
            return "dc"
        if math.hypot(p[0] - mz.x, p[1] - mz.y) < mz.r:
            return "myrranox"
        return "neutral"

    # ------------------------------------------------------------------------------------------------ dressing
    def _place_everything(self):
        rng = self.rng
        hab = self.hab
        keep_clear = [(0, 0, 9.0)] + [(s.x, s.y, 4.0) for s in hab.rest_spots]

        # Myrranox burrows: a ring of boulders round each burrow
        for s in hab.rest_spots:
            if s.kind == "burrow":
                for k in range(4):
                    a = rng.uniform(0, 2 * math.pi) if k == 0 else a + rng.uniform(1.1, 1.9)
                    rr = rng.uniform(3.3, 4.2)
                    self.place(rng.choice(["namaqualand_boulder_02", "namaqualand_boulder_04", "namaqualand_boulder_05"]),
                               s.x + rr * math.cos(a), s.y + rr * math.sin(a), scale=rng.uniform(0.7, 1.2), tilt=8)
            else:   # DC shade spots: a big tree whose crown covers the spot
                out = math.atan2(s.y, s.x)
                self.place("jacaranda_tree", s.x + 3.8 * math.cos(out), s.y + 3.8 * math.sin(out),
                           scale=rng.uniform(0.42, 0.5))

        # cliffs: a broken rock wall along the foot of the slopes, just outside the habitat walls
        for side in range(4):
            for t in np.arange(-56, 57, 5.5):
                off = rng.uniform(1.0, 3.5)
                x, y = [(t, HALF + off), (t, -HALF - off), (HALF + off, t), (-HALF - off, t)][side]
                name = rng.choice(["rock_face_01", "rock_face_02", "namaqualand_cliff_01"])
                sc = {"rock_face_01": 1.6, "rock_face_02": 2.6, "namaqualand_cliff_01": 1.0}[name] * rng.uniform(0.8, 1.3)
                self.place(name, x, y, scale=sc, sink=0.15, tilt=10)

        # trees and big rocks along the inside of the walls and up on the slopes
        def ring(rng, inner, outer):
            x, y = rng.uniform(-HALF - outer, HALF + outer, 2)
            d = max(abs(x), abs(y)) - HALF
            return (x, y) if inner <= d <= outer else None
        for p in self._scatter(70, lambda r: ring(r, -6.0, -1.5), 5.0, keep_clear):
            if self._zone(p) == "dc" or (self._zone(p) == "neutral" and p[0] + p[1] > 0):
                self.place(rng.choice(["island_tree_02", "island_tree_02", "shrub_02", "rock_moss_set_01", "tree_stump_01",
                                       "dead_tree_trunk"]), *p,
                           scale=rng.uniform(1.2, 2.0))
            else:
                self.place(rng.choice(["quiver_tree_01", "quiver_tree_02", "namaqualand_boulder_04", "dead_quiver_trunk",
                                       "namaqualand_boulder_02"]), *p, scale=rng.uniform(1.1, 1.8), tilt=4)
        for p in self._scatter(140, lambda r: ring(r, 6.0, 70.0), 9.0):
            if p[0] + p[1] > 0:
                self.place(rng.choice(["island_tree_02", "jacaranda_tree", "island_tree_02"]), *p,
                           scale=rng.uniform(2.0, 3.2) if rng.random() < 0.7 else rng.uniform(0.35, 0.5))
            else:
                self.place(rng.choice(["quiver_tree_01", "quiver_tree_02", "namaqualand_boulder_04"]), *p,
                           scale=rng.uniform(1.8, 3.0))

        # low plants inside (the creatures walk through them - they stay small)
        def inside(rng):
            return tuple(rng.uniform(-HALF + 2, HALF - 2, 2))
        for p in self._scatter(170, inside, 2.2, keep_clear):
            z = self._zone(p)
            if z == "dc":
                self.place(rng.choice(["fern_02", "shrub_03", "shrub_04", "grass_medium_01"]), *p, scale=rng.uniform(0.6, 1.1))
            elif z == "myrranox":
                self.place(rng.choice(["namaqualand_rocks_01", "shrub_04", "dead_quiver_trunk"]), *p, scale=rng.uniform(0.6, 1.0))
            elif rng.random() < 0.5:
                self.place(rng.choice(["shrub_03", "grass_medium_01", "namaqualand_rocks_01"]), *p, scale=rng.uniform(0.6, 1.0))

        # a few stones round the water hole
        for k in range(7):
            a = k / 7 * 2 * math.pi + rng.uniform(-0.3, 0.3)
            self.place("namaqualand_boulder_05", 5.6 * math.cos(a), 5.6 * math.sin(a), scale=rng.uniform(0.4, 0.7), tilt=10)

    def _grass(self, blades_path):
        """Grass tufts (two crossed alpha cards each), batched per 25 m cell so the camera culls what it does not see."""
        tex = self.base.loader.loadTexture(fpath(blades_path))
        tex.setWrapU(SamplerState.WM_clamp)
        tex.setWrapV(SamplerState.WM_clamp)
        rng = self.rng
        n = 60000
        P = rng.uniform(-HALF + 0.5, HALF - 0.5, (n, 2))
        terr = self.terrain
        X, Y = P[:, 0], P[:, 1]
        ix = np.clip(np.searchsorted(terr.xs, X), 0, len(terr.xs) - 1)
        iy = np.clip(np.searchsorted(terr.ys, Y), 0, len(terr.ys) - 1)
        w = terr.splat[iy, ix]
        dens = w[:, 0] * 1.0 + w[:, 2] * 0.035 + w[:, 1] * 0.12
        keep = (rng.random(n) < dens) & (np.hypot(X, Y) > 4.9)
        P, w = P[keep], w[keep]
        green = w[:, 0] / np.maximum(w[:, 0] + w[:, 2] + w[:, 1], 1e-6)
        cols = np.stack([0.36 + 0.3 * (1 - green), 0.44 + 0.06 * (1 - green), 0.14 + 0.06 * (1 - green)], 1)
        cols *= rng.uniform(0.75, 1.15, (len(P), 1))
        root = self.root.attachNewNode("grass")
        cells = {}
        for i, p in enumerate(P):
            cells.setdefault((int((p[0] + HALF) // 25), int((p[1] + HALF) // 25)), []).append(i)
        for key, ids in cells.items():
            verts, idx = [], []
            for i in ids:
                x, y = P[i]
                h = rng.uniform(0.14, 0.34) * (0.7 + 0.5 * w[i, 0])
                wd = h * 1.4
                a0 = rng.uniform(0, math.pi)
                c = cols[i]
                for a in (a0, a0 + math.pi / 2):
                    dx, dy = math.cos(a) * wd / 2, math.sin(a) * wd / 2
                    b = len(verts)
                    for (px, py, pz, u, v) in ((x - dx, y - dy, 0, 0, 0), (x + dx, y + dy, 0, 1, 0),
                                               (x + dx, y + dy, h, 1, 1), (x - dx, y - dy, h, 0, 1)):
                        verts.append((px, py, pz, 0, 0, 1, u, v, c[0], c[1], c[2], 1))
                    idx += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
            g = _mesh(f"grass{key}", np.array(verts), np.array(idx), [(V, 3, Geom.C_point), (N, 3, Geom.C_normal),
                                                                      (T, 2, Geom.C_texcoord), (C, 4, Geom.C_color)])
            g.reparentTo(root)
        mat = Material()
        mat.setBaseColor((1, 1, 1, 1))
        mat.setRoughness(0.9)
        root.setMaterial(mat)
        root.setTexture(tex)
        root.setTwoSided(True)
        root.setShaderInput("mode", 3)
        root.setShaderInput("wind_strength", 25.0)
        root.hide(SHADOW_MASK)
        self.grass = root

    # ------------------------------------------------------------------------------------------------ per frame
    def update(self, cam_pos, cam_forward, t):
        r = self.base.render
        r.setShaderInput("cam_pos", cam_pos)
        r.setShaderInput("wind_time", t)
        # the shadow area follows the camera (snapped to whole texels so shadows do not shimmer)
        f = Vec3(cam_forward.x, cam_forward.y, 0)
        if f.length() > 1e-3:
            f.normalize()
        c = Vec3(cam_pos.x, cam_pos.y, 0) + f * (self.shadow_size * 0.35)
        texel = self.shadow_size / 4096
        c = Vec3(round(c.x / texel) * texel, round(c.y / texel) * texel, 0)
        self.sun_np.setPos(c + self.sun_dir * 150)
        self.sun_np.lookAt(c)
