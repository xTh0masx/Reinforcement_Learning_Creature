"""Endemic Creatures - walk (fly) through the habitat while the Myrranox and the DC live in it.

Both creatures are NPCs run by their trained brains, live: the predator brains decide where to go, the walking brains
move every leg in MuJoCo, the 3D models follow the physics bodies bone by bone.

    python -m game.play                    # window
    python -m game.play --shot out.png --at 5,-20,6 --look 0,0,1 --wait 8     # screenshot without a window (tests)

Controls: W A S D move, mouse look, Space up, Ctrl down, Shift fast, F follow a creature, T time x1/x2/x4, P pause,
H help on/off, Esc frees the mouse (click into the window to steer again).
"""
import argparse
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
import mujoco  # noqa: E402,F401  (must be loaded before panda3d: DLL conflict on Windows)
from panda3d.core import loadPrcFileData  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shot", default=None, help="render offscreen and save a screenshot")
    ap.add_argument("--at", default="-6,-14,4")
    ap.add_argument("--look", default="0,0,1")
    ap.add_argument("--wait", type=float, default=6.0, help="seconds of simulation before the screenshot")
    ap.add_argument("--follow", default=None, help="screenshot: follow this creature id")
    ap.add_argument("--size", default="1600x900")
    ap.add_argument("--bench", type=int, default=0, help="screenshot mode: also time this many frames")
    ap.add_argument("--off", default="", help="tests: comma list of parts to leave out (shadows, grass, plants, sim, msaa)")
    args = ap.parse_args()
    w, h = args.size.split("x")
    loadPrcFileData("", f"""
        window-title Endemic Creatures
        win-size {w} {h}
        framebuffer-multisample {0 if "msaa" in args.off else 1}
        multisamples {0 if "msaa" in args.off else 4}
        sync-video true
        audio-library-name null
        gl-version 3 3
        textures-power-2 none
        {"window-type offscreen" if args.shot else ""}
    """)
    import gltf  # noqa: F401  (glTF loader plugin)
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import AntialiasAttrib, ClockObject, Vec3

    from game.creature_view import CreatureView
    from game.hud import Hud
    from game.player import Player
    from game.simulation import Simulation
    from game.world import World

    base = ShowBase()
    base.disableMouse()
    base.setBackgroundColor(0.6, 0.7, 0.8, 1)
    base.camLens.setFov(75)
    base.camLens.setNearFar(0.1, 1200)
    base.render.setAntialias(AntialiasAttrib.MMultisample)

    print("building the world ...", flush=True)
    sim = Simulation(seed=args.seed)
    sim.start(wait=False)               # separate process: loads the brains while the world is built
    world = World(base, sim.habitat, off=args.off.split(","))
    while sim.info is None:
        sim.poll(block=True)
    print("loading the creatures ...", flush=True)
    views, spec_of = {}, {}
    info = sim.info
    for cid in info["ids"]:
        sp = info["species"][cid]
        v = CreatureView(sp, info["specs"][cid], info["body_ids"], f"{cid}_", base.render)
        v.actor.setShaderInput("mode", 2)
        if sp == "dc":
            v.actor.setShaderInput("back_color", Vec3(0.16, 0.19, 0.21))
            v.actor.setShaderInput("belly_color", Vec3(0.55, 0.5, 0.42))
        else:
            v.actor.setShaderInput("back_color", Vec3(0.36, 0.18, 0.08))
            v.actor.setShaderInput("belly_color", Vec3(0.86, 0.68, 0.42))
        views[cid] = v
        spec_of[cid] = sp
    torso = {cid: views[cid].body_id["torso"] for cid in views}

    rat = world.model("street_rat")
    rats = []

    player = Player(base, world.terrain, pos=tuple(float(v) for v in args.at.split(",")))
    look = [float(v) for v in args.look.split(",")]
    base.camera.setPos(player.pos)
    base.camera.lookAt(*look)
    player.yaw, player.pitch = base.camera.getH(), base.camera.getP()
    if args.follow:
        player.follow = args.follow
        player.orbit_dist = 4.0 if "myrranox" in args.follow else 6.0
    hud = Hud(base, world, sim)
    show_help = [True]

    def toggle_follow():
        ids = list(views)
        if player.follow is None:
            player.follow = ids[1] if "dc" in ids[1] else ids[0]
        else:
            i = ids.index(player.follow)
            player.follow = ids[i + 1] if i + 1 < len(ids) else None
        player.orbit_yaw = 0.0
        player.orbit_dist = 4.0 if player.follow and "myrranox" in player.follow else 6.0

    def cycle_time():
        sim.speed = {1.0: 2.0, 2.0: 4.0}.get(sim.speed, 1.0)

    def toggle_help():
        show_help[0] = not show_help[0]
        hud.help.show() if show_help[0] else hud.help.hide()

    base.accept("f", toggle_follow)
    base.accept("t", cycle_time)
    base.accept("p", lambda: setattr(sim, "paused", not sim.paused))
    base.accept("h", toggle_help)
    base.accept("escape", lambda: player.capture(False))
    base.accept("mouse1", lambda: player.capture(True))
    base.accept("f3", base.setFrameRateMeter, [True])

    clock = ClockObject.getGlobalClock()
    elapsed = [0.0]

    def frame(task):
        dt = min(clock.getDt(), 0.1)
        elapsed[0] += dt
        sim.advance(dt)
        pose = sim.pose_at(sim.clock)
        heads = {}
        if pose is not None:
            xpos, xquat = pose
            for cid, v in views.items():
                v.update(xpos, xquat)
                heads[cid] = xpos[torso[cid]]
        status = sim.status_at(sim.clock)
        # prey: one rat per prey slot
        while len(rats) < len(status["prey"]):
            r = rat.copyTo(base.render)
            r.setScale(4.0)
            rats.append(r)
        for r, (p, hd, alive) in zip(rats, status["prey"]):
            if alive:
                r.show()
                r.setPos(p[0], p[1], 0.02 + 0.03 * abs(math.sin(elapsed[0] * 11 + p[0])))
                r.setH(math.degrees(hd) - 90)
            else:
                r.hide()
        fc = status["creatures"].get(player.follow) if player.follow else None
        if fc is not None and player.follow in heads:
            player.update(dt, heads[player.follow], fc["heading"])
        else:
            player.update(dt)
        world.update(base.camera.getPos(base.render), base.camera.getQuat(base.render).getForward(), elapsed[0])
        hud.update(status, heads, player)
        return task.cont

    base.taskMgr.add(frame, "frame")

    if args.shot:
        import time
        t0 = time.time()
        while sim.clock < args.wait and time.time() - t0 < args.wait * 4 + 30:
            base.taskMgr.step()
        for _ in range(3):
            base.taskMgr.step()
        from panda3d.core import Filename
        base.win.saveScreenshot(Filename.fromOsSpecific(os.path.abspath(args.shot)))
        if args.bench:
            if "sim" in args.off:
                sim.paused = True
            t1 = time.time()
            for _ in range(args.bench):
                base.taskMgr.step()
            print(f"bench: {args.bench / (time.time() - t1):.1f} fps")
        print("saved", args.shot, "sim clock", round(sim.clock, 1), "fps", round(clock.getAverageFrameRate(), 1))
        sim.stop()
        return
    base.run()


if __name__ == "__main__":
    main()
