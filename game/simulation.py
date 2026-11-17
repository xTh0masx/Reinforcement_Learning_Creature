"""The living habitat behind the game: both predator brains (where to go) and both walking brains (legs, MuJoCo) run in
real time, exactly as in record_world.py - in a separate process, so the simulation and the picture do not share one
Python interpreter (in one process they slowed each other down to ~15 frames per second).

The simulation process sends, per navigation step (0.1 s), the five 50 Hz body snapshots (positions / orientations of
all MuJoCo bodies) and the creatures' needs, activity and the prey. The game interpolates the snapshots at its own clock.
A creature that starves ends the day: after a short pause a new day starts (next seed).
"""
import dataclasses
import multiprocessing as mp
import queue
import time
from collections import deque

import numpy as np

HABITAT = "sim/habitat_no_rocks.json"      # the habitat the final showcase runs in (no rocks inside the walls)
LEAD = 0.3                                  # s the simulation runs ahead of the picture (room to interpolate)

# Game-only tuning (training, showcase and evaluations keep the original values).
# Myrranox: its walking brain v10 tops out at ~1.2 m/s, the navigation limit of 1.2 m/s only got ~1.05 m/s out of it;
# asking for 1.5 m/s gets the full ~1.2 m/s. It also recovers 2.5x faster from fatigue.
GAME_SPECIES = {"myrranox": {"max_speed": 1.5, "fatigue_recovery": 0.025, "fatigue_rest_recovery": 0.075}}
GAME_PREY = {"wander_speed": 0.10, "flee_speed": 0.22}     # was 0.15 / 0.35 m/s


# ------------------------------------------------------------------------------------------------- simulation process
def _status(env, t):
    return {"t": t,
            "creatures": {c.id: {"species": c.species.name, "pos": c.pos.copy(), "heading": c.heading,
                                 "hunger": c.hunger, "fatigue": c.fatigue, "activity": c.activity,
                                 "prey": c.stats["prey_caught"], "starving": c.starve_steps > 0}
                          for c in env.creatures},
            "prey": [(p.copy(), h, a) for p, h, a in zip(env.prey_pos, env.prey_heading, env.prey_alive)]}


def _engine(out, clock, speed, paused, stop, seed):
    import mujoco
    import torch
    torch.set_num_threads(2)
    import record_world
    from sim.creature_env import STARVE_LIMIT_STEPS
    from sim.habitat import Habitat
    from sim.physics_world import PhysicsWorld
    habitat = Habitat.load(HABITAT)
    generation = 0
    while not stop.value:
        generation += 1
        env, brain, obs = record_world.make_env("myrranox", "dc", seed, HABITAT)
        env.habitat.prey.update(GAME_PREY)
        for c in env.creatures:
            if c.species.name in GAME_SPECIES:
                c.species = dataclasses.replace(c.species, **GAME_SPECIES[c.species.name])
        world = PhysicsWorld([{"id": c.id, "species": c.species.name, "loco": record_world.WALKERS[c.species.name],
                               "pos": tuple(c.pos), "yaw": c.heading} for c in env.creatures],
                             habitat, seed=seed, passive=True)
        for c in env.creatures:                       # the walking body must accept the faster commands too
            body = world.bodies[c.id]
            body.top_speed = max(body.top_speed, c.species.max_speed)
        names = {mujoco.mj_id2name(world.model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(world.model.nbody)}
        out.put(("day", generation, {"ids": world.ids,
                                     "species": {i: world.bodies[i].species_name for i in world.ids},
                                     "specs": {i: world.bodies[i].body for i in world.ids},
                                     "body_ids": names},
                 (0.0, world.data.xpos.copy(), world.data.xquat.copy()), _status(env, 0.0)))
        t = 0.0
        while not stop.value:
            if paused.value or t > clock.value + LEAD * max(1.0, speed.value):
                time.sleep(0.003)
                continue
            action = brain(obs)
            others = env.other_actions()
            for c in env.creatures:
                v, w, _ = env.command(action if c is env.ego else others[c.id], c)
                world.set_command(c.id, v, w)
            t0, frames = world.t, []
            state = world.step(on_control=lambda tt: frames.append(
                (t + tt - t0, world.data.xpos.copy(), world.data.xquat.copy())))
            obs, _, _, _, _ = env.step(action, bodies=state, actions=others)
            t += 0.1
            out.put(("step", frames, _status(env, t)))
            dead = [c for c in env.creatures if c.starve_steps >= STARVE_LIMIT_STEPS]
            if dead:
                name = {"myrranox": "Der Myrranox", "dc": "Der DC"}[dead[0].species.name]
                out.put(("message", f"{name} ist verhungert - ein neuer Tag beginnt ..."))
                time.sleep(4.0)
                out.put(("message", ""))
                seed += 1
                break


# ------------------------------------------------------------------------------------------------- game side
class Simulation:
    def __init__(self, seed=0):
        from sim.habitat import Habitat
        self.habitat = Habitat.load(HABITAT)
        ctx = mp.get_context("spawn")
        self._q = ctx.Queue()
        self._clock = ctx.Value("d", 0.0, lock=False)
        self._speed = ctx.Value("d", 1.0, lock=False)
        self._paused = ctx.Value("b", 0, lock=False)
        self._stop = ctx.Value("b", 0, lock=False)
        self.proc = ctx.Process(target=_engine, args=(self._q, self._clock, self._speed, self._paused, self._stop, seed),
                                daemon=True)
        self.frames = deque(maxlen=200)     # (t, xpos, xquat) at 50 Hz
        self.statuses = deque(maxlen=40)    # 10 Hz states
        self.status = None
        self.info = None                    # ids, species, specs, body_ids of the current day
        self.generation = 0
        self.message = ""
        self.clock = 0.0
        self.t = 0.0                        # newest simulated time received

    speed = property(lambda self: self._speed.value, lambda self, v: setattr(self._speed, "value", float(v)))
    paused = property(lambda self: bool(self._paused.value), lambda self, v: setattr(self._paused, "value", int(v)))

    def start(self, wait=True):
        """Start the simulation process; wait=True blocks until the first day is set up."""
        self.proc.start()
        while wait and self.info is None:
            self.poll(block=True)

    def stop(self):
        self._stop.value = 1

    def poll(self, block=False):
        """Take everything the simulation process has sent. Returns True when a new day started."""
        new_day = False
        while True:
            try:
                msg = self._q.get(timeout=120) if block else self._q.get_nowait()
            except queue.Empty:
                return new_day
            block = False
            if msg[0] == "day":
                _, self.generation, self.info, frame, self.status = msg
                self.frames.clear()
                self.frames.append(frame)
                self.statuses.clear()
                self.statuses.append(self.status)
                self.clock = self.t = 0.0
                self._clock.value = 0.0
                new_day = True
            elif msg[0] == "step":
                self.frames.extend(msg[1])
                self.status = msg[2]
                self.statuses.append(msg[2])
                self.t = msg[2]["t"]
            elif msg[0] == "message":
                self.message = msg[1]

    def advance(self, dt):
        """Move the game clock (called every frame); never past what has been simulated."""
        new_day = self.poll()
        if not self.paused:
            self.clock = min(self.clock + dt * self.speed, self.t)
        self._clock.value = self.clock
        return new_day

    def pose_at(self, t):
        """Interpolated (xpos, xquat) of every MuJoCo body at game time t."""
        fr = self.frames
        if not fr:
            return None
        if t <= fr[0][0]:
            return fr[0][1], fr[0][2]
        for i in range(len(fr) - 1, 0, -1):
            a, b = fr[i - 1], fr[i]
            if a[0] <= t <= b[0]:
                u = (t - a[0]) / max(b[0] - a[0], 1e-9)
                q0, q1 = a[2], b[2]
                q1 = np.where((np.sum(q0 * q1, axis=1) < 0)[:, None], -q1, q1)
                q = q0 + u * (q1 - q0)
                q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
                return a[1] + u * (b[1] - a[1]), q
        return fr[-1][1], fr[-1][2]

    def status_at(self, t):
        """The 10 Hz state at game time t, with prey and creature positions interpolated (prey move smoothly)."""
        st = list(self.statuses)
        if not st:
            return self.status
        a = st[0]
        for b in st[1:]:
            if b["t"] >= t:
                break
            a = b
        else:
            return st[-1]
        if b is a or b["t"] <= a["t"]:
            return a
        u = min(max((t - a["t"]) / (b["t"] - a["t"]), 0.0), 1.0)
        prey = [(pa + u * (pb - pa), hb, al) for (pa, _, _), (pb, hb, al) in zip(a["prey"], b["prey"])]
        creatures = {cid: {**c, "pos": a["creatures"][cid]["pos"] * (1 - u) + c["pos"] * u}
                     for cid, c in b["creatures"].items()}
        return {**b, "prey": prey, "creatures": creatures}
