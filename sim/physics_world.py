"""One MuJoCo world with the habitat (floor, walls, rocks from sim/habitat.json) and several creatures,
each walked by its own locomotion brain. Coordinates are habitat coordinates (metres, map centre = 0).

    world = PhysicsWorld([{"id": "myrranox_0", "species": "myrranox", "loco": "models/myr_legs_v1",
                           "pos": (-20, -15), "yaw": 0.0}, ...], Habitat.load())
    world.set_command("myrranox_0", 0.6, 0.0)
    state = world.step()          # advances 0.1 s (5 control steps of every brain)

Creatures collide with the floor, walls and rocks, not with each other.
"""
import math

import mujoco
import numpy as np

from sim.bodies import BODIES, build_world_mjcf
from sim.locomotion_env import CONTROL_DT, FRAME_SKIP, LocomotionEnv

NAV_DT = 0.1
SUBSTEPS = int(round(NAV_DT / CONTROL_DT))


def _yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class PhysicsWorld:
    def __init__(self, creatures, habitat, seed=0, passive=True):
        """passive=True: bodies with spring-damper tail / abdomen / head / arms (stage 1 "alive" bodies)."""
        from play_locomotion import Brain
        self.ids = [c["id"] for c in creatures]
        specs = [(BODIES[c["species"]](passive=passive), f'{c["id"]}_', tuple(c["pos"]), c.get("yaw", 0.0))
                 for c in creatures]
        self.model = mujoco.MjModel.from_xml_string(build_world_mjcf(specs, habitat))
        self.data = mujoco.MjData(self.model)
        self.brains, self.bodies, self.obs = {}, {}, {}
        for c in creatures:
            brain = c["brain"] if "brain" in c else Brain(c["loco"], c["species"])
            body = LocomotionEnv(c["species"], episode_seconds=1e9, version=getattr(brain, "version", 1),
                                 model=self.model, data=self.data, prefix=f'{c["id"]}_', passive=passive)
            self.brains[c["id"]], self.bodies[c["id"]] = brain, body
        self.t = 0.0
        self.falls = {i: 0 for i in self.ids}
        for c in creatures:
            self.obs[c["id"]], _ = self.bodies[c["id"]].reset(
                seed=seed, options={"command": (0.0, 0.0), "pos": tuple(c["pos"]), "yaw": c.get("yaw", 0.0)})
        # Let every body settle with the rest pose held.
        for _ in range(40):
            mujoco.mj_step(self.model, self.data)

    def set_command(self, cid, speed, turn_rate):
        self.bodies[cid].set_command(speed, turn_rate)

    def state(self, cid):
        b = self.bodies[cid]
        q = self.data.qpos[b.q0:b.q0 + 7]
        return {"pos": q[:2].copy(), "heading": _yaw(q[3:7])}

    def step(self, record_poses=False):
        """Advance NAV_DT seconds. Returns per creature: pos, heading, speed, turn_rate, energy (J), fell,
        and with record_poses the pose (LocomotionEnv.pose) after each of the 5 control steps (50 Hz)."""
        energy = {i: 0.0 for i in self.ids}
        poses = {i: [] for i in self.ids}
        fell = {i: False for i in self.ids}
        infos = {}
        for _ in range(SUBSTEPS):
            actions = {i: self.bodies[i].apply_action(self.brains[i](self.obs[i])) for i in self.ids}
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(self.model, self.data)
            for i in self.ids:
                self.obs[i], _, f, _, infos[i] = self.bodies[i].after_physics(actions[i])
                energy[i] += infos[i]["power"] * CONTROL_DT
                if f:
                    # Fell over: stand it up again where it is (counted, shown in the recording).
                    fell[i] = True
                    self.falls[i] += 1
                    s = self.state(i)
                    self.obs[i], _ = self.bodies[i].reset(options={
                        "command": tuple(self.bodies[i].command), "pos": tuple(s["pos"]), "yaw": s["heading"]})
                if record_poses:
                    poses[i].append(self.bodies[i].pose())
            self.t += CONTROL_DT
        out = {}
        for i in self.ids:
            s = self.state(i)
            out[i] = {**s, "speed": infos[i]["speed"], "turn_rate": infos[i]["turn_rate"],
                      "energy": energy[i], "fell": fell[i], "poses": poses[i]}
        return out

    def pose(self, cid):
        return self.bodies[cid].pose()
