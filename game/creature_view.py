"""A creature model in the game whose bones follow the MuJoCo bodies of its physics twin.

Every bone belongs to one physics body (torso, leg segments, spine, neck / head, tail, abdomen). At rest the bodies are
unrotated and the model is in the creature frame, so a bone's world matrix is the body's motion since rest applied to
the bone's rest matrix:  W_bone(t) = M_body(t) * T(-o_body) * W_bone_rest,  o_body = the body's rest origin.
"""
import json
import os

import numpy as np
from direct.actor.Actor import Actor
from panda3d.core import Filename, LMatrix4f, OmniBoundingVolume, Shader, ShaderAttrib

from sim.bodies import hull_origins

HERE = os.path.dirname(os.path.abspath(__file__))


def body_bone_pairs(spec):
    pairs = [("torso", "torso")]
    for leg in spec.legs:
        pairs += [(f"{leg.name}_s{i}", f"{leg.name}_s{i}") for i in range(3)]
    if spec.rear is not None:
        pairs.append((f"{spec.rear.name}_rear", spec.rear.bone))
    for ch in spec.passive:
        pairs += [(f"{ch.name}_s{i}", b) for i, b in enumerate(ch.bones)]
    return pairs


def _np(m):
    """Panda (row-vector) matrix -> numpy column-vector matrix."""
    return np.array([[m.getCell(r, c) for c in range(4)] for r in range(4)]).T


def _panda(a):
    return LMatrix4f(*a.T.ravel().tolist())


class CreatureView:
    def __init__(self, species, spec, body_ids, prefix, parent):
        """body_ids: MuJoCo body name -> index in the world (sent by the simulation process)."""
        path = os.path.join(HERE, "assets", "creatures", f"{species}.glb")
        self.actor = Actor(Filename.fromOsSpecific(path).getFullpath())
        self.actor.reparentTo(parent)
        # the bones move the mesh far from its rest place: never cull it by its (rest pose) bounds
        self.actor.node().setBounds(OmniBoundingVolume())
        self.actor.node().setFinal(True)
        # skin on the graphics card (the DC has ~200k faces - on the CPU that cost two thirds of the frame rate)
        sh = os.path.join(HERE, "shaders")
        shader = Shader.load(Shader.SL_GLSL, vertex=Filename.fromOsSpecific(os.path.join(sh, "lit_skinned.vert")),
                             fragment=Filename.fromOsSpecific(os.path.join(sh, "lit.frag")))
        self.actor.setAttrib(ShaderAttrib.make(shader).setFlag(ShaderAttrib.F_hardware_skinning, True), 10)
        with open(os.path.join(HERE, "assets", "creatures", f"{species}_bones.json")) as f:
            self.parent_of = json.load(f)
        origins = hull_origins(spec)
        self.bones, self.ctrl, self.rest_world, self.body_id, self.origin = [], {}, {}, {}, {}
        rest_local = {}
        for j in self.actor.getJoints():
            rest_local[j.getName()] = _np(j.getDefaultValue())

        def world_rest(b):
            if b not in self.rest_world:
                p = self.parent_of.get(b)
                self.rest_world[b] = (world_rest(p) if p else np.eye(4)) @ rest_local[b]
            return self.rest_world[b]
        for body, bone in body_bone_pairs(spec):
            if bone not in rest_local:
                continue
            world_rest(bone)
            self.bones.append(bone)
            self.body_id[bone] = body_ids[prefix + body]
            o = origins.get(bone if bone in origins else body, np.zeros(3))
            self.origin[bone] = np.asarray(o, float)
            self.ctrl[bone] = self.actor.controlJoint(None, "modelRoot", bone)
        # parents first, so a child's local matrix uses its parent's new world matrix
        depth = lambda b: 0 if not self.parent_of.get(b) else 1 + depth(self.parent_of[b])
        self.bones.sort(key=depth)

    def update(self, xpos, xquat):
        """xpos (nbody, 3) and xquat (nbody, 4, w x y z) of the MuJoCo world - live data or an interpolated snapshot."""
        world = {}
        for bone in self.bones:
            i = self.body_id[bone]
            w, x, y, z = xquat[i]
            m = np.eye(4)
            m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
            m[:3, 3] = xpos[i]
            t = np.eye(4)
            t[:3, 3] = -self.origin[bone]
            world[bone] = m @ t @ self.rest_world[bone]
        for bone in self.bones:
            p = self.parent_of.get(bone)
            pw = world.get(p) if p else None
            if p and pw is None:
                pw = self.rest_world[p]
            local = np.linalg.inv(pw) @ world[bone] if p else world[bone]
            self.ctrl[bone].setMat(_panda(local))
