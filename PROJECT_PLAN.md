# Endemic Creatures – Reinforcement Learning Project Plan

> **For a new chat:** this file is the complete context of the project so far. All decisions below are final unless stated as open. Current status and next tasks: section 13. Walking (locomotion) brains: section 14.

Owner: Thomas Bartels · Status as of 25 Sep 2026 · Local repo: `PycharmProjects/Reinforcement_Learning_Creature`

> **Change 25 Sep 2026:** walking is now **learned by RL too** (owner decision). Each species gets two brains:
> a **navigation brain** (`CreatureEnv`, 2D, where to go, section 8) and a **locomotion brain**
> (`LocomotionEnv`, MuJoCo physics, how to move the legs to follow the navigation brain's speed/turn command).
> MuJoCo is therefore added to the tools. See section 14 for the locomotion design.

---

## 1. Goal

Two creature species from a Blender scene live in one habitat and move around like **endemic creatures**: they stay in their home range, hunt prey, rest, and defend their territory against each other.

- **Every decision** (where to go, how fast, when to stop) is learned with **reinforcement learning in Python using Gymnasium + Stable-Baselines3 (PPO), developed in PyCharm**.
- **Leg movement is learned by RL** in a MuJoCo physics body (locomotion brain) and played back on rigs in Blender.
- Hand-in: **presentation, written paper, and code**.

---

## 2. Tools used (only these)

| Tool | Where | Used for |
| --- | --- | --- |
| Python 3.10+ (conda environment) | PyCharm | All RL code |
| **Gymnasium** | PyCharm | The habitat as an RL environment (`reset`, `step`, observation/action spaces) |
| **Stable-Baselines3** (PPO) | PyCharm | Training the creature brains |
| NumPy | PyCharm | Maths / simulation |
| matplotlib | PyCharm | Live top-down debug viewer, heatmaps, paper figures |
| TensorBoard | Browser | Training curves (`pip install tensorboard`) |
| **MuJoCo** (`pip install mujoco`) | PyCharm | Physics bodies for the locomotion brains |
| Pillow | PyCharm | GIFs of locomotion evaluations |
| Blender + built-in Python (`bpy`) | Blender | Rigging, trajectory/pose import, rendering |
| Git (+ Git LFS for `.blend`/textures) | PyCharm | Version control, code hand-in |

Already installed: Gymnasium, Stable-Baselines3.

**Not used:** Unity, ML-Agents, Godot, Isaac Lab, PettingZoo. Gymnasium and SB3 run only in PyCharm; Blender only reads a CSV file, so nothing is installed into Blender's Python.

---

## 3. Decisions

| Topic | Decision |
| --- | --- |
| Tools | Python (PyCharm) + Gymnasium + Stable-Baselines3, and Blender. No game engine. |
| Creature sizes | Myrranox 1.5 m body length; DC_SKIN_RAW 1.0 m shoulder height |
| Relationship | Both are **predators and rivals**: each hunts small prey animals and defends its territory; they do **not** hunt each other |
| Individuals | Max 2 per species → final scene: 2 Myrranox + 2 DC |
| Demo | 30 fps, third-person follow camera from behind, switchable between creatures |
| Hand-in | Presentation, written paper, code |

**Still open:** deadline, presentation length, paper format (page limit, citation style).

---

## 4. The Blender assets (current state)

| | Myrranox | DC_SKIN_RAW |
| --- | --- | --- |
| Body plan | Arthropod: 3 torso segments + abdomen, 6 legs (3 pairs), mandibles | Hunched quadruped with claws, teeth, tongue |
| Mesh | 21 low-poly parts, ~40k faces, parented to empty `Myrranox_low_2_0` (scale 0.01, cm import) | 13 parts, ~1.47M faces (body ~970k), parented to empty `DC_SKIN_RAW` (scale 0.579) |
| Legs | Split per segment: `TopArms/MidArms/LowerArms` → `long…arms` → `…claw`; each mesh holds left+right (2 loose parts) | Fused into the body mesh; only claws separate |
| Materials | `T_UpperBody`, `T_LowerBody`, `T_Abdomen` (image textures, UVs present) | One default `lambert4` (Maya export) |
| Rig / animation | `Myrranox_rig` (25 Sep): 19 bones, legs split `_L`/`_R`, parts parented rigidly; scaled to 1.5 m | `DC_rig` (25 Sep): 19 bones incl. static neck/head/tail/arms; 76k-face `DC_body_proxy` with automatic weights; scaled to 1.3 m tall |

Scene: `blender/habitat_showcase.blend` (EEVEE) – habitat from `habitat.json` + sun (`Habitat` collection), roots, prey
and follow cameras (`Creatures` collection), both rigs. Myrranox image textures are missing (point to `D:\FINAL MODUL\…`).
Blender MCP is connected and can inspect/edit the scene.

---

## 5. Architecture

```
PyCharm (Python)                                              Blender
┌───────────────────────────────────────────────────┐        ┌──────────────────────────────────┐
│ Navigation brain: CreatureEnv (2D habitat, needs)  │        │ import_trajectories.py           │
│        ↓ speed + turn command (10 Hz)              │        │   root empty ← 10 Hz CSV         │
│ Locomotion brain: LocomotionEnv (MuJoCo body)      │  CSVs  │   rig torso ← body pose (50 Hz)  │
│        ↓ joint targets (50 Hz) → real body pose    │ ─────► │   leg bones ← joint angles       │
│ both trained with SB3 PPO; record_physics.py       │        │ follow cameras → render (30 fps) │
└───────────────────────────────────────────────────┘        └──────────────────────────────────┘
```

- The **navigation brain** outputs a movement command every 0.1 s: **forward speed** and **turn rate**; the
  **locomotion brain** walks the physics body to follow it (section 14).
- The simulation runs at **10 steps/s** (dt = 0.1 s). Blender resamples to 30 fps (3 frames per step).
- The link between both halves is `recordings/trajectories_<date>.csv` with columns:
  `time, id, species, x, y, heading, speed, turn_rate, activity` (activity = walk / idle / eat / rest / confront),
  plus per physics creature `<csv>_<id>_pose.csv` at 50 Hz: `time, x, y, z, qw, qx, qy, qz, <joint angles>, <feet xyz>`.
- The map is stored once in `sim/habitat.json` and used by both Python and Blender (`build_habitat.py`).

---

## 6. Habitat (2D in Python)

About 100 × 100 m, top-down.

| Element | In Python (`habitat.json`) | In Blender |
| --- | --- | --- |
| Boundary | Rectangle 100 × 100 m | Cliffs/walls |
| Home zone Myrranox | Polygon or circle | Sand/rock area |
| Home zone DC | Polygon or circle | Vegetation area |
| Overlap zone | Intersection / shared water source | Water / clearing |
| Prey | Small points that wander slowly and flee when a predator is within 5 m; caught prey respawns after 60–120 s | Small prey creatures moved by the same recording |
| Rest spots | Points with a radius | Burrows, shade |
| Obstacles | Circles (rocks) | Rocks at the same positions |

Prey and spawn positions are randomised every episode during training. SB3 runs 8 parallel environments (`make_vec_env(..., n_envs=8)`).

---

## 7. Movement model (Python, per species)

The brain's actions are in [−1, 1] and get scaled to species limits, with capped acceleration and turning so every path is animatable.

| Limit | Myrranox | DC_SKIN_RAW |
| --- | --- | --- |
| Walk speed (reference for `Walk` cycle) | 0.6 m/s | 0.8 m/s |
| Max speed | 1.2 m/s | 2.5 m/s (`Trot`) |
| Backwards | up to 0.3 m/s | not allowed |
| Max turn rate | 90°/s, can turn on the spot | 60°/s, only while moving |
| Max acceleration | 1.0 m/s² | 1.5 m/s² |

Update per step: `heading += turn_rate * dt`; `pos += speed * dt * (cos(heading), sin(heading))`.

---

## 8. The RL brain (Gymnasium env + PPO)

### Needs (per creature, rates randomised ±20 % per individual)

| Need | Rises when | Falls when | Effect at 100 % |
| --- | --- | --- | --- |
| Hunger | Constantly, faster when moving fast | Catching prey | Penalty every step |
| Fatigue | Moving; strongly at high speed | Standing still, faster on a rest spot | Max speed capped at 30 % |
| Alertness | Rival within 10 m, strongest in own territory | Rival gone, time | Used in rewards |

### Observation space — `Box(-1, 1, shape=(27,))`, egocentric

| Group | Values | Count |
| --- | --- | --- |
| Needs | hunger, fatigue, alertness | 3 |
| Own motion | speed, turn rate | 2 |
| Nearest prey | direction (x, y), distance | 3 |
| Nearest rest spot | direction (x, y), distance | 3 |
| Home zone | direction to centre (x, y), distance, inside yes/no | 4 |
| Nearest other creature | direction (x, y), distance (1 = none seen), rival yes/no | 4 |
| Obstacle rays | 8 rays over 360°, distance to nearest rock/boundary | 8 |
| **Total** | | **27** |

### Action space — `Box(-1, 1, shape=(2,))`
forward speed, turn rate (scaled by the movement model).

### Rewards

| Event | Reward |
| --- | --- |
| Catches prey while hunger > 50 % | +1.0 × hunger |
| Rests on a rest spot while fatigue > 50 % | +0.02 / step |
| Each step outside home zone | −0.01 × distance factor |
| Each step inside the rival's home zone | −0.03 (extra) |
| Rival inside own zone: moving towards it, within 8 m | +0.02 / step |
| Rival leaves own zone after being confronted | +0.5 |
| Closer than 2 m to the rival (fight risk) | −0.1 / step |
| Hunger at 100 % | −0.02 / step |
| Hits obstacle or boundary | −0.5 |

Episode = one simulated day of 10 min (6,000 steps); ends early if hunger stays at 100 % for 60 s.

### Training order
1. One creature alone (prey, rest spots, home zone). Myrranox first, then DC.
2. Two creatures of the same species sharing a territory and competing for prey.
3. Add the rival species as a **scripted** creature that sometimes enters the territory.
4. Replace the scripted rival with the other species' **frozen trained brain**; alternate training rounds (3–4×). Stays within plain Gymnasium + SB3.
5. Final recording: 2 Myrranox + 2 DC.

### Training script (starting values)

```python
# train.py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from sim.creature_env import CreatureEnv

env = make_vec_env(CreatureEnv, n_envs=8, env_kwargs={"species": "myrranox"})
env = VecFrameStack(env, n_stack=3)
env = VecNormalize(env, norm_reward=True)

model = PPO(
    "MlpPolicy", env,
    n_steps=2048, batch_size=512, n_epochs=10,
    learning_rate=3e-4, gamma=0.995, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.005,
    policy_kwargs=dict(net_arch=[128, 128]),
    tensorboard_log="runs/", verbose=1,
)
model.learn(total_timesteps=3_000_000, tb_log_name="myr_brain_v1",
            callback=CheckpointCallback(500_000, "checkpoints/", name_prefix="myr_brain_v1"))
model.save("models/myr_brain_v1")
env.save("models/myr_brain_v1_vecnormalize.pkl")
```

- CPU is enough; expect ~2–5M steps per species for stage 1 (≈1–4 h).
- `tensorboard --logdir runs` → watch `rollout/ep_rew_mean`.
- Run `gymnasium.utils.env_checker.check_env(CreatureEnv())` before long runs.
- Watch the matplotlib viewer at every checkpoint to catch reward loopholes.
- Change one thing per run.

---

## 9. Blender side

> **Changed 25 Sep 2026:** walking is learned (section 14), so hand-made walk cycles, stride-based phase and NLA
> blending (playback steps 2–4 below, `Walk`/`Trot` actions) are **superseded** by physics playback: the torso bone
> follows the recorded body pose, leg bones the recorded joint angles. Done so far: `rig_myrranox.py`, `rig_dc.py`
> (decimated proxy instead of retopology for now), `import_trajectories.py` steps 1 and 6 + pose playback.

**Asset prep – Myrranox:** apply 0.01 scale, scale to 1.5 m length; split limbs into `_L`/`_R` (Separate by Loose Parts); origins at joints; armature with `root` bone + body chain + 6 legs × 3 bones; parent mesh parts rigidly to bones (Ctrl+P → Bone); leg IK to foot targets; actions `Walk` (alternating tripod gait, 1 s loop, in place), `Idle`, `Eat`; measure stride length.

**Asset prep – DC_SKIN_RAW:** scale to 1.0 m shoulder height; retopology to 10–20k faces; UV + bake normal/AO from the sculpt; new material; armature (root, spine, neck, head, jaw, 4 legs × 3, optional tail) with weights; leg IK; actions `Walk` (lateral-sequence), `Trot`, `Idle`, `Eat` with stride lengths.

**Playback (`import_trajectories.py`):**
1. Key `root` X/Y location + Z rotation per frame from the CSV (10 Hz → 30 fps).
2. Ground following: raycast terrain height, tilt body to terrain normal, Shrinkwrap on foot IK targets.
3. No foot sliding: drive the walk cycle by distance, `phase = (phase + speed*dt/stride_length) mod 1`.
4. NLA blending Idle/Walk/Trot/Eat by speed and activity.
5. Bend spine/waist slightly into turns; mandibles/jaw open when eating; head looks at a nearby rival.
6. Cameras: one third-person follow camera per creature (empty following root + Damped Track), switched via timeline markers (Ctrl+B) or active camera.

---

## 10. Phases

| # | Phase | Main tasks | Done when | Est. | Status (25 Sep) |
| --- | --- | --- | --- | --- | --- |
| 0 | Setup | Gymnasium + SB3 installed ✔; add tensorboard; repo structure; PPO on `CartPole-v1` as smoke test | CartPole reaches 500 | 1 day | ✔ done (500.0) |
| 1 | Habitat sim | `CreatureEnv`: 2D map, one creature, prey, home zone, needs, matplotlib viewer; `check_env` | Random-action creature runs in viewer | 1 week | ✔ done |
| 1b | Locomotion (new) | Physics bodies, `LocomotionEnv`, train walking for both species | Follows speed/turn commands without falling | – | v1 done: both walk, no falls; turning weak → v2 |
| 2 | Brain – one species | Train Myrranox alone, then DC; tune rewards | Endemic rules met for one creature | 1–2 weeks | `train.py` ready, not trained |
| 3 | Blender – Myrranox | Rig, import script (walk cycles superseded by physics playback) | Walks a recorded path without foot sliding | 1–2 weeks | rig + playback done; waiting for final brain |
| 4 | Blender – DC | Retopo, bake, rig (cycles superseded) | Same as phase 3 | 2–3 weeks | proxy rig + playback done; retopo/bake open |
| 5 | Two species | 2 + 2 creatures; prey competition + territory rewards | Both species meet all rules together | 2–3 weeks | open |
| 6 | Showcase scene | Terrain from `habitat.json`, props, 4 follow cameras at 30 fps, lighting | Full scene plays from one recording | 1 week | simple habitat + follow cameras exist |
| 7 | Hand-in | Metrics, heatmaps, demo video, paper, slides, code README | Paper, presentation, code ready | 2–3 weeks | locomotion progress material being collected |

Total ≈ 9–13 weeks. Python side (1–2) and Blender side (3–4) can run in parallel.

---

## 11. Repository structure

```
Reinforcement_Learning_Creature/
├─ sim/
│  ├─ habitat.json / habitat.py   # the map (shared with Blender) + geometry helpers
│  ├─ species.py                  # per-species limits and need rates
│  ├─ creature_env.py             # navigation env (Gymnasium)
│  ├─ viewer.py                   # matplotlib top-down view
│  ├─ bodies.py                   # MuJoCo bodies measured on the Blender models
│  └─ locomotion_env.py           # locomotion env (Gymnasium + MuJoCo)
├─ train.py / train_locomotion.py # PPO training
├─ play_locomotion.py             # evaluate one locomotion brain (+ GIF)
├─ locomotion_progress.py         # per-checkpoint clips, progress GIF, metrics, learning curves
├─ record.py / record_physics.py  # recordings for Blender
├─ check_env.py, cartpole_smoke_test.py
├─ evaluate.py                    # (planned) metrics + heatmaps for section 12
├─ models/  recordings/
├─ runs/  checkpoints/  logs/     # gitignored
├─ blender/
│  ├─ habitat_showcase.blend (+ _before_rig.blend backup)
│  └─ scripts/build_habitat.py, import_trajectories.py, rig_myrranox.py, rig_dc.py, rig_common.py
├─ docs/experiments.csv           # one row per training run
├─ docs/locomotion_training_log.md
├─ docs/figures/                  # curves, GIFs, contact sheets (GIFs via Git LFS)
├─ requirements.txt
└─ README.md
```

Rules: run ID `<creature>_brain_v<N>_<change>` (navigation) / `<creature>_legs_v<N>` (locomotion); one row per run in `docs/experiments.csv` (run ID, date, commit, steps, final reward, verdict, notes); change one thing per run; don't change CSV columns or observation layout once something depends on them.

---

## 12. Evaluation and deliverables

Pass all checks over 10 episodes × 10 min, fixed seeds, deterministic brains:

| Check | Target |
| --- | --- |
| Time inside own zone | ≥ 90 % |
| Time with hunger at 100 % | < 5 % |
| Prey caught per creature per 10 min | ≥ 3 |
| Rest-spot visits when fatigue > 50 % | ≥ 1 per 10 min |
| Obstacle/boundary hits per 10 min | ≤ 1 |
| Rival in own territory is confronted and leaves | ≥ 80 % of intrusions |
| Individuals take different paths | No identical trajectories |
| Blender playback | No foot sliding, jitter or animation snapping |

| Deliverable | Content |
| --- | --- |
| Paper | Intro/goal; related work (RL for agent behaviour, Gymnasium, PPO); method (habitat as MDP: observations, actions, rewards, needs, movement model); training setup; results (curves, metrics, heatmaps); discussion & limitations (simplified physics bodies, DC proxy mesh); conclusion |
| Presentation | ~10–15 slides + 1–2 min of demo |
| Code | Repo with README (setup + reproduce), requirements.txt, env, train/record/evaluate scripts with fixed seeds, models, Blender scripts |
| Demo video | 2–3 min, 30 fps, third-person views switching between the 4 creatures |

---

## 13. Current status and next tasks (25 Sep 2026)

Done: phases 0–1 (sim, viewer, `check_env`, CartPole 500); physics bodies + `LocomotionEnv`; Myrranox and DC rigs in
`blender/habitat_showcase.blend`; physics pose playback in Blender (verified to 0.01 mm); progress/presentation
tooling for the walking training.

Next:
1. ✔ `myr_legs_v1` / `dc_legs_v1` trained (4M steps each), released to `models/`, progress material and
   whole-session videos (`docs/videos/`) made. Results: `docs/locomotion_training_log.md`.
2. Turning: `*_legs_v2` — turn reward weight 0.5 → 1.0 and more turn-only/backward commands, resumed from v1.
   One change per run.
3. Train the navigation brains (`train.py`, stage 1) for both species; check against section 12.
4. `record_physics.py` with trained navigation + locomotion brains → import into Blender → render demo.
5. Missing Myrranox textures (`D:\FINAL MODUL\WORKFILES\SubstancePainter\T_*_BC.png`) must be relinked.
6. Open: deadline, presentation length, paper format.

---

## 14. Locomotion brains (added 25 Sep 2026)

| | Myrranox | DC_SKIN_RAW |
| --- | --- | --- |
| Physics body (`sim/bodies.py`) | Hexapod, 8 kg, 6 legs × 3 hinges (coxa yaw, femur lift, claw lift) = 18 actuators; joint points measured on the mesh | Quadruped, 64 kg, 4 legs × 4 hinges (hip abduction + pitch, knee, ankle) = 16 actuators; symmetric approximation from a side render |
| Joint angle 0 | Pose of the model in the `.blend` | Same |
| Blender rig | `Myrranox_rig`: legs split into `_L`/`_R`, parts parented rigidly to bones on the physics joint points | `DC_rig`: 76k-face decimated proxy (`DC_body_proxy`) with automatic weights; claws/teeth/eyes parented rigidly; original sculpt hidden |

- **Env** `sim/locomotion_env.py`: 50 Hz control, position actuators around the rest pose. Observation: gravity, body velocity, angular velocity, height, joint angles/velocities, last action, command. Reward: track commanded forward speed and turn rate; penalties for vertical bounce, roll/pitch rate, tilt, action rate, torque, foot slip, body touching the ground, joint limits; −10 on fall. Commands are resampled every 2–5 s within the section 7 limits (15 % standing).
- **Reward weights** (`sim/locomotion_env.py`): speed tracking 2.0 · exp(−err²/σ²) with σ = ½ max speed; turn tracking 0.5 with σ = ½ max turn rate; penalties vz −0.5, roll/pitch rate −0.05, tilt −1, action rate −0.02, torque −0.02, foot slip −0.1 (capped), torso contact −1, joint limits −0.2. Penalties are multiplied by a curriculum factor that ramps 0.2 → 1 over the first 40 % of training (`PenaltyCurriculum` in `train_locomotion.py`).
- **Training** `train_locomotion.py` (PPO, 8 × SubprocVecEnv, VecNormalize, net 256×256 ELU, 4M steps ≈ 1 h per species on 8 cores). Checkpoints at step 0 and every 250k steps are kept for the presentation.
- **Evaluation** `play_locomotion.py`: fixed command script — walk, full speed, turn left/right, (turn in place, backwards), stop, restart. `locomotion_progress.py` runs it on every checkpoint (clips, progress GIF, contact sheet, metrics, learning curves). History and lessons: `docs/locomotion_training_log.md`.
- **Coupling** `record_physics.py`: every 0.1 s the navigation brain's action → speed/turn command → 5 locomotion steps → the body's real pose goes back into `CreatureEnv` (`step(action, body=...)`). Rocks/walls exist only in the habitat; on a hit the body is pushed back. Outputs the 10 Hz CSV plus a 50 Hz `<csv>_<id>_pose.csv` (body position, quaternion, joint angles, feet).
- **Blender** `import_trajectories.py` keys the rig from the pose file (torso bone = body pose, leg bones = joint angles, resampled to 30 fps). Verified: bone tips match the physics feet to 0.01 mm.
- `record.py --policy heuristic` and `record_physics.py --loco sine` are debugging stand-ins, not RL.
- **Results v1 (4M steps each, 25 Sep 2026):** Myrranox walk 0.61/0.6 m/s, full speed 1.19/1.2, left turn 56/54 °/s,
  right turn 31/54 °/s, no turning on the spot, no backwards. DC walk 0.76/0.8 m/s, trot 2.48/2.5 m/s, turns ≈ 25 % of
  the command. Both stop, restart and never fell. Videos of the whole training: `docs/videos/<run>_training.mp4`.
- **Lessons:** (1) a penalty that is large for the untrained policy's jitter (foot slip) swamps the task reward;
  (2) a narrow tracking reward plus full penalties makes standing still a local optimum; (3) the policy can learn to
  walk once moving before it learns to start from standstill — the evaluation checks both; (4) turning is learned last
  and least (small turn weight, fewer turn commands).

---

## 15. Territory, prey and rival rules (decided 25 Sep 2026, owner)

Both predators may **invade** the other's territory; food is scarce at home, so raids are part of survival.

| Rule | Decision |
| --- | --- |
| Prey eaten in the **own** territory | **No respawn** – the home stock shrinks |
| Prey eaten in the **rival** territory | A new prey spawns **in the eater's own territory** (raids bring food home) |
| Prey eaten in the **neutral** area (inside neither territory) | Respawns after 60–120 s in the neutral area (old rule) |
| Rival-territory cost | Kept but small: −0.01 per step (was −0.03), so raids are short and purposeful |
| Rival | **Present during training**: both species share one world, trained in turns against the other's current (frozen) brain (section 8, training stage 4), incl. the defence rewards (confront intruder, intruder leaves, fight risk < 2 m) |

Defaults chosen by Claude where the owner did not specify (change if needed):
- The **overlap** of the two territories (around the water) counts as **neutral**.
- A prey **stays inside the area it spawned in** (own territory A, territory B or neutral) so the stocks stay meaningful.
- Start of an episode: 8 prey per territory, 4 neutral. Raid respawn is immediate, ≥ 10 m from the eater.
- Hunger and fatigue rise with the **mechanical energy** the physics body spends (section 14 v2), not only with speed.
