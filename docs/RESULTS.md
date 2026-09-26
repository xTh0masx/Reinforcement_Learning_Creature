# Endemic Creatures – what we did and what came out

Status: 26 Sep 2026 · Project plan: [PROJECT_PLAN.md](../PROJECT_PLAN.md) · Walking training:
[locomotion_training_log.md](locomotion_training_log.md) · Predator training:
[predator_training_log.md](predator_training_log.md) · All changes: [CHANGELOG.md](CHANGELOG.md) · Time spent: [TIME_LOG.md](TIME_LOG.md) ·
Every training run: [experiments.csv](experiments.csv)

Two predator species from a Blender scene – **Myrranox** (6-legged arthropod, 1.5 m long) and **DC_SKIN_RAW**
(quadruped, 1.3 m tall) – live in one 100 × 100 m habitat. **Everything they do is learned with reinforcement
learning** (Gymnasium + Stable-Baselines3 PPO, CPU only):

| Brain | Environment | Decides | Rate |
| --- | --- | --- | --- |
| Navigation / predator brain | `sim/creature_env.py` – 2D habitat, prey, hunger, fatigue, territories, rival | where to go: forward speed + turn rate | 10 Hz |
| Locomotion brain | `sim/locomotion_env.py` – MuJoCo physics body | every leg joint, to follow that speed + turn rate | 50 Hz |

The recorded body pose and joint angles are played back on the rigged models in Blender.

---

## 1. Timeline

| Date | Step | Result |
| --- | --- | --- |
| 23 Sep | Project plan; repo structure; `requirements.txt`; PPO on CartPole | CartPole 500.0 ± 0.0 (smoke test passed) |
| 23 Sep | Habitat sim (`CreatureEnv`), species limits, matplotlib viewer, `check_env` | Env passes `check_env`, ~3000 steps/s |
| 23 Sep | Blender: habitat from `habitat.json`, trajectory import, follow cameras | Scripted preview plays in `habitat_showcase.blend` |
| 25 Sep | Blender scene restored after an unsaved session and saved | `blender/habitat_showcase.blend` |
| 25 Sep | Decision: **walking is learned by RL too** (MuJoCo) | Plan changed (section 14) |
| 25 Sep | Physics bodies measured on the Blender meshes; rigs for both models; pose playback | Bone tips match the physics feet to **0.01 mm** |
| 25 Sep | Locomotion pilots 1–3 | Failed (penalties too strong / learned to stand still) – kept as lessons |
| 25 Sep | Locomotion **v1** (4M steps each) | Both walk, never fall – but unnatural gaits |
| 25 Sep | Locomotion **v2** – natural gait (4M steps each) | Insect tripod / quadruped walk→trot, symmetric, 2.4–5× less energy |
| 25 Sep | **B** physics habitat world; **C** predator rules (territories, raids, energy-driven hunger, rival) | Tested; two design problems found and fixed |
| 25 Sep | Prey made slower (0.9 → 0.7 m/s, owner decision); **D1** predator brains started | two failed attempts, then learning |
| 25–26 Sep | Predator round 1 (3M): interrupted by the session end at 1.75M, resumed from the 1.5M checkpoint | Both hunt (4.0–4.4 prey / 10 min) but never raid, rest or defend |
| 26 Sep | Option B (home-stock + intruder inputs, lean days, invading rival, fatigue from 70 %) | DC much better – **released** (`dc_brain_v4_r1`); Myrranox worse → keep `myr_brain_v3_r1` (owner) |
| 26 Sep | Showcase: both predator brains + v2 walking brains in the physics habitat → Blender | DC hunts with real legs; Myrranox cannot turn on the spot and starves (owner: show as is) |
| 26 Sep | Blender: steady side cameras, overview, panel "Creatures", hunger/fatigue display (owner request) | Clear side view of the leg motion; free viewing of both creatures |
| 26 Sep | Prey at half speed (owner decision); showcase re-recorded (5 min) | Myrranox catches prey and survives; DC catches 3 |
| 26 Sep | Showcase rendered side by side (5:00) | `docs/videos/showcase_side_by_side.mp4` |

---

## 2. Habitat and needs (2D, `sim/creature_env.py`)

- Map `sim/habitat.json` (shared with Blender): two home zones (radius 28 m) overlapping at a water hole, 6 rest spots,
  14 rocks, walls. Prey are red dots that wander and flee from predators.
- Needs per creature: **hunger** (rises all the time, faster with effort; catching prey −0.5), **fatigue**
  (rises when moving, falls when standing, faster on a rest spot; at 100 % top speed is capped at 30 %),
  **alertness** (rival nearby).
- Observation: 27 values, egocentric (needs, own motion, nearest prey, rest spot, home, nearest other creature, 8 obstacle rays).
- **Territory rules (owner decision, plan section 15):** prey eaten in the own territory do not come back; prey eaten
  in the rival territory re-appear in the eater's territory (raids bring food home); neutral prey respawn after
  60–120 s. Rivals may invade; being in the rival territory costs a little (−0.01 per 0.1 s).

---

## 3. Walking learned with RL (locomotion brains)

### Physics bodies (`sim/bodies.py`)
- **Myrranox:** hexapod, 8 kg, 18 joints. Joint positions measured on the Blender mesh, so a physics joint angle is
  directly a bone rotation of the rig.
- **DC:** quadruped, 64 kg, 16 joints (hip ×2, knee, ankle per leg), measured from a side render.

### Training sessions

| # | Run | Result | Material |
| --- | --- | --- | --- |
| 1 | `myr_legs_pilot` | Foot-slip penalty swamped the task reward → stopped | learning curve |
| 2 | `myr_legs_pilot2` | Learned to **stand still** (local optimum) → stopped | `figures/myr_legs_pilot2_1M.gif` |
| 3 | `dc_legs_pilot` | Stopped with #2 | learning curve |
| 4 | `myr_legs_v1` | Walks from 1M steps; **3 legs tapping at 9–12 Hz**, front legs in the air | video, figures |
| 5 | `dc_legs_v1` | Walks and trots; **hind legs hop together**; barely turns | video, figures |
| 6 | `myr_legs_v2` | **Insect tripod**, all 6 legs, 2.3–3.7 steps/s, symmetric turning | video, figures |
| 7 | `dc_legs_v2` | **Diagonal walk / trot**, 1.7–2.8 steps/s | video, figures |

What made v1 work (after the pilots): wider speed reward, penalties ramped in over the first 40 % of training.
What made v2 natural: per-leg **gait clock**, reward for following the species' gait (stance legs down + swing legs up,
scored so that standing still earns nothing), foot clearance, air time, energy and smoothness penalties, and a
**mirror-symmetry loss** (`sim/symmetry.py`) – checked before training with an exact mirrored simulation
(`tests/test_locomotion_v2.py`).

### v1 vs v2 (final brains, same command script)

| | Myrranox v1 | **Myrranox v2** | DC v1 | **DC v2** |
| --- | --- | --- | --- | --- |
| Gait | 3 legs, 9–12 Hz tapping | **tripod, all 6 legs** | hind legs hop together | **diagonal walk / trot** |
| Walk / full speed (m/s) | 0.61 / 1.19 (cmd 0.6 / 1.2) | 0.53 / 1.31 | 0.76 / 2.48 (cmd 0.8 / 2.5) | 0.68 / 2.15 |
| Turn left / right | +56 / −31 °/s (cmd ±54) | **+57 / −54** | +10 / −8 °/s (cmd ±36) | **+24 / −22** |
| Left/right asymmetry | 11.9° | 6.0° (2.2° at full speed) | 11.6° | 6.9° (1.3° at trot) |
| Power at full speed | high (~450 W at a walk) | **97 W** | 3.8 kW | **1.56 kW** |
| Falls | none | none | none | none |

Open: Myrranox cannot turn on the spot or walk backwards; both walk ~10–15 % slower than commanded.

---

## 4. Blender (`blender/`)

- `habitat_showcase.blend`: habitat built from `habitat.json`, both rigs, the showcase recording (5 min), cameras and
  the viewing panel. Backup before rigging: `habitat_showcase_before_rig.blend`.
- `rig_myrranox.py`: legs split into left/right parts, rigidly parented to bones placed on the physics joints.
  `rig_dc.py`: 76k-face decimated proxy of the 970k-face sculpt with automatic weights (retopology still planned).
- `import_trajectories.py`: plays a recording – torso bone = physics body pose, leg bones = joint angles, 50 Hz → 30 fps.
  **Verified: bone tips match the recorded physics feet to 0.01 mm.**
- **Cameras (26 Sep):** per creature a steady side camera (`SideCam_<id>`, beside the body at body height, position and
  heading smoothed over 0.7 s / 3 s) with a hunger/fatigue display in the image; `Overview_cam` over the whole habitat.
  No automatic camera switching.
- **Panel "Creatures"** (3D view sidebar, N key; stored in the .blend, Blender asks to allow scripts):

  | Button | Effect |
  | --- | --- |
  | Both side by side | after View → Area → Vertical Split: Myrranox side camera left, DC right, both play together |
  | `<id>` side view | look through that creature's side camera |
  | Overview | whole habitat |
  | Follow `<id>` | view stays centred on the creature, orbit/zoom freely with the mouse while it plays |
  | Free view / Play / Frame | free navigation, playback |

- **Rendering:** `render_camera.py` (background, resumable, one camera) + `make_side_by_side.py` → MP4.
- Known issue: Myrranox textures point to `D:\FINAL MODUL\…` and are missing on this PC (renders pink).

## 5. Physics habitat world (step B, `sim/physics_world.py`)

The MuJoCo world is built from the same `habitat.json`: walls and rocks are solid, several creatures share one
simulation, each walked by its own locomotion brain. Checked: a Myrranox walking into a rock is stopped at its surface;
the refactor left the v1 brains' results identical to the last decimal.

## 6. Predator rules and balance (step C)

Balance tests with two hand-written (scripted) predators, 10-minute days, several seeds:

| Finding | Fix |
| --- | --- |
| The DC got stuck at rocks and starved: a collision set its speed to 0 and it can only turn while moving | Creatures slide along obstacles (as in the physics world); the hit penalty counts once per contact |
| Nobody raided – with 8 prey per territory the home stock lasts a whole day | 5 prey per territory + 3 neutral; the scripted rival raids once its home stock is ≤ 2 |
| Myrranox sprinted to raids and arrived exhausted | Scripted predators sprint only within 8 m of the prey |
| Myrranox hunts worse than the DC (3 vs 5 prey per day, 11 % vs 0 % starving) | Prey slower (0.9 → 0.7 m/s, owner decision). Effect in the scripted test small – the Myrranox's v2 gait costs 58 % of its full-speed power just to walk (DC: 30 %), so it tires twice as fast. Watch in D1. |

Movement limits for the predator brains come from what the v2 locomotion brains can really do
(`sim/locomotion_limits.json`: no backwards / turning on the spot for the Myrranox, DC turns ≤ 40 °/s).
Hunger and fatigue use the power curve measured from the v2 brains (`sim/energy.json`).

---

## 6b. Predator brains (step D1, 25–26 Sep)

| Attempt | Result | Lesson |
| --- | --- | --- |
| `*_brain_v1_r1` – plan reward (section 8) | stayed home and starved (0 prey) | even a good scripted hunter scored negative: the reward hardly separated hunting from starving |
| `*_brain_v2_r1` – reward v2 (fed + every catch + hunting guide) | stood still (0 prey) | half the action range meant "stop"; the guide paid for standing (discount); fleeing prey never caught by chance |
| pilot (400k) – exploration fixes | learns to hunt (0 → 2.5 prey / 10 min) | full speed range, undiscounted guide, prey-flee curriculum, hit counted after 1 s |
| `*_brain_v3_r1` – round 1, 3M steps | hunt 4.0–4.4 prey / 10 min, but never raid, rest or defend | rare situations (empty home, intruder) are seldom experienced |
| `*_brain_v4_r1` – option B (home-stock + intruder inputs, lean days, invading rival, fatigue from 70 %) | **DC: released** – survives every day, 4.6 prey, rests (4/5 checks). Myrranox worse (0/5) | the Myrranox tires ~2× faster than the DC (gait energy) |

Released: **DC `models/dc_brain_v4_r1`**, **Myrranox `models/myr_brain_v3_r1`** (best Myrranox; owner decision to keep it).
Section 12 evaluations: `docs/evaluations/`. Raids and territory defence were not learned by any brain.

## 6c. Showcase: both predators with real walking (step F, 26 Sep)

`record_world.py` runs both predator brains together; their commands drive the v2 walking brains in the MuJoCo habitat
(solid rocks and walls). Recording `recordings/showcase.csv` (+ 50 Hz pose files), imported into
`blender/habitat_showcase.blend` (5182 frames = 2:53 at 30 fps, follow cameras switch at frame 2592, hunger/fatigue bars
in each camera view).

- **DC:** hunts successfully with its real legs – 2 prey in 3 min, stays well fed.
- **Myrranox:** does not transfer from 2D to the physics body (owner decision: show it as is). It cannot turn on the spot
  (walking v2), so it stalls at the start, comes within 1.4–2.4 m of prey but the real body turns and accelerates more
  slowly than the 2D model it was trained with, gets stuck at a rock and starves after 173 s (0 prey).
  Fix options kept for later: walking v3 with turning on the spot, or fine-tuning the predator brain with the physics
  body (plan D2, slow on CPU).
- Hunger/fatigue in the physics world use the same power curve as the 2D training (speed/turn → power): the body's
  instantaneous power includes controller effort for every command change and starved the Myrranox in 132 s.

**Update 26 Sep (owner):** prey at half speed; the showcase was re-recorded for 5 min – the Myrranox now catches prey
with its real legs and survives the whole recording (1 prey), the DC catches 3. Blender: steady side camera per
creature, overview camera, panel "Creatures" (side views, follow + free orbit, both side by side after splitting the
3D view). Preview: `figures/showcase_side_by_side.png`.

## 6d. "Alive" bodies – stage 1: passive body parts (26 Sep)

Owner request: not only the legs – the whole body should move (Myrranox like an ant, DC like a panther with extra
arms). Stage 1 adds body parts that move only by physics: spring-damper joints without motors.

| Creature | Passive parts (joints) | Swing while walking (v2 brain) |
| --- | --- | --- |
| Myrranox | abdomen (up/down, sideways), head (up/down, sideways) | abdomen up to 17–20° sideways, head up to 9° |
| DC | tail 2 segments (up/down, sideways each), neck (2) + head (1), both extra arms (2 each) | tail tip up to 34–46° in turns, arms up to 10°, neck ≈ 1° (steady head) |

Spring rest angles compensate gravity (parts rest in the modelled pose); the collision shapes of the low Myrranox
abdomen and head were slimmed so they do not rest on the ground. The v2 walking brains need no retraining (no falls,
unchanged speed and turning). Recording and Blender rigs include the new joints. Preview:
`videos/alive_stage1_preview.mp4`. Next: stage 2 (actively controlled spine/neck/head, RL) and stage 3 (gaze, eating,
idle motion).

## 6e. "Alive" bodies – stage 2: controlled spine / neck / head (walking v3, 26 Sep)

| | DC v2 + passive parts | **DC v3 (released)** | Myrranox v2 + passive | Myrranox v3 |
| --- | --- | --- | --- | --- |
| Walk / full speed (m/s) | 0.68 / 2.15 | **0.83 / 2.69** | 0.58 / 1.35 | 0.00 / 0.02 (stands still) |
| Head wobble (rad/s) | 1.17 | **0.69** | 1.74 | (1.00, standing) |
| Spine / tail tip swing (°, std) | – / 10.6 | **3.2 / 15.5** | – | – |
| Falls | 0 | 0 | 0 | 0 |

The DC transferred the new control into faster, more fluid, steadier-headed walking. The Myrranox learned to stand
still once the head-steadiness penalty ramped up (lesson 8 in the walking log); it keeps v2 + passive parts until
a fix. Videos: `videos/dc_legs_v2_vs_v3.mp4`, `videos/*_legs_v3_training.mp4`.

## 7. Videos and graphs

| File | What it shows |
| --- | --- |
| `videos/myr_legs_v1_training.mp4`, `videos/dc_legs_v1_training.mp4` | Whole v1 walking training: every 250k checkpoint (untrained → 4M) next to the learning curve, 3:43 each |
| `videos/myr_legs_v2_training.mp4`, `videos/dc_legs_v2_training.mp4` | Same for v2 (natural gait) |
| `videos/*_legs_v1_vs_v2.mp4` | v1 and v2 side by side on the same commands |
| `figures/locomotion/learning_curves_<species>.png` | Reward curves of all walking runs incl. the failed pilots |
| `figures/locomotion/<run>/progress.gif` | All checkpoints back to back (short) |
| `figures/locomotion/<run>/contact_sheet.png` | One frame per checkpoint |
| `figures/locomotion/<run>/metrics.png` / `.csv` | Speed / turn tracking, restart from standstill, falls per checkpoint |
| `figures/locomotion/<run>/<N>_steps.gif` | 12-s clip of each checkpoint |
| `figures/gait/<run>_footfall_<speed>.png` | Footfall diagrams (dark = foot on the ground); v1 vs v2 is the clearest before/after slide |
| `figures/gait/*_legs_v2_<N>_steps_footfall_*.png` | Gait during v2 training (1M, 1.5M, 2.75M, 3.5M) |
| `figures/myr_legs_pilot2_1M.gif` | Pilot 2 standing still at 1M steps |
| `videos/showcase_side_by_side.mp4` | **Showcase, 5:00**: Myrranox (left) and DC (right) living in the habitat, everything learned (predator + walking brains), steady side cameras with hunger/fatigue – rendered from `blender/habitat_showcase.blend` |
| `videos/*_legs_v3_training.mp4`, `videos/*_legs_v2_vs_v3.mp4` | Walking v3 (alive body): whole training; stage 1 vs stage 2 side by side |
| `videos/predator/<run>_training.mp4` | Every predator training run (8): one habitat day per checkpoint, untrained → 3M, next to the learning curve |
| `evaluations/*.txt, *.json` | Section 12 checks of the predator brains and the scripted baseline |
| `../blender/habitat_showcase.blend` + `../recordings/showcase*.csv` | Both predators in the habitat with real walking, hunger/fatigue display |

Live during training: `tensorboard --logdir runs`, `python watch_locomotion.py --run <run> --species <species>`.

---

## 8. Lessons learned

1. A penalty that is large for an untrained policy's jitter swamps the task reward (pilot 1).
2. A narrow tracking reward plus full-strength penalties makes standing still a local optimum (pilot 2) –
   a wider reward and a penalty curriculum fix it.
3. A policy learns to keep walking before it learns to start walking – the evaluation checks both.
4. Rewarding only *what* (speed) gives unnatural motion; rewarding *how* (gait clock, symmetry, energy) gives natural,
   efficient gaits at little cost in accuracy.
5. Test a reward before training on it: the first gait reward paid standing still 55–65 % (caught by a unit test).
6. Mid-training setbacks recover (speed dip at 1.5M, a raised leg at 2.75M) – judge by the end, keep every checkpoint.
7. Simulation details decide survival: a creature that can only turn while moving starved at the first rock.
8. Check a reward with scripted policies before training – the plan's predator reward barely preferred hunting.
9. A good reward still fails without early success: full action range, shaping that never pays for idling, and a
   curriculum (prey that do not flee at first) made the predator brains learn.
10. Rare behaviour (raids, defence) needs practice scenarios and the right inputs – and even then may not appear.
11. 2D-trained decisions must match what the legs can do: the DC transferred to the physics body, the Myrranox did not
    (no turning on the spot). Use the same energy model in training and playback.
12. Training runs inside the Claude session – closing the app stops it; frequent checkpoints make restarts cheap.

## 9. Next (open)

- Myrranox: walking v3 with turning on the spot, or fine-tuning its predator brain with the physics body (plan D2);
  fatigue balance (owner decision).
- Raids and territory defence (not learned); alternating rounds against the other species' trained brain.
- Final scene: 2 Myrranox + 2 DC; terrain/props; relink the Myrranox textures and re-render.
- Paper and presentation (deadline and format still open).
