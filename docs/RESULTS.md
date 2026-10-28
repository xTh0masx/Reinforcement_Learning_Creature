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
| 29–30 Sep | Walking v5 (natural style); stuck joints found and freed | Myrranox knee 4.5 → 20°, steadier; DC panther stride but less steady → owner decision |
| 29 Sep | Showcase v4 without rocks (owner): rocks removed from physics and Blender, re-recorded | `docs/videos/showcase_side_by_side_v4_no_rocks.mp4` |
| 28 Sep | Showcase v4: both creatures on walking v4 (DC 6 prey, Myrranox 2, no falls) | `docs/videos/showcase_side_by_side_v4.mp4` |
| 28 Sep | Walking v4 (precise + fluid) for both, released | Myrranox turns on the spot; DC turns ±30 °/s; ~25 % smoother |
| 27 Sep | Showcase v3: DC walking v3 (alive body) + Myrranox v2 + passive parts; side cameras dodge rocks | `docs/videos/showcase_side_by_side_v3.mp4` |

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
a fix (the retries v3b and v3c failed too, 27 Sep). Videos: `videos/dc_legs_v2_vs_v3.mp4`, `videos/*_legs_v3_training.mp4`.

**Showcase v3 (27 Sep):** the 5-minute showcase re-recorded and rendered with the released DC v3 and Myrranox v2 +
passive parts: `docs/videos/showcase_side_by_side_v3.mp4` (still `figures/showcase_side_by_side_v3.jpg`). The side cameras now orbit to the
other side of the creature when a rock or the wall would block the view (DC: 3 side changes in 5 min).

## 6f. Walking v4 – precise and fluid (28 Sep)

Owner request: movements more precise and fluid. No mesh changes (the brains only see the physics capsules); new
settings in `sim/locomotion_env.py` version 4: joint-target filter, tracking tolerance 1 → 0.5, turn weight 1.5,
soft-landing penalty, ramped commands, 20 % spot turns, head style paid only while tracking, speed reward on spot turns
scaled by the turn. Both brains continued from the released ones for 3M steps (1.1 h, both in parallel), after a
0.5M pilot.

| | Myrranox v2 | **Myrranox v4** | DC v3 | **DC v4** |
| --- | --- | --- | --- | --- |
| Walk / full speed (m/s, command 0.6 / 1.2 and 0.8 / 2.5) | 0.58 / 1.35 | **0.60 / 1.20** | 0.83 / 2.69 | **0.79 / 2.41** |
| Drift when walking straight (°/s) | +10 | **−1** | −2 | **0** |
| Walking turns (°/s, command ±54 / ±36) | +56 / −53 | **+57 / −56** | +21 / −21 | **+29 / −32** |
| Turn on the spot (°/s, command 72) | 7 | **58** | – | – |
| Backwards (m/s, command −0.3) | −0.03 | −0.05 | – | – |
| Mean speed error / turn error (whole script) | 0.10 m/s / 18.6 °/s | **0.10 / 7.9** | 0.13 / 10.7 | **0.08 / 4.6** |
| Joint acceleration (rad/s², lower = smoother) | 84 | **64 (−24 %)** | 102 | **74 (−27 %)** |
| Joint target change per 20 ms | 3.05° | **2.32°** | 3.91° | **3.14°** |
| Head wobble (rad/s) / head tilt | 1.74 / 3.6° | **1.30 / 3.4°** | 0.69 / 10.7° | **0.65 / 2.4°** |
| Power, walking script | 73 W | **57 W** | 947 W | **803 W** |
| Falls | 0 | 0 | 0 | 0 |

Both v4 brains are released and are the default walkers of `record_world.py`. Open: Myrranox walking backwards.
In the physics habitat (1-min test, `record_world.py`): no falls, both catch prey.

**Showcase v4:** 5-minute showcase re-recorded and rendered with both v4 walkers: `docs/videos/showcase_side_by_side_v4.mp4` (still
`figures/showcase_side_by_side_v4.jpg`). No falls; DC 6 prey, Myrranox 2 prey and survives.
Without rocks (owner, 29 Sep): `docs/videos/showcase_side_by_side_v4_no_rocks.mp4` – rocks removed from the physics world and the Blender scene;
no falls, Myrranox 3 prey, DC 4 prey.

## 6g. Walking v5 – natural style (29–30 Sep)

Owner request (29 Sep): the Myrranox should use its legs like an insect (not stiff), the DC should not look like
jelly and should move like a panther. Walking v5 (`sim/locomotion_env.py` version 5): a leg rhythm reward from the
gait clock (each leg reaches forward in the swing and pushes back in stance by the stride the speed needs, the knee
folds mid-swing; measured around each joint's running mean, so the posture stays free), knee mostly exempt from the
energy / joint-acceleration penalties, DC: slower step frequency (long panther strides), hind paw into the front
paw's print, tail and extra arms damped harder (0.15/0.25 → 0.5); after the owner said the DC "does not look stable
at all", a steady-trunk rule (torso rocking / bouncing 10× more expensive, paid in proportion to the tracking).

**Key finding – stuck joints.** The Myrranox knees had been locked since walking v2 at 2M steps: the brain's knee
command sat at about −1.4, beyond the action limit (±1), so every sample gave the same target and the knee could not
learn any more (v2 at 1M: −0.4, knee moving 50°). The same happened to the DC's four abduction joints and its front
shoulders (hence front legs out of step). `unsaturate_brain.py` shifts the output of stuck joints back inside the
range (pose barely changes) and widens their exploration; both v5 runs started from such corrected v4 brains.

| Myrranox | v2 at 1M (lively) | v4 | **v5** |
| --- | --- | --- | --- |
| Knee range of motion | 50° | 4.5° (locked) | **20°** |
| Hip swing / leg lift | 30° / 41° | 23° / 19° | 24° / 21° |
| Foot lift | 16.5 cm | 9 cm | **13 cm** |
| Torso wobble | – | 0.58 rad/s | **0.41 rad/s** |
| Turn on the spot (command 72 °/s) | – | 58 °/s | **65 °/s** |
| Speed error / turn error | – | 0.10 m/s / 7.9 °/s | 0.11 / 7.7 |
| Power / joint acceleration | – | 57 W / 64 rad/s² | 79 W / 86 rad/s² |
| Falls | – | 0 | 0 |

| DC | v4 | v5 at 8.5M (no steady trunk) | **v5b (steady trunk)** |
| --- | --- | --- | --- |
| Steps per second / stride (walk) | 1.75 / 0.45 m | 1.22 / 0.64 m | **1.12 / 0.55 m** |
| Hind paw distance to the front print | 33 cm | 32 cm | **26 cm** |
| Torso wobble | **0.40 rad/s** | 0.59 | 0.44 |
| Head wobble / head tilt | **0.65 rad/s / 2.4°** | 0.85 / 5.9° | 0.80 / 7.7° |
| Body bob | **2.5 cm** | 3.6 cm | 3.7 cm |
| Walk speed (command 0.8) / turns (±36 °/s) | **0.79** / +29 −32 | 0.84 / +33 −31 | 0.62 / +30 −32 |
| Falls | 0 | 0 | 0 |

Myrranox v5 is better in every respect except power and joint acceleration (the knee works now) – recommended for
release. DC v5b walks with longer, slower, better-placed steps but rocks its head more and walks too slowly
(0.62 of 0.8 m/s); no checkpoint of v5 / v5b is as steady as v4 → owner decision after the side-by-side video.
Measurements: `docs/evaluations/v5/` (`style_metrics.py`, `alive_metrics.py`, `precision_metrics.py`), baseline
`docs/evaluations/v5_baseline/`. Videos: `docs/videos/myr_legs_v4_vs_v5.mp4`, `docs/videos/dc_legs_v4_vs_v5.mp4`,
training `docs/videos/myr_legs_v5_training.mp4`, `dc_legs_v5_training.mp4`, `dc_legs_v5b_training.mp4`.
DC skinning check: 20 % of the proxy vertices are blended between several bones (normal for automatic weights);
"preserve volume" skinning not yet tried.

## 6h. Mesh bodies, new reward, walking v6–v11 (30 Sep, 13 Oct)

**30 Sep – DC experiments (none released).** v5c (head ×3, strict speed) became the DC walker (0.78 m/s, panther
stride; the head is held steady in the Blender animation only). v6: the extra arms became actuated limbs (29 motors),
legs ±8° sideways, stricter rhythm rules - a brain transplanted from v5c (`grow_brain.py`) stood still, from scratch it
was stopped by the owner. v7: a step generator (tripod / walk rhythm + learned corrections) - the rhythm alone made the
DC fall every 1.5 s (a quadruped is not statically stable). v8: DC body made of convex hulls cut from the real mesh,
sprint to 4 m/s, ramped commands - it did a "handstand" on its front legs and arms (fixed: 25° tilt = fall) and
collapsed into standing still once the penalties reached full strength (3.2M).

**13 Oct – new reward structure (walking v9, both creatures).** Penalties used to be subtracted, so at full strength
standing still lost less than walking (the trap of Myrranox v3 and DC v8). Now
`reward = tracking × (0.2 + 0.8 × animal-likeness) × quality` with `quality = exp(−Σ c·cost)`; standing still while
walking is commanded earns almost nothing. Both creatures got mesh bodies (Myrranox: 21 hulls; DC: shoulder hump moved
to the chest) and a natural standing pose as the neutral position of their commands. Fixes found on the way: exploration
noise was penalised as jerk (quality 0.06 in training → measured after the filter), resumed brains carried the old
reward statistics (return std ~95 → `--reset-reward-norm`), the fall check used the computed instead of the settled
height (false falls), command ranges too small for the new stances (Myrranox legs ±50°, DC arms ±103°).

**Myrranox walking v10 (released).** On v9 the Myrranox kept finding ways round the rules (stepping in place, a
"wheelbarrow" with the front legs in the air). Root cause was the body: mass by volume gave the abdomen 54 % of the
weight, the centre of mass sat over the hind legs, and even a perfect tripod rhythm tipped it back. Abdomen density
×0.25 (23 % of the weight) and a tripod step generator with learned corrections (version 10) made it walk like an
insect, from scratch in 3M steps:

| Myrranox (walk script) | v5 (previous walker) | **v10 (released 13 Oct)** |
| --- | --- | --- |
| Walk / full speed (command 0.6 / 1.2 m/s) | 0.59 / 1.13 | **0.52 / 1.08** |
| Walking turns (±54 °/s) / turn on the spot (72 °/s) | +65 −58 / 65 | **+57 −49 / 56** |
| Backwards (command −0.3 m/s) | −0.01 (never learned) | **−0.26** |
| Trunk tilt / bob | – | **2.0° / 1.4 cm** |
| Knee / hip swing | 20° / 24° | **60° / 28°** |
| Abdomen on the ground | – | **0 %** |
| Tripod groups in opposite phase | – | **87 %** |
| Front feet on the ground | – | 21–22 % (target ~55 %), stepping too fast (5–6 /s) |
| Falls / joint commands beyond the limit | 0 / – | 0 / 0 % |

**DC walking v11 (training).** The DC hull body was head-heavy (neck + head 21 %, trunk 37 %, centre of mass almost
over the front paws) and its claw hulls caught on the ground. v11: big-cat mass distribution (trunk 55 %, neck + head
9 %, hind legs 15 %, front legs 12 %, arms 6 %, tail 3 %) and rounded paw pads. Best DC on v9 so far: 0.65 m/s, trunk
3.7°, correct lateral footfall order, but hind legs stepping 2–4× per cycle (an "extra step" cost made it stop using the
hind legs altogether and was switched off).

Measurements: `docs/evaluations/` and the session's checkpoints; videos `docs/videos/myr_legs_v10_training.mp4`,
`docs/videos/myr_legs_v5_vs_v10.mp4`; footfall `docs/figures/gait/overview_myrranox.png`; physics bodies
`docs/figures/*_physics_body_mesh.jpg`, standing poses `docs/figures/standing_pose_v9.png`; presentation
`docs/presentation/Endemic_Creatures_RL.pptx`.

## 6i. DC walking v12 – detailed body, wolf reference, T-rex arms, new hand (13–14 Oct)

**Detailed physics body.** The owner wanted the DC's physics body "as close as possible to the real model". The
full-detail sculpt (953k vertices, claws, teeth) is split by bone and decomposed with CoACD (approximate convex
decomposition, the tool used for MuJoCo's model collection) into **218 convex pieces** instead of 24 single hulls;
spikes, horns, the open jaw, the shoulder hump and the belly line are now part of the body. Spikes and horns get a
third of the density (the centre of mass rose 9 cm otherwise). Physics speed is unchanged (1,235 vs 1,311 steps/s).

| Part | single hull → pieces (95 % of the surface within) |
| --- | --- |
| Torso / neck / rear body | 18.1 → 3.3 cm / 13.4 → 4.3 cm / 12.9 → 3.0 cm |
| Upper legs / lower legs and paws | 7.7–10.9 → 3.1–4.2 cm / 1.9–4.9 → 1.2–2.4 cm |
| Head / hand (after the remodel) | 6.9 → 3.7 cm / 3.9 → 1.4 cm |

**Wolf reference.** The owner's target movement is a wolf walk cycle (Sketchfab model, screen recording). Rules from
it (walking only, up to 1.2 m/s): front and hind legs step at the same rate, at least two paws on the ground, paws
lifted at most 8 cm, the front wrist folds under in the swing, head carried low (neck rest pose 16° lower).

**Weight, arms and hand (owner requests, 14 Oct).**
- More weight on the hind legs (9 % each) and tail (5.5 %), lighter head (2.5 %) and neck (4 %): front/hind 55/45 →
  50/50 – the DC fell forward at speed.
- Arms held like a T-rex: upper arm forward, hands extended in front of the chest, above the front legs' swing;
  forearm and hand collide with the front legs; while the arms are not part of the gait the brain may move them only
  ±17° (it had learned to keep the wrists bent 70–90°).
- The hand was remodelled: in the real model the forearm curves down and the hand hangs folded back like a hook. The
  old physics wrist sat 10 cm too high, inside the curve, so the hand could never open. Now the wrist is at the bottom
  of the curve, the hand runs back to the fingertips and opens up to 250° (palm to the ground); the Blender rig was
  changed the same way (`rig_dc.remodel_hands`).
- Arms in the sprint were tried (arms join from 2.0 m/s, crouched sprint pose with palms flat, soft wrist + palm
  reflex, palm and crouch rules, a separate arm brain `sim/arm_brain.py`). The hands still ran on their fingertips
  (−61°) because the DC reared up (nose 11° up, shoulders 0.69 instead of 0.48 m). **Owner decision: no arms in the
  sprint** – T-rex arms at every speed, the DC sprints on four legs.

**Training.** Many short stages (training log rows 23–36); the lessons are in section 8. After the sprint stage the
DC lay down in a "plank" (trunk 16–18 cm low, legs stretched) when standing or walking slowly → the plank counts as a
fall, posture cost. Then the walk looked "clunky" (owner): short hurried steps, 6 cm bob → smoothing stage (70 %
walks, bounce and surge costs up, step rhythm double, gentle start-up help only at slow commands).

**Result `models/dc_legs_v12z` (20.8M steps, 14 Oct):**

| | 0.6 m/s command | 1.0 m/s | 2.5 m/s | 4.0 m/s |
| --- | --- | --- | --- | --- |
| Speed | 0.59 ± 0.14 | 0.86 ± 0.17 | 1.84 | 3.56 (peak 4.1–4.2) |
| Steps/s · timing variation | 1.12 · 9 % | 1.07 · 1 % | 1.89 · 2 % | – |
| Stride · trunk bob | 0.52 m · 1.8 cm | 0.81 m · 2.0 cm | 0.97 m · 1.7 cm | – |

Braking from the sprint to a walk: 3 of 3 without a fall; walking turns +31 / −28 °/s (command ±36); standing at the
normal height (4 cm); no falls in any test. Weak: turning on the spot creeps forward (0.5 m/s) at 33 °/s (command 57).

Figures: `figures/dc_physics_body_v11_vs_v12.jpg`, `figures/wolf_reference_frames.png`,
`figures/dc_arms_folded_vs_trex.png`, `figures/dc_trex_arms_walk.png`, `figures/dc_hand_real_model_closeup.png`,
`figures/dc_hand_v12_poses.png`, `figures/dc_hand_v12_rig_folded_open.png`, `figures/dc_sprint_pose_palms.png`.

## 6j. Final walkers, hunting-game idea, showcase v6 (14–28 Oct)

**Final walkers (owner).** Myrranox `models/myr_legs_v10` (tag `myrranox-v10-final`, 13 Oct) and DC
`models/dc_legs_v12z` (tag `dc-v12-final`, 14 Oct: "this was the final training of the DC"). `record_world.py` walks
with both by default.

**Hunting game (14 Oct, scrapped).** Owner idea: one round = 3 min hunt (+1 point per prey), then 30 s to reach the
water hole or die; the survivor with the most points wins (hunger, fatigue and territories kept, prey respawning). It
was built (`918188e`) and scripted rivals played it: the DC (about twice as fast) scored about 7 points and always
reached the water hole, the Myrranox about 3 and often ran itself tired (at 100 % fatigue it only manages 30 % of its
top speed). The owner scrapped the idea a few minutes into the first training; the code was reverted (`000bcd3`), the
released hunting brains are unchanged.

**Showcase v6 (21 Oct).** Recorded with the final walkers in the habitat without rocks: 5 min, both survive, 4 prey
each, no falls (2 min of compute). The Blender import works with the new DC body (T-rex arms, remodelled hand); the
Myrranox is still shown without textures. Rendered at 24 fps, 8 samples, sharp shadows: `videos/showcase_v6_30s.mp4`
(30 s, 16 min of rendering); the 5-min render is open.

**Why rendering takes long.** The laptop renders with its integrated GPU (Intel Iris Xe). Measured per frame in the
real animation render (960 × 540, incl. rig, mesh deformation, HUD, PNG):

| Settings | s / frame | 5-min video (2 cameras) |
| --- | --- | --- |
| 16 samples, soft shadows, 30 fps (until showcase v5) | 0.83 | ~4.2 h (4.3–5.7 h measured incl. battery) |
| 8 samples | 0.71 | ~3.6 h |
| 8 samples, sharp shadows | 0.56 | ~2.8 h |
| + 24 fps (showcase v6) | ~0.6 | ~2.2 h |
| 4 samples | 0.55 | visible grain – not used |
| Dedicated GPU (desktop / cloud, estimate) | ~0.05–0.1 | ~15–30 min |

## 7. Videos and graphs

| File | What it shows |
| --- | --- |
| `videos/myr_legs_v1_training.mp4`, `videos/dc_legs_v1_training.mp4` | Whole v1 walking training: every 250k checkpoint (untrained → 4M) next to the learning curve, 3:43 each |
| `videos/myr_legs_v2_training.mp4`, `videos/dc_legs_v2_training.mp4` | Same for v2 (natural gait) |
| `videos/*_legs_v1_vs_v2.mp4` | v1 and v2 side by side on the same commands |
| `figures/locomotion/learning_curves_<species>.png` | Reward curves of all walking runs incl. the failed pilots |
| `figures/locomotion/learning_curves_all_<species>.png` | **All brain versions** (one panel per version, own scale; `make_learning_curves.py`) – used in the 10-minute presentation |
| `figures/locomotion/<run>/progress.gif` | All checkpoints back to back (short) |
| `figures/locomotion/<run>/contact_sheet.png` | One frame per checkpoint |
| `figures/locomotion/<run>/metrics.png` / `.csv` | Speed / turn tracking, restart from standstill, falls per checkpoint |
| `figures/locomotion/<run>/<N>_steps.gif` | 12-s clip of each checkpoint |
| `figures/gait/<run>_footfall_<speed>.png` | Footfall diagrams (dark = foot on the ground); v1 vs v2 is the clearest before/after slide |
| `figures/gait/*_legs_v2_<N>_steps_footfall_*.png` | Gait during v2 training (1M, 1.5M, 2.75M, 3.5M) |
| `figures/myr_legs_pilot2_1M.gif` | Pilot 2 standing still at 1M steps |
| `videos/showcase_side_by_side.mp4` | **Showcase, 5:00**: Myrranox (left) and DC (right) living in the habitat, everything learned (predator + walking brains), steady side cameras with hunger/fatigue – rendered from `blender/habitat_showcase.blend` (DC walking v2) |
| `videos/showcase_side_by_side_v4_no_rocks.mp4` | **Showcase v4 without rocks, 5:00 (latest)**: both on walking v4, habitat without rocks (`sim/habitat_no_rocks.json`) |
| `videos/showcase_side_by_side_v4.mp4` | **Showcase v4, 5:00**: both creatures on walking v4 (precise + fluid), rock-dodging side cameras |
| `videos/showcase_side_by_side_v3.mp4` | **Showcase v3, 5:00**: the same with the alive DC (walking v3) and Myrranox v2 + passive parts; cameras dodge rocks |
| `videos/myr_legs_v10_training.mp4`, `videos/myr_legs_v5_vs_v10.mp4` | **Myrranox walking v10**: whole training (0 → 3M) and v5 vs v10 on the same commands |
| `videos/showcase_v6_30s.mp4`, `figures/showcase_v6_still.jpg` | **Showcase v6 (30 s)**: both creatures with their final walkers (Myrranox v10, DC v12z) |
| `presentation/Endemic_Creatures_RL.pptx` | Project presentation (German, updated 28 Oct) |
| `figures/dc_physics_body_v11_vs_v12.jpg`, `figures/dc_hand_*.png`, `figures/dc_*arms*.png`, `figures/wolf_reference_frames.png` | **DC v12**: detailed body, real hand vs physics hand, T-rex arms, sprint pose, wolf reference frames |
| `videos/myr_legs_v4_vs_v5.mp4`, `videos/dc_legs_v4_vs_v5.mp4`, `videos/*_legs_v5*_training.mp4` | **Walking v5 (natural style)**: insect knees / panther stride before vs after; whole v5 trainings |
| `videos/*_legs_v4_training.mp4`, `videos/myr_legs_v2_vs_v4.mp4`, `videos/dc_legs_v3_vs_v4.mp4` | **Walking v4 (precise + fluid)**: whole fine-tuning (4M → 7M, every 250k) next to the learning curve; before / after on the same commands |
| `figures/locomotion/*_legs_v4/`, `figures/gait/*_legs_v4_*` | v4 progress graphs, checkpoint clips, footfall diagrams (incl. Myrranox spot turn `myr_legs_v4_footfall_0.00.png`) |
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
    A sleeping laptop pauses it too (14 Oct: twice, about 6 h and 1.5 h lost) – keep it plugged in and awake.
13. (v12) Scope every rule to the movement it describes: the wolf's walking rules (low paws, two paws down) applied
    in the sprint and set its quality to zero.
14. (v12) A new skill that earns nothing at full penalties makes the brain unlearn the old one (v12q stood still at
    walking commands): keep practising the old skill and ramp the new skill's penalties.
15. (v12) Check the geometry before training a behaviour: the arms could not reach the ground, and the hand's joint
    sat in the wrong place – no reward could have taught a flat palm.
16. (v12) When a body pose changes under a trained brain, shift its observation reference by the same amount;
    otherwise it no longer recognises its own posture and stands still.

## 9. Next (open)

- Showcase v6: render the full 5 minutes (about 2.2 h on the laptop, or minutes on a machine with a dedicated GPU).
- Walking is final for both creatures (Myrranox v10, DC v12z) – no further leg training unless the owner asks.
- Hunting: the brains are still the 2D-trained ones; fine-tuning with the physics bodies (plan D2), fatigue balance
  and the speed gap (DC about twice as fast) are open (owner decisions).
- Raids and territory defence (not learned); alternating rounds against the other species' trained brain.
- Final scene: 2 Myrranox + 2 DC; terrain/props; relink the Myrranox textures and re-render.
- Paper and presentation (deadline and format still open).
