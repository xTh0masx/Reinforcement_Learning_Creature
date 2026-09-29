# Changelog

All changes and owner decisions, newest first. Commit hashes refer to branch `rl-locomotion-setup`
(tags: `legs-v1`, `legs-v2`). Details: [RESULTS.md](RESULTS.md), [locomotion_training_log.md](locomotion_training_log.md),
[predator_training_log.md](predator_training_log.md), [experiments.csv](experiments.csv),
time spent per step: [TIME_LOG.md](TIME_LOG.md).

## 29 Sep 2026

**Showcase v4 without rocks** (`e5d01f1`, owner: "redo the video but delete the rocks")
- Rocks removed from the physics world too (not only hidden), so the creatures do not steer around invisible rocks:
  `sim/habitat_no_rocks.json` (copy of `habitat.json` with no obstacles; training still uses `habitat.json`),
  `record_world.py --habitat`, `import_trajectories.import_csv(path, habitat_path)`; the 14 rocks were deleted from
  `blender/habitat_showcase.blend` (the version with rocks is in git history).
- Recording `recordings/showcase_v4_no_rocks.csv` (5 min, seed 8): no falls, Myrranox 3 prey, DC 4 prey.
  Side cameras: 0 side changes (Myrranox), 1 (DC, wall).
- `docs/videos/showcase_side_by_side_v4_no_rocks.mp4`: 5:00, 2 × 9001 frames. Render 28 Sep 18:25 → 21:50, paused overnight (laptop closed; resumed from the
  existing frames), 29 Sep 08:11 → 09:02, stitched 09:10 – 4.3 h render in total. Still
  `docs/figures/showcase_side_by_side_v4_no_rocks.jpg`.
- Known: the DC looks dark when it walks with the sun behind it from the camera's view (lighting, not rocks).

## 28 Sep 2026

**Showcase v4 rendered** (`7f4aeba`, owner: "yes start")
- Re-recorded with walking v4 (`recordings/showcase_v4.csv`, seed 8, 5 min): no falls, DC 6 prey, Myrranox 2 prey
  and survives. Side cameras changed side 2× (Myrranox) and 5× (DC) to avoid rocks.
- `docs/videos/showcase_side_by_side_v4.mp4`: 5:00, 2 × 9001 frames, render 13:13 → 18:05 (4.9 h), stitched by 18:17. Still
  `docs/figures/showcase_side_by_side_v4.jpg`. The v3 and first showcase videos are kept.
- Known: prey balls can pass in front of a camera (only rocks and walls are avoided); after a side change the camera
  can face against the sun and the creature looks dark.

**Walking v4 – precise and fluid** (`88d54ec`, owner request; both brains released)
- `sim/locomotion_env.py` version 4: joint-target low-pass filter (`ACTION_FILTER` 0.5), tracking tolerance curriculum
  (`set_track_scale`, `train_locomotion.py` `TrackingCurriculum`, `--track-end`), turn weight 1.5, soft-landing
  penalty, ramped commands (`COMMAND_RAMP_S`), 20 % spot turns / 10 % backwards, head style scaled by tracking,
  spot-turn speed reward scaled by the turn; body per species (`ACTIVE`, `body_for`).
- Brain version stored in the brain (`loco_version`), because Myrranox v4 has the v2 observation size;
  `PhysicsWorld` builds bodies via `body_for`.
- `precision_metrics.py` (command tracking per segment), `alive_metrics.py` + joint acceleration / target jumps.
- `train_locomotion.py --resume` saves the starting brain as the first checkpoint; progress tools ignore `<run>_pilot`.
- Pilot 0.5M each (passed), then `models/myr_legs_v4`, `models/dc_legs_v4` (+3M each, 1.1 h). Myrranox turns on the
  spot for the first time (58 of 72 °/s), DC turns ±30 of 36 °/s (was 21), both ~25 % smoother. `record_world.py`
  walks with v4 by default. Videos: `docs/videos/*_legs_v4_training.mp4`, `myr_legs_v2_vs_v4.mp4`,
  `dc_legs_v3_vs_v4.mp4`.

## 27 Sep 2026

**Showcase v3 rendered** (`c40c6b2`, owner: "yes")
- Side cameras avoid rocks: `import_trajectories.py` checks 1.5 s ahead whether a rock or the wall lies between the
  side camera and the creature; if so and the other side is free, the camera orbits to the other side (1.5 s, stays
  ≥ 3 s on a side). In the showcase: DC camera 3 side changes, Myrranox 0.
- `docs/videos/showcase_side_by_side_v3.mp4`: 5:00, Myrranox walking v2 + passive parts (left), DC walking v3 alive body (right); 2 × 9001 frames,
  render 14:35 → 18:54 (4.3 h). Still: `docs/figures/showcase_side_by_side_v3.jpg`. The first showcase
  (`showcase_side_by_side.mp4`, DC v2) is kept for comparison.

## 26 Sep 2026

**27 Sep – Myrranox v3 attempts, showcase with the alive DC** (owner decisions A, then C + A2)
- `myr_legs_v3b` (resume, 10× weaker head penalty) and `myr_legs_v3c` (resume, penalty frozen at 0.6): both failed –
  the Myrranox drifts into standing still after ~1M steps. It stays on walking v2 + passive parts.
- Showcase re-recorded: DC walking v3 (spine, steady head, livelier tail) + Myrranox v2 + passive parts; preview
  `docs/videos/alive_stage2_preview.mp4`. `train_locomotion.py`: `--resume` checkpoints, `--curriculum-total`,
  `--penalty-fixed`, `--head-weight`; `record_world.py --walk-myr/--walk-dc`.
- Known: the side camera can end up behind a rock (fixed later the same day, see above).

**Stage 2 results:** DC `models/dc_legs_v3` released (faster, spine used, head −41 % wobble, tail swings more);
Myrranox v3 failed (learned to stand still once the head penalty ramped up) – stays on v2 + passive parts.
Videos: `docs/videos/*_legs_v3_training.mp4`, `*_legs_v2_vs_v3.mp4`; figures `docs/figures/locomotion/*_legs_v3/`,
`docs/figures/gait/*_legs_v3_*`; aliveness `docs/evaluations/alive_*.json`.

**"Alive" bodies, stage 2: walking v3** (`4032c93`, owner request)
- `sim/bodies.py active=True`: DC spine joint (`RearSegment`: rear body with hind legs and tail), actuated neck/head;
  Myrranox actuated head/abdomen; springs stay; per-joint motor gains and action ranges.
- `LocomotionEnv(version=3)`: head-steadiness and head-level rewards; generic mirror symmetry; tests for v3.
- `PhysicsWorld` builds the body matching each walking brain's version.
- Blender: DC `spine_rear` bone (hind legs + tail on it, proxy re-weighted); `alive_metrics.py`,
  `make_comparison_video.py --passive`; post-processing script for v3 (`logs/post_v3.sh`).
- Training `myr_legs_v3`, `dc_legs_v3` (4M steps each).

**"Alive" bodies, stage 1: passive body parts** (owner request: whole body should move, ant / panther-like)
- `sim/bodies.py` `PassiveChain`: spring-damper joints (no motor, not brain-controlled). Myrranox: abdomen
  (petiole: up/down + sideways) and head. DC: tail in 2 segments, neck + head, both extra arms – on the existing
  DC_rig bones, so the proxy mesh follows without new weights. Spring rest angles cancel the sag under gravity
  (`calibrate_springref`, < 0.2°). Optional (`passive=True`), so training setups stay reproducible.
- v2 walking brains run unchanged with the moving parts: no falls, same speed/turning. Swing in the walk test:
  DC tail tip up to 34–46°, Myrranox abdomen up to 17–20°, DC neck ≈ 1° (head stays steady).
- Recording stores the passive joint angles; rigs: new Myrranox bones `abdomen`, `head` (abdomen, fins, head, eyes,
  mandibles re-parented); DC bones get their joints (`rig_common.add_passive_bones`). Showcase re-recorded.

**Predator training videos** – `make_predator_video.py`, `docs/videos/predator/` (all 8 runs); `CreatureEnv` switches
`action_mapping="legacy"` and `tired_from` replay old brains under their training conditions.

**Showcase video** (`d3e40c5`)
- `docs/videos/showcase_side_by_side.mp4`: 5:00, Myrranox left / DC right, steady side cameras with hunger/fatigue.
- `blender/scripts/render_camera.py` (resumable background render per camera, 16 EEVEE samples, 960×540),
  `make_side_by_side.py` (joins two PNG sequences). `renders/` is gitignored. Render time ≈ 5.5 h for both cameras.

**Prey at half speed – owner decision** (`4cd5d37`)
- `sim/habitat.json`: wander 0.3 → 0.15 m/s, flee 0.7 → 0.35 m/s. Showcase re-recorded (5 min): the Myrranox now
  catches prey with its real legs and survives; the DC catches 3.
- Blender panel button "Both side by side" (after View → Area → Vertical Split: Myrranox camera left, DC right).
  Blender only splits areas interactively – a scripted split creates a broken zero-size area.

**Blender viewing – owner request** (`75098d4`)
- Steady **side camera** per creature (beside the body, position/heading smoothed over 0.7 s / 3 s) replaces the
  swinging follow cameras; **overview camera**; no automatic camera switching.
- **Panel "Creatures"** (3D view sidebar, N): side views, overview, follow a creature with free orbit, free view,
  play/pause. Stored in the .blend as a self-registering text block (`blender/scripts/view_panel.py`).
- Hunger/fatigue display placed from each camera's lens.

**Showcase recording, both predators with real legs** (`e7e31e6`)
- `record_world.py`: predator brains (Myrranox `myr_brain_v3_r1`, DC `dc_brain_v4_r1`) drive the v2 walking brains
  in one MuJoCo habitat (solid rocks/walls); 10 Hz tracks with hunger/fatigue + 50 Hz poses.
- `PhysicsWorld.step(record_poses=True)` returns every control step's pose.
- Hunger/fatigue in the physics world now use the same power curve as the 2D training (the instantaneous controller
  power starved the Myrranox in 132 s).
- Found: the Myrranox brain does not transfer to the physics body (cannot turn on the spot); the DC does.
  Owner decision: **show it as is** (option C).
- Blender importer: hunger/fatigue bars in the camera view.

**Predator brains released – owner decisions** (`9b0d0c4`)
- DC `models/dc_brain_v4_r1` released (4/5 section 12 checks). Myrranox: keep `models/myr_brain_v3_r1`, no fatigue change.

**Option B + evaluation** (`45f4a5c`)
- `evaluate.py`: section 12 checks over fixed seeds, scripted baseline, results to `docs/evaluations/`.
- Observation version 2 (29 values): + prey left at home, + rival inside home. Old brains still run (version detected).
- Training scenarios: 30 % lean days, scripted rival invades on 50 % of days (`ScriptedPredator(intrude_prob)`).
- Fatigue lowers top speed gradually from 70 %.
- Fix: scripted/heuristic policies read the obstacle rays by fixed position (`RAYS`), not "the last 8 values".
- `train.py --resume` also accepts checkpoints (continue an interrupted run); `watch_predator.py` live habitat view;
  viewer fix (a broken line break made `sim/viewer.py` unloadable).
- Round 1 (`*_brain_v3_r1`) interrupted at 1.75M by the end of the Claude session, resumed from 1.5M.

## 25 Sep 2026

**Predator training fixes** (`225dd49`, `fba6b6d`)
- Reward v2 (fed + every catch + potential-based hunting guide), tested with `tests/test_predator_reward.py`.
- Exploration fixes: full speed range, undiscounted guide, prey-flee curriculum (`PreyCurriculum`), hit hysteresis.
- Two failed rounds logged as lessons.

**Documentation, v1 vs v2 videos, slower prey** (`070803f`)
- `docs/RESULTS.md`; `make_comparison_video.py` (two brains side by side); prey flee 0.9 → 0.7 m/s (owner decision);
  `sim/locomotion_limits.json`; `train.py` rival options and `habitat/*` TensorBoard statistics.

**Walking v2, physics habitat, predator rules** (`fdb28dc`, tag `legs-v2`)
- Natural gait: gait clock, tripod / walk→trot rewards, foot clearance, air time, energy, joint acceleration,
  mirror-symmetry loss (`SymmetricPPO`); tests in `tests/test_locomotion_v2.py`.
- `sim/physics_world.py`: habitat walls/rocks as MuJoCo geometry, several creatures per world.
- `CreatureEnv`: rivals sharing the habitat, section 15 prey rules, energy-driven hunger/fatigue, defence rewards,
  sliding along obstacles; `sim/scripted.py`.

**Walking v1 results** (`0bcb91e`, tag `legs-v1`)
- Final v1 brains, whole-session training videos, gait analysis (footfall diagrams), live viewer.

**Setup and walking v1** (`e6020d9`)
- Habitat sim, viewer, MuJoCo bodies, locomotion env and training, rigs, Blender pose playback, docs.
- Owner decision: **walking is learned by RL** (MuJoCo).

## 23 Sep 2026
- Project plan; initial commit.
