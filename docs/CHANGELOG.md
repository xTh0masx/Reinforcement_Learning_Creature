# Changelog

All changes and owner decisions, newest first. Commit hashes refer to branch `rl-locomotion-setup`
(tags: `legs-v1`, `legs-v2`). Details: [RESULTS.md](RESULTS.md), [locomotion_training_log.md](locomotion_training_log.md),
[predator_training_log.md](predator_training_log.md), [experiments.csv](experiments.csv),
time spent per step: [TIME_LOG.md](TIME_LOG.md).

## 26 Sep 2026

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
