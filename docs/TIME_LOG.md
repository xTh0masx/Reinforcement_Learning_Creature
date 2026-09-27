# Time log – from project start to the showcase video

Reconstructed from recorded timestamps: git commits, TensorBoard event times of every training run, file times of
models, recordings, videos and rendered frames, and the Claude Code session. Times are local (CET), 2026.
Working sessions = the Claude Code session was open and work was going on; compute = the machine was training or
rendering (much of it ran in parallel with development work, so the two overlap).

## Summary

| | Time |
| --- | --- |
| Calendar time, first commit → showcase video + docs | 23 Sep 15:48 → 26 Sep ~18:30 = **3 days ≈ 75 h** |
| Working sessions (3) | **≈ 26 h** (23 Sep ~1 h, 25 Sep ~14.5 h, 26 Sep ~10.5 h) |
| Compute: walking training | **≈ 7.1 h** wall time (2 runs in parallel) – 17.6M simulation steps |
| Compute: predator training | **≈ 5.5 h** wall time (2 runs in parallel) – 24.5M simulation steps |
| Compute: rendering the showcase video | **≈ 5.7 h** (2 × 9001 frames) |
| Idle (session closed overnight, runs interrupted) | 25 Sep 23:58 → 26 Sep 08:05 ≈ 8 h |

For comparison, the project plan estimates 9–13 weeks for the whole project (section 10). Done so far: phases 0–1,
walking (new phase 1b), the physics habitat, the predator rules, predator brains and a first showcase – not yet: the
2+2 final scene, the paper and the presentation.

## Session 1 – 23 Sep (≈ 1 h)

| Time | Duration | Step | Result |
| --- | --- | --- | --- |
| 15:48 | – | Project plan + initial commit | `PROJECT_PLAN.md` |
| ~16:00 – 16:30 | 0.5 h | Repo structure, `requirements.txt`, species limits, habitat (`habitat.json`), `CreatureEnv`, viewer, `check_env` | env passes `check_env` |
| 16:27 – 16:28 | 1 min compute | PPO on CartPole (smoke test) | 500.0 |
| ~16:30 – 16:45 | 0.25 h | Blender: habitat builder, trajectory import, follow cameras, scripted preview recording | preview plays in Blender |

## Session 2 – 25 Sep (≈ 14.5 h, 09:30 – 23:58)

| Time | Duration | Step | Result |
| --- | --- | --- | --- |
| ~09:30 – 10:45 | 1.25 h | Blender scene restored and saved; decision: walking learned by RL; models measured; MuJoCo bodies; `LocomotionEnv`; Myrranox rig | bodies stand, rig verified |
| 10:47 – 11:03 | 0.3 h compute | Walking pilots 1–3 | failed, lessons (penalty too strong, standing still) |
| 11:05 – 14:35 | **3.5 h compute** | Walking **v1** training (both species in parallel, 4M steps each) | both walk |
| (in parallel) | | DC rig, pose playback in Blender (0.01 mm), `record_physics.py`, progress/video tooling, docs | |
| 14:35 – 15:33 | 1 h | v1 videos, gait analysis (footfall), commit `e6020d9` / `0bcb91e` | v1 documented, tag `legs-v1` |
| 15:41 – 19:00 | **3.3 h compute** | Walking **v2** training (natural gait) | tripod / walk→trot |
| (in parallel) | | v2 gait rewards + symmetry (tested), B physics habitat world, C predator rules and balance tests; owner away ~1 h | |
| 19:00 – 20:06 | 1.1 h | v2 evaluation, videos, v1 vs v2 comparison, `RESULTS.md`, commits `fdb28dc` / `070803f` | tag `legs-v2` |
| 19:48 – 21:07 | 1.3 h compute | Predator attempt 1 (plan reward) | failed – reward did not prefer hunting |
| 21:07 – 21:47 | 0.7 h | Reward v2 designed and tested | hunting ranked first |
| 21:47 – 22:59 | 1.2 h compute | Predator attempt 2 | failed – brains stood still |
| 22:59 – 23:13 | 0.25 h | Exploration fixes + 400k pilot (10 min compute) | learns to hunt |
| 23:14 – 23:58 | 0.75 h compute | Predator round 1 started | **interrupted** when the session closed |

## Session 3 – 26 Sep (≈ 10.5 h, 08:03 – ~18:30)

| Time | Duration | Step | Result |
| --- | --- | --- | --- |
| 08:03 – 08:50 | 0.75 h compute | Round 1 resumed from the 1.5M checkpoint; live habitat viewer | finished 3M steps |
| 08:50 – 09:58 | 1.1 h | Evaluation (`evaluate.py`), option B implemented and tested | hunts, but no raids/rest/defence |
| 09:59 – 11:14 | **1.26 h compute** | Predator **option B** training | DC much better, Myrranox worse |
| 11:14 – 11:34 | 0.3 h | Evaluation, **DC brain released**, Myrranox kept (owner) | `dc_brain_v4_r1` |
| 11:34 – 12:07 | 0.55 h | Showcase recording with real legs, Blender import, energy-model fix, side cameras + panel, prey half speed, re-recording | showcase in Blender |
| 12:09 – 17:51 | **5.7 h compute** | Rendering both side cameras (9001 frames each) | frames done |
| 17:51 – 18:00 | 0.15 h | Side-by-side video assembled | `showcase_side_by_side.mp4` (5:00) |
| 18:00 – ~18:30 | 0.5 h | Documentation: predator log, changelog, results, README, plan, this time log | commits `d3e40c5`, `60dee38` |

## Session 4 – 27 Sep (excerpt)

Myrranox v3b / v3c retries: see `locomotion_training_log.md` (sessions 8b, 8c).

| Time | Duration | Step | Result |
| --- | --- | --- | --- |
| ~14:30 | 0.1 h | Side cameras dodge rocks, showcase re-imported, test still | `c40c6b2` |
| 14:35 – 18:54 | **4.3 h compute** | Rendering both side cameras of showcase v3 (9001 frames each) | frames done |
| 18:54 – 19:07 | 0.2 h | Side-by-side video assembled | `showcase_side_by_side_v3.mp4` (5:00) |

## Compute per training run

| Run | Start → end | Wall time | Steps | Outcome |
| --- | --- | --- | --- | --- |
| cartpole_smoke | 23.09 16:27 → 16:28 | 1 min | 0.1M | 500 |
| myr/dc_legs_pilot, pilot2 | 25.09 10:47 → 11:03 | 0.3 h | 1.56M | failed (lessons) |
| myr_legs_v1, dc_legs_v1 | 25.09 11:05 → 14:35 | 3.5 h | 2 × 4M | walk |
| myr_legs_v2, dc_legs_v2 | 25.09 15:41 → 19:00 | 3.3 h | 2 × 4M | natural gait |
| *_brain_v1_r1 | 25.09 19:48 → 21:07 | 1.3 h | 2 × 3M | failed |
| *_brain_v2_r1 | 25.09 21:47 → 22:59 | 1.2 h | 2 × 2.9M | failed |
| pilot_*_brain | 25.09 23:03 → 23:13 | 0.2 h | 2 × 0.4M | learns |
| *_brain_v3_r1 | 25.09 23:14 → 23:58, 26.09 08:05 → 08:50 | 1.5 h (+8 h idle) | 2 × 3M | hunts |
| *_brain_v4_r1 | 26.09 09:59 → 11:14 | 1.26 h | 2 × 3M | DC released |
| Showcase render | 26.09 12:09 → 17:51 | 5.7 h | 18 002 frames | video |
| Showcase v3 render | 27.09 14:35 → 18:54 | 4.3 h | 18 002 frames | video |

Speeds on this PC (8 CPU cores, no CUDA GPU): walking training ≈ 650–850 steps/s per run with two runs in parallel,
predator training ≈ 550–650 steps/s per run, EEVEE render ≈ 0.8 frames/s for two cameras in parallel. Everything was
slower whenever live viewers or TensorBoard were open.
