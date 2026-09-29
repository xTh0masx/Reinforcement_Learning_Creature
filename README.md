# Endemic Creatures – Reinforcement Learning

Two predator species (Myrranox and DC_SKIN_RAW) learn to live like endemic creatures in a habitat:
stay in their home range, hunt prey, rest and defend their territory. **Both the decisions and the
walking are learned with reinforcement learning** (Gymnasium + Stable-Baselines3 PPO):

- **Navigation brain** (`sim/creature_env.py`) – 2D habitat, decides where to go: forward speed + turn rate every 0.1 s.
- **Locomotion brain** (`sim/locomotion_env.py`) – MuJoCo physics body, moves the leg joints at 50 Hz to follow that command.

The recorded body pose and joint angles are played back on rigged models in Blender.
Full design: [PROJECT_PLAN.md](PROJECT_PLAN.md) · training history: [docs/locomotion_training_log.md](docs/locomotion_training_log.md).

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ (developed on 3.12, CPU only). Blender 5.x with the Blender MCP add-on for the Blender steps.

## Quick checks

```bash
python cartpole_smoke_test.py                  # PPO on CartPole-v1 reaches 500
python check_env.py                            # check_env + one random-action episode in the viewer
python check_env.py --species dc --no-render
python -m sim.bodies                           # both physics bodies stand in their rest pose
```

## Train

```bash
# Locomotion (walking) brains - ~1 h each on 8 CPU cores, checkpoints every 250k steps from step 0
python train_locomotion.py --species myrranox --steps 4e6 --run myr_legs_v2 --version 2    # natural gait
python train_locomotion.py --species dc       --steps 4e6 --run dc_legs_v2  --version 2

# Predator (navigation) brains: against a scripted rival, then against the other species' frozen brain
python train.py --species myrranox --rival dc --steps 3e6 --run myr_brain_v1_r1
python train.py --species myrranox --rival dc --rival-brain models/dc_brain_v1_r1 --resume models/myr_brain_v1_r1 --steps 3e6 --run myr_brain_v1_r2

tensorboard --logdir runs                      # live curves: rollout/ep_rew_mean, train/penalty_scale
```

Add one row per run to `docs/experiments.csv`.

## Evaluate and present

```bash
# One brain on a fixed command script (walk, full speed, turns, stop, restart) -> numbers + GIF
python play_locomotion.py models/myr_legs_v1 --species myrranox --gif docs/figures/myr_legs_v1.gif

# Whole training run: clip per checkpoint, progress GIF, contact sheet, metrics, learning curves
python locomotion_progress.py --run myr_legs_v1 --species myrranox
```

Output goes to `docs/figures/locomotion/<run>/`.

```bash
# Whole training session as a video: every checkpoint (untrained -> 4M) next to the learning curve
python make_training_video.py --run myr_legs_v1 --species myrranox     # -> docs/videos/myr_legs_v1_training.mp4

# Watch a run live while it trains (MuJoCo window, switches to each new checkpoint)
python watch_locomotion.py --run myr_legs_v1 --species myrranox
```

## Evaluate predator brains

```bash
python evaluate.py --species dc --brain models/dc_brain_v4_r1 --rival myrranox        # section 12 checks
python evaluate.py --species dc --brain scripted --rival myrranox                     # baseline
python watch_predator.py --run dc_brain_v4_r1 --species dc --rival myrranox            # live top-down view
```

## Showcase: both predators with real legs

```bash
python record_world.py --minutes 5 --pick-seed 8 --habitat sim/habitat_no_rocks.json --out recordings/showcase_v4_no_rocks.csv
```

In Blender import it (below), then use the sidebar panel **Creatures** (N key). Render the side-by-side video:

```bash
blender -b blender/habitat_showcase.blend --python blender/scripts/render_camera.py -- --camera SideCam_myrranox_0 --out renders/showcase_v4_no_rocks/myrranox
blender -b blender/habitat_showcase.blend --python blender/scripts/render_camera.py -- --camera SideCam_dc_0 --out renders/showcase_v4_no_rocks/dc
python make_side_by_side.py --left renders/showcase_v4_no_rocks/myrranox --right renders/showcase_v4_no_rocks/dc --out docs/videos/showcase_side_by_side_v4_no_rocks.mp4
```

## Record and play back in Blender

```bash
# Both brains together; --nav omitted = scripted stand-in for the navigation brain
python record_physics.py --species myrranox dc --loco models/myr_legs_v1 models/dc_legs_v1 --seconds 120 \
    --out recordings/walk_v1.csv
```

In Blender (`blender/habitat_showcase.blend`, rigs already built), run in the Python console / via MCP:

```python
repo = r"C:/Users/thoma/PycharmProjects/Reinforcement_Learning_Creature"
g = {}; exec(open(repo + "/blender/scripts/build_habitat.py").read(), g); g["build"](repo + "/sim/habitat.json")
g = {}; exec(open(repo + "/blender/scripts/import_trajectories.py").read(), g); g["import_csv"](repo + "/recordings/walk_v1.csv")
```

Space plays it; timeline markers switch between the follow cameras. Rigging a fresh copy of the models:
`rig_myrranox.py` / `rig_dc.py` (`rig(repo)`).

Debug stand-ins (not RL): `record.py --policy heuristic` (scripted navigation, gliding models) and
`record_physics.py --loco sine sine` (open-loop leg oscillation).

## Layout

| Path | Content |
| --- | --- |
| `sim/habitat.json` | The map (boundary, home zones, water, rest spots, rocks, prey), shared with Blender |
| `sim/habitat.py` | Map loading, free-space checks, obstacle ray casts |
| `sim/species.py` | Per-species movement limits and need rates |
| `sim/creature_env.py` | Navigation env (one creature, training stage 1); `step(action, body=...)` accepts a physics pose |
| `sim/viewer.py` | matplotlib top-down live view |
| `sim/bodies.py` | MuJoCo bodies (joint points measured on the Blender models) → MJCF |
| `sim/locomotion_env.py` | Locomotion env: command tracking with the physics body |
| `train.py`, `train_locomotion.py` | PPO training (navigation / locomotion) |
| `play_locomotion.py`, `locomotion_progress.py` | Evaluation, GIFs, progress material |
| `make_training_video.py`, `watch_locomotion.py` | Training-session videos (`docs/videos/`), live viewer |
| `make_comparison_video.py`, `gait_analysis.py` | Two brains side by side (v1 vs v2); footfall diagrams |
| `calibrate_energy.py` | Power curve of a locomotion brain → `sim/energy.json` (hunger/fatigue in 2D training) |
| `sim/gaits.py`, `sim/symmetry.py` | v2 target gaits; mirror-symmetry loss (`SymmetricPPO`) |
| `sim/physics_world.py` | MuJoCo world from `habitat.json` with several creatures and their brains |
| `sim/scripted.py`, `sim/nav_brain.py` | Scripted rival predator; trained navigation brain for playback |
| `tests/test_locomotion_v2.py` | Mirror-simulation and gait-reward checks |
| `record.py`, `record_physics.py` | Recordings for Blender (10 Hz CSV + 50 Hz pose CSV per creature) |
| `blender/scripts/` | `build_habitat.py`, `import_trajectories.py`, `rig_myrranox.py`, `rig_dc.py`, `rig_common.py` |
| `blender/habitat_showcase.blend` | Scene with habitat, rigs and cameras (Git LFS); `_before_rig.blend` = backup |
| `models/`, `recordings/` | Released brains and recordings |
| `docs/` | `experiments.csv`, `locomotion_training_log.md`, `figures/`, `videos/` (MP4 via Git LFS) |
| `runs/`, `checkpoints/`, `logs/` | TensorBoard logs, checkpoints, training stdout (gitignored) |

## Status (26 Sep 2026)

| Part | State |
| --- | --- |
| Habitat sim, rules (territories, raids, energy-driven hunger), physics habitat world | done, tested |
| Walking brains | **v4 released** (`models/*_legs_v4`): precise + fluid – Myrranox turns on the spot, DC turns ±30 °/s, ~25 % smoother; built on v2 (natural gait) and, for the DC, v3 (alive body) |
| Predator brains | **DC `models/dc_brain_v4_r1`** (4/5 section 12 checks); **Myrranox `models/myr_brain_v3_r1`** (hunts, but starves when its home is empty; fails with real legs) |
| Blender | rigs, pose playback (0.01 mm), steady side cameras, overview, panel "Creatures", hunger/fatigue display |
| Showcase | `docs/videos/showcase_side_by_side_v4_no_rocks.mp4` (5:00, both on walking v4, no rocks; with rocks `showcase_side_by_side_v4.mp4`; earlier versions `showcase_side_by_side_v3.mp4`, `showcase_side_by_side.mp4`) |
| Open | raids/defence, Myrranox turning on the spot, 2+2 creatures, textures, paper/slides |

What happened and all results: **[docs/RESULTS.md](docs/RESULTS.md)** · changes: [docs/CHANGELOG.md](docs/CHANGELOG.md) · time spent: [docs/TIME_LOG.md](docs/TIME_LOG.md) ·
walking: [docs/locomotion_training_log.md](docs/locomotion_training_log.md) ·
predators: [docs/predator_training_log.md](docs/predator_training_log.md).

Note: training runs inside the Claude Code session; closing the app stops it. Resume with
`train.py --resume checkpoints/<run>_<N>_steps --steps <remaining>`.
