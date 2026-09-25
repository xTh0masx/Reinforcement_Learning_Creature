# Locomotion training log (walking learned with RL)

One entry per training session, oldest first.

| Material | Where |
| --- | --- |
| **Whole-session videos** (untrained → 4M steps, every 250k checkpoint next to the learning curve) | `docs/videos/myr_legs_v1_training.mp4`, `docs/videos/dc_legs_v1_training.mp4` |
| Learning curves of all sessions (incl. abandoned pilots) | `docs/figures/locomotion/learning_curves_<species>.png`, live: `tensorboard --logdir runs` |
| Per-checkpoint clips, progress GIF, contact sheet, metrics | `docs/figures/locomotion/<run>/` |
| Final brains | `models/<run>.zip` + `models/<run>_vecnormalize.pkl` |

Regenerate: `python locomotion_progress.py --run <run> --species <species>` and
`python make_training_video.py --run <run> --species <species>`. Watch a run live: `python watch_locomotion.py ...`.

Setup for all sessions: MuJoCo body from `sim/bodies.py`, `LocomotionEnv` (50 Hz control, 20 s episodes, random
speed/turn commands every 2–5 s, 15 % standing), PPO with 8 parallel environments on CPU (8 cores, no GPU).

## Sessions

| # | Date | Run | Species | Steps | What changed | Result |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 25 Sep 2026 | `myr_legs_pilot` | Myrranox | 0.33M (stopped) | First attempt | Reward −3300 → −1800 per episode. The foot-slip penalty (−2.3 per step with the untrained policy's jitter) dwarfed the reward for following the command (max 1.5), and the claws lying on the ground counted as "body contact". Stopped. |
| 2 | 25 Sep 2026 | `myr_legs_pilot2` | Myrranox | 1.1M (stopped) | Foot-slip penalty ÷5 and capped; only torso contacts penalised | Learned to **stand still**: at 1M steps it ignored every walk/turn command (speed 0.00 m/s at a 0.6 m/s command) — a local optimum, because the speed reward was too narrow (σ = 0.3 m/s) and penalties at full strength made moving costly. GIF: `docs/figures/myr_legs_pilot2_1M.gif`. Checkpoints were not kept. |
| 3 | 25 Sep 2026 | `dc_legs_pilot` | DC | ~0.1M (stopped) | Same reward as #2 | Stopped together with #2 because it would hit the same problem. |
| 4 | 25 Sep 2026 | `myr_legs_v1` | Myrranox | **4M** (≈3.5 h, shared CPU) | Speed reward σ = ½ max speed and weight 2.0; turn reward σ = ½ max turn rate; penalties ramped 20 % → 100 % over the first 40 % of training; checkpoints every 250k from step 0 | **Walks.** 0–0.5M: stands still. 0.75M: walks if already moving but stops for good once it slows down. 1.0M: walks (0.56 m/s at 0.6 m/s) and restarts from standstill. 1.5M+: speed error ≈ 0.1 m/s. Final: see below. Reward 1730 / episode at 4M. |
| 5 | 25 Sep 2026 | `dc_legs_v1` | DC | **4M** (≈3.5 h, shared CPU) | Same as #4 | **Walks and trots.** Moves from 250k but unstable in the middle of training (falls at 0.75M, 1.25M, 1.5M; no restart from standstill at 2.25M). Final: no falls, restarts, trots at 2.5 m/s; turning weak. Reward 1950 / episode at 4M. |

| 6 | 25 Sep 2026 | `myr_legs_v2` | Myrranox | **4M** | **Natural gait (v2):** per-leg gait clock in the observation; rewards for following the insect tripod (stance legs down + swing legs up, scored so that standing still earns nothing), foot clearance 3 cm, air time per step, energy (torque × joint speed), joint acceleration; turn reward 0.5 → 1.0; 10 % turn-on-the-spot and 10 % backwards commands; mirror-symmetry loss (`SymmetricPPO`) | **Tripod gait**, all 6 legs step at 2.3–3.7 steps/s (v1: 3 legs tapping at 9–12 Hz). Walking turns now symmetric (+57 / −54 °/s at ±54, v1: +56 / −31). ~5× less power (97 W at full speed). Mid-training: speed dip at 1.5M (penalties at full strength), front-right leg held up at 2.75M, fixed by 3.5M. Still weak: turning on the spot, walking backwards, walk speed 0.53 of 0.6 m/s. |
| 7 | 25 Sep 2026 | `dc_legs_v2` | DC | **4M** | Same as #6 (walk → trot gait) | **Diagonal walk / trot**: hind legs alternate (v1 hopped with both), diagonal pairs together; 1.7 steps/s walking, 2.8 trotting; asymmetry 1.3° at trot. Turns ~65 % of the command (v1 ~25 %). ~2.4× less power (1.56 kW at full speed). No falls after 1.25M. Speed below command: 0.68 of 0.8, 2.15 of 2.5 m/s. |

Reward values are **not comparable between sessions** — the reward definition changed each time.
Compare sessions by the evaluation metrics (`metrics.csv`: speed/turn tracking error, falls, foot slip).

## Final brains (4M steps) – `python play_locomotion.py models/<run> --species <species>`

Fixed command script, deterministic brain, seed 0; values averaged after the first second of each command.

| Command | Myrranox `myr_legs_v1` | DC `dc_legs_v1` |
| --- | --- | --- |
| Walk (0.6 / 0.8 m/s) | **0.61** m/s | **0.76** m/s |
| Full speed (1.2 / 2.5 m/s) | **1.19** m/s | **2.48** m/s |
| Walk + turn left (+54 / +36 °/s) | 0.59 m/s, **+56** °/s | 0.75 m/s, +10 °/s |
| Walk + turn right (−54 / −36 °/s) | 0.53 m/s, −31 °/s | 0.77 m/s, −8 °/s |
| Turn on the spot (+72 °/s, Myrranox only) | +9 °/s (fails) | – |
| Backwards (−0.3 m/s, Myrranox only) | −0.02 m/s (fails) | – |
| Stop | 0.01 m/s | 0.01 m/s |
| Restart from standstill | **0.60** m/s | **0.75** m/s |
| Falls | none | none |

**Strengths:** both follow forward speed commands within ~0.05 m/s over the whole range (DC including the 2.5 m/s
trot), stop and restart reliably and never fell in the evaluation.
**Weaknesses:** turning. The Myrranox turns left well but right only at ~60 %, and cannot turn on the spot or walk
backwards; the DC barely turns (≈25 % of the command). In the habitat the navigation brain will therefore take wider
curves than it asks for.

## v1 vs v2 (final brains, same command script, seed 0)

| | Myrranox v1 | Myrranox v2 | DC v1 | DC v2 |
| --- | --- | --- | --- | --- |
| Walk (0.6 / 0.8 m/s) | 0.61 | 0.53 | 0.76 | 0.68 |
| Full speed (1.2 / 2.5 m/s) | 1.19 | 1.31 | 2.48 | 2.15 |
| Turn left / right (±54 / ±36 °/s) | +56 / −31 | **+57 / −54** | +10 / −8 | **+24 / −22** |
| Turn on the spot / backwards (Myrranox) | fails / fails | fails / fails | – | – |
| Legs used, rhythm | 3 legs, 9–12 Hz tapping | **tripod, all 6, 2.3–3.7 Hz** | hind legs hop together | **diagonal walk/trot, 1.7–2.8 Hz** |
| Left/right asymmetry (walk) | 11.9° | 6.0° (2.2° at full speed) | 11.6° | 6.9° (1.3° at trot) |
| Power at full speed | ~450 W at walk | **97 W** | 3.8 kW | **1.56 kW** |
| Falls | none | none | none | none |

Footfall diagrams: `docs/figures/gait/<run>_footfall_<speed>.png`. v2 trades a little speed accuracy for a natural,
symmetric, far more efficient gait. Open: Myrranox turn on the spot / backwards; both slightly slower than commanded.

## Lessons

1. A penalty that is large for the untrained policy's jitter (foot slip) swamps the task reward.
2. A narrow tracking reward plus full-strength penalties makes **standing still** a local optimum;
   a wider reward and a penalty curriculum fixed it.
3. The policy learns to keep walking before it learns to **start** walking; the evaluation checks both.
4. Forward speed is learned first and best; **turning** is the hardest part — the turn reward (weight 0.5) is small
   next to the speed reward (2.0), and only 70 % of training commands contain a turn.
5. (v2) A gait reward must not pay for standing still: "fraction of legs matching the clock" gave standing 55–65 %
   for free; scoring stance legs and swing legs separately fixed that (checked before training, `tests/test_locomotion_v2.py`).
6. (v2) Mirror symmetry fixed the one-sided turning of v1 (Myrranox right turns 31 → 54 °/s).
7. (v2) Mid-training setbacks recover (speed dip at 1.5M, a raised leg at 2.75M) — judge a run by its end, keep all checkpoints.

## Next ideas (one change per run)

- `*_legs_v3`: more turn-on-the-spot / backwards commands for the Myrranox (both still fail), continue from v2
  (`train_locomotion.py --resume models/<run> --version 2`).
