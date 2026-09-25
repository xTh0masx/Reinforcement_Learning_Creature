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

## Lessons

1. A penalty that is large for the untrained policy's jitter (foot slip) swamps the task reward.
2. A narrow tracking reward plus full-strength penalties makes **standing still** a local optimum;
   a wider reward and a penalty curriculum fixed it.
3. The policy learns to keep walking before it learns to **start** walking; the evaluation checks both.
4. Forward speed is learned first and best; **turning** is the hardest part — the turn reward (weight 0.5) is small
   next to the speed reward (2.0), and only 70 % of training commands contain a turn.

## Next ideas (one change per run)

- `*_legs_v2`: raise the turn reward weight (0.5 → 1.0) and add more turn-only / backward commands; continue from v1
  (`train_locomotion.py --resume models/<run>`).
