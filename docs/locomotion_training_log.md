# Locomotion training log (walking learned with RL)

One entry per training session, oldest first. Curves: `docs/figures/locomotion/learning_curves_<species>.png`
(TensorBoard: `tensorboard --logdir runs`). Per-checkpoint clips, progress GIF, contact sheet and metrics:
`docs/figures/locomotion/<run>/` (made with `python locomotion_progress.py --run <run> --species <species>`).

Setup for all sessions: MuJoCo body from `sim/bodies.py`, `LocomotionEnv` (50 Hz, 20 s episodes, random
speed/turn commands every 2–5 s), PPO with 8 parallel environments on CPU.

| # | Date | Run | Species | Steps | What changed | Result |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 25 Sep 2026 | `myr_legs_pilot` | Myrranox | 0.33M (stopped) | First attempt | Reward −3300 → −1800 per episode. The foot-slip penalty (−2.3 per step with the untrained policy's jitter) dwarfed the reward for following the command (max 1.5), and the claws lying on the ground counted as "body contact". Stopped. |
| 2 | 25 Sep 2026 | `myr_legs_pilot2` | Myrranox | 1.1M (stopped) | Foot-slip penalty ÷5 and capped; only torso contacts penalised | Learned to **stand still**: at 1M steps it ignored every walk/turn command (speed 0.00 m/s at a 0.6 m/s command) — a local optimum, because the speed reward was too narrow (σ = 0.3 m/s) and penalties at full strength made moving costly. GIF: `docs/figures/myr_legs_pilot2_1M.gif`. Checkpoints were not kept. |
| 3 | 25 Sep 2026 | `dc_legs_pilot` | DC | ~0.1M (stopped) | Same reward as #2 | Stopped together with #2 because it would hit the same problem. |
| 4 | 25 Sep 2026 | `myr_legs_v1` | Myrranox | 4M | Speed reward σ = ½ max speed and weight 2.0; turn reward σ = ½ max turn rate; penalties ramped 20 % → 100 % over the first 40 % of training; checkpoints every 250k from step 0 | *running.* 0–0.5M: stands still. 0.75M: walks if already moving but stops for good once it slows down. **1.0M: walks** (0.56 m/s at a 0.6 m/s command) and restarts from standstill. 1.5–2.0M: speed error ≈ 0.1 m/s, full speed 1.2–1.3 m/s, no falls. |
| 5 | 25 Sep 2026 | `dc_legs_v1` | DC | 4M | Same as #4 | *running.* Moves from 250k but unstable: falls at 0.75M, 1.25M and 1.5M; cannot restart from standstill yet (2.25M). |

Reward values are **not comparable between sessions** — the reward definition changed each time.
Compare sessions by the evaluation metrics (`metrics.csv`: speed/turn tracking error, falls, foot slip).
