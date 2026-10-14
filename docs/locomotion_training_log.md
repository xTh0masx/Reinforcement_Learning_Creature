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

| 8 | 26 Sep 2026 | `myr_legs_v3` | Myrranox | 4M | **"Alive" body, stage 2:** head (2) and abdomen (2) get motors (their springs from stage 1 stay and carry the weight) → 22 actuators; new rewards: head steadiness (head rotation not explained by the commanded turn) and a level head; mirror symmetry over all joints | **Failed:** walked at 0.5–0.75M (0.5–0.6 m/s), then **learned to stand still** from 1.25M (0.00 m/s at every command) – once the penalty curriculum made the head-steadiness penalty strong, every step bounced the light head and standing was cheaper (same trap as pilot 2). Not released. |
| 8b | 27 Sep 2026 | `myr_legs_v3b` | Myrranox | 0.75M → 2.33M (stopped) | Continued #8 from its 0.75M checkpoint (still walking) with a **10× weaker head penalty**; penalty ramp continued as in #8 (`--curriculum-total 4e6`) | **Failed again:** collapsed at 1.25M (0.02 m/s) and stayed still at 1.75M / 2.25M. The standing brain tilts its body ~24° with joints at their limits, although walking would still earn ~35 % more reward – a training trap, not the head penalty. |
| 8c | 27 Sep 2026 | `myr_legs_v3c` | Myrranox | 0.75M → 2.07M (stopped) | A2: continued #8 from 0.75M with the **penalty scale frozen at 0.6** (the level at which it still walked), original head weight | **Failed, slower:** 1.5M walks at 0.25 of 0.6 m/s and veers off (−68 °/s at full speed), 2.0M 0.08 m/s. |
| 9 | 26 Sep 2026 | `dc_legs_v3` | DC | 4M | **Stage 2:** spine joint (bend + sideways) splits the chest – the rear body carries hind legs and tail; neck (2) and head (1) get motors; tail and extra arms stay passive (stage 1) → 21 actuators; same rewards as #8 | **Success – released.** Walk 0.83 / 0.8 m/s, trot 2.69 / 2.5 m/s (v2: 0.68 / 2.15 – the spine helps), turns ±21 of ±36 °/s, no falls. Head wobble 0.69 rad/s (v2 + passive parts: 1.17, −41 %), spine ±3.2° sideways / ±1.3° up-down, tail tip ±15.5° (v2: ±10.6°), asymmetry 1.9–3.4°. Power higher (947 W vs 656 W). |
| 10 | 28 Sep 2026 | `myr_legs_v4_pilot`, `dc_legs_v4_pilot` | both | 4M → 4.5M (pilot) | **Precise + fluid (v4)**, continued from Myrranox v2 / DC v3: joint targets low-pass filtered (0.5 per 20 ms), tracking tolerance narrowed 1 → 0.5, turn weight 1.0 → 1.5, soft-landing penalty, half the commands ramp, 20 % spot turns / 10 % backwards, head style terms scaled by tracking quality | **Passed** (no stand-still collapse, no falls). Joint acceleration −32 % (Myr) / −25 % (DC); DC turns 30–32 of 36 °/s (v3: 21); Myrranox walks straight (drift 10 → 4 °/s) but still no spot turns (9 of 72 °/s) or backwards → speed reward on spot-turn commands now scaled by turn tracking (standing still earned it in full). `docs/evaluations/v4_pilot/`. |
| 11 | 28 Sep 2026 | `myr_legs_v4` | Myrranox | 4M → **7M** (continued from `myr_legs_v2`, 1.1 h) | As #10 plus the spot-turn fix (speed reward on spot-turn commands scaled by turn tracking) | **Released.** Turns on the spot for the first time (58 of 72 °/s; v2: 7), walks straight (drift +10 → −1 °/s), full speed on target (1.20 / 1.2; v2 overshot 1.35), turn error 18.6 → 7.9 °/s, joint acceleration −24 %, head wobble −25 %, power −22 %. Still no walking backwards. Midpoint 5.25M already turned on the spot (37 °/s) – the stand-still trap did not come back. |
| 12 | 28 Sep 2026 | `dc_legs_v4` | DC | 4M → **7M** (continued from `dc_legs_v3`, 1.1 h) | Same as #11 | **Released.** Turns +29 / −32 of ±36 °/s (v3: ±21), speed error 0.13 → 0.08 m/s, turn error 10.7 → 4.6 °/s, joint acceleration −27 %, head tilt 10.7 → 2.4°, power −15 %; trot slightly under the command (2.41 / 2.5; v3 overshot 2.69). Best turning at 5.25M (37–38 °/s) but worse speed there – the final brain is the best overall. |
| 13 | 29 Sep 2026 | `myr_legs_v5_pilot`, `dc_legs_v5_pilot` | both | 7M → 7.5M (pilot) | **Natural style (v5)** from v4: gait-clock leg rhythm reward, knee effort relief, DC panther stride + paw register + damped tail/arms | DC responded (1.08 steps/s, hip swing 25 → 33°); Myrranox knee still 4.5° → diagnosis: knee command stuck at −1.4 beyond the action limit since v2 2M (also DC abduction + front shoulders) → `unsaturate_brain.py`. |
| 14 | 29–30 Sep 2026 | `myr_legs_v5` | Myrranox | 7M → **10M** (from `myr_legs_v5_start` = v4 with knees freed; paused 12:50 at 9.75M, resumed next day) | v5 | **Recommended.** Knee 4.5 → 20°, foot lift 9 → 13 cm, torso wobble −29 %, spot turn 58 → 65 °/s, precision unchanged, no falls; power 57 → 79 W. |
| 15 | 29 Sep 2026 | `dc_legs_v5` | DC | 7M → 8.5M (stopped) | v5 from `dc_legs_v5_start` (abduction + front shoulders freed) | Panther stride (1.22 steps/s, 0.64 m) but the owner saw an unstable body: torso wobble 0.40 → 0.59 rad/s, head tilt 2.4 → 5.9°. Stopped. |
| 16 | 29–30 Sep 2026 | `dc_legs_v5b` | DC | 8.5M → **10.5M** (paused at 9.25M, resumed) | #15 + steady trunk (roll/pitch −0.5, bounce −1.5, scaled by tracking) | **Mixed.** Torso wobble back to 0.44, strides 0.55 m at 1.12 steps/s, hind paw 26 cm from the front print (v4: 33); but head wobble 0.80 / tilt 7.7° and walk speed 0.62 of 0.8 m/s. No checkpoint as steady as v4 → owner decision. |
| 17 | 30 Sep 2026 | `dc_legs_v5c` | DC | 10.5M → 12.5M | v5b + head ×3, speed tolerance 0.2 m/s | **DC walker** (showcase): 0.78 m/s, stride 0.62 m at 1.25 steps/s; head wobble unchanged (0.84 rad/s) → head stabilised in the Blender animation only. |
| 18 | 30 Sep 2026 | `dc_legs_v6`, `dc_legs_v6s` | DC | 1M / 0.66M (stopped) | v6: arms as actuated limbs, legs ±8°, stricter rhythm; brain transplant from v5c (`grow_brain.py`) / from scratch | Transplant stood still (9–10 joints stuck); from scratch stopped by the owner while shuffling. |
| 19 | 30 Sep 2026 | `dc_legs_v8` | DC | 8M planned, stopped ~4.3M | v8: mesh hull body, sprint 4 m/s, ramps, no unnecessary movement | Handstand on front legs + arms (→ 25° tilt = fall); best 0.37 m/s at 2.75M, collapsed into standing still from 3.2M. |
| 20 | 13 Oct 2026 | `*_legs_v9_s1`…`s1f` | both | stages of 0.5–2M | v9: multiplicative reward, mesh bodies, standing pose; series of fixes (noise, reward statistics, command ranges, false falls) | DC: correct lateral footfall order, 0.65 m/s, trunk 3.7° (best at 7.5M) but hind legs tripping; Myrranox: stepping in place, then a 'wheelbarrow'. |
| 21 | 13 Oct 2026 | `myr_legs_v10` | Myrranox | **0 → 3M** (abdomen density changed at 1.75M) | v10: tripod step generator + corrections, lighter abdomen | **Released.** 0.52 / 1.08 m/s, spot turns 56 °/s, walks backwards (−0.26) for the first time, trunk level, knee 60°, abdomen clear, tripod 87 %, no falls; front legs still too little on the ground (21 %). |
| 22 | 13 Oct 2026 | `dc_legs_v11` | DC | 7.5M → 9.5M | v11: big-cat mass distribution, rounded paw pads | No more tripping (1.0 steps/s every leg), trunk ≤ 2°, no falls, but slow: 0.23 m/s at 0.6, 0.80 at 1.2. |
| 23 | 13 Oct 2026 | `dc_legs_v12` | DC | 9.5M → 11.5M | v12: 218-piece body, light spikes, wolf rules | +2M: 0.42 / 0.71 m/s, very even rhythm (5 %), but the neck command pinned at its limit (head target unreachable). |
| 24 | 14 Oct 2026 | `dc_legs_v12b` | DC | 11.5M → 12.5M | wolf head as the neck's rest pose | Pulled all legs in to the stop, shuffled at 0.11 m/s – two rule bugs (joint-limit margin, narrow paws free). |
| 25 | 14 Oct 2026 | `dc_legs_v12c` | DC | 11.5M → 12.3M | rule fixes, from the 11.5M brain | Best at +500k: 0.45 / 0.73 m/s, rhythm 1–5 %; the session ended at 12.26M (night lost). |
| 26 | 14 Oct 2026 | `dc_legs_v12d/e/f` | DC | from 12.0M | T-rex arms, weight to the back, arm / front-leg collision | +500k: 0.37 / 0.61 m/s, rhythm 1 %, no falls. |
| 27 | 14 Oct 2026 | `dc_legs_v12s` | DC | 12.5M → 14.0M | sprint stage: arms from 2 m/s, flat paws, braking drills, limit ramp | Walk improved (0.63 / 0.81 m/s); hands stayed tucked → limited arm range. |
| 28 | 14 Oct 2026 | `dc_legs_v12h` | DC | 14.0M → 15.3M | remodelled hand (wrist at the real wrist) | Hands on the fingertips (−61°); the wolf rules had zeroed the sprint's quality. |
| 29 | 14 Oct 2026 | `dc_legs_v12p`, `v12q` | DC | 15.3M → 16.0M | palm rule, soft wrist + palm reflex, crouch cost | DC reared up in the sprint; v12q unlearned walking (0 m/s at walking commands). |
| 30 | 14 Oct 2026 | `dc_legs_v12r` | DC | 14.0M → 14.3M | half the commands walks, sprint costs ramped | Stopped for the arm brain. |
| 31 | 14 Oct 2026 | `dc_arms_v1` | DC arms | 70k (stopped) | separate arm brain, walking brain frozen | Laptop asleep 1.4 h; stopped when the owner dropped the arms from the sprint. |
| 32 | 14 Oct 2026 | `dc_legs_v12u`, `v12v` | DC | 14.0M → 14.5M | T-rex arms at every speed, four-legged sprint; posture cost (capped) | Lay in a "plank" (16–18 cm low, legs stretched) when standing / walking slowly. |
| 33 | 14 Oct 2026 | `dc_legs_v12w` | DC | 14.0M → 15.5M | plank = fall, posture cost uncapped | Plank gone, sprint 2.2 m/s, but no walking off at 0.3 / 0.6 m/s. |
| 34 | 14 Oct 2026 | `dc_legs_v12x` | DC | 15.5M → 18.0M | start-up help (all speeds, fading) | Walks again at every command, sprint 2.8 m/s – but clunky: 37 % timing variation, 0.28 m strides, 6 cm bob. |
| 35 | 14 Oct 2026 | `dc_legs_v12y` | DC | 18.0M → 18.8M | smoothing: 70 % walks, bounce / surge up, rhythm double, no help | Smoother at 1.0 m/s (bob 1.6 cm), but stood still at 0.6 again. |
| 36 | 14 Oct 2026 | `dc_legs_v12z` | DC | 18.8M → 20.8M | gentle permanent start-up help at slow commands only | **Best DC so far:** 0.59 / 0.86 / 1.84 / 3.56 m/s, timing variation 1–9 %, strides 0.52–0.97 m, bob ≤ 2 cm, brakes from 4 m/s without falling. |

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

## "Alive" bodies (stages 1–2, 26 Sep 2026)

Owner request: the whole body should move – Myrranox like an ant, DC like a panther with extra arms.
- **Stage 1** (no training): spring-damper tail, abdomen, head, neck and arms (`sim/bodies.py` `PassiveChain`); the
  v2 brains walk unchanged with them. Preview: `docs/videos/alive_stage1_preview.mp4`.
- **Stage 2** (walking v3): spine / neck / head / abdomen actively controlled, head-steadiness rewards.
  Checks before training: exact mirror simulation for v3 (incl. spine, head and the passive parts), `check_env`,
  both bodies stand. Baseline "aliveness" with v2 + passive parts (`alive_metrics.py`): the loose head wobbles about
  twice as much as the torso (DC 1.17 vs 0.59 rad/s, Myrranox 1.74 vs 0.62 rad/s) – v3 should calm it.

**Stage 2 result:** the DC walks faster, uses its spine, holds its head steadier and swings its tail more
(`docs/videos/dc_legs_v2_vs_v3.mp4`, `docs/videos/dc_legs_v3_training.mp4`, `docs/evaluations/alive_dc.json`).
The Myrranox v3 failed three times (#8 stands still; #8b weaker head penalty; #8c penalty frozen) – every run walks
until about 1M steps and then drifts into standing still. **Owner decision (27 Sep, option C):** the Myrranox keeps
walking v2 with the stage-1 passive parts; the showcase uses DC v3 + Myrranox v2 (`docs/videos/alive_stage2_preview.mp4`).
Untested idea: smaller movement ranges for the new Myrranox head/abdomen joints (A3).

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
8. (v3) A "style" penalty that every step pays (head wobble) can make standing still the best option again as the
   penalty curriculum ramps up – the light Myrranox head bounces with each step and the brain stopped walking; the
   heavy, well-suspended DC head did not. Style terms need to be weak or scaled per species.
9. (v4) Check what standing still earns under every command type: on spot-turn commands the speed reward
   (speed 0 = command 0) paid standing in full – that, not the body, kept the Myrranox from turning on the spot
   through v1–v3. Scaling it by the turn tracking fixed it within 1.25M steps.
10. (v4) Smoothness is easier to build in than to reward: a low-pass filter on the joint targets made both bodies
    ~25 % smoother while tracking got better, not worse. Fine-tuning a released brain (3M steps) was enough.
11. (v5) Watch the raw policy outputs, not only the joint angles: a joint whose mean command drifts beyond the action
    limit is clipped, gets no learning signal and freezes for good (Myrranox knees from v2 2M on). Shifting the
    output bias back inside the range revived the knee within one run.
17. (v12) A "near the joint limit" margin must scale with the range: 5° on a ±8° joint meant "always at the limit".
18. (v12) Penalise both sides of a target: only "paws too wide" cost, so the DC pulled its legs in to the stop.
19. (v12) Walking rules in the sprint and full penalties on an unlearned sprint zeroed the reward; the brain forgot
    to walk. Scope rules to their gait, keep practising walks, ramp the new skill's penalties.
20. (v12) The arms could not reach the ground and the hand's joint sat in the wrong place – check geometry first.
12. (v5) Style changes shift the body dynamics: longer, slower strides made the DC trunk rock until a steady-trunk
    term was added - and even then it did not reach v4's calm. Measure steadiness alongside every style change.
13. (v9) Check what standing still earns under every command type AND with exploration noise: noise penalised as jerk
    wiped out the points (quality 0.06) and the brains learned nothing.
14. (v9) A resumed brain carries its reward statistics: after a reward change they must be reset, or the new rewards
    look like noise (explained variance −4).
15. (v10) Fix the body before the brain: mass by volume made the Myrranox rear-heavy and the DC head-heavy; every rule
    change before that only produced new evasions (stepping in place, wheelbarrow, handstand).
16. (v10) A hexapod tripod is statically stable, so a step generator with learned corrections works for the Myrranox; for
    the quadruped DC the open-loop rhythm falls over.

## Walking v4 – precise and fluid (28 Sep 2026)

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

Fluid by design: the joint targets pass a low-pass filter (0.5 per 20 ms), so the brain cannot make the joints jump;
the action-rate penalty now acts on the brain's raw output. Precise by curriculum: the speed / turn tolerance narrows
from 1 to 0.5 between 20 % and 80 % of the run. Measurements: `docs/evaluations/v4/` (`precision_metrics.py`,
`alive_metrics.py`), baseline `docs/evaluations/v4_baseline/`. Videos: `docs/videos/*_legs_v4_training.mp4`,
`docs/videos/myr_legs_v2_vs_v4.mp4`, `docs/videos/dc_legs_v3_vs_v4.mp4`.

## Walking v5 – natural style (29–30 Sep 2026)

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

## Next ideas (one change per run)

- `*_legs_v3`: more turn-on-the-spot / backwards commands for the Myrranox (both still fail), continue from v2
  (`train_locomotion.py --resume models/<run> --version 2`).
