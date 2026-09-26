# Predator training log (navigation brains, step D)

The predator ("navigation") brain decides where a creature goes – forward speed and turn rate every 0.1 s – in the 2D
habitat (`sim/creature_env.py`): hunt prey, rest, stay in or raid territories, deal with the rival.
Training: `train.py` (PPO, 8 parallel habitats, 3-frame stack, VecNormalize). Evaluation: `evaluate.py`
(section 12 checks, 5 fixed 10-minute days, scripted baseline). Live view: `watch_predator.py`, TensorBoard `habitat/*`.
Every run is also a row in [experiments.csv](experiments.csv); evaluations are in [evaluations/](evaluations/).

**Training videos** – `docs/videos/predator/<run>_training.mp4` (`make_predator_video.py`): every checkpoint
(untrained, then every 500k steps) lives one 10-minute day in the habitat, top-down at 30× speed, next to the learning
curve (prey / 10 min) and live hunger, fatigue, prey, raids. Each run is replayed under its own training conditions
(old speed mapping and fatigue rule for attempts 1–2, prey fleeing at 0.7 m/s).

| Run | Video | What it shows (one day per checkpoint) |
| --- | --- | --- |
| attempt 1 | `myr_brain_v1_r1_training.mp4`, `dc_brain_v1_r1_training.mp4` | starves every day, 0 prey at every checkpoint |
| attempt 2 | `myr_brain_v2_r1_training.mp4`, `dc_brain_v2_r1_training.mp4` | the same – stands still and starves |
| round 1 | `myr_brain_v3_r1_training.mp4`, `dc_brain_v3_r1_training.mp4` | Myrranox: 0 → 2 → 4 prey, survives the day at 2.0–2.5M; DC: survives every day from 0.5M (4–6 prey) |
| option B | `myr_brain_v4_r1_training.mp4`, `dc_brain_v4_r1_training.mp4` | Myrranox: 1 → 3 prey, survives at 1.5–2.5M; DC (released): survives most days, 4–5 prey |

## Setup that stayed the same

- Habitat, needs and territory rules: PROJECT_PLAN.md sections 6, 8 and 15 (own-territory prey do not come back,
  raided prey re-appear at the eater's home, neutral prey respawn).
- Movement limits = what the v2 walking brains can deliver (`sim/locomotion_limits.json`): Myrranox no backwards and
  no turning on the spot, DC max 40 °/s.
- Hunger and fatigue from a power curve measured on the v2 walking brains (`sim/energy.json`).
- Rival: scripted predator (`sim/scripted.py`) – hunts, rests, raids when its home stock is low.

## Sessions

| # | Date | Runs | What changed | Result |
| --- | --- | --- | --- | --- |
| 1 | 25 Sep | `myr_brain_v1_r1`, `dc_brain_v1_r1` (3M each) | Plan reward (section 8), scripted rival | **Failed:** both stayed home and starved (0 prey / 10 min, ~45 % survival). A test showed the reward itself was the problem: a good scripted hunter scored −10 to −12, barely better than doing nothing. |
| 2 | 25 Sep | `*_brain_v2_r1` (stopped at 2.8M) | **Reward v2:** every catch pays (× hunger, no 50 % threshold); +0.01 × (1 − hunger) per step ("well fed"); potential-based guide towards visible prey when hungry. Verified first: hunter +25…+44 vs ≤ −3 for everything else (`tests/test_predator_reward.py`) | **Failed:** brains stood still. Causes: negative speed actions all meant "stop" (half the action range); the guide with discount γ < 1 paid ~+4 per episode for standing; fleeing prey were never caught by chance; sliding along a rock counted many hits. |
| 3 | 25 Sep | pilot (400k) | **Exploration fixes:** action −1…1 = 0…max speed; undiscounted guide; **prey-flee curriculum** (prey do not flee at first, full difficulty after 40 % of training); a new hit only after 1 s without contact | **Learns to hunt:** Myrranox 0 → 2.5 prey / 10 min (survival 31 → 73 %), DC 0 → 1.8 (48 → 66 %). |
| 4 | 25–26 Sep | `myr_brain_v3_r1`, `dc_brain_v3_r1` (3M each) | Fixes of #3, full round vs scripted rival. Interrupted at 1.75M when the Claude session ended (training processes stop with it) – resumed from the 1.5M checkpoints (`train.py --resume checkpoints/<run>_1500000_steps`) | **Hunts:** 4.0–4.4 prey / 10 min (Myrranox better than the scripted predator, 3.5). But **never raids, never rests, never defends**; the Myrranox starves on 3–4 of 5 days once its home stock is empty. |
| 5 | 26 Sep | `myr_brain_v4_r1`, `dc_brain_v4_r1` (3M each) | **Option B:** observation + prey left at home + rival inside home (29 values); 30 % "lean days" (1–2 prey at home); scripted rival invades on 50 % of days; fatigue slows the creature from 70 % (not only at 100 %) | **DC: best brain – released.** Survives every day, 4.6 prey / 10 min, 3 % starving, uses rest spots (2.6 / 10 min), 4/5 checks. **Myrranox: worse** (0/5): spends 21 % in the DC territory without catching there, starves on 4 of 5 days – it tires about twice as fast as the DC. |

## Evaluation (section 12, 5 days × 10 min, rival invades on ~40 % of days)

| Check (target) | Myrranox v3 r1 | Myrranox v4 r1 | Myrranox scripted | DC v3 r1 | **DC v4 r1** | DC scripted |
| --- | --- | --- | --- | --- | --- | --- |
| Time in own zone (≥ 90 %) | 99 % | 69 % | 88 % | 100 % | **93 %** | 93 % |
| Hunger at 100 % (≤ 5 %) | 8 % | 16 % | 11 % | 7 % | **3 %** | 0 % |
| Prey / 10 min (≥ 3) | 4.4 | 1.6 | 3.5 | 2.8 | **4.6** | 4.8 |
| Rest-spot visits when tired (≥ 1) | 0 | 0 | 3.7 | 0 | **2.6** | 1.8 |
| Obstacle hits / 10 min (≤ 1) | 1.0 | 1.4 | 0.4 | 0.2 | 1.4 | 1.4 |
| Days starved | 3/5 | 4/5 | 4/5 | 1/5 | **0/5** | 0/5 |
| Raids / intruders repelled | 0 / 0 | 0 / 0 | 0.2 / 0.25 | 0 / 0 | 0 / 0 | 1.2 / 0.6 |

**Released (owner decisions 26 Sep):** DC `models/dc_brain_v4_r1`; Myrranox `models/myr_brain_v3_r1` (kept as is,
no fatigue change).

## Lessons

1. Check that the reward ranks the wanted behaviour first *before* training (test with scripted policies) – the plan's
   reward did not, and 3M steps were spent on it.
2. A good reward is not enough if the brain never experiences success: map the whole action range to useful actions,
   make sure shaping does not pay for idling, and start easy (curriculum).
3. Rare situations (empty home, intruder) must be practised on purpose (scenario variety) and be observable
   (home-stock input) – otherwise raids and defence never appear. With option B they still did not appear.
4. What works in the fast 2D habitat does not automatically work with real legs: the Myrranox brain failed in the
   physics world (it cannot turn on the spot), the DC brain transferred fine.
5. Training runs in the Claude session – closing the app stops them. Checkpoints every 500k steps made the restart cheap.

## Open

- Raids and territory defence (no brain learned them).
- Myrranox: fatigue balance, turning on the spot (walking v3), or fine-tuning with the physics body (plan D2).
- Alternating rounds against the other species' trained brain (plan stage 4) not run.
- Brains were trained with prey fleeing at 0.7 m/s; the showcase uses 0.35 m/s (owner decision 26 Sep).
