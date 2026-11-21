# Brief: closed-loop success of QwenOFT A0 / AS / MAS with simulator-driven block switching (oracle switcher)

## Goal
Measure task success and step-level progress in LIBERO rollouts for the three Experiment 2 models:
- `e3v5_A0`, `e3v5_AS`, `e3v5_MAS` at step 30000,
- on all 24 E3 held-out tasks × the same 50 LIBERO init states.

For AS and MAS, the block text is switched ONLINE from the simulator state: an oracle step detector that reproduces the offline label boundaries. The question: does the block help the robot do the task when the step is known perfectly?

No training. Same model code, same inference as the existing closed-loop eval of `e3_A0` (job 1175328, `oftvla/libero_client.py`, 323/1200).

## Rules
- Python only via `scripts/common.sh` run_py (container). Never `pip install --user`. One GPU per job.
- Use nothing the original StarVLA implementation does not use. Inference identical to the existing eval: same image preprocessing, same action handling, same deterministic L1 head, env.seed(7), fixed init states, the same step limits as before (400 for L90 tasks, 520 for L10).
- Never overwrite existing results. New output dirs: `runs/oftvla/e3v5_{A0,AS,MAS}/closedloop/`.
- Inode limit on /hnvme: one file per task (no per-step files), plus a summary CSV.
- Git commit messages: no Co-Authored-By line, no mention of Claude or AI. Record decisions in MILESTONES.md (new milestone).
- Stop and report at each CHECKPOINT.

## Step 1: inspect (read-only)
- How `libero_client.py` builds the prompt from text, runs an episode, and writes `e3test_step30000.csv`.
- Where the AS/MAS block texts come from (`A0ASMAS/test.csv` text_AS / text_MAS, columns idx_AS_v2 / idx_MAS_v2) and which code produced their boundaries:
  - step1 v5 segmentation, `~/VLA_FT/language_experiments/analysis/step1_subtasks.py`, with `rolebind_v0` `sim_events.py` / `seg_v4.py`;
  - events per step type (reach/move, pick/grasp, carry, put, operate) and their offsets (e.g. the MAS pick step starts ~3 frames after the close onset).
- Per test task, the canonical block sequence = the most common ordered sequence of distinct text_AS (text_MAS) blocks across that task's test demos. Report how many demos follow it, per task.

**CHECKPOINT 1:** prompt construction, the block sequences per task (step count AS/MAS, share of demos following them), and the event rule that ends each step type, with file:line references.

## Step 2: online switcher + validation on replayed demos
- Implement a causal switcher. At every control step it reads the simulator state (finger width, contacts, object poses, part joint values) and advances from block k to k+1 when the event that ends step k occurs.
  - Use the SAME event definitions as the offline segmentation, applied causally (only past and current states).
  - Rules that need "n steps sustained" confirm after n steps. Report the resulting delay.
  - Events are evaluated for the step's TARGET object. A grasp of a wrong object does not advance the block; log it.
- Validate before any model runs: replay the test demos' recorded actions in the simulator with the switcher attached. Compare its switch frames with the offline boundaries (idx_AS_v2 / idx_MAS_v2 changes).
  - Report per step type: median offset and share within ±1 / ±3 frames.
  - Report the share of demos with the same number of switches.
  - Target: ≥ 90 % of switches within ±3 frames. If a step type misses this, report it and stop.

**CHECKPOINT 2:** validation table, the switcher's file path, and its unit tests.

## Step 3: smoke test
- 2 tasks × 3 init states per model (one T1 task with known A0 success, e.g. L90T06 open top drawer; one multi-step task, e.g. L10T06 soup + sauce). Log the switch frames and print the prompt at each switch.
- Harness check: `e3_A0` (the old run) on 2 tasks × 5 inits through the new harness with the switcher disabled must reproduce the episodes in the existing `e3test_step30000.csv` (success and steps). Report any mismatch; GPU bf16 nondeterminism is possible.

**CHECKPOINT 3:** smoke output, harness check, estimated wall time per model.

## Step 4: full runs
3 jobs (A0, AS, MAS), each 24 tasks × 50 inits, same init order.

Per episode, log:
- success, steps, the block index over time (switch frames);
- max step index reached; first object grasped (object name, frame);
- for each step, whether its ending event occurred.

## Pre-registered analysis (fixed now, before results)
- Unit = task (24), paired across models on the same init states.
- Primary: task success rate, AS vs A0 and MAS vs A0. Wilcoxon signed-rank over tasks + task bootstrap CI; McNemar per task for case studies.
- Secondary (progress, defined from the switcher log so it is available for A0 too, with the switcher running silently): fraction of steps completed, and first-grasp-correct.
- Report by group (T1, T2, S2, two-goal, LIBERO-10), and with/without the leaked tasks L90T41, L90T39.
- One seed per model; interpret per-task differences < ~6/50 episodes with care.

**CHECKPOINT 4:** results table per task and per group, plus the MILESTONES entry.
