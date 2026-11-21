# Brief: QwenOFT with MAS_v5 blocks (noAG), single change vs e3v5_MAS_noAG

## Goal
Train QwenOFT exactly like `e3v5_MAS_noAG`. The only change: the frames at which the MAS block switches around pick grasps follow MAS v5 instead of the v2.3 boundaries. The block text itself (v2.3 wording, noAG format: Current Goal / In-Progress Step / Next Step) stays the same.

Then run the held-out eval and compare it with e3v5_MAS_noAG and e3v5_A0.

## Background (why)
- v2.3 MAS boundaries (Experiment 2): "grasping" starts ~3 frames after the close onset C and ends ~3 frames after the lift onset L.
- MAS v5:
  - grasp starts at ARRIVAL (gripper within 20 mm of its grasp distance; median C-5; fallback C),
  - grasp ends at HELD (object held for 3 steps; median C+8, ~3-4 frames before L; fallback L).
- On the In-Progress VLM, v5 made the grasp->carry switch much easier to recognise (+18 pp near the switch).
- Known flaw, accepted for this run: arrival is too early for handle/side grasps (frying pan, moka pots, ketchup, dressing, wine bottle).

## Inputs
- Current texts: `~/VLA_FT/DataSamples/Libero/A0ASMAS_noAG/{train,val,test}.csv`. Columns: split, suite, task_id, task, demo, t, raw_step, text_A0, text_AS, text_MAS, idx_AS_v2, idx_MAS_v2, mas_unsplit.
- v5 boundaries: `/hnvme/workspace/y307bd13-vgv/runs/step3_model1/labels_v5/boundaries_mas_v5.csv`.
  - Columns: split, task (= task_id), demo, step, type, wording, v4_start, v4_end, v5_start, v5_end, v5_rule, ...
  - Raw steps. Relabelled grasps have `v5_rule == "arrival_held"`; there, `type == "grasp"` rows give the v5 grasp span `[v5_start, v5_end]`.
  - C = v4_start of that grasp row; L = v4_start of the next (carry) row.

## Rules
- Repo rules as always: python only via `scripts/common.sh` run_py (container), never bare python, never `pip install --user`.
- One GPU. Use nothing the original StarVLA implementation does not use. Same recipe, seed and flags as `e3v5_MAS_noAG` (read them from its `args.json`).
- Never overwrite existing files. New text dir: `~/VLA_FT/DataSamples/Libero/A0ASMAS_noAG_v5/`. New run dir: `runs/oftvla/e3v5_MAS_noAG_v5`.
- Git commit messages: no Co-Authored-By line, no mention of Claude or AI.
- Stop and report at each CHECKPOINT.

## Step 1: build A0ASMAS_noAG_v5 (text reassignment, no new strings)
Per demo, the v2.3 MAS steps are contiguous runs of `idx_MAS_v2` (each with one text_MAS). For each relabelled pick grasp:
1. Find the v2.3 step that is the grasp of this object: the run whose In-Progress Step is the grasp text and which overlaps [C, L] most. Its neighbours are the v2.3 reach run (before) and the v2.3 carry run (after). If this match is ambiguous or the neighbours are not reach/carry, skip the grasp and log it.
2. The new grasp span is `[v5_start, v5_end]`, clipped to the union of the three runs. Reassign text_MAS (and idx_MAS_v2) only inside the union of the three runs:
   - frames before v5_start get the reach run's text;
   - frames inside the span get the grasp run's text;
   - frames after v5_end get the carry run's text.
3. All other frames, and all other columns (text_A0, text_AS, idx_AS_v2, ...), are copied unchanged.

Asserted checks; abort if any fails:
- same rows, same order, same row count per split as A0ASMAS_noAG;
- text_A0 and text_AS identical for 100 % of rows;
- every new text_MAS string already occurs in the same demo (no new strings);
- text_MAS changes only inside the three-run unions of relabelled grasps;
- report the share of rows changed per split (expect a few %), the number of grasps relabelled vs skipped (with reasons), and the median shift of each switch (new minus old) for reach->grasp and grasp->carry (expect about -8 and -7 frames);
- print the full prompt text before/after for 2 changed frames of L90T27 demo_13 (ketchup) and L10T06 demo_31.

**CHECKPOINT 1:** checks, shares, shifts, the examples, and the script path (keep it in the new text dir).

## Step 2: smoke test and submit
- Same command as e3v5_MAS_noAG (`train_e3_text.sbatch` with setup MAS, TEXTDIR = A0ASMAS_noAG_v5, same seed), via `submit_chain.sh` with the same number of parts.
- Smoke test first with the exact config (few steps): loss runs, checkpoint saves, the prompt for one sample shows the noAG block.
- Then submit, and chain `heldout_eval.sbatch` with afterok. It must use the test.csv from A0ASMAS_noAG_v5; heldout_eval derives it from args.json, so verify the printed path and md5.

**CHECKPOINT 2:** diff of args vs e3v5_MAS_noAG (only text_csv/run name may differ), smoke-test lines, job ids.

## After that (not part of this brief)
Copy `e3v5_MAS_noAG_v5/heldout_eval/step30000_{frames.csv,pred.npz}` to the laptop for the comparison with e3v5_MAS_noAG and e3v5_A0. The comparison covers L1, gripper agreement exact and ±2, and windows around C / held / L.
