# Milestones

Log of what was done, why it matters, and the evidence that it worked.
A milestone counts as **done** only when its pass criterion was met and the evidence is recorded here.

---

## ✅ M0 — Decision: own standalone repo (2026-10-27)
**What:** Re-implement StarVLA's QwenOFT in a small repo (`oftvla`) instead of developing inside the StarVLA fork.
**Why:** Full control and readability. The model path is about 800 lines in StarVLA, but it is tangled with about 10k lines of dataloader and config machinery.
**Trade-off accepted:** We must *prove* our model equals StarVLA's, otherwise the comparison with GR00T/PI and the paper numbers is lost. The fork (`~/VLA_FT/starVLA` @ `985c223`) stays as the **reference**.

## ✅ M1 — Model + unit tests (2026-10-27)
**What:** `oftvla/model.py` (~230 lines) and `tests/test_model_units.py`.
**Why:** The pure functions (gather, loss, head, prompt) are testable on CPU in seconds, without loading the 8 GB VLM.
**Pass criterion:** all unit tests pass inside `starvla.sif`.
**Evidence:** `6 passed in 5.99s` (helma4).

## ✅ M2 — Equivalence with StarVLA (2026-10-27)
**What:** `scripts/check_equivalence.py` loads both models with identical weights and compares them at 4 points of the pipeline.
**Why:** It guarantees that our results can be compared with StarVLA's and that we introduced no silent bug.
**Pass criterion:** identical tokens and pixels; predictions and loss within 1e-4.
**Evidence (H200, node h24-03):**
```
[PASS] input_ids identical, shape (2, 175)
[PASS] pixel_values: max|diff| = 0.000e+00
       query tokens per sample: [8, 8], query id ours=146663 ref=146663
[PASS] predict_action: max|diff| = 0.000e+00
[PASS] action_loss:    max|diff| = 0.000e+00
ALL PASS
```
**Learned:** 🔍 is a single token (id 146663) in plain `Qwen3-VL-4B-Instruct`, so the `-Action` checkpoint is not needed.
**Does NOT prove:** that StarVLA itself is correct; that training (gradients, optimizer, data) matches; that the test can detect a difference (see M2b).

## ⬜ M2b — Negative control for the equivalence test
**What:** Deliberately change one thing (e.g. head weights, or "next 8" → "next 7" in the prompt) and re-run the check.
**Why:** A test that cannot fail proves nothing.
**Pass criterion:** the expected checks FAIL and the others still PASS.

---

## ✅ M3 — Data (`oftvla/data.py`)
**What:** HDF5 loader for our regenerated LIBERO data, producing StarVLA-format samples.
**Rule (decided 2026-10-27): use nothing the original StarVLA LIBERO pipeline does not use.** Therefore:
every step is a chunk start with end padding by repeating the last action; min_max normalisation of the 6 arm dims, gripper not normalised; PIL default resize 256→224; no augmentation; no proprio; actions as float16.
Kept from our data definition (shared with `e3_A0`): demo selection by `e3_train_ids.csv`, status-ok demos only, A0 sentence, gripper 1 = open / 0 = close (the same convention StarVLA's LIBERO eval expects).
**Guard:** the loader refuses to start if a training demo belongs to a test task (`--forbidden_ids e3_testdemo_ids.csv`). This changes no data; it only turns a silent split error into a crash.
**Pass criterion:** `tests/test_data.py` passes (6 tests).
**Evidence:** `12 passed in 14.12s` on helma (6 model + 6 data tests), commit a82deb5.

## ✅ M4 — Training (`oftvla/train.py`) smoke test
**What:** `scripts/smoke_mini.sbatch`: 300 steps on the mini set (5 tasks), interrupted once and resumed.
**Pass criteria:** (a) the loss clearly falls (gradients reach the head and the VLM); (b) the resume continues at the saved step; (c) GPU memory fits; (d) the measured seconds per step tell us how long the full run takes.
**Evidence (2026-10-27, H200):**
- Job 1156180 (micro-batch 16): loss 0.85 → 0.13, `mse_score` 0.019 → 0.006 in 300 steps; 0.50 s per 16 samples; peak 99 GB; data loading 1–3 % of the time. The resume was *not* exercised: all 300 steps finished before the planned stop (planning error), and the `DONE` guard correctly skipped part 2.
- Job 1156234 (micro-batch 32, stop after 36 s): `[resume] saved at step 36` → `[resume] from step 36`; loss continued 0.37 → 0.34 with no jump (optimizer state restored); final loss 0.085, `mse_score` 0.0038. 0.86 s per 32 samples (13 % faster per sample); peak 119 GB (nvidia-smi 128 GB of 141).
**Decision (revised):** 32 × 4 ran out of memory in the full run (jobs 1156329–31, first step, 132 GB allocated). Cause: the smoke test used no accumulation, so gradients were freed before each forward pass; with accumulation, the 17.6 GB of fp32 gradients stay in memory during micro-batches 2–4. 99 GB (smoke, 16) + 17.6 GB ≈ 117 GB fits, 119 + 17.6 does not. Full run therefore uses 16 × 8 (same 128 per step) → ~4 s/step → 30K steps ≈ 33 h = 2 chained jobs.
**Lesson:** a smoke test must use the exact configuration of the real run (here: accumulation), otherwise it does not test what it claims to.

## ✅ M4b — Full run e3_A0 (`scripts/train_e3_A0.sbatch`)
76 tasks, A0, StarVLA LIBERO recipe: 30K steps × 128 samples, 5K warmup, cosine down to 4 % of peak, AdamW (0.9, 0.95), lr VLM 1e-5 / head 1e-4, full fine-tuning, grad clip 1.0, seed 42.
**Evidence (finished 2026-10-29 05:04):** jobs 1156388 (23:02 h, stopped at step 19,870 and saved resume state) → 1156389 (resumed at 19,870, 11:09 h, finished 30,000, wrote `DONE`) → 1156390 (exited after 16 s: `DONE` exists). Commit f1685e9. 3,636 demos, 581,668 samples; no errors, NaN or OOM; memory constant 105.7 GB; ~3.9 s/step.
Weights: `runs/oftvla/e3_A0/steps_{10000,20000,30000}_model.pt` (9.8 GB each, bf16).
| Steps | Train loss (L1) | mse_score | Grad norm | LR (VLM) |
|---|---|---|---|---|
| 0–2K | 0.164 | 0.0076 | 15.6 | warmup |
| 4–6K | 0.091 | 0.0049 | 9.9 | 1.0e-5 (peak) |
| 10–12K | 0.059 | 0.0037 | 8.6 | 8.3e-6 |
| 18–20K | 0.031 | 0.0023 | 8.3 | 3.7e-6 |
| 28–30K | 0.016 | 0.0012 | 3.7 | 4e-7 |
Resume across the job boundary was seamless (loss 0.029 → 0.028 around step 19,870).
**Caveat:** both numbers are measured on *training* batches. They show the model fits the training data; they say nothing about generalisation or task success.

### Recipe fidelity: unavoidable differences from StarVLA (none change the recipe)
| StarVLA | Here | Why |
|---|---|---|
| 8 GPUs × 16 | 1 GPU × 16 × 8 accumulation | we have 1 H200 per job; same 128 samples per step |
| DeepSpeed ZeRO-2 bf16 | fp32 weights + bf16 autocast | no DeepSpeed, for readable code; small numeric differences |
| LeRobot `libero_all` (40 tasks) | our state-replay data, 76 E3 training tasks | that is our experiment |
| weights saved every 10K | same + resume state | 24 h job limit |

### Known confounds vs the OFT run `e3_A0`
Different backbone and head **and**: no augmentation or proprio (OFT: both used); end padding (OFT: starts t ≤ T−8); min_max (OFT: q01/q99); 3.84M samples seen (OFT: 640K). A difference between the two runs cannot be attributed to the model alone.

## ✅ M4c — Held-out demo error (`oftvla/offline_eval.py`, job 1172087)
Same L1 as training, at fixed weights, on 152 val demos (training tasks, never trained on) vs an equal-sized subset of training demos; every 4th sample start.
| Step | Train L1 | Val L1 | Gap | Val gripper acc | Baseline (mean action) |
|---|---|---|---|---|---|
| 10K | 0.061 | 0.0855 | +0.025 | 0.956 | 0.234 |
| 20K | 0.028 | 0.0846 | +0.056 | 0.957 | 0.234 |
| 30K | 0.015 | 0.0842 | +0.069 | 0.958 | 0.234 |
**Reading:** val error is ~3× below the no-learning baseline, but **flat after 10K** while train error falls 4×: the last 20K steps mostly fit the training demos themselves (memorisation), without making held-out demos worse. Caveat: human demos are multimodal, so val L1 has an irreducible floor and is only a weak proxy for closed-loop success.

## ✅ M5 — Policy + LIBERO evaluation
**What:** `oftvla/policy.py` (checkpoint → unnormalised 8×7 chunks, socket server in starvla.sif) + `oftvla/libero_client.py` (LIBERO loop in sym-openvla-oft.sif, StarVLA eval protocol: 10 settle steps, 180° rotation, bilinear 224, open-loop 8-step chunks, gripper > 0.5 → open, 400/520 step limits, env seed 7, training A0 text) + `scripts/eval_libero.sbatch`.
**Pass criterion:** the pipeline runs end to end, and training-task success is in the range of the OFT baseline.
**Evidence (job 1172088, 30K, sanity set 5 training tasks × 10 inits):** **48/50 = 96 %** (OFT `e3_A0` 80K: 48/50). LR1 soup+cheese 10/10, KS10 drawer+bowl 10/10, KS4 bottom/top drawer 8/10, LR2 butter 10/10, LR4 stack bowls 10/10. ~5–8 s per episode. Training A0 text = LIBERO text for all 5 tasks. Videos: `runs/oftvla/e3_A0/eval/videos_sanity_step30000/`.

## ✅ M6 — Held-out tasks (E3 test)
StarVLA's 96.6 % is on the four standard suites and is not comparable to our split; M6 = E3 test, 24 held-out tasks × 50 inits (`e3_test_ids.csv`).
**Evidence (job 1175328, 30K, 3:07 h, 1200/1200 episodes, 2026-10-29):** **323/1200 = 26.9 %** vs OpenVLA-OFT A0 80K 231/1200 = 19.2 % (OpenVLA A1/A1 25.3 %). Per task 8 better / 5 worse / 11 equal, Wilcoxon p = 0.22, task-bootstrap 95 % CI of the difference [−5.4, +20.8] pts. Without the 2 leaked tasks: 24.7 % vs 16.1 %, 7/4/11, p = 0.11, CI [+1.4, +17.8].
By group (Qwen / OpenVLA A0): T1 231/187 of 400, T2 38/1 of 500, L10 50/37, S2 4/4, two-goal 0/2. Striking: KS9 turn on the stove 50 vs 6; KS10 close the top drawer **1 vs 48**; SS2 book front 40 vs 11; **T2 SS3 white mug right of the caddy 35 vs 0** (first real recombination success); L10 soup+sauce 49 vs 32. Videos: `runs/oftvla/e3_A0/eval/videos_e3test_step30000/`.
**Caveat:** the two models differ in backbone AND recipe (aug, proprio, normalisation, samples seen, eval protocol); one seed each.

---

## ✅ M7 — Per-frame A0 / AS / MAS texts (`text_csv`) + smoke test (2026-10-29)
**Question:** how does the language input (A0 sentence vs AS vs MAS sub-task block) change the learned policy?
**Design (decided 2026-10-29):**
- The sub-task block is an **input**, not a target: no text loss, no new head; model, loss and recipe unchanged (StarVLA QwenOFT).
- Texts are used exactly as built in `~/VLA_FT/DataSamples/Libero/A0ASMAS/` (label dictionary v2.3, frozen): `text_A0` = `Instruction: <sentence>`; `text_AS` / `text_MAS` = `text_A0` + newline + 4-line block (Achieved Goals / Current Goal / In-Progress Step / Next Step). The A0 prompt is a word-for-word prefix of the AS/MAS prompts, so an A0 vs AS/MAS difference can only come from the block.
- Teacher forcing: the loader gives the gold text of each frame; the model never decides which step it is in.
- Demo set: the v5 demos (step1 drops removed): 3,565 of the 3,637 `e3_train_ids.csv` demos. All three runs use the same demos, statistics, seed and code; only the text column differs.
- Evaluation (planned): offline on the held-out E3 test demos (`test.csv`, 1,158 demos) with the same text column; mean L1 over all frames + L1 split by distance to the nearest MAS step change. No closed-loop rollouts for this comparison.
**Code:** `data.py` optional `text_csv` (column `text_<setup>`; demos = split ∩ CSV; crash if CSV frames ≠ HDF5 frames; old `labels_csv` path unchanged); `train.py --text_csv`; `scripts/train_e3_text.sbatch`, `scripts/smoke_text.sbatch`, `submit_chain.sh` NAME + args; `scripts/show_prompt.py`. Commit 4e83f7a.
**Exact prompt** (`show_prompt.py`, real processor): one user turn, no system message:
`<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>{text} Please predict the next 8 robot actions: <action>🔍×8<action>.<|im_end|>\n<|im_start|>assistant\n`.
Tokens at L10T01 demo_31 raw 180: A0 177, AS 225, MAS 236 (images 2 × 64). StarVLA quirk kept: " Please predict ..." follows the last block line after a space.
**Tests:** `22 passed in 5.39s` (helma, container): 18 old + 4 new (per-frame text; A0/AS/MAS give identical starts, images, actions, stats; missing frame raises; bad setup raises). Negative control: breaking the per-frame lookup makes 2 new tests fail.
**Smoke (job 1177748, MAS = longest prompts, exact run config 16 × 8, 14 workers, 40 steps, stopped after 0.03 h and resumed):**
- `[data] setup=MAS demos=3565 samples=565397 texts=289 skipped_not_in_text_csv=72` (289 = train texts; the 360 in the build report counts train + test)
- loss 0.377 → 0.239; mse_score 0.0128 / 0.0134; data loading < 1 % of step time
- peak memory **116.2 GB** (torch; nvidia-smi 120.6 GB) vs 105.7 GB for A0 — fits, ~20 GB headroom; flat from step 5 (batches are padded to their longest prompt)
- **4.5 s/step** (A0: 3.9) → 30K steps ≈ 38 h = 2 chained jobs
- resume: saved at step 23 → resumed at 23, loss continuous (0.280 → 0.276); `DONE`, resume state removed
**Lesson:** the smoke job's log folder must exist before `sbatch` (first submission 1177725 started without it and lost its log; cancelled and resubmitted).
**Known limitations of v2.3 (not changed):** AS/MAS differ also in verb form (plain vs -ing) and in the Achieved-Goals state lines; MAS put steps are often shorter than the 8-action chunk.

## ✅ M8 — Full runs e3v5_A0 / e3v5_AS / e3v5_MAS (2026-10-29 → 2026-10-31)
1 H200 per run, 16 × 8 = 128 per step, 30K steps, same recipe as M4b; only the text column differs. `bash scripts/submit_chain.sh scripts/train_e3_text.sbatch 3 e3v5_<SETUP> <SETUP>`.
| Run | Jobs (part 1 → part 2 → no-op) | Resume at | s/step | Peak mem | Finished |
|---|---|---|---|---|---|
| e3v5_A0 | 1178169 → 1178170 → 1178171 | 20,364 | 3.91 | 105.9 GB | 31 Oct 05:01 |
| e3v5_AS | 1178172 → 1178173 → 1178174 | 16,675 | 4.79 | 114.9 GB | 31 Oct 12:11 |
| e3v5_MAS | 1178175 → 1178176 → 1178177 | 17,233 | 4.71 | 116.2 GB | 31 Oct 11:52 |
- All: 3,565 demos, 565,397 samples, 72 v5-dropped demos skipped; distinct training texts A0 68 / AS 146 / MAS 289. 3,000 logged steps each, no duplicates, no NaN/OOM; loss continuous across the job boundary; `DONE`; weights `steps_{10000,20000,30000}_model.pt`; third job exited on `DONE`.
- Code: part 1 ran at 9616201 (+ uncommitted MILESTONES.md), part 2 at 95849d8 (clean). `oftvla/train.py`, `data.py`, `model.py` were last modified before submission (29 Oct 17:44 / 27 Oct) and are byte-identical to 4e83f7a → both parts ran the same training code. `train.csv` md5 a7051944… in every job.
- Training loss (mean per window; train batches only, NOT a result):
| Window | e3_A0 (old) | e3v5_A0 | e3v5_AS | e3v5_MAS |
|---|---|---|---|---|
| 0–2K | 0.163 | 0.163 | 0.160 | 0.153 |
| 4–6K | 0.091 | 0.090 | 0.090 | 0.088 |
| 10–12K | 0.059 | 0.060 | 0.061 | 0.060 |
| 18–20K | 0.031 | 0.031 | 0.032 | 0.033 |
| 28–30K | 0.016 | 0.016 | 0.017 | 0.017 |
**Reading:** the block lowers the training loss only early (MAS 0.153 vs A0 0.163 in the first 2K steps); from ~5K steps all three fit the training demos equally well. Whether the text changes behaviour on held-out demos needs the test evaluation (M9).
