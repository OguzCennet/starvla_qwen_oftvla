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

## ✅ M9 — Held-out test evaluation of e3v5_A0 / AS / MAS (2026-10-31)
**What:** offline, teacher-forced action error on the held-out E3 test demos (`oftvla/heldout_eval.py`, `scripts/heldout_eval.sbatch`; analysis `oftvla/heldout_analysis.py`). Every frame of the 1,158 test demos (186,874 frames), checkpoints 10K / 20K / 30K, each run with the text column it was trained with. Jobs 1190433–1190435.
**Frame-weighted L1 at 30K:** A0 **0.1226** / AS **0.1201** / MAS **0.1188** (mean-action baseline 0.238); flat from 10K.
**Paired over the 24 test tasks:**
- MAS vs A0: L1 −4.3 % (16 better / 8 worse, p = 0.015); near MAS step changes −8.3 % (p = 0.008); gripper agreement +0.8 pt (20 / 2, p = 1.6e-5).
- AS vs A0: not significant.
- Without the 2 leaked tasks: MAS vs A0 −2.6 % (p = 0.05), gripper 19 / 2.
**Caveats:** one seed per setup (run-to-run noise measured in M10); offline teacher-forced L1, not closed-loop success; near/far uses the v5 step boundaries.

## ✅ M10 — Runs e3v5_AS_noAG / e3v5_MAS_noAG / e3v5_A0_seed2 (2026-11-01 → 2026-11-05)
**Question:** (1) do AS / MAS need the "Achieved Goals" lines? (2) how large is run-to-run noise (seed) compared with the M9 differences?
**Design:** identical to the e3v5 runs (M8: same recipe, demos, statistics, code, 1 H200, 16 × 8, 30K steps) except one factor each. Brief: `docs/briefs/noag_runs.md`.
- Texts without Achieved Goals: `~/VLA_FT/DataSamples/Libero/A0ASMAS_noAG/` (`make_noag.py`, `README.md`, `report.txt`). The block (heading + state lines) removed from `text_AS` / `text_MAS`; everything else identical, checks asserted. Each AS / MAS text is now 4 lines: Instruction / Current Goal / In-Progress Step / Next Step. Removing the block merged **no** texts (train A0 68 / AS 146 / MAS 289, as before): the Achieved-Goals lines were redundant given the other lines. Prompt tokens at L10T01 demo_31 raw 180: A0 177, AS 216 (was 225), MAS 217 (was 236).
- `scripts/train_e3_text.sbatch SETUP [TEXTDIR] [SEED]` (defaults `$TEXTS`, 42 → unchanged behaviour). Commit 6e3f335.
- e3v5_A0_seed2 = e3v5_A0 with seed 43: measures run-to-run noise.

| Run | Setup | Texts (train.csv md5) | Seed | Jobs | Node (part 1) | Peak mem (e3v5) | s/step |
|---|---|---|---|---|---|---|---|
| e3v5_AS_noAG | AS | A0ASMAS_noAG (6e194331…) | 42 | 1194245 → 1194246 → 1194247 | h24-20 | 112.3 GB (114.9) | 4.3–4.6 |
| e3v5_MAS_noAG | MAS | A0ASMAS_noAG (6e194331…) | 42 | 1194248 → 1194249 → 1194250 | h24-19 | 113.2 GB (116.2) | 4.3–4.4 |
| e3v5_A0_seed2 | A0 | A0ASMAS (a7051944…, unchanged) | 43 | 1194251 → 1194252 ✗ → 1194253 ✗ → 1211005 | h24-09 | 105.9 GB (105.9) | 3.8–3.9 |

**Start checks (first ~30 steps):** all at commit 6e3f335, `git status` clean, intended train.csv md5 and echoed SETUP / TEXTDIR / SEED; `[data] demos=3565 samples=565397 skipped_not_in_text_csv=72`, texts 146 / 289 / 68; loss falling (A0_seed2 starts at 0.459 vs 0.362 for e3v5_A0 at step 10 and is close by step 30: 0.271 vs 0.266).
**Mid-run (2 Nov 11:20, part 1 at ~19 h):** AS_noAG step 15,230, MAS_noAG 14,110, A0_seed2 16,550; `steps_10000_model.pt` saved in all three, resume every 1,000 steps, no NaN / OOM / traceback. Training loss (mean per window; train batches only, NOT a result):
| Window | e3v5_A0 | A0_seed2 | e3v5_AS | AS_noAG | e3v5_MAS | MAS_noAG |
|---|---|---|---|---|---|---|
| 0–2K | 0.164 | 0.164 | 0.160 | 0.161 | 0.153 | 0.155 |
| 4–6K | 0.090 | 0.090 | 0.090 | 0.090 | 0.088 | 0.088 |
| 10–12K | 0.060 | 0.059 | 0.061 | 0.062 | 0.060 | 0.061 |
| 13–14K | 0.048 | 0.048 | 0.049 | 0.050 | 0.049 | 0.049 |
Each new run tracks its e3v5 counterpart within ~0.002: removing Achieved Goals barely changes the training fit, and seed 43 gives almost the same curve as seed 42. **Held-out eval:** `heldout_eval.py` now takes the test CSV from the run's args.json (test.csv next to the training text_csv), so the noAG runs are scored on `A0ASMAS_noAG/test.csv` (md5 7e73405d…).
**Training finished:**
| Run | Jobs | Resume at | Finished | Loss 28–30K (e3v5) | Held-out eval job |
|---|---|---|---|---|---|
| e3v5_AS_noAG | 1194245 → 1194246 → 1194247 (exit on `DONE`) | — | 3 Nov 06:21 | 0.0164 (0.017) | 1210759 |
| e3v5_MAS_noAG | 1194248 → 1194249 → 1194250 (exit on `DONE`) | — | 3 Nov 08:12 | 0.0166 (0.017) | 1210760 |
| e3v5_A0_seed2 | 1194251 → 1194252 ✗ → 1194253 ✗ → 1211005 | 20,093; 24,000 | 4 Nov 02:26 | 0.0156 (0.016) | 1211016 |
- All three: 3,000 logged steps, no NaN, `steps_{10000,20000,30000}_model.pt`, `DONE`. Part 1 at 6e3f335, later jobs at db8c183 (training code identical).
- **A0_seed2 failure:** 1194252 resumed at 20,093, saved resume points up to 24,000, then crashed on 2 Nov 21:02 writing the step-25,000 resume file (`PytorchStreamWriter failed writing file data.pkl: file write failed`, a write failure on /hnvme). 1194253 started 6 s later and failed after 4 s (exit 0:53) without creating its log → /hnvme not writable at that time (inode quota near its limit, or a filesystem hiccup; on 5 Nov: 26.5K files of 61K soft). Resubmitted as 1211005: resumed cleanly from 24,000 and finished. `log.jsonl` therefore has 100 duplicate steps (24,010–25,000, replayed with a new shuffle order); keep the last entry per step.
**Held-out eval** (every frame of the 1,158 test demos, 186,874 frames; noAG runs on `A0ASMAS_noAG/test.csv` 7e73405d…, A0_seed2 on `A0ASMAS/test.csv` e2fb4c48…, both resolved from args.json). Frame-weighted L1:
| Run | 10K | 20K | 30K | M9 counterpart (30K) |
|---|---|---|---|---|
| A0_seed2 | 0.1230 | 0.1228 | **0.1224** | A0 0.1226 |
| AS_noAG | 0.1200 | 0.1208 | **0.1202** | AS 0.1201 |
| MAS_noAG | 0.1175 | 0.1196 | **0.1186** | MAS 0.1188 |
**Analysis (30K, 6 runs, paired over the 24 test tasks; `runs/oftvla/analysis_6runs/`, report `step30000_report.txt`).** diff = b − a, L1 negative = b better; grip_agree positive = b better. MAS vs A0 reproduces M9 exactly.
| b vs a | L1 all | L1 near | L1 far | grip all | grip near | grip far |
|---|---|---|---|---|---|---|
| **A0_seed2 vs A0** (noise) | −0.3 %, 9/15, p 0.92 | −2.0 %, 12/12, p 0.53 | +0.3 %, 9/15, p 0.60 | +0.4 pt, 14/8, p 0.15 | +0.9 pt, 10/10, p 0.67 | +0.3 pt, 15/7, p 0.10 |
| MAS vs A0_seed2 | −4.0 %, 18/6, p 0.0018 | −6.4 %, 18/6, p 0.0072 | −3.4 %, 17/7, p 0.0031 | +0.4 pt, 16/5, p 0.018 | +0.6 pt, 15/7, p 0.046 | +0.4 pt, 15/7, p 0.054 |
| MAS_noAG vs A0_seed2 | −4.0 %, 20/4, p 0.0003 | −5.3 %, 18/6, p 0.015 | −3.6 %, 21/3, p 0.0007 | +0.4 pt, 16/6, p 0.0083 | +0.4 pt, 14/8, p 0.098 | +0.3 pt, 17/5, p 0.050 |
| AS vs A0_seed2 | −2.3 %, 15/9, p 0.084 | −2.6 %, 13/11, p 0.58 | −2.2 %, 17/7, p 0.049 | −0.1 pt, 13/9, p 0.66 | +0.1 pt, 9/13, p 0.73 | −0.1 pt, 12/10, p 0.78 |
| AS_noAG vs A0_seed2 | −2.5 %, 15/9, p 0.034 | −4.2 %, 15/9, p 0.11 | −1.9 %, 15/9, p 0.025 | −0.1 pt, 14/8, p 0.25 | +0.1 pt, 13/9, p 0.92 | −0.2 pt, 10/12, p 0.90 |
| MAS vs A0 (M9) | −4.3 %, 16/8, p 0.015 | −8.3 %, 17/7, p 0.0079 | −3.1 %, 15/9, p 0.040 | +0.8 pt, 20/2, p 1.6e-5 | +1.5 pt, 13/9, p 0.13 | +0.6 pt, 19/3, p 6.5e-5 |
| MAS_noAG vs A0 | −4.2 %, 17/7, p 0.0035 | −7.2 %, 18/6, p 0.0031 | −3.3 %, 17/7, p 0.011 | +0.8 pt, 19/3, p 9.9e-5 | +1.3 pt, 11/11, p 0.39 | +0.6 pt, 19/3, p 5.0e-4 |
| MAS_noAG vs MAS | +0.1 %, 13/11, p 0.64 | | | −0.1 pt, 11/11, p 0.95 | | |
| AS_noAG vs AS | −0.2 %, 14/10, p 1.0 | | | −0.0 pt, 11/11, p 0.80 | | |
Per-task L1 differences (all frames): mean |d| / max |d| — A0 vs A0_seed2 0.0033 / 0.0077 (L90T08); A0 vs MAS 0.0070 / 0.0441 (L90T41); A0 vs MAS_noAG 0.0069 / 0.0359 (L90T41).
**Reading:**
- Seed noise is small on average (A0_seed2 vs A0: −0.3 %, p 0.92, CI [−0.0019, +0.0012] contains 0) but not zero per task (mean |d| 0.0033, up to 0.0077): per-task differences below ~0.008 are within seed noise. The MAS effect is about 15× the seed effect on the mean (−0.0049 / −0.0052 vs −0.0003) and is carried mainly by fewer large per-task gains (max 0.044 on L90T41).
- **The MAS gain holds against the second A0:** MAS and MAS_noAG beat A0_seed2 by ~4 % on all frames (18/6 and 20/4 tasks, p ≤ 0.002), more near step changes (−5 to −6 %) than far (−3.5 %). AS gains are about half as large and only borderline.
- Achieved Goals are not needed: removing them changes nothing measurable (MAS_noAG vs MAS +0.1 %, AS_noAG vs AS −0.2 %).
- Gripper: MAS vs A0_seed2 is +0.4 pt (16/5, p 0.018), half the +0.8 pt against A0; A0_seed2 itself is +0.4 pt above A0. So about half of the M9 gripper gain is seed noise. Near step changes the seed difference (+0.9 pt) is as large as the MAS gain → the near-band gripper numbers are not interpretable.
- Caveats: still one seed for AS / MAS / noAG runs (noise estimated from the A0 pair only); offline teacher-forced L1, not closed-loop success.


## ⏳ M11 — Flow heads (QwenGR00T / QwenPI): spec and decisions (2026-11-05)
**What:** read-only analysis of the fork's GR00T and PI heads and their original LIBERO recipe (`docs/briefs/flow_heads_spec.md`, commit f22e6ec), then decisions for step 2 (spec §11). No code yet.
**Counted parameters:** GR00T head 161.5M (model 4.60B), PI head 3.352B (model 7.79B). PI full fine-tuning does not fit on 1 H200 (≈ 181 GB) and needs ≥ 2 H200 with ZeRO-2. GR00T fits on 1 H200 (≈ 120 GB, estimate).
**Decisions:**
- **VLM:** Qwen3-VL-4B-Instruct by default, behind a swappable per-family adapter (fork-style registry). The adapter gives hidden states + mask, hidden size and layer count from the config, and prompt / special tokens. Run dirs carry a VLM tag; `base_vlm` is recorded in args.json.
- **Prompt:** `CoT_prompt` wraps the whole per-frame text for GR00T / PI.
- **Precision:** DeepSpeed ZeRO-2 bf16 (the fork's config) for GR00T / PI.
- **GPUs:** 8 × 16. Fallback 1 × 16 × 8, GR00T only. The same setup for all three texts of a head.
- **Quirks:** keep all fork quirks.
- **Held-out sampling policy:** later.
**Decisions of 2026-11-05 (second round, spec §11):**
- **PI forward = 985c223 legacy switch** (every DiT block cross-attends to its own VLM layer, padding mask applied, VLM in SDPA). The published checkpoint `1208_libero_all_QwenPI_qwen3` was trained with the code at f4473e4 ≡ 09b5652, which had no DiT padding mask and ran the VLM with FlashAttention-2. That forward attends to left padding, whose amount depends on text length and batch partners, so it would confound A0/AS/MAS and make held-out predictions batch-dependent. Recorded as a known difference to the published run (spec §12.1).
- **Schedule:** the original total-step cosine schedule, stopped at step 30K (as the published numbers were produced). PI 100K schedule (released run config). GR00T **80K** schedule: `run_libero_train.sh:47` overrides the yaml's 100K; the released Qwen2.5-VL GR00T LIBERO run also used 80K. Global batch 128; checkpoints at 10K / 20K / 30K.
- **Known OFT deviations (recorded, not changed):** (1) no `CoT_prompt`; (2) fp32 weights + autocast instead of DeepSpeed bf16; (3) a full 30K cosine decay (LR 0.04 × peak at 30K) instead of the published 80K schedule (≈ 0.76 × peak at 30K).
**Evidence from the released checkpoints (spec §12.2):** the README's "30K" numbers are step-30K evaluations of an 80K (OFT, 8 × 16) and a 100K (PI, 16 × 8) schedule, both with CoT_prompt and full fine-tuning.

## ✅ M12 — QwenGR00T / QwenPI frameworks: full-size equivalence with the fork (2026-11-05)
**What:** `oftvla/vlm` (VLM adapter registry, Qwen3-VL), `oftvla/heads/flow_matching` (DiT, encoders, GR00T and PI heads ported from 985c223; changes C1-C5 in the package docstring), `oftvla/flow_model.py` (QwenGR00T, QwenPI). CoT_prompt on (whole per-frame text), robot state off, VLM attention `sdpa` for both heads (QWen3.py:52-53 forces it for every framework), action horizon 8 (yaml:18; = our chunk), PI = legacy switch (`use_canonical_forward=False`). OFT unchanged (commit 9637a0c: `check_equivalence.py` at atol 0 and all captured inputs / predictions / losses `torch.equal` before vs after; prompt ids identical).
**Component tests (CPU fp32, 25):** loss, predict_action, every parameter and input gradient `torch.equal` vs the fork modules for both DiT modes and no / full / padded masks; same RNG order with dropout on; negative controls (weight, mask, t) break the comparison.
**Full size on 1 H200, bf16 (`scripts/checks/flow_equivalence.py`; 6 E3 val frames, A0/AS/MAS mixed, 181-256 tokens, 5 padded rows; real Qwen3-VL-4B; head initialised by the fork):**
| | GR00T | PI |
|---|---|---|
| loss (injected noise / t), `torch.equal` | ✅ 1.14567924 | ✅ 3409.19873047 |
| predict_action (injected init noise), `torch.equal` | ✅ | ✅ |
| gradients, deterministic kernels (fork reproduces itself bitwise) | ✅ 961/961 tensors bitwise equal | ✅ 1241/1241 bitwise equal |
| gradients, default kernels: fork-vs-fork rel_l2 VLM / head | 0.92-1.18e-2 / 0.85-1.28e-4 | 0.79-0.96e-2 / 2.7-3.6e-4 |
| gradients, default kernels: ours-vs-fork rel_l2 VLM / head | 0.86-1.02e-2 / 1.05-1.42e-4 | 0.81-0.90e-2 / 3.2-3.6e-4 |
| negative controls (head weight, mask entry, CoT off) | all differ | all differ |
| peak memory fwd+bwd, batch 6 / micro-batch 16 (bf16 weights + grads, no optimiser) | 26.0 / 55.1 GB | 47.1 / 81.9 GB |
Jobs: GR00T 1238639 (default), 1238731 (deterministic); PI 1238670 (default), 1238758 (deterministic).
**Reading:** both frameworks are bitwise identical to the fork in forward, sampling and backward. With default (non-deterministic) attention-backward kernels the fork itself does not reproduce its gradients; ours lies in the same band (with 4 runs, the head rel_l2 maximum is 1.42e-4 vs 1.28e-4 for fork-vs-fork, a sampling effect, removed by the deterministic run). The PI loss at initialisation is ~3400 in both: the legacy forward cross-attends to the raw (un-normed) residual streams of the VLM layers.

## ✅ M13 — Flow trainer (GR00T / PI): DeepSpeed integration and smoke runs (2026-11-06)
**What:** `oftvla/train_flow.py` (accelerate + DeepSpeed ZeRO-2 bf16, the fork's `ds_config.yaml` byte-identical, launched with `accelerate launch` as in `run_libero_train.sh`), `oftvla/flow_data.py` (deterministic resumable global sample stream), `scripts/train_flow.sbatch` (multi-node launcher). OFT training path unchanged: 3 optimizer steps of the exact e3v5_MAS config give identical loss / grad_norm with the old commit 4b29fd5 and twice with the new code (job 1238967, step 1 loss 0.40377118811011314 / grad_norm 31.3998…).
**Infrastructure:** h200 partition = 4 × H200 (140 GB) per node, 128 CPUs; MaxNodes 64, MaxTime 24 h. /hnvme: no byte quota, files 26.5K / 61K soft. → original layouts kept: **GR00T 2 nodes × 4 GPUs × 16, PI 4 nodes × 4 × 8** (global 128, no accumulation).
**Infra-forced deviation (launcher only):** the container has no libibverbs; the host verbs libraries (and the mlx5 provider targets) are bound into it (`OFTVLA_HOST_IB=1`). Without them NCCL falls back to TCP sockets: NCCL preflight reduce_scatter 5.8 GB/s (sockets) vs 85 GB/s (IB, GPU Direct RDMA) on 2 nodes; GR00T 3.3 s/step vs 0.98 s/step.
**Config-level equivalence vs the fork's own `setup_optimizer_and_scheduler` (`flow_trainer_config.py`): PASS both heads.** Groups `qwen_vl_interface` (LR 1e-5) then `action_model` (1e-4), no base group; GR00T 713 / 248 tensors (4,437,815,808 / 161,472,775 params), PI 713 / 528 (– / 3,352,072,737); AdamW fused, betas (0.9, 0.95), eps 1e-8, wd 1e-8; clip 1.0 (DeepSpeed `gradient_clipping`); `cosine_with_min_lr`, min_lr 1e-6, warmup 5000. LR factor at 0 / 5K / 10K / 30K: GR00T (80K) 0 / 1 / 0.9895 / 0.7600; PI (100K) 0 / 1 / 0.9935 / 0.8451.
**Data order:** first 1000 samples identical for A0 / AS / MAS (md5 622c3087…), texts differ in 1000/1000; 8 × 16 and 16 × 8 read the same global stream.
**Smoke runs (exact config, MAS, 300 steps, stopped at 150 and resumed in a new launch; jobs 1240699 GR00T, 1240700 PI, commit eff5508):**
| | GR00T 2 × 4 × 16 | PI 4 × 4 × 8 |
|---|---|---|
| loss steps 1-25 → 276-300 | 1.160 → 0.433 | 3590 → 1.154 (≈ 1.17 from step ~225) |
| grad norm (pre-clip) | 6-31 (window means), max 139 | 1.6e8 → 5.3 |
| step time (median / mean incl. eval) | 0.975 / 1.028 s | 1.195 / 1.242 s |
| peak memory per GPU | 56.9-64.2 GB | 55.8-61.9 GB |
| projected 30K (incl. eval; + ~30 resume saves) | ≈ 8.6 h + ~0.2 h | ≈ 10.4 h + ~0.3 h |
| resume state (latest only) | 64.39 GB, 21 files | 109.06 GB, 37 files |
| model-only checkpoint (bf16) | 9.98 GB | 16.36 GB |
| resume: step / draws / LR / indices | 150 → 151, draws 151, LR 3.0e-7 → 3.02e-7, rank-0 indices = stream (PASS all 300 steps) | same (PASS) |
One 23 h job per run is enough. Disk for 3 + 3 runs: ≤ 3 × (64 + 30) + 3 × (109 + 49) ≈ 0.76 TB while running, 0.24 TB after (the resume state is removed at DONE).
**PI initial loss:** fork clip = 1.0 (`yaml:64`, `ds_config gradient_clipping`). The released PI checkpoint repo (and its history) has no training logs, so the ~3400-3600 start cannot be compared with the published run; our curve drops to the GR00T-like level ≈ 1.15 within ~225 steps (warmup LR ≤ 4.5e-6 on the head).
**Open:** the released PI run used base LR 3e-5 (fork yaml 2.5e-5). The base group is empty, but `min_lr_rate = min_lr / base LR` changes: LR factor at 30K 0.8444 (3e-5) vs 0.8451 (2.5e-5). Default kept at 2.5e-5.
