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

## ⬜ M3 — Data (`oftvla/data.py`)
**What:** HDF5 loader for our regenerated LIBERO data, producing StarVLA-format samples.
**Rule (decided 2026-10-27): use nothing the original StarVLA LIBERO pipeline does not use.** Therefore:
every step is a chunk start with end padding by repeating the last action; min_max normalisation of the 6 arm dims, gripper not normalised; PIL default resize 256→224; no augmentation; no proprio; actions as float16.
Kept from our data definition (shared with `e3_A0`): demo selection by `e3_train_ids.csv`, status-ok demos only, A0 sentence, gripper 1 = open / 0 = close (the same convention StarVLA's LIBERO eval expects).
**Guard:** the loader refuses to start if a training demo belongs to a test task (`--forbidden_ids e3_testdemo_ids.csv`). This changes no data; it only turns a silent split error into a crash.
**Pass criterion:** `tests/test_data.py` passes (6 tests).

## ⬜ M4 — Training (`oftvla/train.py`) smoke test
**What:** `scripts/smoke_mini.sbatch`: 300 steps on the mini set (5 tasks), interrupted once and resumed.
**Pass criteria:** (a) the loss clearly falls (gradients reach the head and the VLM); (b) the resume continues at the saved step; (c) GPU memory fits; (d) the measured seconds per step tell us how long the full run takes.

## ⬜ M4b — Full run e3_A0 (`scripts/train_e3_A0.sbatch`)
76 tasks, A0, StarVLA LIBERO recipe: 30K steps × 128 samples, 5K warmup, cosine down to 4 % of peak, AdamW (0.9, 0.95), lr VLM 1e-5 / head 1e-4, full fine-tuning, grad clip 1.0, seed 42.

### Recipe fidelity: unavoidable differences from StarVLA (none change the recipe)
| StarVLA | Here | Why |
|---|---|---|
| 8 GPUs × 16 | 1 GPU × 16 × 8 accumulation | we have 1 H200 per job; same 128 samples per step |
| DeepSpeed ZeRO-2 bf16 | fp32 weights + bf16 autocast | no DeepSpeed, for readable code; small numeric differences |
| LeRobot `libero_all` (40 tasks) | our state-replay data, 76 E3 training tasks | that is our experiment |
| weights saved every 10K | same + resume state | 24 h job limit |

### Known confounds vs the OFT run `e3_A0`
Different backbone and head **and**: no augmentation or proprio (OFT: both used); end padding (OFT: starts t ≤ T−8); min_max (OFT: q01/q99); 3.84M samples seen (OFT: 640K). A difference between the two runs cannot be attributed to the model alone.

## ⬜ M5 — Policy + LIBERO evaluation
**What:** `policy.py` (unnormalise, threshold the gripper) and `eval_libero.py`.
**Pass criterion:** the full pipeline runs end to end on one task.

## ⬜ M6 — Reproduce the reference result
**Pass criterion:** LIBERO average close to StarVLA's QwenOFT (96.6 %, ±1.5 for 500 trials).
**Why:** Only after this may the repo be modified for experiments (learned queries, bidirectional query attention, language conditions).

---

### Ideas parked for after M6
- Learned query embeddings instead of 🔍 tokens.
- Bidirectional attention among the queries (OpenVLA-OFT style): does 🔍₁ suffer from seeing less context?
