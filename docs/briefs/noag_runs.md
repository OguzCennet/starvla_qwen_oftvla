# Brief: three new training runs (no Achieved Goals + A0 seed 2)

Repo: ~/VLA_FT/starvla_qwen_oftvla (branch main, remote origin = GitHub). Read MILESTONES.md (M7, M8) and
scripts/common.sh first. Work step by step and STOP at the marked checkpoints to show the user the output.

## Goal
Start three 30K-step QwenOFT training runs, identical to the finished e3v5_* runs except for one factor each:
| Run dir (runs/oftvla/...) | Setup | Text CSV | Seed | Differs from |
|---|---|---|---|---|
| e3v5_AS_noAG | AS | A0ASMAS_noAG/train.csv | 42 | e3v5_AS: no "Achieved Goals" lines |
| e3v5_MAS_noAG | MAS | A0ASMAS_noAG/train.csv | 42 | e3v5_MAS: no "Achieved Goals" lines |
| e3v5_A0_seed2 | A0 | A0ASMAS/train.csv (unchanged) | 43 | e3v5_A0: seed only (measures run-to-run noise) |

## Rules (user's)
- Use nothing the original StarVLA recipe does not use; do NOT change oftvla/model.py, oftvla/train.py, oftvla/data.py.
- One GPU per run (1 x H200, 16 x 8 accumulation, as before). Slurm: --account=y307bd --partition=h200 --gres=gpu:h200:1.
- Python on helma only via the container (`source scripts/common.sh; run_py ...`) or the vgv venv ($PY). Never `pip install --user`.
- No new files on /hnvme except the three run dirs (inode quota is at its limit). The new CSVs go to /home.
- Git commit messages: NO "Co-Authored-By", no mention of Claude or any AI tool. Commit before submitting so the job logs record the commit; then `git push`.
- Do not touch the existing runs/oftvla/e3v5_{A0,AS,MAS} folders.

## Step 1 — build ~/VLA_FT/DataSamples/Libero/A0ASMAS_noAG/
Write `~/VLA_FT/DataSamples/Libero/A0ASMAS_noAG/make_noag.py` (csv module only; imports at the top). For each of
train.csv, val.csv, test.csv in ../A0ASMAS/:
- Read with csv.DictReader (open(..., newline="")); texts contain real newlines.
- In text_AS and text_MAS remove the block that starts with the line beginning "Achieved Goals:" and ends just
  before the line beginning "Current Goal:" (the heading line plus all state lines below it; "Achieved Goals: none"
  is a single line). Nothing else changes: text_A0, all other columns, column order, row order stay identical.
- Write with csv.DictWriter (same fieldnames, newline="").
Checks (the script must assert all of them and print a short report):
1. Same row count per file as the source (train 565,397 / val 23,339 / test 186,874).
2. text_A0 and every non-text column identical to the source, row by row.
3. Every new text_AS / text_MAS has exactly 4 lines, starting in this order with "Instruction:", "Current Goal:",
   "In-Progress Step:", "Next Step:".
4. Every new text equals the source text with only the Achieved-Goals lines removed (rebuild by line filtering and compare).
5. The substring "Achieved" occurs nowhere in the new files.
Report: distinct texts per column and split; the longest text per column; the AS and MAS texts of task L10T01
demo_31 at raw_step 180 (old and new). Also write README.md (what, why, source md5s) and md5sums.txt.
Run it with the vgv venv: `PYTHONNOUSERSITE=1 $PY make_noag.py`.
**CHECKPOINT 1: show the report to the user.**

## Step 2 — prompt check
`run_py scripts/show_prompt.py --base_vlm $BASE_VLM --csv $HOME/VLA_FT/DataSamples/Libero/A0ASMAS_noAG/train.csv --task_id L10T01 --demo demo_31 --raw_step 180`
Expected: A0 unchanged (177 tokens); AS / MAS shorter than before (225 / 236) and without "Achieved Goals".
**CHECKPOINT 2: show the output.**

## Step 3 — make the run script configurable (backward compatible)
In scripts/train_e3_text.sbatch: `SETUP=$1` (as now), `TEXTDIR=${2:-$TEXTS}`, `SEED=${3:-42}`; use
`--text_csv $TEXTDIR/train.csv --seed $SEED` and `md5sum $TEXTDIR/train.csv`; echo SETUP, TEXTDIR, SEED at the top.
With no extra arguments the script must behave exactly as before. Check with `bash -n` and `sbatch --test-only`.
Run the tests: `apptainer exec --env PYTHONPATH=$REPO:$PYLIBS:$HOME/pylibs/pytest $SIF python -m pytest -q tests/` (30 pass).
Commit (scripts/train_e3_text.sbatch and this brief), push.

## Step 4 — submit (3 chained jobs each, same as the e3v5 runs)
```
NOAG=$HOME/VLA_FT/DataSamples/Libero/A0ASMAS_noAG
bash scripts/submit_chain.sh scripts/train_e3_text.sbatch 3 e3v5_AS_noAG  AS  $NOAG
bash scripts/submit_chain.sh scripts/train_e3_text.sbatch 3 e3v5_MAS_noAG MAS $NOAG
bash scripts/submit_chain.sh scripts/train_e3_text.sbatch 3 e3v5_A0_seed2 A0  $TEXTS 43
```
When each first job has logged ~20 steps, check in runs/oftvla/<run>/slurm_<id>.out:
- the commit hash, `git status` clean, md5 of the intended train.csv, the echoed SETUP/TEXTDIR/SEED;
- `[data] setup=... demos=3565 samples=565397 ... skipped_not_in_text_csv=72`; texts: AS_noAG / MAS_noAG fewer or equal
  to 146 / 289, A0_seed2 exactly 68;
- loss falling, gpu_mem_gb at or below the e3v5 values (A0 105.9, AS 114.9, MAS 116.2), sec_per_step ~4-5.
**CHECKPOINT 3: report job IDs, commit hash and these lines to the user.**

## Later (not now)
The held-out evaluation must use the matching test CSV: scripts/heldout_eval.sbatch currently hard-codes
$TEXTS/test.csv, so for the noAG runs it needs the noAG test.csv. Discuss with the user before changing it.
