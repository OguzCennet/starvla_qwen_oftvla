# Equivalence / regression checks

Logs and captured tensors go to `~/VLA_FT/oftvla_checks/` (outside the repo).

| Script | What it checks | How to rerun |
|---|---|---|
| `oft_gate.sbatch` + `capture_oft_io.py` | OFT unchanged by a refactor (step 2a, M11): `check_equivalence.py` at atol 0 vs the fork, plus inputs / predictions / losses for L10T01 demo_31 raw 180 (A0/AS/MAS, alone and batched) | `sbatch scripts/checks/oft_gate.sbatch before` on the old commit, `... after` on the new one, then compare `io_before.pt` / `io_after.pt` with `torch.equal` per key (both runs share `head_state.pt`, so the random OFT head is identical) |
| `show_prompt.py --save_ids` (in `scripts/`) | prompt token ids unchanged | run before and after with `--save_ids FILE`, compare per column |
| `flow_equivalence.py` + `flow_equivalence.sbatch` | full-size GR00T / PI vs the fork on GPU, bf16 (step 3): loss and predict_action `torch.equal`, gradients vs the fork's own run-to-run spread, negative controls, peak memory | `sbatch scripts/checks/flow_equivalence.sbatch gr00t` (or `pi`); add `--deterministic --fork_runs 2` for deterministic kernels (then the gradients must be bitwise equal) |
| `oft_step_check.sbatch` | OFT training path unchanged by the flow trainer (step 4): 3 optimizer steps of the exact e3v5_MAS config with an old commit (git worktree) and twice with the current code; loss / grad_norm per step must match | `git worktree add ../oftvla_checks/wt_OLD OLD`, then `sbatch scripts/checks/oft_step_check.sbatch ../oftvla_checks/wt_OLD` |
| `flow_trainer_config.py` | optimiser / scheduler of `train_flow.py` vs the fork's own `setup_optimizer_and_scheduler` (CPU): groups, tensors / params per group, hyperparameters, LR at 0 / 5K / 10K / 30K | `run_py scripts/checks/flow_trainer_config.py --head gr00t --starvla_root ~/VLA_FT/starVLA --base_vlm $BASE_VLM --out F.json` (and `--head pi`) |
| `flow_data_order.py` | the sample order is identical for A0 / AS / MAS (first 1000 samples) and every GPU layout covers the same global stream | `run_py scripts/checks/flow_data_order.py --data_root $DATA --train_ids $SPLITS/e3_train_ids.csv --text_csv $TEXTS/train.csv --stats $RUNS/e3v5_A0/dataset_statistics.json` |
| `flow_data_order_preflight.py` | preflight of the 6 flow runs: first 1000 sample indices identical for A0 / AS / MAS within each head, in the trained layout (GR00T 8 × 16, PI 16 × 8), with the training split filters; rank-0 indices vs the smoke runs' logs | `run_py scripts/checks/flow_data_order_preflight.py` |
| `nccl_bw.py` | NCCL all_reduce / reduce_scatter bus bandwidth across nodes (sockets vs InfiniBand) | run by `train_flow.sbatch` with `OFTVLA_NCCL_TEST=1` |
| `flow_smoke_report.py` | smoke run of `train_flow.py`: loss / grad-norm curves, step time, peak memory, projected time, and data order + resume continuity (logged rank-0 indices vs `GlobalStream`) | `run_py scripts/checks/flow_smoke_report.py --run_dir R --num_samples 565397 [--png F]` |
