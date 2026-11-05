# Equivalence / regression checks

Logs and captured tensors go to `~/VLA_FT/oftvla_checks/` (outside the repo).

| Script | What it checks | How to rerun |
|---|---|---|
| `oft_gate.sbatch` + `capture_oft_io.py` | OFT unchanged by a refactor (step 2a, M11): `check_equivalence.py` at atol 0 vs the fork, plus inputs / predictions / losses for L10T01 demo_31 raw 180 (A0/AS/MAS, alone and batched) | `sbatch scripts/checks/oft_gate.sbatch before` on the old commit, `... after` on the new one, then compare `io_before.pt` / `io_after.pt` with `torch.equal` per key (both runs share `head_state.pt`, so the random OFT head is identical) |
| `show_prompt.py --save_ids` (in `scripts/`) | prompt token ids unchanged | run before and after with `--save_ids FILE`, compare per column |
| `flow_equivalence.py` + `flow_equivalence.sbatch` | full-size GR00T / PI vs the fork on GPU, bf16 (step 3): loss and predict_action `torch.equal`, gradients vs the fork's own run-to-run spread, negative controls, peak memory | `sbatch scripts/checks/flow_equivalence.sbatch gr00t` (or `pi`); add `--deterministic --fork_runs 2` for deterministic kernels (then the gradients must be bitwise equal) |
