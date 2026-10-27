# oftvla

Minimal, self-contained re-implementation of StarVLA's **QwenOFT** policy
(Qwen3-VL + parallel 🔍 action queries + MLP L1 head), for training and
evaluating on LIBERO with full control over the code.

Reference implementation: [starVLA](https://github.com/starVLA/starVLA) @ `985c223` (MIT).
The action head is adapted from [openvla-oft](https://github.com/moojink/openvla-oft) (MIT).

## Layout
```
oftvla/model.py               QwenOFT model (config, head, gather, loss, train/inference)
tests/test_model_units.py     CPU unit tests (no VLM needed)
scripts/check_equivalence.py  GPU check: same outputs as StarVLA's Qwenvl_OFT
```

## Roadmap
1. [x] model.py + unit tests
2. [ ] equivalence vs StarVLA passes on helma
3. [ ] data.py + train.py (overfit 10 episodes)
4. [ ] policy.py + LIBERO eval (reproduce ~96 %)

## Run (helma, inside starvla.sif)
```bash
SIF=/hnvme/workspace/y307bd13-vgv/containers/starvla.sif
cd ~/VLA_FT/oftvla
apptainer exec --env PYTHONPATH=$PWD $SIF pytest -q tests/test_model_units.py
apptainer exec --nv --env PYTHONPATH=$PWD $SIF python scripts/check_equivalence.py \
    --starvla_root ~/VLA_FT/starVLA --base_vlm <path-or-hf-id of Qwen3-VL-4B-Instruct>
```
