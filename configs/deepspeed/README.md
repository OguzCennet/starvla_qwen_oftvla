# DeepSpeed / accelerate configs

Copied from StarVLA @ 985c223 (`starVLA/config/deepseeds/`):
- `ds_config.yaml`: byte-identical (ZeRO-2, bf16, gradient_clipping 1.0, gradient_accumulation_steps 1).
- `deepspeed_zero2.yaml`: identical except `deepspeed_config_file` points to `./configs/deepspeed/ds_config.yaml` (run from the repo root). `num_processes` / machines are overridden on the `accelerate launch` command line, as the fork does.
