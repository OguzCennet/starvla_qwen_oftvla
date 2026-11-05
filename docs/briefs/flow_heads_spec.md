# Spec: StarVLA QwenGR00T and QwenPI heads in oftvla (step 1, read-only analysis)

Status: analysis only. No code changed, no GPU job run. Written 2026-11-05.

**Reference.** `~/VLA_FT/starVLA` @ `985c223`. The fork's HEAD is `f22001b`, but `git diff 985c223 HEAD -- starVLA/model starVLA/training starVLA/config` is empty, so every model and trainer line cited below is the same at 985c223 and HEAD. The only post-985c223 changes are our own `rb_tools/`, `lang_hdf5` and the LIBERO eval files. Paths are relative to the fork root unless they start with `oftvla/`.

**Environment used for the checks.** Container `starvla.sif`: torch 2.6.0+cu124, transformers 4.57.0, diffusers 0.39.0, deepspeed 0.16.9, accelerate 1.5.2, flash_attn 2.7.4.post1. Base VLM: `Qwen3-VL-4B-Instruct` (text hidden 2560, 36 layers, 32 heads; 4,437,815,808 parameters).

**Main findings** (detail in the sections below):
1. **Prompt wrapper.** The original LIBERO yaml sets `CoT_prompt` (`starvla_cotrain_libero.yaml:40`), and `QWen3.py:126-128` wraps every instruction with it. oftvla never applies it. This also affects our finished OFT runs (§10, D3).
2. **Two versions of the PI attention pattern.** At 985c223, QwenPI runs the DiT with self-attention interleaved. The published StarVLA-π LIBERO checkpoint was trained with the older "legacy all-cross" forward, where every block cross-attends (README:80-90, commits c521dec / 631aae0). Which one counts as "original" needs your decision (§9, Q1).
3. **Repeat factor.** PI's `repeated_diffusion_steps` is hard-coded to 2 (`QwenPI.py:224`), whatever the config says. GR00T uses 8 (default `QwenGR00T.py:97`; the LIBERO yaml does not override it).
4. **Gradient accumulation.** The original runs **without** accumulation. `trainer.gradient_accumulation_steps: 4` (yaml:65) is never read; the DeepSpeed config sets 1. Global batch = 16 × number of processes; the accelerate config defaults to 8 processes, so 128.
5. **Freezing.** The original fine-tunes everything. The yaml freezes the VLM (yaml:57), but the run script passes an empty `--trainer.freeze_modules`. That turns the value into the boolean `true`, which the freeze code ignores (§6).
6. **DeepSpeed.** DeepSpeed works in our container. M4b's "no DeepSpeed, for readable code" is therefore not infrastructure-forced (§10, D2). For the flow heads this matters numerically: the noise, `t` and the timestep bucketing all run in the hidden-state dtype, which is bf16 in the original.
7. **PI memory.** PI full fine-tuning (7.79B parameters) does **not** fit on one H200. It needs at least 2 H200 with ZeRO-2. GR00T fits on 1 H200 (§7).

---

## 1. File map (fork)

| File | Role |
|---|---|
| `starVLA/model/framework/VLM4A/QwenGR00T.py` | framework: defaults dataclass (47-120), VLM + head construction (150-165), `forward` (167-221), `predict_action` (223-278) |
| `starVLA/model/framework/VLM4A/QwenPI.py` | framework: defaults (42-104), construction (121-165), `_encode_vl_hidden_states` (167-184), `forward` (186-245), `predict_action` (247-295), `predict_action_realtime` (297-, RTC, not used on LIBERO) |
| `starVLA/model/modules/action_model/GR00T_ActionHeader.py` | GR00T head: `MLP` (52-59), `ActionEncoder` (62-101), `DiTConfig` table (190-193), `FlowmatchingActionHead` (196-429): `sample_time` 305-307, `forward` 312-363, `predict_action` 365-421 |
| `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py` | PI head: `MLP` (51-58, defaults hidden 1024 / out 2048), `ActionEncoder` (61-100), fallback `DiTConfig` (189-194), `LayerwiseFlowmatchingActionHead` (197-640): `sample_time` 281-283, `forward` 288-347, `predict_action` 349-408; factory 642-652 |
| `starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py` | shared DiT: `TimestepEncoder` (31-41), `AdaLayerNorm` (44-67), `BasicTransformerBlock` (70-185), `DiT` (188-333) |
| `starVLA/model/modules/action_model/flow_matching_head/action_encoder.py` | `SinusoidalPositionalEncoding` (24-54) and `swish` (20-21), imported by both heads. (Its own `ActionEncoder` with W1/W2/W3 is **not** used: both heads define their own with `layer1/2/3`.) |
| `starVLA/model/framework/share_tools.py` | `merge_framework_config` (186-258), `populate_layerwise_dit_cfg` (261-295), `apply_config_compat` (418-524) |
| `starVLA/model/modules/vlm/QWen3.py` | VLM wrapper: forces `sdpa` (53), loads bf16 (63), left padding (67), **sets `config.hidden_size = text_config.hidden_size` (74)**, prompt / `CoT_prompt` (114-171) |
| `examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml` | the LIBERO training config (framework / datasets / trainer) |
| `examples/simBenchmarks/LIBERO/train_files/run_libero_train.sh` | the LIBERO launch script (CLI overrides) |
| `examples/simBenchmarks/LIBERO/train_files/data_registry/data_config.py` | `Libero4in1DataConfig` (12-60): views, action keys, chunk indices, min_max |
| `examples/simBenchmarks/LIBERO/train_files/modality.json` | LIBERO modality slices (no `absolute` key → default `True`) |
| `starVLA/config/deepseeds/deepspeed_zero2.yaml`, `ds_config.yaml` | accelerate + DeepSpeed ZeRO-2 bf16 config |
| `starVLA/training/train_starvla.py` | trainer: Accelerator (50-51), optimiser / scheduler (101-127), seed (144-145), freeze (155-160), train step (415-436), main (472-528) |
| `starVLA/training/trainer_utils/trainer_tools.py` | `normalize_dotlist_args` (31-54), `build_param_lr_groups` (92-148), `freeze_backbones` (192-) |
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | LeRobot loader: all steps are chunk starts (1081), padding (1544-1590, 1762), sample packing (1378-1434) |
| `examples/simBenchmarks/LIBERO/README.md` | published numbers (27-39), legacy PI forward note (78-90) |

## 2. Hyperparameters

"Effective" = the value after the fork's own config path: yaml + CLI → `apply_config_compat` (train_starvla.py:516) → `merge_framework_config(defaults, cfg)` (YAML wins) → runtime overrides in `__init__`. The effective values were verified by building both heads from the LIBERO yaml in the container (§5).

### 2a. QwenGR00T

| Item | Effective value | Source | LIBERO yaml / run script override? |
|---|---|---|---|
| DiT type | `DiT-B` → input_embedding_dim 768, head_dim 64, heads 12 (inner 768) | QwenGR00T.py:81; GR00T_ActionHeader.py:190-193, 217-223 | no |
| DiT depth | 16 blocks | QwenGR00T.py:115 | no |
| interleave_self_attention | True → even blocks cross-attention (8), odd blocks self-attention (8) | QwenGR00T.py:113; cross_attention_dit.py:242-243, 299-318 | no (DiT's own default is False, cross_attention_dit.py:215) |
| cross_attention_dim | 2560 (= Qwen3-VL text hidden) | set at runtime, QwenGR00T.py:155-157, via QWen3.py:74 | yaml `vl_hidden_dim: 2048` (yaml:14) is overwritten at runtime |
| DiT output_dim | 1024 | QwenGR00T.py:116 | no |
| norm / dropout | `ada_norm`; dropout 0.2; final_dropout True; positional_embeddings None | QwenGR00T.py:111-117 | no |
| other DiT args | attention_bias True, activation `gelu-approximate`, norm_elementwise_affine False, norm_eps 1e-5 | DiT defaults, cross_attention_dit.py:200-206 | no |
| MLP hidden (state enc / action dec) | 1024 | QwenGR00T.py:84; GR00T_ActionHeader.py:255-275 | no |
| future tokens | 32 learned (`num_target_vision_tokens`) | QwenGR00T.py:106; GR00T_ActionHeader.py:282-283 | no |
| positional embedding | learned, `max_seq_len` 1024, added to action tokens only | QwenGR00T.py:86-87; GR00T_ActionHeader.py:290-292, 337-340 | no |
| action_dim | 7 | QwenGR00T.py:89 | yaml:16 (same value) |
| action horizon | 8 | default 8 (QwenGR00T.py:95) | yaml:18 = 8. Loader returns 8 steps t..t+7 (data_config.py:39); `actions[:, -8:]` takes all 8 (QwenGR00T.py:197). **= our CHUNK_LEN 8.** |
| state | **not used.** `state_dim` 7 builds a state encoder (QwenGR00T.py:91; GR00T_ActionHeader.py:257-265), but the LIBERO loader packs no `state` (`include_state` not set, datasets.py:1420), so `state=None` and the DiT sequence is [32 future tokens, 8 action tokens] = 40 (GR00T_ActionHeader.py:343-348) | — | yaml:17 `state_dim: 7` (same) |
| time sampling | `s ~ Beta(1.5, 1.0)`, `t = (0.999 − min(s, 0.999)) / 0.999` ∈ [0, 1] | QwenGR00T.py:98-101; GR00T_ActionHeader.py:301, 305-307 | no |
| timestep buckets | `t_disc = (t·1000).long()` | QwenGR00T.py:102; GR00T_ActionHeader.py:330 | no |
| inference steps | 4 Euler steps from t=0 (noise), dt 0.25, buckets 0/250/500/750 | QwenGR00T.py:104; GR00T_ActionHeader.py:381-420 | no |
| repeated_diffusion_steps | **8** (batch, hidden states and mask repeated 8×, each copy with its own noise and t) | default QwenGR00T.py:97; read at 199-203 (fallback 4 never triggers) | no |
| loss | MSE between the predicted velocity and `actions − noise`, averaged over (B·8, 8, 7) | GR00T_ActionHeader.py:326-327, 362 | — |

### 2b. QwenPI

| Item | Effective value | Source | Override? |
|---|---|---|---|
| DiT width | 2560 (= LLM hidden, "no compression") | QwenPI.py:147, 153-157; share_tools.py:291-294 | yaml `vl_hidden_dim` overwritten (QwenPI.py:148) |
| DiT depth | 36 (= number of LLM layers) | QwenPI.py:145-146, 156 | no |
| heads | 40 × 64 | QwenPI.py:101; share_tools.py:289, 294 | no |
| cross_attention_dim | 2560 | share_tools.py:293 | no |
| interleave_self_attention | True (canonical) → 18 cross / 18 self blocks. **The published checkpoint used the legacy all-cross forward; see Q1.** | QwenPI.py:98; cross_attention_dit.py:223-225, 299 | no |
| norm / dropout | `ada_norm`, dropout 0.2, final_dropout True, positional_embeddings None | QwenPI.py:96-100 | no |
| DiT output_dim | 26 (DiT default, cross_attention_dit.py:197). **Unused:** PI calls the DiT with `return_pre_output=True` (LayerwiseFM_ActionHeader.py:338), so `norm_out`/`proj_out_1`/`proj_out_2` (13.18M parameters) get no gradient | — | — |
| action decoder | MLP 2560 → 1024 → 7 | LayerwiseFM_ActionHeader.py:265-269 | no |
| state encoder | MLP 7 → 1024 → 2560 (built, unused on LIBERO) | LayerwiseFM_ActionHeader.py:51-52, 252-259 | yaml:17 |
| future tokens / pos. emb. | 32; learned pos. emb. 1024 × 2560, added to action tokens | QwenPI.py:85-87; LayerwiseFM_ActionHeader.py:270-275, 319-322 | no |
| action horizon | 8 (**code default 16**, QwenPI.py:80) | yaml:18 overrides to 8 | **yes** |
| time sampling | `s ~ Beta(1.5, 1.0)`, `t = (0.999 − s) / 0.999`, **no clamp** → t ∈ (−0.001, 1]; P(t<0) = 1 − 0.999^1.5 ≈ 0.15 %; then `(t·1000).long()` truncates to bucket 0 | LayerwiseFM_ActionHeader.py:277, 281-283, 312 | no |
| inference steps | 4, Euler, same scheme as GR00T | QwenPI.py:84; LayerwiseFM_ActionHeader.py:365-407 | no |
| repeated_diffusion_steps | **2, hard-coded**: QwenPI.py:219-223 read the config value, then line 224 overwrites it with `repeated_diffusion_steps = 2  # NO repeat for big action FM`. The config value (default 2, QwenPI.py:82) never matters | QwenPI.py:224 | — |
| loss | MSE on velocity, averaged over (B·2, 8, 7) | LayerwiseFM_ActionHeader.py:308-309, 346 | — |

### 2c. Shared (both heads; trainer and data)

| Item | Effective value | Source |
|---|---|---|
| LR groups | `qwen_vl_interface` 1e-5, `action_model` 1e-4; `base` 2.5e-5 gets no parameters (both frameworks have only these two top-level modules) | yaml:50-53; trainer_tools.py:105-146 |
| schedule | `cosine_with_min_lr`, min_lr 1e-6 → HF uses `min_lr / optimizer.defaults["lr"]` = 1e-6 / 2.5e-5 = **0.04 of each group's peak**; warmup 5000 | yaml:47, 54-56; train_starvla.py:117-125; transformers `get_cosine_with_min_lr_schedule_with_warmup` |
| scheduler stepping | only when gradients sync | train_starvla.py:431-436 |
| optimiser | AdamW fused, betas (0.9, 0.95), eps 1e-8, weight_decay 1e-8 (`trainer.weight_decay: 0.0` at yaml:62 is unused) | yaml:68-72; train_starvla.py:104-111 |
| grad clipping | 1.0 (accelerator, train_starvla.py:427-428) and DeepSpeed `gradient_clipping 1.0` (ds_config.yaml:21) | yaml:64 |
| freeze | none in effect (see §6) | |
| gradient checkpointing | `gradient_checkpointing: true` (yaml:66) is **not read** by QwenGR00T, QwenPI or QWen3.py (grep: only MiniCPM / Gemma4 / Molmo2 read it) → off | |
| precision | VLM loaded bf16 (QWen3.py:63); DeepSpeed bf16 (ds_config.yaml:5-7); whole forward under `autocast(bf16)` (train_starvla.py:420); heads under `autocast("cuda", dtype=float32)` (QwenGR00T.py:193, QwenPI.py:212) — see §4 for what that does | |
| attention | `sdpa` forced, the yaml's `flash_attention_2` is ignored | QWen3.py:52-53 |
| prompt | one user turn [img][img][text], text = `CoT_prompt` with `{instruction}` replaced, `add_generation_prompt=True`, left padding | yaml:40; QWen3.py:123-144, 67 |
| seed | `set_seed(42 + rank)` | yaml:3; train_starvla.py:144-145 |

## 3. Which VLM hidden states each head reads

Background on transformers 4.57 (checked in the container): `output_hidden_states` for Qwen3-VL is collected by hooks on `Qwen3VLTextDecoderLayer` (`check_model_inputs`). The tuple has 37 entries:
- `[0]` = input to layer 0 (the embeddings);
- `[k]` = raw output of decoder layer k−1, for k = 1..35;
- `[36]` = the **final RMS-normed** `last_hidden_state`, which replaces layer 35's raw output.

The hooks fire on the layer's return value. DeepStack visual features are added *after* the layer (outside the hook) for layers 0-2. So entries 1-3 do not contain the DeepStack addition, even though the next layer's input does.

**GR00T** (QwenGR00T.py:180-219):
- **Which states:** `hidden_states[-1]`, the final normed output, shape [B, L, 2560], bf16. All L tokens are used: chat template, both images' tokens (2 × 64 `image_pad` + vision start/end), text and generation prompt, plus left padding.
- **Mask:** the processor's `attention_mask`, repeated 8× and cast to bool (lines 206-209). It is passed as `encoder_attention_mask` to the 8 **cross-attention** blocks (even indices 0, 2, …, 14), where `BasicTransformerBlock` hands it to `attn1` (cross_attention_dit.py:166-170). diffusers `AttnProcessor2_0` reshapes it to [B, heads, 1, L] and uses it as an SDPA boolean mask (True = attend). Checked: values at masked key positions do not change the output.
- **Self-attention blocks:** the 8 odd-index blocks attend over the 40 DiT tokens with no mask.
- **Entry into the DiT:** the same VLM states go to every cross-attention block. diffusers `Attention(cross_attention_dim=2560)` projects them K/V 2560 → 768. Queries are the DiT tokens [32 future + 8 action]. The timestep enters through AdaLayerNorm in every block (`temb`) and through the output modulation (cross_attention_dit.py:327-329).

**PI** (QwenPI.py:167-184, 208):
- **Which states:** `hidden_states[-36:]` = tuple entries 1..36, a list of 36 tensors [B, L, 2560], bf16. All tokens; same mask as GR00T (repeated 2×).
- **Layer per block:** DiT block i receives `list[i]` (cross_attention_dit.py:308-311), but only the cross-attention blocks use it (even i). The tensors actually used are therefore entries 1, 3, …, 35, i.e. the raw outputs of decoder layers **0, 2, 4, …, 34** (0-based). The final normed output `list[35]` goes to block 35, a self-attention block, so it is **unused** in the canonical forward.
- **Entry into the DiT:** K/V projections 2560 → 2560.
- **Legacy all-cross forward** (`use_canonical_forward=False`, the published checkpoint): every block i cross-attends to `list[i]`. That means layers 0..34 raw plus the final normed output. The 18 odd blocks are built without a `cross_attention_dim`, so their K/V are 2560 → 2560 from `query_dim`. This works only because the DiT width equals the VLM width.

## 4. Randomness, and how to make an oftvla-vs-fork test bitwise

| Source | Where | Generator | Notes |
|---|---|---|---|
| noise ε | `torch.randn(actions.shape, dtype=actions.dtype, device=cuda)` GR00T_ActionHeader.py:322, LayerwiseFM_ActionHeader.py:304 | CUDA default | drawn for the **repeated** batch (B·R, 8, 7); dtype = hidden-state dtype (bf16 in the original) |
| t | `Beta(1.5, 1.0).sample([B·R])` GR00T_ActionHeader.py:306, LayerwiseFM_ActionHeader.py:282 | **CPU default** (the Beta parameters are CPU tensors: `Beta` is not a module and never moves to CUDA; checked) | drawn after ε; `sample()` takes no generator argument; `.to(device, dtype)` → bf16 |
| t bucketing | `(t·1000).long()` with t in bf16 | — | bf16 has ~3 significant digits: near t=1 the reachable buckets are ~4 apart. This is a property of the original numerics |
| dropout | 4 sites per DiT block, p = 0.2: `attn1.to_out.1`, `ff.net.1`, `ff.net.3`, block `final_dropout` (the attention output is dropped twice) | CUDA default | 16 blocks (GR00T) / 36 (PI). The VLM has attention_dropout 0.0. The fork never calls `.eval()` during training, so the periodic `mse_score` eval (train_starvla.py:386-394) also runs with dropout **on** |
| repeated steps | `.repeat(R, 1, 1)`: copy r of sample b is row r·B + b | — | GR00T R=8, PI R=2; independence comes only from ε and t |
| inference noise | `torch.randn(B, 8, 7, dtype=vl_embs.dtype)` GR00T_ActionHeader.py:375, LayerwiseFM_ActionHeader.py:359 | CUDA default | one sample per call, no averaging |
| data order / seed | `set_seed(42 + rank)`, LeRobot sampler | — | not part of the model test |

How autocast behaves: `torch.autocast("cuda", dtype=torch.float32)` is legal in torch 2.6 (no "unsupported dtype" fallback for CUDA, autocast_mode.py checked). It runs autocast-eligible ops (linear, matmul, SDPA) in fp32. **Elementwise ops are not affected**: `noisy = (1−t)·ε + t·a`, `velocity = a − ε` and `t·1000` stay in the input dtype, which is bf16 because `actions` is created with `dtype=last_hidden.dtype` (QwenGR00T.py:194-196). Our port must reproduce this dtype path. That means bf16 hidden states (bf16 VLM weights), not the fp32 weights + autocast of the current oftvla trainer.

Plan for a bitwise test (same structure as M2: identical weights, same batch, bf16 weights in both):
1. **Deterministic path first.** Give the oftvla heads optional `noise=`, `t=` arguments (`None` = sample exactly like the fork, in the same order: ε, then t). On the fork side, wrap the call to capture or override ε and t: set `head.sample_time` to a function returning a fixed tensor, and run under `torch.random.fork_rng(devices=[cuda])` with `torch.manual_seed` / `torch.cuda.manual_seed`. Run both heads with `model.eval()` (dropout off) and compare loss and `predict_action` (with injected initial noise). Expected: max|diff| = 0, as in M2.
2. **Stochastic path.** In train mode, seed the CPU and CUDA RNGs identically right before each forward. If the ported head issues the same RNG calls in the same order (ε on CUDA, Beta on CPU, then the dropout masks block by block), the loss is bitwise equal. This also tests the RNG call order.
3. **Negative controls (M2b lesson).** Change R (8 → 4), remove one dropout site, use `hidden_states[-2]` for GR00T, and use interleave=False for PI. Each must make the test fail.
4. Compare gradients only with a tolerance: SDPA / flash backward kernels are not deterministic.

## 5. Parameter counts (counted, not estimated)

Built in the container on `meta` with the fork's own classes and config path (LIBERO yaml → `apply_config_compat` → framework defaults → runtime dims; `Distribution.set_default_validate_args(False)` was needed because `Beta` validation calls `.item()`). The script is in the session scratchpad, not in the repo.

| | GR00T head | PI head |
|---|---|---|
| **total** (`sum(p.numel())`) | **161,472,775** | **3,352,072,737** |
| DiT (`model`) | 157,032,448 | 3,324,421,146 (13,178,906 of it in the unused output layers) |
| action_encoder | 1,777,152 | 19,686,400 |
| action_decoder | 1,056,775 | 2,629,639 |
| state_encoder (no gradient on LIBERO) | 795,392 | 2,632,192 |
| position_embedding | 786,432 | 2,621,440 |
| future_tokens | 24,576 | 81,920 |
| one DiT block | — | 91,778,560 (attn 26.2M, FF 52.4M, AdaLN 13.1M) |
| **full model with VLM (4,437,815,808)** | **4,599,288,583** | **7,789,888,545** |

Your estimates (~160M, ~3.35B) are confirmed.

## 6. The original StarVLA LIBERO recipe for these heads

GR00T and PI use the **same** yaml and run script; only `--framework.name` differs (run_libero_train.sh:13, 40). Effective recipe:

| Item | Value | Source |
|---|---|---|
| GPUs | **= number of visible GPUs**: `num_processes=${NUM_PROCESSES:-$(nvidia-smi -L \| wc -l)}`. The accelerate config default is 8 processes on 1 machine. The docs name no GPU count for training (README:9 mentions A100 / RTX 4090 for *evaluation*) | run_libero_train.sh:33-37; deepspeed_zero2.yaml:9-10 |
| per-GPU batch | 16 | yaml:41; run_libero_train.sh:44 |
| gradient accumulation | **1**. `Accelerator()` is built without an accumulation argument (train_starvla.py:51); the value comes from the DeepSpeed config, `gradient_accumulation_steps: 1` (ds_config.yaml:10). `trainer.gradient_accumulation_steps: 4` (yaml:65) is never read (grep) | |
| global batch | 16 × #GPUs × 1 = **128 with 8 GPUs** | train_starvla.py:172-178 |
| DeepSpeed | ZeRO stage 2, bf16 on, fp16 off, allgather/reduce buckets 5e8, overlap_comm, contiguous_gradients, no CPU offload, gradient_clipping 1.0 | ds_config.yaml:1-24; deepspeed_zero2.yaml:3-6 |
| precision | bf16 weights (DeepSpeed) with fp32 master copy in the optimiser; bf16 autocast for the step; heads under fp32 autocast (fp32 matmuls, bf16 elementwise) | §2c, §4 |
| LR | VLM 1e-5, head 1e-4; cosine_with_min_lr to 0.04 × peak; warmup 5000 | yaml:47, 50-56 |
| steps | **ambiguous**: yaml 100000 (yaml:46), run script 80000 (run_libero_train.sh:47), README table 30K for every published Qwen3-VL result (README:27-37) | Q2 |
| freeze | **none in effect.** yaml:57 says `'qwen_vl_interface'`, but the run script sets `freeze_module_list=''` (line 14) and passes `--trainer.freeze_modules ${freeze_module_list}` (line 46). With the empty value, `normalize_dotlist_args` turns the flag into `freeze_modules=true`, a boolean (trainer_tools.py:47-51). `freeze_backbones` acts only on strings (trainer_tools.py:211) and `build_param_lr_groups` resets non-strings to "" (109-110) → full fine-tuning | Q5 |
| gradient checkpointing | off (yaml flag unread) | §2c |
| optimiser | AdamW fused (0.9, 0.95), eps 1e-8, wd 1e-8; grad clip 1.0 | yaml:61-72 |
| data | LeRobot `libero_all` (4 suites, 40 tasks), 2 views, 8-step chunks, min_max, no state, no augmentation | yaml:35-43; data_registry/data_config.py:12-60, 119-123 |
| base VLM | yaml: `Qwen3-VL-4B-Instruct` (yaml:12); run script: `Qwen3.5-0.8B` (run_libero_train.sh:15); README rows say "Qwen3-VL" without a size | Q3 |
| prompt | `CoT_prompt` wrapper (yaml:40) | §2c |
| seed | 42 + rank | yaml:3; train_starvla.py:144 |
| logging / eval / save | logging every 100, `mse_score` eval every 100 (train mode), weights every 10000 | run_libero_train.sh:48-50 |
| not used by `train_starvla.py` | `datasets.vlm_data`, `trainer.loss_scale` (VLM co-training belongs to `train_starvla_cotrain.py`), `trainer.max_grad_norm`, `trainer.weight_decay`, `--trainer.vla_data.video_backend` (a misplaced key; `datasets.vla_data.video_backend` comes from yaml:43) | |

Not specified anywhere in the fork: the number of GPUs or the GPU type for the published runs, the base-VLM size behind the README rows, and which step count and freeze setting produced the README numbers.

## 7. Memory estimate: PI full fine-tuning (7.79B parameters, AdamW)

Calibration from our own runs: e3v5_A0 (OFT, 4.46B, fp32 weights + bf16 autocast, micro-batch 16, ~177 tokens, no checkpointing) peaks at 105.9 GB. Model states are 16 B/param ≈ 71 GB and the autocast bf16 weight cache 8.9 GB, so **VLM activations and workspace ≈ 26 GB** at micro-batch 16. AS/MAS prompts (225-236 tokens) add ~9-10 GB. The CoT wrapper adds ~30 tokens and the OFT suffix (~20 tokens) is dropped, so net ≈ +10 tokens.

PI head activations at micro-batch 16, R=2 (32 rows × 40 tokens × 2560, 36 blocks, fp32 matmul outputs under fp32 autocast, ~22 saved T×D tensors per block, plus K/V and the fp32 copies of the 2560-dim VLM states in the 18 cross blocks, plus the repeated bf16 hidden-state list): **≈ 15 GB** (rough; the dropout masks are small).

| Setup | Model states per GPU | + activations | Total | Fits? |
|---|---|---|---|---|
| **Original: ZeRO-2 bf16, 8 GPUs, 16/GPU** | bf16 params 15.6 + fp32 grad partition 31.2/8 + fp32 master 31.2/8 + Adam 62.3/8 + buckets ~2 ≈ **33 GB** | ~41 GB | **≈ 74 GB** | yes on H200; borderline on 80 GB cards |
| ZeRO-2 bf16, 4 × H200 | 15.6 + 124.7/4 + 2 ≈ 49 GB | ~41 | ≈ 90 GB | yes |
| ZeRO-2 bf16, 2 × H200 | 15.6 + 124.7/2 + 2 ≈ 80 GB | ~41 | ≈ 121 GB | yes (tight) |
| ZeRO-2 bf16, **1 × H200** | 15.6 + 124.7 ≈ 140 GB | ~41 | ≈ 181 GB | **no**, even with zero activations |
| current oftvla style (fp32 weights + grads + Adam, autocast), 1 × H200 | 16 B × 7.79B = 124.6 + bf16 cast cache 15.6 | ~41 | ≈ 181 GB | **no** |

For comparison, **GR00T on 1 H200, current oftvla style**: 16 B × 4.60B = 73.6 + cast cache 8.9 + VLM ~26 + GR00T head activations ~11 GB (R=8 makes the DiT batch 128, plus fp32 copies of the 2560-dim states for 8 cross blocks) ≈ **120 GB → fits**.

These are estimates (±10-15 GB). The real number must come from a smoke test with the exact run configuration (M4 lesson).

## 8. What must change in oftvla, and what can be shared

**Can stay shared, unchanged:** `data.py` (sample = 2 PIL views at 224, `lang`, `action` [8, 7] normalised float16, the same min_max stats; per-frame `text_csv` / setup), the E3 split and forbidden-ids guard, `heldout_analysis.py`, `submit_chain.sh`, the run-dir / DONE / resume logic, and `policy.py` / `libero_client.py` apart from model selection.

**`model.py`:**
- Factor out a shared VLM wrapper: loading (sdpa, bf16), processor with left padding, `build_inputs(images, texts)` with a configurable prompt builder. OFT appends its query suffix; GR00T/PI take the raw text, wrapped by `CoT_prompt` if that is adopted (Q6).
- Port, as close to line-for-line as possible:
  - `DiT`, `BasicTransformerBlock`, `AdaLayerNorm`, `TimestepEncoder` (using diffusers `Attention` / `FeedForward` / `Timesteps` / `TimestepEmbedding`, as the original does; diffusers 0.39.0 is in the container);
  - both `ActionEncoder`s (names `layer1/2/3`) and both `MLP`s (PI's has defaults 1024/2048);
  - the two heads with **identical parameter names**, so that `load_starvla_state_dict` works with the prefix map `qwen_vl_interface.model.` → `vlm.`, `action_model.` → `action_head.`;
  - the unused `state_encoder` (and PI's unused DiT output layers), so that state_dicts match.
- `QwenGR00T` / `QwenPI` forward:
  - GR00T uses `hidden_states[-1]`; PI uses `hidden_states[-36:]`.
  - The bool mask is repeated R times; R = 8 for GR00T and **hard-coded 2** for PI.
  - `actions` are created in the hidden-state dtype; the head runs under `autocast(float32)`.
  - RNG call order: ε, then t on CPU. Optional `noise` / `t` arguments for tests.
  - Interleave mode is selectable for PI (Q1).
- `predict_action`: 4 Euler steps; optional generator / initial noise.

**`train.py`:**
- Model class selected by flag. LR groups `vlm` 1e-5 / `head` 1e-4 (same mapping as the fork).
- The loss is MSE (log it under its own name; the `mse_score` eval becomes stochastic).
- **Multi-GPU** for PI (and for the original 8 × 16 recipe for both): accelerate + DeepSpeed ZeRO-2 bf16 with the fork's `ds_config`, `set_seed(seed + rank)`, per-rank data sharding.
- If the original numerics are to be followed, also bf16 weights (DeepSpeed), not fp32 + autocast (D2).
- Resume must save and restore the DeepSpeed state.

**`heldout_eval.py`:**
- `predict_action` is stochastic. Fix a per-frame seed (or pass explicit initial noise) and record it, so that results reproduce and runs can be compared on the same noise. Report one sample (as the original does); optionally also a mean over k samples as a separate metric.
- Use `model.eval()` (dropout off).
- PI needs a smaller batch (7.8B parameters in bf16 = 15.6 GB; the DiT runs 4 times per prediction).

**Tests:**
- Port the M1-style unit tests (time sampling incl. PI's no-clamp case, R, sequence layout, which layers PI reads, the mask).
- An equivalence script like `check_equivalence.py` per head (§4), plus negative controls.

## 9. Open questions (not resolvable from the code)

- **Q1. PI forward: canonical or legacy?** The code at 985c223 defaults to interleaved self-attention (since c521dec, 2026-06-16, "the state token could never influence action predictions"). The README says the released `StarVLA/Qwen3-VL-PI-LIBERO-4in1` checkpoint, i.e. the 95.7 % row, was trained with the **legacy all-cross** forward (README:80-90; commit 631aae0). GR00T is not affected: its DiT always went through `DiT.forward`, and the legacy path would fail with 768 ≠ 2560. Which counts as "original" for PI: the code or the published number?
- **Q2. Step count:** 30K (README table, used for all 3 heads, = our OFT runs) vs 80K (run script) vs 100K (yaml).
- **Q3. Base VLM** of the published rows: the yaml says Qwen3-VL-4B-Instruct, the run script Qwen3.5-0.8B (template; `run_id` says "qwen3oft"), the README just "Qwen3-VL".
- **Q4. GPU count:** not documented. "9.54 epochs at 30K steps" (README) does not match 8 × 16 = 128 samples/step on the 4 no-noops LeRobot suites (273,465 frames according to the dataset cards, not verified locally): that would be 14.0 epochs. 9.54 implies ≈ 87 samples/step, or ≈ 402K frames at 128/step. Unresolved.
- **Q5. Freezing:** the yaml freezes the VLM, but the shipped run script fine-tunes everything (§6). Which produced the README numbers is unknown. Our OFT runs fine-tune everything.
- **Q6. CoT_prompt:**
  - Was it active in the published runs? It is in the shipped yaml.
  - If we adopt it: wrap the whole per-frame text (`Your task is Instruction: …\nCurrent Goal: …`) or only the instruction line? This is a design decision of our experiment.
  - It would also change the OFT baseline (D3).
- **Q7. Numerics:** reproducing the original exactly means bf16 hidden states / weights (DeepSpeed bf16), so that ε, t, bucketing and velocity are in bf16. Do we switch to DeepSpeed (D2), or keep fp32 weights and cast the head inputs to bf16 explicitly? The second is not "the same code path".
- **Q8. Bug-compatibility:** keep PI's unclamped t (bucket 0 for slightly negative t), the double dropout on the attention output, the unused state encoder / output layers, and the dropout-on `mse_score` eval? Proposal: yes for everything that affects training, recorded as such.
- **Q9. Held-out metric for flow heads:** one noise sample per frame (original inference) vs averaging k samples. One sample is the original; averaging would be our addition. The seed policy must be fixed before any run.
- **Q10. Unused parameters under ZeRO-2 AdamW:**
  - The state encoder and PI's output layers get zero gradients under DeepSpeed, so only weight decay (1e-8) applies, which is negligible.
  - Plain torch AdamW skips them entirely.
  - Bitwise identity of the optimiser state therefore depends on the optimiser path.

## 10. Deviations

Categories: **identical** / **infra-forced** / **experiment-intrinsic** / **UNJUSTIFIED**. "Current oftvla pipeline" = what our OFT runs (e3_A0, e3v5_*) use today. The flow heads do not exist yet; rows H1-H8 are requirements for the port.

| # | Item | Original (file:line) | Current oftvla pipeline | Category |
|---|---|---|---|---|
| D1 | GPUs / batch per step | 8 processes × 16, accumulation 1 → 128 (deepspeed_zero2.yaml:10; ds_config.yaml:10; run_libero_train.sh:33, 44) | 1 H200 × 16 × 8 accumulation → 128 (train.py:9, 44-45; MILESTONES M4b) | **infra-forced** for the past OFT runs (1 GPU per job then). For new runs, now that several GPUs are available: **UNJUSTIFIED if kept**. Also unavoidable for PI on memory grounds (§7) |
| D2 | Weight precision / optimiser path | DeepSpeed ZeRO-2 bf16 weights, fp32 master (ds_config.yaml:5-7, 13-21) | fp32 weights + bf16 autocast, plain torch AdamW (train.py:9, 125, 131) | **UNJUSTIFIED**: M4b gives "no DeepSpeed, for readable code"; DeepSpeed 0.16.9 imports in starvla.sif. For the flow heads it changes ε / t / velocity from bf16 to fp32 |
| D3 | Prompt wrapper `CoT_prompt` | "Your task is {instruction}. To identify the key objects for your task. Locate their bounding boxes in [x1,y1,x2,y2] format." (yaml:40; QWen3.py:126-128) | not applied: `{text} Please predict …` (model.py:123-126; check_equivalence.py:44 builds the fork without it) | **UNJUSTIFIED** (not recorded in MILESTONES). Effect on OFT: the 🔍 query states see only the prefix "Your task is " (causal attention; the tail comes after the queries). For GR00T/PI the whole wrapper is cross-attended |
| D4 | Training steps | 30K (README:27-37) / 80K (run_libero_train.sh:47) / 100K (yaml:46) | 30K | **identical** to the published setting (Q2) |
| D5 | LR, groups, schedule, warmup | 1e-5 / 1e-4, cosine_with_min_lr → 0.04, warmup 5K (yaml:47, 50-56) | same (train.py:47-50, 128-133) | identical |
| D6 | Optimiser / clipping | AdamW (0.9, 0.95), eps 1e-8, wd 1e-8, clip 1.0 (yaml:61-72) | same (train.py:51-52, 131, 169) | identical |
| D7 | Freeze | none in effect (run_libero_train.sh:14, 46; trainer_tools.py:47-51, 211) | full fine-tuning | identical (Q5) |
| D8 | Gradient checkpointing | flag set (yaml:66) but unread → off | off | identical |
| D9 | Seed / data order | 42 + rank, LeRobot sampler (train_starvla.py:144) | 42, torch generator with reshuffle per pass (train.py:118-143) | **experiment-intrinsic** (different data source and loader; order cannot match) |
| D10 | Data source / tasks | LeRobot `libero_all`, 40 tasks (yaml:36-37; data_config.py:119-123) | regenerated HDF5, E3 split, 76 training tasks, v5 demos | experiment-intrinsic |
| D11 | Text input | LIBERO task instruction (data_config.py language key) | per-frame `text_A0` / `text_AS` / `text_MAS` | experiment-intrinsic |
| D12 | Chunk starts / padding | every step is a start (datasets.py:1081); end padding `first_last` because LIBERO actions count as absolute (datasets.py:1762; schema.py:68 default True; modality.json has no `absolute`) | every step, last action repeated (data.py:83-86) | identical |
| D13 | Action normalisation | min_max on x..yaw, gripper raw (data_config.py:50-60). (The q01/q99 helper `tools.py:246-275` is legacy; at 985c223 the server inverts through the data-config transform, `policy_norm_processor.py:360`) | min_max on arm dims, gripper raw; inverse min_max in the policy (data.py:88-108) | identical |
| D14 | Image size / resize | 2 views (primary, wrist), PIL default resize to 224 (data_config.py:14-17; datasets.py:1392-1404) | agentview + wrist, PIL resize 256 → 224 (data.py:223) | identical |
| D15 | Inference image resize | no `obs_image_size` in the LIBERO yaml → model does not resize (QwenGR00T.py:245-247); client resizes to 224 (model2libero_interface.py:41) | client 224 + model resize (224, 224) (model.py:38), a no-op | identical in effect |
| D16 | State / proprio | not loaded (`include_state` unset, datasets.py:1420) | none | identical |
| D17 | Augmentation | none (data_config.py:50-60 has no video transforms) | none | identical |
| D18 | Action dtype | float16 (datasets.py:1381) | float16 (M3) | identical |
| D19 | Attention implementation | sdpa forced (QWen3.py:53) | sdpa (model.py:33) | identical |
| D20 | Base VLM | Qwen3-VL-4B-Instruct (yaml:12) vs Qwen3.5-0.8B (run script:15) | Qwen3-VL-4B-Instruct | identical to the yaml (Q3) |
| D21 | Epochs seen | 9.54 at 30K (README) | 30K × 128 / 565,397 ≈ 6.8 | experiment-intrinsic (dataset size) |
| D22 | Periodic eval | `mse_score` every 100 steps, model stays in train mode (train_starvla.py:386-394) | same | identical |
| D23 | Checkpointing | weights every 10K (run_libero_train.sh:48) | same + resume state, 23 h stop | infra-forced (24 h job limit) |
| H1 | GR00T: VLM layer | `hidden_states[-1]` (QwenGR00T.py:190) | — | must be identical |
| H2 | PI: VLM layers | `hidden_states[-36:]`, block i ↔ list[i] (QwenPI.py:183; cross_attention_dit.py:308-311) | — | must be identical; mode per Q1 |
| H3 | Repeats | GR00T 8 (QwenGR00T.py:97, 199-205); PI 2 hard-coded (QwenPI.py:224) | — | must be identical |
| H4 | t sampling | GR00T clamped (GR00T_ActionHeader.py:305-307); PI unclamped (LayerwiseFM_ActionHeader.py:281-283) | — | must be identical (Q8) |
| H5 | Dropout | 0.2 at 4 sites per block (QwenGR00T.py:111-112; QwenPI.py:96-97) | — | must be identical |
| H6 | Mask | processor `attention_mask` as bool key mask in cross blocks (QwenGR00T.py:206-209; QwenPI.py:228-231) | — | must be identical |
| H7 | Head numerics | elementwise in hidden-state dtype (bf16), matmuls fp32 (QwenGR00T.py:193-196; QwenPI.py:212-216) | — | identical only if D2 is resolved (Q7) |
| H8 | Inference | 4 Euler steps from noise, one sample (GR00T_ActionHeader.py:365-421; LayerwiseFM_ActionHeader.py:349-408) | — | must be identical; seed policy per Q9 |

Nothing in this table has been fixed; it is a report. D2 and D3 are the two UNJUSTIFIED items that already affect the finished OFT runs (M4b-M10). D1 becomes one for new runs if multiple GPUs are used for the flow heads but not for OFT.

---

## 11. Decisions (2026-11-05)

| # | Topic | Decision | Status |
|---|---|---|---|
| 1 | PI forward | Use the older forward that trained the published StarVLA-π LIBERO checkpoint (every DiT block cross-attends), **only if** it can be recovered exactly from the fork's git history; no reconstruction by hand | **Condition not met as stated, so PENDING your decision.** The training-time code is pinned exactly in git, but the 985c223 "legacy" switch is **not** the same forward (§12.1) |
| 2 | Steps | 30K (README; same as our OFT runs) | decided. **But see §12.2: the README's 30K rows are the 30K checkpoints of 80K (OFT) / 100K (PI) cosine schedules, not 30K-step runs.** Re-confirmation needed |
| 3 | Base VLM | Default `Qwen3-VL-4B-Instruct`, swappable. Mirror the fork's registry (`get_vlm_model` by `base_vlm` name, `starVLA/model/modules/vlm/__init__.py`): one adapter per VLM family exposing (a) forward → all hidden states + attention mask; (b) `hidden_size`, `num_layers` read from the loaded config (no hard-coded 2560 / 36; PI's DiT depth = number of VLM layers follows from the config); (c) prompt building and special-token ids (action query tokens, image tokens, CoT_prompt). Only the Qwen3-VL adapter is built now; heads depend only on the adapter interface. `base_vlm` goes into args.json; run-directory names carry a short VLM tag, so runs with different backbones cannot be mixed in `heldout_analysis.py` (labels come from folder names) | decided; built in step 2 |
| 4 | Prompt wrapper | GR00T and PI use `CoT_prompt` exactly as the original applies it to the dataset language field. Our language field is the full per-frame text (`text_A0` / `text_AS` / `text_MAS`), so the whole text goes into `{instruction}`. OFT runs stay unchanged; the missing wrapper is recorded as a known OFT deviation (D3) | decided |
| 5 | Precision | DeepSpeed bf16 (the fork's `ds_config.yaml`, ZeRO-2) for GR00T and PI. OFT stays fp32 weights + bf16 autocast, recorded as a known OFT deviation (D2) | decided |
| 6 | GPUs | Target the original 8 GPUs × 16. Fallback, only if queueing makes this impractical: 1 GPU × 16 × 8 accumulation (GR00T only; PI needs ≥ 2 GPUs). All three text runs of a head use the same setup. (The released PI run used 16 GPUs × 8, also 128 per step; §12.2) | decided |
| 7 | Fork quirks | Keep all of them: PI's unclamped t, repeat factor 2 (PI) vs 8 (GR00T), double dropout on the attention output, unused state encoder / DiT output layers, dropout-on `mse_score` eval | decided |
| 8 | Held-out sampling policy | Decided later, in the eval step | open |

## 12. Evidence from the released checkpoints (Hugging Face, read 2026-11-05)

Sources (public metadata only; no weights downloaded):
- `StarVLA/Qwen3-VL-PI-LIBERO-4in1` @ a461ce2: `config.yaml`, `slurm_script`, `README.md`, `summary.jsonl`.
- `StarVLA/Qwen3-VL-OFT-LIBERO-4in1` @ 3917714: `config.yaml`, `run_libero_train.sh`, `README.md`, `summary.jsonl`.
- No Qwen3-VL GR00T LIBERO checkpoint is published (only `Qwen2.5-VL-GR00T-LIBERO-4in1`).

### 12.1 PI forward: what trained the published checkpoint, and is it recoverable?

**Which code.** The run is `1208_libero_all_QwenPI_qwen3` (config.yaml `run_id`; created on HF 2025-12-17). In the fork, the PI forward files `QwenPI.py`, `LayerwiseFM_ActionHeader.py` and `cross_attention_dit.py` were last changed by f4473e4 (2025-12-08 14:03, "auto map dim for qwen3vl"). They are byte-identical from f4473e4 through 09b5652 (2025-12-08 17:20) and up to 2025-12-18; no commit in that window touches them. `QWen3.py` changes only by the removal of `device_map="cuda"` in 09b5652. So, assuming the run used committed code, the training forward is **f4473e4 ≡ 09b5652** (`starVLA/model/framework/QwenPI.py`, `.../LayerwiseFM_ActionHeader.py`, `.../cross_attention_dit.py`, `.../vlm/QWen3.py`). That assumption cannot be checked: the config has `is_resume: True`, and local edits are invisible.

**That forward** (09b5652):
- `QwenPI.forward` (QwenPI.py:103-140): `vl_embs_list = hidden_states[-36:]` (36 hard-coded, = Qwen3-VL-4B); R = 2 hard-coded (line 128); `self.action_model(vl_embs_list_repeated, actions_target_repeated, state_repeated)`, **no attention mask**.
- Head (LayerwiseFM_ActionHeader.py:272-324): hand-written loop `for layer_idx, layer in enumerate(self.model.transformer_blocks): layer(hidden_states=…, encoder_hidden_states=vl_embs_list[layer_idx], temb=temb)`. So every block gets its own layer's states, with no mask. The odd blocks are *built* as self-attention modules (`interleave_self_attention: true` in the run config), but they receive encoder states. diffusers then projects K/V from them with the query-dim projection: 2560 → 2560, the same shapes as a cross block. No `norm_out` / `proj_out` (decoder on the raw block output). `sample_time` without clamp. Inference loop is the same (lines 326-383), also without a mask.
- VLM (QWen3.py @09b5652:59): **`attn_implementation="flash_attention_2"` hard-coded**. `config.hidden_size` is patched (line 70). Same prompt / CoT_prompt code (lines 122-124).

**Diff against 985c223 run with the legacy switch** (`use_canonical_forward=False` → `interleave_self_attention=False`, cross_attention_dit.py:223-225):

| Aspect | Training forward (09b5652) | 985c223, legacy switch | Same? |
|---|---|---|---|
| block i ↔ `hidden_states[-36:][i]`, all 36 blocks cross-attend | yes (hand loop) | yes (`DiT.forward`, layer-wise list, cross_attention_dit.py:299-318) | yes |
| K/V projection shapes of the odd blocks | built with `cross_attention_dim=None` → `to_k/to_v: Linear(2560, 2560)` | built with `cross_attention_dim=2560` → `Linear(2560, 2560)` | yes (same shapes and math; `is_cross_attention` flag differs, `norm_cross` is None in both) |
| timestep embedding, AdaLN, decoder on pre-output, no `norm_out`/`proj_out` | yes | yes (`return_pre_output=True`) | yes |
| **padding mask in the DiT** | **none**: left-padding positions are attended in every block, in training and inference | **applied** (QwenPI.py:174, 228-231, 280-281; LayerwiseFM_ActionHeader.py:337, 399) | **NO** |
| **VLM attention kernel** | **flash_attention_2** | **sdpa forced** (QWen3.py:53) | **NO**: different kernels, and because the old head reads the padding positions, their hidden states (which depend on the kernel's padding handling) feed the head |
| `proj_out_2` (unused) | `Linear(2560, 1024)` (`output_dim` from the run yaml) | `Linear(2560, 26)` (DiT default) | no, but unused (affects only state_dict loading) |
| t sampling, R = 2, dropout 0.2 ×4, Euler 4 steps | same | same | yes |
| layer count | hard-coded 36 | from config (36 for 4B) | yes for Qwen3-VL-4B |
| transformers | 4.57.0 pinned (requirements.txt @09b5652) | 4.57.0 pinned (= our container) | yes |
| diffusers, flash-attn | **unpinned** (`diffusers` without version) | unpinned; container: diffusers 0.39.0, flash_attn 2.7.4.post1 | unknown |

**Result.** The training forward of the published checkpoint *is* identifiable in git history (f4473e4 ≡ 09b5652). But the 985c223 legacy switch does **not** reproduce it: it adds the padding mask in all 36 cross-attention blocks and runs the VLM with SDPA instead of FlashAttention-2. The diffusers / flash-attn versions of the original run are not recorded anywhere. Following decision 1 (no reconstruction by hand), PI is on hold until you decide. Options:
- (a) Port the f4473e4/09b5652 files verbatim (no mask, FA2). This is "taken from git", not hand-reconstructed, but it is a different code snapshot from the rest of our reference (985c223).
- (b) Use the 985c223 canonical forward (interleaved).
- (c) Use the 985c223 legacy switch (all-cross, with mask, SDPA). That is what the fork's maintainers use to *evaluate* the released checkpoint (README:80-90), but it is not what trained it.

### 12.2 Steps, schedule and GPUs of the published runs

| | PI (released) | OFT (released) | README table |
|---|---|---|---|
| run | `1208_libero_all_QwenPI_qwen3` | `1226_libero4in1_qwen3oft` | — |
| GPUs × per-GPU batch | **2 nodes × 8 GPUs × 8 = 128** (slurm_script `-N 2`, `--gres=gpu:8`, `per_device_batch_size 8`) | **8 × 16 = 128** (`--num_processes 8`, `per_device_batch_size 16`) | — |
| gradient accumulation | 1 | 1 | — |
| schedule length (`max_train_steps`) | **100,000** | **80,000** | "30K" |
| released weights | step 100,000 | step 50,000 | — |
| success at step 30,000 | 95.7 % (card: 0.958 / 0.996 / 0.988 / 0.884) | 96.55 % (card) | PI 95.7, OFT 96.6 |
| CoT_prompt | yes | yes | — |
| freeze | `true` → none (full FT) | `true` → none (full FT) | — |
| VLM | Qwen3-VL-4B-Instruct | Qwen3-VL-4B-Instruct | "Qwen3-VL" |
| LR (VLM / head) | 1e-5 / 1e-4 (base 3e-5, empty group) | 1e-5 / 1e-4 | — |

Consequences:
- **Q2 (steps) is answered differently from what decision 2 assumes.** The README's "30K" numbers are evaluations at step 30K of 80K / 100K cosine schedules. At step 30K the learning rate is still high (`cosine_with_min_lr`, 5K warmup: 0.04 + 0.96 · ½(1 + cos(π · progress)) ≈ 0.85 × peak for the 100K schedule, ≈ 0.76 × for 80K). Our OFT runs instead decay to 0.04 × peak at 30K. "30K steps" in the sense of a full 30K schedule is therefore **not** the published setting. That applies to our finished OFT runs too (D4 should read "differs", category UNJUSTIFIED unless re-decided).
- **Q3, Q5 answered:** Qwen3-VL-4B-Instruct; full fine-tuning (packaged `freeze_modules: true` selects nothing).
- **Q4 answered:** 128 per step in both runs. The "9.54 epochs" in the README remains unexplained.
- **Q6:** CoT_prompt was active in both published runs (decision 4 matches).
- The PI card says "one image_0 view". The run config does contain `obs: [image_0]`, but the 09b5652 `Libero4in1DataConfig` (`starVLA/dataloader/gr00t_lerobot/data_config.py:448-`) loads `video.primary_image` + `video.wrist_image` and does not read `obs`. So it is two views, as in our pipeline.
