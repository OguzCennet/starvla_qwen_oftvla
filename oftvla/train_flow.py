"""Multi-GPU training of QwenGR00T / QwenPI, following StarVLA's trainer (train_starvla.py @ 985c223).

Launch (see scripts/train_flow.sbatch): accelerate launch --config_file configs/deepspeed/deepspeed_zero2.yaml
  --num_machines M --num_processes 4M ... -m oftvla.train_flow --head gr00t|pi ...

Recipe (fork, file:line)                                        | here
----------------------------------------------------------------|-------------------------------------------
Accelerator(DeepSpeedPlugin()), accelerate launch, ZeRO-2 bf16  | same; configs/deepspeed/ copied verbatim
  (train_starvla.py:50-51; deepspeed_zero2.yaml; ds_config.yaml) |
global batch = per_device x GPUs x 1 (ds_config accumulation 1)| GR00T 16 x 8 GPUs, PI 8 x 16 GPUs = 128
param groups qwen_vl_interface 1e-5, action_model 1e-4,        | same groups (names kept), same order
  base 2.5e-5 (trainer_tools.py:92-148; yaml:50-53)             |
AdamW fused, betas (0.9,0.95), eps 1e-8, wd 1e-8 (:101-111)     | same
cosine_with_min_lr, min_lr 1e-6, warmup 5000 (:117-125)         | same; total steps GR00T 80K / PI 100K, stop 30K
scheduler stepped only on sync (:431-436)                       | same
grad clip 1.0: accelerator.clip_grad_norm_ (no-op under DS) +   | same calls; logged grad_norm = DeepSpeed's
  ds_config gradient_clipping 1.0                               |   global norm BEFORE clipping
forward under autocast(bf16) (:420)                             | same
no .train()/.eval() calls: VLM in HF eval mode, head train mode | same (dropout 0.2 active in the head)
eval every 100 steps: next batch, predict_action, mse_score     | same (batch consumed, model in train mode)
seed: set_seed(seed + rank) after building (:144-145)           | same; the model is built after
                                                                |   set_seed(seed) so the head init is identical
                                                                |   for A0/AS/MAS runs (fork: unseeded)
weights every 10K: steps_N_pytorch_model.pt (:268-289)          | steps_N_model.pt at 10K/20K/30K (bf16)
resume: reload weights, fast-forward LR (:229-265)              | full state (DeepSpeed + RNG + scheduler +
                                                                |   data position), latest only (24 h job limit)
LeRobot loader, per-epoch reshuffle                             | oftvla.flow_data global stream (D9/D10)
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, set_seed
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler

from oftvla.data import LiberoLangDataset, identity_collate, load_stats, save_stats
from oftvla.flow_data import StreamBatchSampler
from oftvla.flow_model import FlowConfig, build_flow_model
from oftvla.vlm import vlm_tag

# fork LR-config keys -> our module attributes (fork: Qwen_GR00T/Qwen_PI.qwen_vl_interface / .action_model)
LR_GROUP_MODULES = {"qwen_vl_interface": "vlm", "action_model": "action_head"}
HEAD_DEFAULTS = {  # spec §11.2 / §12.2
    "gr00t": {"per_device_batch": 16, "max_train_steps": 80000},  # run_libero_train.sh:44, :47
    "pi": {"per_device_batch": 8, "max_train_steps": 100000},  # released PI run (slurm_script, config.yaml)
}
GB = 2 ** 30


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--head", required=True, choices=["gr00t", "pi"])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--train_ids", nargs="+", required=True)
    p.add_argument("--forbidden_ids", nargs="*", default=[])
    p.add_argument("--text_csv", required=True)
    p.add_argument("--setup", required=True, choices=["A0", "AS", "MAS"])
    p.add_argument("--base_vlm", required=True)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--per_device_batch", type=int, default=None)
    p.add_argument("--global_batch", type=int, default=128)
    p.add_argument("--max_train_steps", type=int, default=None, help="length of the cosine schedule")
    p.add_argument("--stop_at", type=int, default=30000, help="training stops here (decision 11.2)")
    p.add_argument("--warmup_steps", type=int, default=5000)
    p.add_argument("--lr_base", type=float, default=2.5e-5)
    p.add_argument("--lr_vlm", type=float, default=1e-5)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-8)
    p.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--save_steps", type=int, nargs="*", default=[10000, 20000, 30000])
    p.add_argument("--resume_every", type=int, default=1000)
    p.add_argument("--max_hours", type=float, default=23.0)
    p.add_argument("--exit_after_step", type=int, default=None, help="save resume state and exit (resume test)")
    p.add_argument("--log_indices", action="store_true", help="log rank-0 sample indices every logged step")
    a = p.parse_args(argv)
    for k, v in HEAD_DEFAULTS[a.head].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


def build_param_lr_groups(model, lrs: dict):
    """fork: trainer_tools.py:92-148 (nothing frozen, spec §6). Group order = the yaml's learning_rate order."""
    used, groups = set(), []
    for name, lr in lrs.items():
        if name == "base":
            continue
        params = list(getattr(model, LR_GROUP_MODULES[name]).parameters())
        if params:
            groups.append({"params": params, "lr": lr, "name": name})
            used.update(id(p) for p in params)
    other = [p for p in model.parameters() if id(p) not in used]
    if other:
        groups.append({"params": other, "lr": lrs["base"], "name": "base"})
    return groups


def build_optimizer_and_scheduler(model, args):
    """fork: train_starvla.py:101-127."""
    lrs = {"base": args.lr_base, "qwen_vl_interface": args.lr_vlm, "action_model": args.lr_head}  # yaml:50-53 order
    optimizer = torch.optim.AdamW(build_param_lr_groups(model, lrs), lr=args.lr_base, betas=tuple(args.betas),
                                  weight_decay=args.weight_decay, eps=args.eps, fused=True)
    scheduler = get_scheduler(name="cosine_with_min_lr", optimizer=optimizer, num_warmup_steps=args.warmup_steps,
                              num_training_steps=args.max_train_steps,
                              scheduler_specific_kwargs={"min_lr": args.min_lr})
    return optimizer, scheduler


class IndexedDataset(Dataset):
    """Adds the dataset index to each sample (for the data-order logs; the model ignores the key)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return {**self.ds[i], "idx": i}


def main():
    args = parse_args()
    accelerator = Accelerator(deepspeed_plugin=DeepSpeedPlugin())  # train_starvla.py:50-51
    rank, world = accelerator.process_index, accelerator.num_processes
    main_proc = accelerator.is_main_process
    accum = args.global_batch // (world * args.per_device_batch)
    if accum * world * args.per_device_batch != args.global_batch or accum != 1:
        # ds_config.yaml fixes gradient_accumulation_steps = 1 (original layout); other layouts need a decision.
        raise ValueError(f"global batch {args.global_batch} != {world} GPUs x {args.per_device_batch} (accumulation 1)")
    ds_cfg = accelerator.state.deepspeed_plugin.deepspeed_config
    ds_cfg["train_micro_batch_size_per_gpu"] = args.per_device_batch  # "auto" in ds_config (fork: from its loader)
    ds_cfg["train_batch_size"] = args.global_batch

    run_dir = args.run_dir
    paths = {k: os.path.join(run_dir, v) for k, v in dict(
        log="log.jsonl", stats="dataset_statistics.json", done="DONE", resume="resume", resume_tmp="resume.tmp",
        resume_old="resume.old", state="trainer_state.json").items()}
    if os.path.exists(paths["done"]):
        accelerator.print("[done] run already finished (DONE marker exists), nothing to do")
        return
    if main_proc:
        os.makedirs(run_dir, exist_ok=True)

    def log(*a):
        if main_proc:
            print(*a, flush=True)

    # ---- data (statistics fixed at the first start; rank 0 computes, the others wait and load) ----
    def make_ds():
        stats = load_stats(paths["stats"]) if os.path.exists(paths["stats"]) else None
        return LiberoLangDataset(args.data_root, [os.path.expanduser(x) for x in args.train_ids], None, args.setup,
                                 stats=stats, forbidden_id_files=[os.path.expanduser(x) for x in args.forbidden_ids],
                                 text_csv=os.path.expanduser(args.text_csv))
    if main_proc:
        ds = make_ds()
        if not os.path.exists(paths["stats"]):
            save_stats(ds.stats, paths["stats"])
    accelerator.wait_for_everyone()
    if not main_proc:
        ds = make_ds()
    log(f"[data] setup={args.setup} demos={len(ds.demos)} samples={len(ds)} texts={len(ds.texts)} "
        f"skipped_not_in_text_csv={ds.n_skipped_not_in_text}")

    # ---- model, optimiser, scheduler (fork order: build, optimiser/scheduler, set_seed(seed+rank), prepare) ----
    set_seed(args.seed)  # identical head init for every text condition (the fork does not seed here)
    model = build_flow_model(FlowConfig(head=args.head, base_vlm=args.base_vlm))
    optimizer, scheduler = build_optimizer_and_scheduler(model, args)
    for g in optimizer.param_groups:
        log(f"[optim] group {g['name']}: lr={g['lr']} tensors={len(g['params'])} "
            f"params={sum(p.numel() for p in g['params']):,}")
    set_seed(args.seed + rank)  # train_starvla.py:144-145
    model, optimizer = accelerator.prepare(model, optimizer)  # scheduler not prepared (as in the fork)
    unwrapped = accelerator.unwrap_model(model)
    log(f"[model] training flags: vlm={unwrapped.vlm.training} head={unwrapped.action_head.training}; "
        f"zero_stage={ds_cfg['zero_optimization']['stage']} bf16={ds_cfg['bf16']['enabled']} "
        f"micro={ds_cfg['train_micro_batch_size_per_gpu']} global={ds_cfg['train_batch_size']} world={world}")

    # ---- resume ----
    step, draws = 0, 0
    if os.path.isdir(paths["resume"]):
        accelerator.load_state(paths["resume"])  # DeepSpeed model + optimizer partitions + per-rank RNG
        st = json.load(open(os.path.join(paths["resume"], "trainer_state.json")))
        step, draws = st["step"], st["draws"]
        scheduler.load_state_dict(torch.load(os.path.join(paths["resume"], "scheduler.pt"), weights_only=False))
        log(f"[resume] from step {step} (draws {draws}), lr {scheduler.get_last_lr()}")
    if main_proc:
        with open(os.path.join(run_dir, "args.json"), "w") as f:
            json.dump({**vars(args), "world": world, "accum": accum, "vlm_tag": vlm_tag(args.base_vlm)}, f, indent=2)

    sampler = StreamBatchSampler(len(ds), args.seed, world, rank, args.per_device_batch, start_draw=draws)
    loader = DataLoader(IndexedDataset(ds), batch_sampler=sampler, num_workers=args.num_workers,
                        collate_fn=identity_collate, persistent_workers=args.num_workers > 0,
                        prefetch_factor=4 if args.num_workers > 0 else None)
    batches = iter(loader)

    def next_batch():
        nonlocal draws
        b = next(batches)
        draws += 1
        return b

    def save_resume():
        """Full state into resume.tmp; on success on all ranks it replaces resume/ (latest only). One retry."""
        for attempt in (1, 2):
            ok = torch.tensor(1, device=accelerator.device)
            try:
                if main_proc and os.path.exists(paths["resume_tmp"]):
                    shutil.rmtree(paths["resume_tmp"])
                accelerator.wait_for_everyone()
                accelerator.save_state(paths["resume_tmp"])
                if main_proc:
                    torch.save(scheduler.state_dict(), os.path.join(paths["resume_tmp"], "scheduler.pt"))
                    with open(os.path.join(paths["resume_tmp"], "trainer_state.json"), "w") as f:
                        json.dump({"step": step, "draws": draws}, f)
            except Exception as e:  # noqa: BLE001
                print(f"[resume] rank {rank} save attempt {attempt} failed: {e!r}", flush=True)
                ok.zero_()
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if ok.item() == 1:
                if main_proc:
                    if os.path.exists(paths["resume"]):
                        os.replace(paths["resume"], paths["resume_old"])
                    os.replace(paths["resume_tmp"], paths["resume"])
                    if os.path.exists(paths["resume_old"]):
                        shutil.rmtree(paths["resume_old"])
                accelerator.wait_for_everyone()
                log(f"[resume] saved at step {step}")
                return True
        log(f"[resume] save FAILED twice at step {step}; previous resume state kept")
        return False

    def save_weights(s):
        sd = accelerator.get_state_dict(model)  # train_starvla.py:278 (_get_state_dict)
        if main_proc:
            sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in sd.items()}
            tmp = os.path.join(run_dir, f"steps_{s}_model.pt.tmp")
            torch.save(sd, tmp)
            os.replace(tmp, os.path.join(run_dir, f"steps_{s}_model.pt"))
            print(f"[ckpt] weights saved at step {s}", flush=True)
        accelerator.wait_for_everyone()

    # ---- loop (train_starvla.py:351-384, _train_step :415-436) ----
    t_start = t_log = time.time()
    data_time, losses = 0.0, []
    torch.cuda.reset_peak_memory_stats()
    while step < args.stop_at:
        t0 = time.time()
        batch = next_batch()
        data_time += time.time() - t0
        idx_r0 = [s["idx"] for s in batch]
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(batch)["action_loss"]
            accelerator.backward(loss)  # DeepSpeed: backward + clipping (ds_config 1.0) + optimizer step
            accelerator.clip_grad_norm_(model.parameters(), 1.0)  # no-op under DeepSpeed, kept as in the fork
            optimizer.step()  # no-op wrapper under DeepSpeed
            if accelerator.sync_gradients:
                scheduler.step()
        if accelerator.sync_gradients:
            step += 1
        losses.append(loss.detach().float())

        rec = None
        if step % args.log_every == 0:
            torch.cuda.synchronize()
            dt = time.time() - t_log
            loss_local = torch.stack(losses).mean()
            loss_mean = accelerator.reduce(loss_local.clone(), reduction="mean")
            mem = torch.tensor([torch.cuda.max_memory_allocated() / GB], device=accelerator.device)
            mem_min = mem.clone()
            dist.all_reduce(mem, op=dist.ReduceOp.MAX)
            dist.all_reduce(mem_min, op=dist.ReduceOp.MIN)
            rec = {"step": step, "loss": loss_mean.item(), "loss_rank0": loss_local.item(),
                   "grad_norm": float(model.get_global_grad_norm()),
                   **{f"lr_{g['name']}": lr for g, lr in zip(optimizer.param_groups, scheduler.get_last_lr())},
                   "sec_per_step": dt / args.log_every, "data_frac": data_time / dt,
                   "peak_mem_gb_max": mem.item(), "peak_mem_gb_min": mem_min.item(), "draws": draws}
            if args.log_indices:
                rec["idx_rank0"] = idx_r0
            t_log, data_time, losses = time.time(), 0.0, []
        if step % args.eval_every == 0:  # train_starvla.py:369-370, 386-399: consumes a batch, train mode
            ev = next_batch()
            pred = unwrapped.predict_action(ev)
            gt = np.stack([s["action"] for s in ev]).astype(np.float32)
            rec = rec or {"step": step}
            rec["mse_score"] = float(np.linalg.norm(pred - gt) / gt.size)
        if rec and main_proc:
            print(json.dumps(rec), flush=True)
            with open(paths["log"], "a") as f:
                f.write(json.dumps(rec) + "\n")
        if step in args.save_steps:
            save_weights(step)
        stop_now = (args.exit_after_step is not None and step == args.exit_after_step) or \
                   ((time.time() - t_start) / 3600 > args.max_hours)
        if step < args.stop_at and (step % args.resume_every == 0 or stop_now):
            save_resume()
        if stop_now and step < args.stop_at:
            log(f"[exit] stopping at step {step} for resubmission")
            return
    if main_proc:
        open(paths["done"], "w").write(f"finished at step {step}\n")
        if os.path.exists(paths["resume"]):
            shutil.rmtree(paths["resume"])  # final weights saved; the resume state is no longer needed
    accelerator.wait_for_everyone()
    log("[done]")


if __name__ == "__main__":
    main()
