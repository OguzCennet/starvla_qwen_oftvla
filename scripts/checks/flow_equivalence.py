"""Full-size equivalence: oftvla QwenGR00T / QwenPI vs the StarVLA fork (985c223) on one GPU, bf16.

Setup as in the original training: weights in bf16 (DeepSpeed bf16), the step under bf16 autocast
(train_starvla.py:420), the fork's own autocast(float32) around the head kept. Real Qwen3-VL-4B weights;
the head is initialised by the fork (seed 0) and the same state_dict is loaded into ours. Both models in
eval mode (dropout off). Batch: 6 real E3 val frames, texts mixed A0 / AS / MAS (left padding), CoT_prompt on.

Checks:
  forward   loss with injected noise / t                           -> torch.equal required
  predict   predict_action with injected initial noise             -> torch.equal required
  grads     the fork's backward is not deterministic on GPU, so the fork is run K times (K(K-1)/2
            fork-vs-fork pairs) and ours is compared with each fork run; per group (VLM, head):
            max |a-b|, max over tensors of max|a-b|/max|a|, and rel_l2 = ||a-b|| / ||a|| over the group.
            Pass: ours-vs-fork max |a-b| and rel_l2 <= the largest fork-vs-fork value.
  negative  perturbed head weight, flipped mask entry, CoT_prompt off -> ours' loss must differ
  memory    peak memory of ours, forward+backward, at the equivalence batch (6) and at micro-batch 16
Runs one model at a time on the GPU (fork first; its state_dict and its gradients are kept on the CPU).

Usage (GPU node, container): python scripts/checks/flow_equivalence.py --head gr00t|pi --out FILE.json \
    --starvla_root ~/VLA_FT/starVLA --base_vlm $BASE_VLM --data_root $DATA --val_ids $SPLITS/e3_val_ids.csv \
    --text_csv $TEXTS/val.csv --stats $RUNS/e3v5_A0/dataset_statistics.json
"""

import argparse
import gc
import json
import sys
import time
from unittest import mock

import numpy as np
import torch

from oftvla.data import LiberoLangDataset, load_stats
from oftvla.flow_model import FlowConfig, build_flow_model, remap_starvla_state_dict

GB = 1024 ** 3


def log(*a):
    print(*a, flush=True)


def make_batch(args, n_samples=6):
    stats = load_stats(args.stats)
    ds = {s: LiberoLangDataset(args.data_root, [args.val_ids], None, setup=s, stats=stats, text_csv=args.text_csv)
          for s in ("A0", "AS", "MAS")}
    n = len(ds["A0"])
    idx = [n * k // (n_samples + 1) + 13 for k in range(1, n_samples + 1)]
    setups = (["A0", "AS", "MAS"] * n_samples)[:n_samples]
    batch = [ds[s][i] for s, i in zip(setups, idx)]
    if n_samples > 6:
        return batch
    for s, i, b in zip(setups, idx, batch):
        d, t = ds[s].starts[i]
        log(f"  sample {s:3s} {ds[s].demos[d]['key']} t={t} text={len(b['lang'])} chars: {b['lang'][:70]!r}")
    return batch


def build_fork(args):
    sys.path.insert(0, args.starvla_root)
    from omegaconf import OmegaConf
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.model.framework.share_tools import apply_config_compat

    cfg = OmegaConf.load(f"{args.starvla_root}/examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml")
    cfg.framework.name = {"gr00t": "QwenGR00T", "pi": "QwenPI"}[args.head]
    cfg.framework.qwenvl.base_vlm = args.base_vlm
    if args.head == "pi":  # the official legacy switch (run_policy_server.sh:29), as a CLI override would set it
        OmegaConf.update(cfg, "framework.action_model.diffusion_model_cfg.use_canonical_forward", False, force_add=True)
    cfg = apply_config_compat(cfg)  # train_starvla.py:516
    torch.manual_seed(0)
    return build_framework(cfg)


def fork_loss(fork, batch, noise, t):
    with mock.patch.object(torch, "randn", side_effect=lambda *a, **k: noise.clone()) as rn:
        fork.action_model.sample_time = lambda batch_size, device, dtype: t.clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):  # train_starvla.py:420
            loss = fork(batch)["action_loss"]
    assert rn.call_count == 1, rn.call_count
    return loss


def ours_loss(model, batch, noise, t):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(batch, noise=noise.clone(), t=t.clone())["action_loss"]


def fork_predict(fork, batch, init):
    with mock.patch.object(torch, "randn", side_effect=lambda *a, **k: init.clone()) as rn:
        out = fork.predict_action(examples=batch)["normalized_actions"]
    assert rn.call_count == 1
    return torch.as_tensor(out)


def group_of(name):
    return "vlm" if name.startswith("vlm.") else "head"


def collect_grads(named_params, rename=lambda n: n):
    return {rename(n): (p.grad.detach().to("cpu", copy=True) if p.grad is not None else None) for n, p in named_params}


def grad_stats(ref, other, dev):
    """ref, other: {name: CPU grad}. Per group: exact tensors, max|a-b|, max rel (per tensor), rel_l2."""
    st = {g: dict(tensors=0, exact=0, missing=0, max_abs=0.0, max_rel=0.0, worst=None, _d2=0.0, _a2=0.0)
          for g in ("vlm", "head")}
    for n, a in ref.items():
        g = st[group_of(n)]
        b = other[n]
        g["tensors"] += 1
        if a is None or b is None:
            g["missing"] += int((a is None) != (b is None))
            g["exact"] += int(a is None and b is None)
            continue
        if torch.equal(a, b):
            g["exact"] += 1
            g["_a2"] += float(a.to(dev).float().pow(2).sum())
            continue
        af, bf = a.to(dev).float(), b.to(dev).float()
        d = (af - bf).abs()
        dmax, amax = d.max().item(), af.abs().max().item()
        if dmax > g["max_abs"]:
            g["max_abs"], g["worst"] = dmax, n
        g["max_rel"] = max(g["max_rel"], dmax / max(amax, 1e-30))
        g["_d2"] += float(d.pow(2).sum())
        g["_a2"] += float(af.pow(2).sum())
    for g in st.values():
        g["rel_l2"] = (g.pop("_d2") ** 0.5) / max(g.pop("_a2") ** 0.5, 1e-30)
    return st


def summarize(stats_list):
    """min / max over a list of grad_stats results, per group and metric."""
    out = {}
    for g in ("vlm", "head"):
        out[g] = {}
        for m in ("max_abs", "max_rel", "rel_l2", "exact"):
            vals = [s[g][m] for s in stats_list]
            out[g][m] = [min(vals), max(vals)]
        out[g]["tensors"] = stats_list[0][g]["tensors"]
        out[g]["missing"] = max(s[g]["missing"] for s in stats_list)
    return out


def fmt(summary):
    rows = []
    for g, s in summary.items():
        rows.append(f"    {g:4s}: bitwise-equal tensors {s['exact'][0]}-{s['exact'][1]} of {s['tensors']}, "
                    f"missing {s['missing']}, max|diff| {s['max_abs'][0]:.3e}-{s['max_abs'][1]:.3e}, "
                    f"max rel {s['max_rel'][0]:.3e}-{s['max_rel'][1]:.3e}, "
                    f"rel_l2 {s['rel_l2'][0]:.3e}-{s['rel_l2'][1]:.3e}")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    for k in ("head", "out", "starvla_root", "base_vlm", "data_root", "val_ids", "text_csv", "stats"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--fork_runs", type=int, default=4)
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms(True) (needs CUBLAS_WORKSPACE_CONFIG=:4096:8): if the fork "
                         "then reproduces itself bitwise, ours must match it bitwise in the gradients too")
    args = ap.parse_args()
    if args.deterministic:
        import os
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "set CUBLAS_WORKSPACE_CONFIG=:4096:8"
        torch.use_deterministic_algorithms(True, warn_only=False)
    dev = torch.device("cuda")
    res = {"head": args.head, "deterministic": bool(args.deterministic)}

    log("### batch (E3 val frames)")
    batch = make_batch(args)
    batch16 = make_batch(args, 16)

    # ---------------- fork ----------------
    log("### fork")
    fork = build_fork(args).to(dtype=torch.bfloat16).to(dev).eval()
    R = 8 if args.head == "gr00t" else 2
    g = torch.Generator().manual_seed(1)
    noise = torch.randn(len(batch) * R, 8, 7, generator=g).to(dev, torch.bfloat16)
    t = torch.rand(len(batch) * R, generator=g).to(dev, torch.bfloat16)
    init = torch.randn(len(batch), 8, 7, generator=g).to(dev, torch.bfloat16)

    inputs = fork.qwen_vl_interface.build_qwenvl_inputs(
        images=[s["image"] for s in batch], instructions=[s["lang"] for s in batch])
    mask = inputs["attention_mask"]
    res["tokens_per_sample"] = mask.sum(1).tolist()
    res["padded_rows"] = int((mask == 0).any(1).sum())
    log(f"  tokens per sample {res['tokens_per_sample']} (padded rows {res['padded_rows']}, length {mask.shape[1]})")
    assert res["padded_rows"] > 0
    assert "Your task is" in fork.qwen_vl_interface.processor.decode(inputs["input_ids"][0])

    with torch.no_grad():
        loss_f = fork_loss(fork, batch, noise, t)
    pred_f = fork_predict(fork, batch, init)
    log(f"  fork loss {loss_f.item():.6f}")

    rn = lambda n: remap_starvla_state_dict({n: None}).popitem()[0]  # noqa: E731
    fork_grads = []
    for k in range(args.fork_runs):
        torch.cuda.reset_peak_memory_stats()
        fork.zero_grad(set_to_none=True)
        fork_loss(fork, batch, noise, t).backward()
        res["fork_peak_gb_fwd_bwd"] = torch.cuda.max_memory_allocated() / GB
        fork_grads.append(collect_grads(fork.named_parameters(), rn))
    fork.zero_grad(set_to_none=True)
    pairs = [(i, j) for i in range(args.fork_runs) for j in range(i + 1, args.fork_runs)]
    res["fork_vs_fork"] = summarize([grad_stats(fork_grads[i], fork_grads[j], dev) for i, j in pairs])
    log(f"  gradients fork vs fork ({len(pairs)} pairs of {args.fork_runs} runs), range over pairs:\n"
        + fmt(res["fork_vs_fork"]))
    fork_sd = {k: v.detach().to("cpu", copy=True) for k, v in fork.state_dict().items()}
    del fork
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- ours ----------------
    log("### ours")
    ours = build_flow_model(FlowConfig(head=args.head, base_vlm=args.base_vlm))
    log("  " + str(ours.load_starvla_state_dict(fork_sd, strict=True)))
    del fork_sd
    ours = ours.to(dtype=torch.bfloat16).to(dev).eval()
    with torch.no_grad():
        loss_o = ours_loss(ours, batch, noise, t)
    pred_o = torch.as_tensor(ours.predict_action(batch, init_noise=init.clone()))
    res["loss_fork"], res["loss_ours"] = loss_f.item(), loss_o.item()
    res["loss_equal"] = bool(torch.equal(loss_f, loss_o))
    res["predict_equal"] = bool(torch.equal(pred_f, pred_o))
    res["predict_max_abs"] = float((pred_f - pred_o).abs().max())
    log(f"  [{'PASS' if res['loss_equal'] else 'FAIL'}] loss torch.equal: fork {loss_f.item():.8f} ours {loss_o.item():.8f}")
    log(f"  [{'PASS' if res['predict_equal'] else 'FAIL'}] predict_action torch.equal (max|diff| {res['predict_max_abs']:.3e})")

    torch.cuda.reset_peak_memory_stats()
    ours.zero_grad(set_to_none=True)
    try:
        ours_loss(ours, batch, noise, t).backward()
        res["ours_peak_gb_fwd_bwd"] = torch.cuda.max_memory_allocated() / GB
        our_grads = collect_grads(ours.named_parameters())
        res["ours_vs_fork"] = summarize([grad_stats(fg, our_grads, dev) for fg in fork_grads])
        log(f"  gradients ours vs each of the {args.fork_runs} fork runs, range:\n" + fmt(res["ours_vs_fork"]))
        ok = all(res["ours_vs_fork"][g]["max_abs"][1] <= res["fork_vs_fork"][g]["max_abs"][1]
                 and res["ours_vs_fork"][g]["rel_l2"][1] <= res["fork_vs_fork"][g]["rel_l2"][1]
                 and res["ours_vs_fork"][g]["missing"] == 0 for g in ("vlm", "head"))
        del our_grads
        res["grads_within_fork_range"] = ok
        log(f"  [{'PASS' if ok else 'FAIL'}] ours-vs-fork gradient difference within fork-vs-fork range")
    except torch.OutOfMemoryError as e:
        res["ours_backward"] = f"OOM: {str(e)[:200]}"
        log("  backward OOM -> forward only")
    ours.zero_grad(set_to_none=True)
    del fork_grads
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        ours_loss(ours, batch, noise, t)
    res["ours_peak_gb_fwd_only"] = torch.cuda.max_memory_allocated() / GB
    R = ours.repeated_diffusion_steps
    n16, t16 = noise.new_zeros(16 * R, 8, 7).normal_(), t.new_zeros(16 * R).uniform_()
    torch.cuda.reset_peak_memory_stats()
    try:
        ours.zero_grad(set_to_none=True)
        ours_loss(ours, batch16, n16, t16).backward()
        res["ours_peak_gb_fwd_bwd_mb16"] = torch.cuda.max_memory_allocated() / GB
    except torch.OutOfMemoryError as e:
        res["ours_peak_gb_fwd_bwd_mb16"] = f"OOM: {str(e)[:120]}"
    ours.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        ours_loss(ours, batch16, n16, t16)
    res["ours_peak_gb_fwd_only_mb16"] = torch.cuda.max_memory_allocated() / GB

    # ---------------- negative controls ----------------
    log("### negative controls (ours' loss must differ from the fork's)")
    neg = {}
    w = ours.action_head.action_encoder.layer1.weight
    with torch.no_grad():
        old = w[0, 0].clone()
        w[0, 0] += 0.01
        neg["head_weight"] = not torch.equal(ours_loss(ours, batch, noise, t), loss_f)
        w[0, 0] = old
        build = ours.adapter.build_inputs

        def flipped(*a, **k):
            x = build(*a, **k)
            r = int((x["attention_mask"] == 0).any(1).nonzero()[0])
            x["attention_mask"] = x["attention_mask"].clone()
            x["attention_mask"][r, 0] = 1 - x["attention_mask"][r, 0]
            return x

        ours.adapter.build_inputs = flipped
        neg["mask_entry"] = not torch.equal(ours_loss(ours, batch, noise, t), loss_f)
        ours.adapter.build_inputs = build
        ours.cfg.cot_prompt = None
        neg["cot_off"] = not torch.equal(ours_loss(ours, batch, noise, t), loss_f)
        ours.cfg.cot_prompt = FlowConfig(head=args.head).cot_prompt
        neg["restored_equal"] = bool(torch.equal(ours_loss(ours, batch, noise, t), loss_f))
    res["negative_controls"] = neg
    for k, v in neg.items():
        log(f"  {k}: {'differs' if k != 'restored_equal' else 'equal again'} = {v}")

    res["peak_gb_note"] = ("torch.cuda.max_memory_allocated, one model (bf16 weights + bf16 grads, no optimiser "
                           "state) on the GPU; batch 6 (equivalence) and micro-batch 16")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    log(json.dumps({k: v for k, v in res.items() if "peak" in k}, indent=1))


if __name__ == "__main__":
    t0 = time.time()
    main()
    log(f"done in {(time.time() - t0) / 60:.1f} min")
