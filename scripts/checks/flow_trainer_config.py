"""Config-level equivalence of the flow trainer's optimiser / scheduler with the fork's trainer (CPU, no training).

Fork side: the fork's model (build_framework from the LIBERO yaml with the run-script overrides), the fork's
build_param_lr_groups (trainer_tools.py:92-148) and the fork's own setup_optimizer_and_scheduler source
(train_starvla.py:101-127, extracted with ast and executed unchanged; importing train_starvla.py would build an
Accelerator at import time). Our side: oftvla.train_flow.build_optimizer_and_scheduler.
Compared: group names / order, tensors and parameters per group (and the parameter names in each group),
all optimiser hyperparameters, and the LR of every group at steps 0, 5000 (end of warmup), 10000, 30000.

Usage: python scripts/checks/flow_trainer_config.py --head gr00t|pi --starvla_root ~/VLA_FT/starVLA --base_vlm ... --out F.json
"""

import argparse
import ast
import json
import sys

import torch

from oftvla.flow_model import FlowConfig, build_flow_model, remap_starvla_state_dict
from oftvla.train_flow import build_optimizer_and_scheduler, parse_args as flow_args

STEPS = [0, 5000, 10000, 30000]
OPT_KEYS = ["lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused"]


def fork_side(args, max_steps):
    sys.path.insert(0, args.starvla_root)
    from omegaconf import OmegaConf
    from transformers import get_scheduler
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.model.framework.share_tools import apply_config_compat
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups, normalize_dotlist_args

    cfg = OmegaConf.load(f"{args.starvla_root}/examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml")
    # run_libero_train.sh:39-50 CLI overrides (freeze_modules '' -> 'true', as normalize_dotlist_args does)
    cli = ["--framework.name", {"gr00t": "QwenGR00T", "pi": "QwenPI"}[args.head],
           "--framework.qwenvl.base_vlm", args.base_vlm, "--trainer.freeze_modules",
           "--trainer.max_train_steps", str(max_steps)]
    if args.head == "pi":
        cli += ["--framework.action_model.diffusion_model_cfg.use_canonical_forward", "false"]
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(normalize_dotlist_args(cli)))
    cfg = apply_config_compat(cfg)
    model = build_framework(cfg)

    src = open(f"{args.starvla_root}/starVLA/training/train_starvla.py").read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "setup_optimizer_and_scheduler")
    ns = {"torch": torch, "get_scheduler": get_scheduler, "build_param_lr_groups": build_param_lr_groups,
          "dist": torch.distributed, "Tuple": tuple, "logger": None}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "train_starvla.py", "exec"), ns)
    opt, sched = ns["setup_optimizer_and_scheduler"](model=model, cfg=cfg)
    names = {id(p): remap_starvla_state_dict({n: None}).popitem()[0] for n, p in model.named_parameters()}
    return model, opt, sched, names


def our_side(args, max_steps):
    a = flow_args(["--head", args.head, "--run_dir", "x", "--data_root", "x", "--train_ids", "x", "--text_csv", "x",
                   "--setup", "MAS", "--base_vlm", args.base_vlm, "--max_train_steps", str(max_steps)])
    model = build_flow_model(FlowConfig(head=args.head, base_vlm=args.base_vlm))
    opt, sched = build_optimizer_and_scheduler(model, a)
    names = {id(p): n for n, p in model.named_parameters()}
    return model, opt, sched, names


def describe(opt, sched, names):
    groups = []
    for g in opt.param_groups:
        groups.append({"name": g.get("name"), "tensors": len(g["params"]), "params": sum(p.numel() for p in g["params"]),
                       "hyper": {k: (list(g[k]) if isinstance(g[k], tuple) else g[k]) for k in OPT_KEYS if k in g},
                       "param_names": sorted(names[id(p)] for p in g["params"])})
    defaults = {k: (list(v) if isinstance(v, tuple) else v) for k, v in opt.defaults.items()}
    lrs, s = {}, 0
    for target in STEPS:
        while s < target:
            sched.step()
            s += 1
        lrs[target] = list(sched.get_last_lr())
    return {"groups": groups, "defaults": defaults, "scheduler": type(sched).__name__, "lr_at": lrs}


def main():
    ap = argparse.ArgumentParser()
    for k in ("head", "starvla_root", "base_vlm", "out"):
        ap.add_argument(f"--{k}", required=True)
    args = ap.parse_args()
    max_steps = {"gr00t": 80000, "pi": 100000}[args.head]
    import warnings
    warnings.filterwarnings("ignore")
    f = describe(*[x for i, x in enumerate(fork_side(args, max_steps)) if i != 0])
    o = describe(*[x for i, x in enumerate(our_side(args, max_steps)) if i != 0])
    ok = {
        "group_names_order": [g["name"] for g in f["groups"]] == [g["name"] for g in o["groups"]],
        "tensors_params_per_group": [(g["tensors"], g["params"]) for g in f["groups"]] == [(g["tensors"], g["params"]) for g in o["groups"]],
        "param_names_per_group": [g["param_names"] for g in f["groups"]] == [g["param_names"] for g in o["groups"]],
        "group_hyperparameters": [g["hyper"] for g in f["groups"]] == [g["hyper"] for g in o["groups"]],
        "optimizer_defaults": f["defaults"] == o["defaults"],
        "lr_at_steps": f["lr_at"] == o["lr_at"],
    }
    print(f"### {args.head}: schedule {max_steps} steps")
    for side, d in (("fork", f), ("ours", o)):
        print(f"  {side}: " + "; ".join(f"{g['name']} {g['tensors']} tensors / {g['params']:,} params / {g['hyper']}" for g in d["groups"]))
        print(f"  {side} defaults: {d['defaults']}")
        print(f"  {side} LR at steps: " + ", ".join(f"{s}: {v}" for s, v in d["lr_at"].items()))
    for k, v in ok.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    for d in (f, o):
        for g in d["groups"]:
            g.pop("param_names")
    json.dump({"fork": f, "ours": o, "checks": ok}, open(args.out, "w"), indent=1)
    sys.exit(0 if all(ok.values()) else 1)


if __name__ == "__main__":
    main()
