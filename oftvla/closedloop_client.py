"""Closed-loop LIBERO client with the oracle step switcher and the MAS progress scorer (M15).

Inference is identical to oftvla/libero_client.py (its preprocess / binarize_gripper / constants are imported):
env 256x256, env.seed(7) per task, LIBERO fixed init states, 10 dummy steps, one 8-step chunk per query executed
open-loop, gripper > 0.5 -> -1 (open) else +1, step limits 400 / 520, success = env done.

Per policy step (frame f, after the 10 dummy steps) the world state is read BEFORE the action (same convention as
the offline replay): WorldObserver -> AS and MAS Switchers (always running; silent unless the model's setup uses
them) and the Scorer. The prompt of a query is the text of the model's setup: A0 = text_A0, AS / MAS = the text of
the switcher's current block (text_block) at the query frame. The executed (binarised) gripper command is fed to
the switchers after the action is chosen. Nothing here writes to the simulator.

Text sources: --texts spec (switch_spec.json: text_A0 and the AS / MAS block texts of the e3v5 models) or
--texts labels (the old A0 sentence from switch_labels_v1.csv, e3_A0; setup A0 only).

Outputs in --out_dir: episodes.csv (one row per episode, resumable) and episodes_<task_id>.jsonl (one JSON per
episode: switch / event frames, prompt changes, ignored commands, scorer result).
Usage (LIBERO container): python3 -m oftvla.closedloop_client --ids IDS --spec SPEC --setup MAS --texts spec --out_dir D
"""
import argparse
import csv
import json
import os
import pathlib
import time

import numpy as np

from oftvla import ipc
from oftvla.libero_client import DUMMY_ACTION, ENV_RESOLUTION, MAX_STEPS, a0_texts, binarize_gripper, preprocess
from oftvla.scorer import Scorer, WorldObserver
from oftvla.switcher import Switcher

COLS = ["suite", "task", "task_id", "group", "level", "init_id", "setup", "success", "steps", "seconds",
        "mas_block", "as_block", "score_furthest", "score_n_steps", "first_grasp", "first_grasp_correct",
        "n_prompt_changes"]


def run_episode(env, init_state, texts, setup, spec, conn, chunk_len, max_steps, num_steps_wait, log_prompts=False):
    env.reset()
    obs = env.set_init_state(init_state)
    t, step, chunk = 0, 0, None
    world_obs = scorer = None
    sws = {}
    guards = sorted({b["guard_obj"] for s in ("AS", "MAS") for b in spec[s] if b.get("guard_obj")})
    queries, last_lang, success = [], None, 0
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, _, _ = env.step(DUMMY_ACTION)
            t += 1
            continue
        if world_obs is None:          # after reset / settling: the sim may have been rebuilt by reset()
            world_obs = WorldObserver(env, spec["MAS"], guards)
            scorer = Scorer(spec["MAS"], set(world_obs.parts))
            sws = {s: Switcher(spec[s]) for s in ("AS", "MAS")}
        c, z, d = world_obs.observe()
        world = world_obs.world()
        scorer.observe(c, z, d, world)
        for sw in sws.values():
            sw.observe(c, z, d, world["part_goal"])
        lang = texts if setup == "A0" else texts[sws[setup].text_block()]
        main = preprocess(obs["agentview_image"])
        wrist = preprocess(obs["robot0_eye_in_hand_image"])
        if step % chunk_len == 0 or chunk is None:
            chunk = ipc.request(conn, {"images": [main, wrist], "lang": lang})["actions"]  # (8, 7)
            if lang != last_lang:
                blk = None if setup == "A0" else sws[setup].text_block()
                queries.append((step, blk))
                if log_prompts:
                    print(f"[prompt] frame {step} block {blk}:\n{lang}", flush=True)
                last_lang = lang
        a = chunk[step % chunk_len]
        action = np.concatenate([a[:6], [binarize_gripper(float(a[6]))]]).astype(np.float32)
        for sw in sws.values():
            sw.act(float(action[6]))
        obs, _, done, _ = env.step(action.tolist())
        if done:
            success = 1
            step += 1
            break
        t += 1
        step += 1
    log = dict(success=success, steps=step, prompt_changes=queries,
               switcher={s: dict(switch_frames=sw.switch_frames, events=sw.events, ignored=sw.ignored,
                                 block=sw.block(), ended=sw.ended()) for s, sw in sws.items()},
               score=scorer.result() if scorer is not None else None)
    return success, step, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="csv with suite, task, init_id (+ optional group, level)")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--setup", choices=("A0", "AS", "MAS"), required=True)
    ap.add_argument("--texts", choices=("spec", "labels"), required=True)
    ap.add_argument("--labels_csv", default=None, help="for --texts labels")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--port", type=int, default=10093)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--log_prompts", action="store_true", help="print the prompt at every prompt change")
    a = ap.parse_args()
    assert a.texts == "spec" or a.setup == "A0", "--texts labels only for setup A0"

    from libero.libero import benchmark, get_libero_path  # only in the LIBERO container
    from libero.libero.envs import OffScreenRenderEnv

    np.random.seed(a.seed)
    spec = json.load(open(a.spec))["tasks"]
    labels = a0_texts(a.labels_csv) if a.texts == "labels" else None
    with open(a.ids) as f:
        rows = list(csv.DictReader(f))
    suites = {s: benchmark.get_benchmark_dict()[s]() for s in {r["suite"] for r in rows}}
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "episodes.csv")
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {(r["task"], r["init_id"]) for r in csv.DictReader(f)}
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=COLS)
    if not done:
        w.writeheader()

    conn = ipc.connect(a.port)
    chunk_len = ipc.request(conn, {"meta": True})["chunk_len"]
    env, env_task = None, None
    for r in rows:
        if (r["task"], r["init_id"]) in done:
            continue
        ts = suites[r["suite"]]
        tid = ts.get_task_names().index(r["task"])
        task = ts.get_task(tid)
        e = spec[r["task"]]
        if env_task != r["task"]:
            if env is not None:
                env.close()
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RESOLUTION,
                                     camera_widths=ENV_RESOLUTION)
            env.seed(a.seed)
            env_task = r["task"]
            if a.texts == "labels":
                texts = labels[(r["suite"], r["task"])]
            elif a.setup == "A0":
                texts = e["text_A0"]
            else:
                texts = [b["text"] for b in e[a.setup] if b.get("text") is not None]
            print(f"[task] {e['task_id']} setup={a.setup} texts={a.texts}: {len(texts) if isinstance(texts, list) else 1} text(s)",
                  flush=True)
        t0 = time.time()
        success, steps, log = run_episode(env, ts.get_task_init_states(tid)[int(r["init_id"])], texts, a.setup, e,
                                          conn, chunk_len, MAX_STEPS[r["suite"]], a.num_steps_wait, a.log_prompts)
        sc = log["score"]
        fg = sc["first_grasp"]
        row = dict(suite=r["suite"], task=r["task"], task_id=e["task_id"], group=r.get("group", ""),
                   level=r.get("level", ""), init_id=r["init_id"], setup=a.setup, success=success, steps=steps,
                   seconds=round(time.time() - t0, 1), mas_block=log["switcher"]["MAS"]["block"],
                   as_block=log["switcher"]["AS"]["block"], score_furthest=sc["furthest"], score_n_steps=len(sc["done"]),
                   first_grasp="" if fg is None else f"{fg['obj']}@{fg['frame']}",
                   first_grasp_correct="" if fg is None else int(fg["correct"]),
                   n_prompt_changes=len(log["prompt_changes"]))
        with open(os.path.join(a.out_dir, f"episodes_{e['task_id']}.jsonl"), "a") as jf:
            jf.write(json.dumps(dict(task_id=e["task_id"], init_id=int(r["init_id"]), setup=a.setup, **log)) + "\n")
        w.writerow(row)
        f.flush()
        print(f"[ep] {e['task_id']} init {r['init_id']}: success={success} steps={steps} mas_block={row['mas_block']} "
              f"switches MAS={log['switcher']['MAS']['switch_frames']} prompt_changes={log['prompt_changes']} "
              f"score_done={sc['done']}", flush=True)
    conn.close()
    with open(out) as f:
        res = list(csv.DictReader(f))
    print(f"\n[summary] {sum(int(r['success']) for r in res)}/{len(res)} episodes successful", flush=True)


if __name__ == "__main__":
    main()
