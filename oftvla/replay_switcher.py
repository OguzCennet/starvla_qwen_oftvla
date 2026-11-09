"""Step-2 validation of the oracle switcher (M15): replay the E3 test demos in the simulator with the switcher
attached and compare its switch frames with the offline boundaries (idx_AS_v2 / idx_MAS_v2 changes of
A0ASMAS/test.csv).

  --mode actions   reset to the demo's first state, then execute the recorded actions (as a rollout would)
  --mode states    set every recorded state (the offline replay); isolates the causal rules from replay drift

Per demo and setup four switchers see the same frames: guarded (the M15 switcher) and unguarded (to report
whether the 5 cm guard changes any switch). In states mode the observer's contacts / heights of the first demo
of every task are checked against sim_segment.record on the same states (same contact definition).

Writes ONE csv (one row per demo; lists as JSON). Runs in the LIBERO container (sym-openvla-oft.sif):
  python3 -m oftvla.replay_switcher --spec SPEC --texts A0ASMAS/test.csv --mode actions --out OUT.csv --workers 32
"""
import argparse
import collections
import csv
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np

from oftvla.scorer import Scorer, WorldObserver
from oftvla.switcher import Switcher, first_lifts

RAW = {"libero_90": "/hnvme/workspace/y307bd13-vgv/data/libero_raw/libero_90",
       "libero_10": "/hnvme/workspace/y307bd13-vgv/data/libero_raw/libero_10"}
SETUPS = ("AS", "MAS")
csv.field_size_limit(sys.maxsize)


def offline_switches(rows, setup):
    """[(new block idx, first raw step with it)] from the v1 rows of one demo."""
    out, prev = [], None
    for r in rows:
        j = int(r[f"idx_{setup}_v2"])
        if prev is not None and j != prev:
            out.append((j, int(r["raw_step"])))
        prev = j
    return out


def compare(sw_frames, offline, raws, blocks, n_text):
    """Match offline switch into block j with the switcher's start of block j (switch_frames[j-1])."""
    raws = np.asarray(raws)
    def v1(raw):
        return int(np.searchsorted(raws, raw, side="left"))      # first v1 row with raw_step >= raw
    res = []
    for j, b in offline:
        ended = blocks[j - 1]
        s = sw_frames[j - 1] if j - 1 < len(sw_frames) else None
        res.append(dict(j=j, kind=ended["kind"], step=ended.get("step") or ended.get("label"),
                        part=ended["kind"] in ("first_touch", "close_or_touch") or ended.get("is_part", False),
                        off_raw=b, sw_raw=s, off_t=v1(b), sw_t=None if s is None else v1(s)))
    n_sw = sum(1 for i, s in enumerate(sw_frames) if s <= raws[-1] and i + 1 < n_text)
    return res, n_sw


def replay_task(job):
    task, suite, demos, spec, mode, check_record = job
    sys.path.insert(0, os.path.expanduser("~/VLA_FT/libero-subinstructions"))
    import h5py
    import sim_segment as S
    from sim_trace import load_recorded_model, make_env
    from libero.libero import get_libero_path

    bddl = pathlib.Path(get_libero_path("bddl_files")) / suite / f"{task}.bddl"
    env = make_env(bddl)
    sim = env.env.sim
    parts = set(S.articulated_parts(sim))
    guards = sorted({b["guard_obj"] for s in SETUPS for b in spec[s] if b.get("guard_obj")})
    out = []
    with h5py.File(f"{RAW[suite]}/{task}_demo.hdf5", "r") as h:
        for n_demo, (demo, rows) in enumerate(demos):
            t0 = time.time()
            g = h["data"][demo]
            states, actions, xml = g["states"][()], g["actions"][()], g.attrs["model_file"]
            if mode == "actions":
                env.reset()        # resets the robosuite horizon counter and the controller (as libero_client)
            load_recorded_model(env, xml.decode() if isinstance(xml, bytes) else xml)
            env.set_init_state(states[0])
            sim = env.env.sim      # env.reset() may rebuild the sim
            obs = WorldObserver(env, spec["MAS"], guards)
            scorer = Scorer(spec["MAS"], parts)
            for s in SETUPS:
                for b in spec[s]:
                    b["is_part"] = b["obj"] in parts
            sws = {(s, guard): Switcher(spec[s], guard_m=(0.05 if guard else float("inf")))
                   for s in SETUPS for guard in (True, False)}
            hist = collections.defaultdict(list)
            success, T = 0, len(actions)
            for f in range(T):
                if mode == "states":
                    sim.set_state_from_flattened(states[f])
                    sim.forward()
                c, z, d = obs.observe()
                world = obs.world()
                scorer.observe(c, z, d, world)
                for k, v in c.items():
                    hist["c_" + k].append(v)
                for k, v in z.items():
                    hist["z_" + k].append(v)
                for sw in sws.values():
                    sw.observe(c, z, d, world["part_goal"])
                    sw.act(actions[f][-1])
                if mode == "actions":
                    _, _, done, _ = env.step(actions[f].tolist())
                    success = max(success, int(done))
            rec_check = ""
            if mode == "states" and check_record and n_demo == 0:
                rec = S.record(env, states)
                bad = [k for k in rec["c_obj"] if not np.array_equal(rec["c_obj"][k], np.asarray(hist["c_" + k]))]
                bad += [k for k in rec["c_part"] if not np.array_equal(rec["c_part"][k], np.asarray(hist["c_" + k]))]
                bad += [k for k in rec["pos"] if not np.allclose(rec["pos"][k][:, 2], np.asarray(hist["z_" + k]))]
                rec_check = "ok" if not bad else "MISMATCH:" + ",".join(bad)
            objs = [k[2:] for k in hist if k.startswith("z_")]
            lifts = first_lifts({o: hist["c_" + o] for o in objs}, {o: hist["z_" + o] for o in objs},
                                actions[:, -1], objs)
            row = dict(task_id=spec["task_id"], task=task, demo=demo, mode=mode, T=T, success=success,
                       first_lifts=json.dumps(sorted(lifts.items(), key=lambda x: x[1])), record_check=rec_check,
                       seconds=round(time.time() - t0, 1))
            raws = [int(r["raw_step"]) for r in rows]
            for s in SETUPS:
                sw, sw0 = sws[(s, True)], sws[(s, False)]
                res, n_sw = compare(sw.switch_frames, offline_switches(rows, s), raws, spec[s], sw.n_text)
                row[f"{s}_match"] = json.dumps(res)
                row[f"{s}_n_off"] = len(offline_switches(rows, s))
                row[f"{s}_n_sw"] = n_sw
                row[f"{s}_switches"] = json.dumps(sw.switch_frames)
                row[f"{s}_events"] = json.dumps(sw.events)
                row[f"{s}_ignored"] = json.dumps(sw.ignored)
                row[f"{s}_noguard_switches"] = json.dumps(sw0.switch_frames)
                row[f"{s}_guard_changed"] = int(sw.switch_frames != sw0.switch_frames)
            # scorer: done frame of MAS step j vs the offline switch out of it (into j + 1), v1 frames
            sc = scorer.result()
            ra = np.asarray(raws)
            off = dict(offline_switches(rows, "MAS"))
            cmp_ = []
            for j, (typ, d) in enumerate(zip(sc["types"], sc["done"])):
                b = off.get(j + 1)
                cmp_.append(dict(j=j, type=typ, done=d, off_raw=b,
                                 off_t=None if b is None else int(np.searchsorted(ra, b)),
                                 done_t=None if d is None else int(np.searchsorted(ra, d))))
            row["score"] = json.dumps(sc)
            row["score_match"] = json.dumps(cmp_)
            out.append(row)
    env.close()
    print(f"[task] {spec['task_id']} {len(demos)} demos done", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--texts", required=True, help="A0ASMAS/test.csv (offline idx_AS_v2 / idx_MAS_v2)")
    ap.add_argument("--mode", choices=("actions", "states"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None, help="task ids")
    ap.add_argument("--max_demos", type=int, default=None)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    spec = json.load(open(a.spec))["tasks"]
    demos = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(a.texts, newline="") as f:
        for r in csv.DictReader(f):
            demos[r["task"]][r["demo"]].append({k: r[k] for k in ("raw_step", "idx_AS_v2", "idx_MAS_v2")})
    jobs = []
    for task, dd in sorted(demos.items(), key=lambda kv: spec[kv[0]]["task_id"]):
        if a.only and spec[task]["task_id"] not in a.only:
            continue
        items = sorted(dd.items(), key=lambda kv: int(kv[0].split("_")[1]))[:a.max_demos]
        items = [(d, sorted(rows, key=lambda r: int(r["raw_step"]))) for d, rows in items]
        jobs.append((task, spec[task]["suite"], items, spec[task], a.mode, True))
    jobs.sort(key=lambda j: -sum(len(r) for _, r in j[2]))           # longest first
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    n = 0
    with mp.get_context("spawn").Pool(a.workers) as pool, open(tmp, "w", newline="") as f:
        w = None
        for rows in pool.imap_unordered(replay_task, jobs):
            for r in rows:
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(r))
                    w.writeheader()
                w.writerow(r)
                n += 1
            f.flush()
    os.replace(tmp, a.out)
    print(f"[done] {n} demos -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
