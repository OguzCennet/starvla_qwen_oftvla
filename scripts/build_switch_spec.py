"""Build the closed-loop switch spec: per E3 test task, the canonical AS / MAS block sequence and, per block,
the event that ends it (kind + simulator object), for the oracle switcher (oftvla/switcher.py).

Canonical sequence (M15 decision 1): per task and setup the most common ordered sequence of distinct blocks
(text_AS / text_MAS of A0ASMAS/test.csv) across the task's test demos, extended to the longest variant it is a
prefix of (a demo that ends before the release has no final put block). Variants that are neither a prefix nor
an extension are counted and ignored.

Events: block i of AS = Act sub-task i, block i of MAS = MoveAct sub-task i (unmerged) of
language_experiments/data/libero100_subtasks.json, parsed with step1_subtasks.parse_event. Simulator objects:
most common object of that sub-task's event in runs/step1/subtask_steps_final.csv over the task's test demos
(MoveAct rows there merge carry + put: "last_contact:...;open_cmd=" is expanded back into open_cmd + last_contact).
Causal kinds (M15 decision 2): close_cmd -> first close, open_cmd -> first open, last_contact on a part -> end of
the first contact episode (rule A dropped), close_or_touch -> first of close / touch. Guard object of a
close / open command: the object the step acts on (close: the object picked next; open: the carried object).

Run: source scripts/common.sh; run_py scripts/build_switch_spec.py --out <new json>   (refuses to overwrite)
"""
import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/VLA_FT/language_experiments/analysis"))
from step1_subtasks import parse_event  # noqa: E402  (the offline event parser)

TEXTS = os.path.expanduser("~/VLA_FT/DataSamples/Libero/A0ASMAS/test.csv")
RULES = os.path.expanduser("~/VLA_FT/language_experiments/data/libero100_subtasks.json")
STEPS = "/hnvme/workspace/y307bd13-vgv/runs/step1/subtask_steps_final.csv"
TASKS = os.path.expanduser("~/VLA_FT/DataSamples/Libero/Splits/e3_test_tasks.txt")
csv.field_size_limit(sys.maxsize)


def in_progress(text):
    return next(l for l in text.split("\n") if l.startswith("In-Progress Step: "))[len("In-Progress Step: "):]


def canonical(seqs):
    """seqs: Counter of tuples -> (canonical tuple, n following a prefix of it / itself, n other, n most common)."""
    top, n_top = seqs.most_common(1)[0]
    ext = [s for s in seqs if s[:len(top)] == top]
    canon = max(ext, key=len)
    n_pref = sum(n for s, n in seqs.items() if canon[:len(s)] == s)
    return canon, n_pref, sum(seqs.values()) - n_pref, n_top


def expand_moveact(rows):
    """step1 MoveAct rows of one demo -> [(kind, obj)] per unmerged MoveAct sub-task."""
    out = []
    for r in sorted(rows, key=lambda r: int(r["idx"])):
        kind, rest = r["event"].split(":", 1)
        obj = rest.split("->", 1)[1].split(";")[0]
        if ";open_cmd=" in r["event"]:
            out += [("open_cmd", obj), (kind, obj)]
        else:
            out.append((kind, obj))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    groups = {l.split()[1]: (l.split()[0], l.split()[2]) for l in open(TASKS) if l.strip()}
    rules = {t["task_name"]: t for t in json.load(open(RULES))["tasks"]}

    texts = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(TEXTS, newline="") as f:
        for r in csv.DictReader(f):
            texts[r["task"]][r["demo"]].append(r)
    steps = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(STEPS, newline="") as f:
        for r in csv.DictReader(f):
            steps[(r["task"], r["setup"])][r["demo"]].append(r)

    spec, report = {}, []
    for task, demos in sorted(texts.items(), key=lambda kv: next(iter(kv[1].values()))[0]["task_id"]):
        first = next(iter(demos.values()))[0]
        tid, suite = first["task_id"], first["suite"]
        assert groups[task][0] == suite, task
        a0 = {r["text_A0"] for rs in demos.values() for r in rs}
        assert len(a0) == 1, (task, a0)
        rule = rules[task]
        entry = dict(task_id=tid, suite=suite, task=task, group=groups[task][1], text_A0=a0.pop())
        for setup, col, idxcol, rkey in (("AS", "text_AS", "idx_AS_v2", "Act"), ("MAS", "text_MAS", "idx_MAS_v2", "MoveAct")):
            seqs = collections.Counter()
            idx_of = {}
            for demo, rs in demos.items():
                rs = sorted(rs, key=lambda r: int(r["raw_step"]))
                seq = []
                for r in rs:
                    if not seq or seq[-1] != r[col]:
                        seq.append(r[col])
                    assert idx_of.setdefault(r[col], int(r[idxcol])) == int(r[idxcol]), (task, col)
                seqs[tuple(seq)] += 1
            canon, n_pref, n_other, n_top = canonical(seqs)
            subs = rule[rkey]
            # A trailing sub-task may have no text in any test demo (every demo ends before its rows start):
            # no new strings, the last canonical block then runs to the end; its event is still tracked (progress).
            assert len(canon) <= len(subs), (task, setup, len(subs), len(canon))
            if len(canon) < len(subs):
                report.append(f"{tid} {setup}: no text for trailing sub-task(s) {[s['label'] for s in subs[len(canon):]]}"
                              f" in any test demo -> {len(canon)} blocks, events tracked for all {len(subs)}")
            assert [idx_of[t] for t in canon] == list(range(len(canon))), (task, setup, [idx_of[t] for t in canon])
            # simulator objects of each sub-task's event, most common over the task's test demos
            objs = [collections.Counter() for _ in subs]
            for demo in demos:
                rows = steps[(task, rkey)].get(demo)
                assert rows, (task, rkey, demo)
                ev = (expand_moveact(rows) if rkey == "MoveAct" else
                      [(r["event"].split(":", 1)[0], r["event"].split("->", 1)[1]) for r in sorted(rows, key=lambda r: int(r["idx"]))])
                if len(ev) != len(subs):
                    continue
                for i, (k, o) in enumerate(ev):
                    objs[i][o] += 1
            blocks = []
            for i, s in enumerate(subs):
                text = canon[i] if i < len(canon) else None
                kind, phrase = parse_event(s["ends"])
                obj, n_obj = objs[i].most_common(1)[0]
                blocks.append(dict(idx=i, text=text, label=s["label"], step=in_progress(text) if text else None,
                                   ends=s["ends"], kind=kind, obj=obj, obj_share=f"{n_obj}/{sum(objs[i].values())}",
                                   last=(i == len(subs) - 1)))
            # guard objects of commands: close -> the object of the next (pick) block, open -> the carried object
            for i, b in enumerate(blocks):
                if b["kind"] == "close_cmd":   # step1 may record no object for a close command (fallback)
                    b["guard_obj"] = b["obj"] = blocks[i + 1]["obj"]
                elif b["kind"] in ("open_cmd", "close_or_touch"):
                    b["guard_obj"] = b["obj"]
            entry[setup] = blocks
            report.append(f"{tid} {setup}: {len(canon)} blocks, prefix-consistent demos {n_pref}/{len(demos)} "
                          f"(most common {n_top}, other {n_other})")
        spec[task] = entry
    json.dump(dict(source=dict(texts=TEXTS, rules=RULES, steps=STEPS), tasks=spec), open(a.out, "w"), indent=1)
    print("\n".join(report))
    for task, e in spec.items():
        print(f"== {e['task_id']} {e['group']}")
        for setup in ("AS", "MAS"):
            for b in e[setup]:
                print(f"   {setup:3s} {b['idx']} {(b['step'] or '(no text: ' + b['label'] + ')')[:55]:55s} | {b['kind']:14s} {b['obj']:28s} {b['obj_share']:6s}"
                      f" guard={b.get('guard_obj', '')}")


if __name__ == "__main__":
    main()
