"""Validation table for the oracle switcher (M15 Step 2) from replay_switcher CSVs.

Per setup and step type (the type of the block that ENDS at the switch): offline switches, median offset
(switcher - offline, v1 frames, over switches the switcher made), share within +-1 / +-3 frames (over ALL offline
switches; a missed switch counts as outside), missed; per setup the share of demos with the same number of
switches; the effect of the 5 cm guard; the record() contact check; replay success (actions mode).

Run: source scripts/common.sh; run_py scripts/switcher_report.py CSV [CSV ...]
"""
import collections
import csv
import json
import statistics
import sys

csv.field_size_limit(sys.maxsize)
NAMES = {("MAS", "close_cmd"): "reach -> grasp", ("MAS", "first_touch"): "reach -> operate (drawer)",
         ("MAS", "close_or_touch"): "reach -> operate (knob)", ("MAS", "lift"): "grasp -> carry",
         ("MAS", "open_cmd"): "carry -> put", ("MAS", "last_contact", False): "put -> next",
         ("MAS", "last_contact", True): "operate -> next", ("AS", "lift"): "pick -> put",
         ("AS", "last_contact", False): "put -> next pick", ("AS", "last_contact", True): "operate -> next"}


def name(setup, m):
    return NAMES.get((setup, m["kind"], m["part"])) or NAMES[(setup, m["kind"])]


def table(rows, title):
    print(f"\n### {title} ({len(rows)} demos)")
    print("| setup | step type (ended) | offline switches | median offset | within ±1 | within ±3 | missed |")
    print("|---|---|---|---|---|---|---|")
    overall = {}
    for s in ("AS", "MAS"):
        by = collections.defaultdict(list)
        for r in rows:
            for m in json.loads(r[f"{s}_match"]):
                by[name(s, m)].append(None if m["sw_t"] is None else m["sw_t"] - m["off_t"])
        allv = []
        for k in sorted(by, key=lambda k: list(NAMES.values()).index(k)):
            v = by[k]
            allv += v
            hit = [x for x in v if x is not None]
            print(f"| {s} | {k} | {len(v)} | {statistics.median(hit) if hit else '-'} | "
                  f"{100 * sum(abs(x) <= 1 for x in hit) / len(v):.1f} % | {100 * sum(abs(x) <= 3 for x in hit) / len(v):.1f} % | "
                  f"{len(v) - len(hit)} |")
        hit = [x for x in allv if x is not None]
        overall[s] = (len(allv), 100 * sum(abs(x) <= 3 for x in hit) / len(allv))
        print(f"| {s} | **all** | {len(allv)} | {statistics.median(hit)} | {100 * sum(abs(x) <= 1 for x in hit) / len(allv):.1f} % | "
              f"**{overall[s][1]:.1f} %** | {len(allv) - len(hit)} |")
    for s in ("AS", "MAS"):
        same = sum(int(r[f"{s}_n_off"]) == int(r[f"{s}_n_sw"]) for r in rows)
        changed = sum(int(r[f"{s}_guard_changed"]) for r in rows)
        ign = sum(len(json.loads(r[f"{s}_ignored"])) for r in rows)
        print(f"{s}: same number of switches {same}/{len(rows)} = {100 * same / len(rows):.1f} %; "
              f"guard changed the switches in {changed} demos; commands ignored by the guard: {ign}")
    return overall


def main():
    for path in sys.argv[1:]:
        rows = list(csv.DictReader(open(path, newline="")))
        mode = rows[0]["mode"]
        print(f"\n## {path} (mode {mode})")
        rc = collections.Counter(r["record_check"] for r in rows if r["record_check"])
        if rc:
            print(f"record() contact/height check on the first demo of each task: {dict(rc)}")
        table(rows, "all demos")
        if mode == "actions":
            ok = [r for r in rows if r["success"] == "1"]
            print(f"\naction replay succeeds in {len(ok)}/{len(rows)} demos = {100 * len(ok) / len(rows):.1f} %")
            per = collections.Counter(r["task_id"] for r in rows if r["success"] != "1")
            print("failed replays per task:", dict(sorted(per.items())))
            table(ok, "demos whose action replay succeeds")


if __name__ == "__main__":
    main()
