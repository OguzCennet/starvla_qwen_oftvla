"""Unit tests for the MAS progress scorer (oftvla/scorer.py), synthetic world features, no simulator.

Run (LIBERO container): apptainer exec $LIBERO_SIF python3 -m pytest -q tests/test_scorer.py
"""
import numpy as np

from oftvla.scorer import Scorer

OBJ, OTHER, PART = "bowl_1", "plate_1", "drawer_1"
PICK_PLACE = [dict(kind="close_cmd", obj=OBJ), dict(kind="lift", obj=OBJ), dict(kind="open_cmd", obj=OBJ),
              dict(kind="last_contact", obj=OBJ)]


def feats(T=80, near=8, hold=(12, 45), lift=20, dest=(35, 50), goal=48, touch=(12, 45)):
    held = np.zeros(T, bool); held[hold[0]:hold[1] + 1] = True
    c = np.zeros(T, bool); c[touch[0]:touch[1] + 1] = True
    z = np.full(T, 0.9); z[lift:dest[1]] = 0.95
    d = np.full(T, 0.3); d[near:] = 0.03
    de = np.zeros(T, bool); de[dest[0]:dest[1]] = True
    g = np.zeros(T, bool); g[goal:] = True
    return dict(T=T, held={OBJ: held, OTHER: np.zeros(T, bool)}, c={OBJ: c, OTHER: np.zeros(T, bool)},
                z={OBJ: z, OTHER: np.full(T, 0.9)}, d={OBJ: d}, dest={OBJ: de}, goal={OBJ: g}, part={})


def run(blocks, f, parts=()):
    sc = Scorer(blocks, set(parts))
    for t in range(f["T"]):
        pick = lambda d: {k: v[t] for k, v in d.items()}
        world = dict(held=pick(f["held"]), goal_obj=pick(f["goal"]), part_goal=pick(f["part"]), dest=pick(f["dest"]),
                     atoms=[bool(v[t]) for v in list(f["goal"].values()) + list(f["part"].values())])
        sc.observe(pick(f["c"]), pick(f["z"]), pick(f["d"]), world)
    return sc.result()


def test_pick_place_all_done_in_order():
    r = run(PICK_PLACE, feats())
    assert r["types"] == ["reach", "grasp", "carry", "put"]
    assert r["done"] == [8, 20, 35, 48] and r["furthest"] == 4
    assert r["first_grasp"] == dict(obj=OBJ, frame=20, correct=True) and r["success"] == 1 and r["success_frame"] == 48


def test_put_needs_release_and_steps_are_sequential():
    r = run(PICK_PLACE, feats(goal=40, touch=(12, 55)))   # goal true while still touched: put waits for release
    assert r["done"][3] == 56
    r = run(PICK_PLACE, feats(hold=(12, 18)))              # dropped before the lift: no grasp, nothing after it
    assert r["done"][1:] == [None, None, None] and r["furthest"] == 1 and r["first_grasp"] is None


def test_wrong_object_first_grasp():
    f = feats()
    f["held"] = {OTHER: f["held"][OBJ], OBJ: np.zeros(80, bool)}
    f["z"] = {OTHER: f["z"][OBJ], OBJ: np.full(80, 0.9)}
    r = run(PICK_PLACE, f)
    assert r["first_grasp"] == dict(obj=OTHER, frame=20, correct=False) and r["done"][1] is None


def test_operate_uses_the_part_goal():
    T = 40
    goal = np.zeros(T, bool); goal[25:] = True
    d = np.full(T, 0.3); d[10:] = 0.0
    f = dict(T=T, held={}, c={PART: np.zeros(T, bool)}, z={}, d={PART: d}, dest={}, goal={}, part={PART: goal})
    r = run([dict(kind="first_touch", obj=PART), dict(kind="last_contact", obj=PART)], f, parts=[PART])
    assert r["types"] == ["reach", "operate"] and r["done"] == [10, 25] and r["success"] == 1
