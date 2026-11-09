"""Unit tests for the causal oracle switcher (oftvla/switcher.py), synthetic traces, no simulator.

Run (LIBERO container, which has pytest and the offline step1 modules' dependencies):
  apptainer exec $LIBERO_SIF python3 -m pytest -q tests/test_switcher.py
"""
import numpy as np
import pytest

from oftvla.switcher import Switcher, first_lifts
from step1_subtasks import find_event
from sim_events import contact_episodes

OBJ, OTHER, PART = "bowl_1", "plate_1", "drawer_1"
PICK_PLACE = [dict(kind="close_cmd", obj=OBJ, guard_obj=OBJ, text="reach"),
              dict(kind="lift", obj=OBJ, text="grasp"),
              dict(kind="open_cmd", obj=OBJ, guard_obj=OBJ, text="carry"),
              dict(kind="last_contact", obj=OBJ, text="put")]


def trace(T=80, close=10, touch=12, lift=20, open_=40, release=45, near=True):
    """Pick and place of OBJ: gripper closes at `close`, contact from `touch` to `release`, object rises from
    `lift` until `open_`, gripper opens at `open_`. Returns per-frame contacts, z, dists, grip."""
    grip = np.full(T, -1.0)
    grip[close:open_] = 1.0
    c = np.zeros(T, bool)
    c[touch:release + 1] = True
    z = np.full(T, 0.9)
    z[lift:open_] = 0.9 + 0.01 * np.arange(1, open_ - lift + 1)
    d = np.full(T, 0.01 if near else 0.2)
    return dict(contacts={OBJ: c, OTHER: np.zeros(T, bool)}, z={OBJ: z, OTHER: np.full(T, 0.9)},
                dists={OBJ: d}, grip=grip)


def run(blocks, tr, guard_m=0.05):
    sw = Switcher(blocks, guard_m=guard_m)
    T = len(tr["grip"])
    per_frame = []
    for f in range(T):
        sw.observe({k: bool(v[f]) for k, v in tr["contacts"].items()}, {k: float(v[f]) for k, v in tr["z"].items()},
                   {k: float(v[f]) for k, v in tr["dists"].items()},
                   {k: bool(v[f]) for k, v in tr.get("goals", {}).items()})
        per_frame.append(sw.block())
        sw.act(tr["grip"][f])
    return sw, per_frame


def test_pick_place_switches_at_event_plus_3():
    sw, blocks = run(PICK_PLACE, trace())
    assert sw.switch_frames == [13, 23, 43]           # close 10, lift onset 20, open 40 (+3 each)
    assert [e[3] for e in sw.events] == [10, 20, 40, 45]   # last block's event (release 45) tracked, no switch
    assert blocks[12] == 0 and blocks[13] == 1 and blocks[22] == 1 and blocks[23] == 2 and blocks[43] == 3
    assert sw.ended() == [True, True, True, True]


def test_matches_offline_events_on_full_trace():
    """Causal lift onset / last contact = step1 find_event on the full arrays."""
    tr = trace()
    sw, _ = run(PICK_PLACE, tr)
    c, z, grip = tr["contacts"][OBJ], tr["z"][OBJ], tr["grip"]
    eps_of = lambda o: contact_episodes(c, 2)
    assert find_event("lift", OBJ, 11, grip, {OBJ: c}, {OBJ: z}, eps_of) == sw.events[1][3]
    assert find_event("last_contact", OBJ, 41, grip, {OBJ: c}, {OBJ: z}, eps_of) == sw.events[3][3]


def test_confirmation_is_causal():
    """A block never switches before its event is visible: lift confirmed at onset + 2, release at + 3."""
    sw, _ = run(PICK_PLACE, trace())
    lift = next(e for e in sw.events if e[1] == "lift")
    rel = next(e for e in sw.events if e[1] == "last_contact")
    assert lift[4] == lift[3] + 2 and rel[4] == rel[3] + 3
    for _, _, _, t, conf in sw.events:
        assert conf <= t + 3


def test_prefix_independence():
    """Block indices up to frame f do not depend on frames after f (truncated trace gives the same prefix)."""
    tr = trace()
    _, full = run(PICK_PLACE, tr)
    for cut in (15, 30, 44, 50):
        short = {k: ({o: v[:cut] for o, v in tr[k].items()} if isinstance(tr[k], dict) else tr[k][:cut]) for k in tr}
        _, part = run(PICK_PLACE, short)
        assert part == full[:cut]


def test_guard_ignores_far_commands_and_logs_them():
    tr = trace()
    tr["dists"][OBJ][:9] = 0.2                    # an early close/open/close far from the bowl
    tr["grip"][3:5] = 1.0
    sw, _ = run(PICK_PLACE, tr)
    assert sw.switch_frames[0] == 13               # the far close at 3 is ignored, the close at 10 counts
    assert sw.ignored == [(0, "close_cmd", 3, OBJ, 0.2)]
    sw2, _ = run(PICK_PLACE, tr, guard_m=float("inf"))
    assert sw2.switch_frames[0] == 6 and sw2.ignored == []


def test_contact_gap_of_two_merges_three_ends():
    tr = trace()
    c = tr["contacts"][OBJ]
    c[46:] = False
    c[48:50] = True                               # gap 46-47 (2 steps): same episode, ends at 49
    sw, _ = run(PICK_PLACE, tr)
    assert sw.events[3][3] == 49
    c[48:50] = False
    c[49:51] = True                               # gap 46-48 (3 steps): episode ended at 45
    sw, _ = run(PICK_PLACE, tr)
    assert sw.events[3][3] == 45


def test_push_with_open_gripper_is_not_a_lift():
    tr = trace()
    tr["grip"][:] = -1.0                          # never closed: the rise is a push
    blocks = [dict(kind="lift", obj=OBJ, text="grasp"), dict(kind="last_contact", obj=OBJ, text="put")]
    sw, _ = run(blocks, tr)
    assert sw.events == [] and sw.switch_frames == []


def test_wrong_object_lift_does_not_advance():
    tr = trace()
    tr["contacts"] = {OTHER: tr["contacts"][OBJ], OBJ: np.zeros(80, bool)}     # the plate is grasped instead
    tr["z"] = {OTHER: tr["z"][OBJ], OBJ: np.full(80, 0.9)}
    blocks = [dict(kind="lift", obj=OBJ, text="grasp"), dict(kind="last_contact", obj=OBJ, text="put")]
    sw, per_frame = run(blocks, tr)
    assert sw.switch_frames == [] and set(per_frame) == {0}
    assert first_lifts(tr["contacts"], tr["z"], tr["grip"], [OBJ, OTHER]) == {OTHER: 20}


def test_part_events_first_touch_and_first_episode_end():
    T = 60
    c = np.zeros(T, bool)
    c[10:20] = True
    c[30:40] = True                               # second contact episode (re-grasp): rule A dropped
    blocks = [dict(kind="first_touch", obj=PART, text="reach"), dict(kind="last_contact", obj=PART, text="open")]
    tr = dict(contacts={PART: c}, z={}, dists={}, grip=np.full(T, -1.0))
    sw, _ = run(blocks, tr)
    assert sw.switch_frames == [13] and [e[3] for e in sw.events] == [10, 19]


def test_close_or_touch_option1():
    """Close before / during the first contact counts at once; without a close the touch, confirmed at the
    contact end (switch frame = contact end + 3, not touch + 3)."""
    T = 50
    c = np.zeros(T, bool)
    c[15:30] = True
    blocks = [dict(kind="close_or_touch", obj=PART, guard_obj=PART, text="reach"),
              dict(kind="last_contact", obj=PART, text="turn")]
    tr = lambda grip: dict(contacts={PART: c}, z={}, dists={PART: np.zeros(T)}, grip=grip)
    grip = np.full(T, -1.0)
    grip[20:] = 1.0                               # touch at 15, close at 20 inside the contact: the close
    sw, _ = run(blocks, tr(grip))
    assert sw.events[0][3:] == (20, 20) and sw.switch_frames == [23]
    grip = np.full(T, -1.0)
    grip[12:] = 1.0                               # close at 12 before the touch
    sw, _ = run(blocks, tr(grip))
    assert sw.events[0][3] == 12 and sw.switch_frames == [15]
    grip = np.full(T, -1.0)                       # push: never closed -> the touch, confirmed at 29 + 3
    sw, _ = run(blocks, tr(grip))
    assert sw.events[0][3:] == (15, 32) and sw.switch_frames == [32]


def test_part_end_option2b():
    """Operate end = end of the contact in progress when the part's goal first holds; a re-grasp before the goal
    does not end the step; goal reached after the release -> the last contact before it."""
    T = 80
    c = np.zeros(T, bool)
    c[10:20] = True
    c[30:45] = True
    goal = np.zeros(T, bool)
    goal[38:] = True                              # goal holds during the second contact
    blocks = [dict(kind="last_contact", obj=PART, text="open"), dict(kind="close_cmd", obj=OBJ, guard_obj=OBJ, text="reach")]
    tr = dict(contacts={PART: c, OBJ: np.zeros(T, bool)}, z={OBJ: np.full(T, 0.9)}, dists={OBJ: np.zeros(T)},
              grip=np.full(T, -1.0), goals={PART: goal})
    sw, per_frame = run(blocks, tr)
    assert sw.events[0][3:] == (44, 47) and sw.switch_frames == [47] and per_frame[46] == 0 and per_frame[47] == 1
    goal[:] = False
    goal[50:] = True                              # goal only after the release (part keeps moving)
    sw, _ = run(blocks, tr)
    assert sw.events[0][3:] == (44, 50) and sw.switch_frames == [50]


def test_text_block_clipped_for_trailing_blocks_without_text():
    blocks = [dict(b) for b in PICK_PLACE]
    blocks[3]["text"] = None                      # e.g. L90T79: no put text in any test demo
    sw, _ = run(blocks, trace())
    assert sw.block() == 3 and sw.text_block() == 2 and sw.n_text == 3


def test_act_observe_order_enforced():
    sw = Switcher(PICK_PLACE)
    sw.observe({OBJ: False}, {OBJ: 0.9}, {OBJ: 0.0})
    with pytest.raises(AssertionError):
        sw.observe({OBJ: False}, {OBJ: 0.9}, {OBJ: 0.0})
