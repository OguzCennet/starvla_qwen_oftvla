"""Causal oracle step switcher for closed-loop AS / MAS rollouts (M15).

At every control step it reads the simulator state and advances from block k to k+1 when the event that ends
block k has occurred. The events are the offline step1 events (language_experiments/analysis/step1_subtasks.py),
computed with the SAME functions on the history seen so far (past and current states only):

  kind            offline (step1)                                  causal here
  close_cmd       rule B: last close before the next lift onset    first close command at/after `after`
  open_cmd        last open in the carry (apply_dict_v2 split)     first open command at/after `after`
  close_or_touch  first close in the first contact episode, else   first of (close command, first touch)
                  the touch
  first_touch     first robot contact                              same (immediate)
  lift            lift_onset (z > rest + 5 mm for HOLD=3 steps,    same; confirmed at onset + 2
                  gripper closed 3 steps before)
  last_contact    end of the first contact episode ending at/after same for objects; for parts rule A (merge
                  `after` (gaps <= 2 merged); parts: rule A         episodes up to the next event) is dropped;
                                                                    confirmed after 3 steps without contact
Guard (M15 decision 2): a close / open command counts only if the gripper is within GUARD_M of the block's guard
object at that step (distance to the object's geoms, bounding spheres); ignored commands are logged.

Timing as offline: block k ends at event + END_PAD and the next starts at event + 1, the overlap goes to the
lower index, so the block switches at frame event + 3. Every rule above is confirmed at or before event + 3,
so the causal switch frame equals the offline one whenever the event is the same.

Frame convention (same as step1 on the recorded demos): state s_f is observed BEFORE action a_f is executed;
the gripper command of frame f is a_f[-1] (> 0 close, <= 0 open). Per frame:
    sw.observe(contacts_f, z_f, dists_f); k = sw.block(); ...choose a_f...; sw.act(a_f[-1])
Pure numpy + the offline step1 / sim_events functions (no LIBERO import); `SimObserver` reads a MuJoCo sim.
"""
import os
import sys

import numpy as np

for _p in ("~/VLA_FT/language_experiments/analysis", "~/VLA_FT/libero-subinstructions"):
    _p = os.path.expanduser(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sim_events import contact_episodes  # noqa: E402
from step1_subtasks import CONTACT_GAP, END_PAD, HOLD, lift_onset, switches  # noqa: E402

GUARD_M = 0.05
SWITCH_DELAY = END_PAD + 1        # label switches at event + 3
COMMANDS = {"close_cmd": 1, "open_cmd": -1}


class Switcher:
    """blocks: list of dicts with 'kind', 'obj', optional 'guard_obj', optional 'text' (spec of one task/setup).
    Blocks without text at the end (no text in any demo) are tracked for progress, but the text index stays at
    the last block with text (`text_block`)."""

    def __init__(self, blocks, guard_m=GUARD_M):
        self.blocks = blocks
        self.guard_m = guard_m
        self.n_text = sum(1 for b in blocks if b.get("text") is not None) or len(blocks)
        self.k = 0                     # current block
        self.after = 0                 # events of block k are searched at/after this frame
        self.f = -1                    # last observed frame
        self.contacts, self.z, self.dists, self.grip = {}, {}, {}, []
        self.switch_frames = []        # frame at which block i+1 starts (= event_i + 3)
        self.events = []               # (block, kind, obj, event frame, confirm frame)
        self.ignored = []              # (block, kind, frame, guard obj, distance) commands rejected by the guard
        self._ignored_seen = set()

    # ---------------------------------------------------------------- inputs
    def observe(self, contacts, z, dists):
        """State of frame f = self.f + 1 (before its action). contacts: {name: bool} for objects and parts,
        z: {object: height}, dists: {guard object: gripper distance in m}."""
        self.f += 1
        assert len(self.grip) == self.f, "act() must be called once per frame, after observe()"
        for d, src in ((self.contacts, contacts), (self.z, z), (self.dists, dists)):
            for k, v in src.items():
                d.setdefault(k, []).append(v)
        self._advance()

    def act(self, grip_cmd):
        """Gripper command executed at the current frame (> 0 close, <= 0 open)."""
        assert len(self.grip) == self.f
        self.grip.append(float(grip_cmd))

    def block(self):
        """Block index for the current frame (switches with frame <= f applied)."""
        return sum(1 for s in self.switch_frames if s <= self.f)

    def text_block(self):
        return min(self.block(), self.n_text - 1)

    # ---------------------------------------------------------------- events
    def _arr(self, d, name, dtype=float):
        return np.asarray(d[name], dtype=dtype)

    def _command(self, b, i):
        """First guarded command of the block's kind at/after `after` in the gripper history (None if none)."""
        sign = COMMANDS[b["kind"]] if b["kind"] in COMMANDS else 1
        grip = np.asarray(self.grip)
        if len(grip) < 2:
            return None
        guard = b.get("guard_obj")
        for t in switches(grip, sign):
            if t < self.after:
                continue
            d = self.dists[guard][t] if guard else 0.0
            if d <= self.guard_m:
                return t
            if (i, t) not in self._ignored_seen:
                self._ignored_seen.add((i, t))
                self.ignored.append((i, b["kind"], t, guard, round(float(d), 4)))
        return None

    def _episodes(self, obj):
        c = self._arr(self.contacts, obj, bool)
        return c, [ep for ep in contact_episodes(c, CONTACT_GAP) if ep[1] >= self.after]

    def _event(self, i):
        """(event frame, confirm frame) of block i if it has occurred in the history, else None."""
        b, f = self.blocks[i], self.f
        kind, obj = b["kind"], b.get("obj")
        if kind in COMMANDS:
            t = self._command(b, i)
            return (t, t) if t is not None else None
        if obj not in self.contacts:
            return None
        c, eps = self._episodes(obj)
        if kind == "first_touch":
            return (max(eps[0][0], self.after),) * 2 if eps else None
        if kind == "close_or_touch":
            cands = [x for x in (self._command(b, i), max(eps[0][0], self.after) if eps else None) if x is not None]
            return (min(cands),) * 2 if cands else None
        if kind == "last_contact":
            if not eps:
                return None
            e = eps[0][1]
            # no contact in e+1 .. e+CONTACT_GAP+1: nothing can merge into this episode any more
            return (e, e + CONTACT_GAP + 1) if f - e >= CONTACT_GAP + 1 else None
        if kind == "lift":
            z = self._arr(self.z, obj)
            grip = np.asarray(self.grip + [0.0])   # lift_onset reads grip[t-3:t+1] with t <= f - HOLD + 1 < f
            for ep in eps:
                t = lift_onset(z, c, ep, grip)
                if t is not None and t >= self.after:
                    return t, t + HOLD - 1
            return None
        raise ValueError(kind)

    def _advance(self):
        while self.k < len(self.blocks):
            ev = self._event(self.k)
            if ev is None:
                break
            t, conf = ev
            b = self.blocks[self.k]
            self.events.append((self.k, b["kind"], b.get("obj"), int(t), int(conf)))
            if self.k < len(self.blocks) - 1:
                self.switch_frames.append(int(t) + SWITCH_DELAY)
            self.after = int(t) + 1
            self.k += 1

    # ---------------------------------------------------------------- summary
    def ended(self):
        """Per block: True if its ending event occurred."""
        done = {e[0] for e in self.events}
        return [i in done for i in range(len(self.blocks))]


def first_lifts(contacts, z, grip, objects):
    """{object: first lift onset} over a whole episode (logging only: first object grasped, wrong grasps)."""
    out = {}
    grip = np.asarray(grip, dtype=float)
    for o in objects:
        c = np.asarray(contacts[o], dtype=bool)
        for ep in contact_episodes(c, CONTACT_GAP):
            t = lift_onset(np.asarray(z[o], dtype=float), c, ep, grip)
            if t is not None:
                out[o] = int(t)
                break
    return out


class SimObserver:
    """Reads one frame from a MuJoCo sim with the offline definitions: movable objects and articulated parts as in
    sim_segment.movable_objects / articulated_parts, robot = sim_trace.robot_bodies, contact = any robot body
    touching the object / part subtree (the loop of sim_segment.record), z = body_xpos[:, 2].
    Gripper distance (guard): from the gripper0 grip site to the nearest geom bounding sphere of the object."""

    def __init__(self, sim, guard_objs=()):
        import sim_segment as S
        from sim_trace import robot_bodies
        self.sim = sim
        self.robot = robot_bodies(sim)
        self.objs = S.movable_objects(sim, self.robot)
        self.parts = S.articulated_parts(sim)
        self.site = sim.model.site_name2id("gripper0_grip_site")
        bodies = {**{k: v[1] for k, v in self.objs.items()}, **{k: v[1] for k, v in self.parts.items()}}
        missing = [o for o in guard_objs if o not in bodies]
        assert not missing, f"guard objects not in the scene: {missing}"
        self.guard_geoms = {o: np.array([g for g in range(sim.model.ngeom) if int(sim.model.geom_bodyid[g]) in bodies[o]])
                            for o in guard_objs}

    def observe(self):
        sim = self.sim
        contacts = {k: False for k in (*self.objs, *self.parts)}
        for i in range(sim.data.ncon):
            con = sim.data.contact[i]
            b1 = int(sim.model.geom_bodyid[con.geom1])
            b2 = int(sim.model.geom_bodyid[con.geom2])
            if (b1 in self.robot) == (b2 in self.robot):
                continue
            other = b2 if b1 in self.robot else b1
            for k, (_, bodies) in self.objs.items():
                if other in bodies:
                    contacts[k] = True
            for k, (_, bodies) in self.parts.items():
                if other in bodies:
                    contacts[k] = True
        z = {k: float(sim.data.body_xpos[b][2]) for k, (b, _) in self.objs.items()}
        p = np.asarray(sim.data.site_xpos[self.site])
        dists = {}
        for o, geoms in self.guard_geoms.items():
            gp = np.asarray(sim.data.geom_xpos)[geoms]
            r = np.asarray(sim.model.geom_rbound)[geoms]
            dists[o] = float(np.maximum(0.0, np.linalg.norm(gp - p, axis=1) - r).min())
        return contacts, z, dists
