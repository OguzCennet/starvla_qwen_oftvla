"""MAS-level progress scorer for closed-loop episodes (M15), separate from the switcher.

World state only (never the model's gripper command); runs identically for A0, AS and MAS. Per canonical MAS
step of the task (all MoveAct steps of the switch spec, including a trailing step without text), the step is
DONE at the first frame (at or after the previous step's done frame) where, for the step's target object:

  reach    gripper (grip site) within 5 cm of the object / part (nearest geom bounding sphere) or touching it
  grasp    the L rule on the both-fingers hold: held (a left AND a right finger body touch the object) and
           z > z(start of the hold) + 5 mm for 3 steps (step1_subtasks.lift_onset, gripper not used); frame = onset
  carry    held and inside / above the destination of the object's goal atom: site region = within the site box
           world-axis footprint (rotated box projected, ext = abs(R) size) and above its bottom; object = within the destination's horizontal radius and
           above its body
  put      the object's LIBERO goal predicate is true and no robot body touches the object, OR the episode is
           successful at that frame (rollouts and demos end at success, often with the gripper still on it)
  operate  the part's goal predicate (the task's open / close / turnon / turnoff atom) is true

Stacks (L90T65 "the stacked bowls"): for steps whose label names "stacked", held = a left finger on some object and a right finger on some object of
the stack (the target plus every x with an 'on x target' goal atom); the lift is measured on the target.

Episode log: done flag and first frame per step, furthest step reached (number of consecutive done steps),
first object grasped (name, frame, correct = the first grasp step's object), steps used, success.
"""
import numpy as np

from oftvla.switcher import CONTACT_GAP, SimObserver, contact_episodes, lift_onset  # noqa: F401

REACH_M = 0.05
TYPES = {"close_cmd": "reach", "first_touch": "reach", "close_or_touch": "reach", "lift": "grasp",
         "open_cmd": "carry"}
UNARY = ("open", "close", "turnon", "turnoff")


def step_type(block, parts):
    if block["kind"] == "last_contact":
        return "operate" if block["obj"] in parts else "put"
    return TYPES[block["kind"]]


class WorldObserver(SimObserver):
    """SimObserver + the world features of the scorer: goal atoms (LIBERO predicates), destination test,
    part goal predicates. env: the LIBERO ControlEnv / OffScreenRenderEnv (env.env is the BDDL env)."""

    def __init__(self, env, mas_blocks, guard_objs=()):
        import sys, os
        sys.path.insert(0, os.path.expanduser("~/VLA_FT/libero-subinstructions"))
        from sim_trace import goal_atoms
        self.inner = env.env
        targets = {b["obj"] for b in mas_blocks}
        super().__init__(self.inner.sim, sorted(set(guard_objs) | targets))
        self.atoms = goal_atoms(self.inner)
        objs = {b["obj"] for b in mas_blocks if b["obj"] in self.objs}
        self.part_objs = {b["obj"] for b in mas_blocks if b["obj"] in self.parts}
        self.atom_of = {}
        for o in objs:
            idx = [i for i, a in enumerate(self.atoms) if len(a) == 3 and a[1] == o]
            assert len(idx) == 1, (o, self.atoms)
            self.atom_of[o] = idx[0]
        unary = [i for i, a in enumerate(self.atoms) if a[0].lower() in UNARY]
        assert len(unary) <= 1 and (unary or not self.part_objs), self.atoms
        self.part_atom = unary[0] if unary else None
        # stacks: target -> objects with an 'on x target' goal atom (L90T65: right bowl on the left bowl)
        self.stack_of = {o: [a[1] for a in self.atoms if a[0].lower() == "on" and len(a) == 3 and a[2] == o]
                         for o in objs}
        sim = self.sim
        self.dest = {}
        for o, i in self.atom_of.items():
            d = self.atoms[i][2]
            if self.inner.object_states_dict[d].object_state_type == "site":
                self.dest[o] = ("site", d, np.asarray(self.inner.object_sites_dict[d].size, dtype=float))
            else:
                b, bodies = self.objs[d]
                geoms = [g for g in range(sim.model.ngeom) if int(sim.model.geom_bodyid[g]) in bodies]
                gp = np.asarray(sim.data.geom_xpos)[geoms][:, :2]
                r = np.linalg.norm(gp - np.asarray(sim.data.body_xpos[b])[:2], axis=1) + np.asarray(sim.model.geom_rbound)[geoms]
                self.dest[o] = ("object", b, float(r.max()))

    def _in_dest(self, o):
        sim = self.sim
        p = np.asarray(sim.data.body_xpos[self.objs[o][0]])
        kind, d, x = self.dest[o]
        if kind == "site":
            c = np.asarray(sim.data.get_site_xpos(d))
            m = np.asarray(sim.data.get_site_xmat(d)).reshape(3, 3)
            ext = np.abs(m) @ x               # world half extents of the (possibly rotated) site box
            return bool(abs(p[0] - c[0]) <= ext[0] and abs(p[1] - c[1]) <= ext[1] and p[2] >= c[2] - ext[2])
        q = np.asarray(sim.data.body_xpos[d])
        return bool(np.linalg.norm(p[:2] - q[:2]) <= x and p[2] >= q[2])

    def world(self):
        """Call after observe(): {atoms, goal_obj, part_goal, dest, held, held_stack} of the same frame."""
        vals = [bool(self.inner._eval_predicate(a)) for a in self.atoms]
        return dict(atoms=vals, goal_obj={o: vals[i] for o, i in self.atom_of.items()},
                    part_goal={p: vals[self.part_atom] for p in self.part_objs},
                    dest={o: self._in_dest(o) for o in self.atom_of}, held=dict(self.held),
                    # a left finger on some object of the stack and a right finger on some object of the stack
                    # (L90T65: the fingers pinch the two nested rims, one finger on each bowl)
                    held_stack={o: any(self.left_on[x] for x in [o, *xs]) and any(self.right_on[x] for x in [o, *xs])
                                for o, xs in self.stack_of.items()})


class Scorer:
    """blocks: MAS blocks of the switch spec. parts: names of articulated parts in the scene."""

    def __init__(self, blocks, parts):
        self.steps = [(step_type(b, parts), b["obj"]) for b in blocks]
        # steps on a stack (label names "stacked"): held = both fingers on any object of the stack
        self.held_key = ["held_stack" if "stacked" in (b.get("label") or "") else "held" for b in blocks]
        self.h = {}                      # feature -> {name: [per frame]}
        self.n = 0

    def observe(self, contacts, z, dists, world):
        for key, d in (("contacts", contacts), ("z", z), ("dists", dists), ("held", world["held"]),
                       ("held_stack", world.get("held_stack", {})),
                       ("goal_obj", world["goal_obj"]), ("part_goal", world["part_goal"]), ("dest", world["dest"])):
            for k, v in d.items():
                self.h.setdefault(key, {}).setdefault(k, []).append(v)
        self.h.setdefault("success", {}).setdefault("all", []).append(all(world["atoms"]))
        self.n += 1

    def _a(self, key, name, dtype=bool):
        return np.asarray(self.h[key][name], dtype=dtype)

    def grasp_onsets(self, o, start=0, key="held"):
        """L rule on the both-fingers hold (key: held / held_stack), lift of o itself; first onset >= start."""
        held = self._a(key, o)
        z = self._a("z", o, float)
        for ep in contact_episodes(held, CONTACT_GAP):
            if ep[1] < start:
                continue
            t = lift_onset(z, held, ep, None)
            if t is not None and t >= start:
                return t
        return None

    def cond(self, typ, o, key="held"):
        if typ == "reach":
            return self._a("contacts", o) | (self._a("dists", o, float) <= REACH_M)
        if typ == "carry":
            return self._a(key, o) & self._a("dest", o)
        if typ == "put":
            return (self._a("goal_obj", o) & ~self._a("contacts", o)) | self._a("success", "all")
        if typ == "operate":
            return self._a("part_goal", o)
        raise ValueError(typ)

    def result(self):
        done, prev = [], 0
        for (typ, o), key in zip(self.steps, self.held_key):
            if prev is None:
                done.append(None)
                continue
            if typ == "grasp":
                t = self.grasp_onsets(o, prev, key)
            else:
                hit = np.nonzero(self.cond(typ, o, key)[prev:])[0]
                t = int(prev + hit[0]) if len(hit) else None
            done.append(t)
            prev = t
        furthest = next((i for i, t in enumerate(done) if t is None), len(done))
        firsts = [(t, o) for o in self.h.get("held", {}) if (t := self.grasp_onsets(o)) is not None]
        first = min(firsts) if firsts else None
        target = next((o for typ, o in self.steps if typ == "grasp"), None)
        succ = self.h["success"]["all"]
        return dict(types=[t for t, _ in self.steps], done=done, furthest=furthest,
                    first_grasp=None if first is None else dict(obj=first[1], frame=int(first[0]), correct=first[1] == target),
                    steps=self.n, success=int(any(succ)), success_frame=next((i for i, v in enumerate(succ) if v), None))
