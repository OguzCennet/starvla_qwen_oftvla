"""Closed-loop LIBERO evaluation client. Runs where LIBERO lives (sym-openvla-oft.sif), talks to oftvla.policy.

Follows StarVLA's LIBERO eval (examples/simBenchmarks/LIBERO/eval_files @ 985c223):
  * env rendered at 256x256, env.seed(7), episode starts from LIBERO's fixed init states
  * first 10 steps: dummy action [0]*6 + [-1] while objects settle
  * images: agentview and wrist rotated 180 deg, resized to 224x224 with PIL BILINEAR
  * a new 8-step chunk is requested every 8 steps and executed open-loop
  * gripper: model output > 0.5 ("open") -> -1, else +1 (LIBERO convention)
  * step limit: 400 (libero_90) / 520 (libero_10); success = env 'done'
Text: our A0 sentence for the task (most common among ok demos), i.e. exactly the training text.
Imports only numpy, PIL, imageio, LIBERO and oftvla.ipc (no torch model code).

Usage: python -m oftvla.libero_client --ids IDS.csv --labels_csv L.csv --out OUT.csv [--video_dir D]
"""

import argparse
import collections
import csv
import os
import pathlib
import time

import numpy as np
from PIL import Image

from oftvla import ipc

DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RESOLUTION = 256
MODEL_IMAGE_SIZE = (224, 224)
MAX_STEPS = {"libero_90": 400, "libero_10": 520}


def a0_texts(labels_csv: str) -> dict:
    """(suite, task) -> most common A0 sentence among ok demos (= training text)."""
    count = collections.defaultdict(collections.Counter)
    with open(labels_csv) as f:
        for r in csv.DictReader(f):
            if r["status"] == "ok":
                count[(r["suite"], r["task"])][r["A0"]] += 1
    return {k: v.most_common(1)[0][0] for k, v in count.items()}


def preprocess(image: np.ndarray) -> np.ndarray:
    """Rotate 180 deg (training convention) and resize to the model input size (bilinear, as StarVLA's client)."""
    img = np.ascontiguousarray(image[::-1, ::-1])
    return np.asarray(Image.fromarray(img).resize(MODEL_IMAGE_SIZE, Image.BILINEAR))


def binarize_gripper(open_value: float) -> float:
    """Model: 1 = open, 0 = close  ->  LIBERO: -1 = open, +1 = close."""
    return -1.0 if open_value > 0.5 else 1.0


def run_episode(env, init_state, lang, conn, chunk_len, max_steps, num_steps_wait, frames=None):
    env.reset()
    obs = env.set_init_state(init_state)
    t, step, chunk = 0, 0, None
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, _, _ = env.step(DUMMY_ACTION)
            t += 1
            continue
        main = preprocess(obs["agentview_image"])
        wrist = preprocess(obs["robot0_eye_in_hand_image"])
        if frames is not None:
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        if step % chunk_len == 0 or chunk is None:
            chunk = ipc.request(conn, {"images": [main, wrist], "lang": lang})["actions"]  # (8, 7)
        a = chunk[step % chunk_len]
        action = np.concatenate([a[:6], [binarize_gripper(float(a[6]))]]).astype(np.float32)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return 1, step + 1
        t += 1
        step += 1
    return 0, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="csv with suite, task, init_id (+ optional group, level)")
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=10093)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--video_dir", default=None, help="save the first episode of every task as mp4")
    a = ap.parse_args()

    from libero.libero import benchmark, get_libero_path  # imported here: only available in the LIBERO container
    from libero.libero.envs import OffScreenRenderEnv

    np.random.seed(a.seed)
    texts = a0_texts(a.labels_csv)
    with open(a.ids) as f:
        rows = list(csv.DictReader(f))
    suites = {s: benchmark.get_benchmark_dict()[s]() for s in {r["suite"] for r in rows}}
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {(r["task"], r["init_id"]) for r in csv.DictReader(f)}
    f = open(a.out, "a", newline="")
    w = csv.writer(f)
    if not done:
        w.writerow(["suite", "task", "group", "level", "init_id", "success", "steps", "seconds", "lang"])

    conn = ipc.connect(a.port)
    chunk_len = ipc.request(conn, {"meta": True})["chunk_len"]
    env, env_task, videos = None, None, set()
    for r in rows:
        if (r["task"], r["init_id"]) in done:
            continue
        ts = suites[r["suite"]]
        tid = ts.get_task_names().index(r["task"])
        task = ts.get_task(tid)
        if env_task != r["task"]:
            if env is not None:
                env.close()
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RESOLUTION,
                                     camera_widths=ENV_RESOLUTION)
            env.seed(a.seed)
            env_task = r["task"]
            lang = texts[(r["suite"], r["task"])]
            if lang != task.language:
                print(f"[text] training A0 differs from LIBERO text: {lang!r} vs {task.language!r}", flush=True)
        frames = [] if a.video_dir and r["task"] not in videos else None
        t0 = time.time()
        success, steps = run_episode(env, ts.get_task_init_states(tid)[int(r["init_id"])], lang, conn,
                                     chunk_len, MAX_STEPS[r["suite"]], a.num_steps_wait, frames)
        w.writerow([r["suite"], r["task"], r.get("group", ""), r.get("level", ""), r["init_id"],
                    success, steps, round(time.time() - t0, 1), lang])
        f.flush()
        print(f"[ep] {r['task'][:70]} init {r['init_id']}: success={success} steps={steps}", flush=True)
        if frames is not None:
            import imageio
            os.makedirs(a.video_dir, exist_ok=True)
            imageio.mimwrite(os.path.join(a.video_dir, f"{r['task'][:80]}_init{r['init_id']}_s{success}.mp4"),
                             frames, fps=30)
            videos.add(r["task"])
    conn.close()

    with open(a.out) as f:
        res = list(csv.DictReader(f))
    per_task = collections.defaultdict(list)
    for r in res:
        per_task[r["task"]].append(int(r["success"]))
    print("\n[summary]", flush=True)
    for task, s in per_task.items():
        print(f"  {sum(s):>3}/{len(s):<3} {task}", flush=True)
    total = [x for s in per_task.values() for x in s]
    print(f"  TOTAL {sum(total)}/{len(total)} = {100 * sum(total) / max(1, len(total)):.1f} %", flush=True)


if __name__ == "__main__":
    main()
