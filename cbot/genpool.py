"""Data generation shared by several Slurm jobs through the shared filesystem.

    python -m cbot.genpool coordinate --task runs/gen4_task.json --procs 64 \\
        --out data/m3s --games 30000 --seed0 ... --mix ...   # same args as cbot.gen

coordinate (run by the loop's leader, cbot/loop4.sh) writes the task file,
works on it until every chunk is written, and then marks the output DONE. The
other jobs (cbot/node.py) call work() on the same task.

The games (exactly cbot.gen's game list) are split into chunks of --chunk
games. A process claims chunk k by creating claims/<k> in the output directory
(exclusive create), plays it, and writes c<k>.npz atomically (c<k>.empty if no
state was recorded). Claims are refreshed every minute while their games run;
a claim not refreshed for STALE seconds (its job was preempted) is taken over.
Two processes can rarely end up on the same chunk; both write the same file,
so the only cost is the duplicated work. Games already in shard*.npz files
from cbot.gen are skipped, so a cbot.gen run can be finished here.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import queue
import socket
import time
from collections import deque

import numpy as np

from cbot.gen import legacy_seeds, make_jobs, play_record

STALE = int(os.environ.get("GENPOOL_STALE", 300))  # seconds
TOUCH = 60
ME = f"{socket.gethostname()}:{os.environ.get('SLURM_JOB_ID', '-')}:{os.getpid()}"


def log(msg):
    print(f"[{time.strftime('%F %T')}] {ME} {msg}", flush=True)


def chunk_file(out, k):
    return os.path.join(out, f"c{k:05d}.npz")


def is_done(out, k):
    p = chunk_file(out, k)
    return os.path.exists(p) or os.path.exists(p[:-4] + ".empty")


def try_claim(claims, k):
    p = os.path.join(claims, str(k))
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{ME} {time.time():.0f}\n".encode())
            os.close(fd)
            return True
        except FileExistsError:
            pass
        try:
            if time.time() - os.stat(p).st_mtime < STALE:
                return False
            os.rename(p, f"{p}.stale.{os.getpid()}")  # only one taker succeeds
            log(f"taking over stale chunk {k}")
        except FileNotFoundError:
            return False
    return False


def write_chunk(out, k, parts):
    p = chunk_file(out, k)
    if not parts:
        open(p[:-4] + ".empty", "w").close()
        return
    np.savez(
        p + ".tmp.npz",
        X=np.concatenate([r[0] for r in parts]),
        y=np.concatenate([r[1] for r in parts]),
        v=np.concatenate([r[2] for r in parts]),
        g=np.concatenate([r[3] for r in parts]),
    )
    os.replace(p + ".tmp.npz", p)


def work(task, pool, procs):
    """Play claimable chunks of the task until none is left to claim; returns
    the number of chunks this process wrote."""
    out, C = task["out"], task["chunk"]
    claims = os.path.join(out, "claims")
    os.makedirs(claims, exist_ok=True)
    jobs = make_jobs(task["games"], task["seed0"], task["mix"], task["eps"], task["p_record"])
    seen = legacy_seeds(out)
    nchunk = (len(jobs) + C - 1) // C

    results = queue.Queue()
    todo = deque()  # (chunk, job)
    open_ = {}  # chunk -> {"left": games still running, "parts": results}
    inflight, wrote, scan, last_touch = 0, 0, 0, time.time()
    t0 = time.time()

    def claim_next():
        nonlocal scan
        while scan < nchunk:
            k = scan
            scan += 1
            if k in open_ or is_done(out, k) or not try_claim(claims, k):
                continue
            js = [j for j in jobs[k * C : (k + 1) * C] if j[1] not in seen]
            open_[k] = {"left": len(js), "parts": []}
            todo.extend((k, j) for j in js)
            return k
        return None

    def finish(k):
        nonlocal wrote
        write_chunk(out, k, open_.pop(k)["parts"])
        try:
            os.remove(os.path.join(claims, str(k)))
        except FileNotFoundError:
            pass
        wrote += 1

    while True:
        while inflight < 2 * procs:
            if not todo:
                k = claim_next()
                if k is None:
                    break
                if open_[k]["left"] == 0:
                    finish(k)
                continue
            k, j = todo.popleft()
            pool.apply_async(
                play_record, (j,),
                callback=lambda r, k=k: results.put((k, r)),
                error_callback=lambda e, k=k: results.put((k, e)),
            )
            inflight += 1
        if inflight == 0:
            return wrote
        k, r = results.get()
        inflight -= 1
        if isinstance(r, BaseException):
            raise r
        if r is not None:
            open_[k]["parts"].append(r)
        open_[k]["left"] -= 1
        if open_[k]["left"] == 0:
            finish(k)
            done = sum(is_done(out, i) for i in range(nchunk))
            log(f"{out}: chunk {k} written; {done}/{nchunk} chunks done; "
                f"this process {wrote} chunks in {time.time() - t0:.0f}s")
        if time.time() - last_touch > TOUCH:
            for i in open_:
                try:
                    os.utime(os.path.join(claims, str(i)))
                except FileNotFoundError:
                    pass
            last_touch = time.time()


def all_done(task):
    nchunk = (task["games"] + task["chunk"] - 1) // task["chunk"]
    return all(is_done(task["out"], k) for k in range(nchunk))


def read_task(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["coordinate"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    # Same meaning as in cbot.gen:
    ap.add_argument("--out")
    ap.add_argument("--games", type=int)
    ap.add_argument("--mix")
    ap.add_argument("--eps", default="0,0.03")
    ap.add_argument("--p-record", type=float, default=0.2)
    ap.add_argument("--seed0", type=int)
    ap.add_argument("--chunk", type=int, default=200)
    args = ap.parse_args()

    task = {k: getattr(args, k) for k in ("out", "games", "mix", "eps", "p_record", "seed0", "chunk")}
    os.makedirs(args.out, exist_ok=True)
    tmp = f"{args.task}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(task, f)
    os.replace(tmp, args.task)
    log(f"task {task}")
    t0 = time.time()
    with mp.Pool(args.procs) as pool:
        while not all_done(task):
            if work(task, pool, args.procs) == 0:
                time.sleep(30)  # remaining chunks are claimed by other jobs
    open(os.path.join(args.out, "DONE"), "w").close()
    log(f"{args.out} DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
