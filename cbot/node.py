"""One process per Slurm job, on any node: leads the 4-player loop or helps it.

    python -m cbot.node --procs 64

At most one job leads at a time: the one holding the lease runs/loop4.leader
(created exclusively). The leader runs cbot/loop4.sh (generation, training,
evals, gate) and refreshes the lease every minute. Every other job plays
generation chunks of the current round (cbot.genpool) and checks the lease
between chunks; when the lease is released (the leader was preempted or the
loop finished) or not refreshed for LEASE_STALE seconds (its node died), it
takes over and runs loop4.sh, which continues the round from the shared files.

A leader that finds its lease gone (taken over during a filesystem stall) stops
its loop and goes back to helping. A job exits when the loop is finished or
after --idle-exit seconds with nothing to do.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import signal
import subprocess
import sys
import threading
import time

from cbot import genpool
from cbot.genpool import log, read_task

LEASE = "runs/loop4.leader"
LEASE_STALE = int(os.environ.get("LEASE_STALE", 300))
TASK = "runs/gen4_task.json"
loop_proc = None  # loop4.sh while leading


def try_lease():
    for _ in range(2):
        try:
            fd = os.open(LEASE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{genpool.ME}\n".encode())
            os.close(fd)
            return True
        except FileExistsError:
            pass
        try:
            if time.time() - os.stat(LEASE).st_mtime < LEASE_STALE:
                return False
            os.rename(LEASE, f"{LEASE}.stale.{os.getpid()}")  # only one taker succeeds
            log("leader lease is stale; taking over")
        except FileNotFoundError:
            pass
    return False


def own_lease():
    try:
        with open(LEASE) as f:
            return f.read().strip() == genpool.ME
    except FileNotFoundError:
        return False


def release_lease():
    if own_lease():
        os.remove(LEASE)


def lead():
    """Run loop4.sh while holding the lease. Returns its exit code, or None if
    the lease was lost."""
    global loop_proc
    log("leading: running cbot/loop4.sh")
    proc = loop_proc = subprocess.Popen(["cbot/loop4.sh"], start_new_session=True)
    lost = threading.Event()

    def heartbeat():
        while proc.poll() is None:
            time.sleep(60)
            if not own_lease():
                lost.set()
                log("lost the leader lease; stopping the loop")
                os.killpg(proc.pid, signal.SIGTERM)
                return
            os.utime(LEASE)

    threading.Thread(target=heartbeat, daemon=True).start()
    code = proc.wait()
    loop_proc = None
    if lost.is_set():
        return None
    release_lease()
    log(f"loop4.sh exited with {code}")
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--idle-exit", type=int, default=3 * 3600)
    args = ap.parse_args()

    def on_term(signum, frame):  # Slurm preemption / scancel: hand over quickly
        if loop_proc is not None and loop_proc.poll() is None:
            os.killpg(loop_proc.pid, signal.SIGTERM)
        release_lease()
        log(f"signal {signum}; lease released, exiting")
        sys.exit(143)

    signal.signal(signal.SIGTERM, on_term)

    pool = None
    idle_since = time.time()
    while time.time() - idle_since < args.idle_exit:
        if try_lease():
            if pool is not None:  # give the loop every core
                pool.terminate()
                pool.join()
                pool = None
            code = lead()
            if code is not None:
                # 0: the loop reached LAST_ROUND. Otherwise it failed; stop rather
                # than retry, and let the other jobs try once each.
                sys.exit(code)
            idle_since = time.time()
            continue
        task = read_task(TASK)
        if task and not os.path.exists(os.path.join(task["out"], "DONE")):
            if pool is None:
                pool = mp.Pool(args.procs)
            if genpool.work(task, pool, args.procs, TASK):
                idle_since = time.time()
                continue
        time.sleep(30)
    log(f"idle for {args.idle_exit}s, exiting")


if __name__ == "__main__":
    main()
