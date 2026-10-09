"""One process per node (a Slurm job, or a plain process on a node without
Slurm): leads the 4-player loop or helps it.

    python -m cbot.node --procs 64

At most one process leads at a time: the one holding the lease runs/loop4.leader
(created exclusively). The leader runs cbot/loop4.sh (generation, training,
evals, gate) and refreshes the lease every minute. Every other process plays
generation chunks (cbot.genpool) and checks the lease before each chunk claim;
when the lease is released (the leader was preempted or the loop finished) or
not refreshed for LEASE_STALE seconds (its node died), it finishes its open
chunks, takes over and runs loop4.sh, which continues the round from the
shared files.

Leader preference: LEADER_PREFER lists node names, best first (default
"dunes matthew": most cores and memory, no preemption). A free lease goes to
the most preferred node that is up (each process touches runs/nodes/<name>
every minute); any other node takes it only after it has been free for
LEAD_DELAY seconds. A leader that sees a more preferred node up asks loop4.sh
to hand over at the next round (runs/loop4.handoff; exit code 3) and goes back
to helping.

A leader that finds its lease gone (taken over during a filesystem stall) stops
its loop and goes back to helping. A process exits when the loop is finished or
after --idle-exit seconds with nothing to do.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import signal
import socket
import subprocess
import sys
import threading
import time

from cbot import genpool
from cbot.genpool import log, read_task

LEASE = "runs/loop4.leader"
HANDOFF = "runs/loop4.handoff"
NODES = "runs/nodes"
LEASE_STALE = int(os.environ.get("LEASE_STALE", 300))
LEAD_DELAY = int(os.environ.get("LEAD_DELAY", 600))
ALIVE = 180  # a node is up if its runs/nodes file was touched this recently
PREFER = os.environ.get("LEADER_PREFER", "dunes matthew").split()
HOST = os.environ.get("NODE_NAME") or socket.gethostname().split(".")[0]
TASK = "runs/gen4_task.json"
PROCS = os.cpu_count()  # set from --procs
HANDED_OFF = 3  # loop4.sh exit code
loop_proc = None  # loop4.sh while leading


def rank(host):
    return PREFER.index(host) if host in PREFER else len(PREFER)


def nodes_up():
    now = time.time()
    try:
        names = os.listdir(NODES)
    except FileNotFoundError:
        return [HOST]
    up = []
    for n in names:
        try:
            if now - os.stat(os.path.join(NODES, n)).st_mtime < ALIVE:
                up.append(n)
        except FileNotFoundError:
            pass
    return up + [HOST]


def presence():
    os.makedirs(NODES, exist_ok=True)
    while True:
        open(os.path.join(NODES, HOST), "a").close()
        os.utime(os.path.join(NODES, HOST))
        time.sleep(60)


def lease_free():
    """True if the lease is released or stale, i.e. the loop needs a leader."""
    try:
        return time.time() - os.stat(LEASE).st_mtime >= LEASE_STALE
    except FileNotFoundError:
        return True


class Watch:
    """Decides when this node may take a free lease."""

    def __init__(self):
        self.free_since = None

    def can_lead(self):
        if not lease_free():
            self.free_since = None
            return False
        now = time.time()
        if self.free_since is None:
            self.free_since = now
        if rank(HOST) <= min(rank(n) for n in nodes_up()):
            return True
        return now - self.free_since >= LEAD_DELAY


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
    try:
        os.remove(HANDOFF)  # left over from an earlier leader
    except FileNotFoundError:
        pass
    log("leading: running cbot/loop4.sh")
    env = dict(os.environ, P=str(PROCS))  # this node's cores, for the loop's workers
    proc = loop_proc = subprocess.Popen(["cbot/loop4.sh"], start_new_session=True, env=env)
    lost = threading.Event()

    def heartbeat():
        asked = False
        while proc.poll() is None:
            time.sleep(60)
            if proc.poll() is not None:
                return
            if not own_lease():
                lost.set()
                log("lost the leader lease; stopping the loop")
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                return
            os.utime(LEASE)
            better = [n for n in nodes_up() if rank(n) < rank(HOST)]
            if better and not asked:
                log(f"{better[0]} is up and preferred; handing over after this round")
                open(HANDOFF, "w").close()
                asked = True

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
    global PROCS
    PROCS = args.procs

    def on_term(signum, frame):  # Slurm preemption / scancel / kill: hand over quickly
        if loop_proc is not None and loop_proc.poll() is None:
            os.killpg(loop_proc.pid, signal.SIGTERM)
        release_lease()
        log(f"signal {signum}; lease released, exiting")
        sys.exit(143)

    signal.signal(signal.SIGTERM, on_term)
    threading.Thread(target=presence, daemon=True).start()
    log(f"node {HOST} up, {args.procs} procs, leader preference {PREFER}")

    watch = Watch()
    pool = None
    idle_since = time.time()
    while time.time() - idle_since < args.idle_exit:
        if watch.can_lead() and try_lease():
            if pool is not None:  # give the loop every core
                pool.terminate()
                pool.join()
                pool = None
            code = lead()
            watch = Watch()
            if code not in (None, HANDED_OFF):
                # 0: the loop reached LAST_ROUND. Otherwise it failed; stop rather
                # than retry, and let the other processes try once each.
                sys.exit(code)
            idle_since = time.time()
            if code == HANDED_OFF:
                time.sleep(90)  # let the preferred node take the lease first
            continue
        task = read_task(TASK)
        if task and not os.path.exists(os.path.join(task["out"], "DONE")):
            if pool is None:
                pool = genpool.new_pool(args.procs)
            # Stop claiming chunks as soon as this node should lead, so a leader
            # lost mid-training is replaced within a chunk, not a round.
            try:
                if genpool.work(task, pool, args.procs, TASK, stop=watch.can_lead):
                    idle_since = time.time()
                    continue
            except genpool.PoolStuck:
                pool.terminate()
                pool.join()
                pool = None
                continue
        time.sleep(30)
    log(f"idle for {args.idle_exit}s, exiting")


if __name__ == "__main__":
    main()
