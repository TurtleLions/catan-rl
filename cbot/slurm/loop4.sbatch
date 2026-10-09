#!/bin/bash
# 4-player expert-iteration loop, one job per node; all jobs are identical:
#   sbatch -w matthew cbot/slurm/loop4.sbatch     (or cbot/launch.sh for all nodes)
# One process leads (runs cbot/loop4.sh: generation, training, evals); the
# others play generation chunks and take over if the leader goes away (see
# cbot/node.py). Any subset of nodes works, including one.
# Rounds and settings: cbot/loop4.conf, overridden live by runs/loop4.conf.
# Preempted jobs are requeued and rejoin with the repo's current code.
# Status: cbot/slurm/status.sh. Output: runs/slurm-<jobid>.out, runs/loop4.log.
#SBATCH --job-name=catan4
#SBATCH --account=scc
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=runs/slurm-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
exec "$SLURM_SUBMIT_DIR/cbot/run_node.sh"
