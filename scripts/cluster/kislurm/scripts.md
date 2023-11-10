### Run on KISlurm

# information about resources
sinfo
sfree

# start an interactive session
srun -p testbosch_cpu-cascadelake --time=3:00:00 --pty bash 
srun -p ml_gpu-rtx2080 --time=3:00:00 --pty bash 
srun -p ml_gpu-rtx2080 -c 20 --mem 24000 --time=3:00:00 --pty bash 

# tmux
tmux new-session -s <name>
# detach: 
ctrl+b d
# attach: 
tmux attach -t <name>

# run scripts
cd ~/dev/auto_ml/meta_rl
python3 scripts/jrpl/dqn.py


# start a job
sbatch scripts/cluster/kislurm/run_jrpl.sh

# see all jobs
sacct --user=$USER

# see all running jobs
squeue --user=$USER

# see job details
scontrol show job 3868830

# cancel job
scancel 3868830