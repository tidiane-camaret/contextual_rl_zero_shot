### Run on KISlurm

# information about resources
sinfo
sfree

# start an interactive session
srun -p testbosch_cpu-cascadelake --time=3:00:00 --pty bash 
srun -p ml_gpu-rtx2080 --time=3:00:00 --pty bash 
srun -p ml_gpu-rtx2080 -c 20 --mem 24000 --time=3:00:00 --pty bash 

srun -p aisdlc_gpu-rtx2080 --time=3:00:00 --pty bash 
conda activate meta_rl_env
# tmux
tmux new-session -s <name>
# detach: 
ctrl+b d
# attach: 
tmux attach -t <name>

# run scripts

python3 scripts/jrpl/train_dqn.py 
python3 scripts/jrpl/train_ddqn.py --env-id CARLMountainCar

# run HPO using how-to-autorl
conda deactivate
conda activate autorl-sweepers

python3 -m scripts.hpo.how_to_autorl.dehb_for_cartpole_dqn_jrpl --multirun
# OR : 
cd automl/how-to-autorl/
python3 -m examples.dehb_for_pendulum_ppo.py --multirun
automl/how-to-autorl/examples/

# start a job
sbatch scripts/cluster/kislurm/run_jrpl_array.sh

# see all jobs
sacct --user=$USER

# see all running jobs
squeue --user=$USER

# see job details
scontrol show job 3868830

# cancel job
scancel 3868830