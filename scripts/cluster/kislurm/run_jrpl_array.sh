#!/bin/bash

# Array job with 4 scripts, each with 10 seeds
#SBATCH --array=1-40

seed=$(( ($SLURM_ARRAY_TASK_ID - 1) / 4 ))
script_num=$(( ($SLURM_ARRAY_TASK_ID - 1) % 4 ))

# Define the partition on which the job shall run.
#SBATCH --partition ml_gpu-rtx2080    # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name JRPL_array             # short: -J <job name>

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output results/cluster_logs/%x-%A-%a-HelloCluster.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error results/cluster_logs/%x-%A-%a-HelloCluster.err    # STDERR  short: -e logs/%x-%A-job_name.out

# Define the amount of memory required per node
#SBATCH --mem 8GB

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
source ~/miniconda3/bin/activate # Adjust to your path of Miniconda installation
conda activate automl_env

# Define the parameters
env_id="CARLCartPole"
context_name="tau"

# Running the job
start=`date +%s`

case $script_num in
    0)
        echo "Running JRPL DQN experiments for $env_id with context $context_name"
        python3 scripts/jrpl/dqn.py --track --env-id $env_id --context-name $context_name --seed $seed --context-mode explicit
        ;;
    1)
        echo "Running JRPL DQN experiments for $env_id with context $context_name"
        python3 scripts/jrpl/dqn.py --track --env-id $env_id --context-name $context_name --seed $seed --context-mode hidden
        ;;
    2)
        echo "Running JRPL DQN experiments for $env_id with context $context_name"
        python3 scripts/jrpl/dqn.py --track --env-id $env_id --context-name $context_name --seed $seed --context-mode learned --context-encoder mlp_avg
        ;;
    3)
        echo "Running JRPL DQN experiments for $env_id with context $context_name"
        python3 scripts/jrpl/dqn.py --track --env-id $env_id --context-name $context_name --seed $seed --context-mode learned --context-encoder mlp_avg_std
        ;;
esac

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
