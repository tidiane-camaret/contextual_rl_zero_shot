### Run on Nemo cluster

## Submit job 
msub -l nodes=1:ppn=1,walltime=3:00:00, pmem=24GB nemo_jobs/run_striker.moab
# express : 
msub -q express -l nodes=1:ppn=1,walltime=15:00 nemo_jobs/run_striker.moab
# gpu : 
msub -q gpu -l nodes=1:ppn=1:gpus=1,walltime=40:00 nemo_jobs/run_striker.moab

## Interactive session 
msub -l nodes=1:ppn=1,walltime=3:00:00, pmem=24GB -I
# dont forget to activate the env
source miniconda3/etc/profile.d/conda.sh
conda activate tid_env
# run the script
python3 dev/automl/meta_rl/scripts/run_striker.py 