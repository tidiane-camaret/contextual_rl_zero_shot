### Run on Nemo cluster

# CREATE A NEW ENVIRONMENT
# ws_allocate conda 1

# pip install -r dev/automl/meta_rl/requirements.txt

# conda config --prepend envs_dirs $( ws_find conda )/conda/envs
# conda config --prepend pkgs_dirs $( ws_find conda )/conda/pkgs
# conda config --show envs_dirs
# conda config --show pkgs_dirs

# conda create -n meta_rl_env
# conda activate meta_rl_env

# packages for mujoco 
# conda install -c conda-forge glew
# conda install -c conda-forge mesalib
# conda install -c menpo glfw3
# export CPATH=$CONDA_PREFIX/include

## Submit job 
msub -l nodes=1:ppn=1,walltime=10:00:00,pmem=6GB scripts/cluster/run_striker.moab
msub -l nodes=1:ppn=1,walltime=10:00:00 scripts/cluster/run_implicit_dqn.moab
# express : 
msub -q express -l nodes=1:ppn=1,walltime=15:00 scripts/cluster/run_striker.moab
# gpu : 
msub -q gpu -l nodes=1:ppn=1:gpus=1,walltime=40:00 scripts/cluster/run_striker.moab

# see queue
showq -u $USER

## Interactive session 
msub -l nodes=1:ppn=10,walltime=8:00:00,pmem=6GB -I 
# dont forget to activate the env
source miniconda3/etc/profile.d/conda.sh
conda activate tid_env
# run the script
cd dev/automl/meta_rl/
bash scripts/experiments/carl/carl_dqn.bash






# further infos on arguments : 
# https://wiki.calculquebec.ca/w/Utilisation_des_noeuds_de_calcul#Param.C3.A8tres_de_la_commande_msub
# https://wiki.bwhpc.de/e/Batch_Jobs_Moab