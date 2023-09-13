for seed in 0 1 2 3 4
do
    python3 scripts/experiments/carl/cleanrl_dqn.py --total-timesteps 250000 --track --context_name length --seed $seed --hide_context
    python3 scripts/experiments/carl/cleanrl_dqn.py --total-timesteps 250000 --track --context_name length --seed $seed 

done