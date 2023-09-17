for seed in 5 6 7 8 9 10
do

    python3 scripts/experiments/carl/cleanrl_dqn.py --total-timesteps 5000000 --track --context_name length --seed $seed --hide_context
    python3 scripts/experiments/carl/cleanrl_dqn.py --total-timesteps 5000000 --track --context_name length --seed $seed 
    python3 scripts/experiments/carl/cleanrl_dqn_iida.py --total-timesteps 5000000 --track --context_name length --seed $seed 
done