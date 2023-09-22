for seed in 15 16 17 18 19
do

    python3 scripts/experiments/carl/cleanrl_dqn.py --total-timesteps 500000 --track --context_name length --seed $seed --context_state explicit
    python3 scripts/experiments/carl/cleanrl_dqn.py --total-timesteps 500000 --track --context_name length --seed $seed --context_state hidden
    python3 scripts/experiments/carl/cleanrl_dqn_iida.py --total-timesteps 500000 --track --context_name length --seed $seed --context_state implicit
    python3 scripts/experiments/carl/cleanrl_dqn_iida.py --total-timesteps 500000 --track --context_name length --seed $seed --context_state implicit_std

done