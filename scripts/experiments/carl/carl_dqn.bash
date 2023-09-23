env_id="CARLAcrobot"
context_name="LINK_MASS_2"

echo "Running CARL DQN experiments for $env_id with context $context_name"

for seed in 0 1 2 3 4 5 6 7 8 9
do

    python3 scripts/experiments/carl/cleanrl_dqn.py --total-timesteps 500000 --track --env-id $env_id --context-name $context_name --seed $seed --context-state explicit
    python3 scripts/experiments/carl/cleanrl_dqn.py --total-timesteps 500000 --track --env-id $env_id --context-name $context_name --seed $seed --context-state hidden
    python3 scripts/experiments/carl/cleanrl_dqn_iida.py --total-timesteps 500000 --track --env-id $env_id --context-name $context_name --seed $seed --context-state implicit
    python3 scripts/experiments/carl/cleanrl_dqn_iida.py --total-timesteps 500000 --track  --env-id $env_id --context-name $context_name --seed $seed --context-state implicit_std

done