#env_id="CARLLunarLander"
#context_name="GRAVITY_Y"

env_id="CARLCartPole"
context_name="tau" #"gravity" #"length" #


echo "Running JRPL DQN experiments for $env_id with context $context_name"

for seed in 10 11 12 13 14 15 16 17 18 19
do

    python3 scripts/jrpl/dqn.py --track --env-id $env_id --context-name $context_name --seed $seed --context-mode explicit
    python3 scripts/jrpl/dqn.py --track --env-id $env_id --context-name $context_name --seed $seed --context-mode hidden
    python3 scripts/jrpl/dqn.py --track --env-id $env_id --context-name $context_name --seed $seed --context-mode learned --context-encoder mlp_avg
    python3 scripts/jrpl/dqn.py --track --env-id $env_id --context-name $context_name --seed $seed --context-mode learned --context-encoder mlp_avg_std

done