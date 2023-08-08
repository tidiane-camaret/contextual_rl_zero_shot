import argparse
from carl.envs import CARLCartPoleEnv_defaults, CARLCartPoleEnv
from stable_baselines3 import PPO

import wandb
from wandb.integration.sb3 import WandbCallback

from meta_rl.definitions import RESULTS_DIR

task_name = 'cartpole'
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--context', type=str, default='explicit')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--nb_steps', type=int, default=2_000_000)
    parser.add_argument('--nb_runs_per_eval', type=int, default=100)
    args = parser.parse_args()
    context = args.context
    render = args.render
    nb_total_timesteps = args.nb_steps
    nb_runs_per_eval = args.nb_runs_per_eval

    longer_pole = CARLCartPoleEnv_defaults.copy()
    longer_pole["pole_length"] = CARLCartPoleEnv_defaults["pole_length"]*2
    contexts = {0: CARLCartPoleEnv_defaults, 1: longer_pole}

    print(contexts)


    train_env = CARLCartPoleEnv(contexts=contexts)
    model = PPO('MlpPolicy', 
                env=train_env,
                verbose=1,
                tensorboard_log=RESULTS_DIR / "tensorboard" / task_name )
        
    model.learn(total_timesteps=nb_total_timesteps, 
                    #callback=WandbCallback()
                    )