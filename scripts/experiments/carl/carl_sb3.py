import argparse
from carl.envs.gymnasium.classic_control.carl_cartpole import CARLCartPole 
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

    DEFAULT_CONTEXT = {
    "gravity": 9.8,
    "masscart": 1.0,  # Should be seen as 100% and scaled accordingly
    "masspole": 0.1,  # Should be seen as 100% and scaled accordingly
    #"pole_length": 0.5,  # Should be seen as 100% and scaled accordingly
    #"force_magnifier": 10.0,
    #"update_interval": 0.02,  # Seconds between updates
    "initial_state_lower": -0.1,  # lower bound of initial state distribution (uniform) (angles and angular velocities)
    "initial_state_upper": 0.1,  # upper bound of initial state distribution (uniform) (angles and angular velocities)
}

    longer_pole = DEFAULT_CONTEXT.copy()
    longer_pole["gravity"] = DEFAULT_CONTEXT["gravity"]*2
    contexts = {0: DEFAULT_CONTEXT, 1: longer_pole}

    print(contexts)


    train_env = CARLCartPole(contexts=contexts)
    model = PPO('MlpPolicy', 
                env=train_env,
                verbose=1,
                tensorboard_log=RESULTS_DIR / "tensorboard" / task_name )
        
    model.learn(total_timesteps=nb_total_timesteps, 
                    #callback=WandbCallback()
                    )