import argparse
import numpy as np
import itertools

import gym
from gym.wrappers import RecordEpisodeStatistics

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import vec_env, monitor
from stable_baselines3 import PPO

import wandb
from wandb.integration.sb3 import WandbCallback

import meta_rl


NUM_OF_PARAMS = 2
NUM_OF_ENVS = 8
task_name = "striker"


if __name__ == "__main__":
    """
    Comparaison of Invariant (Average) and Oracle implementations 
    for the Striker task.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--nb_steps', type=int, default=2_000_000)
    parser.add_argument('--nb_runs_per_eval', type=int, default=100)
    args = parser.parse_args()
    oracle = args.oracle
    render = args.render
    nb_total_timesteps = args.nb_steps
    nb_runs_per_eval = args.nb_runs_per_eval
    eval_every = nb_total_timesteps // 10

    print("Oracle: ", oracle)



    run = wandb.init(
        project="meta_rl_epi_orig",
        monitor_gym=True, # auto-upload the videos of agents playing the game
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        config={
            "task_name": task_name,
            "oracle": oracle,
            "num_of_params": NUM_OF_PARAMS,
            "total_timesteps": nb_total_timesteps,
        }
        )

        

    # generate the training environment

    if oracle:
        train_env = vec_env.DummyVecEnv([
        lambda: monitor.Monitor(
        RecordEpisodeStatistics(gym.make("StrikerOracle-v0")),
        )
        for _ in range(NUM_OF_ENVS)])
    
    
    else:
        train_env = vec_env.DummyVecEnv([
        lambda: monitor.Monitor(
        RecordEpisodeStatistics(gym.make("StrikerAvg-v0")
            ),
        )
        for _ in range(NUM_OF_ENVS)])

    model = PPO('MlpPolicy', 
                env=train_env,
                verbose=1,
                tensorboard_log="results/tensorboard/"+task_name+"/")

    scale_list = itertools.product(np.arange(6) + 0.5, repeat=NUM_OF_PARAMS)
    scale_list = np.array(list(scale_list))
    scale_list = scale_list * 0.1

    for learning_step in range(0, nb_total_timesteps, eval_every):
        print(f"learning step: {learning_step}")
        model.learn(total_timesteps=eval_every,
                    callback=WandbCallback(),
                    )

        # evaluate the policy on unseen scale values

        global_mean_eval = []
        for s, scale in enumerate(scale_list):
            if oracle:
                eval_env = gym.make('StrikerOracle-v0', eval_scale=scale, eval_mode=True)
            else:
                eval_env = gym.make('StrikerAvg-v0', eval_scale=scale, eval_mode=True)
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=nb_runs_per_eval)
            wandb.log({f"mean_reward_{s}": mean_reward})
            wandb.log({f"std_reward_{s}": std_reward})
            print(f"scale_id: {s}, mean_reward:{mean_reward:.2f} +/- {std_reward}")
            global_mean_eval.append(mean_reward)
            eval_env.close()
        
        wandb.log({"global_mean_eval": np.mean(global_mean_eval)})
        
    # close wandb

    run.finish()

    # render the policy

    obs = eval_env.reset()
    print("obs:", obs.shape, )

    if render:
        for _ in range(10):
            obs = eval_env.reset()
            for _ in range(100):
                action, _states = model.predict(obs)
                obs, reward, done, info = eval_env.step(action)
                eval_env.render()

