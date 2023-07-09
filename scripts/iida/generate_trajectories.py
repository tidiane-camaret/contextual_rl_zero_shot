"""
Train PPO on the striker environment with original parameter
Produce N trajectories and save them as a pickle file
"""
import argparse
import numpy as np
import itertools
import pickle

import gym
from gym.wrappers import RecordEpisodeStatistics

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import vec_env, monitor
from stable_baselines3 import PPO

import meta_rl
from meta_rl.definitions import ROOT_DIR, RESULTS_DIR
NUM_OF_PARAMS = 2

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_train_steps', type=int, default=200_000)
    args = parser.parse_args()

    nb_train_steps = args.nb_train_steps
    
    train_env = gym.make('StrikerAvg-v0', eval_mode=False)
    model = PPO('MlpPolicy', 
                train_env,)

    # learn policy over train env (multiple param settings)
    model.learn(total_timesteps=nb_train_steps, progress_bar=True)

    # save model
    model.save(RESULTS_DIR / "iida/ppo_generator")

    # produce trajectories for each of the 36 training scales
    # in orig. impl, 8_000_000 steps in total
    

    
    def get_traj_dict(scale_list):
        traj_dict = {}
        for scale in scale_list:
            print("generating trajectories for scale: ", scale)
            eval_env = gym.make('StrikerAvg-v0', eval_mode=True, eval_scale=scale)
            traj_list = []
            for _ in range(100):
                obs = eval_env.reset()
                s_, a_ = [], []
                for i in range(200):
                    
                    action, _states = model.predict(obs)
                    obs, reward, done, info = eval_env.step(action)
                    s_.append(obs)
                    a_.append(action)
                tuple = [(s_[i], a_[i], s_[i+1], eval_env.scale) for i in range(len(s_) - 1)]
                traj_list.extend(tuple)
            traj_dict[str(scale)] = traj_list
        return traj_dict

    scale_list_train = itertools.product(np.arange(6), repeat=NUM_OF_PARAMS)
    scale_list_train = np.array(list(scale_list_train))
    scale_list_train = scale_list_train * 0.1
    traj_dict_train = get_traj_dict(scale_list_train)

    with open(RESULTS_DIR / 'iida/traj_dict_train.pkl', 'wb') as f:
        pickle.dump(traj_dict_train, f)

    scale_list_test = itertools.product(np.arange(6) + 0.5, repeat=NUM_OF_PARAMS)
    scale_list_test = np.array(list(scale_list_test))
    scale_list_test = scale_list_test * 0.1
    traj_dict_test = get_traj_dict(scale_list_test)

    with open(RESULTS_DIR / 'iidaiida/traj_dict_test.pkl', 'wb') as f:
        pickle.dump(traj_dict_test, f)
