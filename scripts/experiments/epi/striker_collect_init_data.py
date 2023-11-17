import sys
import pickle
import os
import time 
import copy
import numpy as np
import pandas as pd

import gym
from gym.wrappers import RecordEpisodeStatistics

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import vec_env, monitor
from stable_baselines3 import PPO

import meta_rl

NUM_OF_PARAMS = 2
NUM_OF_ENVS = 8
task_name = "striker"

def main():
    """
    Collection of the initial data for the EPI policy
    """
    """
    train_env = vec_env.DummyVecEnv([
        lambda: monitor.Monitor(
        RecordEpisodeStatistics(gym.make("StrikerAvg-v0")
            ),
        )
        for _ in range(NUM_OF_ENVS)])
    """
    train_env = gym.make("StrikerAvg-v0")
    model = PPO('MlpPolicy', 
                env=train_env,
                verbose=1,
                #batch_size=100000,
                #gae_lambda=1,
                #policy_kwargs=dict(net_arch=[32, 32]),
                )
    
    """
    First, train the policy on the default environment
    
    model.learn(total_timesteps=2_000_000)

    model.save("results/policies/initial/" + task_name)
    """
    model = PPO.load("results/policies/initial/" + task_name)
    
    for _ in range(0):
            obs = train_env.reset()
            for _ in range(200):
                action, _states = model.predict(obs)
                obs, reward, done, info = train_env.step(action)
                train_env.render()
    
    target_sample_size = 50 #1000
    egreedy = 0.2

    data = []
    rollouts = []

    """
    collecting data for for the scale ID 1
    """

    while len(rollouts) < target_sample_size:
        print("step: ", len(rollouts))
        observation = train_env.reset()
        #train_env.change_env(np.array([0.1, 0.1]))
        old_ball_pos = train_env.sim.data.qpos[-9:-7]
        for i in range(200):
            
            if np.random.rand() < egreedy:
                action = train_env.action_space.sample()
            else:
                action, d = model.predict(observation)
            next_observation, reward, terminal, reward_dict = train_env.step(action)
            ball_pos = (train_env.sim.data.qpos[-9:-7])
            # if ball_pos and old_ball_pos are not the same :
            if np.linalg.norm(ball_pos-old_ball_pos) > 0.005: # was the ball moved?
                full_state = train_env.state_vector()
                rollouts.append([full_state, action])
            observation = next_observation
            old_ball_pos = copy.copy(ball_pos)
            if terminal or len(rollouts) == target_sample_size:
                break

    """
    collecting data for all scale IDs
    """
    print('Rollout...')
    for i in range(5):
        for j in range(5):
            env_id = int((i * 5 + j))  # default: 1, 2
            train_env.change_env(scale=np.array([i*0.1, j*0.1]))
            print(train_env.env_id)
            print(train_env.scale)

            for rollout in rollouts:
                state = rollout[0]
                observation = train_env.force_reset_model(qpos=state[:16], qvel=state[16:])
                action = rollout[1]
                before = np.concatenate([train_env.sim.data.qpos[7:9,], train_env.sim.data.qvel[7:9,], train_env.get_body_com("tips_arm")])
                next_observation, reward, terminal, reward_dict = train_env.step(action)
                after = np.concatenate([train_env.sim.data.qpos[7:9,], train_env.sim.data.qvel[7:9,], train_env.get_body_com("tips_arm")])
                data.append(np.concatenate([before, after, np.array([train_env.env_id]), train_env.scale]))
                observation = next_observation

    data = np.array(data)

    g = lambda s, num: [s + str(i) for i in range(num)]
    columns = g('obs', 7)+g('next_obs', 7)+g('env_id', 1)+g('env_vec', 2)
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(os.getcwd() + '/scripts/orig_impl/striker_data_vine.csv')
if __name__ == '__main__':
    main()