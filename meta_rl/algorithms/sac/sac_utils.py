# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecordEpisodeStatistics(gym.wrappers.RecordEpisodeStatistics):
    """
    have to override `self.episode_returns += np.array(rewards)`
    to `self.episode_returns += rewards`
    because jax would modify the `self.episode_returns` to be a
    jax array.
    See https://wandb.ai/costa-huang/brax/reports/Brax-as-Pybullet-replacement--Vmlldzo5ODI4MDk
    """

    def step(self, action):

        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += np.array(rewards)
        self.episode_lengths += 1
        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if "episode" in infos or "_episode" in infos:
                raise ValueError(
                    "Attempted to add episode stats when they already exist"
                )
            else:
                infos["episode"] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(
                        dones,
                        np.round(time.perf_counter() - self.episode_start_times, 6),
                        0.0,
                    ),
                }
                if self.is_vector_env:
                    infos["_episode"] = np.where(dones, True, False)
            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones
            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = 0
            self.episode_start_times[dones] = time.perf_counter()
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def make_env(env_, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            # env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env_, f"videos/{run_name}")
            env = RecordEpisodeStatistics(env)  # TODO causes issues with brax, see why

            env.action_space.seed(seed)
            return env
        # else:
        # env = gym.make(env_id)
        env = RecordEpisodeStatistics(env_)  # TODO causes issues with brax, see why

        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, latent_context_dim=0):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + latent_context_dim
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, latent_context_dim=0):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + latent_context_dim,
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def eval_sac(eval_env, actor, context_encoder, args):
    # TODO : optimize. env should not be redifined at each eval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs, info = eval_env.reset()
    steps = 0
    rewards = []

    if "learned" not in args.context_mode:
        while True:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()
            obs, r, done, truncated, info = eval_env.step(actions)
            steps += 1
            rewards.append(r)
            if done or steps >= args.env_max_episode_steps:
                break
    else:
        traj_actions = []
        traj_obs = []
        context_mu = torch.zeros(1, args.latent_context_dim).to(device)

        while True:
            obs_np = np.array(obs)  # cannot convert jax array to tensor directly
            obs_context = torch.cat(
                [torch.Tensor(obs_np).to(device), context_mu], dim=-1
            )
            actions, _, _ = actor.get_action(torch.Tensor(obs_context).to(device))
            actions = actions.detach().cpu().numpy()

            # add the current transition to the trajectory history
            traj_actions.append(actions)
            traj_obs.append(obs)

            if steps > 0:  # if we have transitions, encode the context
                transitions = np.concatenate(
                    [
                        np.asarray(traj_obs)[:-1],
                        np.asarray(traj_actions)[:-1],
                        np.asarray(traj_obs)[1:],
                    ],
                    axis=-1,
                )
                # if transitions is bigger than context_length, sample a random subset of size context_length
                if transitions.shape[0] > args.nb_input_transitions:
                    idxs = np.random.randint(
                        0, transitions.shape[0], size=args.nb_input_transitions
                    )
                    transitions = transitions[idxs]
                # add a dimension for the batch
                transitions = torch.Tensor(transitions).to(device).unsqueeze(0)
                context_mu, context_sigma = context_encoder(transitions)

            obs, r, done, truncated, info = eval_env.step(actions)
            steps += 1
            rewards.append(r)

            if done or steps >= args.env_max_episode_steps:
                break
    return np.sum(rewards)

    """

    eval_table = [[x, y] for (x, y) in zip(range(len(rewards_list)), rewards_list)]
    eval_table_wandb = wandb.Table(data=eval_table, columns=["context_value", "reward"])
    writer.log(
        {
            "eval_table": wandb.plot.line(
                eval_table_wandb, "context_value", "reward", title="Custom Y vs X Line Plot"
            )
        }
    )
    """
