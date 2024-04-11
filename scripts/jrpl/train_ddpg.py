# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import argparse
import importlib
import os

# to record videos
os.environ["MUJOCO_GL"] = "egl"

import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from carl.context.context_space import UniformFloatContextFeature
from carl.context.sampler import ContextSampler
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from meta_rl.jrpl.carl_wrapper import context_wrapper
from meta_rl.jrpl.context_encoder import ContextEncoder


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
# JRPL arguments
    parser.add_argument("--context-mode", type=str, default="hidden",
        help="how the context is provided to the agent: hidden, explicit, learned")
    parser.add_argument("--context-encoder", type=str, default="mlp_avg",
        help="if context-mode is learned, the type of context encoder to use")
    parser.add_argument("--emb-dim", type=int, default=2,
        help="the dimension of the context embedding") 
    parser.add_argument("--hidden-encoder-dim", type=int, default=16,
        help="the hidden sizes of the context encoder")
    parser.add_argument("--context-length", type=int, default=20,
        help="length of the encoded context")
    parser.add_argument("--context-name", type=str, default="gravity",
        help="the name of the context feature")
    parser.add_argument("--context_lower", type=float, default=0.2,
        help="lower bound of the context feature as a multiple of the default value")   
    parser.add_argument("--context_upper", type=float, default=2.2,
        help="upper bound of the context feature as a multiple of the default value") 
    
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="JRPL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CARLPendulum",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(seed, sampled_contexts, CARLEnv):
    """
    wrapper for monitoring and seeding envs
    Returns envs with a distribution of the context
    """

    def thunk():
        env = CARLEnv(
            # You can play with different gravity values here
            contexts=sampled_contexts,
            obs_context_as_dict=True,
        )

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, encoder_output_size=0):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape)
            + encoder_output_size,
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


class Actor(nn.Module):
    def __init__(self, env, encoder_output_size=0):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + encoder_output_size,
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
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
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = parse_args()

    context_name = args.context_name
    env_module = importlib.import_module("carl.envs")
    CARLEnv = getattr(env_module, args.env_id)
    print("context mode : ", args.context_mode)

    concat_context = True if args.context_mode == "explicit" else False
    CARLEnv = context_wrapper(
        CARLEnv, context_name=args.context_name, concat_context=concat_context
    )
    context_default = CARLEnv.get_default_context()[context_name]

    l, u = context_default * args.context_lower, context_default * args.context_upper
    l, u = min(l, u), max(l, u)
    context_distributions = [UniformFloatContextFeature(context_name, l, u)]

    context_sampler = ContextSampler(
        context_distributions=context_distributions,
        context_space=CARLEnv.get_context_space(),
        seed=args.seed,
    )
    sampled_contexts = context_sampler.sample_contexts(n_contexts=100)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.seed + i, sampled_contexts=sampled_contexts, CARLEnv=CARLEnv)
            for i in range(args.num_envs)
        ]
    )

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    if args.context_mode == "learned":
        from meta_rl.jrpl.buffer import ReplayBuffer

        context_length = args.context_length
        transitions_dim = (
            2 * np.array(envs.single_observation_space.shape).prod()
            + np.array(envs.single_action_space.shape).prod()
        )
        transitions_dim = int(transitions_dim)
        print("transitions_dim : ", transitions_dim)
        if args.context_encoder == "mlp_avg":
            encoder_output_size = args.emb_dim
        elif args.context_encoder == "mlp_avg_std":
            encoder_output_size = 2 * args.emb_dim
        else:
            raise ValueError("context_encoder must be either mlp_avg or mlp_avg_std")

        context_encoder = ContextEncoder(
            transitions_dim,
            args.emb_dim,
            [args.hidden_encoder_dim, args.hidden_encoder_dim],
        ).to(device)

        actor = Actor(envs, encoder_output_size).to(device)
        qf1 = QNetwork(envs, encoder_output_size).to(device)
        qf1_target = QNetwork(envs, encoder_output_size).to(device)
        target_actor = Actor(envs, encoder_output_size).to(device)
        target_actor.load_state_dict(actor.state_dict())
        qf1_target.load_state_dict(qf1.state_dict())
        q_optimizer = optim.Adam(
            list(qf1.parameters()),
            lr=args.learning_rate,
        )
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    else:
        actor = Actor(envs).to(device)
        qf1 = QNetwork(envs).to(device)
        qf1_target = QNetwork(envs).to(device)
        target_actor = Actor(envs).to(device)
        target_actor.load_state_dict(actor.state_dict())
        qf1_target.load_state_dict(qf1.state_dict())
        q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            if args.context_mode == "learned":
                # sample contexts from each element of the batch
                context_ids = info["context_id"]
                # context_id needs to be an int for now. Throw an error if it is not
                if not isinstance(context_ids, int):
                    raise ValueError("context_id should be an int")
                data = rb.sample(
                    batch_size=1,
                    context_length=context_length,
                    add_context=True,
                    context_id=context_ids,
                )
                # encode the contexts
                context_mu, context_sigma = context_encoder(data.contexts)

                # append the context to the observations
                if args.context_encoder == "mlp_avg":
                    obs_context = torch.cat(
                        [torch.Tensor(obs).to(device), context_mu], dim=-1
                    )
                elif args.context_encoder == "mlp_avg_std":
                    # context_sigma = torch.exp(context_sigma)
                    obs_context = torch.cat(
                        [torch.Tensor(obs).to(device), context_mu, context_sigma],
                        dim=-1,
                    )
                with torch.no_grad():
                    actions = actor(torch.Tensor(obs_context).to(device))
                    actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                    actions = (
                        actions.cpu()
                        .numpy()
                        .clip(envs.single_action_space.low, envs.single_action_space.high)
                    )
            else :
                with torch.no_grad():
                    actions = actor(torch.Tensor(obs).to(device))
                    actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                    actions = (
                        actions.cpu()
                        .numpy()
                        .clip(envs.single_action_space.low, envs.single_action_space.high)
                    )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if args.context_mode == "learned":
                # sample contexts from each element of the batch
                data = rb.sample(args.batch_size, context_length, add_context=True)
                # encode the contexts
                context_mu, context_sigma = context_encoder(data.contexts)
                #context_mu, context_sigma = torch.zeros((args.batch_size, encoder_output_size)), torch.zeros((args.batch_size, encoder_output_size))
                #context_mu, context_sigma = context_mu.to(device), context_sigma.to(device)
                # append the context to the observations
                if args.context_encoder == "mlp_avg":
                    data = data._replace(
                        observations=torch.cat(
                            [data.observations, context_mu], dim=-1
                        )
                    )
                    data = data._replace(
                        next_observations=torch.cat(
                            [data.next_observations, context_mu], dim=-1
                        )
                    )
                elif args.context_encoder == "mlp_avg_std":
                    context_sigma = torch.exp(context_sigma)
                    data = data._replace(
                        observations=torch.cat(
                            [data.observations, context_mu, context_sigma], dim=-1
                        )
                    )
                    data = data._replace(
                        next_observations=torch.cat(
                            [data.next_observations, context_mu, context_sigma],
                            dim=-1,
                        )
                    )
            else:
                data = rb.sample(args.batch_size)

            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward(retain_graph=True)
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:

                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(
                    actor.parameters(), target_actor.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ddpg_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Actor, QNetwork),
            device=device,
            exploration_noise=args.exploration_noise,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "DDPG",
                f"runs/{run_name}",
                f"results/videos/{run_name}-eval",
            )

    envs.close()
    writer.close()
