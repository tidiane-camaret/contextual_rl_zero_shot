# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
import importlib
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

env_module = importlib.import_module("carl.envs")
# discrete actions : CARLLunarLander, CARLCartPole
# continuous actions : CARLPendulum
from carl.context.context_space import NormalFloatContextFeature, UniformFloatContextFeature
from carl.context.sampler import ContextSampler

from carl_wrapper import context_wrapper
#from stable_baselines3.common.buffers import ReplayBuffer
from context_encoder import ReplayBuffer, ContextEncoder


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dim", type=int, default=2,
        help="the dimension of the context embedding") 
    parser.add_argument("--context_state", type=str, default="implicit",
        help="the state of the context feature")
    parser.add_argument("--context_name", type=str, default="gravity",
        help="the name of the context feature")
    parser.add_argument("--num_envs", type=int, default=1,
        help="the number of parallel game environments")
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
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
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
    parser.add_argument("--env-id", type=str, default="CARLCartPole",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    args = parser.parse_args()
    # fmt: on
    #assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args







# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, encoder_output_size=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod() + encoder_output_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def make_env(seed, sampled_contexts):
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

if __name__ == "__main__":
    import stable_baselines3 as sb3

    

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = parse_args()
    CARLEnv = getattr(env_module, args.env_id)
    context_name = args.context_name 

    CARLEnv = context_wrapper(CARLEnv, 
                          context_name = context_name, 
                          concat_context = False)

    context_default = CARLEnv.get_default_context()[context_name]
    
    #mu, rel_sigma = 10, 5
    #context_distributions = [NormalFloatContextFeature(context_name, mu, rel_sigma*mu)]            
    l, u = context_default * 0.2, context_default * 2.2
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
    writer = SummaryWriter(f"results/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"device: {device}")
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, sampled_contexts=sampled_contexts) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"



    context_size = 20 #args.batch_size
    context_dim = 2*np.array(envs.single_observation_space.shape).prod() + np.array(envs.single_action_space.shape).prod()
    context_dim = int(context_dim)
    if args.context_state == "implicit":
        encoder_output_size = args.emb_dim 
    elif args.context_state == "implicit_std":
        encoder_output_size = 2*args.emb_dim
    else:
        raise ValueError("context_state must be either implicit or implicit_std")

    q_network = QNetwork(envs, encoder_output_size).to(device)
    
    target_network = QNetwork(envs, encoder_output_size).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # define the context encoder
    context_encoder = ContextEncoder(context_dim, args.emb_dim, [32, 32]).to(device)
    optimizer = optim.Adam(list(q_network.parameters()) + list(context_encoder.parameters()), lr=args.learning_rate)
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        n_envs=args.num_envs,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # sample contexts from each element of the batch
            context_ids = info["context_id"]
            # if context_ids is an int, convert it to a list
            if isinstance(context_ids, int):
                context_ids = [context_ids]
            contexts = [rb.sample(context_size, context_id=c) for c in context_ids]
            context_tensor = torch.zeros((len(context_ids), context_size, context_dim), dtype=torch.float32, device=device)
            for i, context in enumerate(contexts):
                context_tensor[i] = torch.cat([context.observations, context.actions, context.next_observations], dim=-1)       
            
            
            # encode the contexts   
            context_mu, context_sigma = context_encoder(context_tensor)
            # append the context to the observations

            if args.context_state == "implicit":
                obs_context = torch.cat([torch.Tensor(obs).to(device), context_mu], dim=-1)
            elif args.context_state == "implicit_std":
                context_sigma = torch.exp(context_sigma)
                obs_context = torch.cat([torch.Tensor(obs).to(device), context_mu, context_sigma], dim=-1)

            q_values = q_network(obs_context)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

            # remove the context from the observations
            #obs = obs[:, :-encoder_output_size]
            #obs = obs.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None or "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                #copy

                # sample contexts from each element of the batch
                contexts = [rb.sample(context_size, context_id=c.item()) for c in data.context_ids]
                context_tensor = torch.zeros((args.batch_size, context_size, context_dim), dtype=torch.float32, device=device)
                for i, context in enumerate(contexts):
                    context_tensor[i] = torch.cat([context.observations, context.actions, context.next_observations], dim=-1)

                # encode the contexts
                context_mu, context_sigma = context_encoder(context_tensor)
                # append the context to the observations
                if args.context_state == "implicit":
                    data = data._replace(observations = torch.cat([data.observations, context_mu], dim=-1))
                    data = data._replace(next_observations = torch.cat([data.next_observations, context_mu], dim=-1))
                elif args.context_state == "implicit_std":
                    context_sigma = torch.exp(context_sigma)
                    data = data._replace(observations = torch.cat([data.observations, context_mu, context_sigma], dim=-1))
                    data = data._replace(next_observations = torch.cat([data.next_observations, context_mu, context_sigma], dim=-1))
                
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    #print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"results/runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()

    # plot encoder representations

    contexts_in_rb = rb.context_ids
    context_values = []
    context_embs = []
    # filter out the unique contexts
    contexts_in_rb = np.unique(contexts_in_rb)
    for context_id in contexts_in_rb:
        context_value = sampled_contexts[context_id][context_name]
        context = rb.sample(context_size, context_id=context_id)
        context_tensor = torch.cat([context.observations, context.actions, context.next_observations], dim=-1)
        context_tensor = context_tensor.unsqueeze(0)
        context_mu, context_sigma = context_encoder(context_tensor)
        context_values.append(context_value)
        context_embs.append(context_mu.detach().cpu().numpy())

    # plot the context embeddings
    context_values = np.array(context_values)
    context_embs = np.array(context_embs)
    import matplotlib.pyplot as plt
    plt.scatter(context_embs[:, 0, 0], context_embs[:, 0, 1], c=context_values)
    plt.colorbar()
    plt.title(f"Context embeddings for {context_name} using {args.context_state}")

    plt.savefig(f"results/runs/dqn_embeddings_{args.context_state}_{args.seed}.png")

"""
    steps = 0
    while True:
        # get action from agent
        q_values = q_network(torch.Tensor(obs).to(device))
        action = torch.argmax(q_values).cpu().numpy()

        obs, r, done, truncated, info = env.step(action)
        env.render()
        steps += 1
        if done or steps > 100:
            break

    env.close()

"""