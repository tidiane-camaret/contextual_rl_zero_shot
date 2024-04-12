import importlib
import random
import time

import gymnasium as gym
import numpy as np
import torch

from meta_rl.algorithms.sac.sac_utils import (
    Actor,
    Args,
    eval_sac,
    make_env,
)
from meta_rl.jcpl.carl_wrapper import context_wrapper

args = Args()
model_path = "../results/hydra/multirun/2024-02-07/20-31-42/CARLBraxAnt/explicit/0/10/results/models/sac_actor_CARLBraxAnt__sac_utils__0__1707337326.pt"
args.context_name = "mass_torso"
args.env_id = "CARLBraxAnt"
args.context_mode = "explicit"
eval_context_value = 70
args.env_max_episode_steps = 1000

run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"


# TRY NOT TO MODIFY: seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# env setup
env_module = importlib.import_module("carl.envs")
CARLEnv = getattr(env_module, args.env_id)
CARLEnv = context_wrapper(
    CARLEnv,
    context_name=args.context_name,
    concat_context=(args.context_mode == "explicit"),
)
eval_context = CARLEnv.get_default_context()
eval_context[args.context_name] = eval_context_value
env = CARLEnv(
    # You can play with different gravity values here
    contexts={0: eval_context},
)

envs = gym.vector.SyncVectorEnv([make_env(env, args.seed, 0, True, run_name)])
assert isinstance(
    envs.single_action_space, gym.spaces.Box
), "only continuous action space is supported"


# jcpl : if context is learned from transitions, use custom replay buffer
# and create context encoder
if "learned" in args.context_mode:

    from meta_rl.jcpl.context_encoder import ContextEncoder

    latent_context_dim = args.latent_context_dim
    nb_input_transitions = args.nb_input_transitions
    transitions_dim = int(
        2 * np.array(envs.single_observation_space.shape).prod()
        + np.array(envs.single_action_space.shape).prod()
    )
    context_encoder = ContextEncoder(
        d_in=transitions_dim,
        d_out=latent_context_dim,
        hidden_sizes=args.encoder_hidden_sizes,
    ).to(device)
else:
    latent_context_dim = 0
    context_encoder = None


actor = Actor(envs, latent_context_dim).to(device)
actor.load_state_dict(torch.load(model_path, map_location=device))

eval_sac(envs, actor, context_encoder, args)

envs.close()


