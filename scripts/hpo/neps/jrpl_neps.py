# HPO pipeline for JRPL

import logging
import time
import gymnasium as gym
import numpy as np
import torch
import neps
from torch.utils.tensorboard import SummaryWriter
import random
# allows do dynamically import modules from the carl.envs folder
import importlib
env_module = importlib.import_module("carl.envs")

from carl.context.context_space import NormalFloatContextFeature, UniformFloatContextFeature
from carl.context.sampler import ContextSampler
from meta_rl.jrpl.carl_wrapper import context_wrapper
from meta_rl.jrpl.context_encoder import ContextEncoder
from automl.meta_rl.scripts.jrpl.dqn import parse_args
pipeline_space = dict(
    lr=neps.FloatParameter(lower=0.0001, upper=0.1, log=True),
)

def run_pipeline(lr):
        import stable_baselines3 as sb3
        start = time.time()
        """

        args = parse_args()
        args.learning_rate = lr
        print('learning rate : ', args.learning_rate)
        CARLEnv = getattr(env_module, args.env_id)
        from scripts.jrpl.dqn_wrapped import make_env, QNetwork, linear_schedule, eval_agent

        print("context mode : ", args.context_mode)
        context_name = args.context_name 

        concat_context = True if args.context_mode == "explicit" else False


        CARLEnv = context_wrapper(CARLEnv, 
                            context_name = context_name, 
                            concat_context = concat_context)

        context_default = CARLEnv.get_default_context()[context_name]
        
        #mu, rel_sigma = 10, 5
        #context_distributions = [NormalFloatContextFeature(context_name, mu, rel_sigma*mu)]            
        l, u = context_default * args.context_lower, context_default * args.context_upper
        context_distributions = [UniformFloatContextFeature(context_name, min(l,u), max(l,u))]
        
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
        print("device : ", device)

        # env setup
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.seed + i, sampled_contexts=sampled_contexts) for i in range(args.num_envs)]
        )
        
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        
        sum_returns = eval_agent(args, envs)
        print("sum returns : ", sum_returns)
        """        
        sum_returns = lr
        end = time.time()

        return {
            "loss": -sum_returns,
                "info_dict": {  # Optionally include additional information as an info_dict
                    "train_time": end - start,
                            },
        }



logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/hpo_neps/jrpl",
    max_evaluations_total=50,
    ignore_errors=True
)