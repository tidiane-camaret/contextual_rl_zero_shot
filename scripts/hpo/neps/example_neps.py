import logging
import time

import numpy as np

import neps
import importlib    
from meta_rl.jrpl.carl_wrapper import context_wrapper
env_module = importlib.import_module("carl.envs")
CARLEnv = getattr(env_module, "CARLCartPole")
CARLEnv = context_wrapper(CARLEnv, 
                        context_name = "gravity", 
                        concat_context = True)

def run_pipeline(learning_rate, float2, categorical, integer1, integer2):
    import scripts.jrpl.dqn_wrapped as dqn_wrapped
    
    
    from meta_rl.jrpl.context_encoder import ContextEncoder
    from carl.context.context_space import NormalFloatContextFeature, UniformFloatContextFeature
    from carl.context.sampler import ContextSampler
    import torch, random
    from torch.utils.tensorboard import SummaryWriter
    import gymnasium as gym
    
    args = dqn_wrapped.parse_args()
    args.learning_rate = learning_rate
    print('learning rate : ', args.learning_rate)
    
    print("context mode : ", args.context_mode)
    context_name = args.context_name 

    concat_context = True if args.context_mode == "explicit" else False



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
        [dqn_wrapped.make_env(args.seed + i, sampled_contexts=sampled_contexts, CARLEnv=CARLEnv) for i in range(args.num_envs)]
    )
    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    sum_returns = dqn_wrapped.eval_agent(args, envs)
    print("sum returns : ", sum_returns)
    loss = -float(np.sum([learning_rate, float2, int(categorical), integer1, integer2]))
    time.sleep(2)  # For demonstration purposes only
    return loss


pipeline_space = dict(
    learning_rate=neps.FloatParameter(lower=0.00001, upper=0.1, log=True),
    float2=neps.FloatParameter(lower=-10, upper=10),
    categorical=neps.CategoricalParameter(choices=[0, 1]),
    integer1=neps.IntegerParameter(lower=0, upper=1),
    integer2=neps.IntegerParameter(lower=1, upper=1000, log=True),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/hyperparameters_example_7",
    max_evaluations_total=15,
)