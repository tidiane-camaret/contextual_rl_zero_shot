import os
import importlib
import hydra
import pprint
from meta_rl.algorithms.sac.sac import train_sac
from meta_rl.algorithms.sac.sac_utils import Args

@hydra.main(version_base=None, config_path='../configs/', config_name='base_exp')
def main(config):
    pprint.pprint(config)
    print("Working directory : {}".format(os.getcwd()))
    args = Args() 
    # we keep the Args class because it keeps all default values and descriptions
    # but we could migrate those to hydra config files in the future
    args.total_timesteps = config.train.total_timesteps
    args.env_id = config.env.id
    args.seed = config.seed
    args.track = config.wandb.track
    args.wandb_project_name = config.wandb.project_name
    args.wandb_entity = config.wandb.entity
    #print("Total timesteps : {}".format(args.total_timesteps))
    if args.env_id == 'ComplexODEBoundedReward':
        from meta_rl.envs.genrlise.complex_ode_bounded_reward import ComplexODEBoundedReward
        env = ComplexODEBoundedReward([1, 1], 1)
    elif "CARL" in args.env_id :
        from meta_rl.jrpl.carl_wrapper import context_wrapper
        from carl.context.context_space import UniformFloatContextFeature
        from carl.context.sampler import ContextSampler
        env_module = importlib.import_module("carl.envs")
        CARLEnv = getattr(env_module, args.env_id)
        CARLEnv = context_wrapper(
            CARLEnv, context_name=config.context.name, concat_context=config.context.explicit
        )

        context_name = config.context.name

        context_default = CARLEnv.get_default_context()[context_name]

        # mu, rel_sigma = 10, 5
        # context_distributions = [NormalFloatContextFeature(context_name, mu, rel_sigma*mu)]
        l, u = context_default * config.context.lower_bound, context_default * config.context.upper_bound
        l, u = min(l, u), max(l, u)
        context_distributions = [UniformFloatContextFeature(context_name, l, u)]

        context_sampler = ContextSampler(
            context_distributions=context_distributions,
            context_space=CARLEnv.get_context_space(),
            seed=args.seed,
        )
        sampled_contexts = context_sampler.sample_contexts(n_contexts=100)
        env = CARLEnv(contexts=sampled_contexts,)
        
    train_sac(env, args)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        main()  # data processing might error out due to multiple jobs doing the same thing
        print(e)