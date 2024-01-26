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
    args.autotune = config.sac_params.autotune_entropy
    

    # Additional context-related arguments
    # added for logging purposes
    args.context_name = config.context.name
    args.context_lower_bound_coeff = config.context.lower_bound_coeff
    args.context_upper_bound_coeff = config.context.upper_bound_coeff
    args.context_mode = config.context.mode

    # context encoder arguments
    args.nb_input_transitions = config.context_encoder.nb_input_transitions
    args.encoder_hidden_sizes = config.context_encoder.hidden_sizes
    args.latent_context_dim = config.context_encoder.latent_context_dim

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
            CARLEnv, 
            context_name=config.context.name, 
            concat_context= (config.context.mode == 'explicit'),
        )

        context_name = config.context.name

        context_default = CARLEnv.get_default_context()[context_name]

        # mu, rel_sigma = 10, 5
        # context_distributions = [NormalFloatContextFeature(context_name, mu, rel_sigma*mu)]
        lower_bound, upper_bound = context_default * config.context.lower_bound_coeff, context_default * config.context.upper_bound_coeff
        lower_bound, upper_bound = min(lower_bound, upper_bound), max(lower_bound, upper_bound)
        context_distributions = [UniformFloatContextFeature(context_name, lower_bound, upper_bound)]

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