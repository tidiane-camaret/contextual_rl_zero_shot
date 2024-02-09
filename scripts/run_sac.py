import importlib
import os

import hydra

from meta_rl.algorithms.sac.sac import train_sac
from meta_rl.algorithms.sac.sac_utils import Args


@hydra.main(version_base=None, config_path="../configs/", config_name="base_exp")
def main(config):
    # pprint.pprint(config)
    print("Working directory : {}".format(os.getcwd()))
    args = Args()
    # we keep the Args class because it keeps all default values and descriptions
    # but we could migrate those to hydra config files in the future
    args.total_timesteps = config.train.total_timesteps
    args.env_id = config.env.id
    args.env_max_episode_steps = config.env.max_episode_steps
    args.seed = config.seed
    args.track = config.wandb.track
    args.capture_video = config.capture_video
    args.wandb_project_name = config.wandb.project_name
    args.wandb_entity = config.wandb.entity
    args.autotune = config.sac_params.autotune_entropy
    args.train_context_values = config.context.train_values

    # evaluation arguments
    args.eval_context_values = config.context.eval_values
    args.nb_evals_per_seed = config.context.nb_evals_per_seed

    # Additional context-related arguments
    # added for logging purposes
    args.context_name = config.context.name
    args.context_mode = config.context.mode

    # context encoder arguments
    args.nb_input_transitions = config.context_encoder.nb_input_transitions
    args.encoder_hidden_sizes = config.context_encoder.hidden_sizes
    args.latent_context_dim = config.context_encoder.latent_context_dim

    assert (
        "CARL" in args.env_id
    ), "Only CARL environments are supported for context-based training"

    from meta_rl.jrpl.carl_wrapper import context_wrapper

    if args.env_id == "CARLCartPoleContinuous":
        # custom environment, we need to import it directly
        from meta_rl.envs.carl_cartpole import CARLCartPoleContinuous

        CARLEnv = CARLCartPoleContinuous
    else:
        env_module = importlib.import_module("carl.envs")
        CARLEnv = getattr(env_module, args.env_id)

    CARLEnv = context_wrapper(
        CARLEnv,
        context_name=config.context.name,
        concat_context=(config.context.mode == "explicit"),
    )
    context_default = CARLEnv.get_default_context()[args.context_name]

    if args.context_mode == "default_value":
        # train on default context only
        train_envs = CARLEnv(
            context={0: {args.context_name: context_default}}
        )  # not sure if even needed

    elif args.context_mode in ["learned_jrpl", "explicit", "hidden", "learned_iida"]:
        train_contexts = dict()
        for i, train_context_value in enumerate(args.train_context_values):
            print("train_context_value : ", train_context_value)
            c = CARLEnv.get_default_context()
            c[args.context_name] = train_context_value
            train_contexts[i] = c

        train_envs = CARLEnv(
            # You can play with different gravity values here
            contexts=train_contexts,
        )

    else:
        raise ValueError(f"Unknown context mode {args.context_mode}")

    eval_envs = {}

    for eval_context_value in args.eval_context_values:
        print("eval_context_value : ", eval_context_value)
        eval_context = CARLEnv.get_default_context()
        eval_context[args.context_name] = eval_context_value
        eval_envs[eval_context_value] = CARLEnv(
            # You can play with different gravity values here
            contexts={0: eval_context},
        )

    train_sac(train_envs, args, eval_envs=eval_envs)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        main()  # data processing might error out due to multiple jobs doing the same thing
        print(e)

    """
    context_name = config.context.name
    
    # bounds for the context distributions
    lower_bound, upper_bound = (
        context_default * config.context.lower_bound_coeff,
        context_default * config.context.upper_bound_coeff,
    )
    lower_bound, upper_bound = min(lower_bound, upper_bound), max(
        lower_bound, upper_bound
    )

    # train on a context distribution
    context_distributions = [
        UniformFloatContextFeature(context_name, lower_bound, upper_bound)
    ]

    context_sampler = ContextSampler(
        context_distributions=context_distributions,
        context_space=CARLEnv.get_context_space(),
        seed=args.seed,
    )
    sampled_contexts = context_sampler.sample_contexts(n_contexts=100)
    train_envs = CARLEnv(
        contexts=sampled_contexts,
    )

    args.sampled_contexts = sampled_contexts
    """
