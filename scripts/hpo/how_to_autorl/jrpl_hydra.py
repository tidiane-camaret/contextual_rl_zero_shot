"""
Run an HPO using Hydra, via the how_to_autorl package.
https://github.com/facebookresearch/how-to-autorl/
"""

from automl.meta_rl.meta_rl.jrpl.dqn import parse_args, train_agent
from meta_rl.meta_rl.jrpl.context_encoder import ContextEncoder
from meta_rl.meta_rl.jrpl.carl_wrapper import context_wrapper
import importlib
import numpy as np

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

            poetry run pip install "stable_baselines3==2.0.0a1"
            """
        )
    args = parse_args()
    env_module = importlib.import_module("carl.envs")
    CARLEnv = getattr(env_module, args.env_id)
    print("context mode : ", args.context_mode)
    context_name = args.context_name 

    concat_context = True if args.context_mode == "explicit" else False
    CARLEnv = context_wrapper(CARLEnv, 
                          context_name = args.context_name, 
                          concat_context = concat_context)
    episodic_returns = train_agent(args, CARLEnv)
    print("episodic_returns : ", np.asarray(episodic_returns).mean())