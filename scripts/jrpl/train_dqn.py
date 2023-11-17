"""
    function to train a dqn agent
"""

import importlib
import logging

import numpy as np
from meta_rl.jrpl.carl_wrapper import context_wrapper
from meta_rl.jrpl.dqn import parse_args, train_agent

log = logging.getLogger(__name__)


def train_dqn():
    args = parse_args()

    log.info(
        f"Training agent with lr = {args.learning_rate} for {args.total_timesteps} steps"
    )
    env_module = importlib.import_module("carl.envs")
    CARLEnv = getattr(env_module, args.env_id)
    print("context mode : ", args.context_mode)

    concat_context = True if args.context_mode == "explicit" else False
    CARLEnv = context_wrapper(
        CARLEnv, context_name=args.context_name, concat_context=concat_context
    )
    episodic_returns = train_agent(args, CARLEnv)
    print("episodic_returns : ", np.asarray(episodic_returns).mean())
    objective = -np.asarray(episodic_returns).mean()
    return objective


if __name__ == "__main__":
    train_dqn()
