"""function to train a dqn agent
Args:
    cfg (DictConfig): configuration file
"""

import hydra
import logging
import importlib
import sys
import os
import numpy as np
sys.path.append(os.path.abspath("/home/ndirt/dev/automl"))
from omegaconf import DictConfig, OmegaConf 
from meta_rl.jrpl.dqn import parse_args, train_agent
from meta_rl.jrpl.context_encoder import ContextEncoder
from meta_rl.jrpl.carl_wrapper import context_wrapper


log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="dqn_cartpole_pb2")
def train_cleanrl(cfg: DictConfig):
   
    log.info(OmegaConf.to_yaml(cfg))
    
    args = parse_args()
    args.total_timesteps = int(cfg.algorithm.total_timesteps)
    args.learning_rate = cfg.algorithm.model_kwargs.learning_rate
    args.batch_size = cfg.algorithm.model_kwargs.batch_size
    args.tau = cfg.algorithm.model_kwargs.tau
    args.gamma = cfg.algorithm.model_kwargs.gamma
    args.learning_starts = cfg.algorithm.model_kwargs.learning_starts
    args.train_frequency = cfg.algorithm.model_kwargs.train_freq
    #args.gradient_steps = cfg.algorithm.model_kwargs.gradient_steps
    #TODO : see what gradient steps is
    args.exploration_fraction = cfg.algorithm.model_kwargs.exploration_fraction
    args.start_e = cfg.algorithm.model_kwargs.exploration_initial_eps
    args.end_e = cfg.algorithm.model_kwargs.exploration_final_eps
    args.buffer_size = cfg.algorithm.model_kwargs.buffer_size

    #log.info(f"Training agent with lr = {args.learning_rate} for {args.total_timesteps} steps")
    env_module = importlib.import_module("carl.envs")
    CARLEnv = getattr(env_module, args.env_id)
    print("context mode : ", args.context_mode)

    concat_context = True if args.context_mode == "explicit" else False
    CARLEnv = context_wrapper(CARLEnv, 
                          context_name = args.context_name, 
                          concat_context = concat_context)
    episodic_returns = train_agent(args, CARLEnv)
    print("episodic_returns : ", np.asarray(episodic_returns).mean())
    objective = - np.asarray(episodic_returns).mean()
    return objective

if __name__ == "__main__":
    train_cleanrl()
