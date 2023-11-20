# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import hydra
from meta_rl.jrpl.hpo_agent import train_cleanrl

# In order to run, the hydra_plugins containing DEHB must be in the root directory of the project


@hydra.main(config_path="configs", config_name="dqn_cartpole_dehb")
def run_dqn_dehb(cfg):
    return train_cleanrl(cfg)


if __name__ == "__main__":
    run_dqn_dehb()
