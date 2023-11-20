# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys
import hydra
from meta_rl.jrpl.hpo_agent import train_cleanrl



HYDRA_PLUGIN_PATH = "/home/ndirt/dev/automl/how-to-autorl/hydra_plugins"
# add the plugin path to sys.path
sys.path.append(HYDRA_PLUGIN_PATH)

hydra.initialize()

@hydra.main(config_path="configs", config_name="dqn_cartpole_dehb")
def run_dqn_dehb(cfg):
    return train_cleanrl(cfg)


if __name__ == "__main__":
    run_dqn_dehb()


