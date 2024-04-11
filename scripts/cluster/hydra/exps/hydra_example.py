from omegaconf import OmegaConf
import hydra
import submitit
"""
This is a minimal working example of a Hydra app.
Run locally:
    python3 scripts/cluster/hydra/hydra_example.py

Running using the submitit plugin:
    python3 scripts/cluster/hydra/hydra_example.py --multirun hydra/launcher=submitit_slurm
"""
@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg):
    #env = submitit.JobEnvironment()
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()