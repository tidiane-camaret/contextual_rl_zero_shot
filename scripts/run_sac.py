import os
import tyro
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
    train_sac(args)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        main()  # data processing might error out due to multiple jobs doing the same thing
        print(e)