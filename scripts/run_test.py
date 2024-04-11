import os
import pprint

import hydra


@hydra.main(version_base=None, config_path="../configs/", config_name="base_exp")
def main(config):
    pprint.pprint(config)
    print("Working directory : {}".format(os.getcwd()))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        main()  # data processing might error out due to multiple jobs doing the same thing
        print(e)
