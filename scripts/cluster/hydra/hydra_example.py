import hydra
import pprint
import os
import sys
import pathlib
sys.path.insert(1, pathlib.Path(__file__).parent.parent.absolute().as_posix())

""" This is a minimal working example of a Hydra app.

single run with default config parameters
python3 scripts/cluster/hydra/hydra_example.py

sweeper : add the --multirun flag
"""

@hydra.main(version_base=None, config_path='.', config_name='config')
def main(config):
    pprint.pprint(config)
    print("Working directory : {}".format(os.getcwd()))
    print("model : {}".format(config.model.name))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        main()  # data processing might error out due to multiple jobs doing the same thing
        print(e)