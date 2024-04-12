import importlib
import random
import time
import os
import glob
import gymnasium as gym
import numpy as np
import torch

from meta_rl.algorithms.sac.sac_utils import (
    Actor,
    Args,
    get_latent_context_sac,
    make_env,
)
from meta_rl.jcpl.carl_wrapper import context_wrapper

args = Args()




#cartpole

args.date_dir = "2024-02-08/13-15-57"
args.env_max_episode_steps = 200
args.context_name = "tau"
args.env_id = "CARLCartPoleContinuous"
args.eval_context_values = [0.002, 0.0033362 , 0.00556512, 0.00928318, 0.01548527, 0.02583099, 0.04308869, 0.07187627, 0.11989685, 0.2]

# pendulum
args.date_dir = "2024-02-09/13-33-44"
args.env_max_episode_steps = 200
args.context_name = "l"
args.env_id = "CARLPendulum"
args.eval_context_values = [0.002     , 0.0033362 , 0.00556512, 0.00928318, 0.01548527, 0.02583099, 0.04308869, 0.07187627, 0.11989685, 0.2       ]
"""
#mountain_car
args.env_max_episode_steps = 999
args.context_name = "power"
args.env_id = "CARLMountainCarContinuous"
args.eval_context_values = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
args.date_dir = "2024-02-09/18-52-53"
"""

#ant
args.date_dir = "2024-02-07/20-31-42"
args.context_name = "mass_torso"
args.env_id = "CARLBraxAnt"
args.env_max_episode_steps = 1000
args.eval_context_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

## loop over environments
for emes, cn, ei, ecv, dd in zip([999], 
                                 ["mass_torso"], 
                                 ["CARLBraxAnt"], 
                                 [[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]],
                                 [ "2024-02-07/20-31-42"]):
    args.date_dir = dd
    args.env_max_episode_steps = emes
    args.context_name = cn
    args.env_id = ei
    args.eval_context_values = ecv
    for cm in ["learned_jcpl"]:
            args.context_mode = cm
            eval_context_value = 70
            args.latent_context_dim = 2
            args.nb_input_transitions = 20
            args.encoder_hidden_sizes = [8, 4]

            run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
            print(f"run_name: {run_name}")
            print("context_mode : ", args.context_mode)

            models_path_dict = {}
            for dir in glob.glob("results/hydra/multirun/"+args.date_dir +"/" + args.env_id + "/" + args.context_mode + "/*"):
                training_seed = dir.split('/')[-1]
                for run_nb in glob.glob(dir + '/*'):
                    models_path_dict[training_seed] = run_nb+"/results/models/"


            MSEs = []

            for training_seed, models_path in models_path_dict.items():

                # TRY NOT TO MODIFY: seeding
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.backends.cudnn.deterministic = args.torch_deterministic

                device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

                # env setup
                if args.env_id == "CARLCartPoleContinuous":
                    # custom environment, we need to import it directly
                    from meta_rl.envs.carl_cartpole import CARLCartPoleContinuous

                    CARLEnv = CARLCartPoleContinuous
                else:
                    env_module = importlib.import_module("carl.envs")
                    CARLEnv = getattr(env_module, args.env_id)
                CARLEnv = context_wrapper(
                    CARLEnv,
                    context_name=args.context_name,
                    concat_context=(args.context_mode == "explicit"),
                )
                eval_context = CARLEnv.get_default_context()
                eval_context[args.context_name] = eval_context_value
                env = CARLEnv(
                    # You can play with different gravity values here
                    contexts={0: eval_context},
                )

                envs = gym.vector.SyncVectorEnv([make_env(env, args.seed, 0, False, run_name)])
                assert isinstance(
                    envs.single_action_space, gym.spaces.Box
                ), "only continuous action space is supported"


                # jcpl : if context is learned from transitions, use custom replay buffer
                # and create context encoder

                from meta_rl.jcpl.context_encoder import ContextEncoder

                latent_context_dim = args.latent_context_dim
                nb_input_transitions = args.nb_input_transitions
                transitions_dim = int(
                    2 * np.array(envs.single_observation_space.shape).prod()
                    + np.array(envs.single_action_space.shape).prod()
                )
                context_encoder = ContextEncoder(
                    d_in=transitions_dim,
                    d_out=latent_context_dim,
                    hidden_sizes=args.encoder_hidden_sizes,
                ).to(device)



                actor = Actor(envs, latent_context_dim).to(device)


                actor_model_path = [f for f in os.listdir(models_path) if f.startswith("sac_actor")][0]
                actor.load_state_dict(torch.load(models_path + actor_model_path, map_location=device))
                context_encoder_model_path = [f for f in os.listdir(models_path) if f.startswith("context_encoder")][0]
                context_encoder.load_state_dict(torch.load(models_path + context_encoder_model_path, map_location=device))


                latent_contexts = get_latent_context_sac(envs, actor, context_encoder, args)

                envs.close()

                latent_contexts = []
                context_values = []


                for eval_context_value in args.eval_context_values:
                    #print("eval_context_value : ", eval_context_value)
                    eval_context = CARLEnv.get_default_context()
                    eval_context[args.context_name] = eval_context_value
                    eval_env = CARLEnv(
                        contexts={0: eval_context},
                        )
                    eval_env = gym.vector.SyncVectorEnv([make_env(eval_env, args.seed, 0, False, run_name)])
                    l = get_latent_context_sac(eval_env, actor, context_encoder, args)
                    # sample 20 latent contexts for each eval context value
                    l = random.sample(l, min(20, len(l)))
                    latent_contexts.extend(l)

                    context_values.extend([eval_context_value] * len(l))

                import matplotlib.pyplot as plt

                # Convert the latent contexts and context values to numpy arrays
                latent_contexts = np.array(latent_contexts)
                context_values = np.array(context_values)

                # Scatter plot the latent contexts
                plt.scatter(latent_contexts[:, 0], latent_contexts[:, 1], c=context_values)
                plt.colorbar(label=args.context_name)
                plt.xlabel("Latent Dimension 1")
                plt.ylabel("Latent Dimension 2")
                plt.title("Latent Contexts")

                fig_name = f"latent_contexts_{args.env_id}_{args.context_name}_{args.context_mode}_{training_seed}.png"
                plt.savefig("results/latents/" + fig_name)
                # delete the plot
                plt.close()

                # train a regressor to predict the context value from the latent context
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_squared_error

                # cross-validation 5 times
                mse = []
                for i in range(5):
                    X_train, X_test, y_train, y_test = train_test_split(
                        latent_contexts, context_values, test_size=0.2
                    )
                    regressor = RandomForestRegressor(n_estimators=100)
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    mse.append(mean_squared_error(y_test, y_pred))
                #print(f"MSE: {mse}")


                MSEs.extend(mse)

            print(f"Mean MSEs: {np.mean(MSEs)}")
            print(f"Std MSEs: {np.std(MSEs)}")



