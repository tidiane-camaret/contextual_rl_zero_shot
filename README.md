# meta_rl

This repo contains a series of experiments on meta reinforcement learning, aiming to benchmark several meta learning frameworks for reinforcement learning tasks.

Typically, an RL agent is trained on a list of training tasks, and is then evaluated on a list of evaluation tasks. 

The goals of this project are :  
- To create fair and reproducible evaluation methods for the generalization ability of agents 
- To find ways to assess the **level of uncertainty** of an agent when exposed to a task.

The current setup uses a range of RL environments, mostly from from the [OpenAI Gym](https://gym.openai.com/) library, with different dynamics. The environments are generated with different dynamics using [CARL](https://github.com/automl/CARL) and the agent is trained/evaluated on subsets of those environments.

## List of environments used :
* [X] CartPole-v1 (DQN)
    * [X] tau * uniform(0.2,2.2)
    * [X] length * uniform(0.2,2.2)
    * [X] gravity * uniform(0.2,2.2)
* [X] LunarLander-v2 (DQN)
    * [X] gravity * uniform(0.1,2.2)
* [ ] MountainCar-v0 (DQN)
    * [ ] gravity * uniform(0.1,2.2)
* [] Striker (PPO)
    * [] gravity * uniform(0.1,2.2)
* [] Pendulum-v0 (DDPG)
    * [] length * uniform(0.5,2.2)
* [] [Meta World](https://arxiv.org/abs/1910.10897) (PPO)

## Baselines

* [X] Explicit context : the dynamics are given as input to the model as additional state data, both at training and testing time.
* [X] No context : no dynamics are given as input to the model, neither at training nor testing time.
* [] [Context is Everything](https://benevans.zip/iida/) : A predictor model is trained to predict next states from the current state and the action taken. The predictor model is then used as a **context encoder** for the RL agent, which is trained on the training environments. The RL agent is then tested on the testing environments.
* [ ] [Environment Probing Interaction Policies](https://openreview.net/pdf?id=ryl8-3AcFX) : Similar architecture, but the context encoder uses an additional RL agent to generate trajectories from the training environments.
* [X] **Joint Representation and Policy learning (JRPL)**: Similar architecture, but the context encoder is trained jointly with the RL agent on the training environments.


# Usage

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run experiments

### Train DQN methods

```bash
python3 scripts/jrpl/train_dqn.py
```

### Hyperparameter optimization on DQN methods

```bash
python3 scripts/hpo/how_to_autorl/dehb_for_cartpole_dqn_jrpl.py --multirun
```

# Roadmap
* [X] Implement DQN for explicit context, no context and JRPL
* [X] Implement HPO pipelines ([how-to-autorl](https://github.com/facebookresearch/how-to-autorl))
    * [X] Run the pipeline locally/on an slurm interactive session
    * [ ] Make it runnable via slurm jobs using submitit (issues with argparse)
* [ ] Implement evaluation pipelines (hydra, submitit)
* [ ] Standardize and document experiments
* [ ] Implement DDQN for explicit context, no context and JRPL
* [ ] Implement Context is everyting and EPI baselines
* [ ] Implement PPO for explicit context, no context and JRPL


# Previous experiments

## Evaluation of baseline models

We evaluate a model using sets of training and testing environments, with non overlapping sets of dynamics.

Environments can be generated with different dynamics, namely mass, inertia and damping. (see meta_rl/envs/striker_avg.py) The range of those values are taken from [Environment Probing Interaction Policies](https://openreview.net/pdf?id=ryl8-3AcFX).

The project currently contains two baseline models :

- "explicit context" : the dynamics are given as input to the model as additional state data, both at training and testing time.

```bash
scripts/metarl_striker.py --context explicit
```

- "no context" : no dynamics are given as input to the model, neither at training nor testing time.

```bash
scripts/metarl_striker.py --context none
```

# Additional models

- [Context is Everything](https://benevans.zip/iida/)

A previoulsy trained **generator policy** generates trajectories from training environments. A predictor model is then trained to predict next states from the current state and the action taken. The predictor model is then used as a **context encoder** for the RL agent, which is trained on the training environments. The RL agent is then tested on the testing environments.

```bash
scripts/iida/genereate_trajectories.py
scripts/iida/train_predictor.py
scripts/metarl_striker.py --context latent
```

