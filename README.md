# Meta RL : Learning Context in a Zero-Shot setting



The current setup uses a range of RL environments, mostly from from the [OpenAI Gym](https://gym.openai.com/) library, with different dynamics. The environments are generated with different dynamics using [CARL](https://github.com/automl/CARL) and the agent is trained/evaluated on subsets of those environments.

## Base algorithms

* [X] DQN (discrete action space)
* [X] DDQN (discrete action space)
* [X] SAC (continuous action space)
* [ ] PPO (continuous action space)

## List of environments
<!---
**training environments** : default context value * uniform(lower,upper)
**evaluation environments** : 40 linearly spaced values between (lower/2,upper*2)


* [X] CartPole (DQN)
    * tau : (lower,upper) = (0.2,2.2) 
    * length : (lower,upper) = (0.2,2.2)
    * gravity : (lower,upper) = (0.2,2.2)
* [X] LunarLander (DQN)
    * gravity : (lower,upper) = (0.1,2.2)
* [X] MountainCar (DQN, DDQN)
    * gravity : (lower,upper) = (0.1,2.2)
* [X] MountainCar_Continous (SAC)
    * gravity : (lower,upper) = (0.1,2.2)
* [X] Pendulum (SAC)
    * length : (lower,upper) = (0.5,2.2)
* [ ] Ordinary_differential_equation (https://arxiv.org/abs/2310.16686
) (SAC)
* [ ] Mujoco_Ant (SAC)
* [ ] Striker (PPO)
    * gravity : (lower,upper) = (0.1,2.2)
* [ ] [Meta World](https://arxiv.org/abs/1910.10897) (PPO)
-->
## Baselines

* [X] Explicit context : the dynamics are given as input to the model as additional state data, both at training and testing time.
* [X] Hidden context : no dynamics are given as input to the model, neither at training nor testing time.
* [X] [Context is Everything](https://benevans.zip/iida/) : Using previously generated trajectories, a predictor model is trained to predict next states from the current state and the action taken. The predictor model is then used as a **context encoder**, provided as additional state data to the RL agent.

* [X] **Joint Context and Policy learning (JCPL)**: the context encoder is not trained on a prediction task but directly on the trained jointly with the policy network.

<!---
* [ ] [Environment Probing Interaction Policies](https://openreview.net/pdf?id=ryl8-3AcFX) : Similar architecture, but the trajectories are generated by an RL agent whose objective is to actively improve the score of the predictor.
* [ ] **Hypernetworks**: [Dynamics Generalisation in Reinforcement Learning via Adaptive Context-Aware Policies](https://arxiv.org/abs/2310.16686)
-->

# Usage

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run experiments

### Train SAC methods

```bash
python3 scripts/run_sac.py
```


# Roadmap
* [X] Implement DQN for explicit context, no context and JRPL
* [X] Implement HPO pipelines ([how-to-autorl](https://github.com/facebookresearch/how-to-autorl))
    * [X] Run the pipeline locally/on an slurm interactive session
* [X] Implement evaluation pipelines (hydra, submitit)
* [ ] Standardize and document experiments
* [X] Implement DDQN for explicit context, no context and JRPL
* [X] Implement SAC for explicit context, no context and JRPL
* [ ] Implement PPO for explicit context, no context and JRPL



<!---
# Previous experiments

### Train DQN methods

```bash
python3 -m scripts.jrpl.train_dqn
```

### Hyperparameter optimization on DQN methods

```bash
python3 -m scripts.hpo.how_to_autorl.dehb_for_cartpole_dqn_jrpl --multirun 
```

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
-->
