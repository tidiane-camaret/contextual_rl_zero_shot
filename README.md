# meta_rl

A series of experiments on meta reinforcement learning, aiming to develop a meta learning framework for reinforcement learning tasks.

An RL agent is exposed to a series of training tasks, and should be able to perform well on a new task with little to no training, **as well as assessing its level of uncertainty on the new task.**

The current setup utilizes the [OpenAI Gym](https://gym.openai.com/) environment, and particularly the Striker task, which consists of a robotic arm that must hit a ball into a goal.

## Installation

### Install dependencies

```bash
pip install -r requirements.txt
```

# Evaluation of baseline models

We evaluate a model using sets of training and testing environments, with non overlapping sets of dynamics.

Environments can be generated with different dynamics, namely mass, inertia and damping. (see meta_rl/envs/striker_avg.py) The range of those values are taken from [Environment Probing Interaction Policies](https://openreview.net/pdf?id=ryl8-3AcFX).

The project currently contains two baseline models :

- "explicit context" : the dynamics are given as input to the model as additional state data, both at training and testing time.

'''bash
scripts/metarl_striker.py --context explicit
'''

- "no context" : no dynamics are given as input to the model, neither at training nor testing time.

'''bash
scripts/metarl_striker.py --context none
'''

# Additional models

- [Context is Everything](https://benevans.zip/iida/)
A previoulsy trained **generator policy** generates trajectories from training environments. A predictor model is then trained to predict next states from the current state and the action taken. The predictor model is then used as a **context encoder** for the RL agent, which is trained on the training environments. The RL agent is then tested on the testing environments.

'''bash
scripts/iida/genereate_trajectories.py
scripts/iida/train_predictor.py
scripts/metarl_striker.py --context latent
'''

# Roadmap
- [x] Implement baseline models
- [x] Implement Context is Everything model
- [ ] Use the Meta World benchmark