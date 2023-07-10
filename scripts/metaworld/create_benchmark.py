import metaworld
import random

print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

ml1 = metaworld.ML1('pick-place-v2') # Construct the benchmark, sampling tasks

env = ml1.train_classes['pick-place-v2']()  # Create an environment with task `pick_place`

print('nb of tasks : ' , len(ml1.train_tasks))  # How many tasks are there?
'''
for task in ml1.train_tasks:
    env.set_task(task)  # Set task
    print(env.obj_init_pos) # initial position is modified each time
'''


for task in ml1.test_tasks[0:1]:
    print(task.env_name)
    #print(task)
    env.set_task(task)  # Set task
    print(env.model.body_mass[0], env.model.body_inertia[0])
    env.model.body_mass[0] = 0.1
    env.model.body_inertia[0] = [0.0001, 0.0001, 0.0001]
    obs = env.reset()
    print(env.model.body_mass[0], env.model.body_inertia[0])
    """
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
    """
