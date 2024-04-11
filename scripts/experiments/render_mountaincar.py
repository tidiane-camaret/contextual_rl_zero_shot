import importlib
import numpy as np

def eval_random_agent(env_name, context_name, context_value, max_steps, num_episodes=50):
    """
    Evaluate a random agent in the environment
    """
# env setup
    print("env_name : ", env_name)
    if env_name == "CARLCartPoleContinuous":
        # custom environment, we need to import it directly
        from meta_rl.envs.carl_cartpole import CARLCartPoleContinuous
        CARLEnv = CARLCartPoleContinuous
    else:
        env_module = importlib.import_module("carl.envs")
        CARLEnv = getattr(env_module, env_name)

    eval_context = CARLEnv.get_default_context()
    eval_context[context_name] = context_value
    env = CARLEnv(
        # You can play with different gravity values here
        contexts={0: eval_context},
    )

    rewards = []
    env.render()
    for _ in range(num_episodes):
        done = False
        episode_reward = 0
        state = env.reset()
        steps = 0
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            if steps > max_steps:
                break
        rewards.append(episode_reward)
    return np.mean(rewards)

if __name__ == "__main__":
    env_name = "CARLMountainCarContinuous"
    context_name = "power"
    context_value = 0.0025
    max_steps = 1000
    num_episodes = 1
    print(eval_random_agent(env_name, context_name, context_value, max_steps, num_episodes))