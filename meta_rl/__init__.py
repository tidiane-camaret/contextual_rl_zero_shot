from gymnasium.envs.registration import register

register(
     id="CartPoleContinuous-v0",
     entry_point="meta_rl.envs:CartPoleContinuousEnv",
        max_episode_steps=200,
        reward_threshold=195.0,
)


