from gym.envs.registration import register
register(id='StrikerCustom-v0',
         entry_point='meta_rl.envs:StrikerEnv',
         max_episode_steps=100,
         reward_threshold=0.0,)
