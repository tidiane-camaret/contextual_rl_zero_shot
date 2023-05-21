from gym.envs.registration import register


register(id='CartPoleCustom-v0',
         entry_point='meta_rl.envs:CartPoleEnvTid',
         max_episode_steps=200,
         reward_threshold=195.0,)

register(id='StrikerCustom-v0',
         entry_point='meta_rl.envs:StrikerEnvTid',
         max_episode_steps=200,
         reward_threshold=0.0,)


register(
    id='StrikerOriginal-v0',
    entry_point='meta_rl.envs:StrikerEnv',
    max_episode_steps=200,
    reward_threshold=0,
)

register(
    id='StrikerAvg-v0',
    entry_point='meta_rl.envs:StrikerAvgEnv',
    max_episode_steps=200,
    reward_threshold=0,
)

register(
    id='StrikerLSTM-v0',
    entry_point='meta_rl.envs:StrikerAvgEnv',
    max_episode_steps=200,
    reward_threshold=0,
)

register(
    id='StrikerOracle-v0',
    entry_point='meta_rl.envs:StrikerOracleEnv',
    max_episode_steps=200,
    reward_threshold=0,
)

register(
    id='StrikerHistory-v0',
    entry_point='meta_rl.envs:StrikerHistoryEnv',
    max_episode_steps=200,
    reward_threshold=0,
)

register(
    id='StrikerDirect-v0',
    entry_point='meta_rl.envs:StrikerDirectEnv',
    max_episode_steps=200,
    reward_threshold=0,
)

register(
    id='StrikerTask-v0',
    entry_point='meta_rl.envs:StrikerTaskEnv',
    max_episode_steps=200,
    reward_threshold=0,
    kwargs={"reset": False}
)

register(
    id='StrikerTaskReset-v0',
    entry_point='meta_rl.envs:StrikerTaskEnv',
    max_episode_steps=200,
    reward_threshold=0,
    kwargs={"reset": True}
)

register(
    id='StrikerInteraction-v0',
    entry_point='meta_rl.envs:StrikerInteractionEnv',
    max_episode_steps=200,
    reward_threshold=0,
)
