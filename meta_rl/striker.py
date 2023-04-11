import gym

# striker environment from MuJoCo
# add the ability to modify body_mass and damping

class StrikerEnv(gym.Env):
    def __init__(self, body_mass=1.0, damping=0.1):
        self.body_mass = body_mass
        self.damping = damping
        self.env = gym.make("Striker-v2")
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.model = self.env.model
        self.model.body_mass[:] = self.body_mass
        self.model.dof_damping[:] = self.damping
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def get_body_mass(self):
        return self.body_mass

    def get_damping(self):
        return self.damping

    def set_body_mass(self, body_mass):
        self.body_mass = body_mass
        self.model.body_mass[:] = self.body_mass

    def set_damping(self, damping):
        self.damping = damping
        self.model.dof_damping[:] = self.damping

    def get_model(self):
        return self.model

    def get_env(self):
        return self.env

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space

    def __repr__(self):
        return "StrikerEnv(body_mass={}, damping={})".format(self.body_mass, self.damping)

    def __str__(self):
        return "StrikerEnv(body_mass={}, damping={})".format(self.body_mass, self.damping)