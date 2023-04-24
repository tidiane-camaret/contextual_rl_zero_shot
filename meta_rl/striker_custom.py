import gym
import numpy as np

from stable_baselines3.common import vec_env, monitor
from gym.envs.mujoco import StrikerEnv
from gym import spaces

from gym import utils
from gym.envs.mujoco import mujoco_env



class CustomStrikerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, scale=None, oracle=False):
        self.scale = scale
        self.oracle = oracle
        self.max_steps = 1000
        self.step_count = 0
        utils.EzPickle.__init__(self)
        self._striked = False
        self._min_strike_dist = np.inf
        self.strike_threshold = 0.1
        mujoco_env.MujocoEnv.__init__(self, "striker.xml", 5)

        self.original_mass = np.copy(self.model.body_mass)
        self.original_inertia = np.copy(self.model.body_inertia)
        self.original_friction = np.copy(self.model.geom_friction)
        
        self.env_id = int((self.scale[0] * 5 + self.scale[1]) * 10)
        self.model.body_mass[:] = ((self.scale[0] - 0.1) * 5 + 1) * self.original_mass
        self.model.body_inertia[:] = ((self.scale[0] - 0.1) * 5 + 1) * self.original_inertia
        self.model.geom_friction[4, 0] = (self.scale[1] - 0.2) * 0.8 + 0.2

    def step(self, a):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")
        self._min_strike_dist = min(self._min_strike_dist, np.linalg.norm(vec_2))

        if np.linalg.norm(vec_1) < self.strike_threshold:
            self._striked = True
            self._strike_pos = self.get_body_com("tips_arm")

        if self._striked:
            vec_3 = self.get_body_com("object") - self._strike_pos
            reward_near = -np.linalg.norm(vec_3)
        else:
            reward_near = -np.linalg.norm(vec_1)

        reward_dist = -np.linalg.norm(self._min_strike_dist)
        reward_ctrl = -np.square(a).sum()
        reward = 3 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        self.step_count += 1
        done = False #self.step_count >= self.max_steps
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        self.step_count = 0
        self.model.body_mass[:] = ((self.scale[0] - 0.1) * 5 + 1) * self.original_mass
        self.model.body_inertia[:] = ((self.scale[0] - 0.1) * 5 + 1) * self.original_inertia
        self.model.geom_friction[4, 0] = (self.scale[1] - 0.2) * 0.8 + 0.2
        self._min_strike_dist = np.inf
        self._striked = False
        self._strike_pos = None

        qpos = self.init_qpos

        self.ball = np.array([0.5, -0.175])
        while True:
            self.goal = np.concatenate(
                [
                    self.np_random.uniform(low=0.15, high=0.7, size=1),
                    self.np_random.uniform(low=0.1, high=1.0, size=1),
                ]
            )
            if np.linalg.norm(self.ball - self.goal) > 0.17:
                break

        qpos[-9:-7] = [self.ball[1], self.ball[0]]
        qpos[-7:-5] = self.goal
        diff = self.ball - self.goal
        angle = -np.arctan(diff[0] / (diff[1] + 1e-8))
        qpos[-1] = angle / 3.14
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nv
        )
        qvel[7:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        raw_obs = np.concatenate(
            [
                self.sim.data.qpos.flat[:7],
                self.sim.data.qvel.flat[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ]
        )
        if self.oracle:
            raw_obs = np.concatenate([raw_obs, self.scale])
        return raw_obs
    


class OriginalStrikerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self._striked = False
        self._min_strike_dist = np.inf
        self.strike_threshold = 0.1
        mujoco_env.MujocoEnv.__init__(self, "striker.xml", 5)

    def step(self, a):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")
        self._min_strike_dist = min(self._min_strike_dist, np.linalg.norm(vec_2))

        if np.linalg.norm(vec_1) < self.strike_threshold:
            self._striked = True
            self._strike_pos = self.get_body_com("tips_arm")

        if self._striked:
            vec_3 = self.get_body_com("object") - self._strike_pos
            reward_near = -np.linalg.norm(vec_3)
        else:
            reward_near = -np.linalg.norm(vec_1)

        reward_dist = -np.linalg.norm(self._min_strike_dist)
        reward_ctrl = -np.square(a).sum()
        reward = 3 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        self._min_strike_dist = np.inf
        self._striked = False
        self._strike_pos = None

        qpos = self.init_qpos

        self.ball = np.array([0.5, -0.175])
        while True:
            self.goal = np.concatenate(
                [
                    self.np_random.uniform(low=0.15, high=0.7, size=1),
                    self.np_random.uniform(low=0.1, high=1.0, size=1),
                ]
            )
            if np.linalg.norm(self.ball - self.goal) > 0.17:
                break

        qpos[-9:-7] = [self.ball[1], self.ball[0]]
        qpos[-7:-5] = self.goal
        diff = self.ball - self.goal
        angle = -np.arctan(diff[0] / (diff[1] + 1e-8))
        qpos[-1] = angle / 3.14
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nv
        )
        qvel[7:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[:7],
                self.sim.data.qvel.flat[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ]
        )
"""
# striker environment from MuJoCo
class CustomStrikerEnv(StrikerEnv):
    def __init__(self, scale=None, oracle=False):

        self.scale = scale
        self.oracle = oracle
        super().__init__()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        if self.oracle:
            obs = np.concatenate([obs, self.scale])
        
        
        return obs, reward, done, info

class CustomStrikerEnv(StrikerEnv):
    # This environment does not returns rewards for some reason
    def __init__(self, scale=None, oracle=False):
        self.scale = scale
        self.oracle = oracle
        super(CustomStrikerEnv, self).__init__()

        self.original_mass = np.copy(self.model.body_mass)
        self.original_inertia = np.copy(self.model.body_inertia)
        self.original_friction = np.copy(self.model.geom_friction)
        
        self.env_id = int((self.scale[0] * 5 + self.scale[1]) * 10)
        self.model.body_mass[:] = ((self.scale[0] - 0.1) * 5 + 1) * self.original_mass
        self.model.body_inertia[:] = ((self.scale[0] - 0.1) * 5 + 1) * self.original_inertia
        self.model.geom_friction[4, 0] = (self.scale[1] - 0.2) * 0.8 + 0.2

    def reset_model(self):
        self.model.body_mass[:] = ((self.scale[0] - 0.1) * 5 + 1) * self.original_mass
        self.model.body_inertia[:] = ((self.scale[0] - 0.1) * 5 + 1) * self.original_inertia
        self.model.geom_friction[4, 0] = (self.scale[1] - 0.2) * 0.8 + 0.2
        return super(CustomStrikerEnv, self).reset_model()

    def _get_obs(self):
        if self.oracle:
          return np.concatenate(
            [
                self.sim.data.qpos.flat[:7],
                self.sim.data.qvel.flat[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
                self.scale
            ]
        )
        else:
          return np.concatenate(
            [
                self.sim.data.qpos.flat[:7],
                self.sim.data.qvel.flat[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ]
        )
"""

"""
class CustomStrikerEnv(gym.Env):
    def __init__(self, scale=None, oracle=False):

        self.scale = scale
        self.oracle = oracle
        self.env = gym.make("Striker-v2")
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        #self.env = monitor.Monitor(self.env)#, "results/videos/striker/")
        self.model = self.env.model
        self.original_friction = np.copy(self.model.geom_friction)

        self.original_mass = np.copy(self.model.body_mass)
        self.original_inertia = np.copy(self.model.body_inertia)
        self.original_friction = np.copy(self.model.geom_friction)

        mass = np.copy(self.original_mass)
        inertia = np.copy(self.original_inertia)
        friction = np.copy(self.original_friction)
        
        self.env_id = int((self.scale[0] * 5 + self.scale[1]) * 10)
        mass = ((self.scale[0] - 0.1) * 5 + 1) * mass  # 0.5~2.5*mass
        inertia = ((self.scale[0] - 0.1) * 5 + 1) * inertia
        friction[4, 0] = (self.scale[1] - 0.2) * 3 + 2  # 1.4~2.6
        
        self.model.body_mass[:] = mass
        self.model.body_inertia[:] = inertia
        self.model.geom_friction[:] = friction

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        
    def _get_obs(self):
        if self.oracle:

            return np.concatenate(
                [
                    self.env._get_obs(),
                    self.scale
                ]
            )

            return "sauce"
        else:
          return self.env._get_obs()

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
"""
