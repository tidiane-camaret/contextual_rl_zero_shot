import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

NUM_OF_PARAMS = 2

class StrikerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, scale=None, oracle=False):
        if scale is None:
            self.scale = np.random.randint(0, 5, NUM_OF_PARAMS)*0.1  # 0~0.4
        else:
            self.scale = scale
        self.oracle = oracle
        utils.EzPickle.__init__(self)
        self._striked = False
        self._min_strike_dist = np.inf
        self.strike_threshold = 0.1
        mujoco_env.MujocoEnv.__init__(self, "striker.xml", 5)

        self.original_mass = np.copy(self.model.body_mass)
        self.original_inertia = np.copy(self.model.body_inertia)
        self.original_damping = np.copy(self.model.dof_damping)

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

        self.change_env(scale=self.scale)

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


    def change_env(self, scale=None):
        mass = np.copy(self.original_mass)
        inertia = np.copy(self.original_inertia)
        damping = np.copy(self.original_damping)

        if scale is None:
            self.scale = np.random.randint(0, 5, NUM_OF_PARAMS)*0.1  # 0~0.4
        else:
            self.scale = scale

        self.env_id = int((self.scale[0] * 5 + self.scale[1]) * 10)

        mass[11] = ((self.scale[0]-0.1)*8+1) * mass[11]  # 0.2~4.2*mass
        inertia[11, :] = ((self.scale[0]-0.1)*8+1) * inertia[11, :]

        damping[7] = (self.scale[1] - 0.2) * 2 + 0.5  # default 0.5: 0.1~1.1
        damping[8] = (self.scale[1] - 0.2) * 2 + 0.5

        self.model.body_mass[:] = mass
        self.model.body_inertia[:] = inertia
        self.model.dof_damping[:] = damping
        return

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
    