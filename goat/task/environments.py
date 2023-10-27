import importlib
from typing import TYPE_CHECKING, Optional, Type

import gym
import habitat
import numpy as np
from habitat import Dataset
from habitat.core.environments import RLTaskEnv
from habitat.core.registry import registry
from habitat.utils.gym_adapter import HabGymWrapper


class GoatRLEnv(RLTaskEnv):
    def __init__(self, config: "DictConfig", dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)
        self._reward_measure_name = self.config.task.reward_measure
        self._success_measure_name = self.config.task.success_measure
        assert (
            self._reward_measure_name is not None
        ), "The key task.reward_measure cannot be None"
        assert (
            self._success_measure_name is not None
        ), "The key task.success_measure cannot be None"

    def reset(self):
        observations = super().reset()
        return observations

    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        # We don't know what the reward measure is bounded by
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        current_measure = self._env.get_metrics()[self._reward_measure_name]
        reward = self.config.task.slack_reward

        reward += current_measure

        if self._episode_success()["composite_success"] == 1.0:
            reward += self.config.task.success_reward

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        if (
            self.config.task.end_on_success
            and self._episode_success()["composite_success"] == 1.0
        ):
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@habitat.registry.register_env(name="GymGoatEnv")
class GymGoatEnv(gym.Wrapper):
    """
    A registered environment that wraps a RLTaskEnv with the HabGymWrapper
    to use the default gym API.
    """

    def __init__(self, config: "DictConfig", dataset: Optional[Dataset] = None):
        base_env = GoatRLEnv(config=config, dataset=dataset)
        env = HabGymWrapper(base_env)
        super().__init__(env)
