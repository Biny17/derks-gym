import nest_asyncio
nest_asyncio.apply()
from gym_derk.envs import DerkEnv
import gym_derk
import gym
import asyncio
import os
from typing import List, Dict, Tuple
import numpy as np
from gym_derk.derk_app_instance import DerkAppInstance
from gym_derk.derk_server import DerkAgentServer, DerkSession
import logging
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import tensor_spec
from tf_agents.specs import array_spec
import numpy as np
import tensorflow as tf

# Define the observation spec


# Define the reward spec
reward_spec = array_spec.ArraySpec(
    shape=(),
    dtype=np.float32,
    name='reward'
)

# Create the time step spec
# time_step_spec = ts.time_step_spec(observation=observation_spec, reward=reward_spec)

class EnvSix(DerkEnv):
    @property
    def observation_spec(self):
        return tensor_spec.BoundedTensorSpec(
            shape=(64,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='observation'
        )
    @property
    def reward_spec(self):
        return array_spec.ArraySpec(
            shape=(),
            dtype=np.float32,
            name='reward'
        )


class EnvPerso(DerkEnv):
    def observation_spec(self):
        return tensor_spec.BoundedTensorSpec(
            shape=(64,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='observation'
        )

    def action_spec(self):
        return tensor_spec.BoundedTensorSpec(
            shape=(15,), dtype=np.float32, minimum=-1, maximum=1, name='action')

    def reward_spec(self):
        return tensor_spec.BoundedTensorSpec(
            shape=(),
            dtype=tf.float32,
            minimum=0,
            maximum=np.inf,
            name='reward'
        )
    def time_step_spec(self):
        return ts.time_step_spec(observation_spec=self.observation_spec(), reward_spec=self.reward_spec())


    def reset(self) -> np.ndarray:
        return asyncio.get_event_loop().run_until_complete(self.async_reset())[0]

    def step(self, action):
        random_action = [self.action_space.sample() for i in range(5)]
        action_n = np.array([action, *random_action])
        resultats = asyncio.get_event_loop().run_until_complete(self.async_step(action_n))
        return resultats[0][0], resultats[1][0], resultats[2], resultats[3]







class EnvPersoInput(EnvPerso):
    def reset(self) -> np.ndarray:
        return asyncio.get_event_loop().run_until_complete(self.async_reset())[0]

    def pick(self, odds: np.ndarray):
        # make negative value to 0
        odds[odds<0] = 0
        normalized_odds = np.array(odds) / sum(odds)
        # replace NaN with 0
        normalized_odds = np.nan_to_num(normalized_odds)
        return np.random.choice(len(odds), p=normalized_odds)

    def step(self, action):
        real_action = []
        real_action.append(action[0])
        real_action.append(action[1])
        real_action.append(max(action[2], 0))

        spells = self.pick(action[3:7])
        focus = self.pick(action[7:])
        real_action.append(spells)
        real_action.append(focus)
        random_action = [self.action_space.sample() for i in range(5)]
        action_n = np.array([real_action, *random_action])
        resultats = asyncio.get_event_loop().run_until_complete(self.async_step(action_n))
        return resultats[0][0], resultats[1][0], resultats[2], resultats[3]
