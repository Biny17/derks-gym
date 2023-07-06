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
import tf_agents
import tensorflow as tf

class EnvSix(DerkEnv):
    def pick(self, odds: np.ndarray):
        # make negative value to 0
        odds[odds<0] = 0
        odds = odds + 0.001
        normalized_odds = np.array(odds) / sum(odds)
        # replace NaN with 0
        # make sure the sum is 1
        normalized_odds /= normalized_odds.sum()

        return np.random.choice(len(odds), p=normalized_odds)

    def prepoc(self, action):
        real_action = []
        real_action.append(action[0])
        real_action.append(action[1])
        real_action.append(max(action[2], 0))

        spells = self.pick(action[3:7])
        focus = self.pick(action[7:])
        real_action.append(spells)
        real_action.append(focus)
        return real_action

    def step(self, actions):
        actions = actions.reshape(6, 15)
        real_actions = []
        for action in actions:
            real_actions.append(self.prepoc(action))
        resultats = asyncio.get_event_loop().run_until_complete(self.async_step(real_actions))
        return resultats


class EnvPerso(DerkEnv):
    def observation_tensor_spec(self):
        return tensor_spec.BoundedTensorSpec(
            shape=(64,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='observation'
        )
    def observation_spec(self):
        return tf_agents.specs.BoundedArraySpec(
            shape=(64,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='observation'
        )

    def action_tensor_spec(self):
        return  tensor_spec.BoundedTensorSpec(
            shape=(15,), dtype=np.float32, minimum=-1, maximum=1, name='action')
    def action_spec(self):
        return tf_agents.specs.BoundedArraySpec(
            shape=(15,), dtype=np.float32, minimum=-1, maximum=1, name='action')

    def reward_tensor_spec(self):
        return tensor_spec.BoundedTensorSpec(
            shape=(),
            dtype=np.float32,
            minimum=0,
            maximum=np.inf,
            name='reward'
        )
    def reward_spec(self):
        return tf_agents.specs.BoundedArraySpec(
            shape=(),
            dtype=np.float32,
            minimum=0,
            maximum=np.inf,
            name='reward'
        )
    def time_step_spec(self):
        self._time_step_spec = ts.time_step_spec(observation_spec=self.observation_spec(), reward_spec=self.reward_spec())
        return self._time_step_spec


    def reset(self):
        return asyncio.get_event_loop().run_until_complete(self.async_reset())[0]

    def step(self, action):
        random_action = [self.action_space.sample() for i in range(5)]
        action_n = np.array([action, *random_action])
        resultats = asyncio.get_event_loop().run_until_complete(self.async_step(action_n))
        return resultats[0][0], resultats[1][0], resultats[2], resultats[3]

class EnvPersoInput(EnvPerso):
    def reset(self):
        return asyncio.get_event_loop().run_until_complete(self.async_reset())[0]

    def pick(self, odds: np.ndarray):
        # make negative value to 0
        odds[odds<0] = 0
        odds = odds + 0.001
        normalized_odds = np.array(odds) / sum(odds)
        # replace NaN with 0
        # make sure the sum is 1
        normalized_odds /= normalized_odds.sum()

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


class EnvTensorOne(EnvPersoInput, tf_agents.environments.py_environment.PyEnvironment):
    def reset(self):
        obs = asyncio.get_event_loop().run_until_complete(self.async_reset())[0]
        return ts.restart(observation = obs)
    def _reset(self):
        obs = asyncio.get_event_loop().run_until_complete(self.async_reset())[0]
        return ts.restart(observation = obs)

    def observation_spec(self):
        return tensor_spec.BoundedTensorSpec(
            shape=(64,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='observation'
        )
    def action_spec(self):
        return  tensor_spec.BoundedTensorSpec(
            shape=(15,), dtype=np.float32, minimum=-1, maximum=1, name='action')
    def reward_spec(self):
        return tensor_spec.BoundedTensorSpec(
            shape=(),
            dtype=np.float32,
            minimum=0,
            maximum=np.inf,
            name='reward'
        )
    def time_step_spec(self):
        self._time_step_spec = ts.time_step_spec(observation_spec=self.observation_spec(), reward_spec=self.reward_spec())
        return self._time_step_spec

    def current_time_step(self):
        return self._current_time_step

    def _step(self, action):
        observation, reward, done, info = EnvPersoInput.step(self, action=action)
        self._current_time_step = ts.transition(observation=observation, reward=reward, discount=1.0)
        if all(done):
            return ts.termination(observation = observation, reward = reward)
        else:
            return ts.transition(observation=observation, reward=reward, discount=1.0)
    def step(self, action):
        observation, reward, done, info = EnvPersoInput.step(self, action=action)
        self._current_time_step = ts.transition(observation=observation, reward=reward, discount=1.0)
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        if all(done):
            print("done")
            self.reset()
            return ts.termination(observation = observation, reward = reward)
        else:
            return ts.transition(observation=observation, reward=reward, discount=1.0)


# class tensorEnv(py_environment.PyEnvironment, DerkEnv):
#     def observation_spec(self):
#         return tensor_spec.BoundedArraySpec(
#             shape=(64,),
#             dtype=np.float32,
#             minimum=-1.0,
#             maximum=1.0,
#             name='observation'
#         )

#     def action_spec(self):
#         return tensor_spec.BoundedArraySpec(
#             shape=(15,), dtype=np.float32, minimum=-1, maximum=1, name='action')

#     def reward_spec(self):
#         return tensor_spec.BoundedArraySpec(
#             shape=(),
#             dtype=tf.float32,
#             minimum=0,
#             maximum=np.inf,
#             name='reward'
#         )
#     def time_step_spec(self):
#         return ts.time_step_spec(observation_spec=self.observation_spec(), reward_spec=self.reward_spec())

#     def _reset(self):

#         return ts.restart(self.observation_spec())
