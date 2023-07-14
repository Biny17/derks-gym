import nest_asyncio
nest_asyncio.apply()
from gym_derk.envs import DerkEnv
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
    def __init__(self, n_arenas, *args, **kwargs):
        self.n_arenas = n_arenas
        self.space_action = gym.spaces.Box(low=-1, high=1, shape=(15,), dtype=np.float32)
        super().__init__(*args, **kwargs, n_arenas=n_arenas)

    def reset(self):
        return super().reset().reshape(self.n_arenas, 6, 64)

    def prepoc(self, actions):
    # MoveX Rotate Chase Cast0 1 2 3 Focus 0 1 2 3 4 5 6 7
        return np.array([*actions[0:2], max(actions[2], 0), np.argmax(actions[3:7]), np.argmax(actions[7:])])

    def step(self, actions):
        #actions = actions.reshape(self.n_arenas, 6, 15)
        processed_actions = np.apply_along_axis(self.prepoc, -1, actions)
        resultats = asyncio.get_event_loop().run_until_complete(self.async_step(processed_actions))
        return resultats[0].reshape(self.n_arenas, 6, 64), resultats[1].reshape(self.n_arenas, 6), resultats[2], resultats[3]


class EnvSpecTensor(DerkEnv):
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
            minimum=-2000,
            maximum=2000,
            name='reward'
        )
    def time_step_spec(self):
        self._time_step_spec = ts.time_step_spec(observation_spec=self.observation_spec(), reward_spec=self.reward_spec())
        return self._time_step_spec

    def input_converter(self, actions):
    # MoveX Rotate Chase Cast0 1 2 3 Focus 0 1 2 3 4 5 6 7
        return np.array([*actions[0:2], max(actions[2], 0), np.argmax(actions[3:7]), np.argmax(actions[7:])])
    


class EnvTensorOne(EnvSpecTensor, tf_agents.environments.py_environment.PyEnvironment):
    def reset(self, policy):
        self.policy = policy
        self.controlled = 0
        print("controlled: ", self.controlled)
        self.reward_perso = 0
        obs = asyncio.get_event_loop().run_until_complete(self.async_reset())
        self.full_obs = obs
        obs = tf.convert_to_tensor(obs[self.controlled], dtype=tf.float32)
        return ts.restart(observation = obs)
    def _reset(self):
        obs = asyncio.get_event_loop().run_until_complete(self.async_reset())[self.controlled]
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        return ts.restart(observation = obs)

    def current_time_step(self):
        return self._current_time_step
    
    def make_time_step_policy(self, i):
        # self.policy.action(self.full_obs[i])
        timestep = ts.transition(observation = self.full_obs[i], reward = tf.constant([0]), discount = 1.0)
        return self.input_converter(self.policy.action(timestep).action.numpy())
    def random_action(self, i):
        action = np.random.uniform(-1, 1, size=(15,))
        return self.input_converter(action)
    def oner(self, action):
        # add policy actions
        action_n = [self.make_time_step_policy(i) if (i != self.controlled) else action for i in range(6)]
        
        returned = asyncio.get_event_loop().run_until_complete(self.async_step(action_n))
        self.full_obs = tf.convert_to_tensor(returned[0], dtype=np.float32)
        return returned[0][self.controlled], returned[1][self.controlled], returned[2], returned[3]
    
    def step(self, action):
        action = action.numpy()
        action = self.input_converter(action)
        observation, reward, done, info = self.oner(action)
        self.reward_perso += reward
        observation = tf.convert_to_tensor(observation)
        reward = tf.convert_to_tensor(reward)
        if reward.numpy() != 0:
            print(f"{reward :.2f}")
        self._current_time_step = ts.transition(observation=observation, reward=reward, discount=1.0)
        if all(done):
            print(f"total_reward: {self.reward_perso}")
            return ts.termination(observation = observation, reward = reward)
        else:
            return ts.transition(observation = observation, reward = reward, discount=1.0)
    def _step(self, action):
        return self.step(action)


class EnvTensorSix(EnvSpecTensor, tf_agents.environments.py_environment.PyEnvironment):
    def reset(self):
        obs = asyncio.get_event_loop().run_until_complete(self.async_reset())
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        return ts.restart(observation = obs)
    def _reset(self):
        obs = asyncio.get_event_loop().run_until_complete(self.async_reset())
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
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
            minimum=-2000,
            maximum=2000,
            name='reward'
        )
    def time_step_spec(self):
        self._time_step_spec = ts.time_step_spec(observation_spec=self.observation_spec(), reward_spec=self.reward_spec())
        return self._time_step_spec

    def current_time_step(self):
        return self._current_time_step
    
    def make_time_step_policy(self, i):
        # self.policy.action(self.full_obs[i])
        timestep = ts.transition(observation = self.full_obs[i], reward = tf.constant([0]), discount = 1.0)
        return self.input_converter(self.policy.action(timestep).action.numpy())
    
    
    def step(self, action):
        action = action.numpy()
        action = self.input_converter(action)
        observation, reward, done, info = self.oner(action)
        self.reward_perso += reward
        observation = tf.convert_to_tensor(observation)
        reward = tf.convert_to_tensor(reward)
        if reward.numpy() != 0:
            print(f"{reward :.2f}")
        self._current_time_step = ts.transition(observation=observation, reward=reward, discount=1.0)
        if all(done):
            print(f"total_reward: {self.reward_perso}")
            return ts.termination(observation = observation, reward = reward)
        else:
            return ts.transition(observation = observation, reward = reward, discount=1.0)
    def _step(self, action):
        return self.step(action)
    def reset(self):
        obs = asyncio.get_event_loop().run_until_complete(self.async_reset())[0]
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        return ts.restart(observation = obs)
    def _reset(self):
        obs = asyncio.get_event_loop().run_until_complete(self.async_reset())[0]
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
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
    
    def action_process(self, actions):
        print("action_process:")
        print(f"{actions = }")
        return np.apply_along_axis(self.input_converter, -1, actions)
    
    
    def step(self, actions):
        actions = actions.numpy()
        actions = self.action_process(actions)
        observation, reward, done, info = asyncio.get_event_loop().run_until_complete(self.async_step(actions))
        observation = tf.convert_to_tensor(observation)
        reward = tf.convert_to_tensor(reward)
        self._current_time_step = ts.transition(observation=observation, reward=reward, discount=1.0)
        if all(done):
            self.reset()
            return ts.termination(observation = observation, reward = reward)
        else:
            return ts.transition(observation = observation, reward = reward, discount=1.0)
    def _step(self, action):
        return self.step(action)
    
    
    

    # def _step(self, action):
    #     observation, reward, done, info = EnvPersoInput.step(self, action=action)
    #     self._current_time_step = ts.transition(observation=observation.flatten(), reward=reward, discount = 1.0)
    #     print(type(observation))
    #     if all(done):
    #         self.reset()
    #         return ts.termination(observation = observation.flatten(), reward = reward)
    #     else:
    #         return ts.transition(observation=observation.flatten(), reward=reward, discount = 1.0)

    # def step(self, action):
    #     observation, reward, done, info = EnvPersoInput.step(self, action=action)
    #     self._current_time_step = ts.transition(observation=observation.flatten(), reward=reward, discount = 1.0)
    #     print(type(observation))
    #     if all(done):
    #         self.reset()
    #         return ts.termination(observation = observation.flatten(), reward = reward)
    #     else:
    #         return ts.transition(observation = observation.flatten(), reward=reward, discount = 1.0)


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
