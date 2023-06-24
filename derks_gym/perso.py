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

class EnvPerso(DerkEnv):

    def reset(self) -> np.ndarray:
        return asyncio.get_event_loop().run_until_complete(self.async_reset())[0]

    def step(self, action: np.ndarray = None):
        random_action = [self.action_space.sample() for i in range(5)]
        action_n = np.array([action, *random_action])
        resultats = asyncio.get_event_loop().run_until_complete(self.async_step(action_n))
        return resultats[0][0], resultats[1][0], resultats[2], resultats[3]
