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
    def __init__(self):
        super().__init__()

    def step(self, action_n: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, List[bool], List[Dict]]:

        return asyncio.get_event_loop().run_until_complete(self.async_step(action_n))
