import nest_asyncio
nest_asyncio.apply()

from gym_derk.envs import DerkEnv
import gym_derk

env = DerkEnv(turbo_mode=True)
observation_n = env.reset()
