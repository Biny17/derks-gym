import numpy as np
import nest_asyncio
nest_asyncio.apply()
from gym_derk.envs import DerkEnv
import gym_derk



# env = DerkEnv(turbo_mode=True)
# observation_n = env.reset()

# action_n = [env.action_space.sample() for i in range(env.n_agents)]
# observation_n, reward_n, done_n, info_n = env.step(action_n)
# model = action_n

# if all(done_n):
#     env.close()

# total_rewards = np.sum(reward_n)

# print(total_rewards)
