import gymnasium as gym
# create the environment
env = gym.make('CartPole-v1', render_mode="rgb_array")

# initialize the environment
observation = env.reset()
env.render()  # render the environment
