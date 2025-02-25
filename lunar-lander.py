import gymnasium as gym
import numpy as np

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()
    
    if (observation[5] > 0):
        action = 3
    elif (observation[5] < 0):
        action = 1
    else:
        action = 0

    if (observation[2] > 0):
        action = 3
    elif (observation[3] < 0):
        action = 1
    else:
        action = 0

    

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()