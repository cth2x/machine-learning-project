import gymnasium as gym
import numpy as np

# Discretization function
def discretize_state(observation):
    bins = [
        np.linspace(-1.5, 1.5, 4),  # x position
        np.linspace(0, 1.5, 4),    # y position
        np.linspace(-1, 1, 4),     # x velocity
        np.linspace(-1, 1, 4),     # y velocity
        np.linspace(-0.5, 0.5, 4), # angle
        np.linspace(-1, 1, 4),     # angular velocity
        np.linspace(0, 1, 2),      # left leg contact (0 or 1)
        np.linspace(0, 1, 2)       # right leg contact (0 or 1)
    ]
    state = tuple(np.digitize(observation[i], bins[i]) - 1 for i in range(8))  # -1 to start indices at 0
    return state

# ε-greedy policy
def choose_action(state, Q, epsilon, action_space):
    if state not in Q:  # Initialize Q-values for new states
        Q[state] = np.zeros(action_space.n)
    if np.random.random() < epsilon:  # Explore
        return action_space.sample()
    else:  # Exploit
        return np.argmax(Q[state])  # Pick action with highest Q-value

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.3  # Exploration rate (start high, can decay later)
training_episodes = 1000  # Episodes without GUI
visual_episodes = 10      # Episodes with GUI after training

# Q-table as a dictionary
Q = {}  # Shared across both phases

# Phase 1: Training without GUI
env = gym.make("LunarLander-v3", render_mode=None)  # No rendering
action_space = env.action_space
n_actions = action_space.n

print("Training without GUI...")
for episode in range(training_episodes):
    observation, info = env.reset(seed=42)
    state = discretize_state(observation)
    action = choose_action(state, Q, epsilon, action_space)
    
    total_reward = 0
    done = False
    
    while not done:
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_state = discretize_state(next_observation)
        next_action = choose_action(next_state, Q, epsilon, action_space)
        
        if next_state not in Q:
            Q[next_state] = np.zeros(n_actions)
        
        # SARSA update
        Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
        
        state = next_state
        action = next_action
        total_reward += reward
        
        done = terminated or truncated
        if done:
            print(f"Episode {episode + 1}/{training_episodes}: Total Reward = {total_reward}")
            observation, info = env.reset()

    if epsilon > 0.01:
        epsilon *= 0.995

env.close()  # Close the non-rendering environment

# Phase 2: Visualization with GUI
env = gym.make("LunarLander-v3", render_mode="human")  # Enable GUI
print("Switching to GUI for visualization...")

for episode in range(visual_episodes):
    observation, info = env.reset(seed=42)
    state = discretize_state(observation)
    action = choose_action(state, Q, epsilon, action_space)
    
    total_reward = 0
    done = False
    
    while not done:
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_state = discretize_state(next_observation)
        next_action = choose_action(next_state, Q, epsilon, action_space)
        
        if next_state not in Q:
            Q[next_state] = np.zeros(n_actions)
        
        # SARSA update (optional, can remove if only visualizing)
        Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
        
        state = next_state
        action = next_action
        total_reward += reward
        
        done = terminated or truncated
        if done:
            print(f"Visual Episode {episode + 1}/{visual_episodes}: Total Reward = {total_reward}")
            observation, info = env.reset()

env.close()  # Close the rendering environment