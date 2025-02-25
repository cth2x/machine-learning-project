import gymnasium as gym
import numpy as np

# Discretization function
def discretize_state(observation):
    # Define bin ranges based on Lunar Lander's typical observation bounds
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

# Initialize environment
env = gym.make("LunarLander-v3", render_mode="human")
action_space = env.action_space
n_actions = action_space.n  # 4 actions: 0=nothing, 1=left, 2=main, 3=right

# Q-table as a dictionary
Q = {}  # Keys are state tuples, values are arrays of Q-values for each action

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.3  # Exploration rate (start high, can decay later)
episodes = 1000  # Number of episodes to train

# Training loop
for episode in range(episodes):
    observation, info = env.reset(seed=42)
    state = discretize_state(observation)
    action = choose_action(state, Q, epsilon, action_space)
    
    total_reward = 0
    done = False
    
    while not done:
        # Step through environment
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_state = discretize_state(next_observation)
        next_action = choose_action(next_state, Q, epsilon, action_space)
        
        # Ensure next_state is in Q
        if next_state not in Q:
            Q[next_state] = np.zeros(n_actions)
        
        # SARSA update: Q(s,a) += alpha * (reward + gamma * Q(s',a') - Q(s,a))
        Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
        
        # Move to next state and action
        state = next_state
        action = next_action
        total_reward += reward
        
        # Check if episode ends
        done = terminated or truncated
        if done:
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
            observation, info = env.reset()

    # Optional: Decay epsilon to reduce exploration over time
    if epsilon > 0.01:
        epsilon *= 0.995

# Close environment
env.close()