import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt  # Add this import

class LunarLanderAgent:
    def __init__(self, alpha=0.01, gamma=0.99, epsilon=0.3, training_episodes=100000, gui_switch_point=5000, visual_episodes=10):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.training_episodes = training_episodes
        self.gui_switch_point = gui_switch_point
        self.visual_episodes = visual_episodes
        self.Q = {}
        self.env = gym.make("LunarLander-v3", render_mode=None)
        self.action_space = self.env.action_space
        self.n_actions = self.action_space.n
        self.rewards_history = []  # Add this to store rewards
        self.epsilon_history = []  # Add this to store epsilon values

    def discretize_state(self, observation):
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
        return tuple(np.digitize(observation[i], bins[i]) - 1 for i in range(8))

    def choose_action(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions)
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.Q[state])

    def update_q_value(self, state, action, reward, next_state, next_action):
        if next_state not in self.Q:
            self.Q[next_state] = np.zeros(self.n_actions)
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]
        )

    def train(self):
        print("Training without GUI...")
        for episode in range(self.training_episodes):
            observation, _ = self.env.reset(seed=42)
            state = self.discretize_state(observation)
            action = self.choose_action(state)
            total_reward = 0
            done = False

            while not done:
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.discretize_state(next_observation)
                next_action = self.choose_action(next_state)
                
                self.update_q_value(state, action, reward, next_state, next_action)

                state, action = next_state, next_action
                total_reward += reward
                done = terminated or truncated

            self.rewards_history.append(total_reward)  # Store reward
            self.epsilon_history.append(self.epsilon)  # Store epsilon
            print(f"Episode {episode + 1}/{self.training_episodes}: Total Reward = {total_reward}")

            if self.epsilon > 0.01:
                self.epsilon *= 0.995

            if episode + 1 == self.gui_switch_point:
                self.env.close()
                self.env = gym.make("LunarLander-v3", render_mode="human")
                print("Switching to GUI for remaining episodes...")

    def visualize(self):
        print("Visualizing final episodes...")
        for episode in range(self.visual_episodes):
            observation, _ = self.env.reset(seed=42)
            state = self.discretize_state(observation)
            action = self.choose_action(state)
            total_reward = 0
            done = False

            while not done:
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.discretize_state(next_observation)
                next_action = self.choose_action(next_state)
                self.update_q_value(state, action, reward, next_state, next_action)
                state, action = next_state, next_action
                total_reward += reward
                done = terminated or truncated

            print(f"Visual Episode {episode + 1}/{self.visual_episodes}: Total Reward = {total_reward}")
        self.env.close()

    def plot_results(self):  # New method to plot graphs
        # Plot Total Rewards
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards_history, label="Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("SARSA Learning Progress - Total Reward per Episode")
        plt.legend()
        plt.grid(True)
        plt.savefig("rewards_plot.png")  # Save to file for your report
        plt.show()

        # Plot Epsilon Decay
        plt.figure(figsize=(10, 5))
        plt.plot(self.epsilon_history, label="Epsilon", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon Value")
        plt.title("Epsilon Decay Over Training")
        plt.legend()
        plt.grid(True)
        plt.savefig("epsilon_plot.png")  # Save to file
        plt.show()

if __name__ == "__main__":
    agent = LunarLanderAgent(training_episodes=1000)  # Reduce episodes for testing due to your deadline
    agent.train()
    agent.visualize()
    agent.plot_results()  # Call the new plotting method