import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt

class LunarLanderAgentQLearning:
    def __init__(self, alpha=0.01, gamma=0.99, epsilon=0.3, total_episodes=5000, gui_switch_point=4000, visual_episodes=10):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.total_episodes = total_episodes
        self.gui_switch_point = gui_switch_point
        self.visual_episodes = visual_episodes
        self.Q = {}
        self.env = gym.make("LunarLander-v3", render_mode=None)
        self.action_space = self.env.action_space
        self.n_actions = self.action_space.n
        self.rewards_history = []
        self.epsilon_history = []
        self.successful_landings = 0
        self.success_per_episode = []

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

    # Modified update rule for Q-Learning
    def update_q_value(self, state, action, reward, next_state):
        if next_state not in self.Q:
            self.Q[next_state] = np.zeros(self.n_actions)
        # Q-Learning update: use max Q-value over all actions in next_state
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
        )

    def clear_console(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def is_successful_landing(self, total_reward, final_observation):
        success_threshold = 200
        y_pos = final_observation[1]
        x_vel, y_vel = final_observation[2], final_observation[3]
        left_leg, right_leg = final_observation[6], final_observation[7]
        return (total_reward > success_threshold and 
                abs(y_pos - 0) < 0.1 and
                abs(x_vel) < 0.1 and abs(y_vel) < 0.1 and
                left_leg == 1 and right_leg == 1)

    def training_updates_to_console(self, episode, total_reward):
        self.clear_console()
        frac = ((episode + 1) / self.gui_switch_point) * 100
        # success = 1 if self.is_successful_landing(total_reward, self.env.unwrapped.state) else 0
        print(f"Training {round(frac)}% complete")
        print(f"Current reward = {round(total_reward)}")
        # print(f"Success = {bool(success)}")

    def train(self):
        print("Training without GUI...")
        for episode in range(self.total_episodes):
            observation, _ = self.env.reset(seed=42)
            state = self.discretize_state(observation)
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.discretize_state(next_observation)
                
                # Update Q-value using Q-Learning rule (no next_action needed)
                self.update_q_value(state, action, reward, next_state)

                state = next_state
                total_reward += reward
                done = terminated or truncated

            self.rewards_history.append(total_reward)
            self.epsilon_history.append(self.epsilon)
            success = 1 if self.is_successful_landing(total_reward, next_observation) else 0
            self.success_per_episode.append(success)
            self.successful_landings += success

            self.training_updates_to_console(episode, total_reward)

            if self.epsilon > 0.01:
                self.epsilon *= 0.995

            if episode + 1 == self.gui_switch_point:
                self.env.close()
                self.env = gym.make("LunarLander-v3", render_mode="human")
                print("Switching to GUI for remaining episodes...")

        success_rate = (self.successful_landings / self.total_episodes) * 100
        print(f"Success Rate: {success_rate:.2f}% ({self.successful_landings}/{self.total_episodes} successful landings)")

    def visualize(self):
        print("Visualizing final episodes...")
        for episode in range(self.visual_episodes):
            observation, _ = self.env.reset(seed=42)
            state = self.discretize_state(observation)
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.discretize_state(next_observation)

                self.update_q_value(state, action, reward, next_state)

                state = next_state
                total_reward += reward
                done = terminated or truncated

            print(f"Visual Episode {episode + 1}/{self.visual_episodes}: Total Reward = {total_reward}, "
                  f"Success = {bool(self.is_successful_landing(total_reward, next_observation))}")

        self.env.close()

    def plot_results(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards_history, label="Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Q-Learning Progress - Total Reward per Episode")  # Updated title
        plt.legend()
        plt.grid(True)
        plt.savefig("rewards_plot.png")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(self.epsilon_history, label="Epsilon", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon Value")
        plt.title("Epsilon Decay Over Training")
        plt.legend()
        plt.grid(True)
        plt.savefig("epsilon_plot.png")
        plt.show()

        plt.figure(figsize=(10, 5))
        cumulative_success = np.cumsum(self.success_per_episode) / np.arange(1, len(self.success_per_episode) + 1) * 100
        plt.plot(cumulative_success, label="Cumulative Success Rate", color="green")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate (%)")
        plt.title("Q-Learning Progress - Cumulative Success Rate per Episode")  # Updated title
        plt.legend()
        plt.grid(True)
        plt.savefig("success_rate_plot.png")
        plt.show()

if __name__ == "__main__":
    agent = LunarLanderAgentQLearning()
    agent.train()
    agent.visualize()
    agent.plot_results()