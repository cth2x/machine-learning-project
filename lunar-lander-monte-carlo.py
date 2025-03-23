import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt

class LunarLanderAgentMC:
    def __init__(self, gamma=0.99, epsilon=0.4, total_episodes=10000, gui_switch_point=9900, visual_episodes=10):
        self.gamma = gamma
        self.epsilon = epsilon
        self.total_episodes = total_episodes
        self.gui_switch_point = gui_switch_point
        self.visual_episodes = visual_episodes
        self.Q = {}
        self.returns = {}
        self.env = gym.make("LunarLander-v3", render_mode=None)
        self.action_space = self.env.action_space
        self.n_actions = self.action_space.n
        self.rewards_history = []
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
    
    def update_q_values(self, episode_data):
        G = 0
        visited_state_action_pairs = set()
        for state, action, reward in reversed(episode_data):
            G = self.gamma * G + reward
            if (state, action) not in visited_state_action_pairs:
                visited_state_action_pairs.add((state, action))
                if (state, action) not in self.returns:
                    self.returns[(state, action)] = []
                self.returns[(state, action)].append(G)
                # Apply moving average for Q-value update
                alpha = 0.1  # Learning rate for moving average
                self.Q[state][action] = (1 - alpha) * self.Q[state][action] + alpha * np.mean(self.returns[(state, action)])
    
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
        if episode % 5000 == 0:
            avg_reward = np.mean(self.rewards_history[-5000:])
            avg_success_rate = np.mean(self.success_per_episode[-5000:]) * 100
            print(f"Episode {episode}/{self.total_episodes} - Avg Reward: {avg_reward:.2f} - Success Rate: {avg_success_rate:.2f}%")
    
    def train(self):
        print("Training without GUI...")
        for episode in range(self.total_episodes):
            observation, _ = self.env.reset(seed=42)
            state = self.discretize_state(observation)
            episode_data = []
            total_reward = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.discretize_state(next_observation)
                episode_data.append((state, action, reward))
                state = next_state
                total_reward += reward
                done = terminated or truncated
            
            self.rewards_history.append(total_reward)
            self.update_q_values(episode_data)
            success = 1 if self.is_successful_landing(total_reward, next_observation) else 0
            self.success_per_episode.append(success)
            self.successful_landings += success

            self.training_updates_to_console(episode, total_reward)

            # Epsilon Decay
            if self.epsilon > 0.05:
                self.epsilon *= 0.9999

            if episode + 1 == self.gui_switch_point:
                self.env.close()
                self.env = gym.make("LunarLander-v3", render_mode="human")
                print("Switching to GUI for remaining episodes...")
    
    def plot_results(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards_history, label="Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Monte Carlo Learning Progress - Total Reward per Episode")
        plt.legend()
        plt.grid(True)
        plt.savefig("mc_rewards_plot.png")
        plt.show(block=False)
        plt.close()

        plt.figure(figsize=(10, 5))
        cumulative_success = np.cumsum(self.success_per_episode) / np.arange(1, len(self.success_per_episode) + 1) * 100
        plt.plot(cumulative_success, label="Cumulative Success Rate", color="green")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate (%)")
        plt.title("Monte Carlo Learning Progress - Cumulative Success Rate per Episode") 
        plt.legend()
        plt.grid(True)
        plt.savefig("mc_success_rate_plot.png")
        plt.show(block=False)
        plt.close()

if __name__ == "__main__":
    agent = LunarLanderAgentMC()
    agent.train()
    agent.plot_results()
