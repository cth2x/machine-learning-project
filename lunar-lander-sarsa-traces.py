import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
import pygame  # For cleanup


class LunarLanderAgentSARSA:

    def __init__(
        self,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.5,
        epsilon_min=0.01,
        lambda_trace=0.5,
        total_episodes=50000,
        visual_episodes=10,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.lambda_trace = lambda_trace
        self.total_episodes = total_episodes
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
            np.linspace(-1.5, 1.5, 8),  # x position
            np.linspace(0, 1.5, 8),  # y position
            np.linspace(-1, 1, 8),  # x velocity
            np.linspace(-1, 1, 8),  # y velocity
            np.linspace(-0.5, 0.5, 8),  # angle
            np.linspace(-1, 1, 8),  # angular velocity
            np.linspace(0, 1, 2),  # left leg contac
            np.linspace(0, 1, 2),  # right leg contact
        ]
        return tuple(np.digitize(observation[i], bins[i]) - 1 for i in range(8))

    def choose_action(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions)
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.Q[state])

    def clear_console(self):
        os.system("cls" if os.name == "nt" else "clear")

    def is_successful_landing(self, total_reward, final_observation):
        success_threshold = 180
        y_pos = final_observation[1]
        x_vel, y_vel = final_observation[2], final_observation[3]
        left_leg, right_leg = final_observation[6], final_observation[7]
        return (
            total_reward > success_threshold
            and abs(y_pos - 0) < 0.15
            and abs(x_vel) < 0.15
            and abs(y_vel) < 0.15
            and left_leg == 1
            and right_leg == 1
        )

    def training_updates_to_console(self, episode, total_reward):
        self.clear_console()
        frac = ((episode + 1) / self.total_episodes) * 100
        print(f"Training {round(frac)}% complete")
        print(f"Current reward = {round(total_reward)}")

    def train(self):
        print("Training...")
        for episode in range(self.total_episodes):
            observation, _ = self.env.reset()
            state = self.discretize_state(observation)
            action = self.choose_action(state)
            total_reward = 0
            done = False
            E = {}

            while not done:
                next_observation, reward, terminated, truncated, _ = self.env.step(
                    action
                )
                next_state = self.discretize_state(next_observation)
                next_action = self.choose_action(next_state)

                if state not in self.Q:
                    self.Q[state] = np.zeros(self.n_actions)
                if next_state not in self.Q:
                    self.Q[next_state] = np.zeros(self.n_actions)

                delta = (
                    reward
                    + self.gamma * self.Q[next_state][next_action]
                    - self.Q[state][action]
                )

                E[(state, action)] = E.get((state, action), 0) + 1

                for s, a in list(E.keys()):
                    self.Q[s][a] += self.alpha * delta * E[(s, a)]
                    E[(s, a)] *= self.gamma * self.lambda_trace
                    if E[(s, a)] < 1e-5:
                        del E[(s, a)]

                state, action = next_state, next_action
                total_reward += reward
                done = terminated or truncated

            self.rewards_history.append(total_reward)
            self.epsilon_history.append(self.epsilon)
            success = (
                1 if self.is_successful_landing(total_reward, next_observation) else 0
            )
            self.success_per_episode.append(success)
            self.successful_landings += success

            self.training_updates_to_console(episode, total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.9995)

        success_rate = (self.successful_landings / self.total_episodes) * 100
        print(
            f"Success Rate: {success_rate:.2f}% ({self.successful_landings}/{self.total_episodes} successful landings)"
        )

    def visualize(self):
        pass

    def plot_results(self):
        os.makedirs("sarsa/traces", exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards_history, label="Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("SARSA(λ) Learning Progress - Total Reward per Episode")
        plt.legend()
        plt.grid(True)
        plt.savefig("sarsa/traces/rewards_plot.png")
        plt.show(block=False)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.epsilon_history, label="Epsilon", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon Value")
        plt.title("Epsilon Decay Over Training")
        plt.legend()
        plt.grid(True)
        plt.savefig("sarsa/traces/epsilon_plot.png")
        plt.show(block=False)
        plt.close()

        plt.figure(figsize=(10, 5))
        cumulative_success = (
            np.cumsum(self.success_per_episode)
            / np.arange(1, len(self.success_per_episode) + 1)
            * 100
        )
        plt.plot(cumulative_success, label="Cumulative Success Rate", color="green")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate (%)")
        plt.title("SARSA(λ) Learning Progress - Cumulative Success Rate per Episode")
        plt.legend()
        plt.grid(True)
        plt.savefig("sarsa/traces/success_rate_plot.png")
        plt.show(block=False)
        plt.close()


if __name__ == "__main__":
    agent = LunarLanderAgentSARSA()
    agent.train()
    agent.plot_results()
