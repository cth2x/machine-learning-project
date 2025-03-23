import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
import pygame  # For cleanup
import os


class LunarLanderAgentSARSA:
    """
    Implementation of SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm
    for the LunarLander environment. SARSA is an on-policy TD control method that learns
    directly from episodes of experience.
    """

    def __init__(
        self,
        alpha=0.1,  # Learning rate
        gamma=0.99,  # Discount factor
        epsilon=0.5,  # Initial exploration rate
        epsilon_min=0.01,  # Minimum exploration rate
        total_episodes=50000,  # Total training episodes
        visual_episodes=10,  # Number of episodes to visualize
    ):
        """
        Initialize the SARSA agent with hyperparameters.

        Args:
            alpha: Learning rate - controls how much to update the Q-values
            gamma: Discount factor - determines importance of future rewards
            epsilon: Initial exploration rate for epsilon-greedy policy
            epsilon_min: Minimum exploration rate
            total_episodes: Total number of training episodes
            visual_episodes: Number of episodes to visualize after training
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.total_episodes = total_episodes
        self.visual_episodes = visual_episodes
        self.Q = {}  # Q-table to store state-action values
        self.env = gym.make("LunarLander-v3", render_mode=None)
        self.action_space = self.env.action_space
        self.n_actions = self.action_space.n
        self.rewards_history = []  # Store rewards for each episode
        self.epsilon_history = []  # Store epsilon values for plotting
        self.successful_landings = 0  # Count successful landings
        self.success_per_episode = []  # Track success (1) or failure (0) per episode

    def discretize_state(self, observation):
        """
        Convert continuous observation space to discrete state space using binning.

        Args:
            observation: Raw observation from the environment (8-dimensional vector)

        Returns:
            tuple: Discretized state as a tuple of indices
        """
        bins = [
            np.linspace(-1.5, 1.5, 8),  # x position
            np.linspace(0, 1.5, 8),  # y position
            np.linspace(-1, 1, 8),  # x velocity
            np.linspace(-1, 1, 8),  # y velocity
            np.linspace(-0.5, 0.5, 8),  # angle
            np.linspace(-1, 1, 8),  # angular velocity
            np.linspace(0, 1, 2),  # left leg contact
            np.linspace(0, 1, 2),  # right leg contact
        ]
        return tuple(np.digitize(observation[i], bins[i]) - 1 for i in range(8))

    def choose_action(self, state):
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current discretized state

        Returns:
            int: Selected action
        """
        if state not in self.Q:
            self.Q[state] = np.zeros(
                self.n_actions
            )  # Initialize Q-values for new states
        if np.random.random() < self.epsilon:
            return self.action_space.sample()  # Explore - random action
        return np.argmax(self.Q[state])  # Exploit - best known action

    def update_q_value(self, state, action, reward, next_state, next_action):
        """
        Update Q-value using the SARSA update rule:
        Q(s,a) = Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]

        Args:
            state: Current state
            action: Current action
            reward: Received reward
            next_state: Next state
            next_action: Next action
        """
        if next_state not in self.Q:
            self.Q[next_state] = np.zeros(
                self.n_actions
            )  # Initialize Q-values for new states
        self.Q[state][action] += self.alpha * (
            reward
            + self.gamma * self.Q[next_state][next_action]
            - self.Q[state][action]
        )

    def clear_console(self):
        """Clear the console for cleaner display of training progress."""
        os.system("cls" if os.name == "nt" else "clear")

    def is_successful_landing(self, total_reward, final_observation):
        """
        Determine if the landing was successful based on final state and reward.

        Args:
            total_reward: Total reward accumulated in the episode
            final_observation: Final state observation

        Returns:
            bool: True if landing was successful, False otherwise
        """
        success_threshold = 180
        y_pos = final_observation[1]
        x_vel, y_vel = final_observation[2], final_observation[3]
        left_leg, right_leg = final_observation[6], final_observation[7]
        return (
            total_reward > success_threshold
            and abs(y_pos - 0) < 0.15  # Close to ground
            and abs(x_vel) < 0.15  # Low horizontal velocity
            and abs(y_vel) < 0.15  # Low vertical velocity
            and left_leg == 1  # Left leg touching ground
            and right_leg == 1  # Right leg touching ground
        )

    def training_updates_to_console(self, episode, total_reward):
        """
        Display training progress in the console.

        Args:
            episode: Current episode number
            total_reward: Total reward for the current episode
        """
        self.clear_console()
        frac = ((episode + 1) / self.total_episodes) * 100
        print(f"Training {round(frac)}% complete")
        print(f"Current reward = {round(total_reward)}")

    def train(self):
        """
        Train the agent using the SARSA algorithm.
        For each episode:
          1. Reset the environment
          2. Choose initial action using epsilon-greedy
          3. For each step:
             a. Take action, observe reward and next state
             b. Choose next action using epsilon-greedy
             c. Update Q-value using SARSA update rule
             d. Move to next state-action pair
          4. Decay epsilon
          5. Track metrics
        """
        print("Training...")
        for episode in range(self.total_episodes):
            observation, _ = self.env.reset()
            state = self.discretize_state(observation)
            action = self.choose_action(state)
            total_reward = 0
            done = False

            while not done:
                next_observation, reward, terminated, truncated, _ = self.env.step(
                    action
                )
                next_state = self.discretize_state(next_observation)
                next_action = self.choose_action(next_state)
                self.update_q_value(state, action, reward, next_state, next_action)
                state, action = (
                    next_state,
                    next_action,
                )  # Key SARSA difference from Q-learning
                total_reward += reward
                done = terminated or truncated

            # Track performance metrics
            self.rewards_history.append(total_reward)
            self.epsilon_history.append(self.epsilon)
            success = (
                1 if self.is_successful_landing(total_reward, next_observation) else 0
            )
            self.success_per_episode.append(success)
            self.successful_landings += success

            self.training_updates_to_console(episode, total_reward)

            # Decay epsilon for exploration-exploitation trade-off
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.9995)

        # Display final success rate
        success_rate = (self.successful_landings / self.total_episodes) * 100
        print(
            f"Success Rate: {success_rate:.2f}% ({self.successful_landings}/{self.total_episodes} successful landings)"
        )

    def visualize(self):
        """Placeholder for visualization method."""
        pass

    def plot_results(self):
        """
        Plot and save training metrics:
        1. Rewards per episode
        2. Epsilon decay
        3. Success rate
        """
        os.makedirs("sarsa/normal", exist_ok=True)

        # Plot rewards
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards_history, label="Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("SARSA Learning Progress - Total Reward per Episode")
        plt.legend()
        plt.grid(True)
        plt.savefig("sarsa/normal/rewards_plot.png")
        plt.show(block=False)
        plt.close()

        # Plot epsilon decay
        plt.figure(figsize=(10, 5))
        plt.plot(self.epsilon_history, label="Epsilon", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon Value")
        plt.title("Epsilon Decay Over Training")
        plt.legend()
        plt.grid(True)
        plt.savefig("sarsa/normal/epsilon_plot.png")
        plt.show(block=False)
        plt.close()

        # Plot success rate
        plt.figure(figsize=(10, 5))
        cumulative_success = (
            np.cumsum(self.success_per_episode)
            / np.arange(1, len(self.success_per_episode) + 1)
            * 100
        )
        plt.plot(cumulative_success, label="Cumulative Success Rate", color="green")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate (%)")
        plt.title("SARSA Learning Progress - Cumulative Success Rate per Episode")
        plt.legend()
        plt.grid(True)
        plt.savefig("sarsa/normal/success_rate_plot.png")
        plt.show(block=False)
        plt.close()


if __name__ == "__main__":
    agent = LunarLanderAgentSARSA()
    agent.train()
    agent.plot_results()
