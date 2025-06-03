import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bars

class NonstationaryBandit:
    def __init__(self, k=10, q_init=0.0, random_walk_std=0.01):
        self.k = k
        self.q_true = np.ones(k) * q_init
        self.random_walk_std = random_walk_std

    def get_reward(self, action):
        # Sample reward from N(q_true[action], 1)
        reward = np.random.normal(loc=self.q_true[action], scale=1.0)
        # Apply random walk to q_true for all actions
        self.q_true += np.random.normal(loc=0.0, scale=self.random_walk_std, size=self.k)
        return reward

    def optimal_action(self):
        return np.argmax(self.q_true)

class Agent:
    def __init__(self, k=10, epsilon=0.1, alpha=None):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha  # None for sample-average, otherwise fixed-step size
        self.Q = np.zeros(k)
        self.N = np.zeros(k, dtype=int)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        else:
            max_val = np.max(self.Q)
            candidates = np.where(self.Q == max_val)[0]
            return np.random.choice(candidates)

    def update(self, action, reward):
        self.N[action] += 1
        if self.alpha is None:
            n = self.N[action]
            self.Q[action] += (1.0 / n) * (reward - self.Q[action])
        else:
            self.Q[action] += self.alpha * (reward - self.Q[action])

def run_experiment(n_runs=2000, n_steps=10000, epsilon=0.1, alpha=None):
    avg_rewards = np.zeros(n_steps)
    optimal_action_counts = np.zeros(n_steps)

    # Use tqdm to display progress over runs
    for _ in tqdm(range(n_runs), desc=f'Running {"Sample-Average" if alpha is None else "Constant-Step α="+str(alpha)}'):
        env = NonstationaryBandit()
        agent = Agent(epsilon=epsilon, alpha=alpha)

        for t in range(n_steps):
            action = agent.select_action()
            reward = env.get_reward(action)
            agent.update(action, reward)

            avg_rewards[t] += reward
            if action == env.optimal_action():
                optimal_action_counts[t] += 1

    avg_rewards /= n_runs
    optimal_action_percents = (optimal_action_counts / n_runs) * 100.0
    return avg_rewards, optimal_action_percents

# Parameters
n_runs = 2000
n_steps = 10000  # Reduced from 100000 for reasonable output
epsilon = 0.1

# Run and plot for Sample-Average method
avg_rewards_sample, opt_percents_sample = run_experiment(
    n_runs=n_runs, n_steps=n_steps, epsilon=epsilon, alpha=None
)

# Run and plot for Constant-Step method (alpha = 0.1)
avg_rewards_const, opt_percents_const = run_experiment(
    n_runs=n_runs, n_steps=n_steps, epsilon=epsilon, alpha=0.1
)

# Plot average rewards
plt.figure(figsize=(10, 4))
plt.plot(avg_rewards_sample, label="Sample-average")
plt.plot(avg_rewards_const, label="Constant-step α=0.1")
plt.xlabel("Time steps")
plt.ylabel("Average reward")
plt.legend()
plt.title("Average Reward vs. Time (Nonstationary 10-Arm Bandit)")
plt.show()

# Plot % optimal action
plt.figure(figsize=(10, 4))
plt.plot(opt_percents_sample, label="Sample-average")
plt.plot(opt_percents_const, label="Constant-step α=0.1")
plt.xlabel("Time steps")
plt.ylabel("% Optimal action")
plt.legend()
plt.title("Optimal Action % vs. Time (ε=0.1)")
plt.show()
