import numpy as np
import matplotlib.pyplot as plt

# Gradient Bandit with optional baseline
class Bandit:
    def __init__(self, k=10):
        self.k = k
        self.q_star = np.random.normal(loc=0.0, scale=1.0, size=k)
        self.optimal_action = np.argmax(self.q_star)

    def reward(self, a):
        return np.random.normal(loc=self.q_star[a], scale=1.0)

class Agent:
    def __init__(self, k, alpha, use_baseline=True):
        self.k = k
        self.alpha = alpha
        self.use_baseline = use_baseline
        self.H = np.zeros(k)
        self.baseline = 0.0
        self.t = 0

    def select_action(self):
        expH = np.exp(self.H - np.max(self.H))
        pi = expH / np.sum(expH)
        action = np.random.choice(self.k, p=pi)
        return action, pi

    def update(self, a, R, pi):
        self.t += 1
        if self.use_baseline:
            self.baseline += (R - self.baseline) / self.t
            advantage = R - self.baseline
        else:
            advantage = R

        self.H[a] += self.alpha * advantage * (1 - pi[a])
        for i in range(self.k):
            if i != a:
                self.H[i] -= self.alpha * advantage * pi[i]

def simulate(runs=1000, time_steps=1000, k=10, alphas=[0.1, 0.4]):
    optimal_results = {}
    reward_results = {}
    for alpha in alphas:
        for use_baseline in [True, False]:
            optimal_counts = np.zeros(time_steps)
            reward_sums = np.zeros(time_steps)
            for run in range(runs):
                bandit = Bandit(k)
                agent = Agent(k, alpha, use_baseline)
                for t in range(time_steps):
                    a, pi = agent.select_action()
                    R = bandit.reward(a)
                    agent.update(a, R, pi)
                    reward_sums[t] += R
                    if a == bandit.optimal_action:
                        optimal_counts[t] += 1
            optimal_pct = (optimal_counts / runs) * 100
            avg_rewards = reward_sums / runs
            label = f"{'with' if use_baseline else 'without'} baseline, α={alpha}"
            optimal_results[label] = optimal_pct
            reward_results[label] = avg_rewards
    return optimal_results, reward_results

# Run simulation
opt_results, rew_results = simulate(runs=1000, time_steps=1000, k=10, alphas=[0.1, 0.4])

# Plotting Optimal Action %
plt.figure(figsize=(10, 6))
for label, data in opt_results.items():
    plt.plot(data, label=label)
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.title('Gradient Bandit: Optimal Action % over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Average Rewards
plt.figure(figsize=(10, 6))
for label, data in rew_results.items():
    plt.plot(data, label=label)
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Gradient Bandit: Average Reward over Time')
plt.legend()
plt.grid(True)
plt.show()
