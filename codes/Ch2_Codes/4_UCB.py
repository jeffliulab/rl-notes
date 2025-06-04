import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class Bandit:
    def __init__(self, k=10):
        self.k = k
        self.real_q = np.random.normal(0, 1, k)

    def reward(self, a):
        return np.random.normal(self.real_q[a], 1)


class AgentUCB:
    def __init__(self, k=10, c=2):
        self.k = k
        self.c = c
        self.Q = np.zeros(k)
        self.counts = np.zeros(k, dtype=int)
        self.total_count = 0

    def choose_action(self):
        self.total_count += 1
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.total_count) / (self.counts + 1e-10))
        zero_mask = (self.counts == 0)
        if np.any(zero_mask):
            return np.argmax(zero_mask)
        return np.argmax(ucb_values)

    def update(self, a, reward):
        self.counts[a] += 1
        step = 1 / self.counts[a]
        self.Q[a] += step * (reward - self.Q[a])


class AgentEpsilon:
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)
        self.counts = np.zeros(k, dtype=int)

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.Q)

    def update(self, a, reward):
        self.counts[a] += 1
        step = 1 / self.counts[a]
        self.Q[a] += step * (reward - self.Q[a])


def main(runs=2000, steps=1000):
    avg_rewards_ucb = np.zeros(steps)
    avg_rewards_eps = np.zeros(steps)

    for _ in range(runs):
        bandit = Bandit()
        agent_ucb = AgentUCB()
        agent_eps = AgentEpsilon(epsilon=0.1)

        rewards_ucb = np.zeros(steps)
        rewards_eps = np.zeros(steps)

        for t in range(steps):
            a_ucb = agent_ucb.choose_action()
            r_ucb = bandit.reward(a_ucb)
            agent_ucb.update(a_ucb, r_ucb)
            rewards_ucb[t] = r_ucb

            a_eps = agent_eps.choose_action()
            r_eps = bandit.reward(a_eps)
            agent_eps.update(a_eps, r_eps)
            rewards_eps[t] = r_eps

        avg_rewards_ucb += rewards_ucb
        avg_rewards_eps += rewards_eps

    avg_rewards_ucb /= runs
    avg_rewards_eps /= runs

    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards_ucb, label='UCB (c=2)')
    plt.plot(avg_rewards_eps, label='Epsilon-Greedy (ε=0.1)')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward: UCB vs Epsilon-Greedy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
