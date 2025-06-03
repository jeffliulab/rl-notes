import numpy as np
import matplotlib.pyplot as plt

# 老虎机环境
class SlotMachine:
    def __init__(self, k=10):
        self.k = k
        self.real_q = np.random.normal(0, 1, k)  # 每个动作的真实期望奖励

    def reward(self, a):
        return np.random.normal(self.real_q[a], 1)  # 奖励分布 N(real_q[a], 1)

# 代理
class Agent:
    def __init__(self, Q_init, epsilon, alpha=0.1, k=10):
        self.Q = np.ones(k) * Q_init  # 初始化 Q 表
        self.epsilon = epsilon        # ε-贪婪参数
        self.alpha = alpha            # 学习率
        self.k = k

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.k)  # 探索
        else:
            return np.argmax(self.Q)             # 利用

    def update_Q(self, a, reward):
        self.Q[a] += self.alpha * (reward - self.Q[a])  # Q 更新公式

# 主实验函数
def main(runs=2000, steps=1000):
    avg_rewards_1 = np.zeros(steps)  # 乐观初始值策略
    avg_rewards_2 = np.zeros(steps)  # ε-贪婪策略

    for run in range(runs):
        bandit = SlotMachine()
        agent1 = Agent(Q_init=5, epsilon=0)        # 乐观贪婪
        agent2 = Agent(Q_init=0, epsilon=0.1)      # ε-贪婪

        rewards1 = []
        rewards2 = []

        for t in range(steps):
            # Agent 1
            a1 = agent1.choose_action()
            r1 = bandit.reward(a1)
            agent1.update_Q(a1, r1)
            rewards1.append(r1)

            # Agent 2
            a2 = agent2.choose_action()
            r2 = bandit.reward(a2)
            agent2.update_Q(a2, r2)
            rewards2.append(r2)

        avg_rewards_1 += np.array(rewards1)
        avg_rewards_2 += np.array(rewards2)

    # 求平均
    avg_rewards_1 /= runs
    avg_rewards_2 /= runs

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards_1, label='Optimistic Greedy Q=5, ε=0')
    plt.plot(avg_rewards_2, label='ε-Greedy Q=0, ε=0.1')
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Optimistic Initial Values vs ε-Greedy")
    plt.legend()
    plt.grid(True)
    plt.show()

# 运行实验
if __name__ == "__main__":
    main()
