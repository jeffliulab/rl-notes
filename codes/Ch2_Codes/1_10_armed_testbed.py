import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------
# 类：多臂老虎机环境（BanditEnv）
# ----------------------------
class BanditEnv:
    def __init__(self, k=10):
        self.k = k
        # 按照标准正态分布生成 k 个动作的真实平均奖励 q*
        self.q_star = np.random.normal(0, 1, k)
        # 记录当前环境下的最优动作索引
        self.best_action = np.argmax(self.q_star)

    def get_reward(self, action):
        """
        根据动作 a 的真实平均值 self.q_star[action]，从 N(q_star[action], 1) 中采样
        并返回该次的奖励
        """
        return np.random.normal(self.q_star[action], 1)


# ----------------------------
# 类：ε-Greedy 智能体（EpsilonGreedyAgent）
# ----------------------------
class EpsilonGreedyAgent:
    def __init__(self, k, epsilon):
        # 探索概率 ε
        self.epsilon = epsilon
        self.k = k
        # 初始化 k 个动作的估计值 Q[a] = 0
        self.Q = np.zeros(k)
        # 初始化每个动作被选择的计数 N[a] = 0
        self.N = np.zeros(k)

    def select_action(self):
        """
        ε-greedy 策略：
        - 以 ε 的概率随机探索动作；
        - 否则以 1−ε 的概率进行贪心选择（选择当前 Q[a] 最大的动作，若有并列则随机选一个）。
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.k)
        else:
            max_Q = np.max(self.Q)
            candidates = np.where(self.Q == max_Q)[0]
            return np.random.choice(candidates)

    def update(self, action, reward):
        """
        样本平均法更新 Q 值：
            N[a] += 1
            Q[a] ← Q[a] + (1 / N[a]) * (reward − Q[a])
        """
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


# ----------------------------
# 函数：run_one_episode
#   执行一次（单 trial）→ 固定 k, steps, ε
# ----------------------------
def run_one_episode(k, steps, epsilon):
    """
    对应于单次试验（1000 步）：
    - 先初始化 环境（BanditEnv） 和 智能体（EpsilonGreedyAgent）；
    - 每个时间步 t：
        1. agent.select_action() 选择动作 a
        2. reward = env.get_reward(a)
        3. agent.update(a, reward)
        4. 记录 rewards[t] 和 是否选到最优动作 optimal_actions[t]
    返回：
      rewards:           长度为 steps 的奖励序列
      optimal_actions:   长度为 steps 的二值序列（1 表示该步选到最优动作）
    """
    env = BanditEnv(k)
    agent = EpsilonGreedyAgent(k, epsilon)
    rewards = np.zeros(steps)
    optimal_actions = np.zeros(steps)

    for t in range(steps):
        a = agent.select_action()
        r = env.get_reward(a)
        agent.update(a, r)

        rewards[t] = r
        if a == env.best_action:
            optimal_actions[t] = 1

    return rewards, optimal_actions


# ----------------------------
# 函数：run_experiment
#   对不同 ε 值，进行多次 runs（例如 2000 次）试验，统计平均
#   同时在 runs 循环中使用 tqdm 显示进度
# ----------------------------
def run_experiment(k=10, steps=1000, runs=2000, epsilons=[0, 0.01, 0.1]):
    """
    输出：
      avg_rewards:      形状为 (len(epsilons), steps) 的数组，每行对应某个 ε 的“每步平均奖励”
      optimal_rates:    形状为 (len(epsilons), steps) 的数组，每行对应某个 ε 的“每步最优动作命中率（%）”
    """
    avg_rewards = np.zeros((len(epsilons), steps))
    optimal_rates = np.zeros((len(epsilons), steps))

    for idx, epsilon in enumerate(epsilons):
        # tqdm 显示当前 ε 下的 runs 次试验进度
        for _ in tqdm(range(runs), desc=f"ε = {epsilon}"):
            rewards, optimal_actions = run_one_episode(k, steps, epsilon)
            avg_rewards[idx] += rewards
            optimal_rates[idx] += optimal_actions

        # 对 runs 次试验结果求平均
        avg_rewards[idx] /= runs
        # 由命中次数转换为百分比
        optimal_rates[idx] = (optimal_rates[idx] / runs) * 100

    return avg_rewards, optimal_rates


# ----------------------------
# 主程序入口
# ----------------------------
if __name__ == "__main__":
    # 参数设定
    k = 10
    steps = 1000
    runs = 2000
    epsilons = [0, 0.01, 0.1]

    # 运行实验，显示进度条
    avg_rewards, optimal_rates = run_experiment(k, steps, runs, epsilons)

    # 可视化：不同 ε 下的平均奖励曲线
    plt.figure(figsize=(12, 5))
    for idx, epsilon in enumerate(epsilons):
        plt.plot(avg_rewards[idx], label=f"ε = {epsilon}")
    plt.xlabel("Time Step")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.title("Average Reward over Time for Different ε Values")
    plt.grid(True)
    plt.show()

    # 可视化：不同 ε 下的最优动作选择率（百分比）曲线
    plt.figure(figsize=(12, 5))
    for idx, epsilon in enumerate(epsilons):
        plt.plot(optimal_rates[idx], label=f"ε = {epsilon}")
    plt.xlabel("Time Step")
    plt.ylabel("Optimal Action Percentage (%)")
    plt.legend()
    plt.title("Percentage of Optimal Action over Time for Different ε Values")
    plt.grid(True)
    plt.show()
