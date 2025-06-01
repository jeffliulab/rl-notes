import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)

# 模拟一个 10 臂赌博机
k = 10

# 为每个动作生成真实平均奖励 q_*(a)
q_star = np.random.normal(loc=0.0, scale=1.0, size=k)

# 生成每个动作的实际获得的奖励样本
rewards = []  # 最终是一个长度为10的列表，每个元素是该动作的1000次奖励
for i in range(k):
    q = q_star[i]  # 第i个动作的真实平均值
    reward_samples = np.random.normal(loc=q, scale=1.0, size=1000)  # 模拟拉1000次
    rewards.append(reward_samples)

# 创建小提琴图来展示奖励分布
plt.figure(figsize=(12, 6))
plt.violinplot(rewards, showmeans=True, showextrema=False)
plt.axhline(0, color='gray', linestyle='--')
plt.xticks(np.arange(1, k + 1))  # 标出动作编号
plt.xlabel("Action")
plt.ylabel("Reward Distribution")
plt.title("Reward Distributions for 10-Armed Bandit Actions (Using For Loop)")
plt.grid(True)
plt.tight_layout()
plt.show()
