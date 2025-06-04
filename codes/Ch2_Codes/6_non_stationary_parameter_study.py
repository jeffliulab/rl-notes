# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# class NonstationaryBandit:
#     def __init__(self, k=10, q_init=0.0, random_walk_std=0.01):
#         self.k = k
#         self.q_true = np.ones(k) * q_init
#         self.random_walk_std = random_walk_std

#     def get_reward(self, action):
#         reward = np.random.normal(loc=self.q_true[action], scale=1.0)
#         self.q_true += np.random.normal(0.0, self.random_walk_std, size=self.k)
#         return reward

#     def optimal_action(self):
#         return np.argmax(self.q_true)

# class Agent:
#     def __init__(self, k=10, epsilon=0.1, alpha=0.1):
#         self.k = k
#         self.epsilon = epsilon
#         self.alpha = alpha
#         self.Q = np.zeros(k)

#     def select_action(self):
#         if np.random.rand() < self.epsilon:
#             return np.random.randint(self.k)
#         else:
#             max_val = np.max(self.Q)
#             candidates = np.where(self.Q == max_val)[0]
#             return np.random.choice(candidates)

#     def update(self, action, reward):
#         self.Q[action] += self.alpha * (reward - self.Q[action])

# def parameter_study(epsilon_list, n_runs=2000, n_steps=200000, eval_start=100000):
#     avg_final_rewards = []

#     for epsilon in tqdm(epsilon_list, desc="Parameter sweep"):
#         rewards = np.zeros(n_steps)

#         for _ in tqdm(range(n_runs), leave=False, desc=f"ε = {epsilon:.3f}"):
#             env = NonstationaryBandit()
#             agent = Agent(epsilon=epsilon, alpha=0.1)

#             for t in range(n_steps):
#                 a = agent.select_action()
#                 r = env.get_reward(a)
#                 agent.update(a, r)
#                 rewards[t] += r

#         rewards /= n_runs
#         final_avg_reward = np.mean(rewards[eval_start:])
#         avg_final_rewards.append(final_avg_reward)

#     return avg_final_rewards

# # 参数设置
# epsilons = np.logspace(-3, 0, num=10)  # ε from 0.001 to 1.0
# avg_rewards = parameter_study(epsilons)

# # 绘图
# plt.figure(figsize=(8, 5))
# plt.plot(epsilons, avg_rewards, marker='o')
# plt.xscale('log')
# plt.xlabel("ε (exploration rate)")
# plt.ylabel("Average reward (last 100k steps)")
# plt.title("Parameter Study of ε-Greedy with α=0.1 (Nonstationary Bandit)")
# plt.grid(True)
# plt.show()

import jax
import jax.numpy as jnp
from jax import random, vmap
import matplotlib.pyplot as plt
from tqdm import tqdm

def init_bandit_state(key, k, q_init):
    q_true = jnp.ones((k,)) * q_init
    return q_true

def update_q_true(q_true, key, random_walk_std):
    noise = random.normal(key, shape=q_true.shape) * random_walk_std
    return q_true + noise

def get_reward(q_true, action, key):
    reward = random.normal(key) + q_true[action]
    return reward

def run_bandit(key, epsilon, alpha, k=10, steps=200000, eval_start=100000, random_walk_std=0.01):
    q_true = init_bandit_state(key, k, 0.0)
    Q = jnp.zeros(k)
    rewards = []

    def step(carry, t):
        q_true, Q, key = carry
        key, subkey1, subkey2, subkey3 = random.split(key, 4)

        # Action selection
        prob = random.uniform(subkey1)
        action = jnp.argmax(Q)
        action = jnp.where(prob < epsilon, random.randint(subkey2, (), 0, k), action)

        # Reward sampling
        reward = get_reward(q_true, action, subkey3)

        # Q-value update
        Q = Q.at[action].set(Q[action] + alpha * (reward - Q[action]))

        # q_true update (non-stationary)
        q_true = update_q_true(q_true, subkey1, random_walk_std)

        return (q_true, Q, key), reward

    (_, _, _), rewards = jax.lax.scan(step, (q_true, Q, key), jnp.arange(steps))
    avg_reward = jnp.mean(rewards[eval_start:])  # average over last 100k steps
    return avg_reward

# Batch over many runs
def run_all(key, epsilon, alpha, n_runs=2000):
    keys = random.split(key, n_runs)
    run_fn = vmap(lambda k: run_bandit(k, epsilon, alpha))
    return run_fn(keys)

# Parameter sweep
def parameter_sweep(epsilons, alpha, n_runs=2000):
    key = random.PRNGKey(42)
    avg_rewards = []

    for eps in tqdm(epsilons, desc="Sweeping ε"):
        rewards = run_all(key, eps, alpha, n_runs)
        avg_rewards.append(jnp.mean(rewards))

    return jnp.array(avg_rewards)

# 参数设置
epsilons = jnp.logspace(-3, 0, num=10)
alpha = 0.1
avg_rewards = parameter_sweep(epsilons, alpha)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(epsilons, avg_rewards, marker='o')
plt.xscale('log')
plt.xlabel("ε (exploration rate)")
plt.ylabel("Average reward (last 100k steps)")
plt.title("JAX Accelerated ε-Greedy with α=0.1 (Nonstationary Bandit)")
plt.grid(True)
plt.show()
