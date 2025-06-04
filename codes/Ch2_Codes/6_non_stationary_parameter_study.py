import jax
import jax.numpy as jnp
from jax import random, vmap
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_bandit(key, epsilon, alpha,
               k=10,
               steps=200_000,
               eval_start=100_000,
               random_walk_std=0.01):
    """
    在非平稳多臂老虎机环境下跑一次 ε-greedy 实验：
    - 每条臂的初始真实值 q_true(i,0) ~ N(0,1)
    - 每一步拉动臂 i 得到的 reward ~ N(q_true(i,t), 1)
    - q_true 每一步做独立同分布的高斯随机游走：q_true(i,t+1) = q_true(i,t) + N(0, random_walk_std^2)
    - 估计值用常数步长 α 更新：Q[action] += α * (reward - Q[action])
    - 最终返回后半段（steps - eval_start）步的平均 reward。
    """
    # 1) 用一个子键初始化 q_true(0) ~ N(0,1)
    key, init_key = random.split(key)
    q_true = random.normal(init_key, shape=(k,))  # shape=(k,)

    # 2) 估计值 Q 初始为零
    Q = jnp.zeros((k,))

    def step(carry, _):
        q_true, Q, key = carry

        # 3) 从 key 中拆分出 5 个子键：分别用于 ε-greedy 的 uniform、randint、reward、随机游走，以及保留新的 key
        key, sub_a, sub_b, sub_c, sub_d = random.split(key, 5)

        # 3.1) ε-greedy 选动作
        greedy_a = jnp.argmax(Q)
        rand_a = random.randint(sub_b, (), 0, k)           # 随机动作
        is_explore = (random.uniform(sub_a) < epsilon)     # 是否探索
        action = jnp.where(is_explore, rand_a, greedy_a)

        # 3.2) 采样 reward：reward ~ N(q_true[action], 1)
        reward = q_true[action] + random.normal(sub_c)

        # 3.3) 用常数步长 α 更新估计值 Q
        Q = Q.at[action].set(Q[action] + alpha * (reward - Q[action]))

        # 3.4) 非平稳：对每条臂的 q_true 做高斯随机游走
        q_true = q_true + random.normal(sub_d, shape=(k,)) * random_walk_std

        return (q_true, Q, key), reward

    # 4) 用 lax.scan 位置参数形式迭代 steps 次
    #    signature: lax.scan(f, init_carry, xs, length)
    (_, _, _), rewards = jax.lax.scan(
        step,
        (q_true, Q, key),  # init carry
        None,              # xs=None，因为我们不需要用到 xs
        steps              # length=steps
    )

    # 5) 取后半段（eval_start:）的平均 reward
    avg_reward = jnp.mean(rewards[eval_start:])
    return avg_reward


def run_all(key, epsilon, alpha, n_runs=2000):
    """
    并行地跑 n_runs 次 run_bandit，每次用不同的子键。
    返回 shape=(n_runs,) 的 avg_reward 数组。
    """
    subkeys = random.split(key, n_runs)
    batched_fn = vmap(lambda k: run_bandit(k, epsilon, alpha))
    return batched_fn(subkeys)


def parameter_sweep(epsilons, alpha, n_runs=2000):
    """
    对一组不同的 ε 值做参数扫描。对于每个 ε：
    1) 从当前主键 key 中 split 出一个子键用作 this_key
    2) 用 this_key 跑 run_all(..., ε, α) 得到 n_runs 次实验的 avg_reward
    3) 计算 n_runs 次实验中 avg_reward 的平均值
    最终返回 shape=(len(epsilons),) 的 avg_rewards 数组
    """
    key = random.PRNGKey(42)
    avg_rewards = []

    for eps in tqdm(epsilons, desc="Sweeping ε"):
        key, this_key = random.split(key)
        rewards = run_all(this_key, eps, alpha, n_runs)
        avg_rewards.append(jnp.mean(rewards))

    return jnp.array(avg_rewards)


if __name__ == "__main__":
    # 参数设置
    epsilons = jnp.logspace(-3, 0, num=10)  # 从 0.001 到 1.0 的 10 个 ε
    alpha = 0.1
    n_runs = 2000

    # 进行参数扫描
    avg_rewards = parameter_sweep(epsilons, alpha, n_runs)

    # 绘制结果
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, avg_rewards, marker='o')
    plt.xscale('log')
    plt.xlabel("ε (exploration rate)")
    plt.ylabel("Average reward (last 100k steps)")
    plt.title("JAX ε-Greedy with α=0.1 in Nonstationary Bandit")
    plt.grid(True)
    plt.show()
