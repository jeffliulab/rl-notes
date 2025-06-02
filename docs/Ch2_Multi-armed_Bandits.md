# Chapter 2 多臂老虎机

注：图片来源于网络，如果该图片有版权，请联系我删除。本页内容皆为公开免费的笔记，不用于盈利。

## 2.0 导读

从第二章多臂老虎机开始到第八章Planning and Learning with Tabular Methods，作者将这部分统一归纳为本书的“第一部分：Tabular Solution Methods”，也就是“表格解法”。

在上一章，我们已经梳理了强化学习的发展脉络，以及表格解法对现代强化学习的兴起的重要性。表格解法是几乎所有强化学习核心思想的最简形式：在状态空间和动作空间足够小的情况下，近似值函数可以被表示为数组，或者称作表格（tables）。在这种情况下，这些方法往往能够找到精确解（exact solutions），即最优值函数和最优策略。

第九章开始，将介绍另一种只能找到近似解的情况。正如上一章节历史综述中所提到的，现代强化学习的发展开始于表格解法，但在现实中大规模应用则是结合人工神经网络后，强化学习表现出了惊人的产出。

在Part I，我们将分别学习：

| 章节 | 内容                         | 重要概念                  |
| ---- | ---------------------------- | ------------------------- |
| 2    | 赌博机问题                   | ε-greedy                 |
| 3    | 马尔可夫决策过程             | 贝尔曼方程，价值函数      |
| 4    | 动态规划                     | 环境模型                  |
| 5    | 蒙特卡洛方法                 | 基于经验的估计            |
| 6    | 时序差分学习                 | Bootstrapping（自举更新） |
| 7    | 蒙特卡洛与时序差分的结合     | TD(λ) 多步更新           |
| 8    | 时序差分与模型学习方法的结合 | Dyna架构                  |

在本章我们主要学习强化学习中一个比较简单但却非常重要的概念：对于探索和利用的权衡（trade-off between exploitation and exploration）。强化学习和其他类型的学习方法最重要的区别之一，就是RL使用的训练信息是对所采取的动作进行评价（evaluates），而非通过给出正确动作来指示（instructs）。那么，显而易见的，如果环境智能评价agent做的动作好不好，而无法让agent知道什么动作是正确的，那么agent就必须去主动探索，因为如果agent不去探索的话可能就永远无法知道是否还有更好的动作存在了。其根本原因就是agent能得到信息总是处于一种局部可见的状态，是不完整的，因此必须不断通过试错来了解哪些动作的奖励高，进而逐步找到最优策略。因为环境无法给予指示，因而主动探索是找到最优策略的唯一途径。

举例来说，假设你正在玩超级马里奥，指示型反馈会根据训练数据直接告诉你：在这个画面上，应该往右跳。你照着做就行了，因为有明确的指示。这是监督学习的重要特征。在这个情况下，无论你做什么，系统都不会对你的动作进行评价，只会告诉你：在这个状态下，正确动作时往右走。

而在强化学习中，你控制马里奥，按了一个动作，系统将告诉你：你现在得了200分。系统不会告诉你：满分是多少。因此，你只能不断地玩，不断地玩，来看自己的分数能不能继续提高。在这种情况下，你的动作是系统关注的，系统知道怎么去评价，但是无法告诉你正确答案。

## 2.1 K臂老虎机问题

在本章中，我们将用一个具体的例子，来说明评价系统是如何运作的。该例子是一个简化版本的强化学习问题，它不具备联结性（associativity）。联结性的意思就是动作选择要与当前情境或者状态相联系，比如说，当agent在迷宫中探索时，在迷宫不同的位置的时候显然会进行不同的动作。比如在左边是墙壁的时候，显然不能继续向左走了；在右边是墙壁的时候，显然也不能向右走了。这个时候，向哪边走（动作）受到所处位置（状态）的影响，他们不能分开探讨，这种情况就是联结性的情况，也是所有现实问题显然具备的特征。

为了让问题简化，我们先来讨论非联结性（nonassociative）的例子。现在想象这样一个场景：你的面前有一个老虎机：

![1748722008294](image/Ch2_Multi-armed_Bandits/1748722008294.png){style="display:block; margin:auto; width:400px;"}

这个老虎机上面有一个拉杆，你可以拉一下拉杆，然后屏幕上就会转动：

![1748722154314](image/Ch2_Multi-armed_Bandits/1748722154314.png){style="display:block; margin:auto; width:400px;"}

如果转动后停止时屏幕正中心的图表全都是相同的，那么你就中奖了！

读者在拉斯维加斯就玩过这个机器，花了十美元，中了二十美元，然后和朋友买了星巴克。读者个人不喜欢赌博，以前最多的赌博就是买十块钱的刮刮乐，还经常能中20块钱。

我们回到主题上来，现在假设你是一个赌狗，想用自己的知识来解决一下这个问题，你首先会想什么呢？没错，我想每个人都会先环视一圈，看看不同的老虎机，然后决定用哪个老虎机来吐钱，因为在老虎机游戏上这可能是你唯一能做的决策了。那么假设你的面前现在有k个老虎机，你可以选择任何一个去拉，你的目标是找到收益最大的那个老虎机。这就是k臂老虎机问题：

![1748722400299](image/Ch2_Multi-armed_Bandits/1748722400299.png){style="display:block; margin:auto; width:400px;"}

我们来系统性地看一下这个问题：你的目标其实就是在一定时间段内最大化期望地累计奖励，比如说在1000次动作（在强化学习中，我们把1000次动作称为1000个时间步 time steps）中获得的总奖励最大。

那么，假设每个动作a都有一个期望奖励（expected reward）或平均奖励，我们称这个期望奖励就是这个动作的价值（value）。

我们用At代表第t步所选择的动作，Rt代表所获得的奖励，那么某个动作a的“真实”价值就可以记作：

$$
q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a]
$$

这个公式的意思是：在选择动作a的时候，所期望得到的奖励。对于类似读者这种可能没怎么学过数学的人，说明一下该公式的每个元素含义如下：

* a，代表动作，也就是实际选择的那个动作
* At，代表在时间步t采取的动作，At=a就表示给定当前采取的动作是a
* Rt，代表在时间步t收到的奖励
* |，条件符号，Rt | At=a的意思就是在选择动作a的情况下得到的奖励
* $\mathbb{E}[ X | Y]$，条件期望，在Y的条件下，X的期望值
* $\doteq$，定义符号，代表上面是个定义而不是等式
* $q_*$，这里的星号\*代表最优，q代表期望得到的奖励，$q_*(a)$代表选择动作a的时候所期望得到的奖励

复习一下期望值（expected value）：期望指的是试验中每次可能的结果乘以其结果概率的总和。这里要注意，我们这里提到的$q_*(a)$指的是真实值，下一节继续说明如何得到该值。

## 2.2 动作价值方法 Action-value Methods

我们无法一上来就得到真实值，而只能通过多次试验来得到估计值。一种非常直观地估计方法就是简答地采取该动作过去所获得奖励的平均值，也即样本平均法（sample-average method）：

$$
Q_t(a) \doteq 
\frac{\text{在 } t \text{ 之前采取 } a \text{ 时获得的奖励之和}}
     {\text{在 } t \text{ 之前采取动作 } a \text{ 的次数}}
= 
\frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i = a}}
     {\sum_{i=1}^{t-1} \mathbb{1}_{A_i = a}}
$$

其中：

* $\mathbb{1}_{\text{predicate}}$，指示函数，当条件为真时取值为1，否则为0。
* Qt(a)，动作价值函数

如果该动作从未被选择过，那么分母就为0，所以这个时候需要定义一下该特殊情况地Qt(a)，一般取值为0。根据大数定律，当分母趋近于无穷大的时候，$Q_t(a)$会趋近于真实值$q_*(a)$。

现在，在任何一个时刻，当你面对k个老虎机的时候，你都可以通过上述样本平均法来计算出每一个动作对应的估计值。在这些值中，你可以采取两种动作：

* 贪婪动作（greedy action）：选择值最高的动作。因为利用了值的特征，所以我们把贪婪动作叫做利用（exploiting）。
* 非贪婪动作（non-greedy action）：选择值不是最高的动作，一般是随机选择动作。因为没有利用值，所以我们把这种叫做探索（exploring）。

在这个任务中，因为估计值是通过采样计算得来的，所以它只反映了已经发生过的情况的结果，因而是片面的。如果不想遗漏更优可能，那么探索就是必须的。一开始探索的奖励可能会比较低，但一旦能找到更好的动作，就可以反复利用它，从而增加长期收益。由于不能同时探索和利用，所以在探索和利用之间存在一个复杂的权衡，这也是强化学习的一个核心要点。

在权衡中，我们一般考虑：

* 当前估计值的精确程度
* 各个动作的不确定性
* 剩余的时间步数量

在本书中，我们的关注焦点不在探索vs利用的权衡上，因而只考虑简单的平衡策略。当我们基于这些值来进行动作选择的时候，我们就称这类方法为动作价值方法（action-value methods）。一般来说，完整的动作价值函数还要包含状态，即$Q_t(s,a)$，但是老虎机问题中不存在状态，所以这里只探讨简化的动作价值函数。

现在，我们可以定一下动作选择规则，假设只选择估计价值最高的动作，即贪婪动作，我们可以记作：

$$
A_t \doteq \arg\max_a Q_t(a)
$$

其中$\arg\max_a$表示使得Q_t(a)最大的动作a。

但这显然不行。显然，我们需要考虑探索进去。一种常见的做法是大部分时间做贪婪选择，偶尔以小概率ε（epsilon）随机选择任一动作。我们把这种方法称为ε贪婪方法（ε-greedy methods）。

当步数趋近于无穷时，每个动作最终都会被尝试无数次，因此每个a的估计值都能收敛到真实值，而最终最优动作的选择概率也会趋近于1-ε，即几乎总是能选择最优动作。不过，这种渐进性质（asymptotic gurantees）并不能完全反映该方法在优先步数下的实际效果。

## 2.3 多臂老虎机模拟测试

下面我们来设计一个模拟实验。课本中给出了一个十臂测试平台（The 10-armed Testbed）来评估并对比greedy和ε-greedy。

下面将书中的测试复现一下，实验要点如下：

* k=10，设置10个老虎机（或者说10臂老虎机）进行模拟测验
* 预先设定好10个老虎机的真实奖励$q_*(a)$，在下面我会详细说怎么设置；这个奖励是隐藏的，是agent看不到的
* 每次动作选择的时候，在真实奖励的基础上，会进行一个随机的噪音干扰

上述testbed设置好后，就相当于模拟好了一个强化学习的环境，然后我们就可以用强化学习的算法去让agent在这个环境中学习，最后进行评估。在这个例子中，我们将初步学习到如何设置一个简单的模拟环境、测试、以及最终的评估。

首先，在实验开始前，先说明一下testbed的设置。k=10很好理解，就是10个老虎机或者说10臂老虎机。

预先设置的真实奖励用标准正态分布生成：

$$
q_*(a) \sim \mathcal{N}(0, 1)
$$

这个的意思是，每个动作的“好坏”（平均奖励）都是从平均为0、波动为1的分布中随机出来的。

对于每个动作，在最终实际执行的时候，得到的最终奖励会受到噪音的影响，我们可以用类似的方式设置实际奖励Rt为：

$$
R_t \sim \mathcal{N}(q_*(a), 1)
$$

简单来说，就是先按照标准正态分布（中心均值为0，方差为1）生成10个真实奖励值；然后对每个动作，再以该动作的真实奖励为中心均值，并以方差为1，进行实际奖励发放。

我们可以在python中设置该分布：

```python
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

```

因为设置了随机种子，所以其实就是用了一组总是能固定不变的伪随机数。之所以用正态分布，是因为奖励集中在某个值附近（比如平均值），但偶尔有一些波动（好运气/坏运气），并且在数学上好处理。对每个动作进行1000次实际奖励方法取样，可以得到结果如下：

![1748738896389](image/Ch2_Multi-armed_Bandits/1748738896389.png){style="display:block; margin:auto; width:800px"}

这个图叫做“小提琴图”，可以很直观地显示出该testbed设置方法下最终的真实q值的分布。在接下来的实际测试中，agent是无法观测到这些东西的，只能通过实际得到的奖励，不断调整自己的Q值估计，最终当Q值收敛的时候，我们就能看到agent是否找到了正确的答案。

现在知道了环境怎么给奖励，我们就可以开始实验了。假设agent一开始对10个动作的估计Q值都是0，那么在第一回合的时候，无论agent选择的ε是多少（我们在这里的讨论中，将纯greedy视为ε为0的情况），显然面对10个Q值一样的状态，agent都只能随机从中选择一个。当Q值一样的时候随机选一个是一般的处理方法。

无论agent随机选了哪一个动作，这个时候环境会给予agent一个奖励（即上述我们设置的带有噪音的那个奖励），在这里我们可以简单地通过样本平均更新法来更新该动作a对应的Q值，Q(n+1)就是在看了前n次的奖励后估计的动作价值：

$$
Q_{n+1}(a) = \frac{R_1 + R_2 + \cdots + R_n}{n}
$$

该方法在实际实验及编程中并不实用，因为太占内存了。很显然，按照编程的思想，我们总是会设置一个循环，然后在循环内通过不断更新值的方式去更新一个变量（比如非常常见的for i in... i+=1），同理，Q表更新也可以用同样的方式完成：

$$
\begin{aligned}
Q_{n+1} &= \frac{1}{n} \sum_{i=1}^{n} R_i \\
        &= \frac{1}{n} \left( R_n + \sum_{i=1}^{n-1} R_i \right) \\
        &= \frac{1}{n} \left( R_n + (n - 1) Q_n \right) \\
        &= Q_n + \frac{1}{n} (R_n - Q_n)
\end{aligned}
$$

这个公式在n=1时也成立，此时Q2 = R1对任意Q1成立。

那么我们接下来就可以设置一个测试环境，对比一下不同ε之下的不同策略中，agent在找到最高奖励的bandit的表现是如何的。

根据上述公式和我们对奖励的设置，边读边写，可以写出整个程序的草稿框架如下：

```text
伪代码：多臂老虎机测试

def reward（agent选择了老虎机a）：
  设置该函数来按照上面我们已经讨论过的规则，将对应老虎机的奖励返还给agent
  k = 10，代表10个老虎机
  按照标准正态分布设置十个老虎机的真实奖励q*
  然后根据导入的参数老虎机k（代表agent选择的对应的老虎机）返回一个带噪音的Rt
  Rt(a) = 正态分布（以q*为中心，方差为1）
  return Rt(a)

def agent：
  agent就是强化学习智能体，该智能体将和环境交互
  agent初始维护一个值全为0，长度为k=10的Q表
  Q = [0] * 10
  然后agent在每次的选择中，将根据Q表中的最大值来决定选择哪个动作at：
  （1）在Q表中找到最大的值，返回其对应的index，即选择动作；如果有多个最大值，则随机选择一个
  （2）调用reward（a），获得对应的奖励
  （3）根据实际的反馈，更新Q值，这里按照样本平均更新法来更新

def main：
  接着我们进行2000次试验，在每次试验中进行1000个时间步的运行：
    agent和环境交互，以ε的概率随机选择，以1-ε的概率按照agent中定义的方式选择
    选择后更新Q值
  试验结束后绘制图表，展现1000个时间步在2000次试验中的平均奖励，以及选择最优动作的百分比（理想状况下ε策略将按照1-ε的概率选择最优动作）
```

这种写法显然不够优雅，只是无脑的把我们刚才学到的奖励的概念和agent的概念复述了一遍。我们可以提炼一下要点用面向对象的方法来设计一下，整个过程中其实就老虎机本身（环境）和agent两个对象，所以伪代码可以优化为：

```text
class 老虎机：
    def init()：
        初始化k=10
	按照标准正态分布生成上述k个动作的真实平均奖励q*

    def 实际奖励(选择的动作a)：
	reward = 以动作a的真实平均奖励q*为中心，方差为1的正态分布中的值
	return reward

class Agent：
    def init(epsilon)：
	初始化epsilon
	初始化Q表（按照k的值设置对应数量的初始值，这里设置为0）

    def 动作选择()：
	生成一个随机数
	if 该随机数小于epsilon：
	    进行ε-greedy选择一个随机动作
	    return 动作a
	else：
	    进行greedy，选择一个值最大的动作，如果都一样则从中随机选择一个
	    return 动作a

    def 更新Q表(动作a，实际奖励q)：
	根据动作a和实际奖励q，更新Q(a)

def main():
    agent = Agent(epsilon=0/0.01/0.1)
    老虎机 = 老虎机()
    进行1000次循环：
	动作a = agent.动作选择()
	奖励q = 老虎机.实际奖励()
	agent.更新Q表(动作a, 奖励q)

if self = _init_:
    新建三个Q表，用来存放2000次试验下不同epsilon值的平均值
    遍历三种epsilon值：
        进行2000次试验：
	    main()


```

现在看起来像那么回事了，我们继续把一些细节写好，在后续的笔记种，我将省略上述过程，直接展示最终写好细节的伪代码。这里之所以赘述，是为了方便那些不怎么会编程的人学习（如今学习AI的人里不怎么会写代码的人越来越多了，这也正常，因为很多人主要研究数学！但强化学习非常适合应用，特别是和机器人结合在一起，所以会有比较多的编程！但这里读者觉得，在学习强化学习过程中，学会写伪代码就可以了，因为强化学习的代码一般不会太长，最重要的是策略和环境设置，具体的代码实现可以直接让ChatGPT写就行了！）

在写整个程序的伪代码的时候，最关键的就是面向对象的设计，如果你还没有学过面向对象的概念，可以找个python/java的网络教程快速看一下，大概一个下午就能把主要的特点如对象、继承、重写等理解学会了。一般来说，读者认为能把下面我展示的这个水平的伪代码写出来，对于强化学习编程来说就没有问题了：

```python
# 多臂老虎机测试伪代码
class 老虎机:
    def __init__():
        k = 10  # 动作数
        # 按照标准正态分布生成 k 个动作的真实平均奖励 q*
        q_star = [从 N(0, 1) 中采样得到的长度为 k 的列表]
        # 记录当前环境下的最优动作索引
        best_action = argmax(q_star)

    def 实际奖励(选择的动作 a):
        # 根据动作 a 的真实平均值 q_star[a]，从 N(q_star[a], 1) 中采样
        reward = 从 N(q_star[a], 1) 中采样得到的标量
        return reward


class Agent:
    def __init__(epsilon):
        # 探索概率 ε
        self.epsilon = epsilon
        # 初始化 k 个动作的估计值 Q[a] = 0
        self.Q = 长度为 k、全零的列表
        # 初始化每个动作被选择的计数 N[a] = 0
        self.N = 长度为 k、全零的列表

    def 动作选择():
        随机数 = 从 [0, 1) 均匀采样
        if 随机数 < epsilon:
            # 以 ε 的概率随机探索
            a = 在 0 到 k-1 之间随机选一个动作索引
            return a
        else:
            # 以 1 - ε 的概率进行贪心选择
            max_Q = max(self.Q)
            # 找出所有 Q[a] == max_Q 的索引列表
            candidates = [所有满足 Q[a] == max_Q 的索引]
            # 如果有多个相同的最大值，从中随机选择一个
            a = 在 candidates 列表中随机选一个索引
            return a

    def 更新Q表(动作 a, 实际奖励 r):
        # 计数加一
        N[a] += 1
        # 样本平均法更新：Q[a] ← Q[a] + (1 / N[a]) * (r - Q[a])
        Q[a] = Q[a] + (r - Q[a]) / N[a]


def main():
    # k = 10，steps = 1000
    # 在 main 中只跑一次环境交互（一个试验/一次序列）
    agent = Agent(epsilon=0 或 0.01 或 0.1)
    env = 老虎机()
    # 用于记录单次试验中每个时间步 t 的奖励和最优动作命中情况
    rewards_list = 长度为 1000、初始值全零的列表
    optimal_action_flags = 长度为 1000、初始值全零的列表

    for t in range(1000):  # 进行 1000 个时间步
        a = agent.动作选择()
        r = env.实际奖励(a)
        agent.更新Q表(a, r)

        rewards_list[t] = r
        if a == env.best_action:
            optimal_action_flags[t] = 1

    return rewards_list, optimal_action_flags


if self = _init_:
    k = 10
    steps = 1000
    runs = 2000
    epsilons = [0, 0.01, 0.1]

    # 用于累计所有 runs 次试验中，每个 ε 对应的 1000 个时间步的总奖励
    总奖励 = 一个字典，键为 ε，值为长度为 1000 的全零列表
    # 用于累计最优动作命中次数
    总最优命中 = 一个字典，键为 ε，值为长度为 1000 的全零列表

    # 遍历三种 ε
    for ε in epsilons:
        for run in range(runs):
            # 每个 run 都重新执行一次 main，得到该次序列的 rewards 和 optimal flags
            rewards_list, optimal_flags = main()
            # 累加到对应 ε 的“总奖励”和“总最优命中”中
            for t in range(steps):
                总奖励[ε][t] += rewards_list[t]
                总最优命中[ε][t] += optimal_flags[t]

    # 计算平均值：每个ε下，每个时间步的平均奖励 和 最优动作命中率（%）
    平均奖励 = 一个字典，键为 ε，值为长度为 1000 的列表
    最优命中率 = 一个字典，键为 ε，值为长度为 1000 的列表

    for ε in epsilons:
        for t in range(steps):
            平均奖励[ε][t] = 总奖励[ε][t] / runs
            # 将命中次数转换为百分比
            最优命中率[ε][t] = (总最优命中[ε][t] / runs) * 100

    # 最后可以将 平均奖励 和 最优命中率 用于绘图或表格展示
    # 此处省略具体绘图代码，可以直接告诉ChatGPT用 matplotlib 画就行了

```

按照上述伪代码，直接发给ChatGPT，让它写好，然后在python环境中运行即可，具体的代码在本小节笔记最后的代码附录中或者我的github中都能找到，无特殊情况直接在附录中找就行了，因为github repo中不仅有笔记文件和代码文件，还有一些其他比如让这个笔记百科显示出来的文件等，稍微乱一点。

具体代码见**附录 Code 2.1**。

运行后可以得到如下结果：

![1748808472234](image/Ch2_Multi-armed_Bandits/1748808472234.png){style="display:block; margin:auto; width:800px;"}

这张图展示了2000次试验中，1000个时间步内三种ε策略的平均奖励。这张**平均奖励随时间变化图**是几乎所有RL任务都要出的一张图，它能最直观地显示agent的学习趋势。

![1748808496128](image/Ch2_Multi-armed_Bandits/1748808496128.png){style="display:block; margin:auto; width:800px;"}

这张图则是**最优动作选择率图**，用来分析探索vs利用。

从上面两张图可以很直观地看出来，三种策略中明显ε=0.1的效果是最好的。我们根据这两张图来深入分析一下，可以发现：

1. 纯greedy策略很快便进行了收敛，陷入了局部最优
2. ε=0.1的情况收敛到了1.5左右，这已经接近理论最优值1.54，这一点在数学上可以证明，参见附录中的Math 2.1。
3. ε=0.1的最优动作选择率并没有收敛到理论上最优的91%（随机探索中有10%的概率选中最优，所以理论值是90%+10%*10%），这是非常有意思的一点。可能性是多维度的，因为图中展示的是2000次试验的平均结果，所以有可能在一些trial中agent一直都没能找到最好的那个老虎机。具体的原因读者也没有继续深究下去，感兴趣的朋友如果深究出了一个非常具体的原因，或者能找到一个能收敛到理论最优值的ε，麻烦告诉读者，十分感谢！

本节的练习题参考附录中的Exercise 2.2, Exercise 2.3。

## 2.4 Incremental Implementation

在上一节，我们已经介绍Q值的一个比较巧妙的更新公式：

$$
\begin{aligned}
Q_{n+1} = Q_n + \frac{1}{n} (R_n - Q_n)
\end{aligned}
$$

这个公式在书中这里才提出来，但是上一章为了完成代码和练习我们已经使用了，这种公式叫做增量公式。增量实现（Incremental Implementation）在本书中将多次出现，它的一般形式为：

$$
\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize} \cdot (\text{Target} - \text{OldEstimate})
$$

其中：

- $\text{Target} - \text{OldEstimate}$ 表示当前估计的“误差”(error)
- $\text{StepSize}$ 表示步长（在此例中是 $\frac{1}{n}$）

这种思想的核心在于一步一步走，而不是一下子跳到终点（目标值）。为什么这么做呢？因为我们不知道真实的目标值，只能通过观测得到一个大概率带有噪音的目标值，比如奖励可能是随机的、状态转移不确定、数据不完美等等，所以我们不能直接把估计值直接设定为目标值，而是要慢慢靠近她。这种渐进式学习可以比较灵活地适应环境变化，也不会受到单个异常值地剧烈影响。

在这里，书中对上一节我们自己写的老虎机算法进行了一个总结，这里的Q（a）就是Q值，N（a）是动作a被选择的次数。如果在上一小节认真写了代码，这个应该看起来会非常直观明了，因此就不赘述了：

![1748823420106](image/Ch2_Multi-armed_Bandits/1748823420106.png){style="display:block; margin:auto; width:800px;"}

## 2.5 跟踪一个非平稳问题

现在考虑这么一种情况：假如环境在缓慢地变化，那么我们就不得不考虑如何适应这种变化。用2.4的公式，显然不妥，因为无论是一开始的还是最近的，他们的值都会被平等地算进公式里去。

在RL中，其实更多的问题是非平稳（nonstationary）的。在这种情况下，我们应该想办法对最近的奖励赋予更大的权重。这里介绍一个非常常见的方法，即使用固定步长参数（step-size parameter)，即将2.4的公式：

$$
\begin{aligned}
Q_{n+1} = Q_n + \frac{1}{n} (R_n - Q_n)
\end{aligned}
$$

改为：

$$
Q_{n+1} \doteq Q_n + \alpha \left[ R_n - Q_n \right]
$$

这里的α叫做固定步长参数，是一个0到1的常数。显然易见，Q(n+1)就变成了对过去奖励和初始估计Q1的加权平均，并且通过数学公式推导：

$$
\begin{aligned}
Q_{n+1} &= Q_n + \alpha \left[ R_n - Q_n \right] \\
       &= \alpha R_n + (1 - \alpha) Q_n \\
       &= \alpha R_n + (1 - \alpha) \left[ \alpha R_{n-1} + (1 - \alpha) Q_{n-1} \right] \\
       &= \alpha R_n + (1 - \alpha)\alpha R_{n-1} + (1 - \alpha)^2 Q_{n-1} \\
       &= \alpha R_n + (1 - \alpha)\alpha R_{n-1} + (1 - \alpha)^2\alpha R_{n-2} + \cdots \\
       &\quad + (1 - \alpha)^{n-1} \alpha R_1 + (1 - \alpha)^n Q_1 \\
       &= (1 - \alpha)^n Q_1 + \sum_{i=1}^{n} \alpha (1 - \alpha)^{n - i} R_i
\end{aligned}
$$

可以看出，一开始（t=1）的奖励的权重将会随着时间的推移越来越小，并且一开始的奖励的权重是呈指数级衰减的（指数为1-α），因此这种方法也叫做指数加权平均（exponential recency-weighted average）。

根据随机逼近理论，若想让收敛必然发生，必须满足以下两个条件：

$$
\underbrace{\sum_{n=1}^{\infty} \alpha_n(a) = \infty}_{\text{步长足够大以克服初始条件或随机波动}} 
\quad \text{和} \quad 
\underbrace{\sum_{n=1}^{\infty} \alpha_n^2(a) < \infty}_{\text{步长最终变小以确保收敛}}
$$

那么可以看到，对于样本平均法（$\alpha_n(a) = \frac{1}{n}$）是满足的；但是对于固定步长的方法则不满足第二个条件，也即意味着Q值会永远随着最新奖励不断变化。虽然无法收敛，但是因为强化学习中非平稳问题占主流，所以该方法是一般被采用的。

本节练习见 Exercise 2.4 和 Exercise 2.5。

## 2.6 Optimistic Initial Values

周一继续学习这部分


## 附：Exercise 课本练习题

### Exercise 2.1

问：在练习ε-greedy动作选择中，假设有两个动作，且ε=0.5，那么选择贪婪动作的概率是多少？

答：如果epsilon=0.5，那么很显然有0.5的概率进行探索，并由1-0.5=0.5的概率选择贪婪动作。但是要注意，在随机探索的时候，也有可能选中贪婪动作，所以如果题目包含了这种情况，我们就需要

### Exercise 2.2

考虑一个k=4的多臂老虎机问题，考虑使用以下策略：

* 采用ε-greedy策略
* 使用样本平均法（sample-average）来估计动作
* 所有动作的初始动作值估计为$Q_1(a)=0$，对所有a都成立

假设在前几步中，采取的动作和获得的奖励分别为：

| Timestep | Action    | Reward     |
| -------- | --------- | ---------- |
| $t=1$  | $A_1=1$ | $R_1=-1$ |
| $t=2$  | $A_2=2$ | $R_2=1$  |
| $t=3$  | $A_3=2$ | $R_3=-2$ |
| $t=4$  | $A_4=2$ | $R_4=2$  |
| $t=5$  | $A_5=3$ | $R_5=0$  |

那么请问，在哪些时间步中可以确定发生了ε情况，而哪些时间步则是可能呢？

答：

| Timestep | 是否发生了ε | 原因                                                                 |
| -------- | ------------ | -------------------------------------------------------------------- |
| 1        | 有可能       | 第一步Q表都是0，这个时候ε与否并无区别                               |
| 2        | 有可能       | 此时最优值有4个选择，所以这步也都有可能                              |
| 3        | 有可能       | 这一步乍一眼看一定是greedy，但其实也有可能是ε的情况下随机到最优值的 |
| 4        | 一定         | 此时a2已经不是最优值了，选到的话一定是ε                             |
| 5        | 一定         | 此时a3的Q为0，但是a2的Q经过三次选择已经更新为0.33，大于a3            |

### Exercise 2.3

![1748812637285](image/Ch2_Multi-armed_Bandits/1748812637285.png){style="display:block; margin:auto; width:800px;"}

对于图中所示的对比实验中，哪一种方法在长期的表现最好？他会好多少？您能用定量的方式表达吗？

答：这道题有点意思，要求从数学角度分析。虽然图中明显ε=0.1的情况表现最好，但是从数学角度来看的话，如果进行无限时间步，即$t \to \infty$，那么样本平均估计在有探索的情况下将收敛到各个动作的真实值，即：

$$
Q_t(a) \to q_*(a)
$$

令动作中真实价值最大的那个老虎机臂为$a^* = \arg\max_a q^*(a)$，记他的真实价值为$q^*_{\max} \;=\; q^*(a^*) \;=\; \max_{1 \le a \le 10} q^*(a)$，那么在ε策略中，每一步：

* 以1-ε的概率选择$a^*$
* 以ε的概率在10个老虎机臂里随机选一个，选到$a^*$的概率是ε/10

因此当$t\to\infty$时，且 $Q_t$ 已经非常精确时，选择最优臂$a^*$的概率是：

$$
P_\infty(\text{选到最优臂 } a^*) 
= (1 - \varepsilon) + 
\underbrace{\frac{\varepsilon}{10}}_{\text{在“探索”时随机取到 } a^*} 
= 1 - \varepsilon + \frac{\varepsilon}{10}.
$$

换句话说：

$$
P_\infty(\text{optimal}) = 1 - \varepsilon + \frac{\varepsilon}{10}.
$$

同理，选到非最优臂的概率合计为：

$$
\sum_{a \ne a^*} P(\text{选到臂 } a) 
= 9 \times \frac{\varepsilon}{10} 
= \frac{9\varepsilon}{10}.
$$

因此，我们可以得到在epsilon为特定值的情况下，理论上长期的平均奖励：

$$
R_\infty = 
\underbrace{\left(1 - \varepsilon + \frac{\varepsilon}{10}\right)}_{P(\text{选到最优 } a^*)}
q^*_{\text{max}} 
+ \sum_{a \ne a^*} \left(\frac{\varepsilon}{10}\right) q^*(a).
$$

经过数学变换，得到可求解的长期平均奖励（这部分变化的过程参见**附录部分的Math 2.2**）：

$$
\mathbb{E}[R_\infty] 
= (1 - \varepsilon) \underbrace{\mathbb{E}[M_{10}]}_{\approx 1.54} 
+ \frac{\varepsilon}{10} \underbrace{\mathbb{E}\left[\sum_{a=1}^{10} q^*(a)\right]}_{= 0} 
= (1 - \varepsilon) \times 1.54.
$$

因此，我们可以得到理论上（定量方式）的长期（极限状态）选择最优臂的概率，和长期（极限状态）时的平均奖励：

| epsilon | $P_\infty$ | $\mathbb{E}[R_\infty]$ |
| ------- | ------------ | ------------------------ |
| 0       | 不适用*      | 不适用*                  |
| 0.01    | 0.991        | (1-0.1)*1.54             |
| 0.1     | 0.91         | (1-0.01)*1.54            |

*greedy情况下一开始可能会卡在次优臂上，理论上理想化的极限是100%

### Exercise 2.4

周一继续做

### Exercise 2.5

这个是个编程问题，周一继续做


## 附：Math 相关数学证明

### Math 2.1 证明多臂老虎机实验理论最优期望奖励

可以通过数学证明，多臂老虎机问题最终的最优理论值是1.54。

（注：这个证明是读者在看书的时候顺手查的资料边学边推的，如果有问题麻烦联系一下我，以防误导别人）

在我们的设置中，10个真实奖励符合标准正态分布，我们要找的最终值其实就是这10个独立的符合标准正态分布的随机变量的最大值的期望。由于实际奖励是围绕真实奖励以一个标准正态分布抖动的，所以噪声的均值为0，长期来看，其对应的平均奖励在无限长的长期下必然收敛到其真实值。

所以，最优理论值就是查找10个标准正态随机变量中最大值的期望。

对于k个独立分布的标准正态随机变量：

$$
X_1, X_2, \ldots, X_k \sim \mathcal{N}(0, 1)
$$

他们的最大值Mk为：

$$
M_k = \max\{X_1, X_2, \ldots, X_k\}
$$

在统计学中，$\{ X_{(1)}, X_{(2)}, \ldots, X_{(k)} \}$被称作这组样本的顺序统计量（Order Statistics），其中$X_{(1)} \le X_{(2)} \le \cdots \le X_{(k)}$，那么$X_{(k)} = M_k = \max\{X_1, \ldots, X_k\}$。

按照顺序统计量中最大值的分布与期望公式，首先求得累积分布函数CDF：

$$
F_{M_k}(x) = \Pr\{M_k \le x\} = \left[ \Phi(x) \right]^k, \quad \text{其中} \; \Phi(x) = \Pr\{X_i \le x\}
$$

然后对上式求导得到概率密度函数PDF：

$$
f_{M_k}(x) = \frac{d}{dx} F_{M_k}(x) = k \, \Phi(x)^{k-1} \, \phi(x), \quad \text{其中} \; \phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}
$$

得到最终所求的最大值的期望：

$$
\mathbb{E}[M_k] = \int_{-\infty}^{+\infty} x \, f_{M_k}(x) \, dx = \int_{-\infty}^{+\infty} k x \, \Phi(x)^{k-1} \, \phi(x) \, dx
$$

根据上面的公式，将k=10代入，得到：

$$
\mathbb{E}[M_{10}] = \int_{-\infty}^{+\infty} 10x \, \Phi(x)^9 \, \phi(x) \, dx \approx 1.54.
$$

即当k=10的时候，10个老虎机中最大值的数学期望是：

$$
\mathbb{E}\left[ \max_{1 \le i \le 10} q_i^* \right] \approx 1.54.
$$

### Math 2.2 推导多臂老虎机可求解的理论平均最优奖励

下面给出Exercize 2.3中的一个没有详细展开说明的推导过程。

在Exercize 2.3中，经过简单的推导，我们可以得到动作总数为10，最优臂为$a^*$的长期平均奖励：

$$
R_\infty = 
\underbrace{\left(1 - \varepsilon + \frac{\varepsilon}{10}\right)}_{P(\text{选到最优臂 } a^*)}
q^*_{\text{max}} 
+ \sum_{a \ne a^*} \left(\frac{\varepsilon}{10}\right) q^*(a) \tag{1}
$$

观察（1）中左半部分的系数，我们将它拆开，得到：

$$
\left(1 - \varepsilon + \frac{\varepsilon}{10}\right) q^*_{\text{max}}
= (1 - \varepsilon) q^*_{\text{max}} + \frac{\varepsilon}{10} q^*_{\text{max}}.
$$

将其代回（1），得到（2）：

$$
R_\infty =
\underbrace{(1 - \varepsilon)\, q^*_{\text{max}}}_{\text{(A)}}
+ \underbrace{\frac{\varepsilon}{10}\, q^*_{\text{max}}}_{\text{(B)}}
+ \underbrace{\sum_{a \ne a^*} \left(\frac{\varepsilon}{10}\right) q^*(a)}_{\text{(C)}}
\tag{2}
$$

其中：

* A是贪婪阶段直接选到最优臂那部分的期望
* B是探索阶段恰好选到最优臂的那部分的期望
* C是探索阶段选到任何一个非最优臂的期望

接着对（2）中的（B）和（C）进行合并，得到探索阶段所有10根臂的。这里需要用一点 技巧，首先写出“所有10根臂地价值之和”的分解：

$$
\sum_{a=1}^{10} q^*(a) 
= \underbrace{q^*(a^*)}_{=\, q^*_{\text{max}}} 
+ \sum_{a \ne a^*} q^*(a) \tag{I}
$$

然后在（I）两边同时乘以ε/10：

$$
\frac{\varepsilon}{10} \sum_{a=1}^{10} q^*(a)
= \frac{\varepsilon}{10} q^*_{\text{max}} + \frac{\varepsilon}{10} \sum_{a \ne a^*} q^*(a)
\tag{II}
$$

（II）等价于：

$$
\underbrace{\frac{\varepsilon}{10} q^*_{\text{max}}}_{\text{(B)}}
+ \underbrace{\frac{\varepsilon}{10} \sum_{a \ne a^*} q^*(a)}_{\text{(C)}}
= \frac{\varepsilon}{10} \sum_{a=1}^{10} q^*(a)
\tag{\text{III}}
$$

将（III）代回到（2）中，得到（3）：

$$
\begin{aligned}
R_\infty 
= (1 - \varepsilon)\, q^*_{\text{max}} + \frac{\varepsilon}{10} \sum_{a=1}^{10} q^*(a)
\end{aligned}
\tag{3}
$$

在式（3）中，左半部分的$q^*_{\text{max}}$就是在Math 2.1中求出来的$\mathbb{E}\left[ \max_{1 \le i \le 10} q_i^* \right]$，当k=10的时候：

$$
\mathbb{E}[M_{10}] 
= \mathbb{E} \left[ \max_{1 \le i \le 10} \mathcal{N}(0, 1) \right] 
\approx 1.5389.
$$

而右半部分，对于10个以标准正态独立且同分布（i.i.d., independent and identically distributed）进行设置的情况，他们的和的期望是0，即：

$$
对于 q^*(a) \sim \mathcal{N}(\mu = 0,\, \sigma^2 = 1)，
$$

$$
\mathbb{E} \left[ \sum_{a=1}^{10} q^*(a) \right]
= \sum_{a=1}^{10} \mathbb{E}[q^*(a)]
= \sum_{a=1}^{10} 0 = 0.
$$

因此，对于整个（3）式，在k=10并且以标准正态分布的环境下，其相当于：

$$
\mathbb{E}[R_\infty] 
= (1 - \varepsilon)\, \underbrace{\mathbb{E}[M_{10}]}_{\approx 1.54} 
+ \frac{\varepsilon}{10} \underbrace{\mathbb{E} \left[ \sum_{a=1}^{10} q^*(a) \right]}_{=\, 0} 
= (1 - \varepsilon) \times 1.54.
$$

## 附：Code 实验代码

### Code 2.1 多臂老虎机实验

```python
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

```

(end)
