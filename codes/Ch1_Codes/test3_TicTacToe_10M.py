#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# AUTHOR: CLAUDE 4.0
# 该脚本仅作为进化算法vs强化学习的示例
# 并没有经过仔细推敲和设计
# 仅保留在这里以备未来使用
# AUTHOR: CLAUDE 4.0
# 该脚本仅作为进化算法vs强化学习的示例
# 并没有经过仔细推敲和设计
# 仅保留在这里以备未来使用
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 训练参数
TOTAL_GAMES = 10_000_000  # 1000万局
BATCH_SIZE = 1000  # 批量处理游戏数
PROGRESS_INTERVAL = 100_000  # 每10万局显示一次进度
SAVE_INTERVAL = 1_000_000  # 每100万局保存一次数据

# 优化的井字棋环境（向量化版本）
class VectorizedTicTacToe:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.boards = torch.zeros((batch_size, 3, 3), dtype=torch.float32, device=device)
        
    def reset(self):
        self.boards.zero_()
        return self.boards
    
    def get_valid_moves_mask(self):
        return (self.boards.view(self.batch_size, -1) == 0).float()
    
    def make_moves(self, actions, player):
        batch_indices = torch.arange(self.batch_size, device=device)
        row_indices = actions // 3
        col_indices = actions % 3
        self.boards[batch_indices, row_indices, col_indices] = player
    
    def check_winners_batch(self):
        winners = torch.zeros(self.batch_size, dtype=torch.long, device=device)
        
        # 检查行
        for i in range(3):
            row = self.boards[:, i, :]
            row_win = (row[:, 0] == row[:, 1]) & (row[:, 1] == row[:, 2]) & (row[:, 0] != 0)
            winners[row_win] = row[row_win, 0].long()
        
        # 检查列
        for j in range(3):
            col = self.boards[:, :, j]
            col_win = (col[:, 0] == col[:, 1]) & (col[:, 1] == col[:, 2]) & (col[:, 0] != 0)
            winners[col_win] = col[col_win, 0].long()
        
        # 检查对角线
        diag1 = (self.boards[:, 0, 0] == self.boards[:, 1, 1]) & \
                (self.boards[:, 1, 1] == self.boards[:, 2, 2]) & \
                (self.boards[:, 0, 0] != 0)
        winners[diag1] = self.boards[diag1, 0, 0].long()
        
        diag2 = (self.boards[:, 0, 2] == self.boards[:, 1, 1]) & \
                (self.boards[:, 1, 1] == self.boards[:, 2, 0]) & \
                (self.boards[:, 0, 2] != 0)
        winners[diag2] = self.boards[diag2, 0, 2].long()
        
        # 检查平局（棋盘满了但没有赢家）
        board_full = (self.boards.view(self.batch_size, -1) != 0).all(dim=1)
        winners[board_full & (winners == 0)] = -1  # -1表示平局
        
        return winners

# 神经网络Q-Learning玩家
class QNetwork(nn.Module):
    def __init__(self, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(9, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 9)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class NeuralQLearningPlayer:
    def __init__(self, learning_rate=0.001, gamma=0.9, epsilon=0.1):
        self.q_network = QNetwork().to(device)
        self.target_network = QNetwork().to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_target_every = 1000
        self.steps = 0
        
    def get_actions_batch(self, states, valid_moves_mask):
        batch_size = states.shape[0]
        
        # Epsilon-greedy
        random_mask = torch.rand(batch_size, device=device) < self.epsilon
        
        with torch.no_grad():
            q_values = self.q_network(states.view(batch_size, -1))
            q_values[valid_moves_mask == 0] = -float('inf')
            
        actions = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 随机动作
        if random_mask.any():
            random_actions = self._sample_valid_actions(valid_moves_mask[random_mask])
            actions[random_mask] = random_actions
        
        # 贪婪动作
        if (~random_mask).any():
            actions[~random_mask] = q_values[~random_mask].argmax(dim=1)
        
        return actions
    
    def _sample_valid_actions(self, valid_masks):
        batch_size = valid_masks.shape[0]
        actions = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            valid_indices = torch.where(valid_masks[i])[0]
            actions[i] = valid_indices[torch.randint(len(valid_indices), (1,), device=device)]
        
        return actions
    
    def update_batch(self, states, actions, rewards, next_states, next_valid_masks, dones):
        batch_size = states.shape[0]
        
        current_q_values = self.q_network(states.view(batch_size, -1))
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states.view(batch_size, -1))
            next_q_values[next_valid_masks == 0] = -float('inf')
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards + (1 - dones.float()) * self.gamma * max_next_q_values
        
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

# 优化的遗传算法玩家
class VectorizedGeneticPlayer:
    def __init__(self, population_size=100):
        self.population_size = population_size
        self.population = torch.randn(population_size, 9, device=device)
        self.fitness = torch.zeros(population_size, device=device)
        self.generation = 0
        self.mutation_rate = 0.1
        self.elite_ratio = 0.1
        
    def get_actions_batch(self, states, valid_moves_mask, batch_size):
        # 使用最佳个体
        best_idx = self.fitness.argmax()
        weights = self.population[best_idx]
        
        # 计算每个位置的得分
        scores = states.view(batch_size, -1) @ weights
        scores[valid_moves_mask == 0] = -float('inf')
        
        return scores.argmax(dim=1)
    
    def evolve(self):
        # 精英选择
        elite_count = int(self.population_size * self.elite_ratio)
        elite_indices = self.fitness.topk(elite_count)[1]
        
        new_population = torch.zeros_like(self.population)
        new_population[:elite_count] = self.population[elite_indices]
        
        # 生成新个体
        for i in range(elite_count, self.population_size):
            parent1_idx = elite_indices[torch.randint(elite_count, (1,), device=device)]
            parent2_idx = elite_indices[torch.randint(elite_count, (1,), device=device)]
            
            # 交叉
            mask = torch.rand(9, device=device) < 0.5
            new_population[i] = torch.where(mask, self.population[parent1_idx], self.population[parent2_idx])
            
            # 变异
            if torch.rand(1, device=device) < self.mutation_rate:
                mutation = torch.randn(9, device=device) * 0.5
                new_population[i] += mutation
        
        self.population = new_population
        self.fitness.zero_()
        self.generation += 1

# 批量训练函数
def train_agents_gpu():
    print(f"Starting GPU-optimized training for {TOTAL_GAMES:,} games...")
    print(f"Batch size: {BATCH_SIZE}")
    print("-" * 80)
    
    # 初始化
    env = VectorizedTicTacToe(BATCH_SIZE)
    rl_player = NeuralQLearningPlayer()
    ga_player = VectorizedGeneticPlayer()
    
    # 统计
    total_ga_wins = 0
    total_rl_wins = 0
    total_draws = 0
    
    history = {
        'games': [],
        'ga_win_rate': [],
        'rl_win_rate': [],
        'draw_rate': []
    }
    
    start_time = time.time()
    num_batches = TOTAL_GAMES // BATCH_SIZE
    
    with tqdm(total=num_batches, desc="Training Progress") as pbar:
        for batch_idx in range(num_batches):
            # 重置环境
            states = env.reset()
            
            # 随机决定先手（批量）
            first_players = torch.randint(1, 3, (BATCH_SIZE,), device=device)
            current_players = first_players.clone()
            
            game_done = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=device)
            
            # 存储轨迹用于学习
            trajectories = defaultdict(list)
            
            # 游戏循环
            for step in range(9):  # 最多9步
                valid_mask = env.get_valid_moves_mask()
                
                # GA玩家的动作
                ga_mask = (current_players == 1) & ~game_done
                if ga_mask.any():
                    ga_states = states[ga_mask]
                    ga_valid = valid_mask[ga_mask]
                    ga_actions = ga_player.get_actions_batch(ga_states, ga_valid, ga_mask.sum())
                    
                    # 执行动作
                    env.boards[ga_mask] = env.boards[ga_mask].clone()
                    ga_batch_indices = torch.where(ga_mask)[0]
                    for i, (batch_idx, action) in enumerate(zip(ga_batch_indices, ga_actions)):
                        row, col = action // 3, action % 3
                        env.boards[batch_idx, row, col] = 1
                
                # RL玩家的动作
                rl_mask = (current_players == 2) & ~game_done
                if rl_mask.any():
                    rl_states = states[rl_mask]
                    rl_valid = valid_mask[rl_mask]
                    rl_actions = rl_player.get_actions_batch(rl_states, rl_valid)
                    
                    # 保存轨迹
                    trajectories['states'].append(rl_states)
                    trajectories['actions'].append(rl_actions)
                    trajectories['masks'].append(rl_mask)
                    
                    # 执行动作
                    rl_batch_indices = torch.where(rl_mask)[0]
                    for i, (batch_idx, action) in enumerate(zip(rl_batch_indices, rl_actions)):
                        row, col = action // 3, action % 3
                        env.boards[batch_idx, row, col] = 2
                
                # 检查游戏结束
                winners = env.check_winners_batch()
                newly_done = (winners != 0) & ~game_done
                game_done = game_done | newly_done
                
                if game_done.all():
                    break
                
                # 切换玩家
                current_players = 3 - current_players
                current_players[game_done] = 0
            
            # 统计结果
            ga_wins = (winners == 1).sum().item()
            rl_wins = (winners == 2).sum().item()
            draws = (winners == -1).sum().item()
            
            total_ga_wins += ga_wins
            total_rl_wins += rl_wins
            total_draws += draws
            
            # 更新RL玩家
            if len(trajectories['states']) > 0:
                # 计算奖励
                rewards = torch.zeros(BATCH_SIZE, device=device)
                rewards[winners == 2] = 1.0  # RL赢
                rewards[winners == 1] = -1.0  # GA赢
                rewards[winners == -1] = 0.0  # 平局
                
                # 批量更新
                for i in range(len(trajectories['states']) - 1):
                    states_batch = trajectories['states'][i]
                    actions_batch = trajectories['actions'][i]
                    masks_batch = trajectories['masks'][i]
                    
                    next_states = trajectories['states'][i + 1] if i < len(trajectories['states']) - 1 else states_batch
                    next_valid = env.get_valid_moves_mask()[masks_batch]
                    
                    rewards_batch = rewards[masks_batch]
                    dones_batch = game_done[masks_batch]
                    
                    rl_player.update_batch(states_batch, actions_batch, rewards_batch, 
                                         next_states, next_valid, dones_batch)
            
            # 更新GA fitness
            if (batch_idx + 1) % 100 == 0:
                fitness_update = ga_wins - rl_wins
                ga_player.fitness[0] += fitness_update
                
                if (batch_idx + 1) % 1000 == 0:
                    ga_player.evolve()
            
            # 进度更新
            games_played = (batch_idx + 1) * BATCH_SIZE
            if games_played % PROGRESS_INTERVAL == 0:
                ga_rate = total_ga_wins / games_played * 100
                rl_rate = total_rl_wins / games_played * 100
                draw_rate = total_draws / games_played * 100
                
                history['games'].append(games_played)
                history['ga_win_rate'].append(ga_rate)
                history['rl_win_rate'].append(rl_rate)
                history['draw_rate'].append(draw_rate)
                
                elapsed = time.time() - start_time
                speed = games_played / elapsed
                
                tqdm.write(f"\nGames: {games_played:,} | GA: {ga_rate:.2f}% | RL: {rl_rate:.2f}% | Draw: {draw_rate:.2f}% | Speed: {speed:.0f} games/sec")
            
            pbar.update(1)
    
    # 最终结果
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total Games: {TOTAL_GAMES:,}")
    print(f"Total Time: {int(total_time//60)}m {int(total_time%60)}s")
    print(f"Average Speed: {TOTAL_GAMES/total_time:.1f} games/sec")
    print("-" * 80)
    print(f"Genetic Algorithm Wins: {total_ga_wins:,} ({total_ga_wins/TOTAL_GAMES*100:.2f}%)")
    print(f"Q-Learning (Neural) Wins: {total_rl_wins:,} ({total_rl_wins/TOTAL_GAMES*100:.2f}%)")
    print(f"Draws: {total_draws:,} ({total_draws/TOTAL_GAMES*100:.2f}%)")
    print("=" * 80)
    
    # 绘制结果
    plot_results_gpu(history, total_ga_wins, total_rl_wins, total_draws, TOTAL_GAMES)

def plot_results_gpu(history, ga_wins, rl_wins, draws, total_games):
    print("\nGenerating result plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 胜率变化曲线
    ax1.plot(history['games'], history['ga_win_rate'], 'r-', label='Genetic Algorithm', linewidth=2)
    ax1.plot(history['games'], history['rl_win_rate'], 'b-', label='Neural Q-Learning', linewidth=2)
    ax1.plot(history['games'], history['draw_rate'], 'g-', label='Draw', linewidth=2)
    
    ax1.set_xlabel('Games (millions)')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Win Rate Evolution During Training (10M Games)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # X轴显示为百万
    ax1.set_xticklabels([f'{int(x/1e6)}M' for x in ax1.get_xticks()])
    
    # 最终结果饼图
    labels = ['GA Wins', 'RL Wins', 'Draws']
    sizes = [ga_wins, rl_wins, draws]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%',
            shadow=True, startangle=90)
    ax2.set_title('Final Results Distribution (10M Games)')
    
    fig.suptitle('TicTacToe GPU Training: GA vs Neural Q-Learning', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("TicTacToe GPU-Optimized Training (10 Million Games)")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running on CPU (will be slower)")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit()
    
    try:
        train_agents_gpu()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()