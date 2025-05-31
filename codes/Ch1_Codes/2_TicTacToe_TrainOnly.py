#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# AUTHOR: CLAUDE 4.0
# 该脚本仅作为进化算法vs强化学习的示例
# 并没有经过仔细推敲和设计
# 仅保留在这里以备未来使用

import numpy as np
import random
from collections import defaultdict
import time
import sys
import matplotlib.pyplot as plt

# 训练参数
TOTAL_GAMES = 10000000  # 总训练局数
PROGRESS_INTERVAL = 100000  # 每多少局显示一次进度

# 井字棋游戏类
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        
    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        
    def get_valid_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    moves.append((i, j))
        return moves
    
    def make_move(self, row, col, player):
        if self.board[row, col] == 0:
            self.board[row, col] = player
            return True
        return False
    
    def check_winner(self):
        # 检查行
        for i in range(3):
            if self.board[i, 0] == self.board[i, 1] == self.board[i, 2] != 0:
                return self.board[i, 0]
        
        # 检查列
        for j in range(3):
            if self.board[0, j] == self.board[1, j] == self.board[2, j] != 0:
                return self.board[0, j]
        
        # 检查对角线
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:
            return self.board[0, 0]
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] != 0:
            return self.board[0, 2]
        
        # 检查平局
        if len(self.get_valid_moves()) == 0:
            return 0
        
        return None
    
    def get_state(self):
        return tuple(self.board.flatten())
    
    def copy(self):
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        return new_game

# 进化算法玩家
class GeneticPlayer:
    def __init__(self, population_size=50):
        self.population_size = population_size
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.population = []
        self.fitness = [0] * population_size
        self.generation = 0
        
        # 初始化种群
        for _ in range(population_size):
            individual = np.random.randn(9)
            self.population.append(individual)
    
    def get_move(self, game, player_id):
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        
        # 使用最佳个体
        best_idx = 0
        best_fitness = self.fitness[0]
        for i in range(1, len(self.fitness)):
            if self.fitness[i] > best_fitness:
                best_fitness = self.fitness[i]
                best_idx = i
        
        weights = self.population[best_idx]
        
        # 评估每个有效动作
        best_score = -1000
        best_move = valid_moves[0]
        
        for row, col in valid_moves:
            idx = row * 3 + col
            score = weights[idx]
            
            # 检查是否能赢
            temp_game = game.copy()
            temp_game.make_move(row, col, player_id)
            if temp_game.check_winner() == player_id:
                score += 100
            
            # 检查是否需要阻止对手
            temp_game2 = game.copy()
            opponent = 3 - player_id
            temp_game2.make_move(row, col, opponent)
            if temp_game2.check_winner() == opponent:
                score += 50
            
            if score > best_score:
                best_score = score
                best_move = (row, col)
        
        return best_move
    
    def update_fitness(self, idx, result):
        self.fitness[idx] += result
    
    def evolve(self):
        new_population = []
        
        # 精英保留
        elite_count = max(1, int(0.1 * self.population_size))
        elite_indices = []
        for _ in range(elite_count):
            best_idx = 0
            best_fit = -1000
            for i in range(len(self.fitness)):
                if self.fitness[i] > best_fit and i not in elite_indices:
                    best_fit = self.fitness[i]
                    best_idx = i
            elite_indices.append(best_idx)
        
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        # 生成新个体
        while len(new_population) < self.population_size:
            # 选择父母
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # 交叉
            if random.random() < self.crossover_rate:
                child = parent1.copy()
                for i in range(9):
                    if random.random() < 0.5:
                        child[i] = parent2[i]
            else:
                child = parent1.copy()
            
            # 变异
            if random.random() < self.mutation_rate:
                for i in range(9):
                    if random.random() < self.mutation_rate:
                        child[i] += np.random.randn() * 0.5
            
            new_population.append(child)
        
        self.population = new_population
        self.fitness = [0] * self.population_size
        self.generation += 1
    
    def _tournament_selection(self, tournament_size=3):
        candidates = random.sample(range(self.population_size), min(tournament_size, self.population_size))
        best_idx = candidates[0]
        best_fit = self.fitness[candidates[0]]
        
        for idx in candidates[1:]:
            if self.fitness[idx] > best_fit:
                best_fit = self.fitness[idx]
                best_idx = idx
        
        return self.population[best_idx]

# Q-Learning玩家
class QLearningPlayer:
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.1):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.last_state = None
        self.last_action = None
    
    def get_move(self, game, player_id):
        state = game.get_state()
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            return None
        
        # Epsilon-greedy
        if random.random() < self.epsilon:
            action = random.choice(valid_moves)
        else:
            # 选择Q值最高的动作
            best_q = -1000
            best_moves = []
            
            for move in valid_moves:
                q_value = self.q_table[state][move]
                if q_value > best_q:
                    best_q = q_value
                    best_moves = [move]
                elif q_value == best_q:
                    best_moves.append(move)
            
            action = random.choice(best_moves)
        
        self.last_state = state
        self.last_action = action
        return action
    
    def update(self, reward, next_state=None):
        if self.last_state is None or self.last_action is None:
            return
        
        if next_state is None:
            # 终局
            old_q = self.q_table[self.last_state][self.last_action]
            self.q_table[self.last_state][self.last_action] = old_q + self.alpha * (reward - old_q)
        else:
            # 非终局
            next_valid_moves = []
            for i in range(3):
                for j in range(3):
                    if next_state[i*3+j] == 0:
                        next_valid_moves.append((i, j))
            
            if next_valid_moves:
                max_next_q = -1000
                for move in next_valid_moves:
                    q_val = self.q_table[next_state][move]
                    if q_val > max_next_q:
                        max_next_q = q_val
            else:
                max_next_q = 0
            
            old_q = self.q_table[self.last_state][self.last_action]
            self.q_table[self.last_state][self.last_action] = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
    
    def reset(self):
        self.last_state = None
        self.last_action = None

# 训练函数
def train_agents():
    print("Initializing game and players...")
    
    game = TicTacToe()
    ga_player = GeneticPlayer()
    rl_player = QLearningPlayer()
    
    # 统计信息
    ga_wins = 0
    rl_wins = 0
    draws = 0
    
    # 记录历史数据用于绘图
    history = {
        'games': [],
        'ga_win_rate': [],
        'rl_win_rate': [],
        'draw_rate': []
    }
    
    start_time = time.time()
    
    print("Starting training for {} games...".format(TOTAL_GAMES))
    print("Progress updates every {} games".format(PROGRESS_INTERVAL))
    print("-" * 80)
    
    for game_num in range(1, TOTAL_GAMES + 1):
        game.reset()
        rl_player.reset()
        
        # 随机决定先手
        if random.random() < 0.5:
            players = [(1, ga_player, 'GA'), (2, rl_player, 'RL')]
        else:
            players = [(2, rl_player, 'RL'), (1, ga_player, 'GA')]
        
        current_player_idx = 0
        
        while True:
            player_id, player, player_name = players[current_player_idx]
            current_state = game.get_state()
            
            move = player.get_move(game, player_id)
            if move is None:
                break
            
            game.make_move(move[0], move[1], player_id)
            
            winner = game.check_winner()
            if winner is not None:
                # 更新统计
                if winner == 1:
                    ga_wins += 1
                    if players[0][2] == 'GA':
                        rl_player.update(-1)
                    else:
                        rl_player.update(-1, current_state)
                elif winner == 2:
                    rl_wins += 1
                    if players[0][2] == 'RL':
                        rl_player.update(1)
                    else:
                        rl_player.update(1, current_state)
                else:
                    draws += 1
                    rl_player.update(0)
                break
            
            # 切换玩家
            current_player_idx = 1 - current_player_idx
            
            # 更新RL玩家
            if player_name == 'RL':
                next_state = game.get_state()
                rl_player.update(0, next_state)
        
        # 更新GA
        if game_num % 100 == 0:
            best_idx = 0
            best_fit = ga_player.fitness[0]
            for i in range(1, len(ga_player.fitness)):
                if ga_player.fitness[i] > best_fit:
                    best_fit = ga_player.fitness[i]
                    best_idx = i
            
            if winner == 1:
                ga_player.update_fitness(best_idx, 1)
            elif winner == 2:
                ga_player.update_fitness(best_idx, -1)
            else:
                ga_player.update_fitness(best_idx, 0)
            
            if game_num % 1000 == 0:
                ga_player.evolve()
        
        # 记录历史数据
        if game_num % PROGRESS_INTERVAL == 0:
            history['games'].append(game_num)
            history['ga_win_rate'].append(ga_wins / game_num * 100)
            history['rl_win_rate'].append(rl_wins / game_num * 100)
            history['draw_rate'].append(draws / game_num * 100)
            
            # 显示进度
            elapsed = time.time() - start_time
            speed = game_num / elapsed
            eta = (TOTAL_GAMES - game_num) / speed
            
            print("Game: {}/{} ({:.1f}%)".format(game_num, TOTAL_GAMES, game_num/TOTAL_GAMES*100))
            print("GA Wins: {} ({:.2f}%)".format(ga_wins, ga_wins/game_num*100))
            print("RL Wins: {} ({:.2f}%)".format(rl_wins, rl_wins/game_num*100))
            print("Draws: {} ({:.2f}%)".format(draws, draws/game_num*100))
            print("Speed: {:.1f} games/sec, ETA: {}m {}s".format(speed, int(eta//60), int(eta%60)))
            print("-" * 80)
    
    # 最终结果
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print("Total Games: {}".format(TOTAL_GAMES))
    print("Total Time: {}m {}s".format(int(total_time//60), int(total_time%60)))
    print("Average Speed: {:.1f} games/sec".format(TOTAL_GAMES/total_time))
    print("-" * 80)
    print("Genetic Algorithm Wins: {} ({:.2f}%)".format(ga_wins, ga_wins/TOTAL_GAMES*100))
    print("Q-Learning Wins: {} ({:.2f}%)".format(rl_wins, rl_wins/TOTAL_GAMES*100))
    print("Draws: {} ({:.2f}%)".format(draws, draws/TOTAL_GAMES*100))
    print("Final GA Generation: {}".format(ga_player.generation))
    print("=" * 80)
    
    if ga_wins > rl_wins:
        print("\nWINNER: Genetic Algorithm (by {} games)".format(ga_wins - rl_wins))
    elif rl_wins > ga_wins:
        print("\nWINNER: Q-Learning (by {} games)".format(rl_wins - ga_wins))
    else:
        print("\nIt's a TIE!")
    
    # 绘制结果图表
    plot_results(history, ga_wins, rl_wins, draws, TOTAL_GAMES)

def plot_results(history, ga_wins, rl_wins, draws, total_games):
    print("\nGenerating result plots...")
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 胜率变化曲线
    ax1.plot(history['games'], history['ga_win_rate'], 'r-', label='Genetic Algorithm', linewidth=2)
    ax1.plot(history['games'], history['rl_win_rate'], 'b-', label='Q-Learning', linewidth=2)
    ax1.plot(history['games'], history['draw_rate'], 'g-', label='Draw', linewidth=2)
    
    ax1.set_xlabel('Games')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Win Rate Evolution During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, total_games)
    ax1.set_ylim(0, 100)
    
    # 2. 最终结果饼图
    labels = ['GA Wins', 'RL Wins', 'Draws']
    sizes = [ga_wins, rl_wins, draws]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.1, 0.1, 0)  # 突出显示获胜较多的
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax2.set_title('Final Results Distribution')
    
    # 添加总标题
    fig.suptitle('TicTacToe Training Results: GA vs Q-Learning ({} games)'.format(total_games), 
                 fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    print("Plots displayed. Close the window to exit.")

# 主程序
if __name__ == "__main__":
    print("TicTacToe Training Program")
    print("=" * 80)
    try:
        train_agents()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print("\nError occurred: {}".format(e))
        import traceback
        traceback.print_exc()
    finally:
        print("\nProgram finished")