# AUTHOR: CLAUDE 4.0
# 该脚本仅作为进化算法vs强化学习的示例
# 并没有经过仔细推敲和设计
# 仅保留在这里以备未来使用

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import random
from collections import defaultdict
import time
from copy import deepcopy

# 井字棋游戏类
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        
    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        
    def get_valid_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]
    
    def make_move(self, row, col, player):
        if self.board[row, col] == 0:
            self.board[row, col] = player
            return True
        return False
    
    def check_winner(self):
        # 检查行
        for row in self.board:
            if row[0] == row[1] == row[2] != 0:
                return row[0]
        
        # 检查列
        for col in range(3):
            if self.board[0, col] == self.board[1, col] == self.board[2, col] != 0:
                return self.board[0, col]
        
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

# 进化算法玩家
class GeneticPlayer:
    def __init__(self, population_size=50):
        self.population_size = population_size
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.population = self._initialize_population()
        self.fitness = [0] * population_size
        self.generation = 0
        
    def _initialize_population(self):
        # 每个个体是一个策略：从棋盘状态到动作的映射
        population = []
        for _ in range(self.population_size):
            # 使用简化的策略表示：9个位置的权重
            individual = np.random.randn(9)
            population.append(individual)
        return population
    
    def get_move(self, game, player_id):
        # 使用最佳个体的策略
        best_idx = np.argmax(self.fitness)
        weights = self.population[best_idx]
        
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        
        # 计算每个有效位置的得分
        scores = []
        for row, col in valid_moves:
            idx = row * 3 + col
            score = weights[idx]
            
            # 添加一些启发式规则
            temp_game = deepcopy(game)
            temp_game.make_move(row, col, player_id)
            
            # 如果能赢，优先选择
            if temp_game.check_winner() == player_id:
                score += 100
            
            # 阻止对手获胜
            temp_game2 = deepcopy(game)
            opponent = 3 - player_id
            temp_game2.make_move(row, col, opponent)
            if temp_game2.check_winner() == opponent:
                score += 50
                
            scores.append(score)
        
        # 选择得分最高的动作
        best_move_idx = np.argmax(scores)
        return valid_moves[best_move_idx]
    
    def update_fitness(self, idx, result):
        # result: 1 for win, 0 for draw, -1 for loss
        self.fitness[idx] += result
    
    def evolve(self):
        # 选择、交叉和变异
        new_population = []
        
        # 精英保留
        elite_count = int(0.1 * self.population_size)
        elite_indices = np.argsort(self.fitness)[-elite_count:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        # 生成新个体
        while len(new_population) < self.population_size:
            # 锦标赛选择
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # 交叉
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # 变异
            if random.random() < self.mutation_rate:
                self._mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        self.fitness = [0] * self.population_size
        self.generation += 1
    
    def _tournament_selection(self, tournament_size=3):
        indices = random.sample(range(self.population_size), tournament_size)
        best_idx = max(indices, key=lambda i: self.fitness[i])
        return self.population[best_idx]
    
    def _crossover(self, parent1, parent2):
        child = parent1.copy()
        mask = np.random.rand(9) < 0.5
        child[mask] = parent2[mask]
        return child
    
    def _mutate(self, individual):
        for i in range(9):
            if random.random() < self.mutation_rate:
                individual[i] += np.random.randn() * 0.5

# Q-Learning玩家
class QLearningPlayer:
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.1):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.last_state = None
        self.last_action = None
        
    def get_move(self, game, player_id):
        state = game.get_state()
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            return None
        
        # ε-贪婪策略
        if random.random() < self.epsilon:
            action = random.choice(valid_moves)
        else:
            # 选择Q值最高的动作
            q_values = [(move, self.q_table[state][move]) for move in valid_moves]
            max_q = max(q_values, key=lambda x: x[1])[1]
            best_moves = [move for move, q in q_values if q == max_q]
            action = random.choice(best_moves)
        
        self.last_state = state
        self.last_action = action
        return action
    
    def update(self, reward, next_state=None):
        if self.last_state is None or self.last_action is None:
            return
        
        if next_state is None:
            # 终局
            self.q_table[self.last_state][self.last_action] += \
                self.alpha * (reward - self.q_table[self.last_state][self.last_action])
        else:
            # 非终局
            next_valid_moves = [(i, j) for i in range(3) for j in range(3) 
                              if next_state[i*3+j] == 0]
            if next_valid_moves:
                max_next_q = max([self.q_table[next_state][move] 
                                for move in next_valid_moves])
            else:
                max_next_q = 0
            
            self.q_table[self.last_state][self.last_action] += \
                self.alpha * (reward + self.gamma * max_next_q - 
                            self.q_table[self.last_state][self.last_action])
    
    def reset(self):
        self.last_state = None
        self.last_action = None

# 游戏控制器
class GameController:
    def __init__(self):
        self.game = TicTacToe()
        self.ga_player = GeneticPlayer()
        self.rl_player = QLearningPlayer()
        
        # 统计信息
        self.total_games = 0
        self.ga_wins = 0
        self.rl_wins = 0
        self.draws = 0
        self.history = []  # 保存所有游戏结果
        
        # 动画控制
        self.is_paused = False
        self.speed = 0.3
        self.current_moves = []
        self.move_index = 0
        self.displaying_game = 0
        self.is_training = True
        
        # 图形界面
        self.fig, (self.ax_board, self.ax_stats) = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.suptitle('TicTacToe: Genetic Algorithm (Red X) vs Q-Learning (Blue O)', fontsize=16)
        
        self.setup_board()
        self.setup_stats()
        self.setup_controls()
        
    def setup_board(self):
        self.ax_board.set_xlim(-0.5, 2.5)
        self.ax_board.set_ylim(-0.5, 2.5)
        self.ax_board.set_aspect('equal')
        self.ax_board.axis('off')
        
        # 画网格
        for i in range(4):
            self.ax_board.plot([i-0.5, i-0.5], [-0.5, 2.5], 'k-', lw=2)
            self.ax_board.plot([-0.5, 2.5], [i-0.5, i-0.5], 'k-', lw=2)
        
        # 添加进度信息文本
        self.progress_text = self.ax_board.text(1, -0.8, '', ha='center', va='top', 
                                               fontsize=12, bbox=dict(boxstyle='round', 
                                               facecolor='lightgray', alpha=0.8))
    
    def setup_stats(self):
        self.ax_stats.set_xlim(0, 100)
        self.ax_stats.set_ylim(0, 100)
        self.ax_stats.set_xlabel('Games')
        self.ax_stats.set_ylabel('Win Rate (%)')
        self.ax_stats.set_title('Win Rate Statistics (From Game 0)')
        self.ax_stats.grid(True, alpha=0.3)
        
        self.ga_line, = self.ax_stats.plot([], [], 'r-', label='Genetic Algorithm', lw=2)
        self.rl_line, = self.ax_stats.plot([], [], 'b-', label='Q-Learning', lw=2)
        self.draw_line, = self.ax_stats.plot([], [], 'g-', label='Draw', lw=2)
        self.ax_stats.legend()
        
        # 添加文本显示
        self.stats_text = self.ax_stats.text(0.02, 0.98, '', transform=self.ax_stats.transAxes,
                                            verticalalignment='top', fontsize=10,
                                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def setup_controls(self):
        # 暂停/继续按钮
        ax_pause = plt.axes([0.1, 0.02, 0.1, 0.04])
        self.btn_pause = Button(ax_pause, 'Pause')
        self.btn_pause.on_clicked(self.toggle_pause)
        
        # 速度滑块
        ax_speed = plt.axes([0.3, 0.02, 0.3, 0.04])
        self.slider_speed = Slider(ax_speed, 'Speed', 0.1, 2.0, valinit=0.3)
        self.slider_speed.on_changed(self.update_speed)
    
    def toggle_pause(self, event):
        self.is_paused = not self.is_paused
        self.btn_pause.label.set_text('Resume' if self.is_paused else 'Pause')
        plt.draw()
    
    def update_speed(self, val):
        self.speed = val
    
    def should_display(self):
        if self.total_games < 100:
            return self.total_games % 10 == 0
        elif self.total_games < 1000:
            return self.total_games % 100 == 0
        elif self.total_games < 10000:
            return self.total_games % 1000 == 0
        else:
            return self.total_games % 10000 == 0
    
    def play_game(self, display=False):
        self.game.reset()
        self.rl_player.reset()
        
        # 随机决定先手
        if random.random() < 0.5:
            players = [(1, self.ga_player, 'GA'), (2, self.rl_player, 'RL')]
        else:
            players = [(2, self.rl_player, 'RL'), (1, self.ga_player, 'GA')]
        
        moves = []
        current_player_idx = 0
        
        while True:
            player_id, player, player_name = players[current_player_idx]
            
            # 获取当前状态
            current_state = self.game.get_state()
            
            # 玩家走棋
            if player_name == 'GA':
                move = player.get_move(self.game, player_id)
            else:
                move = player.get_move(self.game, player_id)
            
            if move is None:
                break
            
            self.game.make_move(move[0], move[1], player_id)
            moves.append((move, player_id, player_name))
            
            # 检查游戏是否结束
            winner = self.game.check_winner()
            
            if winner is not None:
                # 游戏结束，更新统计
                if winner == 1:  # GA wins
                    self.ga_wins += 1
                    if players[0][2] == 'GA':
                        self.rl_player.update(-1)
                    else:
                        self.rl_player.update(-1, current_state)
                elif winner == 2:  # RL wins
                    self.rl_wins += 1
                    if players[0][2] == 'RL':
                        self.rl_player.update(1)
                    else:
                        self.rl_player.update(1, current_state)
                else:  # Draw
                    self.draws += 1
                    self.rl_player.update(0)
                break
            
            # 切换玩家
            current_player_idx = 1 - current_player_idx
            
            # 更新Q值（中间步骤）
            if player_name == 'RL':
                next_state = self.game.get_state()
                self.rl_player.update(0, next_state)
        
        self.total_games += 1
        
        # 更新进化算法的适应度
        if self.total_games % 100 == 0:
            # 每100局进化一次
            best_idx = np.argmax(self.ga_player.fitness)
            if winner == 1:
                self.ga_player.update_fitness(best_idx, 1)
            elif winner == 2:
                self.ga_player.update_fitness(best_idx, -1)
            else:
                self.ga_player.update_fitness(best_idx, 0)
            
            if self.total_games % 1000 == 0:
                self.ga_player.evolve()
        
        if display:
            self.current_moves = moves
            self.move_index = 0
            self.displaying_game = self.total_games
            self.is_training = False
        
        return winner
    
    def draw_board(self):
        # 清除之前的棋子
        for artist in self.ax_board.texts[1:] + self.ax_board.patches:  # 保留progress_text
            artist.remove()
        
        # 绘制当前棋盘
        for i in range(3):
            for j in range(3):
                if self.game.board[i, j] == 1:  # GA player
                    self.ax_board.text(j, 2-i, 'X', fontsize=40, color='red',
                                     ha='center', va='center', weight='bold')
                elif self.game.board[i, j] == 2:  # RL player
                    self.ax_board.text(j, 2-i, 'O', fontsize=40, color='blue',
                                     ha='center', va='center', weight='bold')
        
        # 显示当前玩家
        if self.move_index < len(self.current_moves):
            _, _, player_name = self.current_moves[self.move_index]
            color = 'red' if player_name == 'GA' else 'blue'
            self.ax_board.set_title(f'Current: {player_name} ({"Red X" if player_name == "GA" else "Blue O"})',
                                   color=color, fontsize=14)
        
        # 更新进度信息
        if self.is_training:
            self.progress_text.set_text(f'Training... Game #{self.total_games}')
        else:
            self.progress_text.set_text(f'Displaying Game #{self.displaying_game}')
    
    def update_stats(self):
        # 更新完整历史的胜率（从第0局开始）
        if self.total_games > 0:
            # 计算累积胜率
            ga_cumsum = 0
            rl_cumsum = 0
            draw_cumsum = 0
            
            ga_rates = []
            rl_rates = []
            draw_rates = []
            
            for i, result in enumerate(self.history):
                if result == 1:
                    ga_cumsum += 1
                elif result == 2:
                    rl_cumsum += 1
                else:
                    draw_cumsum += 1
                
                total = i + 1
                ga_rates.append(ga_cumsum / total * 100)
                rl_rates.append(rl_cumsum / total * 100)
                draw_rates.append(draw_cumsum / total * 100)
            
            x = list(range(len(self.history)))
            
            self.ga_line.set_data(x, ga_rates)
            self.rl_line.set_data(x, rl_rates)
            self.draw_line.set_data(x, draw_rates)
            
            self.ax_stats.set_xlim(0, max(1, len(self.history)))
            self.ax_stats.set_ylim(0, 100)
            
            # 更新x轴标签
            self.ax_stats.set_xlabel(f'Games (Total: {self.total_games})')
        
        # 更新文本统计
        stats_str = f'Total Games: {self.total_games}\n'
        stats_str += f'GA Wins: {self.ga_wins} ({self.ga_wins/max(1,self.total_games)*100:.1f}%)\n'
        stats_str += f'RL Wins: {self.rl_wins} ({self.rl_wins/max(1,self.total_games)*100:.1f}%)\n'
        stats_str += f'Draws: {self.draws} ({self.draws/max(1,self.total_games)*100:.1f}%)\n'
        stats_str += f'GA Generation: {self.ga_player.generation}'
        self.stats_text.set_text(stats_str)
    
    def animate(self, frame):
        if self.is_paused:
            return
        
        # 检查是否需要开始新游戏
        if self.move_index >= len(self.current_moves):
            # 如果刚显示完一局，标记为训练模式，但不清空棋盘
            if not self.is_training:
                self.is_training = True
                # 不再调用 self.game.reset() 和 self.draw_board()
            
            winner = self.play_game(display=self.should_display())
            self.history.append(winner)
            
            # 如果不需要显示，继续下一局
            if not self.should_display():
                # 更新进度显示
                self.progress_text.set_text(f'Training... Game #{self.total_games}')
                self.update_stats()
                return
            else:
                # 需要显示新对局时，才清空棋盘
                self.game.reset()
        
        # 播放当前游戏的动作
        if self.move_index < len(self.current_moves):
            move, player_id, _ = self.current_moves[self.move_index]
            self.game.board[move[0], move[1]] = player_id
            self.move_index += 1
            
            self.draw_board()
            self.update_stats()
            
            # 如果是最后一步，显示结果
            if self.move_index >= len(self.current_moves):
                winner = self.game.check_winner()
                if winner == 1:
                    result = "Genetic Algorithm Wins!"
                    color = 'red'
                elif winner == 2:
                    result = "Q-Learning Wins!"
                    color = 'blue'
                else:
                    result = "Draw!"
                    color = 'green'
                
                self.ax_board.set_title(result, color=color, fontsize=16, weight='bold')
    
    def run(self):
        # 设置动画间隔
        # 训练时用50ms，显示时通过speed控制
        def get_interval():
            if hasattr(self, 'is_training') and not self.is_training:
                return int(self.speed * 1000)  # 显示时使用speed设置
            return 50  # 训练时快速运行
        
        ani = FuncAnimation(self.fig, self.animate, frames=None, 
                          interval=50, repeat=True, blit=False)
        plt.show()
        
        # 显示最终结果
        print("\n" + "="*50)
        print("Final Statistics")
        print("="*50)
        print(f"Total Games: {self.total_games}")
        print(f"Genetic Algorithm Wins: {self.ga_wins} ({self.ga_wins/self.total_games*100:.2f}%)")
        print(f"Q-Learning Wins: {self.rl_wins} ({self.rl_wins/self.total_games*100:.2f}%)")
        print(f"Draws: {self.draws} ({self.draws/self.total_games*100:.2f}%)")
        print(f"Final GA Generation: {self.ga_player.generation}")

# 主程序
if __name__ == "__main__":
    controller = GameController()
    controller.run()