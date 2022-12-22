import chess
from tqdm import tqdm
from operator import itemgetter
from node import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class RMC_Search_Tree:
    def __init__(self, board):
        self._root = Node(board.epd())
        self._current_pos = None
        for i in tqdm(range(100), desc='Creating initial tree'):
            node = self._root.select()
            node.simulate_and_update()
            
    def extend(self, value):
        self._current_pos = None
        for i in tqdm(range(value), desc='Expanding tree'):
            node = self._root.select()
            node.simulate_and_update()
    
    def balance(self):
        max_visits = max([child.visits for child in self._root.children])
        for i in tqdm(range(len(self._root.children)), desc='Move balancing progress'):
            while self._root.children[i].visits < max_visits:
                node = self._root.children[i].select()
                node.simulate_and_update()
    
    @property
    def root(self):
        return self._root
    
    def __str__(self):
        return (
            f'Root: Wins: {self._root.wins} Visits: {self._root.visits}\n'
        )
    
    @property
    def minmax_chain(self):
        chain = []
        board = chess.Board.from_epd(self._root.board)[0]
        color = board.turn
        pos = self._root
        print('###Current board from root node###')
        print(board)
        while not board.is_game_over() and not pos == None:
            moves_by_winratio = [(pos.children[i].move, pos.children[i].wins/pos.children[i].visits, pos.children[i]) for i in range(len(pos.children)) if pos.children[i].visits != 0]
            if not moves_by_winratio == []:
                if color == board.turn:
                    best_choice = max(moves_by_winratio,key=itemgetter(1))
                    pos = best_choice[2]
                    print('Selected',str(best_choice[0]),'as best move for player.')
                    board.push(chess.Move.from_uci(best_choice[0]))
                else:
                    worst_choice = min(moves_by_winratio,key=itemgetter(1))
                    pos = worst_choice[2]
                    print('Selected',str(worst_choice[0]), 'as best move for opponent.')
                    board.push(chess.Move.from_uci(worst_choice[0]))
            else:
                pos = None
        print('--------------------------------------------')
        print('###Final board state by Minmax chain###')
        return board
    
    @property
    def current_pos(self):
        if self._current_pos == None:
            return self._root
        else:
            return self._current_pos
        
    @property
    def move_graph(self):
        moves = [(child.move, child.wins/child.visits) for child in self._root.children if child.visits != 0]
        moves.sort(key=itemgetter(1),reverse=True)
        
        x = np.arange((len(moves)))
        width = 0.6
        
        fig, ax = plt.subplots(figsize=(20,10))
        bars = ax.bar(x,[move[1] for move in moves],width,edgecolor='black',color='red')
        ax.set_ylabel('Winratio%')
        ax.set_title('Move distribution by wins per visit')
        ax.set_xticks(x)
        ax.set_xticklabels([move[0] for move in moves])
        
        plt.show()
    
    def p_val_difference(self, move):
        pass
    
    @property
    def next_nodes(self):
        return [child.move for child in self.current_pos]
    
    def examine_next(self, move):
        self._current_pos = [child for child in self.current_pos if child.move == move]
        return self._current_pos