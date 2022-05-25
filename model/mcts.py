import chess
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from lib.engine import *
from model.model import *
from model.heuristics import *

class MctsNode:
    '''The node object for the MCTS tree.'''
    def __init__(self, move : str, parent : int):
        '''ADD'''
        self.move = move
        self.wins = (0,0)
        self.visits = 0
        self.proven = False
        self.parent = parent
        self.children = None
    
    def is_leaf(self) -> bool:
        if self.visits == 0:
            return True
    
    def update(self, delta : list):
        self.wins = (self.wins[0] + delta[0], self.wins[1] + delta[1])
        self.visits += 1
    
    def prove(self, rule : int):
        self.proven = True
        if rule == 0: #is terminal win for white
            self.wins = (1,-1)
        elif rule == 1: #is terminal win for black
            self.wins = (-1,1)
        elif rule == 2: #is terminal draw
            self.wins = (0,0)
        elif rule == 3: #one child is terminal win for white
            self.wins = (1,-1)
        elif rule == 4: #all children are terminal loss for black
            self.wins = (1,-1)
        elif rule == 5: #all children are terminal loss for white
            self.wins = (-1,1)
        elif rule == 6: #all children are draws # ADD MISSING RULE HERE
            self.wins = (0,0)
        elif rule == 7: #MOSTLY UNUSED SINCE NOT PROVEN STRICTLY: All children are confidently no better than draw
            self.wins = (0,0)
            
    def add_child(self, child : int):
        if type(self.children) == list:
            if child not in self.children:
                self.children += [child]
        else:
            self.children = [child]

            
            
class MCTSTree:
    '''ADD'''
    def __init__(self, root_state : str, games : int, model : ChessModel):
        '''ADD'''
        self.tree = {0 : root_state}
        self.game_pointer = [0 for game in range(games)]
    
    def chainmove(self,node,chain=[]):
        if node != 0:
            chain += [node.move]
            self.chainmove(node.parent,chain=chain) # recursive call of this function on parent
        else:
            return chain[::-1] # reverse for proper order
    
    def expand(self, node : int, move : str, idx : int):
        '''Creates a MctsNode at idx.'''
        self.tree[idx] = MctsNode(move,node)
    
    def select(self, node : int, color : int, c=2):
        '''Selects a child of the current node using the UCBT formula, given the player to move.'''
        values = np.array([[child.wins[color], child.visits] for child in self.tree[node].children]) #make array of visits and wins for children
        ranking = np.divide(values.T[0,:], values.T[1,:]) + c * np.sqrt(np.divide(2 * np.log(np.sum(values[:,1],axis=0)),values.T[1,:])) #UCBT in numpy on values
        return np.argmax(ranking)
    
    def simulate(self, node, T=8, epsilon=0.005):
        chainmoves = self.chainmove(node)
        game = ChessGame(T=T, initial_moves=chainmoves) # creates a ChessGame instance for node position in tree
        
        while not game.terminal:
            '''Extract a next move, chosen by the network'''
            policy = tf.squeeze(self.model(game.get_input_tensor())[1],axis=0).numpy()
            next_move = policy_argmax_choice(policy, game.possible_moves, flipped=not game.board.turn) # game.board.turn returns True for White, meaning we do not need to flip, therefore not
            
            '''Include a small chance to chose a random move still'''
            if np.random.random() <= epsilon:
                next_move = np.random.choice(game.possible_moves)
            
            '''Select the move and update ChessGame accordingly'''
            game.select_move(next_move)
        
        return game.get_result()
    
    def update(self,node,delta):
        if node != 0:
            self.tree[node].update(delta)
            self.update(self.tree[node].parent,delta)
        else:
            self.tree[node].update(delta)
    
    def tree_policy(self):
        pass