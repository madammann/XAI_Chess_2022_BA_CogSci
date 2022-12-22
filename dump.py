import chess
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from multiprocessing import Array
from copy import deepcopy

from lib.engine import *
from model.model import *
from model.heuristics import *


class MctsTree:
    def __init__(self, game):
        '''
        ADD
        '''
        
        self.tree = {0 : MctsNode(None,None)}
        self.root_state = deepcopy(game)
        self.game_results = None
    
    
        
    def __call__(self, model):
        games = []
        while not all([game.completed for game in games]):
            iterations = int(800 * len(games))
            
            for _ in range(iterations):
                pass
    
    
    
                
    def build_model(self, start_idx : int, subtree : dict, model, c=2, choices=1, epsilon=0.005) -> dict:
        '''
        ADD
        '''
        
        game = deepcopy(self.root_state) # creates a deepcopy of the tree's root state for further manipulation
        
        self.tree_policy(game, start_idx, subtree, c=c, choices=choices)
        
        # do a simulation only if the node from which to simulate is not a proven node
        if not subtree[0].proven:
            self.simulate_model(game, model, subtree, epsilon=epsilon)
        
        return subtree
    
    def tree_policy(self, game, idx : int, subtree : dict, c=2, choices=1):
        '''
        The tree policy method which is used to traverse the tree down to a promising leaf.
        Repeatedly calls the select method until it returns False for repeat.
        When this function terminates subtree will have one node inside from which to simulate.
        
        :param game (ChessGame): A game instance located at the step of idx.
        :param idx (int): The index in self.tree from where to start selection from.
        :param subtree (dict): The current subtree dictionary.
        :param c (float): The exploration constant passed down from the builer method.
        :param choices (int): If 1 UCBT will use argmax, if choices > 1 UCBT will be a weighted random choice of the best choices options.
        '''
        
        repeat = True
        node_idx = idx
        
        # until select returns False for whether it has reached a proven node or leaf, a new node is selected.
        while repeat:
            choice,repeat = self.select(game, node_idx, subtree, c=c, choices=choices)
            
            # if choice is not None we navigate to the next node_idx to work down from
            if choice != None:
                node_idx = self.tree[node_idx].children[choice]
                game.select_move(self.tree[node_idx].move) # plays the move of the selected child in the copied game instance
    
    def select(self, game, idx : int, subtree : dict, c=2, choices=1) -> tuple:
        '''
        ADD
        '''
        
        # tests whether node from which to select is a proven node in which case None is returned.
        if self.tree[idx].proven:
            subtree[0] = self.tree[idx]
            
            return None, False
        
        color = 0 if game.board.turn else 1 # 0 : white, 1 : black
        
        # tests whether an exhaustive proven node rule is applicable
        if all([self.tree[node].proven for node in self.tree[idx].children]) and self.tree[idx].children != []:
            best = np.max([node.wins[color] for node in self.tree[idx].children]) # gets the maximum achievable value for the color which is to move now
            
            if best == 1:
                self.tree[idx].prove(0 if color else 1) # result is proven win for white
                
            elif best == -1:
                self.tree[idx].prove(1 if color else 0) # result is proven win for black
                
            elif best == 0:
                self.tree[idx].prove(2) # result is draw
            
            subtree[0] = self.tree[idx]
            
            return None, False
        
        # tests whether not all children were created yet, if true it creates a random remaining node, else it uses UCBT ranking to chose one
        if len(game.possible_moves) > len(self.tree[idx].children):
            existing = set([self.tree[child_idx].move for child_idx in self.tree[idx].children])
            moves = set(game.possible_moves).difference(existing) # gets the remaining moves which do not exist in the tree yet
            move = np.random.choice(list(moves))
            
            subtree[0] = MctsNode(move,idx) # creates the corresponding subtree root
            
            return None,False
        
        else:
            values = np.array([[self.tree[child_idx].wins[color], self.tree[child_idx].visits] for child_idx in self.tree[idx].children]) # make array of visits and wins for children
            ranking = np.divide(values.T[0,:], values.T[1,:]) + c * np.sqrt(np.divide(2 * np.log(np.sum(values[:,1],axis=0)),values.T[1,:])) # UCBT formula in numpy
            
            # makes a weighted random choice allowed, else select the highest valued child
            if choices > 1:
                options = np.argsort(ranking)[::-1]
                options = options[:choices]
                probs = ranking[options]/np.sum(ranking[options])
                
                return np.random.choice(options,p=probs), True
            
            else:
                return np.argmax(ranking), True
    
    
    
    def simulate_model(self, game, model, subtree : dict, epsilon=0.005):
        '''
        ADD
        '''
        
        move_count = 1
        
        # plays the simulation until a checkmate, stalemate, insufficient material, or rule based forced draw is encountered (see ChessGame documentation).
        while not game.terminal:
            policy = model(game.get_input_tensor())[1].numpy()
            next_move = policy_argmax_choice(policy, game.possible_moves, flipped=not game.board.turn) # not game.board.turn since flipped should be true for black
            
            # include a small chance to still chose a random move
            if np.random.random() <= epsilon:
                next_move = np.random.choice(game.possible_moves)
            
            # plays the selected move and expands the subtree
            game.select_move(next_move)
            subtree[move_count] = MctsNode(next_move,move_count-1)
            subtree[move_count-1].children += [move_count]
            move_count += 1
        
        # get the result of the terminal state
        delta = game.get_result()
        
        # apply proven node rules based on result
        last = max(list(subtree.keys()))
        
        if delta == (1,0):
            subtree[last].prove(0) # make last node a proven win for white
            subtree[last-1].prove(1) # make the parent of the last node a proven loss for black
            
        elif delta == (0,1):
            subtree[last].prove(1) # make last node a proven win for black
            subtree[last-1].prove(0) # make the parent of the last node a proven loss for white
            
        else:
            subtree[last].prove(2) # make last node a proven draw
        
        # updates the subtree with the result already
        for key in subtree.keys():
            subtree[key].update(delta, 1)
    
    def update(self, start_idx : int, delta : tuple, visits : int):
        '''
        ADD
        '''
        
        index_chain = [start_idx]
        while not 0 in index_chain:
            index_chain += [self.tree[index_chain[-1]].parent]
        
        for index in index_chain:
            self.tree[index].update(delta, visits)
            
class MctsNode:
    def __init__(self, move : str, parent : int):
        
        # move information
        self.move = move
        
        # heuristic information
        self.wins = (0,0)
        self.visits = 0
        
        # proven information
        self.proven = False
        
        # parent information
        self.parent = parent
        
        # children information
        self.children = []
    
    def update(self, delta : tuple, visits : int):
        self.wins = (self.wins[0] + delta[0], self.wins[1] + delta[1])
        self.visits += visits
    
    def prove(self, case : int):
        self.proven = True
        
        if case == 0:
            self.wins = (1,0)
            
        elif case == 1:
            self.wins = (0,1)
            
        elif case == 2:
            self.wins = (0,0)
    
    def update_parent(self, parent : int):
        self.parent = parent
    
    def update_children(self, children : list, overwrite=False):
        if not overwrite:
            self.children += children
            
        else:
            self.children = children
            
