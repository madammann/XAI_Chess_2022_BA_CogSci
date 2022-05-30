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
        '''ADD'''
        if self.visits == 0:
            return True
    
    def update(self, delta : list):
        '''ADD'''
        self.wins = (self.wins[0] + delta[0], self.wins[1] + delta[1])
        self.visits += 1
    
    def prove(self, case : int):
        '''ADD'''
        self.proven = True
        if case == 0: #proven win for white
            self.wins = (1,-1)
        elif case == 1: #proven win for black
            self.wins = (-1,1)
        elif case == 2: #proven draw
            self.wins = (0,0)
            
    def add_child(self, child : int):
        if type(self.children) == list:
            if child not in self.children:
                self.children += [child]
        else:
            self.children = [child]
    
    def update_reference(self, parent : int, children : list):
        self.parent = parent
        self.children = children
            
class MCTSTree:
    '''ADD'''
    def __init__(self, root_state : str, root_node : MctsNode, games=16):
        '''ADD'''
        self.tree = {0 : root_node}
        self.root_state = root_state
        self.game_pointer = [0 for game in range(games)]
    
    def __add__(self,tree):
        '''ADD'''
        pass
    
    def __len__(self):
        '''ADD'''
        pass
    
    def __str__(self):
        '''ADD'''
        pass
    
    def __contains__(self):
        '''ADD'''
        pass
    
    def __call__(self,obj, iter_lim=100, node_lim=10e8):
        '''
        Starts the tree policy in a specified mode given a set of parameters.
        :param obj: model.ChessModel or heuristics.EvaluationFunction or None.
        :param iter_lim (int): Number of iterations of the tree policy until terminating definitely.
        :param node_lim (int): Number of nodes a tree can contain after iteration without terminating.
        
        :example a: self(model.ChessModel(),iter_lim=50) -> Builds tree for 50 iterations or 10^8 nodes with model heuristic. 
        :example b: self(heuristics.EvaluationFunction(),node_lim=10e10) -> Builds tree for 100 iterations or 10^10 nodes with heuristic simulations.
        :example c: self(None) -> Builds tree for 100 iterations or 10^8 nodes with random simulations.
        '''
        
        iteration = 0
        
        if type(obj) == model.ChessModel:
            '''Define termination function based on game termination (t), iterations (i) and tree size (s).'''
            terminate = lambda t,i,s: any([all(t),i>=iter_lim,s>=node_lim])
            while not terminate([self.tree[ptr] for ptr in self.game_pointer],iteration,len(self.tree.keys())):
                pool = ThreadPool(len(self.game_pointer))
                self.tree_policy(pool)
#                 [pool.map(self.tree_policy, [])]
                iteration += 1
        
        if type(obj) == heuristics.EvaluationFunction:
            '''ADD'''
            terminate = lambda i,s: any([i>=iter_lim,s>=node_lim])
            while terminate(iteration,len(self.tree.keys())):
                iteration += 1
        
        if obj == None:
            '''Expand tree until iteration criterion is met.'''
            terminate = lambda i,s: any([i>=iter_lim,s>=node_lim])
            while terminate(iteration,len(self.tree.keys())):
                iteration += 1
    
    def sort_keys(self):
        '''Updates the key values of self.tree to be a clean natural number list from 0-len(keys).'''
        key2key = dict()
        
        '''Creates a key2key mapping'''
        for i, key in enumerate(self.tree.keys()):
            key2key[key] = i
        
        '''Iterate over all nodes and adjust their parents and children reference'''
        for node in self.values():
            node.update_reference(key2key[node.parent],[key2key[child] for child in node.children])
        
        '''Change dictionary keys to ordered natural number list'''
        self.tree = dict(zip(range(len(dic.keys())),dic.values()))
    
    def chainmove(self,node,chain=[]):
        '''
        Retrieves a list of move strings in order of play for reconstructing the game sequence.
        :param node (MctsNode): A MctsNode object for accessing node.parent.
        :param chain (list): A list of currently recorded moves in reverse order for recursive calls.
        :returns (list): A list of uci-encoded moves as strings in order of occurance.
        '''
        if node != 0:
            chain += [node.move]
            self.chainmove(node.parent,chain=chain) # recursive call of this function on parent
        else:
            return chain[::-1] # reverse for proper order
    
    def expand(self, node : int, move : str, idx : int):
        '''
        Creates a MctsNode with an index key.
        :param node (MctsNode): A MctsNode object.
        :param move (str): A chess move string in uci format.
        :param idx (int): An integer index which will be the key for the node value.
        '''
        self.tree[idx] = MctsNode(move,node)
    
    def select(self, node : int, color : int, c=2):
        '''
        Returns a ranking of children of the current node using the UCBT formula, given the player to move.
        :param node (int): The index of the node to select a child from.
        :param color (int): 0 for white and 1 for black, representing the color to move.
        :returns (numpy.ndarray): An array of indices in order of UCBT ranking score.
        
        :example a: self.select(17,0) -> np.array([2,1,0]) UCBT ranking for children of node 17 with white to move.
        :example b: self.select(10,1,c=1.8) -> np.array([0,3,4,2,1]) UCBT ranking for children of node 10 with black to move and exploration constant set to 1.8.
        '''
        values = np.array([[child.wins[color], child.visits] for child in self.tree[node].children]) #make array of visits and wins for children
        ranking = np.divide(values.T[0,:], values.T[1,:]) + c * np.sqrt(np.divide(2 * np.log(np.sum(values[:,1],axis=0)),values.T[1,:])) #UCBT in numpy on values
        return np.argsort(ranking)[::-1]
    
    def tree_policy(self, pool: ThreadPool, c=2) -> list:
        '''ADD'''
        nodes = [None]*len(self.game_pointer)
        while None in nodes:
            '''Gets all rankings relevant for the current iteration, meaning all unique nodes '''
            rankings = pool.starmap(self.select,[() for in set(self.game_pointer)])
    
    def simulate_model(self, node, T=8, epsilon=0.005):
        '''
        Simulates a chess game from a current state until termination using the ChessModel and returns the result.
        :param node (MctsNode): An MctsNode object for the starting state of the simulation.
        :param T (int): The number of last board states to be considered for network input, default is 8.
        :param epsilon (float): An epsilon value for random choice exploration during simulation, default 0.005.
        :returns (tuple): A tuple for updating the node of play with 1 for a win, 0 for a draw and -1 for a loss for both players.
        
        :example: self.simulate_model(self, node) -> (1,-1) white won and black lost
        '''
        chainmoves = self.chainmove(node)
        game = ChessGame(T=T, initial_moves=chainmoves) # creates a ChessGame instance for node position in tree
        
        while not game.terminal:
            '''Extract a next move, chosen by the network'''
            policy = tf.squeeze(self.model(game.get_input_tensor())[1],axis=0).numpy()
            next_move = policy_argmax_choice(policy, game.possible_moves, flipped=not game.board.turn) # game.board.turn returns True for White, meaning we do not need to flip, therefore "not" is included
            
            '''Include a small chance to chose a random move still'''
            if np.random.random() <= epsilon:
                next_move = np.random.choice(game.possible_moves)
            
            '''Select the move and update ChessGame accordingly'''
            game.select_move(next_move)
        
        return game.get_result()
    
    def update(self,node,delta):
        '''
        Updates a node with a delta result and propagates the result of to all its parents.
        :param node (MctsNode): A node object which is to be updated.
        :param delta (tuple): An update tuple to be applied to the node and its parents.
        '''
        if node != 0:
            self.tree[node].update(delta)
            self.update(self.tree[node].parent,delta)
        else:
            self.tree[node].update(delta)