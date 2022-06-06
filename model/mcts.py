import chess
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from copy import deepcopy

from lib.engine import *
from model.model import *
from model.heuristics import *

class MctsNode:
    '''
    The node object for the MCTS tree.
    ADD MORE DESC
    '''
    def __init__(self, move, parent):
        '''
        ADD
        '''
        
        self.move = move
        self.wins = (0,0)
        self.visits = 0
        self.proven = False
        self.parent = parent
        self.children = []
    
    def __rep__(self):
        '''
        ADD
        '''
        
        return 'MctsNode'
    
    def __str__(self):
        '''
        ADD
        '''
        
        if self.proven:
            if self.wins[0] == 1:
                return f'Node({self.move},{(float('inf'),float('-inf'))})'
            elif self.wins[0] == -1:
                return f'Node({self.move},{(float('-inf'),float('inf'))})'
            else:
                return f'Node({self.move},{self.wins})'
        else:
            return f'Node({self.move},{self.wins})'
    
    def __eq__(self, node):
        '''
        ADD
        '''
        
        if self.move == node.move:
            if self.parent == node.parent:
                return True
            
            if any([child in node.children for child in self.children]):
                return True
        
        return False
    
    def __ne__(self, node):
        '''
        ADD
        '''
        
        if not self.__eq__(self, node):
            return True
        else:
            return False
    
    def update(self, delta : tuple):
        '''
        ADD
        '''
        
        self.wins = (self.wins[0] + delta[0], self.wins[1] + delta[1])
        self.visits += 1
    
    def prove(self, case : int):
        '''
        ADD
        '''
        
        self.proven = True
        
        if case == 0: # proven win for white
            self.wins = (1,-1)
        elif case == 1: # proven win for black
            self.wins = (-1,1)
        elif case == 2: # proven draw
            self.wins = (0,0)
    
    
    # TODO: Implement reference updating in merge op efficiently and rewrite these functions below
    
    def update_reference(self, parent : int, children : list):
        '''
        ADD
        '''
        
        self.parent = parent
        self.children = children
        
    def increment_reference(self, delta : int, parent=True, children=True):
        '''
        ADD
        '''
        
        if parent:
            self.parent = int(self.parent+delta)
        if children:
            self.children = [int(child+delta) for child in self.children]
            
# TODO1: Update docstrings again
# TODO2: Implement tqdm functionality for eval mode
# TODO3: ChessModel functionality with input tensor
# TODO4: ChessGame .terminal and .result adjustment for checkmate and special rules only

class MctsTree:
    '''
    ADD
    '''
    
    def __init__(self, game : ChessGame, games=16):
        '''
        Creates the root node and necessary attributes for Monte-Carlo-Tree-Search.
        :param game (engine.ChessGame): The chess game to act as root node for the search.
        :param games (int): The amount of parallel games to be played on the tree when training with a model.
        '''
        self.tree = {0 : MctsNode(None,None)}
        self.root_state = deepcopy(game)
        self.game_pointer = [0 for _ in range(games)]
    
    def __rep__(self):
        '''
        ADD
        '''
        
        return 'MctsTree'
    
    def __len__(self):
        '''
        Returns the length of the tree which is the number of nodes in it.
        '''
        
        return len(self.tree.values())
    
#     def __str__(self):
#         '''ADD'''
#         pass
    
#     def __contains__(self, obj):
#         '''ADD'''
#         pass

    def merge_subtrees(self, subtrees : list):
        '''
        Merges subtrees into the parent tree (self.tree).
        :param subtrees (list): A list of subtree dictionaries.
        '''

        pass # ADD MORE HERE (add proper merging routine, e.g. update references and integrate at idx with update step from subtree[0] upwards)
    
#         tree_lengths = [len(subtree) for subtree in subtrees[:-1]]
#         subtree_delta = [np.sum([tree_lengths[0:i]]) for i in range(len(subtrees))]
# 
#         # ADD
#         for i, subtree in enumerate(subtrees):
#             root_key = min(subtree.keys())
#             
#             # ADD
#             for key,val in subtree.items():
#                 self.tree[int(key+subtree_delta[i])] = val
#                 
#                 # ADD
#                 if key > root_key:
#                     self.tree[key+subtree_delta[i]].increment_reference(subtree_delta[i])
#                 else:
#                     self.tree[key+subtree_delta[i]].increment_reference(subtree_delta[i],parent=False)
    
    def prune(self):
        '''
        ADD
        '''
        pass # ADD MORE HERE (add pruning method to prune non-parental branches on higher level than game pointers) 
    
    def bestmove(self, idx : int):
        '''
        ADD
        '''
        pass # ADD MORE HERE (add formula choice for best move from index in self.tree)

    
# CURRENTLY UNUSED :below:

#     def sort_keys(self):
#         '''
#         Updates the key values of self.tree to be a clean natural number list from 0-len(keys).
#         This function should ideally only be called for visualization purposes only.
#         '''
#         key2key = dict()
        
#         '''Creates a key2key mapping'''
#         for i, key in enumerate(self.tree.keys()):
#             key2key[key] = i
        
#         '''Iterate over all nodes and adjust their parents and children reference'''
#         for node in self.values():
#             node.update_reference(key2key[node.parent],[key2key[child] for child in node.children])
        
#         '''Change dictionary keys to ordered natural number list'''
#         self.tree = dict(zip(range(len(dic.keys())),dic.values()))
    
#     def chainmove(self,node,chain=[]):
#         '''
#         Retrieves a list of move strings in order of play for reconstructing the game sequence.
#         :param node (MctsNode): A MctsNode object for accessing node.parent.
#         :param chain (list): A list of currently recorded moves in reverse order for recursive calls.
#         :returns (list): A list of uci-encoded moves as strings in order of occurance.
#         '''
        
#         if node != 0:
            
#             chain += [node.move]
#             self.chainmove(node.parent,chain=chain) # recursive call of this function on parent
            
#         else:
            
#             return chain[::-1] # reverse for proper order
        
    def __call__(self,obj, iter_lim=100, node_lim=10e8, gpi=800, T=8, c=2, epsilon=0.005, argmax=True):
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
        
        if type(obj) == type(ChessModel()): # TODO: Better typechecking, you can do better!
            
            # define termination function based on game termination (t), iterations (i) and tree size (s)
            terminate = lambda t,i,s: any([all(t),i>=iter_lim,s>=node_lim])
            
            while not terminate([self.tree[ptr].proven for ptr in self.game_pointer],iteration,len(self.tree.keys())):
                choices = [self.game_pointer.count(ptr) for ptr in self.game_pointer] # sets choices for each game branch to the number of identical game branches
                args = [(ptr,{},model,c,choices,T,epsilon) for i, ptr in enumerate(self.game_pointer)]
                
                # we play gpi games before we select the next move for each game pointer
                for _ in range(gpi):
                    subtrees = ThreadPool(len(self.game_pointer)).starmap(self.build_model,args) # we execute in non-async since we need to await the result anyway for the next step
                    self.merge_subtree(subtrees)
                
                # make the best move for each game according to tree and prune branches which will never be visited again
                self.game_pointer = [self.bestmove(ptr) for ptr in self.game_pointer]
                self.prune()
                
                # ADD MORE HERE (TBD)
                    
                iteration += 1
            
            # ADD MORE HERE (return values for training and do some cleanup)
            
        else:
            # ADD
            terminate = lambda i,s: any([i>=iter_lim,s>=node_lim])
            
            while terminate(iteration,len(self.tree.keys())):
                choices = [self.game_pointer.count(ptr) for ptr in self.game_pointer] # sets choices for each game branch to the number of identical game branches
                args = [(0,{},eval_func,c,choices,T,epsilon) for _ in range(gpi)]
                
                # we play gpi games before we select the next move for each game pointer
                subtrees = ThreadPool(gpi).starmap(self.build_eval,args) # we execute in non-async since we need to await the result anyway for the next step
                self.merge_subtree(subtrees)
                
                # ADD MORE HERE (TBD)
                
                iteration += 1
                
            # ADD MORE HERE (return best move and more and do some cleanup)
                
    def build_model(self, start_idx : int, subtree : dict, model : ChessModel, c=2, choices=1, T=8, epsilon=0.005) -> dict:
        '''
        The builder function which is executed by a thread pool to build the tree using the model.
        This function calles the tree_policy and default_policy routines.
        The returned subtree is composed of a new branch where the expand function created a new leaf and the simulation chain until a terminal leaf.
        ::
        :returns (dir): The subtree dictionary to fuse with the parent tree.
        '''
        
        game = deepcopy(self.root_state) # creates a deepcopy of the tree's root state for further manipulation.
        
        self.tree_policy(game, start_idx, subtree, c=c, choices=choices)
        
        if not subtree[0].proven:
            self.simulate_model(game, model, subtree, T=T, epsilon=epsilon)
        else:
            pass # ADD MORE HERE (case proven node)
        
        return subtree
    
    def build_eval(self, start_idx : int, subtree : dict, eval_func, c=2, choices=1, epsilon=0.01) -> dict:
        '''
        ADD
        '''
        
        game = deepcopy(self.root_state)
        
        self.tree_policy(game, start_idx, subtree, c=c, choices=choices)
        
        if not subtree[0].proven:
            self.simulate_eval(game, eval_func, subtree, T=T, epsilon=epsilon)
        else:
            pass # ADD MORE HERE (case proven node)
        
        return subtree
    
    def tree_policy(self, game : ChessGame, idx : int, subtree : dict, c=2, choices=1):
        '''
        ADD
        '''
        
        repeat = True
        node_idx = idx
        
        # until select returns False for whether it has reached a proven node or leaf, a new node is selected.
        while repeat:
            choice,repeat = self.select(game, node_idx, key, c=c, choices=choices)
            
            # if choice is not None we navigate to the next node_idx to work down from
            if choice != None:
                node_idx = self.tree[node_idx].children[choice]
                game.select_move(self.tree[node_idx].move) # plays the move of the selected child in the copied game instance

                
# EXPAND METHOD CURRENTLY UNUSED, may implement in merge op

#     def expand(self, parent_idx : int, move : str, key : int):
#         '''
#         Creates a MctsNode with an index key.
#         :param parent_idx (int): The index of the parent to be.
#         :param move (str): A chess move string in uci format.
#         :param key (int): An integer index which will be the key for the node value.
#         '''
#         self.tree[key] = MctsNode(move,parent_idx)
#         self.tree[parent_idx].children += [key]
    
    def select(self, game : ChessGame, idx : int, subtree : dict, c=2, choices=1) -> tuple:
        '''
        Selects the child of the parent to be moved to or selects the parent if it is a proven node.
        The select function will traverse the tree until it finds a suitable leaf node to do a simulation on.
        :param game (engine.ChessGame): The ChessGame instance to use for obtaining possible moves and player turn.
        :param idx (int): The index of the node to select a child from.
        :param subtree (dict): The subtree to write to if a new node is created, later merged into self.tree.
        :param c (float): The exploration constant to be used in the UCBT formula.
        :param choices (int): Enables a weighted random choice between the highest n ranked children instead of the maximum child.
        :returns (tuple): Returns a tuple consisting of an int or None and a boolean value.
        
        :example a: self.select(game,17) -> (2,True) 2 was the highest UCBT ranked child index of node 17.
        :example b: self.select(game,23,choices=2) -> (1,True) 1 was the weighted random choice of the two highest UCBT ranked child indices of node 23.
        :example c: self.select(game,28,c=1.8) -> (0) 0 was the highest UCBT ranked child index of node 28 with an exploration constant of 1.8 of node 28.
        :example d: self.select(game,72) -> (None,False) selection reached a proven node and simulation should start from subtree[0].
        :example e: self.select(game,3) -> (None,False) a new child node of 3 was created and added to subtree[0], simulate from there.
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
        if len(game.possible_moves) != len(self.tree[idx].children):
            existing = set([self.tree[child_idx].move for child_idx in self.tree[idx].children])
            moves = set(game.possible_moves).difference(existing) # gets the remaining moves which do not exist in the tree yet
            move = np.random.choice(moves)
            
            subtree[0] = MctsNode(move,None) # creates the corresponding subtree root
            return None,False
        else:
            values = np.array([[child.wins[color], child.visits] for child in self.tree[idx].children]) # make array of visits and wins for children
            ranking = np.divide(values.T[0,:], values.T[1,:]) + c * np.sqrt(np.divide(2 * np.log(np.sum(values[:,1],axis=0)),values.T[1,:])) # UCBT formula in numpy
            
            # makes a weighted random choice allowed, else select the highest valued child
            if choices > 1:
                options = np.argsort(ranking)[::-1]
                options = options[:choices]
                probs = ranking[options]/np.sum(ranking[options])
                return np.random.choice(options,p=probs)
            else:
                return np.argmax(ranking), True
    
    def simulate_eval(self, game, eval_func, subtree : dict, epsilon=0.01):
        '''
        ADD
        '''
        
        move_count = 1
        
        # plays the simulation until a checkmate, stalemate, insufficient material, or rule based forced draw is encountered (see ChessGame documentation).
        while not game.terminal:
            next_move = eval_func(game) if eval_func != None else None # TODO: Implement eval_func class
            
            # include a small chance to still chose a random move
            if np.random.random() <= epsilon or eval_func == None:
                next_move = np.random.choice(game.possible_moves)
            
            # plays the selected move and expands the subtree
            game.select_move(next_move)
            subtree[move_count] = MctsNode(next_move,move_count-1)
            subtree[move_count-1].children += [move_count]
            move_count += 1
        
        delta = game.get_result()
        
        # updates the subtree with the result already
        for key in subtree.keys():
            subtree[key].update(delta)
    
    def simulate_model(self, game, model, subtree : dict, T=8, epsilon=0.005):
        '''
        Simulates a chess game from a current state until termination using the ChessModel and returns the result.
        :param game (engine.ChessGame):
        :param simul_idx (int): The index of the node the simulation is run on as a subtree root node.
        :param T (int): The number of last board states to be considered for network input, default is 8.
        :param epsilon (float): An epsilon value for random choice exploration during simulation, default 0.005.
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
        
        delta = game.get_result()
        
        # updates the subtree with the result already
        for key in subtree.keys():
            subtree[key].update(delta)
    
    def update(self,idx,delta):
        '''
        Updates a node with a delta result and propagates the result to all its parents.
        :param node (int): The index of the node to update.
        :param delta (tuple): An update tuple to be applied to the node and its parents.
        '''
        
        if idx != 0:
            self.tree[idx].update(delta)
            self.update(self.tree[idx].parent,delta)
        else:
            self.tree[idx].update(delta)