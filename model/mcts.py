import chess
import numpy as np
import tensorflow as tf
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
                return f'Node({self.move},(inf,-inf),{self.visits})'
            elif self.wins[0] == -1:
                return f'Node({self.move},(-inf,inf),{self.visits})'
            else:
                return f'Node({self.move},{self.wins},{self.visits})'
        else:
            return f'Node({self.move},{self.wins},{self.visits})'
    
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
        
        if not self.__eq__(node):
            return True
        else:
            return False
    
    def __add__(self, node):
        '''
        ADD
        '''
        
        if self.__eq__(node):
            self.children = list(set(self.children + node.children))
            self.update(node.wins, node.visits)
            
            return self
        else:
            raise ValueError('Nodes are incompatible since they are not identical.')
    
    def update(self, delta : tuple, visits : int):
        '''
        ADD
        '''
        
        self.wins = (self.wins[0] + delta[0], self.wins[1] + delta[1])
        self.visits += visits
    
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
    
    def update_parent(self, parent : int):
        '''
        ADD
        '''
        
        self.parent = parent
    
    def update_children(self, children : list, overwrite=False):
        '''
        ADD
        '''
        
        if not overwrite:
            self.children += children
        else:
            self.children = children

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
    
    def __str__(self):
        '''
        ADD
        '''
        
        string = f'{str(self.tree[0])}('
        
        for index in self.tree[0].children:
            string += f'{str(self.tree[index])}('
            for d in self.tree[index].children:
                string += f'{self.tree[d]})'
        
        string += f')'
        
        return string
            
    
    def __contains__(self, node):
        '''
        ADD
        '''
        
        if type(node) != MctsNode or str:
            raise ValueError('Object can only contain MctsNode objects or move strings.')
        
        if type(node) == str:
            for val in self.tree.values():
                if node == val.move:
                    return True
            
            return False
        
        if type(node) == MctsNode:
            for val in self.tree.values():
                if node == val:
                    return True
            
            return False

    def merge_subtrees(self, subtrees : list, update=True):
        '''
        Merges subtrees into the parent tree (self.tree).
        :param subtrees (list): A list of subtree dictionaries.
        '''
        # step 1 : propagate all deltas upwards to make next steps frictionless
        if update:
            for subtree in subtrees:
                self.update(subtree[0].parent,subtree[0].wins,subtree[0].visits)

        # step 2.1 : find all pairs of indices of subtrees which have an identical root node
        identical_roots = []
        if len(subtrees) >= 2:
            for i, tree_a in enumerate(subtrees):
                for j, tree_b in enumerate(subtrees):
                    if i != j and not (j,i) in identical_roots:
                        if tree_a[0] == tree_b[0]:
                            identical_roots += [(i,j)]
        
        # test if any subtree roots are identical, if none are proceed to base case, else recursive call
        if len(identical_roots) > 0:
            # step 2.2 : create a final list of subtree index pairs which is non-repetitive
            # code credited to TemporalWolf from StackOverflow
            # https://stackoverflow.com/questions/42036188/merging-tuples-if-they-have-one-common-element
            iset = set([frozenset(s) for s in identical_roots])  # Convert to a set of sets
            identical_roots = []
            while(iset):                  # while there are sets left to process:
                nset = set(iset.pop())      # pop a new set
                check = len(iset)           # does iset contain more sets
                while check:                # until no more sets to check:
                    check = False
                    for s in iset.copy():       # for each other set:
                        if nset.intersection(s):  # if they intersect:
                            check = True            # must recheck previous sets
                            iset.remove(s)          # remove it from remaining sets
                            nset.update(s)          # add it to the current set
                identical_roots.append(tuple(nset))  # convert back to a list of tuples
            
            # step 3.1 : split subtrees into identical and non-identical parts
            identical = [val for tup in identical_roots for val in tup]
            non_identical = list(set([i for i in range(len(subtrees))]).difference(identical))
            
            identical = [subtrees[i] for i in range(len(subtrees)) if i in identical]
            non_identical = [subtrees[i] for i in range(len(subtrees)) if i in non_identical]
            
            # step 3.2 : merge non-identical subtrees onto tree if present
            keyrange_sum = np.sum([len(subtree) for subtree in non_identical])
            if keyrange_sum > 0:
                keys = np.add(range(keyrange_sum),len(self))
                pos = 0
            
                for subtree in non_identical:
                    for k,node in subtree.items():
                        self.tree[subtree[k].parent].update_children(children=[keys[pos]])
                        self.tree[keys[pos]] = node # write node to tree
                        self.tree[keys[pos]].update_children([],overwrite=True)
                        if k+1 < len(subtree):
                            subtree[k+1].update_parent(parent=keys[pos]) # upodate original subtree child parent index
                        pos += 1
            
            # step 3.3 : merge identical subtree roots and adjust the remaindure properly
            keys = np.add(range(len(identical_roots)),len(self)) # create the keys for writing to self.tree
            
            subtree_roots = {i : None for i in range(len(identical_roots))}
            for i, pair in enumerate(identical_roots):
                subtree_roots[i] = subtrees[pair[0]][0]
                for j in pair[1:]:
                    subtree_roots[i] = subtree_roots[i] + subtrees[j][0]
            
            for i,node in subtree_roots.items():
                node.update_children([],overwrite=True) # delete children list since it points to the wrong tree now
                self.tree[keys[i]] = node # write node to tree
                self.tree[node.parent].update_children([keys[i]]) # add children ref to parent of node
                
                # update the origin subtrees to point to self.tree parent
                for origin in identical_roots[i]:
                    if len(subtrees[origin]) > 1:
                        subtrees[origin][1].update_parent(keys[i]) # link parent ref to to-be roots
                
                for i in range(len(identical)):
                    # only if the subtree in question has more than one node (since the first node needs to be deleted)
                    if len(identical[i]) > 1:
                        identical[i] = dict(zip(list(range(len(identical[i])-1)),list(identical[i].values())[1:]))
                        for j, node in enumerate(identical[i].values()):
                            if j != 0:
                                node.update_parent(j-1) # ADD
                            if j < len(identical[i])-1:
                                node.update_children([j+1],overwrite=True) # ADD
                    else:
                        identical[i] = {}
                
                if {} in identical:
                    identical.remove({}) # remove empty dicts if there are any
            
            # step 4 : recursive call with only the previously identical subtrees
            self.merge_subtrees(identical,update=False)
            
        else:
            keys = np.add(range(np.sum([len(subtree) for subtree in subtrees])),len(self))
            pos = 0
            
            for subtree in subtrees:
                for k,node in subtree.items():
                    self.tree[subtree[k].parent].update_children(children=[keys[pos]])
                    self.tree[keys[pos]] = node # write node to tree
                    self.tree[keys[pos]].update_children([],overwrite=True)
                    if k+1 < len(subtree):
                        subtree[k+1].update_parent(parent=keys[pos]) # upodate original subtree child parent index
                    pos += 1
    
    def prune(self):
        '''
        ADD
        '''
        
        pass # ADD MORE HERE (add pruning method to prune non-parental branches on higher level than game pointers) 
    
    def bestmove(self, idx : int, epsilon = 0.2):
        '''
        ADD
        '''
        
        children = [self.tree[child_idx] for child_idx in self.tree[idx].children]
        wins = np.argsort([child.wins[self.root_state.board.turn] for child in children])[::-1]
        visits = np.argsort([child.visits for child in children])[::-1]
        
        # we chose the node with highest visit and win count for the player to play
        if wins[0] == visits[0]:
            return children[visits[0]].move

        # if there is no single node with both highest win and visit we chose the most visited node
        return children[visits[0]].move
    
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
        
    def __call__(self,obj, iter_lim=100, node_lim=10e7, gpi=100, T=8, c=2, epsilon=0.005, argmax=True):
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
        
        if node_lim >= 10e7:
            node_lim = 10e7
        if gpi >= 1000:
            gpi = 1000
        
        if type(obj) == ChessModel:
            # define termination function based on game termination (t), iterations (i) and tree size (s)
            terminate = lambda t,i,s: any([all(t),i>=iter_lim,s>=node_lim])
            
            while not terminate([self.tree[ptr].proven for ptr in self.game_pointer],iteration,len(self.tree.keys())):
                if not argmax:
                    choices = [self.game_pointer.count(ptr) for ptr in self.game_pointer] # sets choices for each game branch to the number of identical game branches
                else:
                    choices = [1 for ptr in self.game_pointer]
                args = [(ptr,{},model,c,choices[i],T,epsilon) for i, ptr in enumerate(self.game_pointer)]
                
                # we play gpi games before we select the next move for each game pointer
                for _ in range(gpi):
                    subtrees = ThreadPool(len(self.game_pointer)).starmap(self.build_model,args) # we execute in non-async since we need to await the result anyway for the next step
                    self.merge_subtrees(subtrees)
                
                # make the best move for each game according to tree and prune branches which will never be visited again
                self.game_pointer = [self.bestmove(ptr) for ptr in self.game_pointer]
                self.prune()
                
                # ADD MORE HERE (TBD)
                    
                iteration += 1
            
        else:
            terminate = lambda i,s: any([i>=iter_lim,s>=node_lim])
            
            while not terminate(iteration,len(self.tree.keys())):
                if not argmax:
                    choices = 2
                else:
                    choices = 1
                args = [(0,{},obj,c,choices,epsilon) for i in range(gpi)]
                
                # we play gpi games before we select the next move for each game pointer
                subtrees = ThreadPool(gpi).starmap(self.build_eval,args) # we execute in non-async since we need to await the result anyway for the next step
                self.merge_subtrees(subtrees)
                
                iteration += 1
                
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
            pass # ADD MORE HERE (case proven node, also return empty dict!)
        
        return subtree
    
    def build_eval(self, start_idx : int, subtree : dict, eval_func, c=2, choices=1, epsilon=0.01) -> dict:
        '''
        ADD
        '''
        
        game = deepcopy(self.root_state)
        
        self.tree_policy(game, start_idx, subtree, c=c, choices=choices)
        
        if not subtree[0].proven:
            self.simulate_eval(game, eval_func, subtree, epsilon=epsilon)
        else:
            pass # ADD MORE HERE (case proven node, also return empty dict!)
        
        return subtree
    
    def tree_policy(self, game : ChessGame, idx : int, subtree : dict, c=2, choices=1):
        '''
        ADD
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
                return np.random.choice(options,p=probs)
            else:
                return np.argmax(ranking), True
    
    def simulate_eval(self, game, eval_func, subtree : dict, epsilon=0.01):
        '''
        ADD
        '''
        
        move_count = 1
        
        if type(eval_func) == EvaluationFunction:
            # plays the simulation until a checkmate, stalemate, insufficient material, or rule based forced draw is encountered (see ChessGame documentation).
            while not game.terminal:
                # suggest the next move according to the evaluation function
                next_move = eval_func(game)
            
                # include a small chance to still chose a random move
                if np.random.random() <= epsilon or eval_func == None:
                    next_move = np.random.choice(game.possible_moves)
            
                # plays the selected move and expands the subtree
                game.select_move(next_move)
                subtree[move_count] = MctsNode(next_move,move_count-1)
                subtree[move_count-1].children += [move_count]
                move_count += 1
        else:
            # plays the simulation until a checkmate, stalemate, insufficient material, or rule based forced draw is encountered (see ChessGame documentation).
            while not game.terminal:
                # play a random legal move
                next_move = np.random.choice(game.possible_moves)
            
                # plays the selected move and expands the subtree
                game.select_move(next_move)
                subtree[move_count] = MctsNode(next_move,move_count-1)
                subtree[move_count-1].children += [move_count]
                move_count += 1
                
        delta = game.get_result()

        # updates the subtree with the result already
        for key in subtree.keys():
            subtree[key].update(delta, 1)
    
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
            subtree[key].update(delta, 1)
    
    def update(self, start_idx : int, delta : tuple, visits : int):
        '''
        Updates a node with a delta result and propagates the result to all its parents.
        :param node (int): The index of the node to update.
        :param delta (tuple): An update tuple to be applied to the node and its parents.
        '''
        
        index_chain = [start_idx]
        while not 0 in index_chain:
            index_chain += [self.tree[index_chain[-1]].parent]
        
        for index in index_chain:
            self.tree[index].update(delta, visits)