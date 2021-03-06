import numpy as np
import tensorflow as tf
import chess

from multiprocessing.pool import ThreadPool
from copy import deepcopy

from lib.engine import *
from model.model import *
from model.heuristics import *

class MctsNode:
    '''
    The node object for the MCTS tree.
    Nodes have no access to other nodes and contain only their own information.
    Each node has a children and parent index for tree traversal.
    
    :attribute move (str): The move played on the edge between parent and self as uci-encoded string.
    :attribute wins (tuple): The wins for white and black on this node as tuple (white_wins,black_wins). 
    :attribute visits (int): The visits of this node, needed since wins does not include drawn games.
    :attribute proven (bool): A truth value for whether the node is ground truth proven.
    :attribute parent (int): The index pointing to the parent of this node.
    :attribute children (list): A list of indices pointing to the children of this node.
    '''
    
    def __init__(self, move : str, parent : int, data={}):
        '''
        Method for initializing value for a new MctsNode object.
        
        :param move (str): The move played to reach self, encded in uci format.
        :param parent (int): The index of the parent of this node.
        '''
        
        self.move = move
        self.wins = (0,0)
        self.visits = 0
        self.proven = False
        self.parent = parent
        self.children = []
        
        if len(data) > 0:
            if 'wins' in data.keys():
                self.wins = data['wins']
            
            if 'visits' in data.keys():
                self.visits = data['visits']
            
            if 'proven' in data.keys():
                self.proven = data['proven']
            
            if 'children' in data.keys():
                self.children = data['children']
    
    def __rep__(self) -> str:
        '''
        Method for representing the MctsNode object.
        
        :returns (str): 'MctsNode(move,parent,data=data)'.
        '''
        
        data = {'wins' : self.wins, 'visits' : self.visits, 'proven' : self.proven, 'children' : self.children}
        
        return f'MctsNode({self.move},{self.parent},data={repr(data)})'
    
    def __str__(self) -> str:
        '''
        Method for printing a representative string of the MctsNode object.
        
        :returns (str): A string containing Node(move,wins,visits) for this node.
        '''
        
        if self.proven:
            if self.wins[0] == 1:
                return f'Node({self.move},(inf,-inf),{self.visits})'
            
            elif self.wins[1] == 1:
                return f'Node({self.move},(-inf,inf),{self.visits})'
            
            else:
                return f'Node({self.move},{self.wins},{self.visits})'
        
        else:
            return f'Node({self.move},{self.wins},{self.visits})'
    
    def __eq__(self, node):
        '''
        Method for checking whether another MctsNode object is identical, assuming same tree membership.
        
        :param node (MctsNode): The MctsNode object to compare to.
        :returns (bool): True if node is identical, False if not.
        '''
        
        # node must store the same move
        if self.move == node.move:
            # node must point to same parent
            if self.parent == node.parent:
                return True
            
            elif (self.parent or node.parent) == None:
                # if one parent does not have a listed parent, then one identical child is is enough proof 
                if any([child in node.children for child in self.children]):
                    return True
            
            else:
                return False
            
        return False
    
    def __ne__(self, node):
        '''
        Method for checking whether another MctsNode object is not identical, assuming same tree membership.
        
        :param node (MctsNode): The MctsNode object to compare to.
        :returns (bool): False if node is identical, True if not.
        '''
        
        if not self.__eq__(node):
            return True
        
        else:
            return False
    
    def __add__(self, node):
        '''
        Method for adding two MctsNode objects together if they are identical.
        
        :param node (MctsNode): The node to add to self.
        :returns (MctsNode): The sum of the two nodes, a node with the summed wins, visits, and children.
        '''
        
        # if the node is identical proceed with addition
        if self.__eq__(node):
            self.children = list(set(self.children + node.children))
            self.update(node.wins, node.visits)
            
            return self
        
        else:
            raise ValueError('Nodes are incompatible since they are not identical.')
    
    def update(self, delta : tuple, visits : int):
        '''
        Method for updating the wins and visits of this node.
        
        :param delta (tuple): A tuple (delta_white, delta_black) to add to self.wins.
        :param visits (int): An integer value to add to visits.
        '''
        
        self.wins = (self.wins[0] + delta[0], self.wins[1] + delta[1])
        self.visits += visits
    
    def prove(self, case : int):
        '''
        Method for setting a node to proven for a given base case.
        The value for self.wins will be fixed to the specific case, white won, black won, or draw.
        
        :param case (int): An integer, either 0 (white), 1 (black), or 2 (draw), representing each possible case.
        '''
        
        self.proven = True
        
        if case == 0:
            self.wins = (1,0)
            
        elif case == 1:
            self.wins = (0,1)
            
        elif case == 2:
            self.wins = (0,0)
    
    def update_parent(self, parent : int):
        '''
        Method for changing the parent index of the node.
        
        :param parent (int): A new value for self.parent as integer index.
        '''
        
        self.parent = parent
    
    def update_children(self, children : list, overwrite=False):
        '''
        Method for adding to self.children or for overwriting it.
        
        :param children (list): A list of children indices to add or overwrite self.children with.
        :param overwrite (bool): If True set self.children to children, else add children to self.children.
        '''
        
        if not overwrite:
            self.children += children
            
        else:
            self.children = children

class MctsTree:
    '''
    The tree class for Monte-Carlo-Tree-Search.
    An instance of this class can be created by passing a ChessGame instance as initial state and a number of games to play simulataneously.
    Calling this instance with either a ChessModel, a EvaluationFunction or None instance will result in MCTS starting from the root with specified parameters.
    
    This implementation consists of four steps, selection, simulation, and merge/epdate.
    For n iterations, k "parallel" threads are created which all run selection and simulation from the root node.
    After awaiting the last thread all simulation results, returned as dictionaries of subtrees with update applied, are merged into the parent tree and updates are propagated.
    
    The MctsTree class offers three simulation modes for result generation:
    RANDOM : All moves played in the simulation steps are played randomly.
    EVALUATION-BASED : Moves played in the simulation steps are based on a evaluation function which returns a ranking of possible moves.
    MODEL-BASED: Moves are played according to a chess model's argmax over all moves possible.
    
    The MCTS search terminates if a node or iteration level is hit or, if in model-based simulation mode, all games played on the tree have terminated.
    When MCTS search terminates it will return the move of the best child of the root node as uci-encoded string.
    
    Additional functionalities include:
    Tree pruning, the removal of branches which are guaranteed not to be visited again in model-based execution.
    *Adaptive subtree length, a biased pruning of a subtree before merging to reduce nodes which are unlikely to be visited again.
    *Save & Load, an option to save and load MctsTree objects in a file.
    
    * to be or not to be implemented
    
    :att tree (dict): A dictionary of index and MctsNode pairs which represents the search tree.
    :att root_state (ChessGame): A copied ChessGame instance used as initial state for generating the tree.
    :att game_results: This attribute is either a list of node indices indicating the end state of all games during model-based execution or None.
    '''
    
    def __init__(self, game):
        '''
        Creates the root node and makes a deepcopy of the passed ChessGame instance for Monte-Carlo-Tree-Search.
        The root node has the index 0, None for node.parent and None for node.move.
        
        :param game (engine.ChessGame): The chess game to act as root node for the search.
        '''
        
        self.tree = {0 : MctsNode(None,None)}
        self.root_state = deepcopy(game)
        self.game_results = None
    
    def from_repr(game : str, tree : str):
        '''
        Method for creating a MctsTree object from a game and tree representation string.
        
        :returns (MctsTree): MctsTree based on game and tree repr string.
        '''
        
        tree = MctsTree(eval(game))
        tree.tree = eval(tree)
        
        return tree
    
    def __rep__(self):
        '''
        Method for representing the MctsTree object.
        
        :returns (str): 'MctsTree.from_repr(game,tree)'.
        '''
        
        game = repr(self.root_state)
        tree = repr(self.tree)
        
        return f'MctsTree.from_repr({game},{tree})'
    
    def __len__(self) -> int:
        '''
        Returns the length of the tree which is the number of values in the tree dictionary.
        
        :returns (int): Length of the MctsTree.
        '''
        
        return len(self.tree.values())
    
    def __str__(self) -> str:
        '''
        Method for printing a representative string of the MctsTree object.
        *Subject to future change.
        
        :returns (str): A string containing a tree-like representation of the first three levels of the MctsTree.
        '''
        
        string = f'{str(self.tree[0])}('
        
        for index in self.tree[0].children:
            string += f'{str(self.tree[index])}('
            for d in self.tree[index].children:
                string += f'{self.tree[d]})'
        
        string += f')'
        
        return string
            
    
    def __contains__(self, node) -> bool:
        '''
        Method for testing whether a specific move or node is in the tree.
        
        :param node: Either a uci-encoded string or a MctsNode object.
        :returns (bool): True if the node or move was found inside self.tree, False if not.
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
        
    def merge_branch(self, tree : dict, indices : list):
        '''
        Method for dictionary manipulation to merge specific nodes while keeping proper referencing.
        Created as utility for self.merge_subtrees, this method fuses node pairs correctly.
    
        :param tree (dict): A dictionary to manipulate, has index-node pairs.
        :param indices (list): A list of indices inside tree which are identical and should be merged
        '''
    
        # get the parent idx
        parental_idx = tree[indices[0]].parent
    
        # get a list of all children indices, assumes, as should be the case always, that there are no duplicates in the sum of all children
        children_idx = []
        for idx in indices:
            children_idx += tree[idx].children 
    
        # write merged node in first index position and set others to None (not pretty but functional to keep len(self) intact)
        summands = [tree[idx] for idx in indices]
        tree[indices[0]] = sum(summands[1:],summands[0])
        for idx in indices[1:]:
            tree[idx] = None
    
        # make parent point to correct child at idx indices[0]
        # if currently in the first layer of subtrees we may reference a parent in the parent tree which needs to be handled here
        if parental_idx in tree.keys():
            new_children = [child for child in tree[parental_idx].children if child not in indices[1:]] # remove all references to indices[1:] nodes set to None, assume rest was fine
            tree[parental_idx].update_children(new_children,overwrite=True)
            
        else:
            new_children = [child for child in self.tree[parental_idx].children if child not in indices[1:]] # remove all references to indices[1:] nodes set to None, assume rest was fine
            self.tree[parental_idx].update_children(new_children,overwrite=True)
    
        # make all children point to the correct parent at idx indices[0]
        for child_idx in children_idx:
            tree[child_idx].update_parent(indices[0])
        
    def merge_subtrees(self, subtrees : list):
        '''
        Method for merging a list of subtrees into the parent tree (self.tree).
        
        The parent tree is a dictionary with an integer index and MctsNode pair.
        Inside the parent tree all parent/child references are correct at all times.
        
        Each subtree is a dictionary of index, MctsNode pairs from 0-len(subtree).
        The parent reference for subtree[0].parent is always pointing to the parent tree index where to append.
        All other references are pointing to nodes inside the subtree.
        
        During merging the following tree steps are taken:
        Step 1 - Update: Propagate subtree[0].wins and subtree[0].visits up from self.tree[subtree[0].parent]
        Step 2 - Removal: Remove all subtrees in subtrees which only contain one proven node already present in the tree.
        Step 3 - Reference update: Updates all subtrees to have proper references for merging step.
        Step 4 - Merge preparation: Writes all subtrees into one tree dictionary.
        Step 5 - Identity merge: Intelligently checks all nodes which were added and finds and merges identical nodes with references.
        Step 6 - Final merge: Writes the subtree into the parent tree.
        
        :param subtrees (list): A list of subtree dictionaries.
        '''
        
        # step 1 : propagate all deltas upwards to make next steps frictionless
        for subtree in subtrees:
            self.update(subtree[0].parent,subtree[0].wins,subtree[0].visits)
        
        # step 2 : clean up subtrees since they may contain singular proven nodes which are already in the tree (see builder and select methods)
        remove_indices = []
        for i, subtree in enumerate(subtrees):
            if subtree[0].proven:
                # tests whether the subtree root is already in the tree by checking all children of the parent if one is identical
                if subtree[0].move in [self.tree[child].move for child in self.tree[subtree[0].parent].children]:
                    remove_indices += [i]
        
        for idx in remove_indices:
            del subtrees[idx]
        
        # step 3 : find indices to write to self.tree and also update references of children and parents properly.
        subtree_lengths = [len(subtree) for subtree in subtrees]
        summed_subtree_lengths = [int(np.sum([subtree_lengths[0:i]])) for i in range(1,len(subtrees))]
        delimiter_indices = [len(self)] + [len(self) + length for length in summed_subtree_lengths] # a list of all starting indices for each subtree in the merged tree
        
        # loop over all subtrees and initially only update parent and child references
        for tree_idx, subtree in enumerate(subtrees):
            for item_idx, node in subtrees[i].items():
                # if the node is not the root node we overwrite parent and children with proper indices
                if not item_idx == 0:
                    node.update_parent(node.parent + delimiter_indices[tree_idx])
                    new_children = [child + delimiter_indices[tree_idx] for child in node.children] # is empty if no children
                    node.update_children(new_children, overwrite=True)
                
                # if the node was the root node we only overwrite children
                else:
                    new_children = [child + delimiter_indices[tree_idx] for child in node.children] # is empty if no children
                    node.update_children(new_children, overwrite=True)
        
        # adjust all indices properly
        for i, subtree in enumerate(subtrees):
            subtrees[i] = dict(zip(range(delimiter_indices[i],delimiter_indices[i]+subtree_lengths[i]),subtree.values()))
        
        # step 4 : merge subtrees into a single subtree
        tree = {}
        for subtree in subtrees:
            tree.update(subtree)
            
        # step 5 : find all nodes in subtree which are identical and merge them with correct referencing
        # since all subtrees can only have identical nodes at the same depth we utilize this for a smarter comparison
        
        # store indices to look at smartly to look only at nodes at indices
        layer_indices = delimiter_indices
        
        # store diverged branches, potential branches at current merge iteration and pairs to merge in here
        potential_merges = [node for node in [tree[idx] for idx in layer_indices]]
        diverged = set()
        identical_pairs = []
        
        # step 5.1 : check all potential identical pairs and store them as tuples
        for i, node_a in enumerate(potential_merges):
            for j, node_b in enumerate(potential_merges):
                # compare smartly to not check same pair in reverse and also not identical indices
                if i != j and not j < i:
                    if node_a == node_b:
                        identical_pairs += [(i, j)]
        
        # step 5.2 : fuse all tuples with at least one element identical to merge identical nodes present more than twice properly
        # code credited to TemporalWolf from StackOverflow
        # https://stackoverflow.com/questions/42036188/merging-tuples-if-they-have-one-common-element
        iset = set([frozenset(s) for s in identical_pairs])  # Convert to a set of sets
        identical_pairs = []
        
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
                        
            identical_pairs.append(tuple(nset))  # convert back to a list of tuples
        
        # step 5.3 : store divergent branch delimiter indices to skip in next iteration
        for idx in range(len(subtrees)):
            # if the index is not already confirmed divergent we test if it is in identical pairs
            if not len(list(filter(lambda i_pairs: idx in i_pairs, identical_pairs))) > 0 or idx in diverged:
                diverged.add(idx)
                
        # step 5.4 : loop with incremented layer until all are divergent branches or no more branches are left
        # until there was no identical pair found anymore merge identical pairs
        while len(diverged) < len(subtrees):
            # we merge all pairs immediately with the proper function
            for pair in identical_pairs:
                print('to be merged:', [tree[layer_indices[idx]].children for idx in pair])
                self.merge_branch(tree, [layer_indices[val] for val in pair])
            
            # we need to recalculate delimiters for the next comparison step
            layer_indices = [idx+1 for idx in layer_indices]
            
            # additionally we filter out as divergent all positions where layer_indices[i] - delimiter_indices[i] > subtree_lengths[i] since the subtree i would then have no nodes left
            for i in range(len(layer_indices)):
                if layer_indices[i] - delimiter_indices[i] > subtree_lengths[i]:
                    diverged.add(i)
            
            # get all nodes from the current layer of non-diverged branches to check
            potential_merges = [tree[layer_indices[idx]] for idx in range(len(subtrees)) if idx not in diverged]
            
            # repeat steps 5.1 to 5.3
            identical_pairs = []
            
            for i, node_a in enumerate(potential_merges):
                for j, node_b in enumerate(potential_merges):
                    if i != j and not j < i:
                        if node_a == node_b:
                            identical_pairs += [(i, j)]
            
            iset = set([frozenset(s) for s in identical_pairs])
            identical_pairs = []
        
            while(iset):
                nset = set(iset.pop())
                check = len(iset)
            
                while check:
                    check = False
                
                    for s in iset.copy():
                        if nset.intersection(s):
                            check = True
                            iset.remove(s)
                            nset.update(s)
                        
                identical_pairs.append(tuple(nset))
            
            for idx in range(len(subtrees)):
                if not len(list(filter(lambda i_pairs: idx in i_pairs, identical_pairs))) > 0 or idx in diverged:
                    diverged.add(idx)
            
        # step 6 : merge the tree into self.tree properly
        self.tree.update(tree)
    
    def bestmove(self, idx : int) -> str:
        '''
        Method for obtaining the best children move of a node according to a win/visit ranking.
        A node is treated as the best node if it either has highest wins and visits out of all children.
        If no node has both highest wins and visits, the node with highest visit count is selected instead.
        
        :param idx (int): The index of the node from which the best child move should be selected from.
        '''
        
        children = [self.tree[child_idx] for child_idx in self.tree[idx].children]
        wins = np.argsort([child.wins[self.root_state.board.turn] for child in children])[::-1]
        visits = np.argsort([child.visits for child in children])[::-1]
        
        # we chose the node with highest visit and win count for the player to play
        if wins[0] == visits[0]:
            return children[visits[0]].move

        # if there is no single node with both highest win and visit we chose the most visited node
        return children[visits[0]].move
    
    def move_chain(self, idx, chain=[]):
        '''
        Retrieves a list of uci-encoded move strings in order of play for reconstructing the game sequence.
        
        :param idx (int): The integer index of the node from where to start backpropagating moves.
        :param chain (list): A list used for recursive calls, contains current move list in reversed order.
        :returns (list): A list of uci-encoded moves as strings in order of occurance.
        '''
        
        # recursive case if not yet at root
        if idx != 0:
            chain += [self.tree[idx].move]
            self.move_chain(self.tree[idx].parent,chain=chain)
            
        else:
            # base case if "idx == 0"
            return chain[::-1] # reverse for proper order
    
    def node_chain(self, idx, chain=[]):
        '''
        Retrieves a list of node indices for reconstructing the game sequence.
        
        :param idx (int): The integer index of the node from where to start backpropagating nodes.
        :param chain (list): A list used for recursive calls, contains current move list in reversed order.
        :returns (list): A list of node reference indices in order of occurance.
        '''
        
        # recursive case if not yet at root
        if idx != 0:
            chain += [idx]
            self.node_chain(self.tree[idx].parent,chain=chain)
            
        else:
            # base case if "idx == 0"
            chain += [0]
            return chain[::-1] # reverse for proper order
        
    def __call__(self, obj, iter_lim=100, games=32, node_lim=5e8, c=2, argmax=True, epsilon=0.005):
        '''
        Method for building the MCTS search tree for a specified duration.
        If an instance of this class is called the tree will be built and stored in self.tree and a best move from root will be returned afterwards.
        For a more detailed explanation on hyperparameters and termination criteria view the comments inside the method.
        
        This method can be called in three different modes:
        RANDOM by passing None as obj.
        EVALUATION-BASED by passing a mode.heuristics.EvaluationFunction as obj.
        MODEL-BASED by passing a model.model.ChessModel instance as obj.
        Each mode has a different simulation and termination behavior.
        
        :param obj: model.model.ChessModel or model.heuristics.EvaluationFunction or None.        
        :param iter_lim (int): Number of iterations of the tree policy until terminating definitely.
        :param games (int): Number of game threads to be run in "parallel" during each iteration.
        :param node_lim (int): Number of nodes a tree can contain after iteration without terminating.
        :param c (float): Exploration constant for UCBT formula.
        :param argmax (bool): True if select should always return highest UCBT-ranked node, False if a random weighted choice should be made.
        :param epsilon (float): Hyperparameter representing the chance of selecting a random move instead during simulation.
        
        Input parameters are not checked or clipped, they need to be correctly inputted!
        
        Example usage with "preferred" settings:
        mcts_tree(None, iter_lim=1000, games=64) -> Does random simulations with, ideally, 64 threads per iteration up to 1000 iterations.
        mcts_tree(EvaluationFunction, iter_lim=500, games=32, epsilon=0.01) -> Uses an evaluation function with, ideally, 32 threads per iteraton and slightly higher epsilon for up to 500 iterations.
        mcts_tree(ChessModel, games=32, argmax=False) -> Uses the ChessModel to simulate 32 games in, ideally, separate threads and does allow higher diverging path probability by setting argmax as False.
        '''
        
        iteration = 0
        
        if type(obj) == ChessModel:
            # definition of termination criterion as having completed all games, reaching the iteration limit, or reaching the node limit
            terminate = lambda t,i,s: any([all(t),i>=iter_lim,s>=node_lim])
            
            # create a game pointer for each game which points to the root node
            game_pointer = [0 for _ in range(games)]
            
            while not terminate([self.tree[ptr].proven for ptr in game_pointer],iteration,len(self.tree.keys())):
                # if argmax is not chosen we define for each pointer the amount of nodes to consider for weighted random choice in UCBT for higher game divergence probability 
                if not argmax:
                    choices = [game_pointer.count(ptr) for ptr in game_pointer] # sets choices for each game branch to the number of identical game branches
                    
                else:
                    choices = [1 for ptr in game_pointer]
                    
                # we define arguments for each thread and also only include the non-terminal games
                args = [(ptr,{},model,c,choices[i],epsilon) for i, ptr in enumerate(game_pointer) if not self.tree[ptr].proven]
                
                # we play 800 games before we select the next move for each game pointer, all 800 games per batch are played on the same tree
                for _ in range(800):
                    subtrees = ThreadPool(len(args)).starmap(self.build_model,args) # we execute in non-async since we need to await the result anyway for the next step
                    self.merge_subtrees(subtrees)
                
                # make the best move for each game according to tree and prune branches which will never be visited again
                game_pointer = [self.bestmove(ptr) for ptr in game_pointer]
                    
                iteration += 1 # in this context an iteration is one move in each non-terminal game
            
            # finally store the game pointer in game_results for model training
            self.game_results = game_pointer
            
        else:
            # definition of termination criterion as reaching the iteration limit or the node limit
            terminate = lambda i,s: any([i>=iter_lim,s>=node_lim])
            
            while not terminate(iteration,len(self)):
                if not argmax:
                    choices = 2
                    
                else:
                    choices = 1
                    
                # in each iteration we do selection, expansion, simulation for games number of threads before we merge
                args = [(0,{},obj,c,choices,epsilon) for i in range(games)]
                subtrees = ThreadPool(games).starmap(self.build_eval,args) # we execute in non-async since we need to await the result anyway for the next step
                
                self.merge_subtrees(subtrees)
                
                iteration += 1
        
        # in any case return the best move from root after termination
        return self.bestmove(0)
                
    def build_model(self, start_idx : int, subtree : dict, model, c=2, choices=1, epsilon=0.005) -> dict:
        '''
        The builder function which is executed by a thread pool to build the tree using the model.
        This function calles the tree_policy and default_policy routines.
        *The returned subtree is composed of a new branch where the expand function created a new leaf and the simulation chain until a terminal leaf.
        * Subject to future change
        
        :param start_idx (int): The index from which to start selection from.
        :param subtree (dict): The initial subtree dictionary which is empty.
        :param model (ChessModel): The model to be used in the simulation step.
        :param c (float): The exploration constant passed down from the __call__ function.
        :param choices (int): If 1 UCBT will use argmax, if choices > 1 UCBT will be a weighted random choice of the best choices options.
        :param epsilon (float): Hyperparameter representing the chance of a random choice during simulation.
        :returns (dir): The subtree dictionary to fuse with the parent tree.
        '''
        
        game = deepcopy(self.root_state) # creates a deepcopy of the tree's root state for further manipulation
        
        self.tree_policy(game, start_idx, subtree, c=c, choices=choices)
        
        # do a simulation only if the node from which to simulate is not a proven node
        if not subtree[0].proven:
            self.simulate_model(game, model, subtree, epsilon=epsilon)
        
        return subtree
    
    def build_eval(self, start_idx : int, subtree : dict, eval_func, c=2, choices=1, epsilon=0.01) -> dict:
        '''
        The builder function which is executed by a thread pool to build the tree using the evaluation function or random simulation.
        This function calles the tree_policy and default_policy routines.
        *The returned subtree is composed of a new branch where the expand function created a new leaf and the simulation chain until a terminal leaf.
        * Subject to future change
        
        :param start_idx (int): The index from which to start selection from.
        :param subtree (dict): The initial subtree dictionary which is empty.
        :param eval_func (EvaluationFunction): The evaluation function to be used in the simulation step.
        :param c (float): The exploration constant passed down from the __call__ function.
        :param choices (int): If 1 UCBT will use argmax, if choices > 1 UCBT will be a weighted random choice of the best choices options.
        :param epsilon (float): Hyperparameter representing the chance of a random choice during simulation.
        :returns (dir): The subtree dictionary to fuse with the parent tree.
        '''
        
        game = deepcopy(self.root_state) # creates a deepcopy of the tree's root state for further manipulation
        
        self.tree_policy(game, start_idx, subtree, c=c, choices=choices)
        
        # do a simulation only if the node from which to simulate is not a proven node
        if not subtree[0].proven:
            self.simulate_eval(game, eval_func, subtree, epsilon=epsilon)
        
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
        The method for the selection step of MCTS.
        From a starting index this method traverses self.tree down until it either hits a node which has unexplored children or a terminal node.
        
        The following occurances are handled:
        Case 1 : Select was called on a proven node. 
        Case 2 : Select was called on a node which has all children and all are proven.
        Case 3 : Select was called on a node which has unexplored children. 
        Case 4 : Select was called on a node which has no unexplored children.
        
        The following subtree and return values are created for each case:
        Case 1 -> returns (index=None, repeat=False), subtree={0 : MctsNode} where MctsNode is the proven node from self.tree[idx]
        Case 2 -> returns (index=None, repeat=False), subtree={0 : MctsNode} where MctsNode is the parent of all proven nodes and XYZ
        Case 3 -> returns (index=None, repeat=False), subtree={0 : MctsNode} where MctsNode is a newly created child node of self.tree[idx].
        Case 4 -> returns (index=int, repeat=True), subtree={}
        
        :param game (ChessGame): The ChessGame instance to use for obtaining possible moves and player turn.
        :param idx (int): The index of the node to select a child from.
        :param subtree (dict): The subtree to write to if a new node is created, later merged into self.tree.
        :param c (float): The exploration constant to be used in the UCBT formula.
        :param choices (int): Enables a weighted random choice between the highest n ranked children instead of the maximum child.
        :returns (tuple): Returns a tuple consisting of index and repeat value, index is None or int, repeat is boolean.
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
    
    def simulate_eval(self, game, eval_func, subtree : dict, epsilon=0.01):
        '''
        Simulates a chess game until a terminal state is reached and stores all simulation move nodes in a subtree.
        This method uses the either an evaluation function or random moves for simulation.
        The subtree will contain the result of the simulation and the correct referencing in itself where the root node points to its parent in the parent tree. 
        
        :param game (ChessGame): The ChessGame instance to play the simulation on.
        :param eval_func: Can be a EvaluationFunction class instance or None if simulation should be random.
        :param subtree (dict): The subtree which is to be filled with nodes from the simulation.
        :param epsilon (float): An epsilon value for random choice exploration during simulation.
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
    
    def simulate_model(self, game, model, subtree : dict, epsilon=0.005):
        '''
        Simulates a chess game until a terminal state is reached and stores all simulation move nodes in a subtree.
        This method uses the ChessModel class for simulation, each move is played according to the argmax the model policy output.
        The subtree will contain the result of the simulation and the correct referencing in itself where the root node points to its parent in the parent tree. 
        
        :param game (ChessGame): The ChessGame instance to play the simulation on.
        :param model (ChessModel): The ChessModel instance to generate moves from.
        :param subtree (dict): The subtree which is to be filled with nodes from the simulation.
        :param epsilon (float): An epsilon value for random choice exploration during simulation.
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
        Updates a node with a delta result and propagates the result to all its parents.
        This method is supposed to be called during the merge step on the parent node of each subtree.
        
        :param start_idx (int): The index of the node to update.
        :param delta (tuple): An update tuple to be applied to the node and its parents.
        :param visits (int): The amount of visits to update with.
        '''
        
        index_chain = [start_idx]
        while not 0 in index_chain:
            index_chain += [self.tree[index_chain[-1]].parent]
        
        for index in index_chain:
            self.tree[index].update(delta, visits)