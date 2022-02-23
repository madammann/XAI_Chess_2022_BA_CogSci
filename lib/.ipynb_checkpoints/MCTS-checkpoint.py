import queue
import multiprocessing

# class ModelHook:
#     '''The translator object for connecting to the model.'''
#     def __init__(self):
#         '''None'''
#         pass
    
#     def tensorify(epd):
#         '''None'''
#         pass

class MCTSNode:
    '''The node object for the MCTS tree.'''
    def __init__(self,state):
        '''Initialization method for the node object of the MCTS tree.
        
        Parameters:
            state (str): A string in epd format representing the chess board state.'''
        self.content = state + '#0%0'

    def update(self,result):
        '''This method updates the visit and win part of the node.
        
        Parameters:
            result (bool): A boolean value, true if won simulation, false if loss or draw.'''
        
        '''Get necessary string indices only one to save time'''
        ht_idx = self.content.find('#')
        sp_idx = self.content.find('%')
        
        '''Increment visits and wins based on result (result from the viewpoint of the root node color)'''
        if result:
            self.content[ht_idx+1:] = '#'+str(int(self.content[ht_idx+1:sp_idx])+1)+'/'+str(int(self.content[sp_idx+1:])+1)
        else:
            self.content[ht_idx+1:] = '#'+str(int(self.content[ht_idx+1:sp_idx])+1)+'/'+str(int(self.content[sp_idx+1:]))
    
    def epd(self):
        '''Returns the epd encoded board state string.'''
        return self.content[:self.content.find('#')]
    
    def visits(self):
        '''Returns the visit count for the node as integer.'''
        return int(self.content[self.content.find('#')+1:self.content.find('%')])
    
    def wins(self):
        '''Returns the win count for the node as integer.'''
        return int(self.content[self.content.find('%')+1:])

    
# class MCTSAction:
#     '''The action object for the MCTS tree build queue.'''
#     def __init__(self,name,data):
#         if not name in ['MAKE','UPDATE','']:
#             raise ValueError('This action cannot be performed by the MCTS Queue')
#         self.action = name

class MCTSTree:
    '''The tree object for MCTS.
    Attributes:
        self.tree (dict): A dictionary of addresses and node objects.
        self.node_limit (int): If the number of nodes exceeds this value the build method will terminate.
        self.modus (str): The mode in which MCTS is run, determines simulation result, allowed are "Deep", "Random", "Informed".'''
    
    def __init__(self,root_state,mode='Deep',node_limit=10**5):
        '''Creates initial tree object with attributes and the root node.
        Parameters:
            root_state (str): A string in epd format representing the root chess board state. 
            mode (str): Mode for the simulation method "Deep", "Random", "Informed".
            node_limit (int): Upper limit for the node count in the tree.'''
        self.tree = {'0x0' : MCTSNode(root_state)}
    
    def build(self,condition=None):
        '''Extends the search tree until the condition is met.
        Parameters:
            condition (string): Available conditions are "iter#", "depth#","converge".
        Examples:
            MCTSTree.build('iter2'): Makes two iterations for all parallel processes (One iteration is 500 node updates per writer object).
            MCTSTree.build('depth#'): Builds the tree until a certain depth is reached or the depth did not increase over the last 20 iterations.
            MCTSTree.build('converge'): Builds the tree until a promising move is found with confidence.
            NOTE: The building operation will always abort if there are more than an allowed limit of node in it.'''
        
    def find_n_most_promising_leafs(self):
        pass
    
    def simulation(self,node_address):
        pass
    