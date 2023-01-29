import mmap
import struct

import numpy as np

from copy import deepcopy
from multiprocessing.pool import ThreadPool

from lib.engine import ChessGame
from lib.utils import int2move, move2int

class MctsNode:
    '''
    The MctsNode object is a proxy object that contains methods and attributes for reading from the mmap buffer.
    It is designed to work like a node object for a MCTS tree, but it's attributes are derived from or stored in the memory buffer and not in this object.
    This class is designed for minimal interaction with the buffer.
    
    Reference attributes:
    :attribute buf (mmap.mmap): The buffer mmap object to be read from and to write to.
    :attribute offset (int): The offset inside the buffer at which the node object starts.
    :attribute idx (int): The corresponding index of the node, derived from the offset given each node is 36 bytes long in the buffer.
    :attribute parent_idx (int): The index of the parent of this node, all nodes except root have a parent, the root node references itself.
    :attribute firstborn_idx (int): The index of the first child of this node, will be zero in case the node does not have children yet.
    :attribute child_count (int): The number of children of this node, will be zero in case the node does not have children yet.
    :attribute children_indices (list): A list of the indices of all children this node has, will be empty if the node does not have children yet.
    
    Node attributes:
    :attribute move (str): The uci-encoded chess move used to get from the parent node to this node.
    :attribute proven (bool): A boolean whether the value of this node is proven or not.
    :attribute value (float): The value of the current node, will also apply the virtual loss meaning that the value is discounted if a thread selects this node until it leaves it.
    :attribute games (int): The number of games played on this node and it's children.
    :attribute loss (int): The current loss of this node, will be incremented for each thread selecting this node or a child of it and decreased when the thread leaves the node.
    :attribute depth (int): The depth inside the MCTS tree for this node, starts at zero for the root node.
    
    Node generator attributes:
    :attribute parent (MctsNode): Returns the proxy object of this node's parent or None in case it is the root node.
    :attribute children (list): Returns a list of children proxy objects (MctsNode) of this node, this list is empty if the node does not yet have children.
    The "__getitem__" method is used for accessing a specific child at a key index, it is more efficient than retrieving all children so it should be used instead of the children property.
    '''
    
    def __init__(self, buf, idx : int):
        '''
        Constructor method for the MctsNode proxy object, initialized only from buffer reference and index.
        
        :param buf (mmap.mmap): The buffer mmap object from which to retrieve the node data.
        :param idx (int): The index of the node in the buffer starting from 0 up to a predefined limit.
        '''
        
        self.buf = buf
        self.offset = idx*36
    
    def __len__(self) -> int:
        '''
        Returns the length of the MctsNode proxy, in this context this means the depth of the node inside the tree.
        
        :returns int: The depth of the node inside the tree.
        '''
        
        return self.depth
    
    def __repr__(self) -> str:
        '''
        Method for returning a representative string for the MctsNode object.
        
        :returns str: A formstring containing the most relevant data of the object.
        '''
        
        if self.proven:
            return f'MctsNode({self.move},idx={self.idx},val={int(self.value)},games={self.games},proven={self.proven})'
        
        else:
            return f'MctsNode({self.move},idx={self.idx},val={self.value},games={self.games},proven={self.proven})'
    
    def __getitem__(self, key : int):
        '''
        Getitem method for retrieving an item contained in the node, in this context the object contains children nodes which can be accessed by a key.
        
        :param key (int): The index of the child starting from 0 up to the number of children of the node minus one.
        
        :returns MctsNode or None: Will return a child if it exists and None if it does not, meaning the index was too large or small or there are no children linked to this node yet.
        '''
        
        if key >= 0 and key <= self.child_count and self.child_count > 0:
            return MctsNode(self.buf, self.firstborn_idx + key)
        
        else:
            return None
    
    @property
    def idx(self) -> int:
        '''
        The index inside the buffer.
        '''
        
        return int(self.offset/36)
    
    @property
    def parent_idx(self) -> int:
        '''
        The index of the parent node inside the buffer.
        '''
        
        return int(struct.unpack("q", self.buf[self.offset:self.offset+8])[0])

    @property
    def parent(self):
        '''
        The parent node as MctsNode object or None if self is the root node.
        '''
        
        if not self.idx == 0:
            return MctsNode(self.buf,self.parent_idx)
        
        else:
            return None
    
    @property
    def firstborn_idx(self) -> int:
        '''
        The index of the first child inside the buffer.
        '''
        
        return int(struct.unpack("q", self.buf[self.offset+8:self.offset+16])[0])
    
    @property
    def child_count(self) -> int:
        '''
        The number of children this node posseses.
        '''
        
        return int(struct.unpack("h", self.buf[self.offset+16:self.offset+18])[0])
    
    @property
    def children_indices(self) -> list:
        '''
        A list of indices of the children of this node inside the buffer.
        '''
        
        start_idx = self.firstborn_idx
        
        return list(range(start_idx, start_idx+self.child_count))
    
    @property
    def children(self) -> list:
        '''
        A list of MctsNode objects of the children of this node.
        '''
        
        return [MctsNode(self.buf,idx) for idx in self.children_indices]
    
    @property
    def proven(self) -> bool:
        '''
        The boolean value whether this node is proven or not.
        '''
        
        return bool(struct.unpack("?", self.buf[self.offset+18:self.offset+19])[0])
    
    @property
    def value(self) -> float:
        '''
        The value of this node, potentially discounted by the loss with the formula val*0,5^loss.
        '''
        
        return float(struct.unpack("f", self.buf[self.offset+20:self.offset+24])[0] * 0.5**self.loss)
    
    @property
    def loss(self) -> int:
        '''
        The current loss of this node.
        '''
        
        return int(struct.unpack("b", self.buf[self.offset+24:self.offset+25])[0])
    
    @property
    def move(self) -> str:
        '''
        The uci-encoded move this node represents.
        '''
        
        move = struct.unpack("i", self.buf[self.offset+28:self.offset+32])[0]
        if self.idx > 0 and move != 0:
            return int2move(move)
        
        else:
            return None
    
    @property
    def games(self) -> int:
        '''
        The number of games simulated on this node and it's children.
        '''
        
        return int(struct.unpack("i", self.buf[self.offset+32:self.offset+36])[0])
    
    @property
    def depth(self) -> int:
        '''
        The depth of this node inside the tree
        '''
        
        if self.idx == 0:
            return 0
        
        parent = self.parent
        depth = 1
        while parent.idx > 0:
            depth += 1
            parent = parent.parent
        
        return depth
    
    def create(self, bytestring):
        '''
        Method for creating a node inside the buffer by providing a bytestring for it's values.
        Overwrites all node information with the bytestring provided at the given offset of self.
        Will overwrite and flush.
        
        :param bytestring: A bytestring representing the node data to overwrite the buffer with.
        '''
        
        self.buf.seek(self.offset)
        self.buf.write(bytestring)
        self.buf.flush(self.offset,36)
        
    def expand(self, firstborn_idx : int, children : int):
        '''
        Method for adding a children reference to this node inside the buffer.
        Children will have to be created separately.
        Will overwrite the firstborn and child count values inside the buffer and flush.
        
        :param firstborn_idx (int): The index inside the buffer which marks the first child of this node.
        :param children (int): The number of children this node will have.
        '''
        
        if self.child_count == 0:
            self.buf.seek(self.offset+8)
            self.buf.write(struct.pack("q",firstborn_idx))
            self.buf.flush(self.offset+8,8)
            
            self.buf.seek(self.offset+16)
            self.buf.write(struct.pack("h",children))
            self.buf.flush(self.offset+16,2)

    def update(self, result : float):
        '''
        Method for updating the value of this node using the incremental mean formula.
        Will overwrite the game count and or value of the corresponding buffer position and flush.
        In case the node is proven only the game count will be incremented by one while value remains unchanged.
        
        :param result (float): The float or int value to update the node with, the result of the simulation on the node or it's children.
        '''
        
        if not self.proven:
            games = self.games + 1
            value = self.value
            value = value + ((result-value)/games)
        
            self.buf.seek(self.offset+20)
            self.buf.write(struct.pack("f",value))
            self.buf.flush(self.offset+20, 4)
        
            self.buf.seek(self.offset+32)
            self.buf.write(struct.pack("i",games))
            self.buf.flush(self.offset+32, 4)
            
        else:
            self.buf.seek(self.offset+32)
            self.buf.write(struct.pack("i",self.games + 1))
            self.buf.flush(self.offset+32, 4)
    
    def add_loss(self):
        '''
        Method for incrementing the loss of this node by one within buffer constraints.
        Will overwrite the loss in the corresponding buffer position with loss+1 and flush.
        '''
        
        loss = self.loss
        
        if loss < 128:
            self.buf.seek(self.offset+24)
            self.buf.write(struct.pack("b",loss+1))
            self.buf.flush(self.offset+24, 1)
    
    def remove_loss(self):
        '''
        Method for decreasing the loss of this node by one within buffer constraints.
        Will overwrite the loss in the corresponding buffer position with loss-1 and flush.
        '''
        
        loss = self.loss
        
        if loss > 0:
            self.buf.seek(self.offset+24)
            self.buf.write(struct.pack("b",loss-1))
            self.buf.flush(self.offset+24, 1)
        
    def set_proven(self, value : float):
        '''
        Method for changing a node to a proven node.
        Will overwrite the proven, loss, and value bytes inside the buffer and flush.
        Value will be set, proven set to true, and loss will be set to 0.
        
        :param value (float): The proven game-theoretic value of the node as int or float.
        '''
        
        #sets value to proven
        self.buf.seek(self.offset+18)
        self.buf.write(struct.pack("?",True))
        self.buf.flush(self.offset+18,1)
        
        #removes all loss
        self.buf.seek(self.offset+24)
        self.buf.write(struct.pack("i",0))
        self.buf.flush(self.offset+24,1)
        
        #sets value to fixed value
        self.buf.seek(self.offset+20)
        self.buf.write(struct.pack("f",value))
        self.buf.flush(self.offset+20,4)
    
class MctsTree:
    '''
    The MctsTree object is capable of using MCTS with shared memory and multithreading.
    This class contains methods for running MCTS with different simulation policies for different parameter options.
    
    
    IMPORTANT: It is required to call the close method before creating another instance of this class and ideally also as soon as the MCTS results are no longer needed before deleting the object.
    
    Constant attributes:
    :attribute root_state (ChessGame): The ChessGame instance the root state corresponds to.
    :attribute NODE_LIMIT (int): The number of maximum nodes the MCTS can create in total before the buffer is full.
    :attribute MMAP_FILESIZE (int): The number of bytes the buffer will contain, calculated from node limit times 36 for 36 bytes per node.
    :attribute THREADS (int): The number of threads to use for MCTS.
    :attribute LIMITS (list): The list of allocated minimum and maximum indices each thread is allowed to use.
    
    Buffer and file reference attributes:
    :attribute file: The file object used to create the mmap.mmap buffer object from.
    :attribute buf (mmap.mmap): The buffer object for shared memory.
    
    Hyperparameters (non-constant, they may be changed at a later point):
    :attribute game_limit (int): The currently allowed number of simulations to do before forced termination, may be incremented later.
    :attribute A (int or float): The ADD constant used for the confidence bound when selecting the final move, default value is 1, must be greater than 0.
    :attribute c (int or float): The exploration constant used in the UCBT formula, default value is 2.
    '''
    
    def __init__(self, game : ChessGame, node_limit=23000, threads=6):
        '''
        Constructor method for this class.
        Creates and allocates a shared memory mmap buffer and file.
        Then calculates intervals for each thread to write in and initializes the root node.
        
        :param game (ChessGame): The ChessGame instance to be used as the root node's state from which to search.
        :param node_limit (int): The maximum number of nodes the buffer can contain, default is 23.000 (assuming 1000 nodes with each 23 children on average)
        :param threads (int): The number of threads to run in multithreading, default is six.
        '''
        
        #stores the ChessGame object representing the root state of the tree
        self.root_state = game #store the root state for deepcopying during tree traversal from root
        
        #initializes necessary constants for shared memory and multithreading
        self.NODE_LIMIT = node_limit #1000 nodes searched with 23 avg children count
        self.THREADS = 6
        self.MMAP_FILESIZE = 36*self.NODE_LIMIT
        
        #calculate the writing limits for each singular thread
        limits = np.arange(0,self.NODE_LIMIT+int(self.NODE_LIMIT/self.THREADS),int(self.NODE_LIMIT/self.THREADS))
        self.LIMITS = [(limits[i],limits[i+1]) for i in range(self.THREADS)]
        self.LIMITS[0] = (1,self.LIMITS[0][1]) #add
        
        #initializes the memory mapped file and creates the root node
        self.file = open("buf","w+")
        self.file.truncate(self.MMAP_FILESIZE)
        self.buf = mmap.mmap(self.file.fileno(),length=0,access=mmap.ACCESS_COPY)
        self.buf.write(struct.pack("qqh?fbii",0,0,0,False,0,0,0,0)) #writes the root into the buffer with children count and pointer set
        self.buf.flush(0,36) #flushes root node to memory
        
        #hyperparameter initialization
        self.game_limit = 500
        self.simul_limit = None
        self.A = 1
        self.c = 2
        
    def close(self):
        '''
        Method for closing the mmap buffer and the file object before garbage-collecting self, call this before deleting the object.
        '''
        
        if not self.buf.closed:
            self.buf.close()
        
        if not self.file.closed:
            self.file.close()
    
    def __del__(self):
        '''
        Method to be called before destroying this object to de-allocate the buffer and file properly.
        '''
        
        self.close()
    
    def __len__(self) -> int:
        '''
        Method for returning the number of nodes inside the buffer which have actually been created (IMPORTANT: number of nodes != number of simulations/games).
        
        :returns int: The number of actual nodes inside the buffer.
        '''
        
        return len([child for child in [MctsNode(self.buf, idx) for idx in range(self.NODE_LIMIT)] if child.move != None])+1 #+1 since the root node also has None as move
    
    def __repr__(self) -> str:
        '''
        Method for printing a string representing this object.
        The printed string will contain information about the root node, it's children and the total node count of the tree.
        
        :returns str: A long string with information about root node, it's children, and the total node count.
        '''
        
        string = f'MctsTree object with root node {MctsNode(self.buf, 0)}\n'
        children = MctsNode(self.buf, 0).children
        
        for i, child in enumerate(children):
            string += f'Child {i}: {child} \n'
            
        string += f'Total node count {len(self)}.'
        
        return string
    
    def __contains__(self, moves : list) -> bool:
        '''
        Method for testing whether the tree contains a certain sequenece of moves.
        
        :param moves (list): A list of uci encoded move strings.
        
        :returns bool: True if the sequence is found, False if not.
        '''
        
        node = MctsNode(self.buf, 0)
        
        #goes through all the moves
        for move in moves:
            try:
                #if the move is not inside the children move list return False, else continue until all moves in the list were naviagted through
                if not move in [child.move for child in node.children]:
                    return False
                
                idx = [child.move for child in node.children].index(move)
                node = node[idx]
            
            #if a node has no children yet it will return None and raise this error so we return False if the case and catch it 
            except AttributeError:
                return False
        
        #if this is reached this means all moves were in the tree and we return True
        return True
        
    
    def bestmove(self, move_chain=[]) -> str:
        '''
        Method for receiving the best move from a certain point in the tree, default is from the root node.
        This method calls the policy method and uses the argmax over it to find the best move according to policy.
        
        :param move_chain (list): A list of moves in uci format from the root node state to move before taking the argmax, has to be of even length.
        
        :returns str: The move string in uci format of the node which was selected as best.
        '''
        
        parent = MctsNode(self.buf, 0)
        index = np.argmax(self.policy(move_chain=move_chain))
        
        return parent.children[index].move
    
    def policy(self, move_chain=[]) -> np.ndarray:
        '''
        Method for receiving the policy from a certain point in the tree, default is from the root node.
        If the best move is to be searched from a different position in the tree it has to be one with the same player to move as the root node.
        This method uses the formula val+(A/sqrt(games)) with default value for A being 1.
        The policy will be divided by the sum of values generated from the aforementioned formula to make it sum up to 1.
        
        :param move_chain (list): A list of moves in uci format from the root node state to move before taking the argmax, has to be of even length.
        
        :returns np.ndarray: A numpy array with probabilities for each move, the highest probability is the best move according to evaluation.
        '''
        
        #due to draws being as valuable as losses we cannot invert for the opponent and have to always be picking a move for our side
        if len(move_chain) % 2 != 0:
            raise ValueError('The parameter "move_chain" has to have an even count of moves!')
            
        #creates proxy instance of root node and the initial state
        parent = MctsNode(self.buf,0)
        game = deepcopy(self.root_state)
        
        #makes all moves in the move chain in the buffer and in the ChessGame instance
        for move in move_chain:
            index = game.possible_moves.index(move)
            game.select_move(game.possible_moves[index])
            parent = parent[index]
        
        #make an array for the policy by first applying the formula val+(A/sqrt(games)), then dividing each value by the sum
        policy = np.array([child.value + (self.A/np.sqrt(child.games)) for child in parent.children])
        policy = np.divide(policy,np.sum(policy))
        
        #in all cases except when all nodes are losing or drawing this will already work
        if len([child for child in parent.children if child.proven]) == parent.child_count:
            #if all children are proven and none is a win we adjust the policy
            if max([child.value for child in parent.children]) == 0:
                #we create a list of new values all set to zero
                new_values = [0 for _ in range(parent.child_count)]
                #then we iterate over all moves...
                for i, move in enumerate(game.possible_moves):
                    checker = deepcopy(game)
                    checker.select_move(move)
                    #...and check for each move if it is a draw, if yes we change that index to 1
                    if checker.result == (0,0):
                        new_values[i] = 1
                
                #finally we make sure the sum of the new policy is one by division and return it
                return np.divide(np.array(new_values),np.sum(new_values))
            
        #else we just return the unadjusted policy
        return policy
        
    def __call__(self, simulation_func, game_limit=800, simulation_limit=None) -> np.ndarray:
        '''
        Call method for this class, used for building the MCTS tree.
        Will start building the tree in parallel in the shared memory buffer using multithreading.
        This method may be called again to generate more nodes, the game_limit set in here is incremental.
        
        :param simulation_func (func): Any function which takes the ChessGame state instance as parameter and returns a move as string in uci format.
        :param game_limit (int): The maximum number of simulations to run in total, if the buffer size would be exceeded the algorithm will terminate earlier.
        
        :returns np.ndarray: A numpy array with probabilities for each move, the highest probability is the best move according to evaluation.
        '''
        
        self.game_limit = MctsNode(self.buf,0).games + game_limit #set allowed game limit to current games plus the parameter set
        self.simul_limit = simulation_limit
        
        pool = ThreadPool(self.THREADS)
        args = [(simulation_func, self.LIMITS[i][0], self.LIMITS[i][1]) for i in range(self.THREADS)]
        
        #runs the build_tree method with the ThreadPool instance for each shared memory interval
        results = pool.starmap(self.build_tree, args)
        
        return self.policy()
    
    def expand(self, node : MctsNode, game : ChessGame, buffer_idx : int, buffer_idx_limit : int) -> int:
        '''
        Method for creating a node's children and linking them.
        Creates all children which still fit into the interval of shared memory between buffer_idx and buffer_idx_limit.
        Increments the buffer_idx in the allocated interval of shared memory for the amount of children created.
        Children are created with reference to parent and their move, then the parent is linked to them using the expand method of MctsNode.
        
        :param node (MctsNode): The MctsNode proxy object for which to create the children.
        :param game (ChessGame): The ChessGame instance at the current move for the node, used here to receive the possible moves.
        :param buffer_idx (int): The first index in the index interval allocated to this thread.
        :param buffer_idx_limit (int): The last index in the index interval allocated to this thread marking the maximum index this thread may write to.
        
        :returns int: The new buffer_idx after creating the nodes, used to know which children have already been created.
        '''
        
        #make parent point to firstborn and add children amount to it
        children_count = len(game.possible_moves)
        
        #if the buffer would be full, only create the children for which there is space (this may lead to wrong all-proven rules but is purposefully ingnored since tree policy should terminate anyway already)
        if buffer_idx + children_count >= buffer_idx_limit:
            children_count = buffer_idx_limit - buffer_idx
        
        #for each child node, create it with proper referencing to parent and its move but without proven or playouts
        for i, move in enumerate(game.possible_moves[:children_count]):
            MctsNode(self.buf, buffer_idx+i).create(struct.pack("qqh?fbii",node.idx,0,0,False,0,0,move2int(move),0)) #creates and flushes node via proxy object
        
        #links the children to the parent node by giving it the firstborn index and children_count
        node.expand(buffer_idx, children_count)
        
        return buffer_idx + children_count
    
    def build_tree(self, simulation_func, buffer_idx : int, buffer_idx_limit : int):
        '''
        Method for building a MCTS tree in a given allocated memory interval in shared memory.
        This method is to be given to a thread and will run until the allocated memory interval is full or the desired simulation count is reached.
        
        :param simulation_func (func): A given simulation function for the default policy method, must accept a ChessGame object as input and return a uci-encoded move string of it's legal move list.
        :param buffer_idx (int): The first index in the index interval allocated to this thread.
        :param buffer_idx_limit (int): The last index in the index interval allocated to this thread marking the maximum index this thread may write to.
        '''
        
        #for as long as there is enough memory left, and game limit not exceeded, do: 
        while buffer_idx < buffer_idx_limit and MctsNode(self.buf, 0).games < (self.game_limit - self.THREADS + 1):
            game = deepcopy(self.root_state)
            
            node = MctsNode(self.buf, 0) #we start at the root node
            
            node, game, buffer_idx = self.tree_policy(node, game, buffer_idx, buffer_idx_limit)
            
            result = self.default_policy(node, game, simulation_func)
            
            self.update(node, result)
        
    def tree_policy(self, node : MctsNode, game : ChessGame, buffer_idx : int, buffer_idx_limit : int):
        '''
        Method for selecting the a node to run a simulation on with a selection policy.
        This method will either return a node, game, and the current buffer index, or recursively call itself on a selected node.
        We may encounter different cases which can be categorized into these:
        
        Proven node cases I and II:
        We return the node if it is terminal unless...
        ...all it's siblings are proven to, in which case we will return the parent after setting it to proven with the value of the highest or lowest child value, depending on whose move it is.
        
        Expansion case III:
        If the current node has no children we create them and select one randomly to return.
        
        Unexplored case IV:
        If not all children of the current node have been visited we select a random unvisited child node to return.
        
        UCBT case V:
        If all children have been visited we use the UCBT formula to select a child node from which to recursively call tree_policy on.
        
        :param node (MctsNode): The node which is currently selected in the tree policy.
        :param game (ChessGame): The game instance for the currently selected node.
        :param buffer_idx (int): The first index in the index interval allocated to this thread.
        :param buffer_idx_limit (int): The last index in the index interval allocated to this thread marking the maximum index this thread may write to.
        '''
        
        #proven cases I and II
        if node.proven and node.idx > 0:
            #we test all siblings whether they are proven nodes
            if all([sibling.proven for sibling in node.parent.children]):
                parent = node.parent
                values = [sibling.value for sibling in parent.children]
                
                #if the parent turn is from our perspective, then we will chose the best value since we are to move (parent.turn == not node.turn, therefore the "not" here)
                if not game.board.turn == self.root_state.board.turn:
                    parent.set_proven(max(values))
                    
                    #proven node case II - Our turn
                    return parent, game, buffer_idx #we can still return game even if it is actually the child's state since it won't matter if the returned node is proven
                
                #if it is not our turn we expect our opponent to chose the worst proven state for us
                else:
                    parent.set_proven(min(values))
                    
                    #proven node case II - Opponent's turn
                    return parent, game, buffer_idx #we can still return game even if it is actually the child's state since it won't matter if the returned node is proven
            
            #if they are not we simply return the proven node with case I
            else:
                return node, game, buffer_idx
        
        #if the current node has a children count then it was already expanded so we proceed with case IV and V
        if not node.child_count == 0:
            children = node.children
            
            #generate a list that can be empty of all children not yet played on
            unplayed = [(i, child) for i, child in enumerate(children) if child.games == 0]
            
            #if the list is not empty pick a random unvisited child and return it - case IV
            if len(unplayed) > 0:
                choice_idx = np.random.choice(range(len(unplayed)))
                idx, child = unplayed[choice_idx][0], unplayed[choice_idx][1]
                
                #we do a final security measure in case we encounter an issue in which case we will return the parent instead
                if child != None:
                    child.add_loss()
                    game.select_move(game.possible_moves[idx])
                    
                    return child, game, buffer_idx
                
                else:
                    return node, game, buffer_idx
                    
            #pick a child using UCBT
            ranking = np.array([[child.value, child.games] for child in children])
            ranking = np.divide(ranking.T[0,:], ranking.T[1,:]) + self.c * np.sqrt(np.divide(2 * np.log(np.sum(ranking[:,1],axis=0)),ranking.T[1,:])) # UCBT formula in numpy
            
            #we select the child to maximize the UCBT formula - case V
            child_idx = np.argmax(ranking)
            child = node[child_idx]
            
            #we do a final security measure in case we encounter an issue in which case we will return the parent instead
            if child != None:
                child.add_loss()
                game.select_move(game.possible_moves[child_idx])
            
                return self.tree_policy(child, game, buffer_idx, buffer_idx_limit)
            
            else:
                return node, game, buffer_idx
        
        #if a node does not have children, create all of them via expand, then return the first of them - case III
        else:
            buffer_idx = self.expand(node, game, buffer_idx, buffer_idx_limit)
            
            #we pick a random child (we pick randomly to achieve a higher variance in the multithreaded domain)
            choice_idx = np.random.choice(range(node.child_count))
            child = node[choice_idx]
            
            #we do a final security measure in case we encounter an issue in which case we will return the parent instead
            if child != None:
                child.add_loss()
                game.select_move(game.possible_moves[choice_idx])
            
                return child, game, buffer_idx
            
            else:
                return node, game, buffer_idx
    
    def default_policy(self, node : MctsNode, game : ChessGame, simulation_func) -> int:
        '''
        Method for running a simulation from the current node until a terminal state is reached.
        The simulation policy is provided with the simulation function given and will be used to generate moves from the current state in the simulation.
        If a node is proven this method will immediately return it's value.
        If a node is not proven yet but the provided game is terminal, then it will be set to proven with the terminal reward value.
        If the terminal state is a win but the move is from our opponent's perspective, then we assume optimal play and also set the parent node to a proven loss for us.
        Else we simulate until we reach a terminal state from the current position and return it's value.
        
        :param node (MctsNode): The node from which to run the current simulation.
        :param game (ChessGame): The game instance of ChessGame corresponding to the node selected, used for the actual simulation.
        :param simulation_func (func): A given simulation function, must accept a ChessGame object as input and return a uci-encoded move string of it's legal move list.
        '''
        
        #if the provided node is already proven return it's value
        if node.proven:
            return int(node.value)
        
        #if the provided node is not yet proven but the state is terminal, make it proven
        if game.terminal:
            value = game.result[not self.root_state.board.turn] #since proven nodes returned did not play a move to get to this position we use "not" here
            node.set_proven(value)
            
            #If and only if the terminal state is a loss for us (not a draw), and it resulted from our opponent's move, then we assume optimal play and also update the parent node with the value
            if game.board.turn == self.root_state.board.turn and value == 0 and game.result != (0,0):
                node.parent.set_proven(value)
            
            return value
        
        #we simulate until we reach a terminal state with our simulation policy
        if self.simul_limit == None:
            while not game.terminal:
                move = simulation_func(game)
                game.select_move(move)
                
        else:
            move_count = 0
            while not game.terminal and move_count <= self.simul_limit:
                move = simulation_func(game)
                game.select_move(move)
                move_count += 1
        
        return game.result[self.root_state.board.turn]
    
    def update(self, node : MctsNode, result : float):
        '''
        Method for uppropagating the result of the simulation of a node.
        Will recursively call itself on the parent of the current node until the current node is the root node.
        Additionally, will remove the loss added in the selection part since the thread has left the node now.
        
        :param node (MctsNode): The node proxy object whose value to update and parents to update.
        :param result (float): The float or int result of the simulation.
        '''
        
        #recursive case
        if node.parent != None:
            node.update(result)
            node.remove_loss() #the root node has no virtual loss, other nodes will need to get it removed in this step
            
            #recursive call of update on the parent with parental index
            self.update(node.parent, result)
        
        #base case (at root node)
        else:
            node.update(result)