import numpy as np

from copy import deepcopy
from multiprocessing.pool import ThreadPool

from lib.engine import ChessGame

class MctsNode:
    def __init__(self, parent=None, move=None):
        self.data = {'move' : move, 'wins' : 0, 'proven' : False, 'games' : 0, 'children' : [], 'parent' : parent}
        
    def __getitem__(self, key : int):
        if key >= 0 and key <= self.child_count and self.child_count > 0:
            return self.data['children'][key]
        
        return None

    @property
    def parent(self):
        return self.data['parent']

    @property
    def child_count(self) -> int:
        return len(self.data['children'])

    @property
    def children(self) -> list:
        return self.data['children']

    @property
    def proven(self) -> bool:
        return self.data['proven']

    @property
    def value(self) -> float:
        if self.data['games'] > 0:
            return self.data['wins'] / self.data['games']
        else:
            return 0
        
    @property
    def move(self) -> str:
        return self.data['move']

    @property
    def games(self) -> int:
        return self.data['games']
    
    @property
    def wins(self) -> int:
        return self.data['wins']

    def expand(self, children : list):
        self.data['children'] = children

    def update(self, result : float):
        self.data['games'] += 1
        if result == 1:
            self.data['wins'] += 1

    def set_proven(self, value : float):
        self.data['proven'] = True
        
        if value == 1:
            self.data['wins'] = self.data['games']
        
        else:
            self.data['wins'] = 0

class MctsTree:
    def __init__(self, game : ChessGame, game_limit=1000):
        self.root_state = game #store the root state for deepcopying during tree traversal from root

        self.root = MctsNode()
        
        #hyperparameter initialization
        self.game_limit = game_limit
        self.simul_limit = None
        self.epsilon = 0
        self.A = 1
        self.c = 2
        
    def __len__(self) -> int:
        pass

    def bestmove(self, move_chain=[]) -> str:

        parent = self.root
        index = np.argmax(self.policy(move_chain=move_chain))

        return parent.children[index].move

    def policy(self, move_chain=[]) -> np.ndarray:

        #due to draws being as valuable as losses we cannot invert for the opponent and have to always be picking a move for our side
        if len(move_chain) % 2 != 0:
            raise ValueError('The parameter "move_chain" has to have an even count of moves!')

        #creates proxy instance of root node and the initial state
        parent = self.root
        game = deepcopy(self.root_state)

        #makes all moves in the move chain in the buffer and in the ChessGame instance
        for move in move_chain:
            index = game.possible_moves.index(move)
            game.select_move(game.possible_moves[index])
            parent = parent[index]

        #we handle the case with at least one proven win in the tree
        if any([child.value == 1.0 for child in parent.children if child.proven]):
            new_values = [0 for _ in range(parent.child_count)]
            for i, node in enumerate(parent.children):
                if node.proven and node.value == 1.0:
                    new_values[i] = 1
            
            return np.divide(np.array(new_values),np.sum(new_values))
        
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

        #make an array for the policy by first applying the formula val+(A/sqrt(games)), then dividing each value by the sum
        policy = np.array([child.value + (self.A/np.sqrt(child.games)) for child in parent.children])
        policy = np.divide(policy,np.sum(policy))    
        
        #else we just return the unadjusted policy
        return policy

    def __call__(self, simulation_func, simulation_limit=None, epsilon=0) -> np.ndarray:

        self.simul_limit = simulation_limit
        self.epsilon = epsilon

        results = self.build_tree(simulation_func)

        return self.policy()

    def expand(self, node : MctsNode, game : ChessGame) -> int:
        
        children = []
        for move in game.possible_moves:
            children += [MctsNode(parent=node, move=move)]

        node.expand(children)
        
    def build_tree(self, simulation_func):

        while self.root.games < self.game_limit:
            game = deepcopy(self.root_state)

            node = self.root #we start at the root node

            node, game = self.tree_policy(node, game)

            result = self.default_policy(node, game, simulation_func)

            self.update(node, result)

    def tree_policy(self, node : MctsNode, game : ChessGame):
        #proven cases I and II
        if node.proven and (node.parent != None):
            #we test all siblings whether they are proven nodes
            if all([sibling.proven for sibling in node.parent.children]):
                parent = node.parent
                values = [sibling.value for sibling in parent.children]

                #if the parent turn is from our perspective, then we will chose the best value since we are to move (parent.turn == not node.turn, therefore the "not" here)
                if not game.board.turn == self.root_state.board.turn:
                    parent.set_proven(max(values))

                    #proven node case II - Our turn
                    return parent, game #we can still return game even if it is actually the child's state since it won't matter if the returned node is proven

                #if it is not our turn we expect our opponent to chose the worst proven state for us
                else:
                    parent.set_proven(min(values))

                    #proven node case II - Opponent's turn
                    return parent, game #we can still return game even if it is actually the child's state since it won't matter if the returned node is proven

            #if they are not we simply return the proven node with case I
            else:
                return node, game

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
                    game.select_move(game.possible_moves[idx])

                    return child, game

                else:
                    return node, game

            #pick a child using UCBT
            ranking = np.array([[child.value, child.games] for child in children])
            ranking = np.divide(ranking.T[0,:], ranking.T[1,:]) + self.c * np.sqrt(np.divide(2 * np.log(np.sum(ranking[:,1],axis=0)),ranking.T[1,:])) # UCBT formula in numpy

            #we select the child to maximize the UCBT formula - case V
            child_idx = np.argmax(ranking)
            child = node[child_idx]

            #we do a final security measure in case we encounter an issue in which case we will return the parent instead
            if child != None:
                game.select_move(game.possible_moves[child_idx])

                return self.tree_policy(child, game)

            else:
                return node, game

        #if a node does not have children, create all of them via expand, then return the first of them - case III
        else:
            self.expand(node, game)

            #we pick a random child (we pick randomly to achieve a higher variance in the multithreaded domain)
            choice_idx = np.random.choice(range(node.child_count))
            child = node[choice_idx]

            #we do a final security measure in case we encounter an issue in which case we will return the parent instead
            if child != None:
                game.select_move(game.possible_moves[choice_idx])

                return child, game

            else:
                return node, game

    def default_policy(self, node : MctsNode, game : ChessGame, simulation_func) -> int:

        #if the provided node is already proven return it's value
        if node.proven:
            return int(node.value)

        #if the provided node is not yet proven but the state is terminal, make it proven
        if game.terminal:
            value = game.result[not self.root_state.board.turn] #since proven nodes returned did not play a move to get to this position we use "not" here
            node.set_proven(value)

            #If and only if the terminal state is a loss for us (not a draw), and it resulted from our opponent's move, then we assume optimal play and also update the parent node with the value
            if game.board.turn == self.root_state.board.turn and value == 0 and game.result != (0,0) and node.parent != None:
                node.parent.set_proven(value)

            return value

        #we simulate until we reach a terminal state with our simulation policy

        #version 1 - no simulation limit and no epsilon
        if self.simul_limit == None and self.epsilon == 0:
            while not game.terminal:
                move = simulation_func(game)
                game.select_move(move)

        #version 2 - no simulation limit but epsilon
        elif self.simul_limit == None:
            while not game.terminal:
                if np.random.random() < self.epsilon:
                    move = np.random.choice(game.possible_moves)

                else:
                    move = simulation_func(game)

                game.select_move(move)

        #version 3 - simulation limit and no epislon
        elif self.epsilon == 0:
            move_count = 0
            while not game.terminal and move_count <= self.simul_limit:
                move = simulation_func(game)
                game.select_move(move)
                move_count += 1

        #version 4 - simulation limit and epsilon
        else:
            move_count = 0
            while not game.terminal and move_count <= self.simul_limit:
                if np.random.random() < self.epsilon:
                    move = np.random.choice(game.possible_moves)

                else:
                    move = simulation_func(game)

                game.select_move(move)
                move_count += 1

        return game.result[self.root_state.board.turn]

    def update(self, node : MctsNode, result : float):

        #recursive case
        if node.parent != None:
            node.update(result)

            #recursive call of update on the parent with parental index
            self.update(node.parent, result)

        #base case (at root node)
        else:
            node.update(result)

    def evaluation(self):
        '''
        ADD
        '''

        return self.root.value
    
    def best_three(self):
        '''
        Method for returning the best three moves according to the search.
        '''

        parent = self.root
        indices = np.argsort(self.policy())[::-1][:3]

        return tuple([parent.children[index].move for index in indices])