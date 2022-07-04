import chess
import tensorflow as tf
import numpy as np

'''This module contains the ChessGame base class, the illegal move exception class, and conversion functions from move embedding to uci move'''

class IllegalMoveError(Exception):
    '''IllegalMoveError Exception for error handling.'''
    def __init__(self, string):
        pass

class ChessGame:
    '''ChessGame base class which is designed for MCTS efficiency primarily.'''
    def __init__(self, T=8, initial_moves=[]):
        self.board = chess.Board()
        self.T = T
        self.terminal = False
        self.result = None
        
        if initial_moves == []:
            self.possible_moves = [move.uci() for move in self.board.legal_moves]
            self.tensor_history = [self.board_tensor()]
        else:
            self.tensor_history = []
            self.play_move_chain(initial_moves)
            self.possible_moves = [move.uci() for move in self.board.legal_moves]
    
    def select_move(self, move_uci, history=True):
        '''Push move assuming only legal moves are pushed.'''
        
        #TODO Improve this coding structure
        if not self.terminal:
            try:
                self.board.push(chess.Move.from_uci(move_uci))
            except:
                if move_uci == 'claim draw':
                    if self.board.is_fifty_moves():
                        self.terminal = True
                        self.result = (0,0)
                        
                    elif self.board.is_repetition(3):
                        self.terminal = True
                        self.result = (0,0)
            
            # the game is only counted as terminal if no move could change the result of the game
            if self.board.is_checkmate():
                self.terminal = True
                
                result = self.board.result()
                if result == '1-0':
                    self.result = (1,0)
                
                if result == '0-1':
                    self.result = (0,1)
                
            # the game is stalemate, meaning there is no move for one side and also no check
            elif self.board.is_stalemate():
                self.terminal = True
                self.result = (0,0)
            
            # the game is a forced draw if there is a fivefold repetition according to the "new" draw FIDE rules
            elif self.board.is_fivefold_repetition():
                self.terminal = True
                self.result = (0,0)
                
            # the game is a forced draw if 75 progressless moves were made, according to the "new" draw FIDE rules
            elif self.board.is_seventyfive_moves():
                self.terminal = True
                self.result = (0,0)
            
            # the game is a forced draw if only kings are left on the board, meaning insufficient material in any case    
            elif not ('P' or 'p' or 'N' or 'n' or 'B' or 'b' or 'R' or 'r') in self.board.epd():
                self.terminal = True
                self.result = (0,0)
        
            '''Designed to skip creating unecessary tensors in case of playing an initial move chain in self.__init__'''
            if history:
                '''Add current board tensor to the tensor history'''
                self.tensor_history += [self.board_tensor()]
            
                '''Delete tensors which exceed parameter T'''
                self.tensor_history = self.tensor_history[-self.T:]
            
                self.possible_moves = [move.uci() for move in self.board.legal_moves]
    
    def play_move_chain(self, moves):
        '''Plays all moves efficiently.'''
        
        for i in range(len(moves)):
            if (len(moves) - i) > self.T:
                self.select_move(moves[i],history=False) #do not create unecessary tensors for elements not in the range of T
            else:
                self.select_move(moves[i])
    
    def board_tensor(self):
        '''Creates a tensor of the current board state which includes piece positions and repitition count.'''
        
        tensor = []
        for color in [chess.WHITE,chess.BLACK]:
            for piece in [chess.PAWN,chess.ROOK,chess.KNIGHT,chess.BISHOP,chess.QUEEN,chess.KING]:
                tensor += [np.array(self.board.pieces(piece,color).tolist()).reshape(8,8,1)]
        
        '''TODO: Implement the repetition for opposite color board or leave it like this'''
        '''Get the current board repetition count in the most convenient way'''
        rep = 1
        while not self.board.is_repetition(rep):
            rep += 1
        tensor += [np.full((8,8,1),rep)]
        
        '''Stack into one board tensor'''
        tensor = np.dstack(tensor)
        
        return tensor
    
    def rule_tensor(self):
        '''Creates a tensor of the current rule state which includes Castling rights, turn, and move counters.'''
        
        tensor = []
        
        '''Get the current turn encoding as plane of binary value'''
        tensor += [np.full((8,8,1),self.board.turn)]
        
        '''Get the total move count as plane with integer value'''
        tensor += [np.full((8,8,1),self.board.fullmove_number)]
        
        '''Get the castling rights as a binary value for each color and side'''
        tensor += [np.full((8,8,1),bool(self.board.castling_rights & chess.BB_A1))] # White Queenside
        tensor += [np.full((8,8,1),bool(self.board.castling_rights & chess.BB_H1))] # White Kingside
        tensor += [np.full((8,8,1),bool(self.board.castling_rights & chess.BB_A8))] # Black Queenside
        tensor += [np.full((8,8,1),bool(self.board.castling_rights & chess.BB_H8))] # Black Kingside
        
        '''Get the number of halfmoves (moves without progress)'''
        tensor += [np.full((8,8,1),self.board.halfmove_clock)]
        
        '''Stack into one rule tensor'''
        tensor = np.dstack(tensor)
        
        return tensor
    
    def get_input_tensor(self):
        '''Returns a tensor made out of the board tensor history and the current rule tensor for network input.'''
        
        tensor = self.tensor_history[-self.T:] # get the last t board tensors which exist
        tensor += [self.rule_tensor()]
        tensor = np.dstack(tensor)
        
        '''Flip the board tensor horizontally if black is to move'''
        if self.board.turn == chess.BLACK:
            tensor = np.flip(tensor,axis=0) # We can flip the entire tensor since the rule dimensions are uniform (np.full)
        
        
        # if the output tensor's shape would be smaller than intended it is filled with zero-valued layers
        if tensor.shape[2] < self.T * 13 + 7:
            tensor = np.dstack([tensor,np.dstack([np.full((8,8,1),0)]*(self.T * 13 + 7 - tensor.shape[2]))])
            
        return tf.expand_dims(tf.constant(tensor,dtype='float32'),axis=0)
    
    def get_result(self) -> tuple:
        '''Returns a tuple of length 2 with values 1:win,0:draw,-1:loss for each side if the game is over.'''
        if self.result != None:
            return self.result
    
'''Move embedding to uci conversion functions'''

def idx2field(index : tuple, flipped=False) -> str:
    '''Receives a tuple for a field coordinate and turns it into the field\'s string.'''
    board = np.array([
    ['a1','b1','c1','d1','e1','f1','g1','h1'],
    ['a2','b2','c2','d2','e2','f2','g2','h2'],
    ['a3','b3','c3','d3','e3','f3','g3','h3'],
    ['a4','b4','c4','d4','e4','f4','g4','h4'],
    ['a5','b5','c5','d5','e5','f5','g5','h5'],
    ['a6','b6','c6','d6','e6','f6','g6','h6'],
    ['a7','b7','c7','d7','e7','f7','g7','h7'],
    ['a8','b8','c8','d8','e8','f8','g8','h8']
    ])
    if flipped:
        board = np.flip(board,axis=0)
    return board[index]

def deltamove(origin : tuple, delta : tuple, flipped=False) -> str:
    '''Receives an origin and delta tuple and returns the uci move string for the two fields.'''
    
    '''The delta for the char-axis is inverted so each side's moves are mapped identically on a flipped board'''
    target = None
    if flipped:
        target = (origin[0]+delta[0],origin[1]-delta[1])
    else:
        target = (origin[0]+delta[0],origin[1]+delta[1])
    
    '''Moves that would surpass the boarder are illegal'''
    if min(target) < 0 or max(target) > 7:
        raise IllegalMoveError('Move is outside of board')
        
    return idx2field(origin,flipped=flipped) + idx2field(target,flipped=flipped)

def decode_move(position : tuple, flipped=False) -> tuple:
    '''Decodes a position in the policy embedding into a move.'''
    
    '''Queenmove'''
    if position[2] <= 55:
        direction = int(position[2]/7) # 0-7 direction to move from origin field (N,NE,E,SE,S,SW,W,NW)
        distance = int(position[2]-direction*7)+1 # 1-7 distance to move from origin field (1,2,3,4,5,6,7)
        if direction == 0:
            return (deltamove(position[:2],(distance,0),flipped=flipped),'q')
        if direction == 1:
            return (deltamove(position[:2],(distance,distance),flipped=flipped),'q')
        if direction == 2:
            return (deltamove(position[:2],(0,distance),flipped=flipped),'q')
        if direction == 3:
            return (deltamove(position[:2],(-distance,distance),flipped=flipped),'q')
        if direction == 4:
            return (deltamove(position[:2],(-distance,0),flipped=flipped),'q')
        if direction == 5:
            return (deltamove(position[:2],(-distance,-distance),flipped=flipped),'q')
        if direction == 6:
            return (deltamove(position[:2],(0,-distance),flipped=flipped),'q')
        if direction == 7:
            return (deltamove(position[:2],(distance,-distance),flipped=flipped),'q')
    
    '''Knightmove'''
    if position[2] <= 63:
        direction = position[2]-56 # direction 0-7 for all eight options for a knight to move
        if direction == 0:
            return (deltamove(position[:2],(2,1),flipped=flipped),None)
        if direction == 1:
            return (deltamove(position[:2],(1,2),flipped=flipped),None)
        if direction == 2:
            return (deltamove(position[:2],(-1,2),flipped=flipped),None)
        if direction == 3:
            return (deltamove(position[:2],(-2,1),flipped=flipped),None)
        if direction == 4:
            return (deltamove(position[:2],(-2,-1),flipped=flipped),None)
        if direction == 5:
            return (deltamove(position[:2],(-1,-2),flipped=flipped),None)
        if direction == 6:
            return (deltamove(position[:2],(1,-2),flipped=flipped),None)
        if direction == 7:
            return (deltamove(position[:2],(2,-1),flipped=flipped),None)
    
    '''Pawn promotion move to other than queen'''
    if position[2] == 64:
        return (deltamove(position[:2],(1,0),flipped=flipped),'r') #promote pawn moving forward to rook
    if position[2] == 65:
        return (deltamove(position[:2],(1,0),flipped=flipped),'b') #promote pawn moving forward to bishop
    if position[2] == 66:
        return (deltamove(position[:2],(1,0),flipped=flipped),'n') #promote pawn moving forward to knight
    if position[2] == 67:
        return (deltamove(position[:2],(1,-1),flipped=flipped),'r') #promote pawn capturing left to rook
    if position[2] == 68:
        return (deltamove(position[:2],(1,-1),flipped=flipped),'b') #promote pawn capturing left to bishop
    if position[2] == 69:
        return (deltamove(position[:2],(1,-1),flipped=flipped),'n') #promote pawn capturing left to knight
    if position[2] == 70:
        return (deltamove(position[:2],(1,1),flipped=flipped),'r') #promote pawn capturing right to rook
    if position[2] == 71:
        return (deltamove(position[:2],(1,1),flipped=flipped),'b') #promote pawn capturing right to bishop
    if position[2] == 72:
        return (deltamove(position[:2],(1,1),flipped=flipped),'n') #promote pawn capturing right forward to knight

'''TODO: Replace inefficient argsort with a filtered argmax approach'''
'''TODO: Handle Case where argsort sorts identical prob values in order of appearance'''

def idx2position(idx :int) -> tuple:
    '''Converts index of flattened policy array to tuple of coordinates.'''
    
    depth = int(idx/64)
    field = idx-depth*64
    x = int(field/8)
    y = int(field-x*8)
    
    return (x,y,depth)

def policy_argmax_choice(policy_tensor : np.ndarray, legal_moves : list,flipped=False) -> str:
    '''Finds the first legal move with highest likelyhood in the policy output tensor numpy array with shape=(8,8,73).'''

    '''Argsorts policy_tensor'''
    moves = np.squeeze(policy_tensor,axis=0)
    moves = moves.flatten()
    moves = np.argsort(moves)[::-1]
    
    '''Iterate over all sorted moves until one is legal'''
    for i in range(len(moves)):
        try:
            move = decode_move(idx2position(moves[i]), flipped=flipped)
            if move[1] == None:
                choice_uci = move[0]
            elif move[1] in ['r','b','n']:
                choice_uci = move[0]+move[1]
            elif move[1] == 'q':
                '''If a queen promotion is in the legal moves on this field, then, since only once piece can occupy it, it must be the move'''
                if move[0]+move[1] in legal_moves:
                    return move[0]+move[1]
                else:
                    choice_uci = move[0]
            if choice_uci in legal_moves:
                return choice_uci
        except IllegalMoveError:
            continue
        except Exception as E:
            print(e)
    
'''TODO: Implement optimized argmax search in policy space'''

# def policy_argmax_choice(policy_tensor : np.ndarray, legal_moves : list, flipped=False) -> str:
#     '''Finds the first legal move with highest likelyhood in the policy output tensor numpy array.'''
    
#     '''Restricts search only to legal moves'''
#     searchspace = []
#     for move in legal_moves:
#         searchspace += [policy_tensor[get_policy_index(move,flipped=flipped)]]
    
#     '''Find index of most likely move of legal_moves'''
#     argmax = np.argmax(searchspace)
    
#     return legal_moves[argmax]

# def get_policy_index(move : str,flipped=False) -> tuple:
#     '''ADD'''
    
#     '''Dictionary to map char part of uci to index number for flipped and unflipped case'''
#     char2idx = None
#     if flipped:
#         char2idx = {'a' : 7,'b' : 6,'c' : 5,'d' : 4,'e' : 3,'f' : 2,'g' : 1,'h' : 0}
#     else:
#         char2idx = {'a' : 0,'b' : 1,'c' : 2,'d' : 3,'e' : 4,'f' : 5,'g' : 6,'h' : 7}
        
#     '''Calculate fields to calculate origin and delta'''
#     fields = ((int(move[1])-1,char2idx[move[0]]),(int(move[3])-1,char2idx[move[2]])) # char and num are swapped in board encoding due to numpy array structure
    
#     '''Calculate origin and delta'''
#     delta = (fields[1][0]-fields[0][0],fields[1][1]-fields[0][1])
#     if flipped:
#         delta = (delta[0],-delta[1]) #in flipped case invert the second dimension
#     origin = fields[0]
    
#     '''Limit search in case of promotional or knightmove'''
#     '''If the last character in the uci move string is a b, r, or n, then it is a promotional string'''
#     if move[-1] in ['b','r','n']:
#         if delta[1] == 0: #the pawn moves upwards
#             if move[-1] == 'r':
#                 return (origin[0],origin[1],64)
#             if move[-1] == 'b':
#                 return (origin[0],origin[1],65)
#             if move[-1] == 'n':
#                 return (origin[0],origin[1],66)
#         if delta[1] == -1:
#             if move[-1] == 'r':
#                 return (origin[0],origin[1],67)
#             if move[-1] == 'b':
#                 return (origin[0],origin[1],68)
#             if move[-1] == 'n':
#                 return (origin[0],origin[1],69)
#         if move[-1] == 'r':
#                 return (origin[0],origin[1],70)
#         if move[-1] == 'b':
#                 return (origin[0],origin[1],71)
#         if move[-1] == 'n':
#                 return (origin[0],origin[1],72)
    
    
        
    
#     return fields