import numpy as np
import tensorflow as tf

class IllegalMoveError(Exception):
    '''
    IllegalMoveError Exception subclass for error handling.
    Used for returning which moves could not be played and for catching and handling these errors.
    '''
    
    def __init__(self, *args):
        '''
        Constructor of the IllegalMoveError instance.
        May receive zero to two arguments which are convertable to string format.
        
        :param args: Either None, a move, or a move and a board position.
        '''
        
        self.args = args
    
    def __str__(self):
        '''
        The string conversion method for this error object.
        
        :returns (str): Description of the error taking into accoun the arguments from the instance.
        '''
        
        if self.args != None:
            if len(self.args) == 2:
                return f'The selected chess move {self.args[0]} is illegal in the current chess position {self.args[1]}.'
            
            elif len(self.args) == 1:
                return f'The selected chess move {self.args[0]} is illegal in the current chess position.'
        
        return f'The selected chess move is illegal in the current chess position.'
    
class MoveNotInEmbeddingError(Exception):
    '''
    MoveNotInEmbeddingError Exception subclass for error handling.
    Used for instances in which a conversion function fails to convert move from one encoding to another.
    '''
    
    def __init__(self, *args):
        '''
        Constructor of the MoveNotInEmbeddingError instance.
        May receive zero to exactly three arguments which are convertable to string format.
        In case this criterion is not met a general error message will be used instead.
        
        Example:
        MoveNotInEmbeddingError('uci-encoded','policy space','a2b8') -> raises 'The uci-encoded move a2b8 does not have an equivalent in policy space.'.
        
        :param args: Either None or a move encoding name plus a embedding to be converted to name and the move.
        '''
        
        self.args = args
        
    def __str__(self):
        '''
        The string conversion method for this error object.
        
        :returns (str): Description of the error taking into accoun the arguments from the instance.
        '''
        
        if self.args != None and len(self.args) == 3:
            return f'The {self.args[0]} move {self.args[2]} does not have an equivalent in {self.args[1]}.'
        
        else:
            return f'The move from current embedding has no equivalent in target embedding.'

def idx2field(index : tuple, flipped=False) -> str:
    '''
    Receives a tuple for a field coordinate and turns it into the field's string.
    
    :param index (tuple): A tuple of two integers from 0-7 for the field coordinates.
    :param flipped (bool): A boolean required for the uci move translation when the embedding is for a flipped board, meaning black is to move.
    
    :returns (str): The string representing the field from the index tuple.
    '''
    
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
    
    #if the board has to be flipped we flip it horizontally.
    if flipped:
        board = np.flip(board,axis=0)
        
    return board[index]

def deltamove(origin : tuple, delta : tuple, flipped=False) -> str:
    '''
    Receives an origin and delta tuple and returns the uci move string for the two fields.
    Origin and delta tuple both contain only integers from 0-7 for the respective column and row.
    
    :param origin (tuple): A tuple of two integers between 0 and 7.
    :param delta (tuple): A tuple of two integers representing how much to move in row and colum direction; integers may be negative.
    :param flipped (bool): A boolean required for the uci move translation when the embedding is for a flipped board, meaning black is to move.
    
    :returns (str): The uci-encoded move from the position from origin to delta.
    '''
    
    target = None
    if flipped:
        target = (origin[0]+delta[0],origin[1]-delta[1]) #the delta for the char-axis is inverted so each side's moves are mapped identically on a flipped board.
        
    else:
        target = (origin[0]+delta[0],origin[1]+delta[1])
    
    #moves that would surpass the boarder are illegal
    if min(target) < 0 or max(target) > 7:
        raise IllegalMoveError() #no error arguments since the move does not exist on any chess board.
        
    return idx2field(origin,flipped=flipped) + idx2field(target,flipped=flipped)

def decode_move(position : tuple, flipped=False) -> tuple:
    '''
    Decodes a position in the policy embedding into a uci-encoded move.
    Will return the uci encoded move for the piece, taking into account board mirroring horizontally for black if flipped.
    Also the promotion choice for pawns will be returned as either 'q', 'r', 'n', or 'b', alternatively None for a knightmove.
    
    Example usage:
    decode_move((3,3,49),flipped=True) -> ('d5e4','q') piece move from d5 to e4; queen move in encoding but necessarily queen as a piece.
    decode_move((5,4,56)) -> ('e6f8',None) knight move from e6 to f8.
    decode_move((0,0,67)) -> IllegalMoveError() 'The selected chess move is illegal in the current chess position.'.
    decode_move((6,3,67)) -> ('d7c8','r') white pawn move from d7 to c8 promoting into a rook.
    
    :param position (tuple): A tuple of size 3 of the position of the policy-embedded move (x,y,d).
    :param flipped (bool): A boolean required for the uci move translation in case the move has to be flipped at the end of translation.
    
    :returns (tuple): The move in uci format and the promotional choice for pawns as string (move,promo).
    '''
    
    #queen moves for the 0-55 feature dimensions of the policy tensor.
    if position[2] <= 55:
        direction = int(position[2]/7) #0-7 direction to move from origin field (N,NE,E,SE,S,SW,W,NW).
        
        distance = int(position[2]-direction*7)+1 #1-7 distance to move from origin field (1,2,3,4,5,6,7).
        
        if direction == 0:
            return (deltamove(position[:2],(distance,0),flipped=flipped),'q') #since this can include pawn promotions this embedding includes the queen promotions possible too.
        
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
    
    #knight moves for the 56-63 feature dimensions of the policy tensor.
    if position[2] <= 63:
        direction = position[2]-56 #direction 0-7 for all eight options for a knight to move.
        
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
    
    #pawn promotion moves to another piece than a queen
    if position[2] == 64:
        return (deltamove(position[:2],(1,0),flipped=flipped),'r') #promote pawn moving forward to rook..
    
    if position[2] == 65:
        return (deltamove(position[:2],(1,0),flipped=flipped),'b') #promote pawn moving forward to bishop.
    
    if position[2] == 66:
        return (deltamove(position[:2],(1,0),flipped=flipped),'n') #promote pawn moving forward to knight.
    
    if position[2] == 67:
        return (deltamove(position[:2],(1,-1),flipped=flipped),'r') #promote pawn capturing left to rook.
    
    if position[2] == 68:
        return (deltamove(position[:2],(1,-1),flipped=flipped),'b') #promote pawn capturing left to bishop.
    
    if position[2] == 69:
        return (deltamove(position[:2],(1,-1),flipped=flipped),'n') #promote pawn capturing left to knight.
    
    if position[2] == 70:
        return (deltamove(position[:2],(1,1),flipped=flipped),'r') #promote pawn capturing right to rook.
    
    if position[2] == 71:
        return (deltamove(position[:2],(1,1),flipped=flipped),'b') #promote pawn capturing right to bishop.
    
    if position[2] == 72:
        return (deltamove(position[:2],(1,1),flipped=flipped),'n') #promote pawn capturing right forward to knight.

def idx2position(idx : int) -> tuple:
    '''
    Converts index of flattened policy array to tuple of coordinates.
    
    :param idx (int): An index in the flattened policy embedding.
    
    :returns (tuple): A tuple (x,y,d) for the positional and feature dimensions the index corresponds to in the stacked-up policy embedding.
    '''
    
    depth = int(idx/64)
    field = idx-depth*64
    x = int(field/8)
    y = int(field-x*8)
    
    return (x,y,depth)

def get_policy_index(move : str) -> tuple:
    '''
    Converts an uci move to a coordinate tuple from the policy embedding.
    Used to convert uci to policy embedding to enable masking for legal moves in the policy embedding.
    
    :param move (str): A move string representing the move in uci format.
    
    :returns (tuple): A tuple of format (x,y,d) with x and y ranging from 0-7, and d ranging from 0-63; representing the coordinates in unflattened policy embedding.
    '''
    
    #dictionary to map char part of uci to index number
    char2idx = {'a' : 0,'b' : 1,'c' : 2,'d' : 3,'e' : 4,'f' : 5,'g' : 6,'h' : 7}
        
    #calculate field coordinates for both fields in the uci-embedded move.
    fields = ((int(move[1])-1,char2idx[move[0]]),(int(move[3])-1,char2idx[move[2]])) #char and num are swapped in board encoding due to numpy array structure.
    
    '''Calculate origin and delta'''
    origin = fields[0]
    
    delta = (fields[1][0]-fields[0][0],fields[1][1]-fields[0][1])
    
    #Limit search in case of promotional move or knight move.
    #If the last character in the uci move string is a b, r, or n, then it is a promotional string.
    if move[-1] in ['b','r','n']:
        #the pawn moves upwards
        if delta[1] == 0: 
            if move[-1] == 'r':
                return (origin[0],origin[1],64)
            
            if move[-1] == 'b':
                return (origin[0],origin[1],65)
            
            if move[-1] == 'n':
                return (origin[0],origin[1],66)
        
        #the pawn moves diagonally to the left
        elif delta[1] == -1: 
            if move[-1] == 'r':
                return (origin[0],origin[1],67)
            
            if move[-1] == 'b':
                return (origin[0],origin[1],68)
            
            if move[-1] == 'n':
                return (origin[0],origin[1],69)
        
        #the pawn moves diagonally to the right
        else:
            if move[-1] == 'r':
                return (origin[0],origin[1],70)
            
            if move[-1] == 'b':
                return (origin[0],origin[1],71)
            
            if move[-1] == 'n':
                return (origin[0],origin[1],72)
            
    #if the absolute values for delta are (2,1) or (1,2) then it is a knight move.
    elif (abs(delta[0]) == 2 and abs(delta[1]) == 1) or (abs(delta[0]) == 1 and abs(delta[1]) == 2):
        if delta == (2,1):
            return (origin[0],origin[1],56)
        
        if delta == (1,2):
            return (origin[0],origin[1],57)
        
        if delta == (-1,2):
            return (origin[0],origin[1],58)
        
        if delta == (-2,1):
            return (origin[0],origin[1],59)
        
        if delta == (-2,-1):
            return (origin[0],origin[1],60)
        
        if delta == (-1,-2):
            return (origin[0],origin[1],61)
        
        if delta == (1,-2):
            return (origin[0],origin[1],62)
        
        if delta == (2,-1):
            return (origin[0],origin[1],63)
    
    #in all other cases it is a queen move
    else:
        distance = np.max([delta[0],delta[1]])-1 ##1-7 distance to move from origin field (1,2,3,4,5,6,7), here minus one since we are dealing with indiced (0-6).
        
        direction = 0 #0-7 direction to move from origin field (N,NE,E,SE,S,SW,W,NW).
        if delta[0] > 0:
            if delta[1] > 0:
                direction = 7 #NW
            
            elif delta[1] == 0:
                direction = 0 #N
            
            else:
                direction = 1 #NE
        
        if delta[0] == 0:
            if delta[1] > 0:
                direction = 6 #W
            
            else:
                direction = 2 #E
        
        if delta[0] < 0:
            if delta[1] > 0:
                direction = 5 #SW
            
            elif delta[1] == 0:
                direction = 4 #S
            
            else:
                direction = 3 #SE
                
        return (origin[0],origin[1],direction*7+distance)
    
    raise MoveNotInEmbeddingError('uci-encoded', 'policy space', move)

def flip_uci_move(move : str) -> str:
    '''
    ADD
    '''
    
    flip_dict = {'1' : '8', '2' : '7', '3' : '6', '4' : '5', '5' : '4', '6' : '3', '7' : '2', '8' : '1'}
    
    return move[0]+flip_dict[move[1]]+move[2]+flip_dict[move[3]]+move[4:] #we use slicing to not cause issues with promotional moves
    
def policy_argmax_choice(policy_tensor : np.ndarray, legal_moves : list, flipped=False) -> str:
    '''
    Finds the legal move with highest likelyhood in the numpy array from the policy tensor.
    Since efficiency is key, and a tie is highly unlikely in a well trained network, ties are not considered.
    In case two moves have an equal probability the one with the lower index in the legal moves will be chosen by np.argmax.
    
    :param policy_tensor (np.ndarray): The numpy array of the policy tensor, unflattened with shape (8,8,64).
    :param legal_moves (list): A list of all the uci-encoded legal moves taken from the current position for masking the policy space.
    :param flipped (bool): A boolean whether legal moves have to be flipped before checking in the tensor; meaning the model output is for black.
    
    :returns (str): Returns the uci-encoded move which corresponds to the highest likelihood in the policy embedding.
    '''
    
    #restricts search only to legal moves.
    searchspace = []
    if flipped:
        for move in legal_moves:
            searchspace += [policy_tensor[get_policy_index(flip_uci_move(move))]] #adds to the searchspace the probability of each move in the legal move list flipped horizontally
            
    else:
        for move in legal_moves:
            searchspace += [policy_tensor[get_policy_index(move)]] #adds to the searchspace the probability of each move in the legal move list
            
    #find the index for the legal moves with the highest probability in the policy tensor.
    argmax = np.argmax(searchspace)
    
    return legal_moves[argmax]

def int2move(move : int) -> str:
    '''
    ADD
    '''
    
    int2char = {'1' : 'a', '2' : 'b', '3' : 'c', '4' : 'd', '5' : 'e', '6' : 'f', '7' : 'g', '8' : 'h'}
    move = str(move)
    move = int2char[move[0]]+move[1]+int2char[move[2]]+move[3]+move[4]
    
    if move[4] == '0':
        move = move[:4]
        
    else:
        move = move[:4]+{'1' : 'q', '2' : 'r', '3' : 'n', '4' : 'b'}[move[4]]
    
    return move

def move2int(move : str) -> int:
    '''
    ADD
    '''
    
    char2int = {'a' : '1', 'b' : '2', 'c' : '3', 'd' : '4', 'e' : '5', 'f' : '6', 'g' : '7', 'h' : '8'}
    move = char2int[move[0]]+move[1]+char2int[move[2]]+move[3:]
    
    if move[-1].isnumeric():
        move = move+'0'
        
    else:
        move = move[:4]+{'q' : '1', 'r' : '2', 'n' : '3', 'b' : '4'}[move[-1]]
    
    return int(move)

def get_padded_move_probabilities(legal_moves : list, move_probs : list, flipped=False):
    '''
    ADD
    '''
    
    move_probs_tensor = np.zeros((8,8,73))
    
    #add
    if flipped:
        for move, prob in zip(legal_moves, move_probs):
            move_probs_tensor[get_policy_index(flip_uci_move(move))] = prob
    else:
        for move, prob in zip(legal_moves, move_probs):
            move_probs_tensor[get_policy_index(move)] = prob
    
    #shape and type casting and returning
    return tf.constant(move_probs_tensor,dtype='float32')