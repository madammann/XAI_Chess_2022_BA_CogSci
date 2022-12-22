import chess
import tensorflow as tf
import numpy as np

from lib.utils import IllegalMoveError

class ChessGame:
    '''
    ChessGame base class which is designed for MCTS efficiency primarily.
    This class utilizes the chess.Board class from the python chess module.
    Additionally it is equipped with all necessary components for using it as an environment.
    
    ADD
    ADD note about board tensor efficiency.
    
    :att board (chess.Board): The chess.Board instance used for playing, simulating and retrieving state information.
    :att T: The sequence length for the observation tensor to be generated.
    :att terminal (bool): A boolean as to whether the game is terminal; does not take into account guaranteed lost positions as terminal, only final states.
    :att result: The result of the ongoing chess game, None if not yet decided, else a tuple with either (1,0), (0,0), or (0,1) value.
    :att opponent_rep (int): The repetition count of the opponent for their last position, used for tensor computation only.
    :att possible_moves (list): A list of all possible legal moves in current position in uci-encoding.
    :att tensor_history (list): A list of the last up to T board tensors, used as a memory attribute for tensor generation only.
    '''
    
    def __init__(self, T=8, initial_moves=[]):
        '''
        ADD
        '''
        
        self.board = chess.Board()
        self.T = T
        self.terminal = False
        self.result = None
        self.opponent_rep = 0
        
        if initial_moves == []:
            self.possible_moves = [move.uci() for move in self.board.legal_moves]
            self.tensor_history = [self.board_tensor()]
            
        else:
            self.tensor_history = [self.board_tensor()]
            self.play_move_chain(initial_moves)
            self.possible_moves = [move.uci() for move in self.board.legal_moves]
    
    def select_move(self, move_uci : str, history=True):
        '''
        Push move assuming only legal uci-encoded moves are pushed.
        
        ADD note about illegal uci and illegal non-uci moves.
        
        ADD note about this  being a function only for training, the legal check applies only for non training scenario.
        
        ADD note about 'offer draw' and 'claim draw' moves.
        
        ADD note about claim draw being the only training-allowed option.
        
        ADD MORE LATER
        '''
        
        if not self.terminal:
            #a try statement is used here since this is the most efficient way to handle the rare claim draw case and potential errors
            try:
                self.board.push(chess.Move.from_uci(move_uci))
            
            #in the claim draw case a ValueError will be raised and catched here
            except ValueError:
                if move_uci == 'claim draw':
                    #case 1 - a draw may be claimed by either player if the current position had 50 moves without progress
                    if self.board.is_fifty_moves():
                        self.terminal = True
                        self.result = (0,0)
                    
                    #case 2 - a draw may be claimed by either player if the current position has occured three or more times (here three to four since five is forced draw)
                    elif self.board.is_repetition(3) or self.board.is_repetition(4):
                        self.terminal = True
                        self.result = (0,0)
                        
                #error case for non 'claim draw' string or not to chess.Move convertible passed move_uci
                else:
                    raise IllegalMoveError(move_uci,self.board.fen())
                    
            #error case for moves not in uci-format or moves which cannot be converted to chess.Move or moves which cannot be board.pushed.
            except:
                raise IllegalMoveError(move_uci,self.board.fen())
            
            #the game is only counted as terminal if one of the following rules can be applied, order is in expected liklihood
            #rule 1 - the game is in checkmate
            if self.board.is_checkmate():
                self.terminal = True
                
                #here if-else instead of if-elif since only two options are possible
                if self.board.result() == '1-0':
                    self.result = (1,0)
                
                else:
                    self.result = (0,1)
                
            #rule 2 - the game is in stalemate, meaning no moves for one player but also no check
            elif self.board.is_stalemate():
                self.terminal = True
                self.result = (0,0)
                
            #rule 3 - the game is a forced draw if only kings are left on the board, meaning there is insufficient material in all potentual continuations    
            elif not ('P' or 'p' or 'N' or 'n' or 'B' or 'b' or 'R' or 'r') in self.board.epd():
                self.terminal = True
                self.result = (0,0)

            #rule 4 - the game is a forced draw if 75 progressless moves were made, according to the "new" FIDE draw rules
            elif self.board.is_seventyfive_moves():
                self.terminal = True
                self.result = (0,0)
                
            #rule 5 - the game is a forced draw if there is a fivefold repetition according to the "new" FIDE draw rules
            elif self.board.is_fivefold_repetition():
                self.terminal = True
                self.result = (0,0)
        
            #designed to skip creating unecessary tensors in case of playing an initial move chain in self.__init__
            if history:
                #add current board tensor to the tensor history
                self.tensor_history += [self.board_tensor()]
            
                #delete tensors which exceed length of T
                self.tensor_history = self.tensor_history[-self.T:]
                
                #calculate next set of possible uci moves legal in the current board
                self.possible_moves = [move.uci() for move in self.board.legal_moves]
    
    def play_move_chain(self, moves : list):
        '''
        Plays all moves efficiently.
        ADD MORE LATER
        '''
        
        for i in range(len(moves)):
            if (len(moves) - i) > self.T:
                self.select_move(moves[i],history=False) #do not create unecessary tensors for elements not in the range of T
                
            else:
                self.select_move(moves[i])
    
    def board_tensor(self):
        '''
        Creates a tensor of the current board state which includes piece positions and repetition count.
        ADD MORE LATER
        '''
        
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
        
        #add the opponent's rep count and overwrite it with the current player's rep count thereafter
        tensor += [np.full((8,8,1),self.opponent_rep)]
        self.opponent_rep = rep
        
        #tack into one board tensor
        tensor = np.dstack(tensor)
        
        return tensor
    
    def rule_tensor(self):
        '''
        Creates a tensor of the current rule state which includes Castling rights, turn, and move counters.
        ADD MORE LATER
        '''
        
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
        '''
        Returns a tensor made out of the board tensor history and the current rule tensor for network input.
        ADD MORE LATER
        '''
        
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