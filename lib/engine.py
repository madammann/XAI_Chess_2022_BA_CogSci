import chess
import tensorflow as tf
import numpy as np

class IllegalMoveError(Exception):
    def __init__(self, string):
        pass

class PNG:
    def __init__(self):
        pass
    
class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.possible_moves = [move.uci() for move in self.board.legal_moves]
        self.board_history = []
        self.move_history = [None]
        self.repetitions = {''.join(self.board.fen().split(' ')[:-2]) : 0}
        self.tensor_history = [self.board_tensor()] # already append the first element
        self.png = PNG()
    
    def select_move(self, move_uci):
        if move_uci in self.possible_moves:
            self.board_history += [self.board.fen()]
            self.move_history += [move_uci]
            self.board.push(chess.Move.from_uci(move_uci))
            
            '''Create a new repitition key with value zero or increment it if already present'''
            rep_fen = ''.join(self.board.fen().split(' ')[:-2])
            if rep_fen in list(self.repetitions.keys()):
                self.repetitions[rep_fen] += 1
            else:
                self.repetitions[rep_fen] = 0
                
            '''Add current board tensor to the tensor history'''
            self.tensor_history += [self.board_tensor()]
        else:
            raise IllegalMoveError('Selected move is illegal on this board.')
            
    def get_repetition_count(self, key):
        if key in list(self.repetitions.keys()):
            return self.repetitions[key]
        else:
            return 0
    
    def board_tensor(self):
        tensor = []
        for color in [chess.WHITE,chess.BLACK]:
            for piece in [chess.PAWN,chess.ROOK,chess.KNIGHT,chess.BISHOP,chess.QUEEN,chess.KING]:
                tensor += [np.array(self.board.pieces(piece,color).tolist()).reshape(8,8,1)]
                
        '''Get the repetition count for the position from each color perspective'''
        fen_white,fen_black = None, None
        
        if self.board.turn == chess.WHITE:
            fen_white = self.board.fen().split(' ')[:-2]
            fen_black = [fen_white[0], 'b', fen_white[2], fen_white[3]]
        else:
            fen_black = self.board.fen().split(' ')[:-2]
            fen_white = [fen_black[0], 'w', fen_black[2], fen_black[3]]
            
        tensor += [np.full((8,8,1),self.get_repetition_count(''.join(fen_white)))]
        tensor += [np.full((8,8,1),self.get_repetition_count(''.join(fen_black)))]
        
        '''Stack into one board tensor'''
        tensor = np.dstack(tensor)
        
        '''Flip the board tensor horizontally if black is to move'''
        if self.board.turn == chess.BLACK:
            tensor = np.flip(tensor,axis=0)
        
        return tensor
    
    def rule_tensor(self):
        tensor = []
        
        '''Get the current turn encoding as plane of binary value'''
        tensor += [np.full((8,8,1),self.board.turn)]
        
        '''Get the total move count as plane with integer value'''
        tensor += [np.full((8,8,1),len(self.move_history))]
        
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
    
    def get_input_tensor(self,t=8):
        tensor = self.tensor_history[-t:] # get the last t board tensors which exist
        tensor += [self.rule_tensor()]
        tensor = np.dstack(tensor)
        return tf.constant(tensor,dtype='float32')