import chess
import tensorflow as tf
import numpy as np

from pgn_parser import pgn as PGN
from pgn_parser import parser as Parser
#https://pypi.org/project/pgn-parser/

from engine import IllegalMoveError, MoveNotInEmbeddingError, ChessGame

# class ChessGame:
#     def __init__(self):
#         self.board = chess.Board()
#         self.possible_moves = [move.uci() for move in self.board.legal_moves]
#         self.board_history = []
#         self.move_history = [None]
#         self.repetitions = {''.join(self.board.fen().split(' ')[:-2]) : 0}
#         self.tensor_history = [self.board_tensor()] # already append the first element
#         self.png = PNG()

class ExtendedChessGame(ChessGame):
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
        
        super(ExtendedChessGame, self).__init__()
        
        self.pgn = None
        self.draw_offered = False
    
    def select_move(self, move_uci : str, history=True):
        '''
        ADD
        '''
        
        if self.draw_offered and move_uci == 'accept draw':
            self.terminal = True
            self.result = (0,0)
        
        elif self.draw_offered and move_uci == 'decline draw':
            pass
        
        else:
            pass
        
        if not move_uci in self.possible_moves+['claim draw','offer draw']:
            raise IllegalMoveError(move_uci,self.board.fen())
        
        if not self.terminal:
            #a try statement is used here since this is the most efficient way to handle the rare claim draw case and potential errors
            if move_uci in self.possible_moves:
                self.board.push(chess.Move.from_uci(move_uci))
            
            elif move_uci == 'claim draw':
                #case 1 - a draw may be claimed by either player if the current position had 50 moves without progress
                if self.board.is_fifty_moves():
                    self.terminal = True
                    self.result = (0,0)
                    
                #case 2 - a draw may be claimed by either player if the current position has occured three or more times (here three to four since five is forced draw)
                elif self.board.is_repetition(3) or self.board.is_repetition(4):
                    self.terminal = True
                    self.result = (0,0)
            
            elif move_uci == 'offer draw':
                self.draw_offered = True
            
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
        
        def render(self):
            '''
            ADD
            '''
            
            pass
        
        def load_pgn():
            '''
            ADD
            '''
            
            pass
        
        def save_pgn(self):
            '''
            ADD
            '''
            
            pass
        
        