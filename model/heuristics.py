import chess
import numpy
from multiprocessing.pool import ThreadPool

class EvaluationFunction:
    def __init__(self, weights : list, piece_weights=[0.4,1,1,2,3.5]):
        '''
        ADD
        '''
        
        self.weights = weights
        self.piece_weights = piece_weights
    
    def __call__(self, game):
        '''
        ADD
        '''
        
        parent_board = game.board.epd()
        color = not game.board.turn
        
        args = [(parent_board, move) for move in game.possible_moves]
        children_boards = ThreadPool(len(game.possible_moves)).map(self.create_epd, args)
        
        args = [(children_board, color) for children_board in children_boards]
        scores = ThreadPool(len(game.possible_moves)).map(self.evaluate_position, args)
        
        return game.possible_moves[np.argmax(scores)]
    
    def create_epd(parent_epd : str, move : str) -> str:
        '''
        ADD
        '''
        
        board = chess.Board.from_epd(parent_epd)[0]
        board.push(chess.Move.from_uci(move))
        
        return board.epd()
    
    def evaluate_position(pos : str, color : bool) -> float:
        '''
        ADD
        '''
        
        score = piece_score(pos, color) * self.weights[0]
        score += control_score(pos, color) * self.weights[1]
        score += volatility_score(pos, color) * self.weights[2]
        score += offense_score(pos, color) * self.weights[3]
        score += defense_score(pos, color) * self.weights[4]
        
        return score
    
    def piece_score(pos : str, color : bool) -> float:
        '''
        ADD
        '''
        
        board = chess.Board.from_epd(pos)[0]
        
        score = len(board.pieces(chess.PAWN, color)) * self.piece_weights[0]
        score += len(board.pieces(chess.KNIGHT, color)) * self.piece_weights[1]
        score += len(board.pieces(chess.BISHOP, color)) * self.piece_weights[2]
        score += len(board.pieces(chess.ROOK, color)) * self.piece_weights[3]
        score += len(board.pieces(chess.QUEEN, color)) * self.piece_weights[4]
        
        score -= len(board.pieces(chess.PAWN, not color)) * self.piece_weights[0]
        score -= len(board.pieces(chess.KNIGHT, not color)) * self.piece_weights[1]
        score -= len(board.pieces(chess.BISHOP, not color)) * self.piece_weights[2]
        score -= len(board.pieces(chess.ROOK, not color)) * self.piece_weights[3]
        score -= len(board.pieces(chess.QUEEN, not color)) * self.piece_weights[4]
        
        return score
    
    def control_score(pos : str, color : bool) -> float:
        '''
        ADD
        '''
        
        pass
    
    def volatility_score(pos : str, color : bool) -> float:
        '''
        ADD
        '''
        
        pass
    
    def offense_score(pos : str, color : bool) -> float:
        '''
        ADD
        '''
        
        pass
    
    def defense_score(pos : str, color : bool) -> float:
        '''
        ADD
        '''
        
        pass