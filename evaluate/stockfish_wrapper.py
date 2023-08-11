import numpy as np

from stockfish import Stockfish

from lib.engine import ChessGame

class StockfishModel:
    '''
    ADD
    '''
    
    def __init__(self, path='./stockfish_15.1_win_x64_avx2/stockfish_15_x64_avx2.exe',depth=10, hash_size=2**12, threads=6, elo=1350, movetime=60000):
        '''
        ADD
        '''
        
        self.stockfish = Stockfish(path, depth=depth)
        self.movetime = movetime
        self.stockfish.update_engine_parameters({"Hash" : hash_size, "Threads" : threads, "UCI_LimitStrength" : "true", "UCI_Elo": elo})
    
    def play_next_move(self, game : ChessGame) -> str:
        '''
        ADD
        '''
        
        self.stockfish.set_fen_position(game.board.fen())
        
        move = self.stockfish.get_best_move(wtime=self.movetime, btime=self.movetime)
        game.select_move(move)
        
        return move
    
    def evaluate_pos(self, game : ChessGame) -> float:
        '''
        ADD
        '''
        
        self.stockfish.set_fen_position(game.board.fen())
        stats = self.stockfish.get_wdl_stats()
        
        if stats == None:
            return None
        
        return float(np.mean([1]*stats[0]+[0]*stats[1]+[-1]*stats[2]))
    
    def get_top_k_moves(self, game : ChessGame, k=3) -> list:
        '''
        ADD
        '''
        
        self.stockfish.set_fen_position(game.board.fen())
        
        moves = self.stockfish.get_top_moves(k)
        
        if moves == None:
            return []
        
        return [move['Move'] for move in moves]
    
    def params(self):
        return self.stockfish.get_parameters()