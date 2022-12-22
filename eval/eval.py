import numpy as np

from stockfish import Stockfish
from engine import ChessGame

class StockfishModel:
    '''
    ADD
    '''
    
    def __init__(self, depth=20, hash_size=2048, threads=6):
        '''
        ADD
        '''
        
        path = open('./stockfish_path.txt').read()
        self.stockfish = Stockfish(path, depth=depth)
        self.stockfish.update_engine_parameters({"Hash": hash_size, "Threads": threads})
    
    def play_next(self, game) -> str:
        '''
        ADD
        '''
        
        self.stockfish.set_fen_position(game.board.fen())
        move = self.stockfish.get_best_move()
        game.select_move(move)
        
        return move
    
#     def get_approximate_value(self, game) -> float:
#         '''
#         ADD
#         '''
        
#         stats = self.stockfish.get_wdl_stats()
        
#         winrate = np.divide(stats[0],np.sum(stats))
#         drawrate = np.divide(stats[1],np.sum(stats))
#         loserate = np.divide(stats[2],np.sum(stats))
        
#         stats = winrate - loserate
        
#         if stats > 0:
#             delta = None
        
#         elif stats < 0:
#             pass
        
#         else:
#             return 0
        
#         return winrate - loserate


def model_vs_stockfish(model, setup=None, move_chain=[]):
    '''
    ADD
    '''
    
    game = ChessGame(initial_moves=move_chain)
    stockfish = StockfishModel()
    
    move_list = move_chain
    
    stockfish_white = True
    
    if np.random.random() > 0.5:
        stockfish_white = False
        
    while not game.terminal:
        turn = game.board.turn
        
        if turn == stockfish_white:
            move_list += [stockfish.play_next(game)]
            
        else:
            input_tensor = game.get_input_tensor()
            policy, value = model(input_tensor)
            move = policy_argmax_choice(policy.numpy(), game.possible_moves, flipped=not game.board.turn)
            move_list += [move]
        
        result = game.get_result()
        
        return move_list, result