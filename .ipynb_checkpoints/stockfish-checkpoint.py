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
    
    def get_approximate_value(self, game) -> float:
        '''
        ADD
        '''
        
        stats = self.stockfish.get_wdl_stats()
        
params = {
    "Debug Log File": "",
    "Contempt": 0,
    "Min Split Depth": 0,
    "Threads": 1, # More threads will make the engine stronger, but should be kept at less than the number of logical processors on your computer.
    "Ponder": "false",
    "Hash": 16, # Default size is 16 MB. It's recommended that you increase this value, but keep it as some power of 2. E.g., if you're fine using 2 GB of RAM, set Hash to 2048 (11th power of 2).
    "MultiPV": 1,
    "Skill Level": 20,
    "Move Overhead": 10,
    "Minimum Thinking Time": 20,
    "Slow Mover": 100,
    "UCI_Chess960": "false",
    "UCI_LimitStrength": "false",
    "UCI_Elo": 1350
}

# stockfish.set_position(['e2e4','e7e6']) # to set a position from initial board as move chain

# stockfish.get_top_moves(4)

# stockfish.get_wdl_stats()

# stockfish.make_moves_from_current_position(["d2d4"])

# stockfish.set_elo_rating(number)

# stockfish.set_skill_level(number) # 0-20

# help(stockfish.set_elo_rating)

# stockfish.get_parameters()

# stockfish.get_fen_position()