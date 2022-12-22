import numpy as np
import tensorflow as tf
import chess

from copy import deepcopy
from lib.engine import ChessGame
from lib.utils import policy_argmax_choice
from model.model import ChessModel

def random_simulation(game : ChessGame) -> str:
    '''
    ADD
    '''
    
    return game.possible_moves[np.random.choice(range(len(game.possible_moves)))]

def greedy_lossavoid_simulation(game : ChessGame) -> str:
    '''
    ADD
    '''
    
    successors = [(move, deepcopy(game)) for move in game.possible_moves]
    
    for move, state in successors:
        state.select_move(move)
        
    if game.board.turn:
        successors = [(move, state.terminal, state.result == (1,0)) for move, state in successors]
    else:
        successors = [(move, state.terminal, state.result == (0,1)) for move, state in successors]
        
    for move, terminal, win in successors:
        if terminal and win:
            return move
        
    successors = [move for move, terminal, win in successors if terminal == False]
    
    if len(successors) > 1:
        return successors[np.random.choice(range(len(successors)))]
    
    elif len(successors) == 1:
        return successors[0]
    
    else:
        return game.possible_moves[np.random.choice(range(len(game.possible_moves)))]
    
def lossavoid_simulation(game: ChessGame) -> str:
    '''
    ADD
    '''
    
    moves = game.possible_moves
    
    #add
    while len(moves) > 1:
        #add
        move = game.possible_moves[np.random.choice(range(len(moves)))]
        
        #add
        successor = deepcopy(game.board)
        successor.push(chess.Move.from_uci(move))
        
        #add
        if successor.is_checkmate():
            #add
            if successor.result() == '1-0' and successor.turn:
                moves.remove(move)
            
            elif successor.result() == '0-1' and not successor.turn:
                moves.remove(move)
            
        else:
            return move
    
    #add
    return moves[0]

def heuristic_simulation(game : ChessGame) -> str:
    '''
    ADD
    '''
    
    choice = None
    
    return game.possible_moves[choice]

def cnn_simulation(game : ChessGame, model : ChessModel) -> str:
    '''
    ADD
    '''
    
    policy = model(game.get_input_tensor())
    
    return policy_argmax_choice(np.squeeze(policy[1].numpy(),axis=0), game.possible_moves, flipped=not game.board.turn)