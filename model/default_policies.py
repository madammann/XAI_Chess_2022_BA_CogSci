import random
import chess

import numpy as np
import tensorflow as tf

from copy import deepcopy

from lib.engine import ChessGame
from lib.utils import policy_argmax_choice, policy_probabilities

from model.model import ChessModel
from model.heuristics import heuristic_eval

from multiprocessing.pool import ThreadPool

def random_simulation(game : ChessGame) -> str:
    '''
    Function selects a random move out of all possible moves of a ChessGame instance.
    
    :param game (ChessGame): The ChessGame instance with possible moves attribute.
    
    :returns str: The uci-encoded move string that has been chosen.
    '''
    
    return game.possible_moves[np.random.choice(range(len(game.possible_moves)))]

def lossavoid_simulation(game: ChessGame) -> str:
    '''
    Function selects a move out of all possible moves of a ChessGame instance using loss avoidance.
    Loss avoidance picks a random move and checks if it is losing, if it is not it is chosen, until just one move remains or a move is chosen.
    
    :param game (ChessGame): The ChessGame instance with possible moves attribute.
    
    :returns str: The uci-encoded move string that has been chosen.
    '''
    
    moves = game.possible_moves
    
    while len(moves) > 1:
        move = game.possible_moves[np.random.choice(range(len(moves)))]
        
        successor = deepcopy(game.board)
        successor.push(chess.Move.from_uci(move))
        
        if successor.is_checkmate():
            if successor.result() == '1-0' and successor.turn:
                moves.remove(move)
            
            elif successor.result() == '0-1' and not successor.turn:
                moves.remove(move)
            
        else:
            return move
    
    return moves[0]

def greedy_lossavoid_simulation(game : ChessGame) -> str:
    '''
    Function selects a move out of all possible moves of a ChessGame instance using greedy loss avoidance.
    Greedy loss avoidance picks a checkmate move when available and otherwise a random move which does not lose, if available.
    
    :param game (ChessGame): The ChessGame instance with possible moves attribute.
    
    :returns str: The uci-encoded move string that has been chosen.
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

def cnn_simulation(game : ChessGame, model : ChessModel) -> str:
    '''
    Function selects a move out of all possible moves of a ChessGame instance using a provided model.
    The model receives the input tensor, returning a policy and value, then, according to the policy, a move is picked by the argmax over all possible moves.
    
    :param game (ChessGame): The ChessGame instance with possible moves attribute.
    
    :returns str: The uci-encoded move string that has been chosen.
    '''
    
    val, policy = model(game.get_input_tensor())
    
    return policy_argmax_choice(np.squeeze(policy.numpy(),axis=0), game.possible_moves, flipped=not game.board.turn)

def biased_cnn_simulation(game : ChessGame, model : ChessModel, bias_factor=0.65, epsilon=0.2) -> str:
    '''
    Function selects a move out of all possible moves of a ChessGame instance using a provided model.
    The model receives the input tensor, returning a policy and value, then, according to the policy, a move is picked by the argmax over all possible moves.
    
    Additionally:
    50% chance to take a piece which moved into an attackable position
    
    :param game (ChessGame): The ChessGame instance with possible moves attribute.
    
    :returns str: The uci-encoded move string that has been chosen.
    '''
    
    if len(game.possible_moves) == 1:
        return game.possible_moves[0]
    
    #firstly get the probability distribution from the model for all legal moves
    val, policy = model(game.get_input_tensor())
    model_probs = np.array(policy_probabilities(np.squeeze(policy.numpy(),axis=0), game.possible_moves, flipped=not game.board.turn))
    
    #retrieve and reformat the probability distribution based on a custom biased heuristic
    heuristic_probs = ThreadPool(len(game.possible_moves)).starmap(heuristic_eval,[(move,game.board) for move in game.possible_moves])
    heuristic_probs = np.array(heuristic_probs,dtype='float32')
    heuristic_probs = heuristic_probs.reshape((heuristic_probs.shape[0],)) / np.sum(heuristic_probs)
    
    #weight based on bias factor and then create a combined probability distribution
    total_probs = np.divide(
        np.add(np.multiply(model_probs,(1-bias_factor)),np.multiply(heuristic_probs,bias_factor)),
        np.sum(np.add(np.multiply(model_probs,(1-bias_factor)),np.multiply(heuristic_probs,bias_factor)))
    )
    
    #make a decision as a weighted random choice in epsilon cases
    if random.random() <= epsilon:
        p = np.nan_to_num(total_probs, copy=False, nan=1e-5, posinf=1e-5, neginf=1e-5)
        p = p/np.sum(p)
        choice = np.random.choice(np.arange(len(game.possible_moves)),p=p)
        return game.possible_moves[choice]
    
    else:
        return game.possible_moves[np.argmax(np.nan_to_num(total_probs, copy=False, nan=1e-5, posinf=1e-5, neginf=1e-5))]