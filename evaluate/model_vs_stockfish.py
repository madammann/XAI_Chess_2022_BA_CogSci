import chess
import os
import random

import pandas as pd
import numpy as np

from evaluate.stockfish_wrapper import StockfishModel

from lib.engine import ChessGame
from lib.utils import policy_argmax_choice

from model.model import ChessModel
from model.default_policies import cnn_simulation
from model.mcts import MctsTree

def load_model(model_path):
    '''
    ADD
    '''

    model = ChessModel()
    model(ChessGame(T=8).get_input_tensor())
    model.load(model_path)

    return model

def playout(model, model_name, stockfish_path, stockfish_elo, mcts=True):
    '''
    ADD
    '''
    
    game = ChessGame(T=8)
    
    stockfish = StockfishModel(stockfish_path, elo=stockfish_elo)
    
    starter = random.random() <= 0.5
    
    moves = []
    while not game.terminal:
        if game.board.turn == starter:
            #model turn
            if mcts:
                tree = MctsTree(game)
                simul_func = lambda game: cnn_simulation(game, model)
                move_probs = tree(simul_func,simulation_limit=30)
                move = tree.bestmove()
                game.select_move(move)
                moves += [move]
                
                try:
                    tree.close()
                    del tree
                except Exception as e:
                    print(e)
                    
            else:
                val, pol = model(game.get_input_tensor())
                move = policy_argmax_choice(np.squeeze(pol.numpy(),axis=0), game.possible_moves, flipped=not game.board.turn)
                game.select_move(move)
                moves += [move]
            
        else:
            #stockfish turn
            move = stockfish.play_next_move(game)
            moves += [move]
        
        print(f'\nPLAYER {"Model" if not starter == game.board.turn else "Stockfish"}: {move} in {game.board.fen()}\n{game.board}\n')
        
    
    result = 'DRAW'
    if game.result == (1,0):
        result = 'WHITE'
        
    elif game.result == (0,1):
        result = 'BLACK'
    
    appendix = pd.DataFrame(
        [[
            model_name,
            stockfish_elo,
            mcts,
            ",".join(moves),
            len(moves),
            result,
            'White' if starter else 'Black',
            'Black' if starter else 'White'
        ]],
        columns=['Model','Stockfish_ELO','MCTS','Moves','Move Count','Result','Model color', 'Stockfish color']
    )
    
    return appendix

def model_vs_stockfish(model_name, stockfish_path, n=1):
    '''
    ADD
    '''
    
    data = None
    if os.path.exists('./data/stockfish_vs.csv'):
        data = pd.read_csv('./data/stockfish_vs.csv',index_col=None)
    
    else:
        data = pd.DataFrame(columns=['Model','Stockfish_ELO','MCTS','Moves','Move Count','Result','Model color', 'Stockfish color'])
    
    model_path = None
    if model_name == 'Supervised':
        model_path = './weights/supervised/model_weights_at_45.h5'
    
    elif model_name == 'Montecarlo':
        model_path = './weights/montecarlo/model_weights_at_23.h5'
        
    elif model_name == 'DDPG':
        model_path = './weights/ddpg/target_model_weights_at_84.h5'
    
    else:
        raise ValueError('Model must have a name that is either Supervised, Montecarlo, or DDPG.')
    
    model = load_model(model_path)
    
    for elo in [900, 1350,1500,2000]:
        for mcts_allow in [True, False]:
            appendix = playout(model, model_name, stockfish_path, elo, mcts=mcts_allow)
            data = pd.concat([data,appendix],axis=0)
            data.to_csv('./data/stockfish_vs.csv',index=None)
    
    #later implement functionality for specific elo and mcts on and off in eval script
    
    if n > 1:
        model_vs_stockfish(model_name, stockfish_path, n=n-1)