import random
import os

import numpy as np
import pandas as pd

from model.mcts_vanilla import MctsTree as MctsTreeVanilla
from model.mcts import MctsTree

from lib.engine import ChessGame

from model.default_policies import random_simulation, cnn_simulation
from model.model import ChessModel

from evaluate.stockfish_wrapper import StockfishModel

from datetime import datetime, timedelta

from tqdm import tqdm

def sample_random_position():
    '''
    ADD
    '''
    
    game = ChessGame(T=8)
    
    moves = []
    while not game.terminal:
        moves += [random.choice(game.possible_moves)]
        game.select_move(moves[-1])
        
    moves = moves[:random.randint(1,len(moves)-1)] #make sure at least one move from terminal and one move from initial
    
    return moves

def evaluate_mcts(initial_moves : list, stockfish_value : float, stockfish_three : list):
    '''
    ADD
    '''
    
    game = ChessGame(T=8,initial_moves=initial_moves)
    
    model = ChessModel()
    model(game.get_input_tensor())
        
    model.load('./weights/ddpg/target_model_weights_at_84.h5')
    
    columns = ['MCTS Mode','Length','Moves','Perspective','Simulation function','Time', 'Policy MEAN', 'Policy STD', 'Best three', 'Stockfish best three', 'MCTS value', 'Stockfish value']
    data = pd.DataFrame(columns=columns)
    
    #Multithreaded CNN
    simul_func = lambda game: cnn_simulation(game, model)
    
    start = datetime.now()
    tree = MctsTree(game)
    
    policy = tree(simul_func)
    end = datetime.now()
    
    best_three = tree.best_three()
    value = tree.evaluation()
    
    try:
        tree.close()
        del tree
        
    except Exception as e:
        print(e)
        
    appendix = pd.DataFrame(
        [['Multithreaded',len(initial_moves),",".join(initial_moves),f'{"White" if (len(initial_moves) % 2) == 0 else "Black"}','CNN',float((end - start).seconds), np.mean(policy), np.std(policy), ",".join(best_three), ",".join(stockfish_three), value, stockfish_value]],
        columns=columns
    )
    
    data = pd.concat([data,appendix],axis=0)
    
    #Vanilla CNN
    start = datetime.now()
    tree = MctsTreeVanilla(game)
    
    policy = tree(simul_func)
    end = datetime.now()
    
    best_three = tree.best_three()
    value = tree.evaluation()
    
    try:
        del tree
        
    except Exception as e:
        print(e)
        
    appendix = pd.DataFrame(
        [['Vanilla',len(initial_moves),",".join(initial_moves),f'{"White" if (len(initial_moves) % 2) == 0 else "Black"}','CNN',float((end - start).seconds), np.mean(policy), np.std(policy), ",".join(best_three), ",".join(stockfish_three), value, stockfish_value]],
        columns=columns
    )
    
    data = pd.concat([data,appendix],axis=0)
    
    #Multithreaded RAN
    simul_func = lambda game: random_simulation(game)
    
    start = datetime.now()
    tree = MctsTree(game)
    
    policy = tree(simul_func)
    end = datetime.now()
    
    best_three = tree.best_three()
    value = tree.evaluation()
    
    try:
        tree.close()
        del tree
        
    except Exception as e:
        print(e)
        
    appendix = pd.DataFrame(
        [['Multithreaded',len(initial_moves),",".join(initial_moves),f'{"White" if (len(initial_moves) % 2) == 0 else "Black"}','RAN',float((end - start).seconds), np.mean(policy), np.std(policy), ",".join(best_three), ",".join(stockfish_three), value, stockfish_value]],
        columns=columns
    )
    
    data = pd.concat([data,appendix],axis=0)   
        
    #Vanilla RAN
    start = datetime.now()
    tree = MctsTreeVanilla(game)
    
    policy = tree(simul_func)
    end = datetime.now()
    
    best_three = tree.best_three()
    value = tree.evaluation()
    
    try:
        del tree
        
    except Exception as e:
        print(e)
        
    appendix = pd.DataFrame(
        [['Vanilla',len(initial_moves),",".join(initial_moves),f'{"White" if (len(initial_moves) % 2) == 0 else "Black"}','RAN',float((end - start).seconds), np.mean(policy), np.std(policy), ",".join(best_three), ",".join(stockfish_three), value, stockfish_value]],
        columns=columns
    )
    
    data = pd.concat([data,appendix],axis=0)
    
    return data

def stockfish_eval(initial_moves, path):
    '''
    ADD
    '''
    game = ChessGame(T=1,initial_moves=initial_moves)
    
    stockfish = StockfishModel(path)
    
    return stockfish.evaluate_pos(game), stockfish.get_top_k_moves(game,3)

def do_statistics(stockfish_path, n=1):
    '''
    ADD
    '''
    
    data = None
    if os.path.exists('./data/mcts_eval.csv'):
        data = pd.read_csv('./data/mcts_eval.csv',index_col=None)
    
    else:
        data = pd.DataFrame(columns=['MCTS Mode','Length','Moves','Perspective','Simulation function','Time', 'Policy MEAN', 'Policy STD', 'Best three', 'Stockfish best three','MCTS value', 'Stockfish value'])
        data.to_csv('./data/mcts_eval.csv',index=None)
    
    for _ in tqdm(range(n),desc='Running evaluation...'):
        moves = sample_random_position()
        
        stockfish_val, stockfish_moves = stockfish_eval(moves, path=stockfish_path)
        
        appendix = evaluate_mcts(moves, stockfish_val, stockfish_moves)
        
        data = pd.concat([data,appendix],axis=0)
        data.to_csv('./data/mcts_eval.csv',index=None)