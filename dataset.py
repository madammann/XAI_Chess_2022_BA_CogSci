import re
import chess
import os
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from stockfish import Stockfish
from multiprocessing.pool import ThreadPool

from lib.engine import ChessGame
from lib.utils import get_policy_index

parser = argparse.ArgumentParser(description='empty')
parser.add_argument('--path', default='./dataset/')

args = parser.parse_args()

def read_pgn(path : str) -> list:
    '''
    ADD
    '''

    #read file in
    file = open(path,'r')
    content = file.read()
    file.close()

    #filter only the game notation string
    games = []
    for line in content.split('\n'):
        if re.search(r'1. ',line) != None:
            games += [line]

    return games

def convert_game(game_data : str, stockfish_path='./stockfish_15.1_win_x64_avx2/stockfish_15_x64_avx2.exe') -> tuple:
    '''
    ADD
    '''

    #get the result from the data and store it as number where 1 means white won, 0 means draw, -1 means black won
    result = re.search(r'1-0|0-1|1\/2-1\/2',game_data)[0]
    if result == '1-0':
        result = 1
    elif result == '0-1':
        result = -1
    else:
        result = 0

    moves = re.findall(r'[abcdefghx]+[1234567890][=QRBN]{0,2}|[KQRBN][12345678]{0,1}[abcdefghx]+[123456789]|[O\-]{3,5}',game_data)

    board = chess.Board()
    try:
        for move in moves:
            board.push_san(move)
        
        #make a list of uci moves with comma separation
        moves = [move.uci()+',' for move in board.move_stack]
        
        #no game with less than three moves can be lost (fastest mate is in four moves) 
        if len(moves) < 3:
            return (None, None)

        return (''.join(moves)[:-1], result)

    except ValueError:
        return (None, None)

if __name__ == "__main__":
    #collect all paths to .png files
    affix = args.path
    paths = [affix+path for path in os.listdir(affix) if path[-4:] == '.pgn']

    print('Checking for PGN files...')
    if len(paths) > 0:
        print(f'Found {len(paths)} PGN files, beginning conversion to csv (this can be interrupted at any time and resumed by executing this script again at a later time).')
    else:
        print('All PGN files were already converted to CSV.')

    print('\n')
    #iterate over all .png files
    for i, path in tqdm(enumerate(paths),desc=f'Converting pgn to csv',total=len(paths)):
        #generate DataFrame to append to
        df = pd.DataFrame(columns=['Moves','Result'])
        games = read_pgn(path)
        
        total_games, valid_games = 0, 0
        invalid_game_samples = [] #collect all invalid games in a separate txt file for analysis

        #iterate over all games in each .png file
        thread_results = ThreadPool(100).map(convert_game, games)
        
        #filter the valid games and append them to the dataframe
        df = pd.concat([df,pd.DataFrame(zip([res[0] for res in thread_results], [res[1] for res in thread_results]),columns=['Moves','Result'])],axis=0)
        df = df.dropna()

        #store data, rename processed file and save chunk
        os.rename(path,path+'.processed')
        n = len([affix+path for path in os.listdir(affix) if path[-10:] == '.processed'])
        df.to_csv(f'{affix}chess_dataset_{n}.csv',index=None,sep=';')

    print('\n')
    print('Deleting processed PGN files if present...')
    for path in tqdm([path for path in os.listdir(affix) if path[-10:] == '.processed'],desc='Deleting processed PGN files...'):
        os.remove(f'{affix}{path}')

    if not os.path.exists(f'{affix}/chess_dataset.csv'):
        print('Creating merged dataset csv file...')
        pd.DataFrame(columns=['Moves','Result']).to_csv(f'{affix}chess_dataset.csv',index=None,sep=';')

    print('\n')
    print('Checking for unmerged csv files...')
    csv_paths = [
        affix+path for path in os.listdir(affix)
        if (path[-4:] == '.csv')
        and (path[-23:] != 'chess_dataset.csv')
    ]

    for path in tqdm(csv_paths,desc='Merging singular csv files into main csv...'):
        appendix = pd.read_csv(path,index_col=None,sep=';')

        #define cutoff point and shuffle
        appendix = appendix.sample(frac=1)

        #append train and test appendix
        train = pd.read_csv(f'{affix}chess_dataset.csv',index_col=None,sep=';')
        train = pd.concat([train,appendix],axis=0)
        train.to_csv(f'{affix}chess_dataset.csv',index=None,sep=';')

        os.remove(path)

    print('\n')
    print('Beginning splitting the train dataset into shards (each with a total move count of 1024[batch_size]*moves<=1.000)')

    if os.path.exists(f'{affix}chess_dataset.csv') and (not os.path.exists(f'{affix}chess_dataset_shard_{0}.csv')):
        train = pd.read_csv(f'{affix}chess_dataset.csv',index_col=None,sep=';')
        train = train.sample(frac=1)

        sumlim = 1024000
        shatter_indices = [0]
        for i, moves in enumerate(list(train['Moves'])):
            sumlim -= moves.count(',')+1
            if (sumlim+moves.count(',')+1) <= 0:
                sumlim = 1024000
                shatter_indices += [i]

        for i in range(len(shatter_indices)):
            if i != 0:
                train.iloc[shatter_indices[i-1]:shatter_indices[i]].to_csv(f'{affix}chess_dataset_shard_{i}.csv',index=None,sep=';')

    else:
        if os.path.exists(f'{affix}chess_dataset.csv') and os.path.exists(f'{affix}chess_dataset_shard_{0}.csv'):
            print('Detected previously created shards, please delete all shards before trying to create new ones.')
    
    #create folder for test cache files
    os.mkdir(f'{affix}test_cache/')
    
    print('Creating test data split...')
    shard_paths = [affix+path for path in os.listdir(affix) if ('_shard_' in path) and ('.csv' in path)]
    
    #make last three shards the test data
    df = pd.DataFrame(columns=['Moves','Result'])
    for path in shard_paths[len(shard_paths)-3:]:
        appendix = pd.read_csv(path,sep=';',index_col=None)
        df = pd.concat([df,appendix],axis=0)
        
    df.to_csv(f'{affix}chess_dataset_test.csv',index=None,sep=';')
    os.remove(shard_paths[-1])
    os.remove(shard_paths[-2])
    os.remove(shard_paths[-3])
            
    print('\n')
    print('Finished creating the dataset for pretraining!')