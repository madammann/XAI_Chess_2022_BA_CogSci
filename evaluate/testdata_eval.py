import chess
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from datetime import datetime, timedelta
from tqdm import tqdm

from model.model import ChessModel

from lib.engine import ChessGame
from lib.utils import get_policy_index

class DatasetGenerator:
    def __init__(self, path : str, T=8):
        self.path = path
        self.T = T
        self.generator = self.move_generator()

    def move_generator(self,gamma=0.95):
        for a, b in pd.read_csv(self.path,sep=';',index_col=None).values:
            game = ChessGame(T=self.T)
            for move, disc_fac in zip(a.split(','),[gamma**(len(a.split(','))-i-1) for i in range(len(a.split(',')))]):
                input_tensor = tf.squeeze(game.get_input_tensor(),axis=0)

                move_tensor = np.full((8,8,73),1e-10)
                move_tensor[get_policy_index(move)] = 0.9999
                move_tensor = tf.constant(move_tensor)

                value_tensor = tf.constant([b*disc_fac if game.board.turn else -b*disc_fac])

                game.select_move(move)

                yield input_tensor, move_tensor, value_tensor

    def __call__(self):
        return self.move_generator()
    
def load_model(model_path):
    '''
    ADD
    '''

    model = ChessModel()
    model(ChessGame(T=8).get_input_tensor())
    model.load(model_path)

    return model

def load_test_dataset(datapath : str):
    '''
    ADD
    '''
    
    expected_input_shape = (8,8,119)

    dataset = tf.data.Dataset.from_generator(
        DatasetGenerator(datapath+'chess_dataset_test.csv',T=8),
        output_signature=(
            tf.TensorSpec(shape=expected_input_shape,dtype='float32'),
            tf.TensorSpec(shape=(8,8,73),dtype='float32'),
            tf.TensorSpec(shape=(1,),dtype='float32')
        )
    )

    dataset = dataset.take(1024000*2)
    dataset = dataset.batch(1024)

    dataset = dataset.cache(datapath+'test_cache/')

    return dataset

def testeval_epoch(datapath : str, model_path : str):
    '''
    ADD
    '''

    model = load_model(model_path)
    testdata = load_test_dataset(datapath)

    loss = []
    for datum in testdata:
        input_tensor, move_distribution, empiric_value = datum

        pred_value, policy = model(input_tensor)

        #calculate the loss with the known game outcome and the move_probabilities
        loss += [model.loss(pred_value, empiric_value, policy, move_distribution)]

    return tf.reduce_mean(loss)

def testeval(datapath : str):
    '''
    ADD
    '''

    supervised = pd.DataFrame(columns=['Epoch','Loss']).set_index('Epoch')
    montecarlo = pd.DataFrame(columns=['Epoch','Loss']).set_index('Epoch')
    ddpg = pd.DataFrame(columns=['Epoch','Loss']).set_index('Epoch')
    
    if os.path.exists('./data/test_eval_supervised.csv'):
        supervised = pd.read_csv('./data/test_eval_supervised.csv',index_col='Epoch')
    
    if os.path.exists('./data/test_eval_montecarlo.csv'):
        montecarlo = pd.read_csv('./data/test_eval_montecarlo.csv',index_col='Epoch')
        
    if os.path.exists('./data/test_eval_ddpg.csv'):
        ddpg = pd.read_csv('./data/test_eval_ddpg.csv',index_col='Epoch')
    
    supervised_data = pd.read_csv('./data/training_data_supervised.csv',index_col='Epoch')
    montecarlo_data = pd.read_csv('./data/training_data_montecarlo.csv',index_col='Episode')
    ddpg_data = pd.read_csv('./data/training_data_ddpg.csv',index_col='Epoch')
    
    supervised_epochs = np.linspace(1,len(supervised_data),20,dtype='int32').tolist()
    montecarlo_epochs = np.linspace(1,len(montecarlo_data),20,dtype='int32').tolist()
    ddpg_epochs = np.linspace(1,len(ddpg_data),20,dtype='int32').tolist()
    
    #calculate remaining epochs to check
    supervised_epochs = list(set(supervised_epochs).difference(set(supervised.index.to_list())))
    montecarlo_epochs = list(set(montecarlo_epochs).difference(set(montecarlo.index.to_list())))
    ddpg_epochs = list(set(ddpg_epochs).difference(set(ddpg.index.to_list())))
    
    #do each test eval
    for epoch in tqdm(supervised_epochs, desc=f'Running test evaluation on supervised model:'):
        path = supervised_data['Path'].iloc[epoch-1]
        loss = float(testeval_epoch(datapath, path).numpy())
        appendix = pd.DataFrame([[epoch, loss]],columns=['Epoch','Loss']).set_index('Epoch')
        supervised = pd.concat([supervised,appendix],axis=0)
        supervised = supervised.sort_index()
        supervised.to_csv('./data/test_eval_supervised.csv',index='Epoch')
        
    for epoch in tqdm(montecarlo_epochs, desc=f'Running test evaluation on montecarlo model:'):
        path = montecarlo_data['Path'].iloc[epoch-1]
        loss = float(testeval_epoch(datapath, path).numpy())
        appendix = pd.DataFrame([[epoch, loss]],columns=['Epoch','Loss']).set_index('Epoch')
        montecarlo = pd.concat([montecarlo,appendix],axis=0)
        montecarlo = montecarlo.sort_index()
        montecarlo.to_csv('./data/test_eval_montecarlo.csv',index='Epoch')
        
    for epoch in tqdm(ddpg_epochs, desc=f'Running test evaluation on ddpg model:'):
        path = ddpg_data['Path'].iloc[epoch-1]
        loss = float(testeval_epoch(datapath, path).numpy())
        appendix = pd.DataFrame([[epoch, loss]],columns=['Epoch','Loss']).set_index('Epoch')
        ddpg = pd.concat([ddpg,appendix],axis=0)
        ddpg = ddpg.sort_index()
        ddpg.to_csv('./data/test_eval_ddpg.csv',index='Epoch')