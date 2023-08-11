import chess
import os
import time
import argparse

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds

from datetime import datetime, timedelta
from tqdm import tqdm

from model.model import ChessModel
from model.default_policies import cnn_simulation
from lib.engine import ChessGame
from lib.utils import get_policy_index, get_padded_move_probabilities

parser = argparse.ArgumentParser(description='empty')
parser.add_argument('--shutdown',default='False')
parser.add_argument('--time',default=None)
parser.add_argument('--path',default='./dataset/dataset.csv')

args = parser.parse_args()

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

def load_dataset(path : str):
    return tf.data.Dataset.from_generator(
        DatasetGenerator(path),
        output_signature=(
            tf.TensorSpec(shape=(8,8,119),dtype='float32'),
            tf.TensorSpec(shape=(8,8,73),dtype='float32'),
            tf.TensorSpec(shape=(1,),dtype='float32')
        )
    )

class TrainingHandler:
    '''
    ADD
    '''

    def __init__(self, dataset_path):
        '''
        ADD
        '''

        #declaring constants
        self.T = 8

        #declaring variables
        self.epoch = 1
        self.runtimes = [60] #initially assume an epoch requires 60mins

        #loading in and preparing the model
        self.model = ChessModel()
        self.model(ChessGame(T=self.T).get_input_tensor())

        #initialize list of shards with proper order
        self.shards = [dataset_path+path for path in os.listdir(dataset_path) if ('_shard_' in path) and ('.csv' in path)]

        mapping = []
        for shard in self.shards:
            mapping += [int(shard.split('.')[0].split('_')[-1])]

        self.shards = list(sorted(zip(mapping,self.shards),key=lambda elem: elem[0]))
        self.shards = [shard for idx, shard in self.shards]

        #if no dataset is provided for pretraining print error
        if len(self.shards) == 0:
            raise FileNotFoundError('The passing of a dataset for pretraining is required.')

        #loading in the training dataframe
        if os.path.exists('./data/training_data_supervised.csv'):
            df = pd.read_csv('./data/training_data_supervised.csv',index_col=None)

            if len(df) > 0:
                self.epoch = int(df['Epoch'].iloc[-1]) + 1 #we add one since we start the epoch after the last
                self.runtimes = [int(runtime) for runtime in list(df['Runtime'])[-10:]]

        else:
            data = pd.DataFrame(columns=['Epoch','Startdatetime','Enddatetime','Runtime','Loss','Path'])
            data.to_csv('./data/training_data_supervised.csv',index=False)
            self.epoch = 1

        if self.epoch > 1:
            self.model.load(f'./weights/supervised/model_weights_at_{self.epoch-1}.h5')

    @property
    def est_epoch_time(self) -> timedelta:
        '''
        ADD
        '''

        return timedelta(minutes=np.mean(self.runtimes[-10:]))

    @property
    def weights_path(self) -> str:
        return f'./weights/supervised/model_weights_at_{self.epoch}.h5'

    def preprocess(self, dataset):
        dataset = dataset.take(1024000)
        dataset = dataset.batch(1024)
        dataset = dataset.prefetch(20)

        return dataset

    def episode(self, batch):
        '''
        ADD
        '''

        input_tensor, move_distribution, empiric_value = batch

        #weight update with training batch
        with tf.GradientTape() as tape:
            pred_value, policy = self.model(input_tensor,training=True)

            #calculate the loss with the known game outcome and the move_probabilities
            loss = self.model.loss(pred_value, empiric_value, policy, move_distribution)

            #apply gradients
            gradient = tape.gradient(loss, self.model.trainable_variables)

            self.model.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

        return tf.reduce_mean(loss)

    def training(self, endtime=None, shutdown=False):
        '''
        ADD

        :param endtime (datetime.datetime):
        :param shutdown (bool):
        '''

        if endtime != None:
            if datetime.now() >= (endtime or (datetime.now()+timedelta(minutes=15))):
                raise ValueError('Must specify an endtime which is at least 15 minutes in the future.')

        #training loop
        while self.continue_training(endtime):
            #add
            if self.epoch > len(self.shards):
                print(f'Not enough data to continue training, shard limit of {len(self.shards)} exceeded!')
                break

            #get start time
            start = datetime.now()

            #initialize empty list for loss
            losses = []

            dataset = load_dataset(self.shards[self.epoch-1])
            dataset = self.preprocess(dataset)

            print(f'Starting training epoch on {self.shards[self.epoch-1]} dataset split.')
            for batch in tqdm(dataset,desc=f'Running epoch {self.epoch} for 1.000 iterations:'):
                try:
                    loss = self.episode(batch)
                    losses += [loss]

                except Exception as e:
                    print('An exception has occured and is purposefully ignored.')
                    print(e)

            #get end time
            end = datetime.now()

            #process data ADD HERE
            losses = float(tf.reduce_mean(losses).numpy())

            #print small eval message CHANGE ADD
            print(f'Epoch {self.epoch} finished with an average loss of {losses}.')

            #append results to training_data_supervised.csv
            appendix = pd.DataFrame(
                [[
                    self.epoch,
                    start.strftime('%d-%m-%Y %H:%M:%S'),
                    end.strftime('%d-%m-%Y %H:%M:%S'),
                    ((end-start).seconds / 60),
                    np.mean(losses),
                    self.weights_path
                ]],
                columns=['Epoch','Startdatetime','Enddatetime','Runtime','Loss','Path']
            )

            #load, append, save
            data = pd.read_csv('./data/training_data_supervised.csv',index_col=None)
            data = pd.concat([data,appendix],axis=0)
            data.to_csv('./data/training_data_supervised.csv',index=False)

            #save model weights
            self.model.save(self.weights_path)

            #increment epoch
            self.epoch += 1

            self.runtimes = self.runtimes+[((end-start).seconds / 60)]
            self.runtimes = self.runtimes[-10:]

        #schedules a shutdown after training if desired
        if shutdown:
            print('Scheduled a shutdown in 3 minutes which may be stopped with "shutdown /a" in the cmd prompt.')
            os.system("shutdown -s -t 180") # schedule a shutdown that can be stopped with cmd("shutdown /a") if necessary.

    def continue_training(self, endtime=None) -> bool:
        '''
        ADD
        '''

        if endtime == None:
            print('Without a specified endtime this script will not be executed.')
            return False

        #abandon if time constraints violated
        if datetime.now() > endtime:
            print('Abandoning training due to overtime.')
            return False

        #abandon if another estimated epoch would take too long
        if (datetime.now()+self.est_epoch_time) > (endtime+timedelta(minutes=30)):
            print('Abandoning training due to estimated time exceeding endtime by more than 30 minutes.')
            return False

        #if no constraints violated continue
        return True

def main(args):
    time = None
    if args.time != None:
        time = int(args.time)
        time = datetime.now()+timedelta(minutes=time)

    #add
    shutdown = False
    if args.shutdown == 'True':
        shutdown = True

    #add
    training_handler = TrainingHandler(args.path)

    #add
    if not os.path.exists(args.path):
        raise FileNotFoundError(f'Could not find dataset file at {args.path}.')

    print('\n')
    training_handler.training(endtime=time,shutdown=shutdown)

if __name__ == "__main__":
    main(args)
