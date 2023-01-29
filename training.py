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
from model.mcts import MctsTree
from model.default_policies import cnn_simulation
from lib.engine import ChessGame
from lib.utils import get_policy_index, get_padded_move_probabilities

parser = argparse.ArgumentParser(description='empty')
parser.add_argument('--pretraining', default='False')
parser.add_argument('--shutdown',default='False')
parser.add_argument('--time',default=None)
parser.add_argument('--path',default='./dataset/dataset.csv')

args = parser.parse_args()

class DatasetGenerator:
    def __init__(self, path : str):
        self.path = path
        self.generator = self.move_generator()

    def move_generator(self):
        for a, b in pd.read_csv(self.path,sep=';',index_col=None).values:
            game = ChessGame(T=6)
            for move in a.split(','):
                input_tensor = tf.squeeze(game.get_input_tensor(),axis=0)

                move_tensor = np.full((8,8,73),1e-10)
                move_tensor[get_policy_index(move)] = 0.9999
                move_tensor = tf.constant(move_tensor)

                value_tensor = tf.constant([b if game.board.turn else -b])

                game.select_move(move)

                yield input_tensor, move_tensor, value_tensor

    def __call__(self):
        return self.move_generator()

def load_dataset(path : str):
    return tf.data.Dataset.from_generator(
        DatasetGenerator(path),
        output_signature=(
            tf.TensorSpec(shape=(8,8,91),dtype='float32'),
            tf.TensorSpec(shape=(8,8,73),dtype='float32'),
            tf.TensorSpec(shape=(1,),dtype='float32')
        )
    )

class TrainingHandler:
    '''
    ADD
    '''

    def __init__(self, pretraining=False, dataset_path='./dataset/'):
        '''
        ADD
        '''

        #declaring constants
        self.T = 6
        self.filters = 32
        self.residual = 6
        self.pretraining = pretraining

        #declaring variables
        self.epoch = 1
        self.runtimes = [60] #initially assume an epoch requires 60mins

        #loading in and preparing the model
        self.model = ChessModel(filters=self.filters,residual_size=self.residual,T=self.T)
        self.model(ChessGame(T=self.T).get_input_tensor())

        #initialize list of shards with proper order
        self.shards = [dataset_path+path for path in os.listdir(dataset_path) if ('_shard_' in path) and ('.csv' in path)] if pretraining else None

        if self.pretraining:
            mapping = []
            for shard in self.shards:
                mapping += [int(shard.split('.')[0].split('_')[-1])]

            self.shards = list(sorted(zip(mapping,self.shards),key=lambda elem: elem[0]))
            self.shards = [shard for idx, shard in self.shards]

        #if no dataset is provided for pretraining print error
        if self.pretraining and self.shards == None:
            raise FileNotFoundError('The passing of a dataset for pretraining is required.')

        #loading in the training dataframe
        if os.path.exists('./data/training_data.csv'):
            df = pd.read_csv('./data/training_data.csv',index_col=None)
            df_pretrained = df.where(df['Mode'] == 'pretraining').dropna()
            df_selfplay = df.where(df['Mode'] == 'selfplay').dropna()

            #we load the most recent pretrained data if either...
            #...we continue pretraining
            #...we start selfplay
            if (self.pretraining and len(df_pretrained) > 0) or (self.pretraining == False and len(df_selfplay) == 0):
                self.model.load(df_pretrained['Path'].iloc[-1])
                self.epoch = int(df_pretrained['Epoch'].iloc[-1])
                self.runtimes = [int(runtime) for runtime in list(df_pretrained['Runtime'])[-10:]]

                #ADD
                self.epoch += 1

            #if we continue selfplay we load the most recent selfplay weights
            elif self.pretraining == False and len(df_selfplay) > 0:
                self.model.load(df_selfplay['Path'].iloc[-1])
                self.epoch = int(df_selfplay['Epoch'].iloc[-1])
                self.runtimes = [int(runtime) for runtime in list(df_selfplay['Runtime'])[-10:]]

                #since epoch is incremented AFTER training we have to increment it here again
                self.epoch += 1

            #if neither is true this means we are starting from scratch, meaning we don't load in any weights
        elif self.pretraining:
            print('No prior training data found, continue [y/n]?')
            if input() == 'y':
                data = pd.DataFrame(columns=['Mode','Epoch','Startdatetime','Enddatetime','Runtime','Loss','Moves','Exceptions','Path'])
                data.to_csv('./data/training_data.csv',index=False)

            else:
                raise FileNotFoundError('The previous training data could not be found.')

        else:
            raise FileNotFoundError('The previous training data could not be found and is necessary for selfplay training.')

    @property
    def weights_path(self) -> str:
        if self.pretraining:
            return f'./weights/model_weights_at_pretraining_{self.epoch}.h5'

        else:
            return f'./weights/model_weights_at_selfplay_{self.epoch}.h5'

    @property
    def est_epoch_time(self) -> timedelta:
        '''
        ADD
        '''

        return timedelta(minutes=np.mean(self.runtimes[-10:]))

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
            #get start time
            start = datetime.now()

            #initialize empty list for loss and move count
            losses, move_counts = [], []
            exceptions = 0

            if self.pretraining == False:
                #do 100 episodes for a tenth of an epoch (a milestone so to say) CHANGE ADD
                for episode in tqdm(range(100),desc=f'Running epoch {self.epoch}:'):
                    try:
                        loss, move_count = self.selfplay_episode()
                        losses += [loss]
                        move_counts += [move_count]

                    except Exception as e:
                        print('An exception has occured and is purposefully ignored.')
                        print(e)
                        exceptions += 1
            else:
                dataset = load_dataset(self.shards[self.epoch-1])
                dataset = self.preprocess(dataset)

                print(f'Starting training epoch on {self.shards[self.epoch-1]} dataset split.')
                for batch in tqdm(dataset,desc=f'Running epoch {self.epoch} for 1.000 iterations:'):
                    try:
                        loss = self.pretrain_episode(batch)
                        losses += [loss]

                    except Exception as e:
                        print('An exception has occured and is purposefully ignored.')
                        print(e)
                        exceptions += 1

            #get end time
            end = datetime.now()

            #process data ADD HERE
            losses = float(tf.reduce_mean(losses).numpy())

            #print small eval message CHANGE ADD
            print(f'Epoch {self.epoch} finished with an average loss of {losses}.')

            #append results to training_data.csv
            appendix = pd.DataFrame(
                [[
                    'pretraining' if self.pretraining else 'selfplay',
                    self.epoch,
                    start.strftime('%d-%m-%Y %H:%M:%S'),
                    end.strftime('%d-%m-%Y %H:%M:%S'),
                    ((end-start).seconds / 60),
                    np.mean(losses),
                    np.mean(move_counts) if not self.pretraining else 0,
                    exceptions,
                    self.weights_path
                ]],
                columns=['Mode','Epoch','Startdatetime','Enddatetime','Runtime','Loss','Moves','Exceptions','Path']
            )

            #load, append, save
            data = pd.read_csv('./data/training_data.csv',index_col=None)
            data = data.append(appendix)
            data.to_csv('./data/training_data.csv',index=False)

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

    def selfplay_episode(self):
        '''
        ADD
        '''

        game = ChessGame(T=self.T)
        simulation = lambda game: cnn_simulation(game, self.model)
        input_tensors, move_probs_list = [], []
        loss = None

        move_count = 0
        while not game.terminal:
            #do MCTS
            tree = MctsTree(game)
            move_probs = tree(simulation,simulation_limit=1)
            move = tree.bestmove()

            #store necessary input and move_prob data
            input_tensors += [tf.squeeze(game.get_input_tensor())]
            move_probs_list += [get_padded_move_probabilities(game.possible_moves , move_probs, flipped=not game.board.turn)]

            #play the move and continue with the successor state as opposite color
            game.select_move(move)
            move_count += 1

            #deallocate buffer for MCTS
            tree.close()

        #get true value tensor for white and black
        outcomes = tf.expand_dims(tf.constant([game.result[i%2] for i in range(len(move_probs_list))],dtype='float32'),axis=-1)

        #preprocess move probability list
        move_probs_list = tf.stack(move_probs_list)
        input_tensors = tf.stack(input_tensors)

        #weight update with training batch
        with tf.GradientTape() as tape:
            pred_values, policies = self.model(input_tensors,training=True)

            #calculate the loss with the known game outcome and the move_probabilities
            loss = self.model.loss(pred_values, outcomes, policies, move_probs_list)

            #apply gradients
            gradient = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

        return tf.reduce_mean(loss), move_count

    def pretrain_episode(self, batch):
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

    def preprocess(self, dataset):
        dataset = dataset.take(1024000)
        dataset = dataset.batch(1024)
        dataset = dataset.prefetch(20)

        return dataset

if __name__ == "__main__":
    #add
    time = None
    if args.time != None:
        time = int(args.time)
        time = datetime.now()+timedelta(minutes=time)

    shutdown = False
    if args.shutdown == 'True':
        shutdown = True

    #add
    training_handler = None
    if args.pretraining == 'False':
        training_handler = TrainingHandler(pretraining=False)

    #check that, if pretraining is selected, the dataset is present
    elif args.pretraining == 'True':
        if not os.path.exists(args.path):
            raise FileNotFoundError(f'Could not find dataset file at {args.path}.')

        training_handler = TrainingHandler(pretraining=True, dataset_path=args.path)

    else:
        raise ValueError('You must specify either True or False for the pretrainng argument.')

    print('\n')
    training_handler.training(endtime=time,shutdown=shutdown)
