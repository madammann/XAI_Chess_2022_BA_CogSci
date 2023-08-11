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
from model.mcts import MctsTree

from lib.engine import ChessGame
from lib.utils import get_policy_index, get_padded_move_probabilities

parser = argparse.ArgumentParser(description='empty')
parser.add_argument('--shutdown',default='False')
parser.add_argument('--time',default=None)

args = parser.parse_args()

class TrainingHandler:
    '''
    ADD
    '''

    def __init__(self):
        '''
        ADD
        '''

        #declaring constants
        self.T = 8

        #declaring variables
        self.episode = 1
        self.runtimes = [60] #initially assume an episode requires 60mins

        #loading in and preparing the model
        self.model = ChessModel()
        self.model(ChessGame(T=self.T).get_input_tensor())

        #loading in the training dataframe
        if os.path.exists('./data/training_data_montecarlo.csv'):
            df = pd.read_csv('./data/training_data_montecarlo.csv',index_col=None)

            if len(df) > 0:
                self.episode = int(df['Episode'].iloc[-1]) + 1 #we add one since we start the episode after the last
                self.runtimes = [int(runtime) for runtime in list(df['Runtime'])[-10:]]

        else:
            data = pd.DataFrame(columns=['Episode','Startdatetime','Enddatetime','Runtime','Loss','Move Count','Moves','Path'])
            data.to_csv('./data/training_data_montecarlo.csv',index=False)
            self.episode = 1

        if self.episode > 1:
            self.model.load(f'./weights/montecarlo/model_weights_at_{self.episode-1}.h5')

    @property
    def est_episode_time(self) -> timedelta:
        '''
        ADD
        '''

        return timedelta(minutes=np.mean(self.runtimes[-10:]))

    @property
    def weights_path(self) -> str:
        return f'./weights/montecarlo/model_weights_at_{self.episode}.h5'

    def do_episode(self, gamma=0.95):
        '''
        ADD
        '''

        game = ChessGame(T=self.T)
        simulation = lambda game: cnn_simulation(game, self.model)
        input_tensors, move_probs_list = [], []

        loss = None
        moves = []

        while not game.terminal:
            #do MCTS
            tree = MctsTree(game)
            move_probs = tree(simulation,simulation_limit=30)
            move = tree.bestmove()

            #store necessary input and move_prob data
            input_tensors += [tf.squeeze(game.get_input_tensor())]
            move_probs_list += [get_padded_move_probabilities(game.possible_moves , move_probs, flipped=not game.board.turn)]

            #play the move and continue with the successor state as opposite color
            game.select_move(move)
            print(f'\n{move} in {game.board.fen()}\n{game.board}\n')
            moves += [move]

            #deallocate buffer for MCTS
            try:
                tree.close()
                del tree
            except:
                print('Warning: Ran into error deallocating buffer space.')

        #get true value tensor for white and black (discounted by gamma**step)
        outcomes = tf.expand_dims(tf.constant([game.result[i%2]*gamma**(len(move_probs_list)-i-1) for i in range(len(move_probs_list))],dtype='float32'),axis=-1)

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
        
        print(f'checking output of episode: {tf.reduce_mean(loss)} and {moves}')
        return float(tf.reduce_mean(loss).numpy()), moves

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

            #initialize empty list for loss
            loss, moves = 0.0, []

            print(f'Starting training episode {self.episode}:')
            try:
                loss, moves = self.do_episode()
                print("Moves:" ,",".join(moves))
                print('Length:',len(moves))

            except Exception as e:
                raise e

            #get end time
            end = datetime.now()

            #print small eval message CHANGE ADD
            print(f'Episode {self.episode} finished with an average loss of {loss}.')
            
            #append results to training_data_montecarlo.csv
            appendix = pd.DataFrame(
                [[
                    self.episode,
                    start.strftime('%d-%m-%Y %H:%M:%S'),
                    end.strftime('%d-%m-%Y %H:%M:%S'),
                    ((end-start).seconds / 60),
                    float(loss),
                    len(moves),
                    ",".join(moves),
                    self.weights_path
                ]],
                columns=['Episode','Startdatetime','Enddatetime','Runtime','Loss','Move Count','Moves','Path']
            )

            #load, append, save
            data = pd.read_csv('./data/training_data_montecarlo.csv',index_col=None)
            data = pd.concat([data,appendix],axis=0)
            data.to_csv('./data/training_data_montecarlo.csv',index=False)

            #save model weights
            self.model.save(self.weights_path)
            
            #increment episode
            self.episode += 1

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

        #abandon if another estimated episode would take too long
        if (datetime.now()+self.est_episode_time) > (endtime+timedelta(minutes=30)):
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
    training_handler = TrainingHandler()

    print('\n')
    training_handler.training(endtime=time,shutdown=shutdown)

if __name__ == "__main__":
    main(args)
