import chess
import os
import time
import argparse
import random
import base64

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds

from datetime import datetime, timedelta
from tqdm import tqdm

from model.model import ChessModel
from model.default_policies import cnn_simulation
from lib.engine import ChessGame
from lib.utils import get_policy_index, get_padded_move_probabilities, policy_argmax_choice, idx2position, position2idx

from copy import deepcopy

from multiprocessing.pool import ThreadPool

parser = argparse.ArgumentParser(description='empty')
parser.add_argument('--shutdown',default='False')
parser.add_argument('--time',default=None)
parser.add_argument('--path',default='./buffer/')

args = parser.parse_args()

class ExperienceReplayBuffer:
    '''
    The class for implementing the experience replay buffer.

    :att memory: None if empty, else a numpy array in form of a stack of elements for sampling.
    :att size (int): The maximum size of the buffer.
    :att batch_size (int): The size of a return-batch, used later for sampling in batch size.
    '''

    def __init__(self, path, size=160000, batch_size=64):
        '''
        Method for initializing the experience replay buffer.

        :param size (int): The maximum size of the buffer.
        :param batch_size (int): The size of a return-batch, used later for sampling in batch size.
        ADD : Size is exactly 10 iterations memory of all batches for n=100 batches over a size of 64
        '''

        self.path = path+'replay_buffer.csv'
        self.size = size
        self.batch_size = batch_size

        #load in the existing buffer if it exists
        try:
            self.buffer = pd.read_csv(self.path,index_col=None).values

        except FileNotFoundError:
            self.buffer = pd.DataFrame(columns=['State','Action','Reward','Successor','Terminal']).values

        except Exception as e:
            raise e

    def append(self, state : bytes, action : int, reward : float, successor : bytes, terminal : bool):
        '''
        Method for appending an element to the memory of the replay buffer.
        Follows first-in-first-out scheme for appending elements if the memory size is exceeded.

        :param element (list/tuple): A list of ADD.
        '''

        appendix = np.array([[state, action, reward, successor, terminal]],dtype='object')
        self.buffer = np.vstack([self.buffer,appendix])

        #every now and then clear working memory by deleting and saving a new dataframe
        if len(self.buffer) > self.size * 1.5:
            self.buffer = self.buffer[-self.size:]

    def sample(self, n=500):
        '''
        Method for sampling self.batch_size elements from memory and returning them in dataset batch form.

        :returns (tf.data.Dataset): A tensorflow dataset slice with shape (batch_size,5).
        '''

        if len(self.buffer) > self.batch_size * n:
            sample_indices = np.random.choice(np.arange(0, len(self.buffer)), self.batch_size*n, replace=False)
            samples = [self.buffer[idx] for idx in sample_indices]

            #do multithreaded sample preprocessing
            samples = ThreadPool(32).map(self.preprocess_samples, samples)

            #do dataset preprocessing with batching
            data = self.preprocess(samples)

            return data

        else:
            raise AttributeError('A sample was requested but the memory was not yet filled enough to provide one.')

    def preprocess_samples(self, sample):
        '''
        ADD
        '''

        return (
            tf.cast(np.frombuffer(base64.b64decode(sample[0]),dtype='float32').reshape((8,8,119)),dtype='float32'),
            tf.constant([sample[1]],dtype='int32'),
            tf.constant([sample[2]],dtype='float32'),
            tf.cast(np.frombuffer(base64.b64decode(sample[3]),dtype='float32').reshape((8,8,119)),dtype='float32'),
            tf.constant([sample[4]],dtype='bool')
        )

    def preprocess(self, dataset):
        '''
        ADD
        '''

        #convert buffer data to tf.dataset
        a = tf.stack([a for a,b,c,d,e in dataset])
        b = tf.stack([b for a,b,c,d,e in dataset])
        c = tf.stack([c for a,b,c,d,e in dataset])
        d = tf.stack([d for a,b,c,d,e in dataset])
        e = tf.stack([e for a,b,c,d,e in dataset])

        dataset = tf.data.Dataset.from_tensor_slices((a,b,c,d,e))

        #batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.batch_size//4)

        return dataset

    def flush(self):
        '''
        ADD
        '''

        pd.DataFrame(self.buffer,columns=['State','Action','Reward','Successor','Terminal']).to_csv(self.path,index=None)

class TrainingHandler:
    '''
    ADD
    '''

    def __init__(self, path, gamma=0.95, polyak_avg_fac=0.8, epsilon=0.1, target_update_epochs=4):
        '''
        path: for exp rep buf
        '''

        #declaring constants
        self.T = 8
        self.gamma = gamma
        self.epsilon = epsilon
        self.polyak_avg_fac = polyak_avg_fac
        self.target_update_epochs = target_update_epochs

        #declaring variables
        self.epoch = 1
        self.runtimes = [60] #initially assume an epoch requires 60mins

        #loading in and preparing the model
        self.model = ChessModel()
        self.model(ChessGame(T=self.T).get_input_tensor())

        #loading in and preparing the target model
        self.target_model = ChessModel()
        self.target_model(ChessGame(T=self.T).get_input_tensor())

        #initialize buffer
        self.buffer = ExperienceReplayBuffer(path)

        #loading in the training dataframe
        if os.path.exists('./data/training_data_ddpg.csv'):
            df = pd.read_csv('./data/training_data_ddpg.csv',index_col=None)

            if len(df) > 0:
                self.epoch = int(df['Epoch'].iloc[-1]) + 1 #we add one since we start the epoch after the last
                self.runtimes = [int(runtime) for runtime in list(df['Runtime'])[-10:]]

            else:
                self.epoch = 1

        else:
            data = pd.DataFrame(columns=['Epoch','Startdatetime','Enddatetime','Runtime','Loss','Path','Target-Path'])
            data.to_csv('./data/training_data_ddpg.csv',index=False)
            self.epoch = 1

        if self.epoch > 1:
            self.model.load(f'./weights/ddpg/model_weights_at_{self.epoch-1}.h5')
            self.target_model.load(f'./weights/ddpg/target_model_weights_at_{self.epoch-1}.h5')
        else:
            self.target_model = deepcopy(self.model) #for the start the target model has to be identical

    @property
    def est_epoch_time(self) -> timedelta:
        '''
        ADD
        '''

        return timedelta(minutes=np.mean(self.runtimes[-10:]))

    @property
    def weights_path(self) -> str:
        return (f'./weights/ddpg/model_weights_at_{self.epoch}.h5',f'./weights/ddpg/target_model_weights_at_{self.epoch}.h5')

    def sample_game(self):
        '''
        ADD
        '''

        game = ChessGame(T=8)
        trajectory = []

        state, move, state_prime = base64.b64encode(game.get_input_tensor().numpy().reshape(8,8,119).tobytes()).decode('utf-8'), None, None

        while not game.terminal:
            val, pol = self.model(game.get_input_tensor())

            #for initial epoch we do random actions fully
            if random.random() <= self.epsilon or self.epoch == 1:
                move = random.choice(game.possible_moves)

            else:
                move = policy_argmax_choice(np.squeeze(pol.numpy(),axis=0), game.possible_moves, flipped=not game.board.turn)

            game.select_move(move)
            state_prime = base64.b64encode(game.get_input_tensor().numpy().reshape(8,8,119).tobytes()).decode('utf-8')

            #we don't know the reward yet so we add none each time
            trajectory += [(state, position2idx(get_policy_index(move)), None, state_prime, game.terminal)] #we do not need the successor state in this case

            state = state_prime

        outcome = 0
        if game.result == (1,0):
            outcome = 1

        elif game.result == (0,1):
            outcome = -1

        rewards = [(outcome*self.gamma**i)*(1,-1)[i%2] for i in range(len(trajectory))] #we transform the reward into a discounted proper reward here

        return [(trajectory[i][0], trajectory[i][1], rewards[i], trajectory[i][3], trajectory[i][4]) for i in range(len(trajectory))]

    def update_step(self, batch):
        '''
        ADD
        '''
        #weight update with training batch
        with tf.GradientTape() as tape:
            tar_vals, tar_pols = [], []

            states, actions, rewards, successors, terminals = batch

            pred_values, pred_policies = self.model(states,training=True)
            tar_pred_values, tar_policies = self.target_model(successors)

            for i, terminal in enumerate(terminals):
                #we generate the sampled policy as in supervised learning from the sample present
                sampled_policy = np.full((8,8,73),1e-10)
                sampled_policy[idx2position(actions[i])] = 0.9999

                #if the item is a move which ends in a terminal successor state:
                if terminal:
                    tar_vals.append(rewards[i]) #the true value is known and therefore used in training

                else:
                    tar_vals.append(rewards[i] + tf.multiply(self.gamma, tar_pred_values[i]))

                tar_pols.append(tf.constant(sampled_policy))

            #ADD
            loss = self.model.loss(
                pred_values,
                tf.cast(tf.stack(tar_vals),dtype='float32'),
                pred_policies,
                tf.cast(tf.stack(tar_pols),dtype='float32')
            )

            #apply gradients
            gradient = tape.gradient(loss, self.model.trainable_variables)

            self.model.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

        return tf.reduce_mean(loss)

    def update_target_network(self):
        '''
        ADD
        '''

        for target_var, model_var in zip(self.target_model.trainable_variables, self.model.trainable_variables):
            target_var.assign(self.polyak_avg_fac * target_var + (1 - self.polyak_avg_fac) * model_var)


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
            losses = []

            #DDPG Training

            #sampling step


            print(f'Starting sampling process of n>={self.buffer.batch_size*500} chess positions with the current model')
            positions = 0
            while positions < self.buffer.batch_size*500:
                trajectories = self.sample_game()
                positions += len(trajectories)

                for state, action, reward, successor, terminal in trajectories:
                    self.buffer.append(state, action, reward, successor, terminal)

                print(f'Progress: {positions}/{int(self.buffer.batch_size*500)} = {round((positions/(self.buffer.batch_size*500))*100, 2)}%',end="\r")

            self.buffer.flush()

            #training on buffer
            print(f'Starting training episode {self.epoch} on the Experience Replay Buffer:')
            samples = self.buffer.sample()

            for batch in tqdm(samples,desc=f'Training process:'):
                losses += [self.update_step(batch)]

            #get end time
            end = datetime.now()
            
            if self.epoch % self.target_update_epochs == 0:
                print(f'Due to {self.epoch} being a target update epoch we update the target network:')
                self.update_target_network()

            #process data ADD HERE
            losses = float(tf.reduce_mean(losses).numpy())

            #print small eval message CHANGE ADD
            print(f'Epoch {self.epoch} finished with an average loss of {losses}.')

            #append results to training_data_ddpg.csv
            appendix = pd.DataFrame(
                [[
                    self.epoch,
                    start.strftime('%d-%m-%Y %H:%M:%S'),
                    end.strftime('%d-%m-%Y %H:%M:%S'),
                    ((end-start).seconds / 60),
                    np.mean(losses),
                    self.weights_path[0],
                    self.weights_path[1]
                ]],
                columns=['Epoch','Startdatetime','Enddatetime','Runtime','Loss','Path','Target-Path']
            )

            #load, append, save
            data = pd.read_csv('./data/training_data_ddpg.csv',index_col=None)
            data = pd.concat([data,appendix],axis=0)
            data.to_csv('./data/training_data_ddpg.csv',index=False)

            #save model weights
            self.model.save(self.weights_path[0])
            self.target_model.save(self.weights_path[1])

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
