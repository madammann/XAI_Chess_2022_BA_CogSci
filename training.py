# IMPORTS
import chess
import os

import numpy as np
import pandas as pd

import tensorflow as tf

from tqdm import tqdm
from datetime import datetime

from model.mcts import MctsTree
from lib.engine import ChessGame
from model.model import ChessModel

# TRAINING UTILITY FUNCTIONS
def load_progress() -> tuple:
    '''
    Reads the file progress.txt in the weights folder to obtain milestone data.
    
    :returns (tuple): A tuple of current milestone data in form of (epoch,milestone)
    '''
    
    with open('./weights/progress.txt','r') as f:
        content = f.read()
        lines = content.split('\n')
    
    return (int(lines[0].split('=')[1]),int(lines[1].split('=')[1]))

def save_progress(progress : tuple):
    '''
    Writes the current progress to the progress.txt file in the weights folder to store milestone data.
    
    :param progress (tuple): A tuple representing current milestone progress as (epoch,milestone)
    '''
    
    with open('./weights/progress.txt','w') as f:
        f.write(f'epoch={progress[0]}\nmilestone={progress[1]}')

def verify_continue(starttime, timer, progress, goal) -> bool:
    '''
    This function is called after each milestone to verify whethet the training shall be continued.
    
    :param starttime (datetime): A datetime object created at training start to calculate passed time with.
    :param timer (int): A number representing the distance in minutes from start at which the training shall be stopped at earliest convenience.
    :param progress (tuple): The tuple representing current milestone progress.
    :param goal (tuple): The tuple representing milestone goal progress.
    
    :returns (bool): True if training is to be resumed, False if training is to be stopped.
    '''
    
    if progress == goal:
        return False
    
    delta = datetime.now()-starttime
    
    if timer > delta.minutes:
        return False
        
    return True

def append_training_stats(data : list):
    '''
    This function appends a row to the csv file in data containing training data information.
    
    :param data (list): A list with ordered data to append.
    '''
    columns = ['Milestone','Start','End','Seconds','Loss','Samples','...']
    df = pd.load_csv('./data/training.csv', index=False)
    appendix = pd.DataFrame([data], columns=columns)
    
    
    pass

# TRAINING FUNCTIONS
def train(model, data):
    for input_tensor, error in data:
        pass
    
# z : the (discounted?) result of the game -1,0,1
# v : predicted outcome e.g. val of network
# pi : distribution of search probabilities(?)
# p : policy output e.g. pol of network
# c : parameter for controling l2 weight norm
# s : input tensor

# INITIALISATION

# training setup
pc_care = True # if set to true then the PC will be shut down 5mins after training milestone for the day was reached.
shutdown_timer = None # will be set manually and represents the distance from now at which the pc will be shut down as soon as model weights are stored.
now = datetime.now()

if pc_care:
    print('Please select a time in hours from now on at which the training shall be terminated at earliest convenience:')
    while not type(shutdown_times) == int:
        shutdown_timer = input()
        
        try:
            shutdown_times = int(shutdown_timer)
        
        except:
            print('Incorrect input, please try again and select a time in hours:')

# training parameters
epoch_goal = 100
episodes_per_epoch = 3125 # 100.000 games per epoch since 32 batches are used
batch_size = 32

# hyperparameters
epsilon = 0.005
learning_rate = 0.001

# training preparation
progress = (0, 0)
progress_target = (100, 3125) # 100 epochs with 3125 episodes

try:
    progress = load_progress()
    print('Loaded in current progress successfully.')
except:
    print('Could not load in current progress, starting from scratch.')

# loading in and preparing the model
model = ChessModel()
model(tf.random.normal(1,1,(1,8,8,111))) # feed forward once to make sure a graph is created

# we load model weights or, if model is new, store initial version after initialization
try:
    model.load('./weights/milestone_' + str(progress[1]) + '_' + str(progress[0]) + '.h5')

except:
    pass # initialize and save initial model here since it was not yet created

# TRAINING LOOP

# do training until set goal is completed or the pc needs some rest
while verify_continue(now, shutdown_timer, progress, progress_target):
    
    # Since one milestone is defined as 625 episodes we do that many and then store data appropriately
    for episode in tqdm(range(625),desc='Progress to next milestone ' + str(progress) + ':'):
        pass # do episode and training here properly
    
    # After a training milestone is hit we update the porgress tuple
    if progress + 625 == episodes_per_epoch:
        progress = (progress[0]+1, 0) # we increment the epoch and set episodes to 0 since we have reached a new epoch
    else:
        progress = (progress[0],progress[1]+625) # we increment the episode counter of progress since we are not in the next epoch yet
    
    # save model weights and progress
    model.save('./weights/milestone_' + str(progress[1]) + '_' + str(progress[0]) + '.h5')
    save_progress(progress)
    
    # ADD
    
# termination routine to ensure safe termination
cleanup_milestones(progress) # deletes weight files in-between epochs except the current one if present

# shutdown if allowed
if pc_care:
    os.system("shutdown -s -t 300") # schedule a shutdown that can be stopped with cmd("shutdown /a") if necessary.