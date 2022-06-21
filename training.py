import chess

import numpy as np
import pandas as pd

import tensorflow as tf


from model.mcts import *
from lib.engine import *
from model.model import *

# training parameters
epoch_goal = 100
episodes_per_epoch = 3125 # 100.000 games per epoch since 32 batches are used
batch_size = 32

# preparation functions

def validate_dependancies():
    pass

def load_progress():
    pass

# hyperparameters
epsilon = 0.005


# training preparation
progress = (0, 0)
progress_target = (100,3125)

model = ChessModel()
# load weights and more info


# training functions

def append_milestone():
    pass

def do_episode():
    pass

# training loop
for epoch in range(epoch_goal):
    # do epoch only if it has not been done before
    if progress[0] > epoch+1:
        for episode in range(episodes_per_epoch):
            if progress[1] > (episode+1):
                pass