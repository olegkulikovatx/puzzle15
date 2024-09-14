# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:59:17 2024

@author: the_f
"""

#import numpy as np
#import random
#from IPython.display import clear_output
#from collections import deque
#import progressbar

#from timeit import default_timer as timer
#import time as t

#import tensorflow as tf
#from tensorflow import keras

#from tensorflow.keras import Model, Sequential
#from tensorflow.keras.layers import Dense, Embedding, Reshape, BatchNormalization, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam

from Puzzle15.Puzzle15 import Enviroment15
from Puzzle15.Puzzle15 import Agent

enviroment = Enviroment15()
optimizer = Adam(learning_rate=0.001)
agent = Agent(enviroment, optimizer)

from Puzzle15.Puzzle15 import test
test(agent, enviroment, 1, 10, improve_limit=128)

agent.load_models('15_latest.keras')

test(agent, enviroment, 20, 15, improve_limit=128)
