# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:07:15 2024

@author: the_f
"""

from tensorflow.keras.optimizers import Adam

from Puzzle15.Puzzle15 import Enviroment15
from Puzzle15.Puzzle15 import Agent

enviroment = Enviroment15()
optimizer = Adam(learning_rate=0.001)
agent = Agent(enviroment, optimizer)

from Puzzle15.Puzzle15 import train

nmoves=31

#agent.load_models('15_latest.keras')

improve_limit = 64
#agent.load_models('15_latest.keras')
#%run -i "15_saved_history.py"

min_epsilon = 0.1
max_epsilon = 0.9
decay = 0.0009
epsilon = max_epsilon
num_of_episodes = 5000 
timesteps_per_episode = 1000
batch_size = 32

history, infos = train(agent, num_of_episodes, timesteps_per_episode, epsilon, nmoves, 
                           min_epsilon, max_epsilon, batch_size, decay, improve_limit=improve_limit)
agent.q_network.save('15_after_latest_continue.keras')
