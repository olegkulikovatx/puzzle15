import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import progressbar

from timeit import default_timer as timer
import time as t

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, BatchNormalization, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam


class Enviroment15:
    def __init__(self, initial_state = None):
        self.state = ""
        self.num_actions = 4
        self.state_size = 5 * 16 
        self.action_counter = 0
        self.reset()
        self.improve_limit = 64
        if initial_state is not None:
            self.state = initial_state
        self.where0 = self.find0()
        self.best_score = self.score()
        ##             North       East      South      West
        self.moves = {0:[-1, 0], 1:[0, 1], 2:[1, 0], 3: [0, -1]}
    
    def find0(self):
        where0 = np.where(self.state==0)
        return np.array([where0[0][0], where0[1][0]])
    
    def find_tile(self, tile):
        where = np.where(self.state==tile)
        return np.array([where[0][0], where[1][0]])
    
    def place_tile(self, tile, y, x):
        tile_yx = self.find_tile(tile)
        old_y = tile_yx[0]
        old_x = tile_yx[1]
        self.state[old_y, old_x] = self.state[y, x]
        self.state[y, x] = tile
        
    def move0(self, y, x):
        n = 0
        e = 1
        s = 2
        w = 3
        tile_yx = self.find0()
        new_y = tile_yx[0]
        new_x = tile_yx[1]        
        actions = []
        
        ## Move to new_y = y +- 1
        if y == 0:
            while new_y > 1:
                actions.append(n)
                new_y -= 1
            if new_y == 0:
                actions.append(s)
                new_y += 1
        else:
            while new_y > y - 1:
                actions.append(n)
                new_y -= 1
            while new_y < y - 1:
                actions.append(s)
                new_y += 1
        ## Move to new_x = x
        while new_x > x:
            actions.append(w)
            new_x -= 1
        while new_x < x:
            actions.append(e)
            new_x += 1
        
        ## Move to new_y = y
        if y == 0:
            actions.append(n)
            new_y -= 1
        else:
            actions.append(s)
            new_y += 1
        
        return actions
    
    def move_tile(self, y, x):
        pass
    
    def render(self):
        print(self.state)
                
    
    def sample(self):
        '''
        Return random action
        0 - North
        1 - East
        2 - South
        3 - West
        '''
        action = np.random.randint(0, 4)
        while(not self.is_valid_action(action)):
            action = np.random.randint(0, 4)
        return action
    
    def is_valid_action(self, action):
        new0 = self.where0        
        new0 = new0 + np.array(self.moves[action])
        if new0.max() > 3 or new0.min() < 0:
            return False
        else:
            return True
    
    def get_state_vector(self):        
        ## Build 5 x 16 matrix A:
        ## A[0, min_placed] = 1 if all tiles min_placed and greater are in target locations 15
        ## A[1, i] = 1 if tile (min_placed - 1) is in cell i 15
        ## A[2, i] = 1 if tile (min_placed - 2) is in cell i 15
        ## A[3, i] = 1 if tile (min_placed - 3) is in cell i 15 
        ## A[4, i] = 1 if tile 0 is in cell i  16
        ## Then flatten
        ## Total number of observations: 16 * 15^4 = 810_000
        arr = self.state.reshape(16)
        vector = np.zeros((5,16))
        min_placed = 16
        for i in range(15, -1, -1):
            if arr[i] != i:
                min_placed = i + 1
                break
        if min_placed < 16:
            vector[0, min_placed] = 1
        for i in range(min_placed - 1, min_placed - 4, -1):
            if i >= 0:
                #####
                where_i = np.where(arr==i)
                if len(where_i) == 0:
                    self.render()
                #####
                where_i = np.where(arr==i)[0][0]
                vector[min_placed - i, where_i] = 1
        where_0 = np.where(arr==0)[0][0]
        vector[4, where_0] = 1
            
        vector = vector.reshape(5*16)
        return vector
    
    def step(self, action, debug=True):
        next_state, reward, terminated, info = (0,0,False,"Legal move")
        self.action_counter += 1
        
        ##Update where0
        self.where0 = self.find0()
        
        if not self.is_valid_action(action):
            ## illegal move
            ## The same state
            next_state = self.get_state_vector()
            ## High penalty
            reward = -100
            ## Terminate
            #terminated = True
            info = "Illegal move. Terminated"
            sign = '!'
        else:
            old_score = self.score()
                       
            ## next_state            
            new0 = self.where0        
            new0 = new0 + np.array(self.moves[action]) 
            
            ## Move a number to the empty space 
            self.state[self.where0[0], self.where0[1]] = self.state[new0[0], new0[1]]
            self.state[new0[0], new0[1]] = 0
            self.where0 = new0
                                               
            next_state = self.get_state_vector()
            
            ##reward
            sign = '+'
            new_score = self.score()
            if new_score > old_score:
                sign = '+'
                reward = new_score - old_score                
            else:
                sign = '-'
                reward = new_score - old_score
                
            ##extra rewards and termination
            if self.total_distance() == 0:
                ## Puszzle solved
                reward += 150
                sign = '$'
                terminated = True
                info = "Puzzle solved"
            elif new_score > self.best_score:
                ##New Best score
                reward += 1
                sign = '^'
                terminated = False
                ##update best score and reset counter
                self.best_score = new_score
                self.action_counter = 0 
                
        if self.action_counter > self.improve_limit:
            ## could not improve the best score
            ## after max number of iterations. Game lost
            #reward -= 1
            terminated = True
            info = "Iteration limit reached"

        if debug:
            print(sign, end='', flush=True)
            
        return next_state, reward, terminated, info 
    
    def score(self):
        arr = self.state.reshape(16)
        #score = -1 * abs(arr - np.array(range(0,16))).sum()
        score = 0
        max_unplaced = 0
        for i in range(15, 0, -1):
            if arr[i] == i:
                score += i * 2
            else:
                max_unplaced = i 
                break
                
        ## extra bonus for completeng a row
        if max_unplaced < 12:
            score += 120
        if max_unplaced < 8:
            score += 120
        if max_unplaced < 4:
            score += 120
            
        ## extra bonut for being close
        where12 = self.find_tile(12)
        ## x == 0, y ==2
        if (where12[0] == 2 and where12[1] == 0 and max_unplaced == 12):
            score += 10
        
        
        actual_i = np.where(self.state==max_unplaced)
        
        actual_x   = actual_i[0][0]
        actual_y   = actual_i[1][0]        
        expected_y   = max_unplaced % 4
        expected_x   = int(max_unplaced / 4)        
        #print(f"max_unplaced {max_unplaced}, expected {expected_x},{expected_y} actual {actual_x} {actual_y}")
        diff = abs(expected_x-actual_x) + abs(expected_y-actual_y)
        score -= diff
            
        return score
    
    def total_distance(self):
        arr = self.state.reshape(16)
        d = abs(arr - np.array(range(0,16))).sum()
        return d
    
    def action_name(self, action):
        names = {0: "North", 1:"East", 2: "South", 3:"West"}
        return names[action]        
    
    def reset(self):
        arr = np.array(range(0,16))
        np.random.shuffle(arr)
        self.state = arr.reshape(4,4)
        self.action_counter = 0
        self.best_score = self.score()
        self.where0 = self.find0()
        vector = self.get_state_vector()
        return vector


    def init_state(self):
        '''
        All tiles in the right place
        '''
        arr = np.array(range(0,16))
        self.state = arr.reshape(4,4)
        self.action_counter = 0
        self.best_score = self.score()
        self.where0 = self.find0()
        vector = self.get_state_vector()
        return vector
        
    
    def shuffle_state(self, nmoves: int): 
        '''
        Randomly move 0 tile 'nmoves' times
        '''
        arr = self.state
        moves = [[0,-1],[0,1],[-1,0],[1,0]]
        remaining_moves = nmoves
        while remaining_moves > 0:
            m = moves[np.random.randint(4)]
            
            where0 = np.where(arr==0)
            x = where0[0][0]
            y = where0[1][0]
            new_x = x + m[0]
            new_y = y + m[1]
            if 0 <= new_x <= 3  and 0 <= new_y <= 3:
                arr[x][y] = arr[new_x][new_y]
                arr[new_x][new_y] = 0
                remaining_moves -= 1
                self.state = arr.reshape(4,4)
        self.action_counter = 0
        self.best_score = self.score()
        self.where0 = self.find0()
        vector = self.get_state_vector()
        return vector


class Agent:
    def __init__(self, enviroment, optimizer):
        
        # Initialize atributes
        self._state_size = enviroment.state_size
        self._action_size = enviroment.num_actions
        self._optimizer = optimizer
        self._lr = 0.7
        self._enviroment = enviroment
        
        self.expirience_replay = deque(maxlen=100_000)
        
        self.initializer = tf.keras.initializers.HeNormal()
        
        # Initialize discount and exploration rate
        self.gamma = 0.9
        self.epsilon = 0.1
        
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.align_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append([state, action, reward, next_state, terminated])
    
    def _build_compile_model(self):
        model = Sequential()

        #model.add(InputLayer(input_shape=(self._state_size,)))  
        
        model.add(Dense(256, input_shape=(self._state_size,), activation='relu', kernel_initializer=self.initializer))
        model.add(Dense(128, activation='relu', kernel_initializer=self.initializer))
        model.add(Dense(64, activation='relu', kernel_initializer=self.initializer))
        model.add(Dense(self._action_size, activation='linear', kernel_initializer=self.initializer))
                
        model.compile(loss=tf.keras.losses.Huber(), optimizer=self._optimizer, metrics=['accuracy'])
        return model

    def align_target_model(self, debug = True):
        self.target_network.set_weights(self.q_network.get_weights())
        if debug:
            print("A", end='', flush=True)
    
    def act(self, state, do_rand=True):
        if do_rand:
            r = np.random.rand()
            #print("r={}, e={}".format(r, self.epsilon))
            if r <= self.epsilon:
                action = self._enviroment.sample()
                #print("Random {}".format(action))
                return action        
        #print(state.shape)            
        q_values = self.q_network.predict(state, verbose=0)
        ## take an action according to the probablily, not just max()
        #action = self.get_action_by_probability(q_values[0])
        action = np.argmax(q_values[0])
        #print("Predicted {}".format(action))
        return action
    
    def get_action_by_probability(self, w):
        tolerance = 1e-6

        m = w.min()
        w = w - m
        w += np.abs(m)
        w += tolerance
        w /= sum(w)
        #print(w)
        ranges = {}
        left = 0
        for i in range(len(w)-1):
            right = left + w[i]
            ranges[i]=(left, right)
            left = right

        ranges[len(w-1)] = (left, 1.0)
        #print(ranges)
        r = np.random.rand()
        for i in range(len(w)-1):
            if r >= ranges[i][0] and r < ranges[i][1]:
                return i
        return len(w)-1

    def load_models(self, path):
        print("loading models from {}".format(path))
        self.target_network = tf.keras.models.load_model(path)
        print("Target model loaded")
        self.q_network = tf.keras.models.load_model(path)
        print("Q model loaded")
        
    def old_retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)
        X = []
        Y = []
        
        for state, action, reward, next_state, terminated in minibatch:
            state = np.reshape(state, (-1, self._enviroment.state_size))
            target = self.q_network.predict(state, verbose=0)
            
            if terminated:
                max_future_q = reward
            else:                
                t = self.target_network.predict(next_state, verbose=0)
                ## TODO : Use probabylty[i] * t[i] 
                max_future_q = reward + self.gamma * np.amax(t)
                
            target[0][action] = (1 - self._lr) * target[0][action] + self._lr * max_future_q
            
            X.append(state)
            Y.append(target)
        
        
        X = np.array(X).reshape(-1, self._enviroment.state_size)
        Y = np.array(Y).reshape(-1, self._enviroment.num_actions)
        
        self.q_network.fit(X, Y, batch_size=batch_size, verbose=0)
            
    def retrain(self, batch_size, debug = True):
        minibatch = random.sample(self.expirience_replay, batch_size)

        states       = np.array([transition[0] for transition in minibatch], dtype = np.int32)
        actions      = np.array([transition[1] for transition in minibatch], dtype = np.int32)
        rewards      = np.array([transition[2] for transition in minibatch], dtype = np.float32)
        next_states  = np.array([transition[3] for transition in minibatch], dtype = np.int32)        
        terminateds  = np.array([transition[4] for transition in minibatch])

        states  = states.reshape(-1, self._enviroment.state_size)
        next_states  = states.reshape(-1, self._enviroment.state_size)
        targets = self.q_network.predict(states, verbose=0) 
        ts = self.target_network.predict(next_states, verbose=0)

        Y = np.array([]) 
        for i, (state, action, reward, next_state, terminated) in enumerate(minibatch):
            #state         = states[i]
            #next_state    = next_states[i]
            #reward        = rewards[i]
            action        = int(action)
            terminated    = terminateds[i]
            target        = targets[i]

            if terminated:
                max_future_q = reward
            else:                
                t = ts[i]                
                max_future_q = reward + self.gamma * np.amax(t)

            target[action] = (1 - self._lr) * target[action] + self._lr * max_future_q
            
            target = np.array(target).reshape(-1, self._enviroment.num_actions)
            #self.q_network.fit(state, target, batch_size=1, verbose=0, shuffle=True, epochs=1) 
            
            Y = np.append(Y, target)

        Y = Y.reshape(batch_size, self._enviroment.num_actions)
        self.q_network.fit(states, Y, batch_size=batch_size, verbose=0, shuffle=True, epochs=1)
        if debug:
            print("T", end='', flush=True)
        
        return 0
            
        

def train(agent, num_of_episodes, timesteps_per_episode, epsilon, nmoves,
          min_epsilon, max_epsilon, batch_size, decay, improve_limit=64):
    
    start = timer()    

    history = []
    infos   = []
    total_steps = 0
    enviroment = Enviroment15()

    bar = progressbar.ProgressBar(maxval=num_of_episodes+1, widgets=\
    [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]) 
    bar.start()
    for e in range(0, num_of_episodes):
        # Reset the enviroment
        #state = enviroment.reset()
        enviroment.init_state()
        state = enviroment.shuffle_state(nmoves)
        state = np.reshape(state, (-1, enviroment.state_size))
        enviroment.improve_limit = improve_limit
        
        debug = (e < 10 or e >= num_of_episodes - 10)
        if debug:
            print("Original state")
            enviroment.render()
        # Initialize variables
        reward = 0
        terminated = False
        num_trained = 0
        num_aligned = 0

        actions_peformed = np.array([0,0,0,0]) 
        
        start_score = enviroment.score()
        end_score = start_score
        
        info = "-"

        for timestep in range(timesteps_per_episode):
            agent.epsilon = epsilon
            total_steps += 1

            # Run Action
            action = agent.act(state)        
            actions_peformed[action] += 1

            # Take action    
            next_state, reward, terminated, info = enviroment.step(action, debug)        
            next_state = np.reshape(next_state, (-1, enviroment.state_size))
            agent.store(state, action, reward, next_state, terminated)

            state = next_state
            end_score = enviroment.score()

            if terminated:                                
                break

            if len(agent.expirience_replay) > batch_size:
                if total_steps % 16 == 0:
                    agent.retrain(batch_size, debug)
                    num_trained += 1
            if total_steps % 2048 == 1:
                agent.align_target_model(debug)  
                num_aligned += 1

        if debug:
            print("\n{} done in {} steps. {}. Start score {} End score {}. epsilon {}".format(
                    e+1, timestep + 1, info, start_score, end_score, epsilon))        
            if (e + 1) % 1 == 0:
                print("**********************************")
                print("Episode: {}  epsilon {}".format(e, epsilon))
                enviroment.render()
                print ("actions: {} last action No {} ".format(actions_peformed, enviroment.action_name(action), epsilon))
                print("Retrained {} times aligned {} times".format(num_trained, num_aligned))
                print("**********************************")            
            
        if (e + 1) % 1000 == 0:             
            agent.q_network.save("15_latest.keras")
            
        history.append(timestep+1)
        infos.append(info)
        
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * e)
        bar.update(e + 1)
    bar.finish()
        
    stop = timer()
    print ("{} seconds elapsed".format(stop-start))
    agent.q_network.save("15_latest.keras")
    return history, infos    


def test(agent, enviroment, num_of_episodes, nmoves, show=False,
         improve_limit=64):
    total_epochs = 0
    total_penalties = 0
    total_solved = 0
    for _ in range(num_of_episodes):
            # Reset the enviroment
        enviroment.init_state()
        state = enviroment.shuffle_state(nmoves)
        state = np.reshape(state, (-1, enviroment.state_size))
        enviroment.improve_limit = improve_limit

        # Initialize variables
        reward = 0
        terminated = False
        epochs = 0
        penalties = 0
        reward = 0

        terminated = False
        print("Episode: {}.".format(_), flush=True)
        #enviroment.render()
        while not terminated:
            action = agent.act(state)
            next_state, reward, terminated, info = enviroment.step(action)

            next_state = np.reshape(next_state, (-1, enviroment.state_size))

            state = next_state

            if reward < -10:
                penalties += 1

            epochs += 1
            if show:
                clear_output(wait=True)
                enviroment.render()
                t.sleep(0.25)

            if (epochs > 1024):
                penalties += 1
                break
        if enviroment.total_distance() == 0:
            total_solved += 1
            
        total_penalties += penalties
        total_epochs += epochs
        print("Episode: {} {} epochs {} penalties".format(_, epochs, penalties), flush=True)
        #print("Final state:".format(_), flush=True)
        #enviroment.render()        

    print("**********************************")
    print("Results")
    print("**********************************")
    print("Epochs per episode: {}".format(total_epochs / num_of_episodes))
    print("Penalties per episode: {}".format(total_penalties / num_of_episodes))    
    print(f"Solved: {total_solved} of {num_of_episodes}")
    