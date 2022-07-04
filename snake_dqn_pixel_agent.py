from .base_agent import Agent

import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import os


class DQN:

    """ Deep Q Network """

    def __init__(self, env, params):

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.epsilon = params['epsilon'] 
        self.gamma = params['gamma'] 
        self.batch_size = params['batch_size'] 
        self.epsilon_min = params['epsilon_min'] 
        self.epsilon_decay = params['epsilon_decay'] 
        self.learning_rate = params['learning_rate']
        self.layer_sizes = params['layer_sizes']
        self.memory = deque(maxlen=2500)
        self.model = self.build_model()


    def build_model(self):
        model = Sequential()
        for i in range(len(self.layer_sizes)):
            if i == 0:
                model.add(Dense(self.layer_sizes[i], input_shape=(self.observation_space.n,), activation='relu'))
            else:
                model.add(Dense(self.layer_sizes[i], activation='relu'))
        model.add(Dense(self.action_space.n, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])


    def replay(self):

        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class DQNAGENT(Agent):
    
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.ACTIONS = ["up", "right", "down", "left"]
        self.training_episodes = 10000
        self.log_path = os.path.join(os.path.dirname(__file__), '..','logs', 'log.txt')
        
        # variables for reward enhancement
        # self.prev_dist = None
        self.prev_length = None
        
        
    def train(self):
        params = dict()
        params['name'] = None
        params['epsilon'] = 1
        params['gamma'] = .95
        params['batch_size'] = 100
        params['epsilon_min'] = .01
        params['epsilon_decay'] = .995
        params['learning_rate'] = 0.00025
        params['layer_sizes'] = [128, 128, 128]

        agent = DQN(self.env, params)
        for e in range(self.training_episodes):
            observation = self.env.reset()
            # self.prev_dist = self._measure_distance(observation)
            self.prev_length = 0
            state = self._enhance_state(observation)
            state = np.reshape(state, (1, self.env.observation_space.n))
            score = 0
            max_steps = 10000
            for i in range(max_steps):
                action = agent.act(state)
                observation, reward_indicators, done, _ = self.env.step(self.ACTIONS[action])
                reward = self._enhance_reward(reward_indicators, done)
                score = reward_indicators["length"]
                next_state = self._enhance_state(observation)
                print_state = next_state
                next_state = np.reshape(next_state, (1, self.env.observation_space.n))
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if params['batch_size'] > 1:
                    agent.replay()
                if done:
                    print(f'final state before dying: \n')
                    print(print_state)
                    print(f'episode: {e+1}/{self.training_episodes}, score: {score}')
                    break
            log_line = f'{e}, {score} \n'
            with open(self.log_path, "a") as file_object:
                file_object.write(log_line)
    
    def _enhance_state(self, observation):
        head_x = int(observation["head_x"])
        head_y = int(observation["head_y"])
        apple_x = int(observation["food_x"])
        apple_y = int(observation["food_y"])
        field_size = int(observation["field_size"])
        percent_size = int(100 / field_size)
        snake_dots = observation["snake_dots"]
       
        # image has axes: image[y][x]
        image = np.full((field_size, field_size), 0)
        
        for dot in snake_dots:
            if len(dot) == 2:
                x = int(dot[0] / percent_size)
                y = int(dot[1] / percent_size)
                if (x <= field_size - 1) & (y <= field_size - 1):
                    image[y][x] = 1
                else:
                    print("Snake POSITION error")
        
        image[apple_y][apple_x] = 3
        if (head_x <= field_size - 1) & (head_y <= field_size - 1):
                image[head_y][head_x] = 2
        else:
            print("Snake HEAD error")

        # print(image)
        state = np.interp(image, (image.min(), image.max()), (0, 1))
        
        return state
    
    def _enhance_reward(self, reward_indicators, done):
        reward = 0
        length = int(reward_indicators["length"])
        
        if done:
            reward = -100
        else:
            if length > self.prev_length:
                print("ate food")
                reward = 30
            else:
                reward = -1
        
        self.prev_length = length
        
        return reward
    
    
    def test(self):
        print("Test not implemented yet")
                
    def load_model(self):
        print("Load model not implemented yet")
    
    def store_model(self):
        print("Store model not implemented yet")
    