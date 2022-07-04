from .base_agent import Agent

import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import time
import math


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
        
        # variables for reward enhancement
        self.prev_dist = None
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
        episode = 1000

        sum_of_rewards = []
        agent = DQN(self.env, params)
        for e in range(episode):
            observation = self.env.reset()
            self.prev_dist = self._measure_distance(observation)
            self.prev_length = 0
            state = self._enhance_state(observation)
            state = np.reshape(state, (1, self.env.observation_space.n))
            score = 0
            max_steps = 10000
            for i in range(max_steps):
                action = agent.act(state)
                prev_state = state
                observation, reward_indicators, done, _ = self.env.step(self.ACTIONS[action])
                reward = self._enhance_reward(observation, reward_indicators, done)
                score += reward
                next_state = self._enhance_state(observation)
                next_state = np.reshape(next_state, (1, self.env.observation_space.n))
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if params['batch_size'] > 1:
                    agent.replay()
                if done:
                    print(f'final state before dying: {str(prev_state)}')
                    print(f'episode: {e+1}/{episode}, score: {score}')
                    break
            sum_of_rewards.append(score)
        return sum_of_rewards
    
    def _enhance_state(self, observation):
        snake_x = int(observation["head_x"])
        snake_y = int(observation["head_y"])
        apple_x = int(observation["food_x"])
        apple_y = int(observation["food_y"])
        field_size = int(observation["field_size"])
        snake_direction = observation["direction"]

        # wall check
        if snake_y == 0:
            wall_up, wall_down = 1, 0
        elif snake_y == (field_size - 1):
            wall_up, wall_down = 0, 1
        else:
            wall_up, wall_down = 0, 0
        if snake_x == (field_size - 1):
            wall_right, wall_left = 1, 0
        elif snake_x == 0:
            wall_right, wall_left = 0, 1
        else:
            wall_right, wall_left = 0, 0
        
        # observation: apple_down, apple_left, apple_up, apple_right, obstacle_up, obstacle_right, obstacle_down, obstacle_left, direction_up, direction_right, direction_down, direction_left
        state = [int(snake_y < apple_y), int(snake_x < apple_x), int(snake_y > apple_y), int(snake_x > apple_x), \
                    wall_up, wall_right, wall_down, wall_left, \
                    int(snake_direction == 'UP'), int(snake_direction == 'RIGHT'), int(snake_direction == 'DOWN'), int(snake_direction == 'LEFT')]
        
        return state
    
    def _measure_distance(self, observation):
        snake_x = int(observation["head_x"])
        snake_y = int(observation["head_y"])
        apple_x = int(observation["food_x"])
        apple_y = int(observation["food_y"])  
        return round(math.sqrt((snake_x-apple_x)**2 + (snake_y-apple_y)**2), 8)
    
    def _enhance_reward(self, obs, reward_indicators, done):
        reward = 0
        dist = self._measure_distance(obs)
        length = int(reward_indicators["length"])
        
        if done:
            reward = -50
        else:
            if length > self.prev_length:
                print("ate food")
                reward = 10
            else:
                if dist < self.prev_dist:
                    reward = 1
                else:
                    self.prev_dist = dist
                    reward = -1
        
        self.prev_dist = dist
        self.prev_length = length
        
        return reward
    
    # def _enhance_reward(self, obs, reward_indicators, done):
    #     reward = 0
    #     length = int(reward_indicators["length"])
        
    #     if done:
    #         reward = -50
    #     else:
    #         if length > self.prev_length:
    #             print("ate food")
    #             reward = 20
    #         else:
    #             reward = -1
        
    #     self.prev_length = length
        
    #     return reward
    
    
    def test(self):
        print("Test not implemented yet")
                
    def load_model(self):
        print("Load model not implemented yet")
    
    def store_model(self):
        print("Store model not implemented yet")
    