import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from gym import spaces
from IPython.display import clear_output


class DQN:

    """ Deep Q Network """

    def __init__(self, action_space, observation_space, params):

        self.action_space = action_space
        self.observation_space = observation_space
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

class DQNAGENT():
    
    def __init__(self, env, processor, render):
        self.env = env
        # Enable state space reduction and reward enhancements
        self.processor = processor
        self.env.observation_space = self.processor.observation_space
        self.render = render
        
        
    def train(self):
        params = dict()
        params['name'] = None
        params['epsilon'] = 1
        params['gamma'] = .95
        params['batch_size'] = 200
        params['epsilon_min'] = .01
        params['epsilon_decay'] = .995
        params['learning_rate'] = 0.00025
        params['layer_sizes'] = [128, 128, 128 ,128]
        episode = 10000

        agent = DQN(action_space=self.env.action_space, observation_space=self.env.observation_space, params=params)
        high_score = 0
        for e in range(episode):
            observation = self.env.reset()
            state = self.processor.process_state(observation, {"direction": self.env.actions[0]})

            max_steps = 10000
            for i in range(max_steps):
                action = agent.act(state)
                observation, score, done, info = self.env.step(self.env.actions[action])
                reward = self.processor.process_reward(observation, score, done, info)
                next_state = self.processor.process_state(observation, info)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if self.render:
                    self.env.render()
                if params['batch_size'] > 1:
                    agent.replay()
                if done:
                    clear_output()
                    if score > high_score:
                        high_score = score
                    print(f"Episode {e}/{episode}, Score: {score}, HighScore: {high_score}")
                    break

    