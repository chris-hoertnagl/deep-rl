from time import sleep
from gym import Env, spaces
import event_api_client as api_client
import json
import os

param_path = os.path.join(os.path.dirname(__file__), 'params.json')

class ChromeEnv(Env):

    def __init__(self):
        print("created Env")
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(100)
        
        self.current_observation = None
        self.observation_updated = False
        
        self.api_client = api_client.ApiClient()
        self.api_client.subscribe_state(callback=self._updateObservation)
        
    def _updateObservation(self, client, userdata, message):
        self.current_observation = json.loads(message.payload.decode())
        self.observation_updated = True

    def step(self, action):
        status = self.api_client.publish_action(action=action)
        if status != 0:
            # TODO implement fail logic
            print(f"Failed to send action to topic {action}")
        
        # Wait for state update from the Chrome Extension
        while not self.observation_updated:
            # print("waiting for state")
            pass
        
        observation = self.current_observation["observation"]
        done = self.current_observation["done"]
        reward = self.current_observation["rewardIndicators"]
        info = {}
                
        self.observation_updated = False
        
        return observation, reward, done, info
        
    def reset(self):
        self.current_observation = None
        self.observation_updated = False
        self.api_client.publish_reset()
        
        # Wait for state update from the Chrome Extension
        while not self.observation_updated:
            # print("waiting for reset")
            pass
        
        observation = self.current_observation["observation"]
        self.observation_updated = False
        return observation

    def render(self):
        print("rendered environment")
        
    def close(self):
        self.api_client.close()
        print("Connection closed")
        

        
        
    
