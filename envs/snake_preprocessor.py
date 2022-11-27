import numpy as np
from gym import spaces

from envs.python_snake_env import SnakeEnv

class FullState:

    def __init__(self) -> None:
        # variables for reward enhancement
        self.prev_score = 0
        self.observation_space = spaces.Discrete(40)

    def process_state(self, state, info):
        processed_s = state       
        snake_direction = [int(info["direction"] == 'UP'), int(info["direction"] == 'RIGHT'), int(info["direction"] == 'DOWN'), int(info["direction"] == 'LEFT')]
        processed_s = np.append(np.ravel(processed_s), snake_direction)
        processed_s = np.reshape(processed_s, (1, len(processed_s)))
        return processed_s

    def process_reward(self, state, score, is_terminal, info):
        processed_r = 0
        
        if is_terminal:
            processed_r = 0
        else:
            processed_r = score - self.prev_score
        
        self.prev_score = score

        return processed_r


class ReduceState:

    def __init__(self) -> None:
        # variables for reward enhancement
        self.prev_score = 0
        self.observation_space = spaces.Discrete(12)

    def process_state(self, state, info):
        processed_s = state
        field_size = len(state)

        for y in range(field_size):
            for x in range(field_size):
                if state[y][x] == SnakeEnv.APPLE:
                    apple_x = x
                    apple_y = y
                elif state[y][x] == SnakeEnv.HEAD:
                    snake_x = x
                    snake_y = y
        
        snake_direction = info["direction"]

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
        processed_s = [int(snake_y < apple_y), int(snake_x < apple_x), int(snake_y > apple_y), int(snake_x > apple_x), \
                    wall_up, wall_right, wall_down, wall_left, \
                    int(snake_direction == 'UP'), int(snake_direction == 'RIGHT'), int(snake_direction == 'DOWN'), int(snake_direction == 'LEFT')]
        

        processed_s = np.reshape(processed_s, (1, len(processed_s)))
        return processed_s

    def process_reward(self, state, score, is_terminal, info):
        processed_r = 0
        
        if is_terminal:
            processed_r = 0
        else:
            processed_r = score - self.prev_score
        
        self.prev_score = score

        return processed_r


    # def _measure_distance(self, observation):
    #     snake_x = int(observation["head_x"])
    #     snake_y = int(observation["head_y"])
    #     apple_x = int(observation["food_x"])
    #     apple_y = int(observation["food_y"])  
    #     return round(math.sqrt((snake_x-apple_x)**2 + (snake_y-apple_y)**2), 8)