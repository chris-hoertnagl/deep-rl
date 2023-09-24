from copy import deepcopy
from gym import Env, spaces
import numpy as np
import random
from IPython.display import clear_output

class  SnakeEnv(Env):
    APPLE = "A"
    HEAD = "H"
    BODY = "S"
    EMPTY = "O"
    
    UP = 'UP'
    DOWN = 'DOWN'
    RIGHT = 'RIGHT'
    LEFT = 'LEFT'
    
    def __init__(self, tiles:int):
        super(SnakeEnv, self).__init__()
        self.TILES = tiles

        self.actions = [SnakeEnv.UP, SnakeEnv.DOWN, SnakeEnv.LEFT, SnakeEnv.RIGHT]
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(self.TILES * self.TILES)
        
        # Inititialze game
        self.reset()
        
    def reset(self):
        # Initialize empty grid
        self.grid = np.array([[SnakeEnv.EMPTY for x in range(self.TILES)] for y in range(self.TILES)])
        
        # Add snake head to the grid at random position
        start_y = random.randint(0 , self.TILES - 1)
        start_x = random.randint(0 , self.TILES - 1)
        self.grid[start_y][start_x] = SnakeEnv.HEAD
        self.snake = np.array([[start_y, start_x]])
        
        # Add apple at random position
        apple_y, apple_x = self.__generate_apple()
        self.grid[apple_y][apple_x] = SnakeEnv.APPLE
        self.apple = (apple_y, apple_x)
        
        # Initialize game variables & last_action memory
        self.score = 0
        self.terminal = False
        self.last_action = None
        
        return self.grid
        
    def step(self, action):
        # Move the snake
        self.__move(direction=action)
        
        observation = self.grid
        done = self.terminal
        self.score = len(self.snake)
        score = self.score
        info = {"direction": action}
        
        return observation, score, done, info
    
    def get_copy(self):
        # Returns a full copy of the game
        instance = SnakeEnv(self.TILES)
        instance.reset()
        instance.grid = deepcopy(self.grid)
        instance.snake = deepcopy(self.snake)
        instance.apple = self.apple
        instance.score = self.score
        instance.terminal = self.terminal
        instance.last_action = self.last_action
        return instance

    def __generate_apple(self):
        all_grid_positions = {(y,x) for x in range(0, self.TILES) for y in  range(0, self.TILES)}
        allowed_grid_positions = list(all_grid_positions - set([tuple(e) for e in self.snake]))
        if allowed_grid_positions:
            return random.choice(allowed_grid_positions)
        else:
            # In case the board is full of snake the game will end anyway
            return random.choice(list(all_grid_positions))
        
        
    def __move(self, direction):
        head_y, head_x = self.snake[-1]
        match direction:
            case SnakeEnv.UP:
                new_head = (head_y -1, head_x)

            case SnakeEnv.DOWN:
                new_head = (head_y + 1, head_x)

            case SnakeEnv.RIGHT:
                new_head = (head_y, head_x + 1)

            case SnakeEnv.LEFT:
                new_head = (head_y, head_x - 1)
                

        if self.__did_wall_crash(new_head):
            self.terminal = True
            return
        
        if self.__did_self_crash(new_head):
            self.terminal = True
            return
        
        self.snake = np.append(self.snake, [new_head], axis=0)
        self.grid[head_y][head_x] = SnakeEnv.BODY
        head_y, head_x = self.snake[-1]
        self.grid[head_y][head_x] = SnakeEnv.HEAD
        
        if self.__did_eat():
            apple_y, apple_x = self.__generate_apple()
            self.grid[apple_y][apple_x] = SnakeEnv.APPLE
            self.apple = (apple_y, apple_x)   
        else:
            tail_y, tail_x = self.snake[0]
            self.snake = self.snake[1:]
            self.grid[tail_y][tail_x] = SnakeEnv.EMPTY
        
        self.last_action = direction
    
    def __did_eat(self):
        eaten = False
        head_y, head_x = self.snake[-1]
        apple_y, apple_x = self.apple
        if (apple_y == head_y) & (apple_x == head_x):
            eaten = True
        return eaten
    
    def __did_wall_crash(self, new_head):
        head_y, head_x = new_head
        if head_y >= self.TILES:
            return True
        if head_y < 0:
            return True
        if head_x >= self.TILES:
            return True
        if head_x < 0:
            return True
        return False
    
    def __did_self_crash(self, new_head):
        head_y, head_x = new_head
        for body_y, body_x in self.snake:
            if (body_y == head_y) & (body_x == head_x):
                return True
        return False
        
    def render(self, clear:bool = True):
        if clear:
            clear_output(wait=True)
        grid = ""
        for x in range(self.TILES):
            for y in range(self.TILES):
                grid += str(self.grid[x][y]) + " "
            grid += "\n"
        print(grid)