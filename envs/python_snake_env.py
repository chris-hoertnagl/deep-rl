from copy import deepcopy
from gym import Env, spaces
import random
import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
COLORS = [WHITE, BLACK, YELLOW, RED]     
        
                
class Direction:
    UP = 'UP'
    DOWN = 'DOWN'
    RIGHT = 'RIGHT'
    LEFT = 'LEFT'
    DIRECTIONS = [UP, DOWN, LEFT, RIGHT]


class  SnakeEnv(Env):
    APPLE = 3
    HEAD = 2
    BODY = 1
    EMPTY = 0
    
    def __init__(self, tiles:int, tile_size:int):
        super(SnakeEnv, self).__init__()
        self.TILES = tiles
        self.BLOCK_SIZE = tile_size

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.TILES * self.TILES)
        
        self.actions = Direction.DIRECTIONS
        
        self.reset()
        
    def reset(self):
        self.grid = [[SnakeEnv.EMPTY for x in range(self.TILES)] for y in range(self.TILES)]
        
        start_y = random.randint(0 , self.TILES - 1)
        start_x = random.randint(0 , self.TILES - 1)
        self.grid[start_y][start_x] = SnakeEnv.HEAD
        self.snake = [(start_y, start_x)]
        apple_y, apple_x = self.generate_apple()
        self.grid[apple_y][apple_x] = SnakeEnv.APPLE
        self.apple = (apple_y, apple_x)
        
        self.score = 0
        self.terminal = False
        self.last_action = None
        return self.grid
        
    def step(self, action):
        self.action_input(action)
        observation = self.grid
        done = self.terminal
        self.score = len(self.snake)
        score = self.score
        info = {}
        
        return observation, score, done, info
    
    def get_copy(self):
        instance = SnakeEnv(self.TILES, self.BLOCK_SIZE)
        instance.reset()
        instance.grid = deepcopy(self.grid)
        instance.snake = deepcopy(self.snake)
        instance.apple = self.apple
        instance.score = self.score
        instance.terminal = self.terminal
        instance.last_action = self.last_action
        return instance

    def generate_apple(self):
        choices = {(y,x) for x in range(0, self.TILES) for y in  range(0, self.TILES)}
        allowed = list(choices - set(self.snake))
        if allowed:
            return random.choice(allowed)
        else:
            # In case the board is full of snake the game will end anyway
            return random.choice(list(choices))
            
    def action_input(self, direction):
        # TODO: maybe think about movement in opposite directions
        self.move(direction=direction)
        
    def move(self, direction):
        head_y, head_x = self.snake[-1]
        match direction:
            case Direction.UP:
                new_head = (head_y -1, head_x)

            case Direction.DOWN:
                new_head = (head_y + 1, head_x)

            case Direction.RIGHT:
                new_head = (head_y, head_x + 1)

            case Direction.LEFT:
                new_head = (head_y, head_x - 1)
                

        if self.did_wall_crash(new_head):
            self.terminal = True
            return
        
        if self.did_self_crash(new_head):
            self.terminal = True
            return
        
        self.snake.append(new_head)
        self.grid[head_y][head_x] = SnakeEnv.BODY
        head_y, head_x = self.snake[-1]
        self.grid[head_y][head_x] = SnakeEnv.HEAD
        
        if self.did_eat():
            apple_y, apple_x = self.generate_apple()
            self.grid[apple_y][apple_x] = SnakeEnv.APPLE
            self.apple = (apple_y, apple_x)   
        else:
            tail_y, tail_x = self.snake[0]
            self.snake = self.snake[1:]
            self.grid[tail_y][tail_x] = SnakeEnv.EMPTY
        
        self.last_action = direction
    
    def did_eat(self):
        eaten = False
        head_y, head_x = self.snake[-1]
        apple_y, apple_x = self.apple
        if (apple_y == head_y) & (apple_x == head_x):
            eaten = True
        return eaten
    
    def did_wall_crash(self, new_head):
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
    
    def did_self_crash(self, new_head):
        head_y, head_x = new_head
        for body_y, body_x in self.snake:
            if (body_y == head_y) & (body_x == head_x):
                return True
        return False
    
    def render(self):
        screen = pygame.display.set_mode((self.TILES * (self.BLOCK_SIZE + 1), self.TILES * (self.BLOCK_SIZE + 1)))
        pygame.init()
        pygame.event.pump()
        for x in range(self.TILES):
            for y in range(self.TILES):
                color = self.grid[y][x]
                pygame.draw.rect(screen, COLORS[color], [x * self.BLOCK_SIZE, y * self.BLOCK_SIZE, (x + 1) * self.BLOCK_SIZE , (y + 1) * self.BLOCK_SIZE])
        pygame.display.update()