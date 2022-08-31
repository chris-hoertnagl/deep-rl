from gym import Env, spaces
import time
import json
import os
import random
import pygame

param_path = os.path.join(os.path.dirname(__file__), 'params.json')

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
    
    FPS = 1
    BLOCK_SIZE = 50
    
    def __init__(self, tiles):
        super(SnakeEnv, self).__init__()
        self.WIDTH = tiles
        self.HEIGHT = tiles
        self.screen = pygame.display.set_mode((self.WIDTH * (SnakeEnv.BLOCK_SIZE + 1), self.HEIGHT * (SnakeEnv.BLOCK_SIZE + 1)))
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(tiles * tiles)
        
        self.reset()
        
    def reset(self):
        pygame.init()
        self.grid = [[SnakeEnv.EMPTY for x in range(self.WIDTH)] for y in range(self.HEIGHT)]
        
        start_x, start_y = self.get_random_position()
        self.grid[start_x][start_y] = SnakeEnv.HEAD
        self.snake = [(start_x, start_y)]
        apple_x, apple_y = self.generate_apple()
        self.grid[apple_x][apple_y] = SnakeEnv.APPLE
        self.apple = (apple_x, apple_y)
        
        self.game_over = False
        self.last_action = None
        return self.grid
        
    def step(self, action):
        self.action_input(action)
        observation = self.grid
        done = self.game_over
        reward = len(self.snake) - 1
        info = {}
        
        return observation, reward, done, info
            
    def get_random_position(self):
        return random.randint(0 , self.WIDTH - 1), random.randint(0 , self.HEIGHT - 1)

    def generate_apple(self):
        x, y = self.get_random_position()
        while ((x,y) == self.snake[-1]):
            x, y = self.get_random_position()
            
        return x,y
            
    def action_input(self, action):
        direction = Direction.DIRECTIONS[action]
        match direction:
            case Direction.UP:
                if self.last_action == Direction.DOWN:
                    self.move(Direction.DOWN)
                else:
                    self.move(Direction.UP)
                return

            case Direction.DOWN:
                if self.last_action == Direction.UP:
                    self.move(Direction.UP)
                else:
                    self.move(Direction.DOWN)
                return

            case Direction.RIGHT:
                if self.last_action == Direction.LEFT:
                    self.move(Direction.LEFT)
                else:
                    self.move(Direction.RIGHT)
                return

            case Direction.LEFT:
                if self.last_action == Direction.RIGHT:
                    self.move(Direction.RIGHT)
                else:
                    self.move(Direction.LEFT)
                return
        
    def move(self, direction):
        head_x, head_y = self.snake[-1]
        match direction:
            case Direction.UP:
                new_head = (head_x, head_y - 1)
                self.snake.append(new_head)

            case Direction.DOWN:
                new_head = (head_x, head_y + 1)
                self.snake.append(new_head)

            case Direction.RIGHT:
                new_head = (head_x + 1, head_y)
                self.snake.append(new_head)

            case Direction.LEFT:
                new_head = (head_x - 1, head_y)
                self.snake.append(new_head)

        if self.did_wall_crash():
            self.game_over = True
            return
        
        self.grid[head_x][head_y] = SnakeEnv.BODY
        
        head_x, head_y = self.snake[-1]
        self.grid[head_x][head_y] = SnakeEnv.HEAD
        
        if self.did_eat():
            print("eat happened")
            apple_x, apple_y = self.generate_apple()
            self.grid[apple_x][apple_y] = SnakeEnv.APPLE
            self.apple = (apple_x, apple_y)            
        else:
            tail_x, tail_y = self.snake[0]
            self.snake = self.snake[1:]
            self.grid[tail_x][tail_y] = SnakeEnv.EMPTY
        
        self.last_action = direction
    
    def did_eat(self):
        eaten = False
        head_x, head_y = self.snake[-1]
        apple_x, apple_y = self.apple
        if (apple_x == head_x) & (apple_y == head_y):
            eaten = True
        return eaten
    
    def did_wall_crash(self):
        head_x, head_y = self.snake[-1]
        if head_x >= self.WIDTH:
            return True
        if head_x < 0:
            return True
        if head_y >= self.HEIGHT:
            return True
        if head_y < 0:
            return True
        return False
    
    def did_self_crash(self):
        pass
    
    def render(self):
        pygame.event.pump()
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                color = self.grid[x][y]
                pygame.draw.rect(self.screen, COLORS[color], [x * SnakeEnv.BLOCK_SIZE, y * SnakeEnv.BLOCK_SIZE, (x + 1) * SnakeEnv.BLOCK_SIZE , (y + 1) * SnakeEnv.BLOCK_SIZE])
        pygame.display.update()