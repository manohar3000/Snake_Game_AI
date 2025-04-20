import pygame, random, cv2
from pygame.math import Vector2
from collections import deque
import numpy as np

# Pygame initial setup (you can also initialize this in main.py)
pygame.init()
cell_size = 40
cell_number = 20
screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))

# --- SNAKE CLASS ---
class Snake:
    def __init__(self):
        self.body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
        self.direction = Vector2(0, 0)
        self.new_block = False

    def draw(self, surface):
        for block in self.body:
            rect = pygame.Rect(int(block.x * cell_size), int(block.y * cell_size), cell_size, cell_size)
            pygame.draw.rect(surface, (176, 166, 0), rect)

    def move(self):
        if self.new_block:
            new_body = self.body.copy()
            new_body.insert(0, self.body[0] + self.direction)
            self.body = new_body[:]
            self.new_block = False
        else:
            new_body = self.body[:-1]
            new_body.insert(0, self.body[0] + self.direction)
            self.body = new_body[:]

    def grow(self):
        self.new_block = True

    def reset(self):
        self.__init__()

# --- FRUIT CLASS ---
class Fruit:
    def __init__(self):
        self.randomize()

    def randomize(self):
        self.pos = Vector2(random.randint(0, cell_number - 1), random.randint(0, cell_number - 1))
    
    def draw(self, surface):
        rect = pygame.Rect(int(self.pos.x * cell_size), int(self.pos.y * cell_size), cell_size, cell_size)
        pygame.draw.rect(surface, (255, 41, 74), rect)

# --- FRAME STACKER CLASS ---
class FrameStacker:
    def __init__(self, max_frames=2):
        self.frames = deque(maxlen=max_frames)

    def add_frame(self, surface):
        screen_array = pygame.surfarray.array3d(surface)
        screen_array = np.transpose(screen_array, (1, 0, 2))
        screen_array = cv2.resize(screen_array, (20, 20))
        gray_image = cv2.cvtColor(screen_array, cv2.COLOR_BGR2GRAY)
        # Convert to binary image: invert values if necessary.
        binary_image = 1 - gray_image // (gray_image.max() if gray_image.max() > 0 else 1)
        self.frames.append(binary_image.astype(float))

    def get_stacked_frames(self):
        # Ensure we always have enough frames.
        while len(self.frames) < self.frames.maxlen:
            if not self.frames:
                self.frames.append(np.zeros((20, 20), dtype=float))
            else:
                self.frames.append(self.frames[-1])
        return np.stack(self.frames, axis=0)

    def reset(self):
        self.frames.clear()

# --- MAIN ENVIRONMENT CLASS ---
class MainEnv:
    def __init__(self):
        self.snake = Snake()
        self.fruit = Fruit()
        self.frame_stacker = FrameStacker()
    
    def draw_elements(self, surface):
        self.fruit.draw(surface)
        self.snake.draw(surface)
    
    def is_collision(self):
        head = self.snake.body[0]
        # Check wall collision
        if not 0 <= head.x < cell_number or not 0 <= head.y < cell_number:
            return True
        # Check self collision
        if head in self.snake.body[1:]:
            return True
        return False

    def play_step(self, action, surface):
        # Define directions: up, right, down, left
        directions = [Vector2(0, -1), Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0)]
        # Simple check to avoid reversing direction:
        if action < 4 and self.snake.direction != -directions[action]:
            self.snake.direction = directions[action]

        self.snake.move()
        self.frame_stacker.add_frame(surface)

        reward = 0
        game_over = self.is_collision()

        if game_over:
            reward = -10
            self.snake.reset()
            self.fruit.randomize()
            self.frame_stacker.reset()
            return self.frame_stacker.get_stacked_frames(), reward, True

        # Check if snake eats fruit
        if self.snake.body[0] == self.fruit.pos:
            reward = 10
            self.snake.grow()
            self.fruit.randomize()

        return self.frame_stacker.get_stacked_frames(), reward, False
