import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from os import environ

# Suppress the Pygame support prompt
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._load_resources()

        # Define observation and action spaces
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {0: [1, 0], 1: [0, 1], 2: [-1, 0], 3: [0, -1]}

        self._initialize_item_locations()

    def _load_resources(self):
        pygame.mixer.init()
    
        # Load images and sounds
        self.images = {
            "wood": pygame.image.load("minewood.png"),
            "iron": pygame.image.load("mineiron.png"),
            "grass": pygame.image.load("minegrass.png"),
            "toolshed": pygame.image.load("mineshed.png"),
            "gold": pygame.image.load("gold.png"),
            "agent": pygame.image.load("mineplayer.png"),
            "background": pygame.image.load("minewood.png")
        }
        self.sounds = {
            "collect": pygame.mixer.Sound("sample1.wav"),
            "move": pygame.mixer.Sound("sample2.wav")
        }

    def _initialize_item_locations(self):
        self.item_locations = {
            "wood": self._random_location(),
            "iron": self._random_location(),
            "grass": self._random_location(),
            "toolshed": self._random_location(),
            "gold": self._random_location()
        }

    def _random_location(self):
        return np.random.randint(0, self.size, size=2)

    def reset(self):
        self._initialize_item_locations()
        self.agent_location = np.array([0, 0])
        self.target_location = self.item_locations["gold"]
        return self._get_obs()

    def step(self, action):
        self.sounds['move'].play()
        direction = self._action_to_direction[action]
        new_agent_location = np.clip(self.agent_location + direction, 0, self.size - 1)
        self.agent_location = new_agent_location
        reward, terminated = self._check_item_collection()
        return self._get_obs(), reward, terminated, {}

    def _check_item_collection(self):
        reward = 0
        terminated = False
        for item, location in self.item_locations.items():
            if np.array_equal(self.agent_location, location):
                self.sounds['collect'].play()
                reward = 1
                self.item_locations[item] = None  # Remove the item
                if item == "gold":
                    terminated = True
                break
        return reward, terminated

    def _get_obs(self):
        return {"agent": self.agent_location, "target": self.target_location}

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window.get_size())
        canvas.blit(self.images['background'], (0, 0))
        self._draw_grid(canvas)
        self._draw_items(canvas)
        self._draw_agent(canvas)
        self.window.blit(canvas, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])

    def _draw_items(self, canvas):
        pix_square_size = self.window_size // self.size
        for item, location in self.item_locations.items():
            if location is not None:
                item_image = pygame.transform.scale(self.images[item], (pix_square_size, pix_square_size))
                canvas.blit(item_image, (location[0] * pix_square_size, location[1] * pix_square_size))

    def _draw_agent(self, canvas):
        pix_square_size = self.window_size // self.size
        agent_image = pygame.transform.scale(self.images['agent'], (pix_square_size, pix_square_size))
        canvas.blit(agent_image, (self.agent_location[0] * pix_square_size, self.agent_location[1] * pix_square_size))

    def close(self):
        pygame.quit()

# Example usage
env = GridWorldEnv(render_mode='human')
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
