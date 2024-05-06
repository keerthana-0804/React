import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from win32gui import SetWindowPos
import tkinter as tk
print("THE GYM VERSION IS :",gym.__version__)
class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.collected_items = []
        self.specified_items=[]
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.graph = None  # Graph representing the grid environment

             # Initialize item locations
        self._initialize_item_locations()

    def _initialize_item_locations(self):
        # Get the initial agent location
        agent_location = (0, 0)

        # List to store already assigned locations
        assigned_locations = []

        # Randomly place items in the grid
        for item_name in ["wood", "iron", "grass", "toolshed", "gold"]:
            item_location = np.random.randint(0, self.size, size=2)
            
            # Ensure the new location doesn't overlap with the agent's initial location
            while np.array_equal(item_location, agent_location):
                item_location = np.random.randint(0, self.size, size=2)
            
            # Ensure the new location doesn't overlap with already assigned locations
            while item_location.tolist() in assigned_locations:
                item_location = np.random.randint(0, self.size, size=2)
            
            # Update the assigned locations list
            assigned_locations.append(item_location.tolist())
            
            # Set the location of the item
            setattr(self, f"_{item_name}_location", item_location)