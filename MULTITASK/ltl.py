import numpy as np
import pygame
import os
import gymnasium as gym
from gymnasium import spaces
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from sympy import symbols, And, Or, simplify_logic
from sympy.abc import F

# Define symbols for each item
wood, iron, grass, toolshed, gold = symbols('wood iron grass toolshed gold')

def evaluate_task(simplified_task, items_list):
    # Split the simplified expression into terms based on logical operators
    terms = str(simplified_task).split('&')

    for term in terms:
        # Remove whitespace and parentheses
        term = term.strip().replace('(', '').replace(')', '')

        # If term contains '|', it indicates OR condition
        if '|' in term:
            # Split the term based on '|'
            sub_terms = term.split('|')
            # Check if any sub-term is already present in items_list
            found = False
            for sub_term in sub_terms:
                if sub_term.strip() in items_list:
                    found = True
                    break
            # If none of the sub-terms are present in items_list, add the first sub-term
            if not found:
                items_list.append(sub_terms[0].strip())
        else:
            # Add the term to the items_list
            items_list.append(term.strip())

    return items_list

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.collected_items = []
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
        
        # Set initial agent location
        self._agent_location = agent_location

    def get_location(self, item):
        return getattr(self, f"_{item}_location")

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "gold": self._gold_location,
            "wood": self._wood_location,
            "iron": self._iron_location,
            "grass": self._grass_location,
            "toolshed": self._toolshed_location,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        new_agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # Check if the new location contains an item
        item_type = None
        if np.array_equal(new_agent_location, self._wood_location):
            item_type = "wood"
        elif np.array_equal(new_agent_location, self._iron_location):
            item_type = "iron"
        elif np.array_equal(new_agent_location, self._grass_location):
            item_type = "grass"
        elif np.array_equal(new_agent_location, self._toolshed_location):
            item_type = "toolshed"
        elif np.array_equal(new_agent_location, self._gold_location):
            item_type = "gold"

        # Remove the item from the environment if it exists
        if item_type is not None:
            print("Collected ", item_type)
            self.collected_items.append(item_type)
            setattr(self, f"_{item_type}_location", None)
            reward = 1  # You can assign a reward for picking up the item
        else:
            reward = 0

        # Update agent's location
        self._agent_location = new_agent_location

        # Check if the episode is terminated
        terminated = False
        if np.array_equal(self._agent_location, self._toolshed_location):
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info

    def render(self, mode="human", close=False):
        if mode == "rgb_array":
            # No point rendering the environment just to return an image of it
            raise NotImplementedError

        assert mode == "human"
        if close:
            pygame.quit()
            return

        # The first time we're called, we need to set up the PyGame window
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
            pygame.display.set_caption("GridWorld")
            self.clock = pygame.time.Clock()

        self._render_frame()
        self.clock.tick(self.metadata["render_fps"])

    def _render_frame(self):
        if self.window is None:
            return  # Avoid rendering if the window is not initialized

        # Clear the window
        self.window.fill((0, 0, 0))

        # Draw the agent
        agent_x, agent_y = self._agent_location
        agent_rect = pygame.Rect(
            agent_x * (self.window_size // self.size),
            agent_y * (self.window_size // self.size),
            self.window_size // self.size,
            self.window_size // self.size,
        )
        pygame.draw.rect(self.window, (0, 255, 0), agent_rect)

        # Draw the target
        target_x, target_y = self._toolshed_location
        target_rect = pygame.Rect(
            target_x * (self.window_size // self.size),
            target_y * (self.window_size // self.size),
            self.window_size // self.size,
            self.window_size // self.size,
        )
        pygame.draw.rect(self.window, (255, 0, 0), target_rect)

        # Draw the items
        for item in ["wood", "iron", "grass", "toolshed", "gold"]:
            location = getattr(self, f"_{item}_location")
            if location is not None:
                x, y = location
                item_rect = pygame.Rect(
                    x * (self.window_size // self.size),
                    y * (self.window_size // self.size),
                    self.window_size // self.size,
                    self.window_size // self.size,
                )
                pygame.draw.rect(self.window, (255, 255, 255), item_rect)

        # Display the frame
        pygame.display.flip()

    def _get_info(self):
        return {"collected_items": self.collected_items}

from sympy.logic.boolalg import And, Or
from sympy import symbols, simplify_logic, satisfiable

# Define LTL formulas
ltl_formulas = {
    "task1": "F(wood & iron)",             # Eventually collect wood and iron
    "task2": "G(toolshed >> F(gold))"      # Always eventually collect gold if the agent reaches the toolshed
}

def evaluate_ltl(formula, collected_items):
    # Define symbols for each collected item
    collected_symbols = [symbols(item) for item in collected_items]

    # Parse the LTL formula string
    ltl_expr = parse_ltl(formula, collected_symbols)

    # Evaluate the LTL formula with the collected items
    result = satisfiable(ltl_expr)

    return result

def parse_ltl(formula, symbols):
    # Replace '&' with 'And' and '|' with 'Or'
    formula = formula.replace('&', ' & ').replace('|', ' | ')

    # Replace item names with corresponding symbols
    for symbol in symbols:
        formula = formula.replace(symbol.name, f'symbols("{symbol.name}")')

    # Parse the LTL formula string
    ltl_expr = eval(formula)

    return ltl_expr

# Create the environment
env = GridWorldEnv(render_mode="human")
# Run the environment loop
done = False
while not done:
    # Agent takes an action
    agent_action = env.action_space.sample()

    # Execute the action in the environment
    observation, reward, done, info = env.step(agent_action)

    # Evaluate LTL formulas
    for formula_name, ltl_formula in ltl_formulas.items():
        if evaluate_ltl(ltl_formula, env.collected_items):
            print(f"LTL formula '{formula_name}' satisfied!")

    # Render environment
    env.render()

# Close the environment
env.close()

