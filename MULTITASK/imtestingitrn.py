import numpy as np
import pygame
import os
import gymnasium as gym
from gymnasium import spaces
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from win32gui import SetWindowPos
import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
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
        self._wood_location = np.random.randint(0, self.size, size=2)
        self._iron_location = np.random.randint(0, self.size, size=2)
        self._grass_location = np.random.randint(0, self.size, size=2)
        self._toolshed_location = np.random.randint(0, self.size, size=2)
        self._gold_location = np.random.randint(0, self.size, size=2)

    def get_location(self, item):
        return getattr(self, f"_{item}_location")

    def set_window_position(self):
        root = tk.Tk()  # create only one instance for Tk()
        root.withdraw()  # keep the root window from appearing
  
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        win_w = 512  # Adjust this according to your window size
        win_h = 512  # Adjust this according to your window size

        x = round((screen_w - win_w) / 2)
        y = round((screen_h - win_h) / 2 * 0.8)  # 80 % of the actual height

        # pygame screen parameter for further use in code
        screen = pygame.display.set_mode((win_w, win_h))

        # Set window position center-screen and on top of other windows
        # Here 2nd parameter (-1) is essential for putting window on top
        SetWindowPos(pygame.display.get_wm_info()['window'], -1, x, y, 0, 0, 1)

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "gold": self._gold_location,
            "wood": self._wood_location,
            "iron": self._iron_location,
            "grass": self._grass_location,
            "toolshed": self._toolshed_location,
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        """ # Randomly place items in the grid
        self._wood_location = np.random.randint(0, self.size, size=2)
        self._iron_location = np.random.randint(0, self.size, size=2)
        self._grass_location = np.random.randint(0, self.size, size=2)
        self._toolshed_location = np.random.randint(0, self.size, size=2) """

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
            setattr(self, f"_{item_type}_location", None)
            reward = 1  # You can assign a reward for picking up the item
        else:
            reward = 0
        
        # Update the agent's location
        self._agent_location = new_agent_location

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
      
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((173, 216, 230))  # Light Blue background

        pix_square_size = self.window_size // self.size

        # Draw items
        item_images = {
            "wood": pygame.image.load("minewood.png"),
            "iron": pygame.image.load("mineiron.png"),
            "grass": pygame.image.load("grass.png"),
            "toolshed": pygame.image.load("mineshed.png"),
            "gold": pygame.image.load("gold.png")
        }

        for item, location in self._get_obs().items():
            if location is not None and item != "agent" and item != "target":
                # Load image and scale it to fit grid square
                item_image = pygame.transform.scale(item_images[item], (pix_square_size, pix_square_size))
                # Draw image onto canvas at correct location
                canvas.blit(item_image, (pix_square_size * location[0], pix_square_size * location[1]))

        # Draw gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (0, 0, 0),  # Black gridlines
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                (0, 0, 0),  # Black gridlines
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        # Draw agent
        agent_image = pygame.image.load("mineplayer.png")  # Replace with your agent image file
        agent_image = pygame.transform.scale(agent_image, (pix_square_size, pix_square_size))
        canvas.blit(agent_image, (pix_square_size * self._agent_location[0], pix_square_size * self._agent_location[1]))

        """ # Draw target
        target_image = pygame.image.load("gold.png")  # Replace with your target image file
        target_image = pygame.transform.scale(target_image, (pix_square_size, pix_square_size))
        canvas.blit(target_image, (pix_square_size * self._target_location[0], pix_square_size * self._target_location[1])) """

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def set_agent_and_target_location(self, agent_location, target_location):
        """
        Set the initial locations of the agent and target.

        Args:
            agent_location (tuple): Initial location of the agent as a tuple (x, y).
            target_location (tuple): Initial location of the target as a tuple (x, y).
        """
        assert isinstance(agent_location, tuple) and len(agent_location) == 2, "Agent location must be a tuple of length 2"
        assert isinstance(target_location, tuple) and len(target_location) == 2, "Target location must be a tuple of length 2"
        assert all(0 <= loc < self.size for loc in agent_location), "Agent location must be within the grid bounds"
        assert all(0 <= loc < self.size for loc in target_location), "Target location must be within the grid bounds"
        
        self._agent_location = np.array(agent_location)
        self._target_location = np.array(target_location)

  
    def convert_to_graph(self):
        G = nx.grid_2d_graph(self.size, self.size)
        # Convert agent and target locations to tuples
        agent_location_tuple = tuple(self._agent_location)
        target_location_tuple = tuple(self._target_location)
        # Add self-loop to the target location
        G.add_edge(target_location_tuple, target_location_tuple)
        self.graph = G  # Assign the graph to the class attribute
        nodes_and_neighbors = {}
        c=0
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            nodes_and_neighbors[node] = neighbors
            #print(f"{node}: {neighbors}")
            c+=1
        print("NNDICT",nodes_and_neighbors)
        print("TOTAL COUNT:",c)
        return nodes_and_neighbors

    def display_graph(self):
        # Convert agent and target locations to tuples
        agent_location_tuple = tuple(self._agent_location)
        target_location_tuple = tuple(self._target_location)

        # Initialize colors for nodes
        node_colors = []
        for node in self.graph.nodes():
            if tuple(node) == agent_location_tuple:
                node_colors.append('yellow')
            elif tuple(node) == target_location_tuple:
                node_colors.append('red')
            elif tuple(node) == tuple(self._wood_location):
                node_colors.append('brown')  # Color for wood
                # Print coordinates of wood
                print("Wood location:", tuple(self._wood_location))
            elif tuple(node) == tuple(self._grass_location):
                node_colors.append('green')  # Color for grass
                # Print coordinates of grass
                print("Grass location:", tuple(self._grass_location))
            elif tuple(node) == tuple(self._iron_location):
                node_colors.append('black')  # Color for iron
                # Print coordinates of iron
                print("Iron location:", tuple(self._iron_location))
            elif tuple(node) == tuple(self._toolshed_location):
                node_colors.append('pink')  # Color for toolshed
                # Print coordinates of toolshed
                print("Toolshed location:", tuple(self._toolshed_location))
            else:
                node_colors.append('blue')

        # Determine the maximum y-coordinate
        max_y = max(node[1] for node in self.graph.nodes())
        # Shift y-coordinates to match the desired orientation
        pos = {node: (node[0] + 0.5, max_y - node[1] + 0.5) for node in self.graph.nodes()}
        labels = {node: f"{node[0]}, {node[1]}" for node in self.graph.nodes()}  # Labels for nodes
        # Draw nodes
        nx.draw(self.graph, pos=pos, with_labels=True, labels=labels, node_color=node_colors)
        plt.show()

    def dfs(self,graph, start, goal):
        stack = [(start, [start])]
        visited = set()

        while stack:
            (node, path) = stack.pop()
            if node not in visited:
                if node == goal:
                    return path
                visited.add(node)
                for neighbor in graph[node]:
                    stack.append((neighbor, path + [neighbor]))

        return None
    
    def bfs(self, graph, start, goal):
        queue = deque([(start, [start])])
        visited = set()

        while queue:
            node, path = queue.popleft()
            if node not in visited:
                if node == goal:
                    return path
                visited.add(node)
                for neighbor in graph[node]:
                    queue.append((neighbor, path + [neighbor]))

        return None
    
    
    def determine_action(self, current_location, next_location):
        dx = next_location[0] - current_location[0]
        dy = next_location[1] - current_location[1]

        if dx == 1:
            return 0  # Right
        elif dx == -1:
            return 2  # Left
        elif dy == 1:
            return 1  # Up
        elif dy == -1:
            return 3  # Down
        else:
            return None  # Invalid action

    def step_through_path(self, path):
        for i in range(len(path) - 1):
            current_location = path[i]
            next_location = path[i + 1]

            action = self.determine_action(current_location, next_location)
            if action is not None:
                observation, reward, done, _, info = self.step(action)
                self.render()

                if done:
                    print("Goal reached!")
                    break
            else:
                print("Invalid action")

# Example usage:
if __name__ == "__main__":
    env = GridWorldEnv(render_mode="human", size=5)
        # Get the location of the wood
    wood_location =tuple( env.get_location("wood"))
    iron_location=env.get_location("iron")
    tool_location=env.get_location("toolshed")
    grass_location=env.get_location("grass")
    print("Wood location:", wood_location,"iron location:", iron_location,"tool location:", tool_location,"gold location:", grass_location)
    agent_location = (0,0)
    env.set_agent_and_target_location(agent_location, wood_location)

    # Set item locations
    env.reset()

    # Call convert_to_graph to create the graph
    n = env.convert_to_graph()

    # Display the graph
    env.display_graph()




    # Find path using BFS
    initial_state = agent_location
    goal_state = tuple(wood_location)
    path = env.bfs(n, initial_state, goal_state)
    print("Path from", initial_state, "to", goal_state, ":", path)

    # Step through the path
    env.step_through_path(path)

     # Find path using BFS
    initial_state = wood_location
    goal_state = tuple(iron_location)
    path = env.bfs(n, initial_state, goal_state)
    print("Path from", initial_state, "to", goal_state, ":", path)
    # Step through the path
    env.step_through_path(path)


    initial_state = tuple(iron_location)
    goal_state = tuple(tool_location)
    path = env.bfs(n, initial_state, goal_state)
    print("Path from", initial_state, "to", goal_state, ":", path)
    # Step through the path
    env.step_through_path(path)

    env.close()