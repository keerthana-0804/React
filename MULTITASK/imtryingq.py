import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import pdb

""" print(gymnasium.__version__ )
print()
print(gym.__version__) """

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, color_options=None):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.color_options = color_options  # List of colors the agent can pick from

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

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    @staticmethod
    def calculate_reward(agent_location, target_location):
        distance = np.linalg.norm(agent_location - target_location)
        reward_threshold = 0.5  # Example threshold
        if distance < reward_threshold:
            reward = 1
        else:
            reward = 0
        return reward  
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action % len(self._action_to_direction)]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # Calculate reward based on distance to target
        reward = self.calculate_reward(self._agent_location, self._target_location)
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        print(f"Returned observation: {observation}, reward: {reward}, terminated: {terminated}")
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
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    

import numpy as np
import random

# Define Q-learning parameters
learning_rate = 0.9  # Learning rate
discount_factor = 0.95  # Discount factor
epsilon = 0.1 # Epsilon-greedy exploration parameter
num_episodes = 10  # Number of episodes to train for

# Create a Q-table with dimensions (state_space, action_space)
# Filled with zeros initially (all actions have equal value for each state)
q_table = np.zeros((5**2, 4))  # Assuming size x size grid

env = GridWorldEnv(render_mode="human")

for episode in range(num_episodes):
    # Reset the environment
    observation, info = env.reset()
    state = observation["agent"]  # Extract the agent's location for Q-table lookup
    
    total_reward = 0
    
    # Training loop within an episode
    terminated = False
    while not terminated:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore randomly
        else:
            # Exploit the learned Q-values
            action = np.argmax(q_table[state])
        
        # Ensure the selected action is within the valid range
        if action not in range(env.action_space.n):
            action = env.action_space.sample()  # Choose a random action
        
        # Take action, observe reward and next state
        next_observation, reward, terminated, _, _ = env.step(action)
        next_state = next_observation["agent"]

        total_reward += reward

        # Update Q-table using the Bellman equation
        q_table[state, action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action])

        # Update the current state
        state = next_state

    if episode % 100 == 0:
        print("Episode:", episode, "Total Reward:", total_reward)

# Once training is done, you can test the learned policy
# Test the learned policy
def test_policy(env, q_table):
    observation, info = env.reset()
    state = observation["agent"]  # Extract the agent's location for Q-table lookup
    terminated = False

    while not terminated:
        action = np.argmax(q_table[state])  # Choose action greedily based on Q-values
        next_observation, _, terminated, _, _ = env.step(action)  # Take action
        state = next_observation["agent"]  # Extract next state
        env.render()  # Visualize the environment


print("TESTING POLICY-----------------------------------------")
# Test the learned policy
test_policy(env, q_table)

