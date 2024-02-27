import gym
import time
import pygame
from pygame.locals import QUIT

start_time = time.time()

# Set up Pygame for rendering

clock = pygame.time.Clock()

from rewardcalc import highest_paths

# Define
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

# Determine the size of the grid from the environment's observation space
grid_size = int(env.observation_space.n ** 0.5)

# Map states from the environment to (row, column) coordinates
state_mapping = {state: divmod(state, grid_size) for state in range(env.observation_space.n)}

# Convert state transitions into actions
def convert_path_to_actions(path):
    actions = []
    for i in range(len(path) - 1):
        current_state = state_mapping[path[i]]
        next_state = state_mapping[path[i + 1]]

        # Determine the action based on the state transition
        if next_state[0] == current_state[0] and next_state[1] == current_state[1] + 1:
            actions.append(2)  # 'right' action in FrozenLake
        elif next_state[0] == current_state[0] and next_state[1] == current_state[1] - 1:
            actions.append(0)  # 'left' action in FrozenLake
        elif next_state[0] == current_state[0] + 1 and next_state[1] == current_state[1]:
            actions.append(1)  # 'down' action in FrozenLake
        elif next_state[0] == current_state[0] - 1 and next_state[1] == current_state[1]:
            actions.append(3)  # 'up' action in FrozenLake
        else:
            raise ValueError("Invalid state transition in the path")

    return actions

# Example paths from your graph

# Apply paths to FrozenLake
for path in highest_paths:
    actions = convert_path_to_actions(path)
    print(f"Executing path: {actions}")
    pygame.init()
    screen_size = (1000, 1000)  # Adjust screen size as needed
    screen = pygame.display.set_mode(screen_size)
    observation = env.reset()
    for action in actions:
        observation, reward, done, _ = env.step(action)[:4]  # Unpack the first four values

        # Render the environment
        env.render(mode='human')

        # Slow down the rendering speed
        clock.tick(2)  # Adjust the value as needed for your desired speed

        # Handle events, including quitting the Pygame window
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                quit()

        if done:
            break

elapsed_time = time.time() - start_time
print(f"----{elapsed_time:.6f} seconds----")

# Close the Pygame window at the end
pygame.quit()
