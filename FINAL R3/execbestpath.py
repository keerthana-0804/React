import gymnasium as gym
import time

from test1rew import best_paths

# Define environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False,render_mode="human")

# Determine the size of the grid from the environment's observation space
grid_size = int(env.observation_space.n**0.5)

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
best_path=best_paths[0]
start_time=time.time()
actions = convert_path_to_actions(best_path)
print(f"Executing path: {actions}")
observation = env.reset()
for action in actions:
    observation, reward, done, _ = env.step(action)[:4]  # Unpack the first four values
    env.render()
    if done:
        break
    
elapsed_time = time.time() - start_time
print(f"----{elapsed_time:.6f} seconds----")