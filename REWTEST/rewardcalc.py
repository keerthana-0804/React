from rmatrix import reward_matrix
from search import all_paths
import gym

# Function to calculate total reward for a path
def calculate_total_reward(path, reward_matrix):
    total_reward = 0
    for i in range(len(path) - 1):
        current_state = path[i]
        next_state = path[i + 1]
        total_reward += reward_matrix[current_state][next_state]
    return total_reward

# Store paths with highest total reward
highest_paths = []
highest_reward = float('-inf')

# Calculate total reward for each path and print the results
for path in all_paths:
    total_reward_path = calculate_total_reward(path, reward_matrix)

    if total_reward_path > highest_reward:
        highest_paths = [path]
        highest_reward = total_reward_path
    elif total_reward_path == highest_reward:
        highest_paths.append(path)

if __name__=='__main__':
# Print the results including both path and reward
    for path in highest_paths:
        print(f"Highest Total Reward Path: {path}, Total Reward: {calculate_total_reward(path, reward_matrix)}")
