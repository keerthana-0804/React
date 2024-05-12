from searchstr import mira_graph,dfs
from rewardingscheme import evaluate
from finalgraphlayout import rm
import time
initial_state = 0
goal_state = 15

    # Find all paths using Depth-First Search
all_paths = dfs(mira_graph, initial_state, goal_state)
# Part 1: DFS and path finding
start_time_dfs = time.time()

    # Display all paths
if all_paths:
    print(f"All possible paths from {initial_state} to {goal_state}:")
    for path in all_paths:
            print(path)
else:
    print("No valid paths found.")
elapsed_time_dfs = time.time() - start_time_dfs
print(f"----{elapsed_time_dfs:.6f} seconds----")

max_reward = float('-inf')
best_paths = []
for path in all_paths:
    result,tot = evaluate(path)
    print(f"Path {path}: Result = {result}")
    print("Total Reward: {}".format(tot))
    if tot> max_reward:
        max_reward = tot
        best_paths = [path]  # Store the current path
    elif tot == max_reward:
        best_paths.append(path)  # Append the current path to the list

#print(best_paths)
