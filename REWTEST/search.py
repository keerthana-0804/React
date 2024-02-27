from rmatrix import mira_l

print("\nMira List:")
for l in mira_l:
    print(l)
#print(mira_l)
    
mira_graph = {}

for action in mira_l:
    current_state, next_state, _ = action
    if current_state not in mira_graph:
        mira_graph[current_state] = []
    mira_graph[current_state].append(next_state)

print("Generated MIRA Graph:")
print(mira_graph)


def dfs(graph, current, goal, path=[]):
    path = path + [current]
    if current == goal:
        return [path]
    if current not in graph:
        return []
    paths = []
    for neighbor in graph[current]:
        if neighbor not in path:
            new_paths = dfs(graph, neighbor, goal, path)
            for new_path in new_paths:
                paths.append(new_path)
    return paths



# Ask the user for initial and goal states
initial_state = int(input("Enter initial state: "))
goal_state = int(input("Enter goal state: "))

# Find all paths using Depth-First Search
all_paths = dfs(mira_graph, initial_state, goal_state)

# Display all paths
if all_paths:
    print(f"All possible paths from {initial_state} to {goal_state}:")
    for path in all_paths:
        print(path)
else:
    print("No valid paths found.")

