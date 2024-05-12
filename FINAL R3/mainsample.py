

import sys
import json

# Parse command-line arguments
args = sys.argv[1:]

# Find the index of '--items_list' argument
items_list_index = args.index("--items_list") + 1

# Deserialize items_list from JSON string
items_list = json.loads(args[items_list_index])
print('MAIN LOOP ITEM LIST :',items_list)

from maingrid import GridWorldEnv
from sympy import symbols, And, Or, simplify_logic
import time
import numpy as np
import pygame
import os
import gymnasium as gym
from gymnasium import spaces
from os import environ
import streamlit as st




start_time = time.time()
  
agent_location = (0, 0) 
env = GridWorldEnv(render_mode="human", size=5)





print("THE TASKS ARE :----------------------------------")
print("TASK 1: collect iron and collect wood and the reach the toolshed")
print("TASK 2: collect  wood or grass and then pick up gold ")
print("-----------------------------------------------------")

task1 = "iron & wood & toolshed"
task2 = "gold & (grass ⊕ wood)"

# Set the agent location and target item
env.set_agent_and_target_location(agent_location, items_list[0])
# Find unspecified locations
all_items = ['wood', 'iron', 'toolshed', 'grass', 'gold','furnace']
unspecified_items = list(set(all_items) - set(items_list))
#print("Unspecified items:", unspecified_items)
# Get unspecified item locations
unspecified_locations = [env.get_location(item) for item in unspecified_items]
# Convert unspecified locations to tuples
unspecified_locations = [tuple(loc) for loc in unspecified_locations if loc is not None]
print("T->r(i(S->(S->p(S))))")
#print("T->r(i(V ⊕ S))")
# Call convert_to_graph to create the graph
graph = env.convert_to_graph(unspecified_locations)
env.specified_items=items_list
# Display the graph
env.display_graph()
pygame.time.wait(5000)
    # Example usage



# Store paths for each pair of items
paths = []
print("ITEM LIST BEFORE VISITING:",items_list)
# Convert items to their respective locations
for item in items_list:
    item_location = env.get_location(item)
    if item_location is None:
        print(f"Item '{item}' not found or already picked up.")
    elif item in env.collected_items:
        print(f"Item '{item}' has already been collected.")
    else:
        # Compute path from current agent location to the item
        path = env.bfs(graph, tuple(agent_location), tuple(item_location))
        #print("Path to", item, ":", path)
        paths.append(path)
        # Update agent location to the current item location
        agent_location = item_location
        # Mark item as collected
        #env.collected_items.append(item)




env.set_window_position() 
r=0
# Step through each path in the environment
for path in paths:
    if path:
        citmain,rew=env.step_through_path(path,items_list,start_time)
        print("VISITED:",citmain)
        if citmain==items_list:
            exit()
            # You need to define 'collected_item'
        pygame.time.wait(2000)  # Adjust the wait time as needed
# Close the environment
end_time = time.time()  # Record the end time
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
env.close()






