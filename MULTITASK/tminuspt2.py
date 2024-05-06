from itstminus import GridWorldEnv
from sympy import symbols, And, Or, simplify_logic
import time
import numpy as np
import pygame
import os
import gymnasium as gym
from gymnasium import spaces
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from expmiralogic import generate_output_expression,parse_input_expression
# Define symbols for each item
wood, iron, grass, toolshed, gold = symbols('wood iron grass toolshed gold')
start_time = time.time()

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






def main():
    
    agent_location = (0, 0)  # Initial agent location
    #items = input("Enter items separated by commas (e.g., wood,iron): ").strip().lower().split(',')
    #print("The items entered by the user is:", items)
    # Create the environment
    env = GridWorldEnv(render_mode="human", size=5)


    print("THE TASKS ARE :----------------------------------")
    print("TASK 1: collect iron and collect wood and the reach the toolshed")
    print("TASK 2: collect  wood or grass and then pick up gold ")
        # Example usage:
    print("-----------------------------------------------------")
    task1 = "iron & wood & toolshed"
    task2 = "gold & (grass ⊕ wood)"

    # Initialize an empty list to store items
    new_items_list = []

    # Parse and evaluate Task 1
    parsed_task1 = And(wood, iron, toolshed)
    simplified_task1 = simplify_logic(parsed_task1)
    new_items_list = evaluate_task(simplified_task1, new_items_list)

    input_expression = "And(iron, toolshed, wood)"
    output_expression = generate_output_expression(input_expression)
    print("OUTPUT EXPRESSION:",output_expression)


    # Parse and evaluate Task 2
    parsed_task2 = Or(And(wood, gold), And(grass, gold))
    simplified_task2 = simplify_logic(parsed_task2)
        # Test the function
    input_exp = "EXP=gold & (grass | wood)"
    output_exp = parse_input_expression(input_exp)
    #print("OUTPUT EXPRESSION:", output_exp)

    new_items_list = evaluate_task(simplified_task2, new_items_list)

    print("PARSED TASK 1: ",simplified_task1)
    print("PARSED TASK 2: ",simplified_task2)
    print("***************PROCESSING BOTH TASKS **********")
    #print("Final TASKS to be done:", items_list)
    print()
    print()
        # Create a new list with items except 'toolshed' and 'gold'
    items_list = [item for item in new_items_list if item not in ['toolshed', 'gold']]

    # Add 'toolshed' and 'gold' at the end of the new list
    items_list.extend(['toolshed', 'gold'])

    print("Final TASKS to be done:", items_list)
    
    # Set the agent location and target item
    env.set_agent_and_target_location(agent_location, items_list[0])
    
    # Find unspecified locations
    all_items = ['wood', 'iron', 'toolshed', 'grass', 'gold']
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

if __name__ == "__main__":
    main()
    
